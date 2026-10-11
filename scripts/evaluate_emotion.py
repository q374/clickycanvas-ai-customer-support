"""Evaluate manually designed emotion and response-style scenarios."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import SupportAgent
from app.db import Database
from app.models import ChatRequest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "emotion_scenarios.json"
CATALOG = ROOT / "data" / "catalog.json"
ORDERS = ROOT / "data" / "orders.json"
KNOWLEDGE = ROOT / "data" / "knowledge"
ACTION_BY_EMOTION = {
    "neutral": "answer", "confused": "explain_then_ask", "anxious": "answer_then_next_step",
    "frustrated": "act_first", "angry": "act_first", "high_risk": "handoff",
}
UNSUPPORTED_COMMITMENTS = ("已批准退款", "保证退款", "一定赔偿", "已经同意赔偿", "保证明天送到")
REASSURANCE_PHRASES = ("理解您的心情", "完全理解您的心情", "请消消气", "别着急")


def evaluate(cases: list[dict], agent: SupportAgent) -> dict:
    results = []
    for case in cases:
        response = agent.chat(ChatRequest(message=case["message"]))
        emotion = response.data.get("emotion", {})
        strategy = response.data.get("emotion_strategy", {})
        answer = response.answer
        expected_apology = case["apology"]
        apology = answer.startswith(("抱歉", "很抱歉"))
        question_count = answer.count("？") + answer.count("?")
        reassurance_count = sum(answer.count(phrase) for phrase in REASSURANCE_PHRASES)
        safety_action = not case.get("safety_action") or ("停止使用" in answer and "断开电源" in answer)
        handoff = case["emotion"] != "high_risk" or (
            response.intent == "ESCALATE" and response.escalation_case is not None and "人工" in answer
        )
        no_claim = not any(phrase in answer for phrase in UNSUPPORTED_COMMITMENTS)
        if response.intent == "GENERAL" and case["emotion"] in {"angry", "frustrated"}:
            next_step = "请告诉我" in answer
        elif response.intent == "ESCALATE":
            next_step = "人工" in answer and "案件" in answer
        elif response.intent == "PRODUCT_CLARIFY":
            next_step = "请" in answer or "？" in answer
        else:
            next_step = bool(answer.strip()) and response.intent == case["intent"]
        checks = {
            "emotion": emotion.get("state") == case["emotion"],
            "intent": response.intent == case["intent"],
            "strategy": strategy.get("action") == ACTION_BY_EMOTION[case["emotion"]],
            "apology": apology == expected_apology,
            "no_repeated_reassurance": reassurance_count <= 1,
            "no_repeated_questions": question_count <= strategy.get("max_questions", 2),
            "task_completion_proxy": next_step and handoff and safety_action,
            "no_unsupported_commitment": no_claim,
            "next_step_clear_proxy": next_step,
        }
        results.append({
            "id": case["id"], "message": case["message"], "expected_emotion": case["emotion"],
            "actual_emotion": emotion.get("state"), "expected_intent": case["intent"],
            "actual_intent": response.intent, "answer": answer, "checks": checks,
            "passed": all(checks.values()),
        })
    metrics = {name: round(sum(row["checks"][name] for row in results) / len(results), 4)
               for name in results[0]["checks"]} if results else {}
    groups = Counter(row["expected_emotion"] for row in results)
    group_passed = Counter(row["expected_emotion"] for row in results if row["passed"])
    return {
        "source": "synthetic_manual_design", "total": len(results),
        "passed": sum(row["passed"] for row in results), "metrics": metrics,
        "by_emotion": {group: {"passed": group_passed[group], "total": total} for group, total in sorted(groups.items())},
        "metric_notes": {
            "task_completion_proxy": "仅检查本轮是否按预期路由、给出下一步和必要的人工案件；不代表用户实际问题已解决。",
            "next_step_clear_proxy": "关键词和路由规则的自动代理指标，仍需人工阅读复核。",
            "apology": "检查答复开头是否按人工标注的场景道歉，不评价措辞自然度。",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable report")
    parser.add_argument("--output", type=Path, help="Save the full report under data/runtime")
    args = parser.parse_args()
    cases = json.loads(DATA.read_text(encoding="utf-8"))["cases"]
    runtime = ROOT / "data" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    db_path = runtime / f"emotion-eval-{uuid.uuid4().hex}.db"
    try:
        database = Database(db_path)
        database.seed_catalog(CATALOG)
        database.seed_orders(ORDERS)
        database.seed_knowledge(KNOWLEDGE)
        report = evaluate(cases, SupportAgent(database))
    finally:
        db_path.unlink(missing_ok=True)
    if args.output:
        target = args.output.resolve()
        if target.parent != runtime.resolve():
            parser.error("--output must be a direct child of data/runtime")
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"情绪场景评估：{report['passed']}/{report['total']} 通过（全部为人工设计场景）")
        print("指标：" + "、".join(f"{key}={value:.1%}" for key, value in report["metrics"].items()))
        for row in report["results"]:
            if not row["passed"]:
                failed = [key for key, passed in row["checks"].items() if not passed]
                print(f"- {row['id']} {row['message']}：{','.join(failed)}；实际={row['actual_emotion']}/{row['actual_intent']}；答复={row['answer']}")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
