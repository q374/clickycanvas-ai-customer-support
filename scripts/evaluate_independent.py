"""Evaluate the frozen independent challenge set using customer-visible behavior.

This evaluator deliberately avoids response.intent, emotion_strategy and other
implementation labels as quality assertions. It checks the answer text and the
customer-visible product cards only. Internal model status is retained as an
operational field in the report and is not part of the pass decision.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import ChatRequest, ShoppingContext
from scripts.evaluate_demo import build_agent


ROOT = Path(__file__).resolve().parents[1]

METRIC_DEFINITIONS = {
    "observable_pass_rate": "每个案例所有轮次的顾客可见行为检查全部通过的比例。",
    "turn_observable_pass_rate": "所有对话轮次的顾客可见行为检查通过比例。",
    "criterion_pass_rates": "每类外部行为检查的通过比例；不读取内部意图或策略字段。",
    "safety_violation_count": "出现禁止承诺或缺少安全处置的轮次数量，目标为 0。",
    "response_latency_ms_p95": "单进程 agent.chat 墙钟延迟 P95，不含服务启动、网络服务和页面渲染。",
}


def _product_titles(response) -> list[str]:
    data = response.data if isinstance(response.data, dict) else {}
    products = data.get("products")
    if not isinstance(products, list):
        product = data.get("product")
        products = [product] if isinstance(product, dict) else []
    return [str(item.get("title")) for item in products if isinstance(item, dict) and item.get("title")]


def _product_count(response) -> int:
    data = response.data if isinstance(response.data, dict) else {}
    products = data.get("products")
    if isinstance(products, list):
        return len(products)
    return 1 if isinstance(data.get("product"), dict) else 0


def _has_any(answer: str, values: list[str]) -> bool:
    return any(value in answer for value in values)


def _check_behavior(response, expected: dict[str, Any]) -> dict[str, Any]:
    answer = response.answer.strip()
    titles = _product_titles(response)
    checks: dict[str, bool] = {}
    missing: list[str] = []
    forbidden: list[str] = []

    for value in expected.get("answer_must_include", []):
        ok = value in answer
        checks[f"include:{value}"] = ok
        if not ok:
            missing.append(value)
    for values in expected.get("answer_must_include_any", []):
        ok = _has_any(answer, values)
        label = " / ".join(values)
        checks[f"include_any:{label}"] = ok
        if not ok:
            missing.append(label)
    for value in expected.get("answer_must_not_include", []):
        ok = value not in answer
        checks[f"exclude:{value}"] = ok
        if not ok:
            forbidden.append(value)
    for values in expected.get("answer_must_not_include_any", []):
        found = [value for value in values if value in answer]
        ok = not found
        label = " / ".join(values)
        checks[f"exclude_any:{label}"] = ok
        if not ok:
            forbidden.extend(found)

    for values in expected.get("product_titles_any", []):
        ok = _has_any(answer, values) or any(_has_any(title, values) for title in titles)
        label = " / ".join(values)
        checks[f"product_title_any:{label}"] = ok
        if not ok:
            missing.append(label)

    count = _product_count(response)
    if "product_count_min" in expected:
        minimum = int(expected["product_count_min"])
        checks[f"product_count>={minimum}"] = count >= minimum
        if count < minimum:
            missing.append(f"至少 {minimum} 个商品卡片")
    if "product_count_max" in expected:
        maximum = int(expected["product_count_max"])
        checks[f"product_count<={maximum}"] = count <= maximum
        if count > maximum:
            forbidden.append(f"超过 {maximum} 个商品卡片")

    if "next_step_any" in expected:
        ok = _has_any(answer, expected["next_step_any"])
        checks["next_step_any"] = ok
        if not ok:
            missing.append("下一步：" + " / ".join(expected["next_step_any"]))

    if expected.get("apology_required"):
        ok = bool(re.match(r"^(?:很抱歉|抱歉)", answer))
        checks["apology_required"] = ok
        if not ok:
            missing.append("合适的道歉或承接")
    if expected.get("apology_forbidden"):
        ok = not bool(re.match(r"^(?:很抱歉|抱歉)", answer))
        checks["apology_forbidden"] = ok
        if not ok:
            forbidden.append("不必要的开场道歉")

    if expected.get("safety_stop"):
        ok = "停止使用" in answer and "断开电源" in answer
        checks["safety_stop"] = ok
        if not ok:
            missing.append("停止使用和断开电源")

    passed = all(checks.values()) if checks else True
    safety_terms = expected.get("safety_must_not_include", [])
    safety_violations = [value for value in safety_terms if value in answer]
    if safety_violations:
        passed = False
        forbidden.extend(safety_violations)
    return {
        "passed": passed,
        "checks": checks,
        "missing": list(dict.fromkeys(missing)),
        "forbidden": list(dict.fromkeys(forbidden)),
        "safety_violations": list(dict.fromkeys(safety_violations)),
        "answer": answer,
        "product_titles": titles,
        "product_count": count,
    }


def _turns(case: dict[str, Any]) -> list[dict[str, Any]]:
    turns = case.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{case.get('id')}: turns must be a non-empty list")
    return turns


def evaluate(agent, cases: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    case_results: list[dict[str, Any]] = []
    turn_results: list[dict[str, Any]] = []
    latencies: list[float] = []
    criteria: Counter[str] = Counter()
    criteria_passed: Counter[str] = Counter()
    group_totals: Counter[str] = Counter()
    group_passed: Counter[str] = Counter()
    safety_violations = 0

    for case in cases:
        context: ShoppingContext | None = None
        pending_order_id: str | None = None
        case_turns: list[dict[str, Any]] = []
        for index, turn in enumerate(_turns(case), start=1):
            message = str(turn["original_query"])
            order_id = pending_order_id if pending_order_id and "验证码" in message else None
            pending_order_id = None
            started = time.perf_counter()
            response = agent.chat(ChatRequest(message=message, shopping_context=context, order_id=order_id))
            latencies.append((time.perf_counter() - started) * 1000)
            data = response.data if isinstance(response.data, dict) else {}
            next_context = data.get("shopping_context")
            if isinstance(next_context, dict):
                context = ShoppingContext.model_validate(next_context)
            if data.get("order_id") and "验证码" in response.answer:
                pending_order_id = str(data["order_id"])
            expected = turn.get("observable", {})
            metrics = _check_behavior(response, expected)
            for key, ok in metrics["checks"].items():
                criteria[key] += 1
                if ok:
                    criteria_passed[key] += 1
            safety_violations += len(metrics["safety_violations"])
            turn_result = {
                "case_id": case["id"],
                "group": case["group"],
                "turn": index,
                "query": message,
                "expected_emotion": turn.get("emotion_label"),
                "metrics": metrics,
                "llm_status": data.get("llm", {}).get("status") if isinstance(data.get("llm"), dict) else None,
            }
            turn_results.append(turn_result)
            case_turns.append(turn_result)
        passed = all(item["metrics"]["passed"] for item in case_turns)
        group_totals[case["group"]] += 1
        if passed:
            group_passed[case["group"]] += 1
        case_results.append({
            "id": case["id"],
            "source": case["source"],
            "blind": bool(case.get("blind")),
            "group": case["group"],
            "passed": passed,
            "turns": case_turns,
        })

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    p95 = None
    if latencies:
        p95 = round(statistics.quantiles(latencies, n=20, method="inclusive")[-1], 2) if len(latencies) >= 2 else round(latencies[0], 2)
    return {
        "version": "p6.6-observable-v1",
        "mode": mode,
        "cases": len(case_results),
        "turns": len(turn_results),
        "metrics": {
            "observable_pass_rate": ratio(sum(item["passed"] for item in case_results), len(case_results)),
            "turn_observable_pass_rate": ratio(sum(item["metrics"]["passed"] for item in turn_results), len(turn_results)),
            "criterion_pass_rates": {
                key: ratio(criteria_passed[key], total) for key, total in sorted(criteria.items())
            },
            "safety_violation_count": safety_violations,
            "response_latency_ms_p95": p95,
            "llm_statuses": dict(Counter(item["llm_status"] or "deterministic" for item in turn_results)),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "group_metrics": {
            group: {
                "cases": group_totals[group],
                "passed": group_passed[group],
                "pass_rate": ratio(group_passed[group], group_totals[group]),
            }
            for group in sorted(group_totals)
        },
        "results": case_results,
        "failures": [item for item in case_results if not item["passed"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行独立挑战集（只检查顾客可见行为）")
    parser.add_argument("--cases", default="data/independent_challenge_cases.json")
    parser.add_argument("--database", default="data/runtime/p6-independent.db")
    parser.add_argument("--mode", choices=["deterministic", "deepseek"], default="deterministic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
        missing = selected - {case["id"] for case in cases}
        if missing:
            parser.error("未知案例 ID：" + ", ".join(sorted(missing)))
    if args.limit:
        cases = cases[: args.limit]
    report = evaluate(build_agent(ROOT / args.database, args.mode), cases, args.mode)
    if args.output:
        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        metrics = report["metrics"]
        print(f"独立挑战集：{report['cases']} 条、{report['turns']} 轮")
        print(f"案例可观察行为通过率：{metrics['observable_pass_rate']:.1%}")
        print(f"轮次可观察行为通过率：{metrics['turn_observable_pass_rate']:.1%}")
        print(f"安全边界违规：{metrics['safety_violation_count']} 轮")
        print(f"P95：{metrics['response_latency_ms_p95']} ms")
        for group, values in report["group_metrics"].items():
            print(f"- {group}: {values['passed']}/{values['cases']} 通过")
        if report["failures"]:
            print("失败案例：")
            for failure in report["failures"]:
                details = [
                    f"第{turn['turn']}轮: {turn['metrics']['missing'] or turn['metrics']['forbidden']}"
                    for turn in failure["turns"] if not turn["metrics"]["passed"]
                ]
                print(f"- {failure['id']}: {'; '.join(details)}")


if __name__ == "__main__":
    main()
