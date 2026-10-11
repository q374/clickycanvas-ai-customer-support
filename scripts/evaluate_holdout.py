"""Run the frozen P6.1 holdout set and emit reproducible quality counters.

The holdout file keeps expectations separate from implementation examples.  The
default run evaluates only ``source == new_holdout`` cases with ``blind: true``;
``--include-reference`` is useful when checking the observed and manually
designed reference rows as well.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import ChatRequest, ShoppingContext
from scripts.evaluate_demo import build_agent


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_PATH = ROOT / "data" / "p6_holdout_cases.json"

METRIC_DEFINITIONS = {
    "intent_accuracy": "所有评估轮次中，实际 intent 等于 expected_intent 的比例。",
    "slot_accuracy": "所有声明 expected_slots 的槽位键值比较中，实际 shopping_context 值相等的比例。",
    "unnecessary_clarification_rate": "期望不是 PRODUCT_CLARIFY 却返回 PRODUCT_CLARIFY 的轮次，占所有轮次的比例。",
    "hard_condition_violation_rate": "声明 expected_product_ids 的商品轮次中，返回候选违反任一 required_conditions 或应无匹配却返回商品的比例。",
    "candidate_coverage": "声明 expected_product_ids 的轮次中，期望 ID 全部被召回且空期望保持空结果的比例。",
    "preference_order_accuracy": "声明 expected_product_ids 的轮次中，实际候选顺序与冻结顺序完全一致的比例。",
    "fact_consistency": "回答边界、政策主文档和订单状态均符合期望的轮次比例。",
    "no_evidence_commitment_rate": "回答包含 answer_boundary.must_not_claim 的轮次，占所有轮次的比例。",
    "model_fallback_rate": "DeepSeek 可调用轮次中 status=fallback 的比例；deterministic 为 null。",
    "response_latency_ms_p95": "单进程 agent.chat wall-clock 延迟 P95，不含启动、网络服务和页面渲染。",
}


def _turns(case: dict[str, Any]) -> list[dict[str, Any]]:
    turns = case.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{case.get('id')}: turns must be a non-empty list")
    return turns


def _slot_value(response, key: str) -> Any:
    context = response.data.get("shopping_context")
    if isinstance(context, dict):
        return context.get(key)
    return None


def _condition_holds(product: dict[str, Any], condition: dict[str, Any]) -> bool:
    field = condition["field"]
    op = condition["op"]
    expected = condition["value"]
    value = product.get(field)
    if op == "includes":
        values = value if isinstance(value, list) else [value]
        return any(str(expected).lower() in str(item).lower() for item in values)
    if op == "excludes":
        values = value if isinstance(value, list) else [value]
        return all(str(expected).lower() not in str(item).lower() for item in values)
    if op == "eq":
        return value == expected
    if op == "le":
        return isinstance(value, (int, float)) and value <= expected
    if op == "ge":
        return isinstance(value, (int, float)) and value >= expected
    raise ValueError(f"unsupported condition operator: {op}")


def _response_boundary_ok(response, expected: dict[str, Any]) -> bool:
    boundary = expected.get("answer_boundary", {})
    answer = response.answer
    if any(text not in answer for text in boundary.get("must_include", [])):
        return False
    if any(text in answer for text in boundary.get("must_not_claim", [])):
        return False
    return True


def _evaluate_turn(response, expected: dict[str, Any]) -> dict[str, Any]:
    actual_ids = [item.get("id") for item in response.data.get("products", []) if isinstance(item, dict)]
    expected_ids = expected.get("expected_product_ids")
    has_product_expectation = expected_ids is not None
    candidate_coverage = None
    ranking_correct = None
    if has_product_expectation:
        candidate_coverage = set(expected_ids).issubset(actual_ids) and (
            bool(expected_ids) or not actual_ids
        )
        ranking_correct = actual_ids == expected_ids

    hard_conditions = expected.get("required_conditions", [])
    hard_condition_ok = True
    if hard_conditions:
        products = response.data.get("products", [])
        hard_condition_ok = all(
            _condition_holds(product, condition)
            for product in products
            for condition in hard_conditions
        )
        if expected_ids == [] and products:
            hard_condition_ok = False

    policy_ok = True
    expected_doc = expected.get("expected_policy_document")
    if expected_doc:
        documents = response.data.get("documents", [])
        policy_ok = bool(documents) and documents[0].get("document_id") == expected_doc

    expected_order = expected.get("expected_order")
    order_ok = True
    if expected_order:
        order = response.data.get("order")
        if expected_order.get("state") == "verified":
            order_ok = (
                isinstance(order, dict)
                and order.get("order_id") == expected_order.get("order_id")
                and order.get("fulfillment_status") == expected_order.get("fulfillment_status")
            )
        else:
            order_ok = order is None and expected_order.get("state_phrase", "") in response.answer

    return {
        "intent_correct": response.intent == expected.get("expected_intent"),
        "slot_checks": [
            {
                "slot": key,
                "expected": value,
                "actual": _slot_value(response, key),
                "correct": _slot_value(response, key) == value,
            }
            for key, value in expected.get("expected_slots", {}).items()
        ],
        "unnecessary_clarification": (
            expected.get("expected_intent") != "PRODUCT_CLARIFY" and response.intent == "PRODUCT_CLARIFY"
        ),
        "candidate_coverage": candidate_coverage,
        "ranking_correct": ranking_correct,
        "hard_condition_ok": hard_condition_ok,
        "policy_ok": policy_ok,
        "order_ok": order_ok,
        "fact_consistent": _response_boundary_ok(response, expected) and policy_ok and order_ok,
        "no_evidence_commitment_ok": not any(
            text in response.answer for text in expected.get("answer_boundary", {}).get("must_not_claim", [])
        ),
        "actual_product_ids": actual_ids,
        "answer": response.answer,
        "llm_status": response.data.get("llm", {}).get("status") if isinstance(response.data.get("llm"), dict) else None,
    }


def evaluate(agent, cases: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    slot_checks: list[dict[str, Any]] = []
    all_turn_metrics: list[dict[str, Any]] = []

    for case in cases:
        context: ShoppingContext | None = None
        pending_order_id: str | None = None
        case_turns = []
        for turn in _turns(case):
            message = turn["original_query"]
            order_id = pending_order_id if pending_order_id and "验证码" in message else None
            pending_order_id = None
            started = time.perf_counter()
            response = agent.chat(ChatRequest(message=message, shopping_context=context, order_id=order_id))
            latencies_ms.append((time.perf_counter() - started) * 1000)
            if response.data.get("shopping_context"):
                context = ShoppingContext.model_validate(response.data["shopping_context"])
            if response.intent == "ORDER_STATUS" and response.data.get("order_id") and not response.data.get("order"):
                pending_order_id = response.data["order_id"]
            metrics = _evaluate_turn(response, turn)
            slot_checks.extend(metrics["slot_checks"])
            all_turn_metrics.append(metrics)
            case_turns.append({"query": message, "expected": turn, "metrics": metrics})
        results.append({
            "id": case["id"],
            "source": case["source"],
            "blind": case.get("blind", False),
            "group": case["group"],
            "passed": all(
                turn["metrics"]["intent_correct"]
                and all(item["correct"] for item in turn["metrics"]["slot_checks"])
                and not turn["metrics"]["unnecessary_clarification"]
                and (turn["metrics"]["candidate_coverage"] is None or turn["metrics"]["candidate_coverage"])
                and (turn["metrics"]["ranking_correct"] is None or turn["metrics"]["ranking_correct"])
                and turn["metrics"]["hard_condition_ok"]
                and turn["metrics"]["policy_ok"]
                and turn["metrics"]["order_ok"]
                and turn["metrics"]["fact_consistent"]
                and turn["metrics"]["no_evidence_commitment_ok"]
                for turn in case_turns
            ),
            "turns": case_turns,
        })

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    candidate_metrics = [m for m in all_turn_metrics if m["candidate_coverage"] is not None]
    ranking_metrics = [m for m in all_turn_metrics if m["ranking_correct"] is not None]
    fallback_eligible = [m for m in all_turn_metrics if m["llm_status"] not in {None, "skipped_high_risk", "skipped_general", "skipped_clarification"}]
    fallback_count = sum(m["llm_status"] == "fallback" for m in fallback_eligible)
    p95 = round(statistics.quantiles(latencies_ms, n=20, method="inclusive")[-1], 2) if len(latencies_ms) >= 2 else (round(latencies_ms[0], 2) if latencies_ms else None)
    metrics = {
        "turns": len(all_turn_metrics),
        "cases": len(results),
        "intent_accuracy": ratio(sum(m["intent_correct"] for m in all_turn_metrics), len(all_turn_metrics)),
        "slot_accuracy": ratio(sum(item["correct"] for item in slot_checks), len(slot_checks)),
        "unnecessary_clarification_rate": ratio(sum(m["unnecessary_clarification"] for m in all_turn_metrics), len(all_turn_metrics)),
        "hard_condition_violation_rate": ratio(sum(not m["hard_condition_ok"] for m in all_turn_metrics if m["candidate_coverage"] is not None), sum(m["candidate_coverage"] is not None for m in all_turn_metrics)),
        "candidate_coverage": ratio(sum(m["candidate_coverage"] for m in candidate_metrics), len(candidate_metrics)),
        "preference_order_accuracy": ratio(sum(m["ranking_correct"] for m in ranking_metrics), len(ranking_metrics)),
        "fact_consistency": ratio(sum(m["fact_consistent"] for m in all_turn_metrics), len(all_turn_metrics)),
        "no_evidence_commitment_rate": ratio(sum(not m["no_evidence_commitment_ok"] for m in all_turn_metrics), len(all_turn_metrics)),
        "model_fallback_rate": ratio(fallback_count, len(fallback_eligible)) if mode == "deepseek" else None,
        "response_latency_ms_p95": p95,
        "llm_statuses": dict(Counter(m["llm_status"] or "deterministic" for m in all_turn_metrics)),
    }
    return {
        "mode": mode,
        "metric_definitions": METRIC_DEFINITIONS,
        "metrics": metrics,
        "source_counts": dict(Counter(case["source"] for case in cases)),
        "group_counts": dict(Counter(case["group"] for case in cases)),
        "passed": sum(result["passed"] for result in results),
        "total": len(results),
        "failures": [result for result in results if not result["passed"]],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 P6.1 冻结盲测集并输出 P6.2 基线统计")
    parser.add_argument("--cases", default=str(HOLDOUT_PATH))
    parser.add_argument("--database", default="data/runtime/p6-holdout.db")
    parser.add_argument("--mode", choices=["deterministic", "deepseek"], default="deterministic")
    parser.add_argument("--include-reference", action="store_true", help="包含 user_observed/manual_design 参考案例")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", help="将 JSON 报告以 UTF-8 写入指定路径")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))
    cases = payload["cases"]
    if not args.include_reference:
        cases = [case for case in cases if case.get("blind") and case.get("source") == "new_holdout"]
    if args.limit:
        cases = cases[: args.limit]
    report = evaluate(build_agent(ROOT / args.database, args.mode), cases, args.mode)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(f"P6.2 盲测基线：{report['passed']}/{report['total']} 案例通过；轮次 {report['metrics']['turns']}")
    for key, value in report["metrics"].items():
        print(f"- {key}: {value}")
    print(f"- 来源：{report['source_counts']}")
    if report["failures"]:
        print("失败案例：")
        for failure in report["failures"]:
            print(f"- {failure['id']}")


if __name__ == "__main__":
    main()
