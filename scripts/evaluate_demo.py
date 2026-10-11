"""Run the deterministic ClickyCanvas evaluation set."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Allow `python scripts/evaluate_demo.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import SupportAgent
from app.db import Database
from app.llm import DeepSeekOrchestrator
from app.main import CATALOG_PATH, KNOWLEDGE_PATH, ORDERS_PATH
from app.models import ChatRequest


ROOT = Path(__file__).resolve().parents[1]

SOURCE_BY_INTENT = {
    "PRODUCT_SEARCH": "ClickyCanvas 商品快照",
    "PRODUCT_COMPARE": "ClickyCanvas 商品快照",
    "POLICY_FAQ": None,
    "ORDER_STATUS": "ClickyCanvas 模拟订单库",
    "ESCALATE": "ClickyCanvas 售后流程",
}

FULFILLMENT_LABELS = {
    "processing": "处理中",
    "shipped": "已发货",
    "delayed": "物流延迟",
    "delivered": "已送达",
}


def structural_issues(case: dict, response) -> list[str]:
    """Check evidence and safety invariants that keyword matching misses."""
    issues: list[str] = []
    expected_state = case.get("expected_state")
    expected_source = SOURCE_BY_INTENT.get(case["expected_intent"])
    if expected_source and expected_source not in response.sources:
        issues.append(f"缺少来源：{expected_source}")
    if case["expected_intent"] == "POLICY_FAQ" and not response.sources:
        issues.append("政策回答缺少来源")

    if response.intent == "PRODUCT_SEARCH" and not isinstance(response.data.get("products"), list):
        issues.append("商品搜索缺少结构化 products")
    expected_product_ids = case.get("expected_product_ids")
    if expected_product_ids is not None:
        actual_product_ids = [product.get("id") for product in response.data.get("products", [])]
        if actual_product_ids != expected_product_ids:
            issues.append(f"商品结果不符：期望 {expected_product_ids}，实际 {actual_product_ids}")
    if expected_state == "no_products":
        if response.data.get("products") != []:
            issues.append("无结果搜索却返回商品")
        if "没有找到" not in response.answer:
            issues.append("无结果搜索未说明没有找到商品")

    if response.intent == "PRODUCT_COMPARE":
        product_ids = response.data.get("product_ids", [])
        products = response.data.get("products")
        if len(product_ids) >= 2 and expected_state != "missing_product" and not isinstance(products, list):
            issues.append("商品对比缺少结构化 products")
        if isinstance(products, list) and len(products) < 2 and len(product_ids) >= 2:
            issues.append("商品对比结构化商品少于两个")
        if isinstance(products, list) and len(products) >= 2:
            for product in products:
                if product.get("title") not in response.answer and product.get("id") not in response.answer:
                    issues.append(f"商品对比回答缺少商品：{product.get('id')}")
        if expected_state == "missing_product":
            if products:
                issues.append("未知商品对比却返回商品结果")
            for missing_id in case.get("expected_missing_product_ids", []):
                if missing_id not in response.answer:
                    issues.append(f"未说明未知商品：{missing_id}")

    if response.intent == "POLICY_FAQ" and expected_state != "policy_unknown" and not isinstance(response.data.get("documents"), list):
        issues.append("政策回答缺少检索文档")
    if expected_state == "policy_unknown":
        if response.data.get("documents"):
            issues.append("无依据政策问题却返回文档")
        if "没有找到明确的政策说明" not in response.answer:
            issues.append("无依据政策问题未说明无法确认")

    if response.intent == "ORDER_STATUS":
        message_without_order_id = re.sub(r"\bCC-\d{4}\b", "", case["message"], flags=re.IGNORECASE)
        has_access_code = bool(re.search(r"(?<!\d)\d{4}(?!\d)", message_without_order_id))
        has_order = isinstance(response.data.get("order"), dict)
        if has_access_code and expected_state not in {"invalid_code", "order_not_found"} and not has_order:
            issues.append("提供验证码后缺少订单结构化数据")
        if not has_access_code and has_order:
            issues.append("缺少验证码却返回订单结构化数据")
        if expected_state in {"invalid_code", "order_not_found", "needs_code"}:
            if has_order:
                issues.append("未验证订单却返回订单结构化数据")
            expected_notice = {"invalid_code": "不匹配", "order_not_found": "没有找到订单", "needs_code": "验证码"}[expected_state]
            if expected_notice not in response.answer:
                issues.append(f"订单边界回答缺少提示：{expected_notice}")
        order = response.data.get("order")
        if has_order and isinstance(order, dict):
            expected_status = FULFILLMENT_LABELS.get(order.get("fulfillment_status"))
            if order.get("order_id") not in response.answer:
                issues.append("订单回答缺少订单号")
            if expected_status and expected_status not in response.answer:
                issues.append("订单回答缺少当前状态")

    if response.intent == "ESCALATE":
        case_id = response.data.get("case_id")
        escalation_case = response.escalation_case
        if not case_id or escalation_case is None:
            issues.append("人工转交缺少案件结构化数据")
        elif case_id not in response.answer:
            issues.append("人工转交回答缺少案件编号")
        llm = response.data.get("llm")
        if isinstance(llm, dict) and llm.get("status") != "skipped_high_risk":
            issues.append("高风险问题未跳过模型改写")

    return issues


def build_agent(database_path: Path, mode: str) -> SupportAgent:
    database = Database(database_path)
    database.seed_catalog(CATALOG_PATH)
    database.seed_orders(ORDERS_PATH)
    database.seed_knowledge(KNOWLEDGE_PATH)
    llm = None
    if mode == "deepseek":
        llm = DeepSeekOrchestrator.from_env()
        if llm is None:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    return SupportAgent(database, llm=llm)


def evaluate(agent: SupportAgent, cases: list[dict]) -> dict:
    results = []
    for case in cases:
        response = agent.chat(ChatRequest(message=case["message"]))
        intent_ok = response.intent == case["expected_intent"]
        missing = [text for text in case.get("expected_contains", []) if text not in response.answer]
        forbidden = [text for text in case.get("expected_excludes", []) if text in response.answer]
        issues = structural_issues(case, response)
        passed = intent_ok and not missing and not forbidden and not issues
        llm_status = response.data.get("llm", {}).get("status") if isinstance(response.data, dict) else None
        results.append({
            "id": case["id"],
            "passed": passed,
            "intent": response.intent,
            "expected_intent": case["expected_intent"],
            "missing": missing,
            "forbidden": forbidden,
            "issues": issues,
            "llm_status": llm_status,
        })

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    intent_correct = sum(1 for result in results if result["intent"] == result["expected_intent"])
    by_intent: dict[str, dict[str, int]] = {}
    expected_counts = Counter(case["expected_intent"] for case in cases)
    for intent, count in expected_counts.items():
        intent_results = [result for result in results if result["expected_intent"] == intent]
        by_intent[intent] = {
            "total": count,
            "passed": sum(1 for result in intent_results if result["passed"]),
            "intent_correct": sum(1 for result in intent_results if result["intent"] == intent),
        }
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "intent_correct": intent_correct,
        "intent_accuracy": round(intent_correct / total, 4) if total else 0,
        "llm_statuses": dict(Counter(result["llm_status"] or "deterministic" for result in results)),
        "by_intent": by_intent,
        "failures": [result for result in results if not result["passed"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 ClickyCanvas 固定评估集")
    parser.add_argument("--cases", default="data/eval_cases.json")
    parser.add_argument("--database", default="data/runtime/eval.db")
    parser.add_argument("--mode", choices=["deterministic", "deepseek"], default="deterministic")
    parser.add_argument("--limit", type=int, default=None, help="只运行前 N 条案例，联网模式建议先使用小数量")
    parser.add_argument("--case-id", action="append", help="只运行指定案例 ID；可重复传入")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    args = parser.parse_args()

    payload = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["id"] in requested]
        missing_ids = requested - {case["id"] for case in cases}
        if missing_ids:
            parser.error(f"未知案例 ID：{', '.join(sorted(missing_ids))}")
    if args.limit:
        cases = cases[: args.limit]
    report = evaluate(build_agent(ROOT / args.database, args.mode), cases)
    report["mode"] = args.mode
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"评估集：{report['total']} 条；通过：{report['passed']} 条；通过率：{report['pass_rate']:.1%}")
        print(f"意图路由准确率：{report['intent_accuracy']:.1%}")
        if report["mode"] == "deepseek":
            print(f"模型状态：{report['llm_statuses']}")
        for intent, stats in report["by_intent"].items():
            print(f"- {intent}: {stats['passed']}/{stats['total']} 通过，路由 {stats['intent_correct']}/{stats['total']}")
        if report["failures"]:
            print("失败案例：")
            for failure in report["failures"]:
                print(f"- {failure['id']}: {failure['missing'] or failure['forbidden'] or failure['issues'] or '意图不匹配'}")


if __name__ == "__main__":
    main()
