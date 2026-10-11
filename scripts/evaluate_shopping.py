"""Run synthetic multi-turn shopping cases and report failure stages."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import ChatRequest, ShoppingContext


ROOT = Path(__file__).resolve().parents[1]


def catalog_probe_cases(catalog: dict) -> list[dict]:
    """Derive fact checks from the fixture, without calling the search implementation."""
    products = catalog["products"]
    probes = []
    for product in products:
        name, product_id = product["title"], product["id"]
        for suffix, expected in (
            ("多少钱", [f"{product['price']:.0f}"]),
            ("是什么轴", [product["switch_type"]]),
            ("支持 Mac 吗", ["支持 macOS" if "macOS" in product["compatibility"] else "无法确认"]),
        ):
            probes.append({"id": f"catalog-fact-{product_id}-{len(probes)}", "source": "catalog_derived",
                           "stage": "商品事实", "turns": [{"message": f"{name}{suffix}",
                           "intent": "PRODUCT_DETAIL", "contains": expected}]})
    for axis in sorted({product["switch_type"] for product in products}):
        expected_ids = [product["id"] for product in sorted(products, key=lambda item: item["price"])
                        if product["layout"] != "Numpad" and axis in product["switch_type"]]
        probes.append({"id": f"catalog-axis-{axis}", "source": "catalog_derived", "stage": "候选召回",
                       "turns": [{"message": f"有{axis}的键盘吗", "intent": "PRODUCT_SEARCH", "ids": expected_ids}]})
    return probes


def evaluate(cases: list[dict], agent) -> dict:
    results = []
    for case in cases:
        context = None
        pending_order_id = None
        previous_answer = None
        turns = []
        for turn in case["turns"]:
            message = turn["message"]
            order_id = pending_order_id if pending_order_id and ("验证码" in message or any(ch.isdigit() for ch in message)) else None
            pending_order_id = None
            response = agent.chat(ChatRequest(message=message, shopping_context=context, order_id=order_id))
            if response.data.get("shopping_context"):
                context = ShoppingContext.model_validate(response.data["shopping_context"])
            if response.intent == "ORDER_STATUS" and response.data.get("order_id") and not response.data.get("order"):
                pending_order_id = response.data["order_id"]
            actual_ids = [product["id"] for product in response.data.get("products", [])]
            issues = []
            if response.intent != turn["intent"]:
                issues.append("需求理解：意图不符")
            for key, expected in turn.get("context", {}).items():
                if context is None or getattr(context, key) != expected:
                    issues.append(f"需求理解：{key} 不符")
            if "ids" in turn:
                expected_ids = turn["ids"]
                if set(actual_ids) != set(expected_ids):
                    issues.append("候选召回：候选集合不符")
                elif actual_ids != expected_ids:
                    issues.append("排序：候选顺序不符")
            for phrase in turn.get("contains", []):
                if phrase not in response.answer:
                    issues.append(f"回答：缺少 {phrase}")
            if turn.get("not_same_as_previous") and response.answer == previous_answer:
                issues.append("回答：重复上一轮回答")
            previous_answer = response.answer
            turns.append({"message": message, "intent": response.intent, "ids": actual_ids, "issues": issues})
        results.append({"id": case["id"], "source": case.get("source", "synthetic_seed"), "stage": case["stage"],
                        "passed": all(not turn["issues"] for turn in turns), "turns": turns})
    return {"total": len(results), "passed": sum(result["passed"] for result in results), "results": results}


def main() -> None:
    cases = json.loads((ROOT / "data" / "shopping_cases.json").read_text(encoding="utf-8"))["cases"]
    catalog = json.loads((ROOT / "data" / "catalog.json").read_text(encoding="utf-8"))
    cases = [*cases, *catalog_probe_cases(catalog)]
    runtime = ROOT / "data" / "runtime"
    suffix = uuid.uuid4().hex
    app_db = runtime / f"shopping-eval-app-{suffix}.db"
    eval_db = runtime / f"shopping-eval-{suffix}.db"
    old_database = os.environ.get("CLICKYCANVAS_DB")
    old_mode = os.environ.get("LLM_MODE")
    try:
        os.environ["CLICKYCANVAS_DB"] = str(app_db)
        os.environ["LLM_MODE"] = "deterministic"
        from scripts.evaluate_demo import build_agent
        report = evaluate(cases, build_agent(eval_db, "deterministic"))
    finally:
        if old_database is None:
            os.environ.pop("CLICKYCANVAS_DB", None)
        else:
            os.environ["CLICKYCANVAS_DB"] = old_database
        if old_mode is None:
            os.environ.pop("LLM_MODE", None)
        else:
            os.environ["LLM_MODE"] = old_mode
        for path in (app_db, eval_db):
            if path.resolve().parent == runtime.resolve():
                path.unlink(missing_ok=True)
    print(f"售前评估：{report['passed']}/{report['total']} 通过（对话案例与目录衍生探针）")
    for result in report["results"]:
        if not result["passed"]:
            for turn in result["turns"]:
                if turn["issues"]:
                    print(f"- {result['id']} / {turn['message']}：{'；'.join(turn['issues'])}；实际候选={turn['ids']}")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
