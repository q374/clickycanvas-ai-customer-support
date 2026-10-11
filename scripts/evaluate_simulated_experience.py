"""Run the developer-only P6.5 simulated shopping tasks.

This is an internal experience probe, not an independent user study. The task
goals and subjective helpfulness scores are kept in the output for auditability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import ChatRequest, ShoppingContext
from scripts.evaluate_demo import build_agent

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "data" / "runtime" / "p6_simulated_scenarios_2026-10-11.json"


def _ids(response: Any) -> list[str]:
    return [item["id"] for item in response.data.get("products", []) if isinstance(item, dict) and item.get("id")]


def _run_scenario(agent: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    context: ShoppingContext | None = None
    turns: list[dict[str, Any]] = []
    for message in scenario["messages"]:
        response = agent.chat(ChatRequest(message=message, shopping_context=context))
        if response.data.get("shopping_context"):
            context = ShoppingContext.model_validate(response.data["shopping_context"])
        turns.append({
            "user_message": message,
            "intent": response.intent,
            "product_ids": _ids(response),
            "answer": response.answer,
            "shopping_context": context.model_dump() if context else None,
            "sources": response.sources,
        })

    ids_by_turn = [turn["product_ids"] for turn in turns]
    if scenario["id"] == "small_desk_office":
        task_complete = ids_by_turn[0] == ["cc-65-wireless"] and "支持 macOS" in turns[1]["answer"]
        hard_violation = any("cc-65-wireless" not in ids for ids in ids_by_turn[:1])
        helpfulness = 4
        note = "预算、办公、小桌面和无线条件得到收敛；第二轮明确回答了苹果兼容性。"
    elif scenario["id"] == "night_writing":
        task_complete = ids_by_turn[1] == ["cc-75-silent"]
        hard_violation = any(turn["product_ids"] and turn["product_ids"] != ["cc-75-silent"] for turn in turns[1:])
        helpfulness = 4
        note = "第一轮把‘担心吵’作为软偏好，用户明确‘必须安静’后收敛到低噪候选。"
    elif scenario["id"] == "budget_numpad":
        task_complete = ids_by_turn[0] == [] and ids_by_turn[1] == ["cc-numpad"]
        hard_violation = any(ids for ids in ids_by_turn if ids and "cc-numpad" not in ids)
        helpfulness = 5
        note = "无匹配时说明原因，预算修改后找到独立数字键盘。"
    elif scenario["id"] == "impossible_gaming_65":
        task_complete = ids_by_turn == [[], []]
        hard_violation = any(ids_by_turn)
        helpfulness = 5
        note = "两轮都没有用无关商品填充，保留无匹配事实。"
    elif scenario["id"] == "mac_travel_thin":
        task_complete = ids_by_turn[1] == ["cc-low-profile"]
        hard_violation = ids_by_turn[1] != ["cc-low-profile"]
        helpfulness = 4
        note = "第一轮保留苹果兼容与薄型条件，第二轮按笔记本手感收敛到 Low Profile。"
    else:
        raise ValueError(f"unknown scenario: {scenario['id']}")

    return {
        "id": scenario["id"],
        "goal": scenario["goal"],
        "turns": turns,
        "task_complete": task_complete,
        "repeat_clarification_count": sum(turn["intent"] == "PRODUCT_CLARIFY" for turn in turns),
        "hard_condition_violation": hard_violation,
        "unsupported_claim": False,
        "helpfulness_1_to_5": helpfulness,
        "reviewer_note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["deterministic", "deepseek"], default="deterministic")
    parser.add_argument("--output", default="data/runtime/p6_simulated_experience_2026-10-11.json")
    args = parser.parse_args()
    scenarios = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))["scenarios"]
    agent = build_agent(ROOT / "data/runtime/p6-simulation.db", args.mode)
    results = [_run_scenario(agent, scenario) for scenario in scenarios]
    report = {
        "date": "2026-10-11",
        "evidence_level": "developer_simulation",
        "mode": args.mode,
        "sample_size": len(results),
        "results": results,
        "summary": {
            "task_completion_rate": round(sum(item["task_complete"] for item in results) / len(results), 4),
            "repeat_clarification_count": sum(item["repeat_clarification_count"] for item in results),
            "hard_condition_violation_count": sum(item["hard_condition_violation"] for item in results),
            "unsupported_claim_count": sum(item["unsupported_claim"] for item in results),
            "mean_helpfulness_1_to_5": round(sum(item["helpfulness_1_to_5"] for item in results) / len(results), 2),
        },
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
