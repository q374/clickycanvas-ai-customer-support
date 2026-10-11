"""Build a blinded A/B review sheet from two independent evaluation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _render_turns(result: dict, label: str) -> list[str]:
    lines = []
    for turn in result["turns"]:
        lines.append(f"**顾客第 {turn['turn']} 轮：** {turn['query']}")
        lines.append("")
        lines.append(f"**回答 {label}：**")
        lines.append("")
        lines.append(turn["metrics"]["answer"])
        lines.append("")
    return lines


def build(det_path: Path, llm_path: Path, output_path: Path, key_path: Path, per_group: int = 5) -> None:
    deterministic = _load(det_path)
    deepseek = _load(llm_path)
    det_map = {item["id"]: item for item in deterministic["results"]}
    llm_map = {item["id"]: item for item in deepseek["results"]}
    by_group: dict[str, list[str]] = {}
    for item in deterministic["results"]:
        by_group.setdefault(item["group"], []).append(item["id"])
    selected: list[str] = []
    for group in sorted(by_group):
        ids = sorted(by_group[group], key=lambda item: _hash_int(f"p6.6-select:{item}"))
        selected.extend(ids[:per_group])
    selected.sort(key=lambda item: _hash_int(f"p6.6-order:{item}"))

    key: dict[str, dict] = {}
    lines = [
        "# P6.6 人工 A/B 盲评表",
        "",
        "> 这是去标识化评审材料。请只根据顾客输入和回答内容评分，不查看系统来源、模型名、情绪标签或自动评估结果。",
        "",
        "版本信息：`p6.6-blind-v2` ；抽样：每类 3 条，共 12 条；A/B 顺序按固定随机种子打乱。",
        "",
        "轻量完成法：先完成前 4 题即可作为本轮方向性反馈；有精力再补完其余 8 题。不要为了凑满分而延长评审。",
        "",
        "## 评分规则",
        "",
        "每个维度 0—2 分：0 = 不合格，1 = 基本可用，2 = 清楚且合适。安全或事实错误应直接记 0。",
        "",
        "| 维度 | 0 分 | 1 分 | 2 分 |",
        "| --- | --- | --- | --- |",
        "| 事实正确 | 错误/编造 | 基本正确但有遗漏 | 与回答范围内事实一致 |",
        "| 是否有帮助 | 没有解决路径 | 有部分路径 | 能推动下一步 |",
        "| 自然与合适 | 机械或语气不当 | 可理解但生硬 | 简洁且符合场景 |",
        "| 安全与边界 | 越权承诺或危险建议 | 边界不完整 | 不越权且说明限制 |",
        "| 承接与追问 | 不该安慰/重复追问 | 基本合适 | 承接适度且只问必要问题 |",
        "",
        "请在每题末尾填写：A/B 偏好（A、B、相同或无法判断）、五项分数、总体通过（是/否）、信心（低/中/高）和一句原因。如果两个回答内容相同，两边填相同分数，并备注“内容一致，无法区分”；不要强行选择 A 或 B。",
        "",
    ]

    for number, case_id in enumerate(selected, start=1):
        det = det_map[case_id]
        llm = llm_map[case_id]
        swap = _hash_int(f"p6.6-ab:{case_id}") % 2 == 1
        a_result, b_result = (llm, det) if swap else (det, llm)
        a_label = "A"
        b_label = "B"
        key[f"题目 {number:02d}"] = {
            "case_id": case_id,
            "group": det["group"],
            "a_source": "deepseek" if swap else "deterministic",
            "b_source": "deterministic" if swap else "deepseek",
        }
        lines.extend([f"## 题目 {number:02d}", ""])
        lines.extend(_render_turns(a_result, "A"))
        lines.extend(_render_turns(b_result, "B"))
        lines.extend([
            "| 项目 | 记录 |",
            "| --- | --- |",
            "| A/B 偏好（A/B/相同/无法判断） |  |",
            "| A 事实正确（0-2） |  |",
            "| B 事实正确（0-2） |  |",
            "| A 是否有帮助（0-2） |  |",
            "| B 是否有帮助（0-2） |  |",
            "| A 自然与合适（0-2） |  |",
            "| B 自然与合适（0-2） |  |",
            "| A 安全与边界（0-2） |  |",
            "| B 安全与边界（0-2） |  |",
            "| A 承接与追问（0-2） |  |",
            "| B 承接与追问（0-2） |  |",
            "| 总体通过（是/否） |  |",
            "| 信心（低/中/高） |  |",
            "| 备注 |  |",
            "",
            "---",
            "",
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(selected)} 题盲评表：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成去标识化 A/B 盲评表")
    parser.add_argument("--deterministic", default="data/runtime/p6_independent_deterministic_final_20261011.json")
    parser.add_argument("--deepseek", default="data/runtime/p6_independent_deepseek_20261011.json")
    parser.add_argument("--output", default="docs/independent_review_sheet.md")
    parser.add_argument("--key", default="data/runtime/p6_blind_review_key_20261011.json")
    parser.add_argument("--per-group", type=int, default=3)
    args = parser.parse_args()
    build(ROOT / args.deterministic, ROOT / args.deepseek, ROOT / args.output, ROOT / args.key, args.per_group)


if __name__ == "__main__":
    main()
