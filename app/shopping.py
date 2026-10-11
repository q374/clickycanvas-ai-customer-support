"""Bounded, explainable extraction of keyboard shopping conditions."""

from __future__ import annotations

import re
from typing import Any

from .models import ProductSearchFilters, ShoppingContext


ORDINALS = {"一": 0, "1": 0, "二": 1, "两": 1, "2": 1, "三": 2, "3": 2, "四": 3, "4": 3, "五": 4, "5": 4}
ALLOWED_VALUES = {
    "use_case": {"办公", "游戏", "数据录入", "日常"},
    "layout": {"75%", "65%", "60%", "TKL", "Numpad"},
    "connectivity": {"Bluetooth", "USB-C", "2.4GHz", "wireless"},
    "compatibility": {"macOS", "Windows", "iPadOS", "Android"},
    "noise_level": {"低"},
    "required_tag": {"薄型", "入门", "低延迟"},
    "preference": {"quiet", "wireless"},
}
EVIDENCE_CUES = {
    "办公": ("办公", "开会", "会议", "工作", "打字"),
    "游戏": ("游戏", "电竞"),
    "数据录入": ("录入", "财务", "表格"),
    "日常": ("日常", "平时"),
    "低": ("安静", "静音", "低噪", "小声", "不吵"),
    "quiet": ("安静", "静音", "低噪", "小声", "不吵", "别太响", "声音别太大"),
    "wireless": ("无线", "蓝牙", "2.4"),
    "Bluetooth": ("蓝牙", "bluetooth"),
    "USB-C": ("usb-c", "有线"),
    "2.4GHz": ("2.4",),
    "macOS": ("mac", "苹果电脑"),
    "Windows": ("windows", "微软电脑"),
    "iPadOS": ("ipad", "平板"),
    "Android": ("android", "安卓"),
    "薄型": ("薄", "矮轴", "笔记本手感"),
    "入门": ("入门", "新手", "第一次买"),
    "低延迟": ("低延迟", "延迟低"),
}
SWITCH_TERMS = (
    "快速触发线性轴", "静音线性轴", "矮轴线性轴", "轻段落轴",
    "线性轴", "段落轴", "静音轴", "矮轴", "青轴", "红轴", "茶轴",
    "黑轴", "银轴", "白轴", "粉轴", "紫轴", "绿轴", "黄轴",
)


def ordinal_index(message: str) -> int | None:
    match = re.search(r"第([一二两三四五1-5])款", message)
    return ORDINALS[match.group(1)] if match else None


def wants_pair_comparison(message: str) -> bool:
    return bool(re.search(r"(?:这|前|上面)(?:两|2)款", message)) or (
        "哪个" in message and any(term in message for term in ["更安静", "更静音", "更适合", "哪个好"])
    )


def is_reset(message: str) -> bool:
    return any(term in message for term in ["重新开始选购", "清空选购条件", "重新选键盘", "重新推荐"])


def _add_unique(items: list[str], value: str) -> list[str]:
    return items if value in items else [*items, value]


def switch_term(message: str) -> str | None:
    """Recognize an axis type as an inventory field, including absent common axes."""
    found = next((term for term in SWITCH_TERMS if term in message), None)
    if found:
        return "静音" if found == "静音轴" else found
    unknown = re.search(r"(?:有没有|有|想要|要|需要|必须|改成|换成|不要)\s*([\u4e00-\u9fff]{1,4}轴)", message)
    if not unknown:
        unknown = re.search(r"^([\u4e00-\u9fff]{1,4}轴)(?:的|有|吗|？|\?|$)", message.strip())
    return unknown.group(1) if unknown else None


def _normalize_budget_numerals(message: str) -> str:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000}

    def replace(match: re.Match[str]) -> str:
        value = current = 0
        for char in match.group():
            if char in digits:
                current = digits[char]
            else:
                value += (current or 1) * units[char]
                current = 0
        return str(value + current)

    return re.sub(r"[零〇一二两三四五六七八九十百千]{2,8}", replace, message)


def parse_conditions(message: str) -> dict[str, Any]:
    """Only extract supported facts; later turns replace the same slot."""
    lower = message.lower()
    budget_message = _normalize_budget_numerals(message)
    patch: dict[str, Any] = {}
    if any(term in message for term in ["预算不限", "不考虑预算", "预算无所谓"]):
        patch.update(min_price=None, max_price=None, budget_unlimited=True)
    else:
        interval = re.search(r"(?<!\d)(\d{2,5})\s*(?:元|块)?\s*(?:到|至|[-~～])\s*(\d{2,5})\s*(?:元|块)?", budget_message)
        if interval:
            low, high = (int(interval.group(1)), int(interval.group(2)))
            if low <= high:
                patch.update(min_price=float(low), max_price=float(high), budget_unlimited=False)
        else:
            upper = re.search(r"(?:预算\s*(?:是|为|约|大约|提高到|改成)?\s*|(?:价格|金额)?\s*(?:不超过|别超过|不要超过|不高于|低于|至多|最多)\s*)(\d{2,5})", budget_message)
            if not upper:
                upper = re.search(r"(?<!\d)(\d{2,5})\s*(?:元|块|人民币|cny)?\s*(?:以内|以下)", budget_message.lower())
            if upper:
                patch.update(min_price=None, max_price=float(upper.group(1)), budget_unlimited=False)

    if any(term in message for term in ["办公", "办公室", "共享空间", "共享工位"]):
        patch["use_case"] = "办公"
    elif any(term in message for term in ["游戏", "电竞"]):
        patch["use_case"] = "游戏"
    elif any(term in message for term in ["数据录入", "财务", "表格"]):
        patch["use_case"] = "数据录入"
    elif any(term in message for term in ["日常", "平时用"]):
        patch["use_case"] = "日常"

    layout = re.search(r"(?<!\d)(75|65|60)\s*(?:%|配列)", message)
    if layout:
        patch["layout"] = f"{layout.group(1)}%"
    elif "TKL" in message.upper():
        patch["layout"] = "TKL"
    elif "数字键盘" in message or "数字区" in message:
        patch["layout"] = "Numpad"

    axis = switch_term(message)
    if axis:
        if re.search(r"(?:不要|不想要|不需要|别用|排除).{0,3}" + re.escape(axis), message):
            patch.update(switch_type=None, excluded_switch_types=[axis])
        else:
            patch.update(switch_type=axis, excluded_switch_types=[])

    negative_bluetooth = bool(re.search(r"(?:不要|不想要|不需要|别带|不带|排除|没有)\s*蓝牙", message))
    negative_wireless = bool(re.search(r"(?:不要|不想要|不需要|排除)\s*无线", message))
    if "有线" in message and negative_wireless:
        patch.update(connectivity="USB-C", excluded_connectivity=["Bluetooth", "2.4GHz"])
    elif negative_bluetooth:
        patch.update(connectivity=None, excluded_connectivity=["Bluetooth"])
    elif "蓝牙" in message or "bluetooth" in lower:
        patch["excluded_connectivity"] = []
        if any(term in message for term in ["希望", "最好", "尽量", "优先"]):
            patch["preferences_add"] = ["wireless"]
        else:
            patch["connectivity"] = "Bluetooth"
    elif "2.4ghz" in lower:
        patch.update(connectivity="2.4GHz", excluded_connectivity=[])
    elif "usb-c" in lower or "有线" in message:
        patch["connectivity"] = "USB-C"
    elif "无线" in message and not negative_wireless:
        if any(term in message for term in ["希望", "最好", "尽量", "优先"]):
            patch["preferences_add"] = ["wireless"]
        else:
            patch["connectivity"] = "wireless"

    if "mac" in lower or "苹果电脑" in message:
        patch["compatibility"] = "macOS"
    elif "windows" in lower:
        patch["compatibility"] = "Windows"

    negative_thin = bool(re.search(r"(?:不要|不需要|不想要|不用)\s*(?:薄型|薄键盘)", message))
    if negative_thin:
        patch["required_tags_remove"] = ["薄型"]
    elif any(term in message for term in ["薄型", "薄键盘", "薄一些", "笔记本手感", "笔记本那种薄手感"]):
        patch["required_tags_add"] = ["薄型"]
    if "低延迟" in message:
        patch["required_tags_add"] = _add_unique(patch.get("required_tags_add", []), "低延迟")
    if "入门" in message or "第一次买" in message:
        patch["required_tags_add"] = _add_unique(patch.get("required_tags_add", []), "入门")
    quiet = any(term in message for term in ["安静", "静音", "低噪", "声音小", "声音比较小", "不要吵", "别太响", "别太吵"])
    if quiet:
        negative_quiet = bool(re.search(r"(?:不要|不需要|不想要|不用)\s*(?:安静|静音|低噪)", message))
        quiet_required = bool(re.search(r"(?:必须|只要|一定要|务必).{0,5}(?:安静|静音|低噪|声音小)", message))
        if "希望安静" in message or "最好安静" in message:
            quiet_required = False
            patch["noise_level"] = None
        if negative_quiet:
            patch["noise_level"] = None
            patch["preferences_remove"] = ["quiet"]
        elif quiet_required:
            patch["noise_level"] = "低"
            patch["preferences_remove"] = ["quiet"]
        else:
            patch["preferences_add"] = _add_unique(patch.get("preferences_add", []), "quiet")
    return patch


def merge_context(previous: ShoppingContext | None, patch: dict[str, Any]) -> ShoppingContext:
    data = (previous or ShoppingContext()).model_dump()
    for key, value in patch.items():
        if key == "required_tags_add":
            for item in value:
                data["required_tags"] = _add_unique(data["required_tags"], item)
        elif key == "required_tags_remove":
            data["required_tags"] = [item for item in data["required_tags"] if item not in value]
        elif key == "preferences_add":
            for item in value:
                data["preferences"] = _add_unique(data["preferences"], item)
        elif key == "preferences_remove":
            data["preferences"] = [item for item in data["preferences"] if item not in value]
        else:
            data[key] = value
    if patch:
        data["last_product_ids"] = []
    return ShoppingContext.model_validate(data)


def context_memory_events(previous: ShoppingContext | None, current: ShoppingContext, source: str = "rules") -> list[dict[str, Any]]:
    """把本轮槽位变化转成可复盘事件；不记录完整聊天历史或身份信息。"""
    before = (previous or ShoppingContext()).model_dump()
    after = current.model_dump()
    ignored = {"last_product_ids", "recommendation_pending"}
    events: list[dict[str, Any]] = []
    for field in (
        "min_price", "max_price", "budget_unlimited", "use_case", "layout", "switch_type",
        "excluded_switch_types", "connectivity", "excluded_connectivity", "compatibility",
        "noise_level", "required_tags", "preferences",
    ):
        if field in ignored or before.get(field) == after.get(field):
            continue
        events.append({"field": field, "before": before.get(field), "after": after.get(field), "source": source})
    return events


def missing_recommendation_slots(context: ShoppingContext) -> list[str]:
    if any((context.use_case, context.layout, context.switch_type, context.connectivity,
            context.excluded_connectivity, context.excluded_switch_types, context.compatibility,
            context.noise_level, context.required_tags, context.preferences,
            context.min_price is not None, context.max_price is not None, context.budget_unlimited)):
        return []
    return ["主要使用场景或在意的键盘特点"]


def filters_for(context: ShoppingContext) -> ProductSearchFilters:
    return ProductSearchFilters(
        full_keyboards_only=context.layout != "Numpad" and context.use_case != "数据录入",
        min_price=context.min_price,
        max_price=context.max_price,
        use_case=context.use_case,
        layout=context.layout,
        switch_type=context.switch_type,
        excluded_switch_types=context.excluded_switch_types,
        connectivity=context.connectivity,
        excluded_connectivity=context.excluded_connectivity,
        compatibility=context.compatibility,
        noise_level=context.noise_level,
        required_tags=context.required_tags,
        preferences=context.preferences,
    )


def criteria_label(context: ShoppingContext) -> str:
    parts = []
    if context.min_price is not None and context.max_price is not None:
        parts.append(f"预算{context.min_price:.0f}—{context.max_price:.0f}元")
    elif context.max_price is not None:
        parts.append(f"预算不超过{context.max_price:.0f}元")
    if context.use_case:
        parts.append(f"{context.use_case}用途")
    if context.layout:
        parts.append(f"{context.layout}配列")
    if context.switch_type:
        parts.append(f"{context.switch_type}轴体")
    if context.excluded_switch_types:
        parts.append("不要" + "、".join(context.excluded_switch_types))
    if context.connectivity:
        if context.connectivity == "wireless":
            parts.append("无线连接")
        elif context.connectivity == "Bluetooth":
            parts.append("支持蓝牙（Bluetooth）")
        else:
            parts.append(f"支持{context.connectivity}")
    if context.excluded_connectivity:
        parts.append("不带" + "、".join(context.excluded_connectivity))
    if context.compatibility:
        parts.append(f"兼容{context.compatibility}")
    if context.noise_level:
        parts.append("低噪声")
    for tag in context.required_tags:
        parts.append(tag)
    for preference in context.preferences:
        parts.append("优先安静" if preference == "quiet" else "优先无线")
    return "、".join(parts) or "当前筛选条件"


def valid_model_suggestions(message: str, raw: Any, rule_patch: dict[str, Any], previous: ShoppingContext | None = None) -> dict[str, Any]:
    """Accept only supported values backed by an exact quote from this turn."""
    if not isinstance(raw, dict) or not isinstance(raw.get("suggestions"), list):
        raise ValueError("invalid shopping extraction")
    suggestions = raw["suggestions"]
    if len(suggestions) > 5:
        raise ValueError("too many shopping suggestions")
    patch: dict[str, Any] = {}
    for item in suggestions:
        if not isinstance(item, dict):
            raise ValueError("invalid shopping suggestion")
        field, value, evidence = item.get("field"), item.get("value"), item.get("evidence")
        if field not in ALLOWED_VALUES or value not in ALLOWED_VALUES[field]:
            raise ValueError("unsupported shopping suggestion")
        if not isinstance(evidence, str) or not evidence.strip() or evidence not in message:
            raise ValueError("shopping suggestion has no quoted evidence")
        if re.search(r"(?:不要|不需要|不想要|不用|排除)", evidence):
            raise ValueError("negated shopping suggestion")
        cues = EVIDENCE_CUES.get(value, (value.lower(),))
        if not any(cue.lower() in evidence.lower() for cue in cues):
            raise ValueError("shopping suggestion is not supported by its quote")
        target = {"required_tag": "required_tags_add", "preference": "preferences_add"}.get(field, field)
        if target in rule_patch:
            continue
        if target == "connectivity" and previous and value in previous.excluded_connectivity:
            raise ValueError("shopping suggestion conflicts with excluded connection")
        if target in {"required_tags_add", "preferences_add"}:
            patch[target] = _add_unique(patch.get(target, []), value)
        else:
            patch[target] = value
    return patch
