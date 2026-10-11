from __future__ import annotations

import logging
import re
from typing import Any

from .db import Database
from .emotion import classify_emotion, merge_deepseek_emotion, strategy_for_emotion, style_deterministic_answer
from .llm import DeepSeekOrchestrator
from .models import ChatRequest, ChatResponse, Product, ProductSearchFilters, ShoppingContext
from .tools import ACTIVE_TOOL_EXECUTOR, BoundedToolExecutor, ToolBudgetExceeded, invoke_tool
from .shopping import (
    context_memory_events, criteria_label, filters_for, is_reset, merge_context, missing_recommendation_slots,
    ordinal_index, parse_conditions, valid_model_suggestions, wants_pair_comparison,
    switch_term,
)


LOGGER = logging.getLogger(__name__)


ORDER_ID_RE = re.compile(r"\bCC-\d{4}\b", flags=re.IGNORECASE)
CODE_RE = re.compile(r"(?<!CC-)(?<!\d)\d{4}(?!\d)", flags=re.IGNORECASE)
PRODUCT_ID_RE = re.compile(r"\bcc-[a-z0-9]+(?:-[a-z0-9]+)+\b", flags=re.IGNORECASE)

PRODUCT_TERMS = [
    "办公室", "共享空间", "静音", "低噪", "无线", "有线", "蓝牙", "2.4GHz",
    "Mac", "macOS", "iPad", "Windows", "Android", "游戏", "电竞", "入门",
    "预算", "紧凑", "薄型", "笔记本", "数字键盘", "快捷键", "75%", "65%", "60%", "TKL",
]
PRODUCT_CONTEXT_TERMS = ["键盘", "键帽", "数字区", "商品", "产品", "机械轴", "轴体", *PRODUCT_TERMS]
GREETING_RE = re.compile(r"^(?:你好|您好|嗨|哈喽|hello|hi|早上好|上午好|中午好|下午好|晚上好|在吗|有人吗)[呀啊嘛吗！!，,。.?？~～\s]*$", re.IGNORECASE)
RECOMMENDATION_TERMS = ["推荐", "帮我挑", "帮我选", "选购", "想买"]
USAGE_TERMS = ["办公室", "办公", "共享空间", "游戏", "电竞", "数据录入", "打字", "日常", "学习", "编程"]

POLICY_TERMS = [
    "发货", "配送", "物流", "退货", "退换", "退款", "保修", "保价", "政策", "进水", "按键失灵", "无法连接",
    "进液", "按键没有反应", "按键没反应", "连不上", "可以退", "交给快递", "换新", "赔付", "终身", "自燃",
    "配对失败", "无线配对", "蓝牙配对", "配不上", "没反应", "按下去没反应", "快递两天没动静", "送到国外", "海外", "国外", "寄到",
    "无理由退", "拆封", "签收", "不喜欢", "价格保护", "出库",
]

ESCALATION_TERMS = [
    "退款争议", "退款金额", "投诉", "赔偿", "严重损坏", "到货损坏", "破损", "法律",
    "我要退款", "退款一直", "退款没", "退款未", "人工处理", "找人工", "转人工", "人工核对",
]


FULFILLMENT_LABELS = {
    "processing": "处理中",
    "shipped": "已发货",
    "delayed": "物流延迟",
    "delivered": "已送达",
}


def _find_order_id(message: str, request_order_id: str | None) -> str | None:
    if request_order_id:
        return request_order_id.strip().upper()
    match = ORDER_ID_RE.search(message)
    return match.group(0).upper() if match else None


def _find_access_code(message: str, request_code: str | None) -> str | None:
    if request_code:
        return request_code.strip()
    match = CODE_RE.search(message)
    return match.group(0) if match else None


def _find_product_ids(message: str) -> list[str]:
    seen: set[str] = set()
    product_ids: list[str] = []
    for match in PRODUCT_ID_RE.findall(message):
        product_id = match.lower()
        if product_id not in seen:
            seen.add(product_id)
            product_ids.append(product_id)
    return product_ids[:4]


def _product_terms(message: str) -> list[str]:
    lower = message.lower()
    found: list[str] = []
    for term in PRODUCT_TERMS:
        if term.lower() in lower and term not in found:
            found.append(term)
    for layout in re.findall(r"(?<!\d)(75|65|60)(?:%|配列)?", message):
        normalized = f"{layout}%"
        if normalized not in found:
            found.append(normalized)
    for phrase, term in {"苹果电脑": "macOS", "薄键盘": "薄型", "办公": "办公室"}.items():
        if phrase in message and term not in found:
            found.append(term)
    return found


def _has_product_context(message: str) -> bool:
    lower = message.lower()
    return bool(_find_product_ids(message)) or any(term.lower() in lower for term in PRODUCT_CONTEXT_TERMS)


def _is_recommendation_request(message: str) -> bool:
    lower = message.lower()
    return any(term.lower() in lower for term in RECOMMENDATION_TERMS)


def _missing_recommendation_slots(message: str) -> list[str]:
    lower = message.lower()
    missing: list[str] = []
    if not any(term in lower for term in USAGE_TERMS):
        missing.append("主要使用场景（办公、游戏或日常）")
    has_budget = bool(
        re.search(
            r"(?:预算\s*(?:是|为|约|大约)?\s*\d{2,5}|\d{2,5}\s*(?:元|块|人民币|cny)|(?:不超过|不高于|低于|以内)\s*\d{2,5})",
            lower,
        )
    )
    has_budget = has_budget or any(term in lower for term in ["预算不限", "预算无所谓", "不考虑预算"])
    if not has_budget:
        missing.append("预算范围")
    return missing


def _product_to_card(product: Product) -> dict[str, Any]:
    return {
        "id": product.id,
        "title": product.title,
        "price": product.price,
        "currency": product.currency,
        "layout": product.layout,
        "switch_type": product.switch_type,
        "noise_level": product.noise_level,
        "connectivity": product.connectivity,
        "compatibility": product.compatibility,
        "use_cases": product.use_cases,
        "tags": product.tags,
        "available_for_sale": product.available_for_sale,
        "source_url": product.source_url,
        "fetched_at": product.fetched_at,
    }


def _preserves_required_facts(deterministic: ChatResponse, answer: str) -> bool:
    """Reject rewrites that drop a required verification or comparison fact."""
    if deterministic.intent == "PRODUCT_SEARCH" and deterministic.data.get("products") == []:
        if "没有找到" not in answer:
            return False
        context = deterministic.data.get("shopping_context", {})
        axis = context.get("switch_type") if isinstance(context, dict) else None
        if axis and "没有标注" in deterministic.answer:
            return axis in answer and "没有" in answer
        return True

    if deterministic.intent == "PRODUCT_SEARCH":
        products = deterministic.data.get("products", [])
        if any(product["title"] not in answer and product["id"] not in answer for product in products[:3]):
            return False
        if any(f"{product['price']:.0f}" not in answer or product["switch_type"] not in answer
               for product in products[:3]):
            return False
        known_ids = {product["id"] for product in products}
        if any(product_id not in known_ids for product_id in PRODUCT_ID_RE.findall(answer)):
            return False

    if deterministic.intent == "PRODUCT_DETAIL":
        product = deterministic.data.get("product", {})
        if product.get("id") not in answer and product.get("title") not in answer:
            return False
        if "无法确认" in deterministic.answer and "无法确认" not in answer:
            return False
        if "标注支持 macOS" in deterministic.answer and "支持 macOS" not in answer:
            return False
        if "未标注" in deterministic.answer and "未标注" not in answer:
            return False
        if "标注支持蓝牙" in deterministic.answer and "支持蓝牙" not in answer:
            return False
        switch = product.get("switch_type")
        if switch and switch in deterministic.answer and switch not in answer:
            return False

    if deterministic.intent == "PRODUCT_COMPARE":
        product_ids = deterministic.data.get("product_ids", [])
        products = deterministic.data.get("products", [])
        if len(product_ids) < 2:
            return "至少两个" in answer
        for product in products:
            if product.get("title") not in answer and product.get("id") not in answer:
                return False
        if "无法据此判断哪款实际更安静" in deterministic.answer and "无法" not in answer:
            return False
        if "低于" in deterministic.answer and "低于" not in answer:
            return False

    if deterministic.intent == "ORDER_STATUS":
        order = deterministic.data.get("order")
        if not isinstance(order, dict):
            return "验证码" in answer
        if order.get("order_id") not in answer:
            return False
        status_label = FULFILLMENT_LABELS.get(order.get("fulfillment_status"))
        if status_label and status_label not in answer:
            return False

    if deterministic.intent == "POLICY_FAQ" and not deterministic.data.get("documents"):
        return "没有找到明确的政策说明" in answer

    return True


class SupportAgent:
    def __init__(self, database: Database, llm: DeepSeekOrchestrator | None = None):
        self.database = database
        self.llm = llm

    def chat(self, request: ChatRequest) -> ChatResponse:
        executor = BoundedToolExecutor(max_calls=3)
        token = ACTIVE_TOOL_EXECUTOR.set(executor)
        rule_emotion = classify_emotion(request.message)
        try:
            deterministic = self._deterministic_chat(request)
            # Safety wording such as smoke, electric shock or a serious injury
            # must enter the existing human workflow even if the business
            # router did not see a legacy escalation keyword.
            if rule_emotion.needs_human and deterministic.intent != "ESCALATE":
                deterministic = self._escalate(request.message, request)
            if (request.shopping_context is not None or deterministic.intent == "PRODUCT_COMPARE") and "shopping_context" not in deterministic.data:
                data = dict(deterministic.data)
                context = request.shopping_context.model_copy(deep=True) if request.shopping_context else ShoppingContext()
                if deterministic.intent == "PRODUCT_COMPARE" and data.get("products"):
                    context.last_product_ids = [product["id"] for product in data["products"]]
                data["shopping_context"] = context.model_dump()
                deterministic = deterministic.model_copy(update={"data": data})
        except ToolBudgetExceeded:
            deterministic = ChatResponse(
                intent="GENERAL",
                answer="本轮操作步骤超过安全上限，请把问题拆成一到两个具体问题后再试。",
                data={},
            )
        finally:
            ACTIVE_TOOL_EXECUTOR.reset(token)
        data = dict(deterministic.data)
        data["agent"] = executor.trace()
        data["emotion"] = rule_emotion.model_dump()
        data["emotion_strategy"] = strategy_for_emotion(rule_emotion).model_dump()
        deterministic = deterministic.model_copy(update={
            "answer": style_deterministic_answer(deterministic.answer, rule_emotion, deterministic.intent, request.message),
            "data": data,
        })
        if self.llm is None:
            return deterministic
        if deterministic.intent in {"ESCALATE", "GENERAL", "PRODUCT_CLARIFY"}:
            data = dict(deterministic.data)
            status = {
                "ESCALATE": "skipped_high_risk",
                "GENERAL": "skipped_general",
                "PRODUCT_CLARIFY": "skipped_clarification",
            }[deterministic.intent]
            data["llm"] = {"status": status, "model": self.llm.model}
            return deterministic.model_copy(update={"data": data})
        try:
            draft = self.llm.rewrite(request.message, deterministic)
            if not _preserves_required_facts(deterministic, draft.answer):
                raise ValueError("DeepSeek rewrite omitted required deterministic facts")
        except Exception as exc:
            LOGGER.warning("DeepSeek rewrite fallback (%s): %s", type(exc).__name__, str(exc)[:200])
            data = dict(deterministic.data)
            data["llm"] = {"status": "fallback", "model": self.llm.model}
            return deterministic.model_copy(update={"data": data})
        data = dict(deterministic.data)
        data["llm"] = {
            "status": "rewritten",
            "model": self.llm.model,
            "intent": draft.intent,
            "confidence": draft.confidence,
        }
        emotion = merge_deepseek_emotion(
            rule_emotion,
            state=draft.emotion_state,
            urgency=draft.emotion_urgency,
            confidence=draft.emotion_confidence,
            needs_human=draft.emotion_needs_human,
        )
        data["emotion"] = emotion.model_dump()
        data["emotion_strategy"] = strategy_for_emotion(emotion).model_dump()
        return deterministic.model_copy(update={
            "answer": style_deterministic_answer(draft.answer, emotion, deterministic.intent, request.message),
            "data": data,
        })

    def _deterministic_chat(self, request: ChatRequest) -> ChatResponse:
        message = request.message.strip()
        lower = message.lower()

        if any(term in lower for term in [term.lower() for term in ESCALATION_TERMS]):
            return self._escalate(message, request)

        order_id = _find_order_id(message, request.order_id)
        policy_message = any(term.lower() in lower for term in POLICY_TERMS)
        if order_id or (any(term in message for term in ["订单", "快递", "到哪", "到哪里", "发到哪里"]) and not policy_message):
            return self._order_status(message, request, order_id)

        if any(term.lower() in lower for term in POLICY_TERMS):
            return self._policy_answer(message)

        if is_reset(message):
            return ChatResponse(
                intent="PRODUCT_CLARIFY",
                answer="选购条件已清空。请告诉我主要使用场景或在意的键盘特点；预算可以之后再补。",
                data={"shopping_context": ShoppingContext(recommendation_pending=True).model_dump()},
            )

        context = request.shopping_context
        index = ordinal_index(message)
        if index is not None:
            ids = context.last_product_ids if context else []
            if index >= len(ids):
                return self._reference_clarification(context)
            return self._product_reference_answer(message, ids[index], context)

        if any(term in message for term in ["这款", "这个键盘", "它"]):
            ids = context.last_product_ids if context else []
            if len(ids) != 1:
                return self._reference_clarification(context)
            return self._product_reference_answer(message, ids[0], context)

        product_ids = _find_product_ids(message)
        if len(product_ids) < 2 and any(term in message for term in ["比较", "对比", "哪个", "谁", "区别"]):
            normalized = message.lower().replace(" ", "")
            named_ids: list[str] = []
            for product in self.database.list_products(ProductSearchFilters(limit=20)):
                names = [product.title, product.handle, product.id]
                words = product.title.split()
                if len(words) >= 2:
                    names.append(" ".join(words[:2]))
                if any(name.lower().replace(" ", "") in normalized for name in names) and product.id not in named_ids:
                    named_ids.append(product.id)
            if len(named_ids) >= 2:
                product_ids = named_ids[:4]

        if wants_pair_comparison(message) and not _find_product_ids(message):
            ids = context.last_product_ids if context else []
            if len(product_ids) >= 2:
                return self._compare_answer(message, product_ids)
            if re.search(r"(?:前|上面)(?:两|2)款", message) and len(ids) >= 2:
                ids = ids[:2]
            elif len(ids) != 2:
                return self._reference_clarification(context)
            return self._compare_answer(message, ids)

        if len(product_ids) < 2 and any(term in message for term in ["比较", "对比", "哪个", "区别"]):
            normalized = message.lower().replace(" ", "")
            named_ids: list[str] = []
            for product in self.database.list_products(ProductSearchFilters(limit=20)):
                if product.title.lower().replace(" ", "") in normalized and product.id not in named_ids:
                    named_ids.append(product.id)
            if len(named_ids) >= 2:
                product_ids = named_ids[:4]
        is_compare_request = (
            "对比" in message
            or "区别" in message
            or ("比较" in message and len(product_ids) >= 2)
            or (len(product_ids) >= 2 and any(term in message for term in ["哪个", "谁", "更适合", "怎么选"]))
        )
        if is_compare_request:
            return self._compare_answer(message, product_ids)

        named_product = self._named_product(message)
        if named_product is not None and any(term in lower for term in [
            "支持", "兼容", "多少钱", "价格", "轴", "连接", "蓝牙", "噪声", "安静", "配列", "参数", "适合",
            "怎么样", "好不好", "值得", "介绍", "热插拔", "保证", "延迟", "实际", "手感",
        ]):
            return self._product_reference_answer(message, named_product.id, context)

        if _is_recommendation_request(message) or _has_product_context(message) or parse_conditions(message) or (
            context is not None and (context.recommendation_pending or bool(parse_conditions(message)))
        ):
            return self._product_answer(message, context, _is_recommendation_request(message))

        return self._general_answer(message)

    def _product_clarification_answer(self, missing_slots: list[str]) -> ChatResponse:
        requested = "和".join(missing_slots)
        answer = (
            f"可以帮您推荐键盘。为了缩小范围，请告诉我{requested}。"
            "如果有偏好，也可以一并说是否需要蓝牙、希望安静或特定配列。"
        )
        return ChatResponse(
            intent="PRODUCT_CLARIFY",
            answer=answer,
            data={"missing_slots": missing_slots},
        )

    def _general_answer(self, message: str) -> ChatResponse:
        greeting_with_presence = bool(re.fullmatch(r"(?:你好|您好|嗨|哈喽)[，,]\s*在吗[？?!！。]*", message.strip(), flags=re.IGNORECASE))
        if GREETING_RE.fullmatch(message.strip()) or greeting_with_presence:
            answer = "您好！我是 ClickyCanvas 演示客服，可以帮您挑选或比较键盘、咨询配送/退货/保修政策，或查询模拟订单。您想了解哪方面？"
        elif any(term in message for term in ["谢谢", "感谢", "多谢"]):
            answer = "不客气！如果需要，我可以帮您挑选或比较键盘、咨询配送/退货/保修政策，或查询模拟订单。"
        else:
            answer = "我目前主要能帮您挑选或比较键盘、咨询配送/退货/保修政策，或查询模拟订单。您想了解哪方面？如果是选键盘，可以告诉我预算、连接方式或使用场景。"
        return ChatResponse(intent="GENERAL", answer=answer)

    def _named_product(self, message: str) -> Product | None:
        lower = message.lower().replace(" ", "")
        for product in self.database.list_products(ProductSearchFilters(limit=20)):
            words = product.title.split()
            names = [product.title, product.handle, product.id]
            if len(words) >= 2:
                names.append(" ".join(words[:2]))
            if any(name.lower().replace(" ", "") in lower for name in names):
                return product
        return None

    def _reference_clarification(self, context: ShoppingContext | None) -> ChatResponse:
        return ChatResponse(
            intent="PRODUCT_CLARIFY",
            answer="我还不能确定您指的是哪款商品。请提供商品 ID，或明确说第一款、第二款；比较时请指定两款。",
            data={"shopping_context": (context or ShoppingContext()).model_dump()},
        )

    def _product_reference_answer(self, message: str, product_id: str, context: ShoppingContext | None) -> ChatResponse:
        product = invoke_tool(
            "get_product_detail", {"product_id": product_id},
            lambda: self.database.get_product(product_id),
        )
        if product is None:
            return self._reference_clarification(context)
        card = _product_to_card(product)
        lower = message.lower()
        if any(term in message for term in ["多少钱", "价格", "价位"]) and any(term in message for term in ["连接", "蓝牙", "无线", "有线"]):
            connection_text = "、".join(
                "蓝牙（" + value + "）" if "bluetooth" in value.lower() else value
                for value in product.connectivity
            )
            conclusion = f"价格为{product.price:.0f} {product.currency}；连接方式为{connection_text}。"
        elif "mac" in lower or "苹果电脑" in message:
            conclusion = "商品资料显示支持 macOS。" if "macOS" in product.compatibility else "商品资料未标注 macOS，无法确认是否支持。"
        elif any(term in message for term in ["多少钱", "价格", "价位"]):
            conclusion = f"价格为{product.price:.0f} {product.currency}。"
        elif "轴" in message:
            requested_axis = switch_term(message)
            if requested_axis:
                matches = requested_axis.lower() in product.switch_type.lower()
                conclusion = (f"商品资料显示为{product.switch_type}，符合您问的{requested_axis}。" if matches else
                              f"商品资料未标注{requested_axis}；轴体是{product.switch_type}。")
            else:
                conclusion = f"商品资料显示轴体为{product.switch_type}。"
        elif "热插拔" in message:
            conclusion = "商品资料没有提供热插拔信息，无法确认。"
        elif any(term in message for term in ["保证", "延迟", "实际手感"]):
            conclusion = "商品资料没有提供这项实测信息，无法替您断言；实际手感和延迟需要以真实使用为准。"
        elif "蓝牙" in message:
            supports = any("bluetooth" in value.lower() for value in product.connectivity)
            conclusion = ("商品资料显示支持蓝牙，连接方式为" if supports else "商品资料未标注蓝牙，连接方式为") + "、".join(product.connectivity) + "。"
        elif any(term in message for term in ["蓝牙", "连接", "无线"]):
            conclusion = f"商品资料显示连接方式为{'、'.join(product.connectivity)}。"
        elif any(term in message for term in ["游戏", "办公", "打字", "适合"]):
            conclusion = f"商品资料列出的适用场景为{'、'.join(product.use_cases)}；未列出的用途无法确认。"
        elif any(term in message for term in ["安静", "静音", "噪声"]):
            conclusion = f"商品资料中标为{product.noise_level}噪声，实际听感仍会受环境影响。"
        elif any(term in message for term in ["支持", "兼容", "能否", "可以"]):
            conclusion = "商品资料没有提供您问的这项信息，无法确认。"
        else:
            conclusion = (
                f"价格为{product.price:.0f} {product.currency}，配列为{product.layout}，轴体为{product.switch_type}，"
                f"资料中标为{product.noise_level}噪声；连接方式为{'、'.join(product.connectivity)}，"
                f"适用场景为{'、'.join(product.use_cases)}。实际手感没有实测资料，无法替您断言好不好用。"
            )
        return ChatResponse(
            intent="PRODUCT_DETAIL",
            answer=f"{product.title}（{product.id}）：{conclusion}",
            data={"product": card, "shopping_context": (context or ShoppingContext()).model_dump()},
            sources=["ClickyCanvas 商品快照"],
        )

    def _compare_answer(self, message: str, product_ids: list[str] | None = None) -> ChatResponse:
        product_ids = product_ids if product_ids is not None else _find_product_ids(message)
        if len(product_ids) < 2:
            return ChatResponse(
                intent="PRODUCT_COMPARE",
                answer="请提供至少两个商品 ID 再进行对比，例如：比较 cc-75-silent 和 cc-65-wireless。",
                data={"product_ids": product_ids},
                sources=["ClickyCanvas 商品快照"],
            )
        products = invoke_tool(
            "compare_products", {"product_ids": product_ids},
            lambda: self.database.compare_products(product_ids),
        )
        if len(products) < 2:
            found_ids = {product.id for product in products}
            missing = [product_id for product_id in product_ids if product_id not in found_ids]
            return ChatResponse(
                intent="PRODUCT_COMPARE",
                answer=f"无法找到可对比的商品：{', '.join(missing)}。请检查商品 ID。",
                data={"product_ids": product_ids},
                sources=["ClickyCanvas 商品快照"],
            )
        cards = [_product_to_card(product) for product in products]
        lines = ["我把这几款的主要差异列出来："]
        for card in cards:
            lines.append(
                f"- {card['title']}：{card['price']:.0f} {card['currency']}；{card['layout']}；{card['switch_type']}；"
                f"噪声{card['noise_level']}；连接方式：{'、'.join(card['connectivity'])}。"
            )
        if any(term in message for term in ["更安静", "更静音", "哪个安静", "哪个噪声", "声音更小"]):
            noise_levels = {card["noise_level"] for card in cards}
            if len(noise_levels) == 1:
                lines.append(f"按商品资料的噪声标注，两款均为{next(iter(noise_levels))}噪声，无法据此判断哪款实际更安静。")
            else:
                ranks = {"低": 0, "中": 1, "高": 2}
                ranked = sorted(cards, key=lambda card: ranks.get(card["noise_level"], 3))
                lines.append(
                    f"按商品资料标注，{ranked[0]['title']}为{ranked[0]['noise_level']}噪声，"
                    f"低于{ranked[-1]['title']}的{ranked[-1]['noise_level']}噪声；实际听感仍会受环境影响。"
                )
        lines.append("我只比较商品资料里明确提供的字段，未提供的体验和性能不会猜测。")
        return ChatResponse(
            intent="PRODUCT_COMPARE",
            answer="\n".join(lines),
            data={"products": cards, "product_ids": product_ids},
            sources=["ClickyCanvas 商品快照"],
        )

    def _product_answer(self, message: str, previous: ShoppingContext | None = None, recommendation: bool = False) -> ChatResponse:
        patch = parse_conditions(message)
        extraction = "rules"
        if self.llm is not None:
            try:
                suggestions = self.llm.extract_shopping(message)
                supplemental = valid_model_suggestions(message, suggestions, patch, previous)
                patch = {**supplemental, **patch}
                extraction = "model" if supplemental else ("rules" if patch else "clarify")
            except Exception as exc:
                LOGGER.warning("DeepSeek shopping extraction fallback (%s): %s", type(exc).__name__, str(exc)[:200])
                extraction = "fallback"
        context = merge_context(previous, patch)
        memory_events = context_memory_events(previous, context, source=extraction)
        if recommendation or (previous is not None and previous.recommendation_pending):
            missing = missing_recommendation_slots(context)
            if missing:
                context.recommendation_pending = True
                result = self._product_clarification_answer(missing)
                return result.model_copy(update={"data": {
                    **result.data, "shopping_context": context.model_dump(), "shopping_extraction": extraction,
                    "shopping_memory": {"updates": memory_events, "scope": "page_session"},
                }})
        context.recommendation_pending = False
        if previous is not None and previous.recommendation_pending and not patch:
            result = self._product_clarification_answer(missing_recommendation_slots(context) or ["要补充的具体条件"])
            return result.model_copy(update={"data": {
                **result.data, "shopping_context": context.model_dump(), "shopping_extraction": extraction,
                "shopping_memory": {"updates": memory_events, "scope": "page_session"},
            }})
        filters = filters_for(context)
        products = invoke_tool(
            "search_products", {"filters": filters.model_dump()},
            lambda: self.database.list_products(filters),
        )
        criteria = criteria_label(context)
        context.last_product_ids = [product.id for product in products]
        if not products:
            no_axis_in_store = bool(context.switch_type) and not invoke_tool(
                "search_products", {"filters": {"switch_type": context.switch_type, "limit": 1}},
                lambda: self.database.list_products(ProductSearchFilters(switch_type=context.switch_type, limit=1)),
            )
            detail = (f"商品资料里没有标注{context.switch_type}的键盘。" if no_axis_in_store else
                      "请告诉我您愿意调整哪一项，我再重新筛选。")
            return ChatResponse(
                intent="PRODUCT_SEARCH",
                answer=f"按{criteria}，我没有找到同时满足这些条件的商品。{detail}",
                data={"products": [], "filters": filters.model_dump(), "criteria": criteria,
                      "shopping_context": context.model_dump(), "shopping_extraction": extraction,
                      "shopping_memory": {"updates": memory_events, "scope": "page_session"}},
                sources=["ClickyCanvas 商品快照"],
            )
        cards = [_product_to_card(product) for product in products]
        answer_lines = [f"为您筛选到 {len(cards)} 款："]
        for card in cards:
            answer_lines.append(
                f"- {card['title']}：{card['price']:.0f} {card['currency']}，{card['layout']}，{card['switch_type']}，"
                f"{card['noise_level']}噪声；适用场景：{'、'.join(card['use_cases'])}。"
            )
        if any(term in message for term in ["宿舍", "寝室", "室友"]):
            answer_lines.append("宿舍使用还可以考虑声音；我没有把“宿舍”自行当作静音硬条件。")
        if "quiet" in context.preferences and not any(card["noise_level"] == "低" for card in cards):
            answer_lines.append("您希望声音小，但这些候选都没有标注低噪声；若安静是必须条件，我会按硬条件重新筛选。")
        if context.max_price is None and not context.budget_unlimited:
            answer_lines.append("如果您有预算上限，可以告诉我，我再继续筛选。")
        return ChatResponse(
            intent="PRODUCT_SEARCH",
            answer="\n".join(answer_lines),
            data={"products": cards, "filters": filters.model_dump(), "criteria": criteria,
                  "shopping_context": context.model_dump(), "shopping_extraction": extraction,
                  "shopping_memory": {"updates": memory_events, "scope": "page_session"}},
            sources=["ClickyCanvas 商品快照"],
        )

    def _policy_answer(self, message: str) -> ChatResponse:
        if any(term in message for term in ["价格保护", "保价"]):
            return ChatResponse(
                intent="POLICY_FAQ",
                answer="我没有找到明确的政策说明，建议转人工确认。",
                sources=["ClickyCanvas 政策知识库"],
            )
        if any(term in message for term in ["配不上", "连不上", "无法连接", "配对失败", "蓝牙配对"]):
            retrieval_query = "无法连接"
        elif any(term in message for term in ["按键没反应", "按键没有反应", "按键失灵", "几个键", "没反应"]):
            retrieval_query = "个别按键失灵"
        elif any(term in message for term in ["新加坡", "海外", "国外", "寄到"]):
            retrieval_query = "配送范围"
        elif any(term in message for term in ["拆开", "不喜欢", "无理由退", "签收"]):
            retrieval_query = "无理由退货"
        else:
            retrieval_query = message
        hits = invoke_tool(
            "retrieve_policy", {"query": retrieval_query, "limit": 6},
            lambda: self.database.retrieve_knowledge(retrieval_query, limit=6),
        )
        if not hits:
            return ChatResponse(
                intent="POLICY_FAQ",
                answer="我没有找到明确的政策说明，建议转人工确认。",
                sources=["ClickyCanvas 政策知识库"],
            )
        section_markers: list[str] = []
        if "退货" in message and "流程" in message:
            section_markers = ["退货流程"]
        elif "退货" in message and any(term in message for term in ["条件", "几天", "无理由"]):
            section_markers = ["无理由退货", "不符合条件的情况"]
        elif any(term in message for term in ["什么时候", "交给快递", "发货"]):
            section_markers = ["发货时间"]
        elif "保修" in message and any(term in message for term in ["进水", "进液", "跌落", "拆解", "改装"]):
            section_markers = ["保修范围"]
        elif "保修" in message and any(term in message for term in ["期限", "多久", "几个月"]):
            section_markers = ["保修期限"]

        selected_hits = [
            hit for hit in hits
            if not section_markers or any(hit.excerpt.lstrip().startswith(marker) for marker in section_markers)
        ]
        if not selected_hits:
            selected_hits = hits[:1]
        selected_hits = selected_hits[:2]

        headings = (
            "无理由退货", "不符合条件的情况", "退货流程", "发货时间", "物流延迟", "配送范围",
            "保修期限", "保修范围", "处理方式", "个别按键失灵", "无法连接", "到货损坏",
        )
        excerpts: list[str] = []
        for hit in selected_hits:
            excerpt = hit.excerpt.strip()
            for heading in headings:
                if excerpt.startswith(heading):
                    excerpt = excerpt[len(heading):].strip(" ：:。")
                    break
            if excerpt and excerpt not in excerpts:
                excerpts.append(excerpt)
        primary = selected_hits[0]
        answer = " ".join(excerpts) or primary.content.strip()
        if "退货" in message and "流程" in message and "订单号" not in answer:
            answer += " 请提供订单号，我先帮您核对是否还在退货期限内。"
        elif "保修" in message and "订单号" not in answer:
            answer += " 如需确认您的订单是否在保修期内，请提供订单号；提交检测还需要故障描述。"
        return ChatResponse(
            intent="POLICY_FAQ",
            answer=answer,
            data={
                "documents": [hit.model_dump(include={"document_id", "title", "section", "content", "source_path", "updated_at"}) for hit in hits[:1]],
                "retrieval": {
                    "method": hits[0].retrieval_method,
                    "query": message,
                    "hits": [hit.model_dump(exclude={"content"}) for hit in hits],
                },
            },
            sources=[hit.title for hit in hits[:1]],
        )

    def _order_status(self, message: str, request: ChatRequest, order_id: str | None) -> ChatResponse:
        if not order_id:
            return ChatResponse(
                intent="ORDER_STATUS",
                answer="请提供演示订单号，例如 CC-1002。查询订单还需要 4 位演示验证码。",
                sources=["ClickyCanvas 模拟订单库"],
            )
        access_code = _find_access_code(message, request.demo_access_code)
        if not access_code:
            return ChatResponse(
                intent="ORDER_STATUS",
                answer=f"已识别订单 {order_id}。为了查询演示订单，请再提供 4 位演示验证码。",
                data={"order_id": order_id},
                sources=["ClickyCanvas 模拟订单库"],
            )
        order, status = invoke_tool(
            "lookup_order", {"order_id": order_id, "demo_access_code": access_code},
            lambda: self.database.lookup_order(order_id, access_code),
        )
        if status == "not_found":
            return ChatResponse(
                intent="ORDER_STATUS",
                answer=f"没有找到订单 {order_id}。请检查订单号是否以 CC- 开头并包含 4 位数字。",
                sources=["ClickyCanvas 模拟订单库"],
            )
        if status == "invalid_code":
            return ChatResponse(
                intent="ORDER_STATUS",
                answer="订单号或演示验证码不匹配，我不能继续显示订单信息。",
                sources=["ClickyCanvas 模拟订单库"],
            )
        assert order is not None
        status_label = FULFILLMENT_LABELS.get(order.fulfillment_status, order.fulfillment_status)
        last_event = order.tracking_events[-1] if order.tracking_events else None
        answer = f"订单 {order.order_id} 当前状态：{status_label}。"
        if order.tracking_number:
            answer += f" 演示物流单号：{order.tracking_number}。"
        if last_event:
            answer += f" 最近节点是 {last_event.occurred_at} 在 {last_event.location or '未知地点'} 显示“{last_event.status}”。"
        if order.issue_type == "damaged_on_arrival":
            answer += " 如果商品到货损坏，请保留外包装并提交照片，客服会创建售后案件。"
        return ChatResponse(
            intent="ORDER_STATUS",
            answer=answer,
            data={"order": order.model_dump()},
            sources=["ClickyCanvas 模拟订单库"],
        )

    def _escalate(self, message: str, request: ChatRequest) -> ChatResponse:
        order_id = _find_order_id(message, request.order_id)
        reason = "售后问题需要人工判断"
        if "退款" in message or "赔偿" in message:
            reason = "退款或赔偿争议"
        elif any(term in message for term in ["起火", "冒烟", "触电", "人身"]):
            reason = "疑似安全风险"
        elif "损坏" in message or "进水" in message:
            reason = "商品损坏或疑似人为损坏"
        case = invoke_tool(
            "create_after_sale_case", {"order_id": order_id, "reason": reason},
            lambda: self.database.create_escalation_case(order_id, reason, message[:500]),
        )
        return ChatResponse(
            intent="ESCALATE",
            answer=f"这个问题需要人工审核，我已经创建售后案件 {case.case_id}。请保留订单、商品和包装信息，人工客服会继续确认处理方式。",
            data={"case_id": case.case_id},
            sources=["ClickyCanvas 售后流程"],
            escalation_case=case,
        )
