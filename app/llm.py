from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import load_local_env
from .emotion import EmotionResult, EmotionState, UrgencyLevel, strategy_instruction
from .models import ChatResponse


IntentName = Literal[
    "PRODUCT_SEARCH",
    "PRODUCT_DETAIL",
    "PRODUCT_COMPARE",
    "POLICY_FAQ",
    "ORDER_STATUS",
    "AFTER_SALES",
    "ESCALATE",
    "UNKNOWN",
]


class LLMDraft(BaseModel):
    intent: IntentName
    answer: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(default=0.5, ge=0, le=1)
    emotion_state: EmotionState | None = None
    emotion_urgency: UrgencyLevel | None = None
    emotion_confidence: float | None = Field(default=None, ge=0, le=1)
    emotion_needs_human: bool | None = None


class DeepSeekOrchestrator:
    """Optional DeepSeek answer rewriter over deterministic evidence.

    The deterministic agent still owns routing, data lookup and escalation.
    DeepSeek may only improve the wording and report a comparable intent label.
    """

    def __init__(self, api_key: str, model: str = "deepseek-flash", base_url: str = "https://api.deepseek.com"):
        self.model = model
        self.base_url = base_url
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=1)

    @classmethod
    def from_env(cls) -> "DeepSeekOrchestrator | None":
        load_local_env()
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip() or "deepseek-flash",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip() or "https://api.deepseek.com",
        )

    def rewrite(self, user_message: str, deterministic: ChatResponse) -> LLMDraft:
        emotion = deterministic.data.get("emotion", {})
        # The local classifier is the safety source of truth. The prompt only
        # carries its bounded response behavior to the wording model.
        strategy_text = "正常回答，必要时再追问关键信息。"
        if isinstance(emotion, dict) and emotion.get("state") in {
            "neutral", "confused", "anxious", "frustrated", "angry", "high_risk",
        }:
            emotion_value = EmotionResult.model_validate(emotion)
            strategy_text = strategy_instruction(emotion_value)
        evidence = {
            "deterministic_intent": deterministic.intent,
            "deterministic_answer": deterministic.answer,
            "data": deterministic.data,
            "sources": deterministic.sources,
            "response_strategy": strategy_text,
        }
        instructions = (
            "你是 ClickyCanvas 电商客服的回答润色器。只能使用 EVIDENCE 中的事实。"
            "不要新增商品规格、价格、订单状态、政策承诺、退款结论或物流信息。"
            "如果 deterministic_intent 是 ESCALATE，必须保留人工案件编号和转人工结论。"
            "输出严格 JSON，字段为 intent、answer、confidence、emotion_state、emotion_urgency、emotion_confidence、emotion_needs_human；intent 必须与 deterministic_intent 相同。"
            "emotion_state 只能是 neutral/confused/anxious/frustrated/angry/high_risk；emotion_urgency 只能是 low/medium/high/critical。"
            f"当前情绪策略是：{strategy_text}"
            "JSON 示例：{\"intent\":\"PRODUCT_SEARCH\",\"answer\":\"...\",\"confidence\":0.8,\"emotion_state\":\"neutral\",\"emotion_urgency\":\"low\",\"emotion_confidence\":0.8,\"emotion_needs_human\":false}。"
            "answer 使用简洁、自然的中文，只回答顾客需要的信息。保留必要的商品 ID、订单号和安全边界；不要在自然回答中重复内部来源、模型名称或追踪信息，来源由系统单独展示。"
        )
        prompt = json.dumps({"user_message": user_message, "evidence": evidence}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]
        last_error: ValueError | None = None
        for _attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=900,
                response_format={"type": "json_object"},
                # JSON rewriting is short and bounded; disabling reasoning keeps
                # the completion budget for the actual JSON answer.
                extra_body={"thinking": {"type": "disabled"}},
                stream=False,
            )
            message = response.choices[0].message
            content = message.content or ""
            try:
                payload = self._parse_json(content)
                draft = LLMDraft.model_validate(payload)
                if draft.intent != deterministic.intent:
                    raise ValueError("DeepSeek returned an intent different from the deterministic route")
                return draft
            except ValueError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def extract_shopping(self, user_message: str) -> dict[str, Any]:
        """Suggest bounded shopping slots; the agent validates every suggestion."""
        instructions = (
            "你只提取顾客本条消息中的商品选购条件，不回答问题。"
            "输出 JSON 对象，格式为 suggestions 数组；每项含 field、value、evidence。"
            "evidence 必须逐字摘自顾客消息，无法确定时返回空数组。"
            "可用字段和值：use_case=办公/游戏/数据录入；"
            "layout=75%/65%/60%/TKL/Numpad；"
            "connectivity=Bluetooth/USB-C/2.4GHz/wireless；"
            "compatibility=macOS/Windows/iPadOS/Android；"
            "noise_level=低；required_tag=薄型/入门/低延迟；"
            "preference=quiet/wireless。不要推测价格或商品事实。"
            "例如顾客说‘开会时用’，输出"
            "{\"suggestions\":[{\"field\":\"use_case\",\"value\":\"办公\",\"evidence\":\"开会时用\"}]}。"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": user_message}],
            temperature=0,
            max_tokens=700,
            response_format={"type": "json_object"},
            # The extractor is validated locally; it needs a complete JSON
            # object rather than a long reasoning trace.
            extra_body={"thinking": {"type": "disabled"}},
            stream=False,
        )
        return self._parse_json(response.choices[0].message.content or "")

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("DeepSeek response content must be text")
        text = content.strip()
        if not text:
            raise ValueError("DeepSeek response content was empty")

        # Some compatible gateways wrap the answer in a reasoning block or a
        # Markdown fence even when JSON mode is requested. Remove those wrappers
        # before parsing while keeping the actual JSON payload unchanged.
        text = re.sub(r"<think(?:ing)?[^>]*>.*?</think(?:ing)?>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json\n"):
                text = text[5:].lstrip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("DeepSeek response did not contain JSON")
            value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("DeepSeek response JSON must be an object")
        return value
