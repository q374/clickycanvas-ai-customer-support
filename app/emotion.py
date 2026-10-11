from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EmotionState = Literal["neutral", "confused", "anxious", "frustrated", "angry", "high_risk"]
UrgencyLevel = Literal["low", "medium", "high", "critical"]


class EmotionResult(BaseModel):
    state: EmotionState = "neutral"
    urgency: UrgencyLevel = "low"
    confidence: float = Field(default=0.72, ge=0, le=1)
    needs_human: bool = False
    method: Literal["rules", "deepseek", "fallback"] = "rules"
    signals: list[str] = Field(default_factory=list, max_length=8)


class EmotionStrategy(BaseModel):
    """可审计的客服动作策略，不把内部标签直接写进顾客回答。"""

    state: EmotionState
    action: Literal["answer", "explain_then_ask", "answer_then_next_step", "act_first", "handoff"]
    acknowledge_once: bool = False
    max_questions: int = Field(default=2, ge=0, le=2)
    no_compensation_promise: bool = False


_STRATEGIES: dict[EmotionState, EmotionStrategy] = {
    "neutral": EmotionStrategy(state="neutral", action="answer", max_questions=2),
    "confused": EmotionStrategy(
        state="confused", action="explain_then_ask", acknowledge_once=True, max_questions=1,
    ),
    "anxious": EmotionStrategy(
        state="anxious", action="answer_then_next_step", acknowledge_once=True, max_questions=1,
    ),
    "frustrated": EmotionStrategy(
        state="frustrated", action="act_first", acknowledge_once=True, max_questions=1,
    ),
    "angry": EmotionStrategy(
        state="angry", action="act_first", acknowledge_once=True, max_questions=1,
    ),
    "high_risk": EmotionStrategy(
        state="high_risk", action="handoff", acknowledge_once=True, max_questions=1,
        no_compensation_promise=True,
    ),
}


def strategy_for_emotion(result: EmotionResult) -> EmotionStrategy:
    """Map an observable wording signal to a bounded customer-service action."""
    return _STRATEGIES[result.state]


def style_deterministic_answer(answer: str, result: EmotionResult, intent: str, message: str = "") -> str:
    """Add one short, state-specific handoff to deterministic answers.

    Facts and tool results stay untouched. The function is intentionally idempotent
    so a later model rewrite cannot cause repeated reassurance when the safe
    deterministic answer is used again.
    """
    text = answer.strip()
    if not text or result.state == "neutral":
        return text

    # Apologize when the wording indicates a concrete service failure or a
    # safety incident. A bare insult still gets a short action-first reply,
    # without an automatic apology for an unverified cause.
    apology = _apology_prefix(result.state, message)
    if result.state == "high_risk":
        if any(term in message for term in ("起火", "冒烟", "触电", "漏电", "人身")):
            return f"很抱歉遇到安全问题，请立即停止使用并断开电源。{text}"
        return f"很抱歉给您带来麻烦。{text}"

    if text.startswith(("很抱歉", "抱歉")):
        return text

    # A standalone abusive/general message needs a useful next step rather than
    # the generic capability list, which otherwise sounds dismissive.
    if result.state == "angry" and intent == "GENERAL":
        return f"{apology or '我先帮您处理。'}请告诉我订单号和遇到的具体问题，我会继续查。"
    if result.state == "frustrated" and intent == "GENERAL":
        return f"{apology or '我先帮您处理。'}请告诉我订单号或具体问题，我马上帮您确认。"

    # For a precise order or policy question, retain the verified facts. For
    # clarifications, ask only for the one field the next tool step needs.
    if intent == "PRODUCT_CLARIFY" and result.state == "confused":
        if "商品 ID" in text or "哪款商品" in text:
            return "我先确认您说的是哪两款键盘。请发两款商品名称或商品 ID。"
        if "预算" in text and "场景" in text:
            return "您主要是办公、游戏，还是日常使用？"
        return text

    leading_markers = ("我先帮您", "我先直接", "我们直接", "先给您", "我把重点", "抱歉", "理解")
    if text.startswith(leading_markers):
        return text
    prefixes = {
        "confused": "我把重点说清楚：",
        "anxious": "先给您明确答复：",
        "frustrated": "我先帮您继续处理：",
        "angry": "我先直接处理：",
    }
    return f"{apology or prefixes[result.state]}{text}"


def _apology_prefix(state: EmotionState, message: str) -> str:
    """Choose apology only when the message contains a concrete failure cue."""
    failure_terms = ("一直", "反复", "还是", "又", "无法", "失败", "不行", "没反应", "太慢", "催", "不解决", "烦")
    if state == "frustrated" and any(term in message for term in failure_terms):
        return "抱歉让您反复尝试仍没解决，我先帮您继续排查："
    if state == "anxious" and any(term in message for term in ("等", "延迟", "来不及", "赶不上", "什么时候到")):
        return "抱歉让您久等了，先给您明确答复："
    if state == "angry" and ("CC-" in message.upper() or any(term in message for term in ("订单", "退款", "失败", "没反应", "连不上", "不解决", "等"))):
        return "抱歉让您遇到这个问题，我先直接处理："
    return ""


def strategy_instruction(result: EmotionResult) -> str:
    """Return the short instruction supplied to an optional answer rewriter."""
    instructions = {
        "neutral": "正常回答，必要时再追问关键信息。",
        "confused": "先用一句话解释重点，只追问一个最关键的问题。",
        "anxious": "先给明确结论或当前进度，再给一个可执行的下一步，只追问一个问题。",
        "frustrated": "只承接一次，用一句短话即可；不要反复说理解或让用户消消气，优先推进处理。",
        "angry": "只承接一次，用一句短话即可；不反驳、不重复安慰，优先推进订单或售后处理。",
        "high_risk": "保持人工转交流程，不承诺退款、赔偿或其他结果，只说明案件和下一步。",
    }
    return instructions[result.state]


_HIGH_RISK_SIGNALS = (
    ("退款争议", "退款争议"), ("退款金额", "退款争议"), ("赔偿", "赔偿"),
    ("投诉", "投诉"), ("法律", "法律风险"), ("严重损坏", "严重损坏"),
    ("到货损坏", "到货损坏"), ("我要退款", "退款请求"), ("人工处理", "人工请求"),
    ("找人工", "人工请求"), ("转人工", "人工请求"), ("起火", "安全风险"),
    ("冒烟", "安全风险"), ("触电", "安全风险"), ("漏电", "安全风险"), ("人身", "安全风险"),
    ("退款一直", "退款争议"), ("退款没", "退款争议"), ("退款未", "退款争议"),
)
_ANGRY_SIGNALS = (
    "垃圾", "骗子", "sb", "傻", "脑残", "滚", "坑爹", "去死", "妈的", "他妈", "我日",
)
_FRUSTRATED_SIGNALS = (
    ("一直", "反复未解决"), ("反复", "反复未解决"), ("还是", "问题仍未解决"),
    ("又", "重复失败"), ("无法", "操作失败"), ("失败", "操作失败"), ("不行", "操作失败"),
    ("没反应", "操作失败"), ("没有反应", "操作失败"), ("太慢", "等待过久"),
    ("催", "催促处理"), ("不解决", "问题未解决"), ("失望", "负面体验"), ("烦", "负面体验"),
)
_ANXIOUS_SIGNALS = (
    ("着急", "着急"), ("急", "着急"), ("担心", "担心"), ("害怕", "担心"),
    ("怕", "担心"), ("来不及", "时间压力"), ("赶不上", "时间压力"),
    ("尽快", "希望尽快处理"), ("什么时候到", "等待确认"), ("能不能快", "希望尽快处理"),
)
_CONFUSED_SIGNALS = (
    ("不懂", "不理解"), ("不明白", "不理解"), ("什么意思", "需要解释"),
    ("看不懂", "需要解释"), ("怎么选", "选择困难"), ("哪个好", "选择困难"),
    ("区别", "需要比较"), ("对比", "需要比较"), ("能解释", "需要解释"),
    ("不确定", "需要确认"), ("麻烦说明", "需要解释"),
)


def _find_signals(message: str, rules: tuple[tuple[str, str], ...]) -> list[str]:
    return list(dict.fromkeys(label for term, label in rules if term in message))


def classify_emotion(message: str) -> EmotionResult:
    """Classify only observable wording; never claim to know a user's inner state."""
    text = message.strip().lower()
    if not text:
        return EmotionResult(method="fallback", confidence=0.5)

    high_risk = _find_signals(text, _HIGH_RISK_SIGNALS)
    if high_risk:
        return EmotionResult(
            state="high_risk", urgency="critical", confidence=0.96,
            needs_human=True, signals=high_risk,
        )

    angry = [term for term in _ANGRY_SIGNALS if term in text]
    if angry:
        return EmotionResult(
            state="angry", urgency="high", confidence=0.93,
            needs_human=False, signals=list(dict.fromkeys(angry)),
        )

    frustrated = _find_signals(text.replace("麻烦", ""), _FRUSTRATED_SIGNALS)
    if frustrated:
        return EmotionResult(
            state="frustrated", urgency="high", confidence=0.87,
            needs_human=False, signals=frustrated,
        )

    anxious = _find_signals(text, _ANXIOUS_SIGNALS)
    if anxious:
        return EmotionResult(
            state="anxious", urgency="high", confidence=0.85,
            needs_human=False, signals=anxious,
        )

    confused = _find_signals(text, _CONFUSED_SIGNALS)
    if confused:
        return EmotionResult(
            state="confused", urgency="medium", confidence=0.84,
            needs_human=False, signals=confused,
        )

    # Keep neutral confidence below 1: rules can only observe wording, not intent.
    return EmotionResult()


def merge_deepseek_emotion(rule_result: EmotionResult, *, state: EmotionState | None,
                           urgency: UrgencyLevel | None, confidence: float | None,
                           needs_human: bool | None) -> EmotionResult:
    """Accept model classification without weakening a locally detected safety boundary."""
    if state is None or urgency is None or confidence is None or needs_human is None:
        return rule_result
    if rule_result.state == "high_risk":
        return rule_result.model_copy(update={"method": "deepseek"})
    return EmotionResult(
        state=state,
        urgency=urgency,
        confidence=confidence,
        needs_human=bool(needs_human) or rule_result.needs_human,
        method="deepseek",
        signals=rule_result.signals,
    )
