from __future__ import annotations

import unittest

from app.emotion import (
    EmotionResult,
    classify_emotion,
    merge_deepseek_emotion,
    strategy_for_emotion,
    style_deterministic_answer,
)


class EmotionClassificationTests(unittest.TestCase):
    def test_classifies_six_observable_states(self) -> None:
        cases = {
            "你好，想看看办公键盘": "neutral",
            "我不明白这两款有什么区别": "confused",
            "我有点着急，明天要用，能尽快发吗": "anxious",
            "怎么又失败了，一直连不上": "frustrated",
            "你们就是骗子，太垃圾了": "angry",
            "我要退款争议，不解决我就投诉": "high_risk",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                result = classify_emotion(message)
                self.assertEqual(result.state, expected)
                self.assertTrue(result.signals or expected == "neutral")

    def test_high_risk_requires_human_and_cannot_be_downgraded(self) -> None:
        result = classify_emotion("商品冒烟了，请马上人工处理")
        self.assertEqual(result.state, "high_risk")
        self.assertEqual(result.urgency, "critical")
        self.assertTrue(result.needs_human)

        merged = merge_deepseek_emotion(
            result, state="neutral", urgency="low", confidence=0.99, needs_human=False,
        )
        self.assertEqual(merged.state, "high_risk")
        self.assertTrue(merged.needs_human)
        self.assertEqual(merged.method, "deepseek")

    def test_empty_message_falls_back_without_claiming_emotion(self) -> None:
        result = classify_emotion("")
        self.assertEqual(result, EmotionResult(method="fallback", confidence=0.5))

    def test_strategy_limits_questions_and_keeps_high_risk_handoff(self) -> None:
        angry = strategy_for_emotion(classify_emotion("你们就是骗子，太垃圾了"))
        self.assertEqual(angry.action, "act_first")
        self.assertTrue(angry.acknowledge_once)
        self.assertEqual(angry.max_questions, 1)

        high_risk = strategy_for_emotion(classify_emotion("商品冒烟了"))
        self.assertEqual(high_risk.action, "handoff")
        self.assertTrue(high_risk.no_compensation_promise)

    def test_style_is_direct_and_idempotent(self) -> None:
        result = classify_emotion("你们就是骗子")
        answer = style_deterministic_answer("请提供订单号。", result, "ORDER_STATUS")
        self.assertEqual(answer, "我先直接处理：请提供订单号。")
        self.assertEqual(style_deterministic_answer(answer, result, "ORDER_STATUS"), answer)

    def test_apology_depends_on_a_concrete_failure(self) -> None:
        frustrated = classify_emotion("怎么又一直连不上")
        answer = style_deterministic_answer("我来继续排查。", frustrated, "POLICY_FAQ", "怎么又一直连不上")
        self.assertTrue(answer.startswith("抱歉"))
        self.assertNotIn("我们直接继续处理", answer)

        angry = classify_emotion("你们就是骗子")
        self.assertFalse(style_deterministic_answer("请提供订单号。", angry, "ORDER_STATUS", "你们就是骗子").startswith("抱歉"))

        high_risk = classify_emotion("商品冒烟了")
        safety = style_deterministic_answer("已创建售后案件。", high_risk, "ESCALATE", "商品冒烟了")
        self.assertIn("停止使用并断开电源", safety)


if __name__ == "__main__":
    unittest.main()
