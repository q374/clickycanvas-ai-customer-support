from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from app.agent import SupportAgent
from app.db import Database
from app.llm import LLMDraft
from app.main import CATALOG_PATH, KNOWLEDGE_PATH, ORDERS_PATH
from app.models import ChatRequest
from app.tools import BoundedToolExecutor, ToolBudgetExceeded


ROOT = Path(__file__).resolve().parents[1]


class BoundedAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = ROOT / "data" / "runtime" / f"tools-test-{uuid.uuid4().hex}.db"
        self.database = Database(self.db_path)
        self.database.seed_catalog(CATALOG_PATH)
        self.database.seed_orders(ORDERS_PATH)
        self.database.seed_knowledge(KNOWLEDGE_PATH)
        self.agent = SupportAgent(self.database)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_agent_records_verified_tool_trace(self) -> None:
        response = self.agent.chat(ChatRequest(message="多久发货？"))
        trace = response.data["agent"]
        self.assertEqual(trace["mode"], "bounded_tool_agent")
        self.assertEqual(trace["max_tool_calls"], 3)
        self.assertEqual(trace["tool_calls"][0]["name"], "retrieve_policy")
        self.assertTrue(trace["verified"])

    def test_agent_records_emotion_without_changing_business_route(self) -> None:
        response = self.agent.chat(ChatRequest(message="怎么又一直连不上？"))
        self.assertEqual(response.intent, "POLICY_FAQ")
        self.assertEqual(response.data["emotion"]["state"], "frustrated")
        self.assertEqual(response.data["emotion"]["method"], "rules")

    def test_deepseek_can_return_structured_emotion(self) -> None:
        class StructuredLLM:
            model = "fake"

            def extract_shopping(self, _message):
                return {"suggestions": []}

            def rewrite(self, _message, deterministic):
                return LLMDraft(
                    intent=deterministic.intent,
                    answer=deterministic.answer,
                    confidence=0.8,
                    emotion_state="anxious",
                    emotion_urgency="high",
                    emotion_confidence=0.91,
                    emotion_needs_human=False,
                )

        response = SupportAgent(self.database, llm=StructuredLLM()).chat(
            ChatRequest(message="办公用的键盘")
        )
        self.assertEqual(response.data["emotion"]["state"], "anxious")
        self.assertEqual(response.data["emotion"]["method"], "deepseek")

    def test_angry_general_message_gets_one_action_first_prompt(self) -> None:
        response = self.agent.chat(ChatRequest(message="你们就是骗子，太垃圾了"))
        self.assertEqual(response.intent, "GENERAL")
        self.assertEqual(response.answer, "我先帮您处理。请告诉我订单号和遇到的具体问题，我会继续查。")
        self.assertEqual(response.data["emotion_strategy"]["action"], "act_first")

    def test_safety_language_enters_human_flow_even_without_legacy_term(self) -> None:
        response = self.agent.chat(ChatRequest(message="商品刚刚冒烟了"))
        self.assertEqual(response.intent, "ESCALATE")
        self.assertEqual(response.data["emotion"]["state"], "high_risk")
        self.assertTrue(response.data["emotion_strategy"]["no_compensation_promise"])
        self.assertIn("人工客服会继续确认处理方式", response.answer)

    def test_order_tool_trace_redacts_demo_code(self) -> None:
        response = self.agent.chat(ChatRequest(message="订单 CC-1002 到哪了？验证码 1937"))
        trace = response.data["agent"]
        self.assertEqual(trace["tool_calls"][0]["name"], "lookup_order")
        self.assertEqual(trace["tool_calls"][0]["arguments"]["demo_access_code"], "[redacted]")

    def test_tool_budget_is_enforced(self) -> None:
        executor = BoundedToolExecutor(max_calls=2)
        executor.invoke("one", {}, lambda: 1)
        executor.invoke("two", {}, lambda: 2)
        with self.assertRaises(ToolBudgetExceeded):
            executor.invoke("three", {}, lambda: 3)
        self.assertEqual(executor.trace()["tool_calls"][-1]["status"], "blocked_budget")


if __name__ == "__main__":
    unittest.main()
