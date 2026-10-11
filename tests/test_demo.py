from __future__ import annotations

import unittest
import uuid
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent import SupportAgent
from app.db import Database
from app.llm import DeepSeekOrchestrator, LLMDraft
from app.main import CATALOG_PATH, KNOWLEDGE_PATH, ORDERS_PATH, create_app
from app.models import ChatRequest, ProductSearchFilters
from scripts.sync_catalog import normalize


ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    model = "fake"

    def extract_shopping(self, user_message):
        return {"suggestions": []}

    def rewrite(self, user_message, deterministic):
        return LLMDraft(intent=deterministic.intent, answer=deterministic.answer + "（已润色）", confidence=0.9)


class DemoDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = ROOT / "data" / "runtime" / f"test-{uuid.uuid4().hex}.db"
        self.database = Database(self.db_path)
        self.database.seed_catalog(CATALOG_PATH)
        self.database.seed_orders(ORDERS_PATH)
        self.database.seed_knowledge(KNOWLEDGE_PATH)
        self.agent = SupportAgent(self.database)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_seeded_models_have_expected_counts(self) -> None:
        self.assertEqual(self.database.product_count(), 6)
        self.assertEqual(self.database.order_count(), 7)
        self.assertEqual(len(self.database.search_knowledge("多久发货")), 1)

    def test_product_search_uses_structured_fixture(self) -> None:
        products = self.database.list_products(ProductSearchFilters(query="办公室 静音", max_price=800))
        self.assertTrue(products)
        self.assertEqual(products[0].id, "cc-75-silent")

    def test_explicit_thin_requirement_excludes_other_mac_compatible_products(self) -> None:
        response = self.agent.chat(ChatRequest(message="苹果电脑用，预算 800 元以内的薄型键盘"))
        self.assertEqual([item["id"] for item in response.data["products"]], ["cc-low-profile"])

    def test_compare_answers_direct_quieter_question(self) -> None:
        response = self.agent.chat(ChatRequest(message="cc-low-profile 和 cc-75-silent 哪个更安静"))
        self.assertIn("两款均为低噪声", response.answer)
        self.assertIn("无法据此判断", response.answer)

    def test_policy_answer_uses_most_relevant_document(self) -> None:
        response = self.agent.chat(ChatRequest(message="键盘进水后还能走保修吗"))
        self.assertEqual([doc["document_id"] for doc in response.data["documents"]], ["warranty"])
        self.assertIn("进液", response.answer)

    def test_product_compare_preserves_requested_order(self) -> None:
        products = self.database.compare_products(["cc-65-wireless", "cc-75-silent"])
        self.assertEqual([product.id for product in products], ["cc-65-wireless", "cc-75-silent"])

    def test_order_lookup_requires_demo_code(self) -> None:
        order, status = self.database.lookup_order("cc-1002", "1937")
        self.assertEqual(status, "ok")
        self.assertIsNotNone(order)
        self.assertEqual(order.fulfillment_status, "shipped")

        order, status = self.database.lookup_order("CC-1002", "0000")
        self.assertIsNone(order)
        self.assertEqual(status, "invalid_code")

    def test_order_verification_reply_keeps_pending_order_context(self) -> None:
        prompt = self.agent.chat(ChatRequest(message="订单 CC-1002 到哪了？"))
        self.assertEqual(prompt.intent, "ORDER_STATUS")
        self.assertEqual(prompt.data["order_id"], "CC-1002")

        reply = self.agent.chat(ChatRequest(message="验证码 1937", order_id=prompt.data["order_id"]))
        self.assertEqual(reply.intent, "ORDER_STATUS")
        self.assertEqual(reply.data["order"]["order_id"], "CC-1002")
        self.assertIn("已发货", reply.answer)

    def test_general_messages_do_not_fall_back_to_product_search(self) -> None:
        agent = SupportAgent(self.database, llm=FakeLLM())
        for message in ["你好", "今天天气怎么样？"]:
            with self.subTest(message=message):
                response = agent.chat(ChatRequest(message=message))
                self.assertEqual(response.intent, "GENERAL")
                self.assertNotIn("products", response.data)
                self.assertIn("配送/退货/保修政策", response.answer)
                self.assertEqual(response.data["llm"]["status"], "skipped_general")

        product_question = agent.chat(ChatRequest(message="你好，我想找一把适合办公室的键盘"))
        self.assertEqual(product_question.intent, "PRODUCT_SEARCH")
        self.assertTrue(product_question.data["products"])

    def test_vague_keyboard_recommendation_asks_for_missing_slots(self) -> None:
        agent = SupportAgent(self.database, llm=FakeLLM())
        response = agent.chat(ChatRequest(message="推荐一下"))
        self.assertEqual(response.intent, "PRODUCT_CLARIFY")
        self.assertEqual(response.data["missing_slots"], ["主要使用场景或在意的键盘特点"])
        self.assertIn("蓝牙", response.answer)
        self.assertEqual(response.data["llm"]["status"], "skipped_clarification")

        office = self.agent.chat(ChatRequest(message="推荐一把办公室用的键盘"))
        self.assertEqual(office.intent, "PRODUCT_SEARCH")
        self.assertTrue(office.data["products"])

        layout_only = self.agent.chat(ChatRequest(message="推荐一把 75% 配列键盘"))
        self.assertEqual(layout_only.intent, "PRODUCT_SEARCH")
        self.assertTrue(layout_only.data["products"])

        followup = self.agent.chat(ChatRequest(message="办公，预算800以内"))
        self.assertEqual(followup.intent, "PRODUCT_SEARCH")
        self.assertEqual(followup.data["filters"]["max_price"], 800.0)
        self.assertEqual(followup.data["filters"]["use_case"], "办公")
        self.assertTrue(followup.data["products"])
        self.assertTrue(all(product["price"] <= 800 for product in followup.data["products"]))
        self.assertTrue(all(any("办公" in case for case in product["use_cases"]) for product in followup.data["products"]))

        no_match = self.agent.chat(ChatRequest(message="办公，预算500以内"))
        self.assertEqual(no_match.data["products"], [])

    def test_escalation_cases_can_be_listed_and_loaded(self) -> None:
        response = self.agent.chat(ChatRequest(message="订单 CC-1006 到货损坏，我要退款"))
        case_id = response.data["case_id"]
        cases = self.database.list_escalation_cases()
        self.assertEqual(cases[0].case_id, case_id)
        loaded = self.database.get_escalation_case(case_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.order_id, "CC-1006")

    def test_agent_routes_product_policy_order_and_escalation(self) -> None:
        product_response = self.agent.chat(ChatRequest(message="有没有适合办公室、声音比较小的键盘？"))
        self.assertEqual(product_response.intent, "PRODUCT_SEARCH")
        self.assertTrue(product_response.data["products"])

        policy_response = self.agent.chat(ChatRequest(message="多久发货？"))
        self.assertEqual(policy_response.intent, "POLICY_FAQ")
        self.assertIn("工作日", policy_response.answer)

        synonym_policy_response = self.agent.chat(ChatRequest(message="蓝牙连不上怎么办"))
        self.assertEqual(synonym_policy_response.intent, "POLICY_FAQ")
        self.assertTrue(synonym_policy_response.data["documents"])

        shipping_policy_response = self.agent.chat(ChatRequest(message="订单什么时候交给快递"))
        self.assertEqual(shipping_policy_response.intent, "POLICY_FAQ")
        self.assertIn("1 个工作日", shipping_policy_response.answer)

        refund_policy_response = self.agent.chat(ChatRequest(message="退款政策是什么？"))
        self.assertEqual(refund_policy_response.intent, "POLICY_FAQ")

        order_response = self.agent.chat(ChatRequest(message="订单 CC-1002 到哪了？验证码 1937"))
        self.assertEqual(order_response.intent, "ORDER_STATUS")
        self.assertIn("已发货", order_response.answer)

        compare_response = self.agent.chat(ChatRequest(message="比较 cc-75-silent 和 cc-65-wireless"))
        self.assertEqual(compare_response.intent, "PRODUCT_COMPARE")
        self.assertEqual([product["id"] for product in compare_response.data["products"]], ["cc-75-silent", "cc-65-wireless"])

        escalation_response = self.agent.chat(ChatRequest(message="订单 CC-1006 到货损坏，我要退款"))
        self.assertEqual(escalation_response.intent, "ESCALATE")
        self.assertIsNotNone(escalation_response.escalation_case)

    def test_optional_llm_rewrites_safe_answers_but_preserves_escalation(self) -> None:
        agent = SupportAgent(self.database, llm=FakeLLM())
        product_response = agent.chat(ChatRequest(message="有没有适合办公室的键盘？"))
        self.assertIn("（已润色）", product_response.answer)
        self.assertEqual(product_response.data["llm"]["status"], "rewritten")

        escalation_response = agent.chat(ChatRequest(message="我要退款争议人工处理"))
        self.assertIn("售后案件", escalation_response.answer)
        self.assertEqual(escalation_response.data["llm"]["status"], "skipped_high_risk")

    def test_llm_rewrite_falls_back_when_required_compare_prompt_is_dropped(self) -> None:
        class IncompleteLLM:
            model = "fake"

            def rewrite(self, user_message, deterministic):
                return LLMDraft(intent=deterministic.intent, answer="这是对比结果。", confidence=0.8)

        agent = SupportAgent(self.database, llm=IncompleteLLM())
        response = agent.chat(ChatRequest(message="请对比 cc-75-silent"))
        self.assertIn("至少两个", response.answer)
        self.assertEqual(response.data["llm"]["status"], "fallback")

    def test_llm_rewrite_preserves_no_evidence_boundaries(self) -> None:
        class OverconfidentLLM:
            model = "fake"

            def extract_shopping(self, user_message):
                return {"suggestions": []}

            def rewrite(self, user_message, deterministic):
                return LLMDraft(intent=deterministic.intent, answer="可以，已确认。", confidence=0.9)

        agent = SupportAgent(self.database, llm=OverconfidentLLM())
        no_product = agent.chat(ChatRequest(message="预算 250 元以内"))
        self.assertIn("没有找到", no_product.answer)
        self.assertEqual(no_product.data["llm"]["status"], "fallback")

        unknown_policy = agent.chat(ChatRequest(message="支持保价吗？"))
        self.assertIn("没有找到明确的政策说明", unknown_policy.answer)
        self.assertEqual(unknown_policy.data["llm"]["status"], "fallback")

    def test_deepseek_parser_accepts_reasoning_and_markdown_wrappers(self) -> None:
        payload = DeepSeekOrchestrator._parse_json(
            '<think>先检查事实来源</think>\n```json\n'
            '{"intent":"PRODUCT_SEARCH","answer":"可用","confidence":0.8}\n```'
        )
        self.assertEqual(payload["intent"], "PRODUCT_SEARCH")
        self.assertEqual(payload["answer"], "可用")

    def test_deepseek_retries_an_empty_json_response_once(self) -> None:
        class CompletionStub:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **_kwargs):
                self.calls += 1
                content = "" if self.calls == 1 else '{"intent":"PRODUCT_SEARCH","answer":"可用","confidence":0.8}'
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        completion = CompletionStub()
        orchestrator = object.__new__(DeepSeekOrchestrator)
        orchestrator.model = "fake"
        orchestrator.client = SimpleNamespace(chat=SimpleNamespace(completions=completion))
        deterministic = self.agent._deterministic_chat(ChatRequest(message="有没有适合办公室的键盘？"))
        draft = orchestrator.rewrite("有没有适合办公室的键盘？", deterministic)
        self.assertEqual(completion.calls, 2)
        self.assertEqual(draft.answer, "可用")


class ApiTests(unittest.TestCase):
    def test_minimal_http_loop(self) -> None:
        db_path = ROOT / "data" / "runtime" / f"api-test-{uuid.uuid4().hex}.db"
        try:
            with patch.dict("os.environ", {"LLM_MODE": "deterministic"}):
                client = TestClient(create_app(db_path))
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["products"], 6)

            chat = client.post("/api/chat", json={"message": "订单 CC-1002 到哪了？验证码 1937", "conversation_id": "test-session"})
            self.assertEqual(chat.status_code, 200)
            self.assertEqual(chat.json()["intent"], "ORDER_STATUS")
            self.assertTrue(chat.json()["data"]["trace"]["trace_id"])
            traces = client.get("/api/traces", params={"conversation_id": "test-session"})
            self.assertEqual(traces.status_code, 200)
            self.assertEqual(traces.json()["traces"][0]["message"], "订单 CC-1002 到哪了？验证码 [验证码]")

            compare = client.post("/api/products/compare", json={"product_ids": ["cc-75-silent", "cc-65-wireless"]})
            self.assertEqual(compare.status_code, 200)
            self.assertEqual(compare.json()["product_ids"], ["cc-75-silent", "cc-65-wireless"])

            missing = client.post("/api/orders/lookup", json={"order_id": "CC-1002", "demo_access_code": "0000"})
            self.assertEqual(missing.status_code, 403)

            created = client.post("/api/chat", json={"message": "订单 CC-1006 到货损坏，我要退款"})
            case_id = created.json()["data"]["case_id"]
            cases = client.get("/api/cases")
            self.assertEqual(cases.status_code, 200)
            self.assertEqual(cases.json()["cases"][0]["case_id"], case_id)
            detail = client.get(f"/api/cases/{case_id}")
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.json()["case"]["order_id"], "CC-1006")
            self.assertEqual(client.get("/api/cases/CASE-missing").status_code, 404)
        finally:
            db_path.unlink(missing_ok=True)

    def test_data_sources_explain_demo_boundaries(self) -> None:
        db_path = ROOT / "data" / "runtime" / f"api-test-{uuid.uuid4().hex}.db"
        try:
            with patch.dict("os.environ", {"LLM_MODE": "deterministic"}):
                client = TestClient(create_app(db_path))
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn("当前数据来源", home.text)
            response = client.get("/api/data-sources")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["demo_only"])
            self.assertEqual(payload["products"]["count"], 6)
            self.assertEqual(payload["knowledge"]["count"], 5)
            self.assertFalse(payload["products"]["live"])
            self.assertFalse(payload["orders"]["live"])
            self.assertFalse(payload["cases"]["human_connected"])
        finally:
            db_path.unlink(missing_ok=True)

    def test_mockshop_catalog_is_read_only_and_separate(self) -> None:
        db_path = ROOT / "data" / "runtime" / f"api-test-{uuid.uuid4().hex}.db"
        snapshot_path = ROOT / "data" / "runtime" / f"snapshot-test-{uuid.uuid4().hex}.json"
        snapshot_path.write_text(json.dumps({
            "store": {"name": "Mock", "source": "mock_shop_storefront_api", "source_url": "https://mock.shop/api"},
            "products": [{
                "id": "gid://shopify/Product/1", "handle": "sample", "title": "Sample Shoes",
                "description": "A sample product", "product_type": "product", "vendor": "Mock",
                "price": 19.5, "currency": "CAD", "available_for_sale": True,
                "layout": "未提供", "switch_type": "未提供", "noise_level": "未提供",
                "connectivity": [], "compatibility": [], "use_cases": [], "tags": ["shoes"],
                "variants": [], "source_url": "https://mock.shop/api",
            }],
        }, ensure_ascii=False), encoding="utf-8")
        try:
            with patch.dict("os.environ", {
                "LLM_MODE": "deterministic",
                "MOCKSHOP_SNAPSHOT_PATH": str(snapshot_path),
            }):
                client = TestClient(create_app(db_path))
            response = client.get("/api/catalogs/mockshop/products", params={"query": "shoes"})
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["independent"])
            self.assertEqual(payload["catalog"]["product_count"], 1)
            self.assertEqual(payload["products"][0]["id"], "gid://shopify/Product/1")
            primary_products = client.get("/api/products").json()["products"]
            self.assertEqual(len(primary_products), 6)
            self.assertNotIn("gid://shopify/Product/1", {item["id"] for item in primary_products})
        finally:
            db_path.unlink(missing_ok=True)
            snapshot_path.unlink(missing_ok=True)


class CatalogSyncTests(unittest.TestCase):
    def test_normalize_uses_storefront_money_amount_and_currency(self) -> None:
        payload = {"data": {"shop": {"name": "Mock"}, "products": {"nodes": [{
            "id": "gid://shopify/Product/1", "title": "Sample", "variants": {"nodes": [{
                "id": "gid://shopify/ProductVariant/1", "title": "Default",
                "price": {"amount": "19.50", "currencyCode": "CAD"}, "availableForSale": True,
            }]},
        }]}}}
        catalog = normalize(payload, "https://mock.shop/api")
        product = catalog["products"][0]
        self.assertEqual(product["price"], 19.5)
        self.assertEqual(product["currency"], "CAD")
        self.assertEqual(product["variants"][0]["price"], 19.5)

    def test_normalize_rejects_partial_catalog(self) -> None:
        payload = {"data": {"products": {"pageInfo": {"hasNextPage": True}, "nodes": []}}}
        with self.assertRaisesRegex(ValueError, "partial snapshot"):
            normalize(payload, "https://mock.shop/api")


if __name__ == "__main__":
    unittest.main()
