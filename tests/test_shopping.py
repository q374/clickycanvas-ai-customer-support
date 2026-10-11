from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


from app.agent import SupportAgent
from app.db import Database
from app.llm import LLMDraft
from app.main import CATALOG_PATH, KNOWLEDGE_PATH, ORDERS_PATH, create_app
from app.models import ChatRequest, ShoppingContext


ROOT = Path(__file__).resolve().parents[1]


class ShoppingFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = ROOT / "data" / "runtime" / f"shopping-test-{uuid.uuid4().hex}.db"
        self.database = Database(self.db_path)
        self.database.seed_catalog(CATALOG_PATH)
        self.database.seed_orders(ORDERS_PATH)
        self.database.seed_knowledge(KNOWLEDGE_PATH)
        self.agent = SupportAgent(self.database)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def send(self, message: str, previous=None, agent=None, order_id=None):
        return (agent or self.agent).chat(ChatRequest(message=message, shopping_context=previous, order_id=order_id))

    @staticmethod
    def context(response):
        return ShoppingContext.model_validate(response.data["shopping_context"])

    def test_recommendation_keeps_slots_and_resolves_previous_products(self) -> None:
        first = self.send("推荐一下")
        office = self.send("办公", self.context(first))
        self.assertEqual(office.intent, "PRODUCT_SEARCH")
        self.assertEqual(len(office.data["products"]), 4)
        result = self.send("预算800以内", self.context(office))
        self.assertEqual([p["id"] for p in result.data["products"]], ["cc-65-wireless", "cc-75-silent", "cc-low-profile"])
        detail = self.send("第一款支持Mac吗", self.context(result))
        self.assertEqual(detail.intent, "PRODUCT_DETAIL")
        self.assertEqual(detail.data["product"]["id"], "cc-65-wireless")
        ambiguous = self.send("这两款哪个更安静", self.context(detail))
        self.assertEqual(ambiguous.intent, "PRODUCT_CLARIFY")
        narrowed = self.send("预算改成700以内", self.context(ambiguous))
        compare = self.send("这两款哪个更安静", self.context(narrowed))
        self.assertEqual([p["id"] for p in compare.data["products"]], ["cc-65-wireless", "cc-75-silent"])
        self.assertIn("低于", compare.answer)

    def test_hard_conditions_soft_ranking_and_no_silent_relaxation(self) -> None:
        hard = self.send("办公用，预算800以内，必须安静且带蓝牙")
        self.assertEqual([p["id"] for p in hard.data["products"]], ["cc-75-silent", "cc-low-profile"])
        soft = self.send("办公用，预算800以内，希望安静")
        self.assertEqual([p["id"] for p in soft.data["products"]], ["cc-75-silent", "cc-low-profile", "cc-65-wireless"])
        self.assertIsNone(self.context(soft).noise_level)
        excluded = self.send("办公用，预算800以内，不要蓝牙")
        self.assertEqual(excluded.data["products"], [])
        self.assertIn("不带Bluetooth", excluded.answer)

    def test_budget_interval_override_and_reset(self) -> None:
        range_result = self.send("预算500到800，办公")
        self.assertEqual((self.context(range_result).min_price, self.context(range_result).max_price), (500, 800))
        changed = self.send("预算改成500以内", self.context(range_result))
        self.assertEqual(changed.data["products"], [])
        self.assertEqual(self.context(changed).max_price, 500)
        reset = self.send("重新开始选购", self.context(changed))
        self.assertIsNone(self.context(reset).max_price)
        self.assertEqual(self.context(reset).last_product_ids, [])

    def test_order_verification_keeps_shopping_context_separate(self) -> None:
        products = self.send("办公，预算800以内")
        order = self.send("订单 CC-1002 到哪了？", self.context(products))
        self.assertEqual(self.context(order).use_case, "办公")
        verified = self.send("验证码 1937", self.context(order), order_id=order.data["order_id"])
        self.assertEqual(verified.data["order"]["order_id"], "CC-1002")
        self.assertEqual(self.context(verified).last_product_ids, self.context(products).last_product_ids)

    def test_model_only_supplements_supported_evidenced_slot(self) -> None:
        class Extractor:
            model = "fake"

            def extract_shopping(self, message):
                return {"suggestions": [{"field": "use_case", "value": "办公", "evidence": "开会"}]}

            def rewrite(self, user_message, deterministic):
                return LLMDraft(intent=deterministic.intent, answer=deterministic.answer, confidence=0.8)

        agent = SupportAgent(self.database, llm=Extractor())
        start = self.send("推荐一下", agent=agent)
        result = self.send("开会时用", self.context(start), agent=agent)
        self.assertEqual(result.intent, "PRODUCT_SEARCH")
        self.assertEqual(self.context(result).use_case, "办公")
        self.assertEqual(result.data["shopping_extraction"], "model")

    def test_invalid_model_extraction_falls_back_without_inventing_conditions(self) -> None:
        class BadExtractor:
            model = "fake"

            def extract_shopping(self, message):
                return {"suggestions": [{"field": "use_case", "value": "办公", "evidence": "并不存在的原话"}]}

            def rewrite(self, user_message, deterministic):
                raise AssertionError("clarification should not be rewritten")

        agent = SupportAgent(self.database, llm=BadExtractor())
        start = self.send("推荐一下", agent=agent)
        result = self.send("开会时用", self.context(start), agent=agent)
        self.assertEqual(result.intent, "PRODUCT_CLARIFY")
        self.assertIsNone(self.context(result).use_case)
        self.assertEqual(result.data["shopping_extraction"], "fallback")

    def test_model_can_supplement_a_partial_rule_parse(self) -> None:
        class Extractor:
            model = "fake"

            def extract_shopping(self, message):
                return {"suggestions": [{"field": "preference", "value": "quiet", "evidence": "声音别太大"}]}

            def rewrite(self, user_message, deterministic):
                return LLMDraft(intent=deterministic.intent, answer=deterministic.answer, confidence=0.8)

        result = self.send("打游戏，声音别太大", agent=SupportAgent(self.database, llm=Extractor()))
        self.assertEqual(result.intent, "PRODUCT_SEARCH")
        self.assertEqual(self.context(result).use_case, "游戏")
        self.assertIn("quiet", self.context(result).preferences)
        self.assertEqual(result.data["shopping_extraction"], "model")

    def test_product_rewrite_cannot_change_catalog_axis(self) -> None:
        class WrongRewrite:
            model = "fake"

            def extract_shopping(self, message):
                return {"suggestions": []}

            def rewrite(self, user_message, deterministic):
                return LLMDraft(intent=deterministic.intent,
                                answer=deterministic.answer.replace("线性轴", "青轴"), confidence=0.9)

        result = self.send("打游戏用的键盘", agent=SupportAgent(self.database, llm=WrongRewrite()))
        self.assertEqual(result.data["llm"]["status"], "fallback")
        self.assertNotIn("青轴", result.answer)

    def test_invalid_extraction_keeps_rule_based_quiet_preference(self) -> None:
        class InvalidExtractor:
            model = "fake"

            def extract_shopping(self, message):
                raise ValueError("empty model response")

            def rewrite(self, user_message, deterministic):
                return LLMDraft(intent=deterministic.intent, answer=deterministic.answer, confidence=0.8)

        result = self.send("打游戏，别太响", agent=SupportAgent(self.database, llm=InvalidExtractor()))
        self.assertEqual(self.context(result).preferences, ["quiet"])
        self.assertEqual(result.data["shopping_extraction"], "fallback")
        self.assertIn("都没有标注低噪声", result.answer)

    def test_observed_gaming_conversation_answers_without_budget_and_checks_axis(self) -> None:
        first = self.send("我想买一个打游戏用的键盘")
        self.assertEqual(first.intent, "PRODUCT_SEARCH")
        self.assertEqual([p["id"] for p in first.data["products"]], ["cc-60-starter", "cc-tkl-pro"])
        dorm = self.send("就宿舍打游戏", self.context(first))
        self.assertIn("宿舍", dorm.answer)
        self.assertNotIn("请告诉我预算范围", dorm.answer)
        axis = self.send("青轴的键盘有吗?", self.context(dorm))
        self.assertEqual(axis.data["products"], [])
        self.assertIn("没有标注青轴", axis.answer)

    def test_named_product_fact_and_unavailable_fact(self) -> None:
        known = self.send("Canvas 75 支持 Mac 吗")
        self.assertEqual(known.intent, "PRODUCT_DETAIL")
        self.assertIn("支持 macOS", known.answer)
        unknown = self.send("Canvas 75 支持热插拔吗")
        self.assertEqual(unknown.intent, "PRODUCT_DETAIL")
        self.assertIn("无法确认", unknown.answer)

    def test_customer_copy_hides_internal_source_phrasing(self) -> None:
        internal_phrases = ("演示商品快照", "以上信息来自", "当前目录", "商品目录", "目录标注")

        search = self.send("办公，预算800以内")
        self.assertTrue(all(phrase not in search.answer for phrase in internal_phrases))
        self.assertIn("ClickyCanvas 商品快照", search.sources)

        detail = self.send("Canvas 75 支持 Mac 吗")
        self.assertTrue(all(phrase not in detail.answer for phrase in internal_phrases))
        self.assertIn("ClickyCanvas 商品快照", detail.sources)

        compare = self.send("比较 cc-75-silent 和 cc-65-wireless")
        self.assertTrue(all(phrase not in compare.answer for phrase in internal_phrases))
        self.assertIn("主要差异", compare.answer)

        unknown_policy = self.send("支持保价吗")
        self.assertNotIn("当前政策资料", unknown_policy.answer)
        self.assertIn("没有找到明确的政策说明", unknown_policy.answer)

        returns = self.send("我不明白退货到底怎么走流程")
        self.assertNotIn("退换货政策", returns.answer)
        self.assertIn("提供订单号", returns.answer)
        self.assertLess(returns.answer.count("订单号"), 2)

    def test_product_search_copy_matches_candidate_count(self) -> None:
        result = self.send("想买无线键盘")
        self.assertIn("为您筛选到 4 款", result.answer)
        self.assertEqual(result.answer.count("\n- "), 4)
        self.assertNotIn("我根据", result.answer)
        self.assertNotIn("实际声音和使用感受受环境影响", result.answer)
        self.assertNotIn("以上信息适合作为初步筛选参考", result.answer)

    def test_chinese_budget_and_wired_followup_replace_previous_connection(self) -> None:
        result = self.send("第一次买入门的游戏键盘，预算四百以内")
        self.assertEqual(self.context(result).max_price, 400)
        self.assertIn("入门", self.context(result).required_tags)
        office = self.send("办公，八百以内，想要蓝牙")
        wired = self.send("连接方式换成有线，不要无线", self.context(office))
        self.assertEqual(self.context(wired).connectivity, "USB-C")
        self.assertEqual(self.context(wired).excluded_connectivity, ["Bluetooth", "2.4GHz"])

    def test_named_product_multi_fact_and_pair_reference(self) -> None:
        detail = self.send("Canvas 75 的连接方式和价格是多少")
        self.assertIn("699", detail.answer)
        self.assertIn("蓝牙", detail.answer)
        unknown = self.send("Canvas 75 能不能热插拔")
        self.assertIn("无法确认", unknown.answer)
        first = self.send("办公，预算800以内")
        comparison = self.send("前两款哪个更适合办公", self.context(first))
        self.assertEqual(comparison.intent, "PRODUCT_COMPARE")
        self.assertEqual([p["id"] for p in comparison.data["products"]], ["cc-65-wireless", "cc-75-silent"])

    def test_chat_endpoint_shopping_context_round_trip(self) -> None:
        db_path = ROOT / "data" / "runtime" / f"shopping-api-{uuid.uuid4().hex}.db"
        try:
            with patch.dict("os.environ", {"LLM_MODE": "deterministic"}):
                application = create_app(db_path)
            endpoint = next(route.endpoint for route in application.routes if getattr(route, "path", None) == "/api/chat")
            first = endpoint(ChatRequest(message="推荐一下"))
            context = first.model_dump(mode="json")["data"]["shopping_context"]
            second = endpoint(ChatRequest.model_validate({"message": "办公，预算800以内", "shopping_context": context}))
            self.assertEqual(second.intent, "PRODUCT_SEARCH")
            self.assertEqual(len(second.data["products"]), 3)
        finally:
            db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
