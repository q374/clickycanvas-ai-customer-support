from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from scripts.evaluate_demo import build_agent
from scripts.evaluate_holdout import evaluate
from app.llm import DeepSeekOrchestrator, LLMDraft
from app.models import ChatResponse


ROOT = Path(__file__).resolve().parents[1]


class HoldoutTests(unittest.TestCase):
    def test_frozen_holdout_has_required_fields_and_sources(self) -> None:
        payload = json.loads((ROOT / "data" / "p6_holdout_cases.json").read_text(encoding="utf-8"))
        cases = payload["cases"]
        blind = [case for case in cases if case.get("blind") and case.get("source") == "new_holdout"]
        self.assertGreaterEqual(len(blind), 30)
        self.assertIn("user_observed_2026-10-09", {case["source"] for case in cases})
        self.assertIn("manual_design", {case["source"] for case in cases})
        for case in cases:
            self.assertTrue(case["id"])
            self.assertTrue(case["group"])
            self.assertTrue(case["turns"])
            for turn in case["turns"]:
                for field in ("original_query", "context", "expected_intent", "expected_slots",
                              "required_conditions", "ranking_reason", "answer_boundary"):
                    self.assertIn(field, turn, f"{case['id']} missing {field}")
                self.assertIn("expected_product_ids", turn)

    def test_holdout_evaluator_emits_reproducible_metric_keys(self) -> None:
        db_path = ROOT / "data" / "runtime" / f"holdout-test-{uuid.uuid4().hex}.db"
        try:
            cases = json.loads((ROOT / "data" / "p6_holdout_cases.json").read_text(encoding="utf-8"))["cases"]
            cases = [case for case in cases if case.get("blind") and case.get("source") == "new_holdout"]
            report = evaluate(build_agent(db_path, "deterministic"), cases, "deterministic")
            for key in (
                "slot_accuracy", "unnecessary_clarification_rate", "hard_condition_violation_rate",
                "candidate_coverage", "preference_order_accuracy", "fact_consistency",
                "no_evidence_commitment_rate", "model_fallback_rate", "response_latency_ms_p95",
            ):
                self.assertIn(key, report["metrics"])
            self.assertEqual(report["total"], len(cases))
            self.assertEqual(report["metrics"]["model_fallback_rate"], None)
        finally:
            db_path.unlink(missing_ok=True)

    def test_deepseek_requests_disable_thinking_for_bounded_json(self) -> None:
        class CompletionStub:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {
                    "choices": [type("Choice", (), {
                        "message": type("Message", (), {"content": '{"intent":"PRODUCT_SEARCH","answer":"可用","confidence":0.8}'})()
                    })()]
                })()

        completion = CompletionStub()
        orchestrator = object.__new__(DeepSeekOrchestrator)
        orchestrator.model = "fake"
        orchestrator.client = type("Client", (), {"chat": type("Chat", (), {"completions": completion})()})()
        orchestrator.rewrite("推荐办公室键盘", ChatResponse(intent="PRODUCT_SEARCH", answer="可用"))
        self.assertEqual(completion.kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertGreaterEqual(completion.kwargs["max_tokens"], 700)


if __name__ == "__main__":
    unittest.main()
