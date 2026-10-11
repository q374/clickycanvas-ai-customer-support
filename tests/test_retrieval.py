from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from app.db import Database
from scripts.evaluate_retrieval import evaluate


ROOT = Path(__file__).resolve().parents[1]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_chunk_retrieval_is_repeatable_and_explainable(self) -> None:
        database_path = ROOT / "data" / "runtime" / f"retrieval-test-{uuid.uuid4().hex}.db"
        cases = json.loads((ROOT / "data" / "retrieval_cases.json").read_text(encoding="utf-8"))["cases"]
        try:
            report = evaluate(cases, database_path)
            self.assertEqual(report["total"], 24)
            self.assertEqual(report["top1_hits"], 24)
            self.assertEqual(report["top2_hits"], 24)
            self.assertEqual(report["mrr"], 1.0)
            self.assertEqual(report["retrieval_method"], "lexical_chunk_with_scores")
            self.assertTrue(report["results"][0]["hits"][0]["matched_terms"])
        finally:
            database_path.unlink(missing_ok=True)

    def test_hybrid_retrieval_uses_semantic_score_when_provider_is_available(self) -> None:
        class FakeEmbeddingProvider:
            model = "fake-embedding"

            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] if "退" in text else [0.0, 1.0] for text in texts]

        database_path = ROOT / "data" / "runtime" / f"hybrid-retrieval-test-{uuid.uuid4().hex}.db"
        try:
            database = Database(database_path, embedding_provider=FakeEmbeddingProvider())
            database.seed_knowledge(ROOT / "data" / "knowledge")
            hits = database.retrieve_knowledge("开封了还能退吗", limit=3)
            self.assertTrue(hits)
            self.assertEqual(hits[0].document_id, "returns")
            self.assertEqual(hits[0].retrieval_method, "hybrid_lexical_semantic")
            self.assertIsNotNone(hits[0].semantic_score)
        finally:
            database_path.unlink(missing_ok=True)

    def test_hybrid_retrieval_falls_back_when_embedding_provider_fails(self) -> None:
        class BrokenEmbeddingProvider:
            model = "broken"

            def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("simulated embedding API failure")

        database_path = ROOT / "data" / "runtime" / f"hybrid-fallback-test-{uuid.uuid4().hex}.db"
        try:
            database = Database(database_path, embedding_provider=BrokenEmbeddingProvider())
            database.seed_knowledge(ROOT / "data" / "knowledge")
            hits = database.retrieve_knowledge("多久发货", limit=3)
            self.assertTrue(hits)
            self.assertEqual(hits[0].document_id, "shipping")
            self.assertEqual(hits[0].retrieval_method, "lexical_chunk")
            self.assertIsNone(hits[0].semantic_score)
        finally:
            database_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
