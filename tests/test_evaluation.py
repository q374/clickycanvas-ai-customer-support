from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from app.agent import SupportAgent
from app.db import Database
from app.main import CATALOG_PATH, KNOWLEDGE_PATH, ORDERS_PATH
from scripts.evaluate_demo import evaluate


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_structured_evaluation_passes_fixture_cases(self) -> None:
        db_path = ROOT / "data" / "runtime" / f"eval-test-{uuid.uuid4().hex}.db"
        try:
            database = Database(db_path)
            database.seed_catalog(CATALOG_PATH)
            database.seed_orders(ORDERS_PATH)
            database.seed_knowledge(KNOWLEDGE_PATH)
            cases = json.loads((ROOT / "data" / "eval_cases.json").read_text(encoding="utf-8"))["cases"]
            report = evaluate(SupportAgent(database), cases)
            self.assertEqual(report["total"], len(cases))
            self.assertEqual(report["passed"], len(cases))
            self.assertEqual(report["failures"], [])
        finally:
            db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
