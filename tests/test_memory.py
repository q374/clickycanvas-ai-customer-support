from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from app.db import Database, _now
from app.models import ShoppingContext


ROOT = Path(__file__).resolve().parents[1]


class AnonymousConversationMemoryTests(unittest.TestCase):
    def test_memory_round_trip_and_clear(self) -> None:
        database_path = ROOT / "data" / "runtime" / f"memory-test-{uuid.uuid4().hex}.db"
        try:
            database = Database(database_path)
            context = ShoppingContext(max_price=800, use_case="办公", last_product_ids=["cc-75-silent"])
            database.save_conversation_memory("session-1", context)
            loaded = database.get_conversation_memory("session-1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.max_price, 800)
            self.assertEqual(loaded.last_product_ids, ["cc-75-silent"])

            database.clear_conversation_memory("session-1")
            self.assertIsNone(database.get_conversation_memory("session-1"))
        finally:
            database_path.unlink(missing_ok=True)

    def test_expired_or_corrupt_memory_is_not_used(self) -> None:
        database_path = ROOT / "data" / "runtime" / f"memory-expiry-test-{uuid.uuid4().hex}.db"
        try:
            database = Database(database_path)
            with database.connect() as connection:
                connection.execute(
                    "INSERT INTO anonymous_conversation_memory VALUES (?, ?, ?, ?, ?)",
                    ("expired", json.dumps({"max_price": 800}), _now(), _now(), "2000-01-01T00:00:00+00:00"),
                )
                connection.execute(
                    "INSERT INTO anonymous_conversation_memory VALUES (?, ?, ?, ?, ?)",
                    ("corrupt", "{bad", _now(), _now(), "2999-01-01T00:00:00+00:00"),
                )
            self.assertIsNone(database.get_conversation_memory("expired"))
            self.assertIsNone(database.get_conversation_memory("corrupt"))
            with database.connect() as connection:
                count = connection.execute("SELECT COUNT(*) FROM anonymous_conversation_memory").fetchone()[0]
            self.assertEqual(count, 0)
        finally:
            database_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
