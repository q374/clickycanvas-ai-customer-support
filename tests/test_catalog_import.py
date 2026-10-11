from __future__ import annotations

import json
import csv
import io
import sqlite3
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from app.catalog_import import CatalogImportError, import_catalog, parse_csv, read_catalog, validate_catalog
from app.db import Database


ROOT = Path(__file__).resolve().parents[1]


class CatalogImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = ROOT / "data" / "runtime" / f"import-test-{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)

    def tearDown(self) -> None:
        for file in self.directory.iterdir():
            file.unlink()
        self.directory.rmdir()

    def _product(self, **overrides: object) -> dict[str, object]:
        product: dict[str, object] = {
            "id": "demo-1",
            "handle": "demo-1",
            "title": "Demo Keyboard",
            "description": "A demo product",
            "product_type": "mechanical_keyboard",
            "vendor": "ClickyCanvas",
            "price": 299,
            "currency": "cny",
            "layout": "60%",
            "switch_type": "linear",
            "noise_level": "medium",
            "connectivity": ["USB-C"],
            "compatibility": ["Windows"],
            "use_cases": ["office"],
            "tags": ["demo"],
            "variants": [{"id": "demo-1-blue", "title": "Blue", "price": 299}],
        }
        product.update(overrides)
        return product

    def test_json_normalizes_currency_and_defaults(self) -> None:
        catalog = validate_catalog({
            "store": {"source_url": "https://example.com/demo", "fetched_at": "2026-10-09T00:00:00Z"},
            "products": [self._product()],
        })
        self.assertEqual(catalog["products"][0]["currency"], "CNY")
        self.assertTrue(catalog["products"][0]["available_for_sale"])
        self.assertEqual(catalog["products"][0]["source_url"], "https://example.com/demo")
        self.assertEqual(catalog["products"][0]["fetched_at"], "2026-10-09T00:00:00Z")

    def test_csv_import_supports_lists_and_variants(self) -> None:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "id", "handle", "title", "description", "product_type", "vendor", "price", "currency",
            "layout", "switch_type", "noise_level", "available_for_sale", "connectivity",
            "compatibility", "use_cases", "tags", "variants",
        ])
        writer.writeheader()
        writer.writerow({
            "id": "demo-1", "handle": "demo-1", "title": "Demo Keyboard", "description": "A demo product",
            "product_type": "mechanical_keyboard", "vendor": "ClickyCanvas", "price": 299, "currency": "CNY",
            "layout": "60%", "switch_type": "linear", "noise_level": "medium", "available_for_sale": "true",
            "connectivity": "USB-C;Bluetooth", "compatibility": "Windows;macOS", "use_cases": "office",
            "tags": "demo", "variants": json.dumps([{"id": "demo-1-blue", "title": "Blue", "price": 299}]),
        })
        csv_text = output.getvalue()
        catalog = parse_csv(csv_text)
        product = catalog["products"][0]
        self.assertEqual(product["connectivity"], ["USB-C", "Bluetooth"])
        self.assertEqual(product["variants"][0]["id"], "demo-1-blue")

    def test_missing_required_field_is_rejected(self) -> None:
        product = self._product()
        del product["vendor"]
        with self.assertRaisesRegex(CatalogImportError, "missing required field.*vendor"):
            validate_catalog({"products": [product]})

    def test_price_and_currency_are_validated(self) -> None:
        for price in [-1, "not a number", float("nan"), float("inf"), "1e9999", True]:
            with self.subTest(price=price), self.assertRaisesRegex(CatalogImportError, "price"):
                validate_catalog({"products": [self._product(price=price)]})
        for currency in ["人民币", "", "CN", "123", None]:
            with self.subTest(currency=currency), self.assertRaisesRegex(CatalogImportError, "currency"):
                validate_catalog({"products": [self._product(currency=currency)]})
        catalog = validate_catalog({"products": [self._product(price=0, variants=[])]})
        self.assertEqual(catalog["products"][0]["price"], 0)

    def test_fields_and_variants_are_validated(self) -> None:
        invalid_products = [
            self._product(title="  "), self._product(connectivity="USB-C"),
            self._product(tags=["demo", 1]), self._product(available_for_sale="maybe"),
            self._product(variants=[{"id": "v", "title": "Blue"}]),
            self._product(variants=[{"id": "v", "title": "Blue", "price": -1}]),
            self._product(variants=[{"id": "v", "title": "Blue", "price": 1, "currency": "USD"}]),
            self._product(prcie=100),
        ]
        for product in invalid_products:
            with self.subTest(product=product), self.assertRaises(CatalogImportError):
                validate_catalog({"products": [product]})

    def test_empty_or_invalid_catalog_is_rejected(self) -> None:
        for payload in [[], {}, {"products": []}, {"products": {}}, {"products": [None]}]:
            with self.subTest(payload=payload), self.assertRaises(CatalogImportError):
                validate_catalog(payload)

    def test_malformed_csv_is_rejected(self) -> None:
        header = "id,handle,title,description,product_type,vendor,price,currency,layout,switch_type,noise_level"
        for text in [
            "id,title\np1,Product", header + ",title\n", header + ",typo\n",
            header + "\na,b", header + "\n" + ",".join(["x"] * 12),
        ]:
            with self.subTest(text=text), self.assertRaises(CatalogImportError):
                parse_csv(text)

    def test_csv_preserves_multiline_description(self) -> None:
        header = "id,handle,title,description,product_type,vendor,price,currency,layout,switch_type,noise_level"
        catalog = parse_csv(header + '\np1,p1,Keyboard,"Line 1\nLine 2",keyboard,Demo,299,CNY,60%,linear,medium\n')
        self.assertEqual(catalog["products"][0]["description"], "Line 1\nLine 2")

    def test_mixed_currency_is_rejected(self) -> None:
        second = self._product(id="demo-2", handle="demo-2", currency="USD", variants=[])
        with self.assertRaisesRegex(CatalogImportError, "mixed currencies"):
            validate_catalog({"products": [self._product(), second]})

    def test_duplicate_product_and_variant_ids_are_rejected(self) -> None:
        duplicate_id = self._product(id="DEMO-1", handle="different")
        with self.assertRaisesRegex(CatalogImportError, "duplicate product id"):
            validate_catalog({"products": [self._product(), duplicate_id]})

        duplicate_variant = self._product(id="demo-2", handle="demo-2")
        with self.assertRaisesRegex(CatalogImportError, "duplicate variant id"):
            validate_catalog({"products": [self._product(), duplicate_variant]})

        duplicate_handle = self._product(id="demo-2", handle="DEMO-1", variants=[])
        with self.assertRaisesRegex(CatalogImportError, "duplicate product handle"):
            validate_catalog({"products": [self._product(), duplicate_handle]})

    def test_failed_import_keeps_existing_output(self) -> None:
        input_path = self.directory / "invalid.json"
        output_path = self.directory / "catalog.json"
        original = b"keep this catalog\r\n"
        output_path.write_bytes(original)
        input_path.write_text(json.dumps({"products": [self._product(price=-5)]}), encoding="utf-8")

        with self.assertRaises(CatalogImportError):
            import_catalog(input_path, output_path)
        self.assertEqual(output_path.read_bytes(), original)

    def test_replace_failure_keeps_file_and_cleans_temporary_file(self) -> None:
        input_path = self.directory / "valid.json"
        output_path = self.directory / "catalog.json"
        input_path.write_text(json.dumps({"products": [self._product()]}), encoding="utf-8")
        output_path.write_bytes(b"previous catalog")
        with patch("app.catalog_import.os.replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                import_catalog(input_path, output_path)
        self.assertEqual(output_path.read_bytes(), b"previous catalog")
        self.assertEqual(sorted(path.name for path in self.directory.iterdir()), ["catalog.json", "valid.json"])

    def test_successful_import_replaces_output_atomically(self) -> None:
        input_path = self.directory / "catalog.csv"
        output_path = self.directory / "catalog.json"
        output_path.write_text("old catalog", encoding="utf-8")
        input_path.write_text(
            "id,handle,title,description,product_type,vendor,price,currency,layout,switch_type,noise_level\n"
            "demo-1,demo-1,Demo Keyboard,A demo product,mechanical_keyboard,ClickyCanvas,299,CNY,60%,linear,medium\n",
            encoding="utf-8-sig",
        )
        catalog = import_catalog(input_path, output_path)
        self.assertEqual(len(catalog["products"]), 1)
        stored = read_catalog(output_path)
        self.assertEqual(stored["products"][0]["id"], "demo-1")

    def test_json_import_can_replace_input_file(self) -> None:
        path = self.directory / "catalog.json"
        path.write_text(json.dumps({"products": [self._product()]}), encoding="utf-8-sig")
        import_catalog(path, path)
        self.assertEqual(read_catalog(path)["products"][0]["currency"], "CNY")

    def test_cli_reports_failure_without_changing_target(self) -> None:
        from scripts.import_catalog import main

        path = self.directory / "invalid.json"
        output = self.directory / "catalog.json"
        path.write_text("{invalid json", encoding="utf-8")
        output.write_bytes(b"original")
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            exit_code = main(["--input", str(path), "--output", str(output)])
        self.assertEqual(exit_code, 2)
        self.assertIn("invalid JSON", stderr.getvalue())
        self.assertEqual(output.read_bytes(), b"original")

    def test_database_replaces_stale_products_without_changing_orders(self) -> None:
        database = Database(self.directory / "catalog.db")
        database.seed_catalog(ROOT / "data" / "catalog.json")
        database.seed_orders(ROOT / "data" / "orders.json")
        path = self.directory / "catalog.json"
        path.write_text(json.dumps({"products": [self._product()]}), encoding="utf-8")
        database.seed_catalog(path)
        self.assertEqual(database.product_count(), 1)
        self.assertIsNone(database.get_product("cc-75-silent"))
        self.assertEqual(database.get_product("demo-1").variants[0].id, "demo-1-blue")
        self.assertEqual(database.order_count(), 7)
        self.assertEqual(database.lookup_order("CC-1002", "1937")[1], "ok")

    def test_database_validation_failure_preserves_products(self) -> None:
        database = Database(self.directory / "catalog.db")
        database.seed_catalog(ROOT / "data" / "catalog.json")
        path = self.directory / "invalid.json"
        path.write_text(json.dumps({"products": [self._product(), self._product(title="")] }), encoding="utf-8")
        with self.assertRaises(CatalogImportError):
            database.seed_catalog(path)
        self.assertEqual(database.product_count(), 6)
        self.assertEqual(database.get_product("cc-75-silent").variants[0].id, "cc-75-silent-black")

    def test_database_write_failure_rolls_back_catalog(self) -> None:
        database = Database(self.directory / "catalog.db")
        database.seed_catalog(ROOT / "data" / "catalog.json")
        with database.connect() as connection:
            connection.execute("""
                CREATE TRIGGER fail_new_product BEFORE INSERT ON products WHEN NEW.id = 'demo-2'
                BEGIN SELECT RAISE(ABORT, 'test write failure'); END;
            """)
        path = self.directory / "catalog.json"
        second = self._product(id="demo-2", handle="demo-2", variants=[])
        path.write_text(json.dumps({"products": [self._product(), second]}), encoding="utf-8")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "test write failure"):
            database.seed_catalog(path)
        self.assertEqual(database.product_count(), 6)
        self.assertIsNone(database.get_product("demo-1"))
        self.assertEqual(database.get_product("cc-75-silent").variants[0].id, "cc-75-silent-black")


if __name__ == "__main__":
    unittest.main()
