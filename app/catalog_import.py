"""Validate and normalize replaceable local product catalogs.

The application consumes one stable JSON shape (``{"store": ..., "products": [...]}``).
This module keeps file parsing and validation outside the support and database
code so a future data source can produce the same shape without changing the
business logic.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from .models import Product


class CatalogImportError(ValueError):
    """Raised when a catalog cannot be safely imported."""


REQUIRED_PRODUCT_FIELDS = (
    "id",
    "handle",
    "title",
    "description",
    "product_type",
    "vendor",
    "price",
    "currency",
    "layout",
    "switch_type",
    "noise_level",
)
LIST_FIELDS = ("connectivity", "compatibility", "use_cases", "tags")
CSV_FIELDS = (*REQUIRED_PRODUCT_FIELDS, "available_for_sale", *LIST_FIELDS, "variants", "source_url", "fetched_at")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


def _error(index: int, message: str) -> CatalogImportError:
    return CatalogImportError(f"product {index}: {message}")


def _text(value: Any, field: str, index: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _error(index, f"{field} must be a string")
    value = value.strip()
    if not value and not allow_empty:
        raise _error(index, f"{field} must not be empty")
    return value


def _boolean(value: Any, field: str, index: int, *, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "是"}:
            return True
        if normalized in {"false", "0", "no", "n", "否"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise _error(index, f"{field} must be a boolean")


def _price(value: Any, field: str, index: int) -> float:
    if isinstance(value, bool) or value is None:
        raise _error(index, f"{field} must be a non-negative number")
    try:
        number = float(Decimal(str(value).strip()))
    except (InvalidOperation, ValueError, TypeError):
        raise _error(index, f"{field} must be a non-negative number") from None
    if not math.isfinite(number) or number < 0:
        raise _error(index, f"{field} must be a non-negative finite number")
    return number


def _currency(value: Any, index: int) -> str:
    currency = _text(value, "currency", index).upper()
    if not _CURRENCY_RE.fullmatch(currency):
        raise _error(index, "currency must be a three-letter ISO-style code such as CNY or USD")
    return currency


def _string_list(value: Any, field: str, index: int) -> list[str]:
    if value is None or value == "":
        return []
    if not isinstance(value, list):
        raise _error(index, f"{field} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise _error(index, f"{field} must contain non-empty strings")
        result.append(item.strip())
    return result


def _variant(raw: Any, product_index: int, variant_index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _error(product_index, f"variants[{variant_index}] must be an object")
    required = ("id", "title", "price")
    missing = [field for field in required if field not in raw]
    if missing:
        raise _error(product_index, f"variants[{variant_index}] missing required field(s): {', '.join(missing)}")
    unknown = set(raw) - {*required, "available_for_sale"}
    if unknown:
        raise _error(product_index, f"variants[{variant_index}] unknown field(s): {', '.join(sorted(unknown))}")
    variant_id = _text(raw["id"], f"variants[{variant_index}].id", product_index)
    title = _text(raw["title"], f"variants[{variant_index}].title", product_index)
    return {
        "id": variant_id,
        "title": title,
        "price": _price(raw["price"], f"variants[{variant_index}].price", product_index),
        "available_for_sale": _boolean(raw.get("available_for_sale"), f"variants[{variant_index}].available_for_sale", product_index),
    }


def validate_catalog(payload: Any, *, source: str = "catalog") -> dict[str, Any]:
    """Validate a catalog and return a normalized, JSON-serializable payload.

    A catalog has one currency across all products. Product IDs, handles and
    variant IDs are unique case-insensitively. Defaults are applied only after
    all required fields have been checked.
    """

    if not isinstance(payload, Mapping):
        raise CatalogImportError(f"{source}: top-level value must be an object")
    raw_products = payload.get("products")
    if not isinstance(raw_products, list):
        raise CatalogImportError(f"{source}: products must be a list")
    if not raw_products:
        raise CatalogImportError(f"{source}: products must not be empty")

    store = payload.get("store", {})
    if store is None:
        store = {}
    if not isinstance(store, Mapping):
        raise CatalogImportError(f"{source}: store must be an object")
    allowed_store_fields = {"name", "source", "source_url", "fetched_at"}
    unknown_store_fields = set(store) - allowed_store_fields
    if unknown_store_fields:
        raise CatalogImportError(f"{source}: unknown store field(s): {', '.join(sorted(unknown_store_fields))}")
    normalized_store = dict(store)
    for field, value in normalized_store.items():
        if value is not None and not isinstance(value, str):
            raise CatalogImportError(f"{source}: store.{field} must be a string or null")
    seen_ids: dict[str, int] = {}
    seen_handles: dict[str, int] = {}
    seen_variant_ids: dict[str, tuple[int, int]] = {}
    currencies: set[str] = set()
    products: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_products, start=1):
        if not isinstance(raw, Mapping):
            raise _error(index, "must be an object")
        missing = [field for field in REQUIRED_PRODUCT_FIELDS if field not in raw]
        if missing:
            raise _error(index, f"missing required field(s): {', '.join(missing)}")
        unknown = set(raw) - set(CSV_FIELDS)
        if unknown:
            raise _error(index, f"unknown field(s): {', '.join(sorted(unknown))}")

        product_id = _text(raw["id"], "id", index)
        handle = _text(raw["handle"], "handle", index)
        id_key = product_id.casefold()
        handle_key = handle.casefold()
        if id_key in seen_ids:
            raise _error(index, f"duplicate product id {product_id!r} (already used by product {seen_ids[id_key]})")
        if handle_key in seen_handles:
            raise _error(index, f"duplicate product handle {handle!r} (already used by product {seen_handles[handle_key]})")
        seen_ids[id_key] = index
        seen_handles[handle_key] = index

        currency = _currency(raw["currency"], index)
        currencies.add(currency)
        variants_raw = raw.get("variants", [])
        if variants_raw is None:
            variants_raw = []
        if not isinstance(variants_raw, list):
            raise _error(index, "variants must be a list")
        variants = []
        for variant_index, variant_raw in enumerate(variants_raw, start=1):
            variant = _variant(variant_raw, index, variant_index)
            variant_key = variant["id"].casefold()
            if variant_key in seen_variant_ids:
                previous_product, previous_variant = seen_variant_ids[variant_key]
                raise _error(index, f"duplicate variant id {variant['id']!r} (already used by product {previous_product}, variant {previous_variant})")
            seen_variant_ids[variant_key] = (index, variant_index)
            variants.append(variant)

        normalized: dict[str, Any] = {
            "id": product_id,
            "handle": handle,
            "title": _text(raw["title"], "title", index),
            "description": _text(raw["description"], "description", index, allow_empty=True),
            "product_type": _text(raw["product_type"], "product_type", index),
            "vendor": _text(raw["vendor"], "vendor", index),
            "price": _price(raw["price"], "price", index),
            "currency": currency,
            "available_for_sale": _boolean(raw.get("available_for_sale"), "available_for_sale", index),
            "layout": _text(raw["layout"], "layout", index),
            "switch_type": _text(raw["switch_type"], "switch_type", index),
            "noise_level": _text(raw["noise_level"], "noise_level", index),
            "connectivity": _string_list(raw.get("connectivity", []), "connectivity", index),
            "compatibility": _string_list(raw.get("compatibility", []), "compatibility", index),
            "use_cases": _string_list(raw.get("use_cases", []), "use_cases", index),
            "tags": _string_list(raw.get("tags", []), "tags", index),
            "variants": variants,
            "source_url": raw.get("source_url") or store.get("source_url"),
            "fetched_at": raw.get("fetched_at") or store.get("fetched_at"),
        }
        try:
            normalized = Product.model_validate(normalized).model_dump(mode="json")
        except ValidationError as exc:
            detail = exc.errors()[0].get("msg", "invalid product")
            raise _error(index, detail) from exc
        products.append(normalized)

    if len(currencies) > 1:
        values = ", ".join(sorted(currencies))
        raise CatalogImportError(f"{source}: mixed currencies are not allowed ({values})")

    return {"store": normalized_store, "products": products}


def _csv_value(value: str | None, field: str, row_number: int) -> Any:
    value = (value or "").strip()
    if field in LIST_FIELDS:
        if not value:
            return []
        if value.startswith("["):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise CatalogImportError(f"row {row_number}: {field} must be a semicolon-separated list or JSON list") from exc
            return parsed
        return [part.strip() for part in re.split(r"[;|]", value) if part.strip()]
    if field == "variants":
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CatalogImportError(f"row {row_number}: variants must be a JSON list") from exc
        return parsed
    return value


def parse_csv(text: str, *, source: str = "catalog.csv", store_name: str = "Imported catalog") -> dict[str, Any]:
    """Parse the documented one-product-per-row CSV format."""

    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff"), newline=""), strict=True)
    if not reader.fieldnames:
        raise CatalogImportError(f"{source}: CSV header is required")
    reader.fieldnames = [header.strip() for header in reader.fieldnames]
    headers = set(reader.fieldnames)
    if len(headers) != len(reader.fieldnames):
        raise CatalogImportError(f"{source}: duplicate CSV column names")
    unknown = headers - set(CSV_FIELDS)
    if unknown:
        raise CatalogImportError(f"{source}: unknown CSV column(s): {', '.join(sorted(unknown))}")
    missing = [field for field in REQUIRED_PRODUCT_FIELDS if field not in headers]
    if missing:
        raise CatalogImportError(f"{source}: CSV missing required column(s): {', '.join(missing)}")
    products: list[dict[str, Any]] = []
    try:
        for row in reader:
            row_number = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise CatalogImportError(f"row {row_number}: number of cells must match the CSV header")
            if not any(value.strip() for value in row.values()):
                continue
            raw = {field: _csv_value(row.get(field), field, row_number) for field in CSV_FIELDS if field in row}
            products.append(raw)
    except csv.Error as exc:
        raise CatalogImportError(f"{source}: invalid CSV at line {reader.line_num}: {exc}") from exc
    return validate_catalog({"store": {"name": store_name, "source": "csv_import"}, "products": products}, source=source)


def read_catalog(path: str | Path, *, format: Literal["auto", "json", "csv"] = "auto") -> dict[str, Any]:
    """Read JSON or CSV and return the normalized canonical catalog."""

    input_path = Path(path)
    try:
        text = input_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise CatalogImportError(f"cannot read {input_path}: {exc}") from exc
    selected = format
    if selected == "auto":
        selected = input_path.suffix.lower().lstrip(".")
    if selected not in {"csv", "json"}:
        raise CatalogImportError("input format must be JSON or CSV (use --format for other extensions)")
    if selected == "csv":
        return parse_csv(text, source=str(input_path))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogImportError(f"{input_path}: invalid JSON at line {exc.lineno}, column {exc.colno}") from exc
    return validate_catalog(payload, source=str(input_path))


def write_catalog_atomic(catalog: Mapping[str, Any], path: str | Path) -> None:
    """Write a validated catalog without exposing a partially-written target."""

    destination = Path(path)
    normalized = validate_catalog(catalog, source=str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(normalized, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def import_catalog(input_path: str | Path, output_path: str | Path, *, format: Literal["auto", "json", "csv"] = "auto") -> dict[str, Any]:
    """Read, validate and atomically replace an output catalog file."""

    catalog = read_catalog(input_path, format=format)
    write_catalog_atomic(catalog, output_path)
    return catalog
