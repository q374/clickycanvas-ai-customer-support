"""Read-only helpers for catalogs kept outside the primary support catalog."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import Product


class CatalogSnapshotError(ValueError):
    """Raised when an external catalog snapshot cannot be trusted."""


def load_snapshot(path: str | Path) -> tuple[dict[str, Any], list[Product]]:
    """Load and validate an external catalog without writing it to the main DB."""
    snapshot_path = Path(path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CatalogSnapshotError("快照根节点必须是对象")

    raw_products = payload.get("products", [])
    if not isinstance(raw_products, list):
        raise CatalogSnapshotError("快照商品列表格式无效")

    products: list[Product] = []
    for index, raw in enumerate(raw_products):
        try:
            products.append(Product.model_validate(raw))
        except ValidationError as exc:
            raise CatalogSnapshotError(f"第 {index + 1} 个商品格式无效") from exc

    store = payload.get("store") or {}
    if not isinstance(store, dict):
        raise CatalogSnapshotError("快照店铺信息格式无效")
    metadata = {
        "id": "mockshop",
        "name": store.get("name") or "Mock.shop Store",
        "source": store.get("source") or "mock_shop_storefront_api",
        "source_url": store.get("source_url"),
        "fetched_at": store.get("fetched_at"),
        "product_count": len(products),
    }
    return metadata, products


def search_snapshot(products: list[Product], query: str = "", max_price: float | None = None, limit: int = 10) -> list[Product]:
    """Apply small, deterministic read-only filters to a snapshot."""
    query_parts = [part.lower() for part in re.findall(r"[A-Za-z0-9%+.-]+|[\u4e00-\u9fff]+", query) if len(part) > 1]
    matched: list[Product] = []
    for product in products:
        if max_price is not None and product.price > max_price:
            continue
        haystack = " ".join([
            product.title, product.description, product.product_type,
            product.vendor, *product.tags,
        ]).lower()
        if query_parts and not all(part in haystack for part in query_parts):
            continue
        matched.append(product)
    matched.sort(key=lambda product: (product.price, product.title.lower()))
    return matched[:limit]
