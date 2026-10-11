"""Fetch a Shopify Storefront API catalog into the local demo schema.

This is an optional development command. The runnable demo uses the checked-in
ClickyCanvas fixture by default, so the app remains usable without a network.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


QUERY = """
query Catalog {
  shop { name }
  products(first: 100) {
    pageInfo { hasNextPage }
    nodes {
      id
      handle
      title
      description
      productType
      vendor
      tags
      variants(first: 100) {
        pageInfo { hasNextPage }
        nodes { id title price { amount currencyCode } availableForSale }
      }
    }
  }
}
"""


def endpoint(value: str) -> str:
    value = value.rstrip("/")
    return value if value.endswith("/api") else f"{value}/api"


def fetch_catalog(store_url: str) -> dict:
    request = Request(
        endpoint(store_url),
        data=json.dumps({"query": QUERY}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:  # noqa: S310 - explicit developer-provided endpoint
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    return payload


def normalize(payload: dict, source_url: str) -> dict:
    data = payload.get("data") or {}
    store_name = (data.get("shop") or {}).get("name") or "Mock.shop Store"
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    product_connection = data.get("products") or {}
    if (product_connection.get("pageInfo") or {}).get("hasNextPage"):
        raise ValueError("Product catalog exceeds one page; refusing to save a partial snapshot")
    products = []
    for item in product_connection.get("nodes", []):
        variant_connection = item.get("variants") or {}
        if (variant_connection.get("pageInfo") or {}).get("hasNextPage"):
            raise ValueError(f"Product {item['id']} variants exceed one page; refusing a partial snapshot")
        variants = []
        currencies = set()
        for variant in variant_connection.get("nodes", []):
            money = variant.get("price") or {}
            currencies.add(money.get("currencyCode"))
            variants.append({
                "id": variant["id"],
                "title": variant.get("title") or "Default",
                "price": float(money["amount"]),
                "available_for_sale": bool(variant.get("availableForSale", True)),
            })
        if len(currencies) > 1:
            raise ValueError(f"Product {item['id']} has mixed variant currencies")
        price = variants[0]["price"] if variants else 0
        products.append({
            "id": item["id"],
            "handle": item.get("handle") or item["id"].split("/")[-1],
            "title": item.get("title") or "未命名商品",
            "description": item.get("description") or "",
            "product_type": item.get("productType") or "product",
            "vendor": item.get("vendor") or store_name,
            "price": price,
            "currency": next(iter(currencies)) if currencies else "未提供",
            "available_for_sale": any(variant["available_for_sale"] for variant in variants) if variants else True,
            "layout": "未提供",
            "switch_type": "未提供",
            "noise_level": "未提供",
            "connectivity": [],
            "compatibility": [],
            "use_cases": [],
            "tags": item.get("tags") or [],
            "variants": variants,
            "source_url": source_url,
            "fetched_at": fetched_at,
        })
    return {
        "store": {"name": store_name, "source": "mock_shop_storefront_api", "source_url": source_url, "fetched_at": fetched_at},
        "products": products,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="同步 mock.shop Storefront API 商品快照")
    parser.add_argument("--store-url", default="https://mock.shop/api", help="Storefront API 地址或商店根地址")
    parser.add_argument("--output", default="data/catalog.snapshot.json", help="规范化快照输出路径")
    args = parser.parse_args()

    source_url = endpoint(args.store_url)
    payload = fetch_catalog(source_url)
    normalized = normalize(payload, source_url)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存 {len(normalized['products'])} 个商品到 {output}")


if __name__ == "__main__":
    main()
