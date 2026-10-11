"""Import a local JSON or CSV catalog into the canonical demo catalog file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_import import CatalogImportError, import_catalog  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="校验并原子替换 ClickyCanvas 本地商品目录（JSON 或 CSV）",
    )
    parser.add_argument("--input", required=True, help="待导入的 JSON/CSV 文件")
    parser.add_argument(
        "--output", default=str(ROOT / "data" / "catalog.json"),
        help="规范化 JSON 目录输出路径（默认 data/catalog.json）",
    )
    parser.add_argument("--format", choices=("auto", "json", "csv"), default="auto", help="输入格式，默认按扩展名判断")
    args = parser.parse_args(argv)

    try:
        catalog = import_catalog(args.input, args.output, format=args.format)
    except (CatalogImportError, OSError) as exc:
        print(f"导入失败：{exc}", file=sys.stderr)
        return 2
    currency = next(iter({product["currency"] for product in catalog["products"]}), "未设置")
    print(f"导入成功：{len(catalog['products'])} 个商品，币种 {currency}，已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
