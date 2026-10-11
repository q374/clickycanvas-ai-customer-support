"""Measure the explainable chunk retrieval layer used by policy RAG."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import Database
from app.embeddings import EmbeddingProvider, embedding_provider_from_env
from app.main import CATALOG_PATH, KNOWLEDGE_PATH, ORDERS_PATH


ROOT = Path(__file__).resolve().parents[1]


def evaluate(cases: list[dict], database_path: Path, embedding_provider: EmbeddingProvider | None = None) -> dict:
    database = Database(database_path, embedding_provider=embedding_provider)
    database.seed_catalog(CATALOG_PATH)
    database.seed_orders(ORDERS_PATH)
    database.seed_knowledge(KNOWLEDGE_PATH)
    results = []
    durations_ms: list[float] = []
    for case in cases:
        started = time.perf_counter()
        hits = database.retrieve_knowledge(case["query"], limit=2)
        elapsed_ms = (time.perf_counter() - started) * 1000
        durations_ms.append(elapsed_ms)
        documents = [hit.document_id for hit in hits]
        rank = documents.index(case["expected_document"]) + 1 if case["expected_document"] in documents else None
        results.append({
            "id": case["id"],
            "query": case["query"],
            "expected_document": case["expected_document"],
            "documents": documents,
            "hits": [{
                "document_id": hit.document_id,
                "score": hit.score,
                "semantic_score": hit.semantic_score,
                "matched_terms": hit.matched_terms,
                "retrieval_method": hit.retrieval_method,
            } for hit in hits],
            "top1_hit": bool(documents and documents[0] == case["expected_document"]),
            "top2_hit": case["expected_document"] in documents,
            "reciprocal_rank": 1 / rank if rank else 0,
            "elapsed_ms": round(elapsed_ms, 3),
        })
    sorted_durations = sorted(durations_ms)
    p95_index = min(len(sorted_durations) - 1, max(0, int(len(sorted_durations) * 0.95) - 1)) if sorted_durations else 0
    return {
        "total": len(results),
        "top1_hits": sum(result["top1_hit"] for result in results),
        "top2_hits": sum(result["top2_hit"] for result in results),
        "mrr": round(sum(result["reciprocal_rank"] for result in results) / len(results), 4) if results else 0,
        "retrieval_method": "hybrid_lexical_semantic" if embedding_provider is not None else "lexical_chunk_with_scores",
        "semantic_enabled": embedding_provider is not None,
        "latency_ms": {
            "mean": round(statistics.mean(durations_ms), 3) if durations_ms else 0,
            "p95": round(sorted_durations[p95_index], 3) if sorted_durations else 0,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评估本地政策知识检索基线或 Hybrid RAG")
    parser.add_argument("--cases", default="data/retrieval_cases.json")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    parser.add_argument("--semantic", action="store_true", help="使用 EMBEDDING_* 配置启用语义召回")
    parser.add_argument("--output", help="将完整 JSON 报告写入项目内文件")
    args = parser.parse_args()

    payload = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))
    database_path = ROOT / "data" / "runtime" / f"retrieval-eval-{uuid.uuid4().hex}.db"
    try:
        provider = embedding_provider_from_env() if args.semantic else None
        if args.semantic and provider is None:
            raise SystemExit("已请求 --semantic，但未配置 EMBEDDING_API_KEY/EMBEDDING_MODEL/EMBEDDING_BASE_URL")
        report = evaluate(payload["cases"], database_path, embedding_provider=provider)
        report["case_file"] = args.cases
        report["case_source"] = payload.get("source", "unspecified")
        if args.output:
            output_path = ROOT / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        database_path.unlink(missing_ok=True)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"检索探针：{report['total']} 条；Top-1 命中：{report['top1_hits']}/{report['total']}；Top-2 命中：{report['top2_hits']}/{report['total']}")
        for result in report["results"]:
            if not result["top1_hit"]:
                print(f"- {result['id']}：期望 {result['expected_document']}，实际 {result['documents'] or '无结果'}")


if __name__ == "__main__":
    main()
