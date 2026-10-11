from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .agent import SupportAgent
from .catalog_import import read_catalog
from .catalogs import CatalogSnapshotError, load_snapshot, search_snapshot
from .config import load_local_env
from .db import Database
from .embeddings import embedding_provider_from_env
from .llm import DeepSeekOrchestrator
from .models import ChatRequest, ChatResponse, OrderLookupRequest, ProductCompareRequest, ProductSearchFilters, ShoppingContext


ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "catalog.json"
ORDERS_PATH = ROOT / "data" / "orders.json"
KNOWLEDGE_PATH = ROOT / "data" / "knowledge"
STATIC_PATH = ROOT / "static" / "index.html"

load_local_env()


def create_app(database_path: str | Path | None = None) -> FastAPI:
    path = Path(database_path or os.getenv("CLICKYCANVAS_DB", ROOT / "data" / "runtime" / "clickycanvas.db"))
    mockshop_snapshot_path = Path(os.getenv("MOCKSHOP_SNAPSHOT_PATH", ROOT / "data" / "runtime" / "mockshop.snapshot.json"))
    if not mockshop_snapshot_path.is_absolute():
        mockshop_snapshot_path = ROOT / mockshop_snapshot_path
    embedding_provider = embedding_provider_from_env()
    database = Database(path, embedding_provider=embedding_provider)
    database.seed_catalog(CATALOG_PATH)
    catalog_store = read_catalog(CATALOG_PATH)["store"]
    database.seed_orders(ORDERS_PATH)
    database.seed_knowledge(KNOWLEDGE_PATH)
    llm = DeepSeekOrchestrator.from_env() if os.getenv("LLM_MODE", "deterministic").lower() == "deepseek" else None
    agent = SupportAgent(database, llm=llm)

    api = FastAPI(title="ClickyCanvas 客服 Demo", version="0.1.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_PATH)

    @api.get("/health")
    def health() -> dict[str, str | int]:
        return {"status": "ok", "products": database.product_count(), "orders": database.order_count()}

    @api.get("/api/stats")
    def stats() -> dict[str, int]:
        return {"products": database.product_count(), "orders": database.order_count()}

    @api.get("/api/data-sources")
    def data_sources() -> dict[str, object]:
        return {
            "demo_only": True,
            "products": {
                "name": catalog_store.get("name") or "本地商品目录",
                "source": catalog_store.get("source") or "local_import",
                "file": "data/catalog.json",
                "count": database.product_count(),
                "fetched_at": catalog_store.get("fetched_at"),
                "live": False,
            },
            "knowledge": {
                "name": "本地政策与 FAQ",
                "count": database.knowledge_count(),
                "chunks": database.knowledge_chunk_count(),
                "retrieval": "hybrid_lexical_semantic" if embedding_provider is not None else "lexical_chunk_with_scores",
                "semantic_enabled": embedding_provider is not None,
            },
            "orders": {"name": "固定模拟订单", "count": database.order_count(), "live": False},
            "cases": {"name": "本地模拟售后案件", "human_connected": False},
            "answers": {"mode": "deepseek" if llm is not None else "deterministic"},
        }

    @api.get("/api/products")
    def products(
        query: str = Query(default="", max_length=200),
        max_price: float | None = Query(default=None, ge=0),
        layout: str | None = Query(default=None, max_length=20),
        connectivity: str | None = Query(default=None, max_length=30),
        compatibility: str | None = Query(default=None, max_length=30),
        limit: int = Query(default=10, ge=1, le=20),
    ) -> dict[str, object]:
        filters = ProductSearchFilters(
            query=query, max_price=max_price, layout=layout, connectivity=connectivity,
            compatibility=compatibility, limit=limit,
        )
        return {"products": [product.model_dump() for product in database.list_products(filters)], "filters": filters.model_dump()}

    @api.get("/api/products/{product_id}")
    def product(product_id: str) -> dict[str, object]:
        item = database.get_product(product_id)
        if item is None:
            raise HTTPException(status_code=404, detail="商品不存在")
        return {"product": item.model_dump()}

    @api.get("/api/catalogs/mockshop/products")
    def mockshop_products(
        query: str = Query(default="", max_length=200),
        max_price: float | None = Query(default=None, ge=0),
        limit: int = Query(default=10, ge=1, le=20),
    ) -> dict[str, object]:
        if not mockshop_snapshot_path.exists():
            raise HTTPException(status_code=404, detail="mock.shop 独立目录快照不可用，请先运行同步脚本")
        try:
            catalog, snapshot_products = load_snapshot(mockshop_snapshot_path)
        except (OSError, json.JSONDecodeError, CatalogSnapshotError) as exc:
            raise HTTPException(status_code=503, detail=f"mock.shop 独立目录快照无效：{exc}") from exc
        filters = {"query": query, "max_price": max_price, "limit": limit}
        products = search_snapshot(snapshot_products, query=query, max_price=max_price, limit=limit)
        return {
            "catalog": catalog,
            "products": [product.model_dump() for product in products],
            "filters": filters,
            "independent": True,
        }

    @api.post("/api/products/compare")
    def compare_products(request: ProductCompareRequest) -> dict[str, object]:
        products = database.compare_products(request.product_ids)
        if len(products) < 2:
            raise HTTPException(status_code=404, detail="至少需要找到两个可对比商品")
        return {
            "products": [product.model_dump() for product in products],
            "product_ids": request.product_ids,
        }

    @api.post("/api/orders/lookup")
    def lookup_order(request: OrderLookupRequest) -> dict[str, object]:
        order, status = database.lookup_order(request.order_id, request.demo_access_code)
        if status == "not_found":
            raise HTTPException(status_code=404, detail="订单不存在")
        if status == "invalid_code":
            raise HTTPException(status_code=403, detail="订单号或演示验证码不匹配")
        assert order is not None
        return {"order": order.model_dump()}

    @api.get("/api/cases")
    def cases(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, object]:
        return {"cases": [case.model_dump() for case in database.list_escalation_cases(limit)]}

    @api.get("/api/cases/{case_id}")
    def case(case_id: str) -> dict[str, object]:
        item = database.get_escalation_case(case_id)
        if item is None:
            raise HTTPException(status_code=404, detail="售后案件不存在")
        return {"case": item.model_dump()}

    @api.get("/api/traces")
    def traces(
        conversation_id: str | None = Query(default=None, max_length=80),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, object]:
        """为需求理解、召回、排序和回答问题提供脱敏复盘记录。"""
        return {"traces": database.list_traces(conversation_id=conversation_id, limit=limit)}

    @api.delete("/api/conversations/{conversation_id}")
    def clear_conversation(conversation_id: str) -> dict[str, object]:
        """清除匿名选购记忆；聊天追踪保留用于演示复盘。"""
        database.clear_conversation_memory(conversation_id)
        return {"conversation_id": conversation_id, "memory_cleared": True}

    @api.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        effective_request = request
        if request.shopping_context is None and request.conversation_id:
            stored_context = database.get_conversation_memory(request.conversation_id)
            if stored_context is not None:
                effective_request = request.model_copy(update={"shopping_context": stored_context})
        response = agent.chat(effective_request)
        context_payload = response.data.get("shopping_context")
        if effective_request.conversation_id and isinstance(context_payload, dict):
            database.save_conversation_memory(
                effective_request.conversation_id,
                ShoppingContext.model_validate(context_payload),
            )
        trace_id = database.record_trace(effective_request, response)
        data = dict(response.data)
        data["trace"] = {"trace_id": trace_id, "conversation_id": effective_request.conversation_id}
        return response.model_copy(update={"data": data})

    return api


app = create_app()
