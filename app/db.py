from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .catalog_import import read_catalog
from .embeddings import EmbeddingProvider
from .models import (
    ChatRequest,
    ChatResponse,
    EscalationCase,
    KnowledgeDocument,
    KnowledgeHit,
    Order,
    OrderItem,
    Product,
    ProductSearchFilters,
    ProductVariant,
    ShoppingContext,
    TrackingEvent,
)

LOGGER = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    handle TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    product_type TEXT NOT NULL,
    vendor TEXT NOT NULL,
    price REAL NOT NULL,
    currency TEXT NOT NULL,
    available_for_sale INTEGER NOT NULL,
    layout TEXT NOT NULL,
    switch_type TEXT NOT NULL,
    noise_level TEXT NOT NULL,
    connectivity_json TEXT NOT NULL,
    compatibility_json TEXT NOT NULL,
    use_cases_json TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    source_url TEXT,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS product_variants (
    id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    price REAL NOT NULL,
    available_for_sale INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    section TEXT NOT NULL,
    content TEXT NOT NULL,
    source_path TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    content TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document ON knowledge_chunks(document_id, ordinal);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    demo_access_code TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payment_status TEXT NOT NULL,
    fulfillment_status TEXT NOT NULL,
    tracking_number TEXT,
    return_eligible_until TEXT,
    issue_type TEXT
);

CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    title TEXT NOT NULL,
    quantity INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tracking_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    occurred_at TEXT NOT NULL,
    status TEXT NOT NULL,
    location TEXT
);

CREATE TABLE IF NOT EXISTS escalation_cases (
    case_id TEXT PRIMARY KEY,
    order_id TEXT,
    reason TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_traces (
    trace_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    created_at TEXT NOT NULL,
    message TEXT NOT NULL,
    intent TEXT NOT NULL,
    answer TEXT NOT NULL,
    data_json TEXT NOT NULL,
    sources_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversation_traces_conversation ON conversation_traces(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS anonymous_conversation_memory (
    conversation_id TEXT PRIMARY KEY,
    context_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_anonymous_memory_expires ON anonymous_conversation_memory(expires_at);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _expires_at(ttl_hours: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(timespec="seconds")


def _search_tokens(text: str) -> list[str]:
    """Keep useful Chinese bigrams while retaining normal ASCII words."""
    tokens: list[str] = []
    for chunk in re.findall(r"[A-Za-z0-9%+.-]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.append(chunk.lower())
            tokens.extend(chunk[index:index + 2].lower() for index in range(len(chunk) - 1))
        else:
            tokens.append(chunk.lower())
    return [token for token in tokens if len(token) > 1]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算向量相似度；无效向量直接交给上层触发词法回退。"""
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have the same non-zero dimension")
    if not all(math.isfinite(float(value)) for value in [*left, *right]):
        raise ValueError("embedding vectors must contain finite numbers")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


POLICY_SEARCH_STOPWORDS = {"支持", "政策", "什么", "是否", "请问", "可以", "有没有", "多久", "怎么办"}


def _knowledge_aliases(query: str) -> list[str]:
    """Return a small, local alias set before keyword scoring."""
    lowered = query.lower()
    aliases: list[str] = []
    if "蓝牙" in lowered and any(term in lowered for term in ["连不上", "无法连接", "连接不上"]):
        aliases.append("无法连接")
    if "按键没反应" in lowered:
        aliases.append("按键失灵")
    if "苹果电脑" in lowered:
        aliases.extend(["Mac", "macOS"])
    if "进液" in lowered:
        aliases.extend(["退货", "人为损坏"])
    if "可以退" in lowered or "几天可以退" in lowered:
        aliases.extend(["退货", "7 个自然日"])
    if "交给快递" in lowered:
        aliases.extend(["发货", "物流"])
    if "配对失败" in lowered:
        aliases.append("无法连接")
    if "按键" in lowered and "没反应" in lowered:
        aliases.append("按键失灵")
    if "快递" in lowered and "没动静" in lowered:
        aliases.extend(["物流", "物流延迟", "48 小时"])
    if "送到国外" in lowered:
        aliases.extend(["海外配送", "配送范围"])
    if "出库" in lowered:
        aliases.extend(["发货", "发货时间"])
    return aliases


def _expand_knowledge_query(query: str) -> str:
    return " ".join([query, *_knowledge_aliases(query)])


def _knowledge_chunks(content: str, max_chars: int = 700) -> list[str]:
    """按段落切分政策资料，尽量让每个命中片段自洽且可展示。"""
    paragraphs = [line.strip() for line in content.splitlines() if line.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if paragraph.startswith("#") and current:
            chunks.append(current)
            current = paragraph
            continue
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [content.strip()]


def _redact_trace_message(message: str) -> str:
    """追踪只用于评估，避免把演示验证码原样写入数据库。"""
    return re.sub(r"(?<!CC-)(?<!\d)\d{4}(?!\d)", "[验证码]", message[:1000], flags=re.IGNORECASE)


class Database:
    def __init__(self, path: str | Path, embedding_provider: EmbeddingProvider | None = None):
        self.path = Path(path)
        self.embedding_provider = embedding_provider
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def seed_catalog(self, path: str | Path) -> int:
        """Replace only product tables, after validating the complete catalog."""
        payload = read_catalog(path)
        with self.connect() as connection:
            connection.execute("DELETE FROM product_variants")
            connection.execute("DELETE FROM products")
            for raw in payload["products"]:
                connection.execute(
                    """
                    INSERT INTO products (
                        id, handle, title, description, product_type, vendor, price,
                        currency, available_for_sale, layout, switch_type, noise_level,
                        connectivity_json, compatibility_json, use_cases_json, tags_json,
                        source_url, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        raw["id"], raw["handle"], raw["title"], raw["description"],
                        raw["product_type"], raw["vendor"], raw["price"], raw["currency"],
                        int(raw.get("available_for_sale", True)), raw["layout"],
                        raw["switch_type"], raw["noise_level"], _json(raw.get("connectivity", [])),
                        _json(raw.get("compatibility", [])), _json(raw.get("use_cases", [])),
                        _json(raw.get("tags", [])), raw.get("source_url"), raw.get("fetched_at"),
                    ),
                )
                for variant in raw.get("variants", []):
                    connection.execute(
                        "INSERT INTO product_variants (id, product_id, title, price, available_for_sale) VALUES (?, ?, ?, ?, ?)",
                        (variant["id"], raw["id"], variant["title"], variant["price"], int(variant.get("available_for_sale", True))),
                    )
        return len(payload["products"])

    def seed_orders(self, path: str | Path) -> int:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        with self.connect() as connection:
            for raw in payload.get("orders", []):
                connection.execute(
                    """
                    INSERT INTO orders (
                        order_id, demo_access_code, created_at, payment_status,
                        fulfillment_status, tracking_number, return_eligible_until, issue_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(order_id) DO UPDATE SET
                        demo_access_code=excluded.demo_access_code, created_at=excluded.created_at,
                        payment_status=excluded.payment_status, fulfillment_status=excluded.fulfillment_status,
                        tracking_number=excluded.tracking_number,
                        return_eligible_until=excluded.return_eligible_until, issue_type=excluded.issue_type
                    """,
                    (
                        raw["order_id"], raw["demo_access_code"], raw["created_at"], raw["payment_status"],
                        raw["fulfillment_status"], raw.get("tracking_number"),
                        raw.get("return_eligible_until"), raw.get("issue_type"),
                    ),
                )
                connection.execute("DELETE FROM order_items WHERE order_id = ?", (raw["order_id"],))
                connection.execute("DELETE FROM tracking_events WHERE order_id = ?", (raw["order_id"],))
                for item in raw.get("line_items", []):
                    connection.execute(
                        "INSERT INTO order_items (order_id, product_id, variant_id, title, quantity) VALUES (?, ?, ?, ?, ?)",
                        (raw["order_id"], item["product_id"], item["variant_id"], item["title"], item.get("quantity", 1)),
                    )
                for event in raw.get("tracking_events", []):
                    connection.execute(
                        "INSERT INTO tracking_events (order_id, occurred_at, status, location) VALUES (?, ?, ?, ?)",
                        (raw["order_id"], event["occurred_at"], event["status"], event.get("location")),
                    )
        return len(payload.get("orders", []))

    def seed_knowledge(self, directory: str | Path, updated_at: str = "2026-10-08") -> int:
        files = sorted(Path(directory).glob("*.md"))
        with self.connect() as connection:
            for file in files:
                content = file.read_text(encoding="utf-8")
                heading = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
                title = heading.group(1).strip() if heading else file.stem
                document_id = file.stem
                connection.execute(
                    """
                    INSERT INTO knowledge_documents (document_id, title, section, content, source_path, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        title=excluded.title, section=excluded.section, content=excluded.content,
                        source_path=excluded.source_path, updated_at=excluded.updated_at
                    """,
                    (document_id, title, "document", content, str(file), updated_at),
                )
                connection.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (document_id,))
                for ordinal, chunk in enumerate(_knowledge_chunks(content)):
                    connection.execute(
                        "INSERT INTO knowledge_chunks (chunk_id, document_id, ordinal, content) VALUES (?, ?, ?, ?)",
                        (f"{document_id}:{ordinal}", document_id, ordinal, chunk),
                    )
        return len(files)

    def product_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM products").fetchone()[0])

    def order_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0])

    def knowledge_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM knowledge_documents").fetchone()[0])

    def knowledge_chunk_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0])

    def list_products(self, filters: ProductSearchFilters) -> list[Product]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM products ORDER BY price ASC").fetchall()
            products = [self._product_from_row(connection, row) for row in rows]

        query_tokens = _search_tokens(filters.query)
        scored: list[tuple[int, Product]] = []
        for product in products:
            if not product.available_for_sale:
                continue
            if filters.min_price is not None and product.price < filters.min_price:
                continue
            if filters.max_price is not None and product.price > filters.max_price:
                continue
            if filters.full_keyboards_only and product.layout == "Numpad":
                continue
            if filters.use_case and filters.use_case != "日常" and not any(filters.use_case.lower() in value.lower() for value in product.use_cases):
                continue
            if filters.required_tags and not all(
                any(required.lower() == tag.lower() for tag in product.tags)
                for required in filters.required_tags
            ):
                continue
            if filters.layout and filters.layout.lower() not in product.layout.lower():
                continue
            if filters.switch_type and filters.switch_type.lower() not in product.switch_type.lower():
                continue
            if any(excluded.lower() in product.switch_type.lower() for excluded in filters.excluded_switch_types):
                continue
            if filters.connectivity == "wireless":
                if not any("bluetooth" in value.lower() or "2.4ghz" in value.lower() for value in product.connectivity):
                    continue
            elif filters.connectivity and not any(filters.connectivity.lower() in value.lower() for value in product.connectivity):
                continue
            if any(any(excluded.lower() in value.lower() for value in product.connectivity) for excluded in filters.excluded_connectivity):
                continue
            if filters.compatibility and not any(filters.compatibility.lower() in value.lower() for value in product.compatibility):
                continue
            if filters.noise_level and product.noise_level != filters.noise_level:
                continue
            haystack = " ".join([
                product.title, product.description, product.layout, product.switch_type,
                product.noise_level, *product.connectivity, *product.compatibility,
                *product.use_cases, *product.tags,
            ]).lower()
            score = sum(1 for token in query_tokens if token in haystack)
            if query_tokens and score == 0:
                continue
            if "quiet" in filters.preferences and product.noise_level == "低":
                score += 3
            if "wireless" in filters.preferences and any("bluetooth" in value.lower() or "2.4ghz" in value.lower() for value in product.connectivity):
                score += 2
            scored.append((score, product))

        scored.sort(key=lambda item: (-item[0], item[1].price))
        return [product for _, product in scored[: filters.limit]]

    def get_product(self, product_id: str) -> Product | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
            return self._product_from_row(connection, row) if row else None

    def compare_products(self, product_ids: list[str]) -> list[Product]:
        """Return products in the caller's order so comparison output is stable."""
        products: list[Product] = []
        for product_id in product_ids:
            product = self.get_product(product_id.strip().lower())
            if product is not None:
                products.append(product)
        return products

    def retrieve_knowledge(self, query: str, limit: int = 3) -> list[KnowledgeHit]:
        """检索知识片段；有 embedding 配置时融合词法和语义分数。"""
        aliases = {alias.lower() for alias in _knowledge_aliases(query)}
        query_tokens = [token for token in _search_tokens(_expand_knowledge_query(query)) if token not in POLICY_SEARCH_STOPWORDS]
        if not query_tokens:
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.document_id, c.ordinal, c.content AS chunk_content, d.content AS document_content,
                       d.title, d.section, d.source_path, d.updated_at
                FROM knowledge_chunks c JOIN knowledge_documents d ON d.document_id = c.document_id
                ORDER BY c.document_id, c.ordinal
                """
            ).fetchall()
        candidates: list[dict[str, Any]] = []
        for row in rows:
            chunk_content = row["chunk_content"]
            haystack = f"{row['title']} {chunk_content}".lower()
            matched_terms = [token for token in query_tokens if token in haystack]
            score = sum(haystack.count(token) * (3 if token in aliases else 1) for token in matched_terms)
            if score > 0 or self.embedding_provider is not None:
                candidates.append({"row": row, "lexical_score": float(score), "matched_terms": matched_terms})

        method = "lexical_chunk"
        semantic_scores: dict[str, float] = {}
        if self.embedding_provider is not None and candidates:
            try:
                texts = [query, *[item["row"]["chunk_content"] for item in candidates]]
                vectors = self.embedding_provider.embed(texts)
                query_vector = vectors[0]
                for item, vector in zip(candidates, vectors[1:], strict=True):
                    semantic_scores[item["row"]["document_id"] + ":" + str(item["row"]["ordinal"])] = _cosine_similarity(query_vector, vector)
                method = "hybrid_lexical_semantic"
            except Exception as exc:
                LOGGER.warning("Embedding retrieval fallback (%s): %s", type(exc).__name__, str(exc)[:200])

        max_lexical = max((item["lexical_score"] for item in candidates), default=0.0)
        max_semantic = max((max(score, 0.0) for score in semantic_scores.values()), default=0.0)
        scored: list[KnowledgeHit] = []
        for item in candidates:
            row = item["row"]
            key = row["document_id"] + ":" + str(row["ordinal"])
            semantic_score = semantic_scores.get(key)
            lexical_part = item["lexical_score"] / max_lexical if max_lexical else 0.0
            semantic_part = max(semantic_score or 0.0, 0.0) / max_semantic if max_semantic else 0.0
            combined_score = (0.65 * lexical_part + 0.35 * semantic_part) if method != "lexical_chunk" else item["lexical_score"]
            excerpt = " ".join(line.lstrip("# ").strip() for line in row["chunk_content"].splitlines() if line.strip())
            scored.append(KnowledgeHit(
                document_id=row["document_id"], title=row["title"], section=row["section"],
                content=row["document_content"], excerpt=excerpt[:500], source_path=row["source_path"],
                updated_at=row["updated_at"], score=float(combined_score), semantic_score=semantic_score,
                matched_terms=item["matched_terms"], retrieval_method=method,
            ))
        scored.sort(key=lambda item: (-item.score, item.document_id, item.section))
        return scored[:limit]

    def search_knowledge(self, query: str, limit: int = 3) -> list[KnowledgeDocument]:
        """兼容旧调用方的文档级检索封装。"""
        hits = self.retrieve_knowledge(query, limit=limit)
        return [KnowledgeDocument(
            document_id=hit.document_id, title=hit.title, section=hit.section,
            content=hit.content, source_path=hit.source_path, updated_at=hit.updated_at,
        ) for hit in hits]

    def record_trace(self, request: ChatRequest, response: ChatResponse) -> str:
        trace_id = uuid.uuid4().hex[:16]
        safe_data = dict(response.data)
        # 评估需要召回与槽位信息，回答本身不应把访问凭据写入追踪。
        safe_message = _redact_trace_message(request.message)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_traces
                    (trace_id, conversation_id, created_at, message, intent, answer, data_json, sources_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (trace_id, request.conversation_id, _now(), safe_message, response.intent,
                 response.answer[:4000], _json(safe_data), _json(response.sources)),
            )
        return trace_id

    def list_traces(self, conversation_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if conversation_id:
                rows = connection.execute(
                    "SELECT * FROM conversation_traces WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                    (conversation_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM conversation_traces ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [{
            "trace_id": row["trace_id"], "conversation_id": row["conversation_id"],
            "created_at": row["created_at"], "message": row["message"], "intent": row["intent"],
            "answer": row["answer"], "data": json.loads(row["data_json"]),
            "sources": json.loads(row["sources_json"]),
        } for row in rows]

    def lookup_order(self, order_id: str, demo_access_code: str) -> tuple[Order | None, str]:
        normalized_id = order_id.strip().upper()
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM orders WHERE order_id = ?", (normalized_id,)).fetchone()
            if not row:
                return None, "not_found"
            if row["demo_access_code"] != demo_access_code.strip():
                return None, "invalid_code"
            items = [
                OrderItem(
                    product_id=item["product_id"], variant_id=item["variant_id"],
                    title=item["title"], quantity=item["quantity"],
                )
                for item in connection.execute("SELECT * FROM order_items WHERE order_id = ?", (normalized_id,)).fetchall()
            ]
            events = [
                TrackingEvent(occurred_at=event["occurred_at"], status=event["status"], location=event["location"])
                for event in connection.execute("SELECT * FROM tracking_events WHERE order_id = ? ORDER BY id", (normalized_id,)).fetchall()
            ]
        return Order(
            order_id=row["order_id"], demo_access_code="****", created_at=row["created_at"],
            payment_status=row["payment_status"], fulfillment_status=row["fulfillment_status"],
            tracking_number=row["tracking_number"], return_eligible_until=row["return_eligible_until"],
            issue_type=row["issue_type"], line_items=items, tracking_events=events,
        ), "ok"

    def create_escalation_case(self, order_id: str | None, reason: str, summary: str) -> EscalationCase:
        case_id = f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[-10:]}"
        case = EscalationCase(case_id=case_id, order_id=order_id, reason=reason, summary=summary, created_at=_now())
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO escalation_cases (case_id, order_id, reason, summary, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (case.case_id, case.order_id, case.reason, case.summary, case.status, case.created_at),
            )
        return case

    def list_escalation_cases(self, limit: int = 20) -> list[EscalationCase]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM escalation_cases ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            EscalationCase(
                case_id=row["case_id"], order_id=row["order_id"], reason=row["reason"],
                summary=row["summary"], status=row["status"], created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_escalation_case(self, case_id: str) -> EscalationCase | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM escalation_cases WHERE case_id = ?", (case_id.strip(),)
            ).fetchone()
        if row is None:
            return None
        return EscalationCase(
            case_id=row["case_id"], order_id=row["order_id"], reason=row["reason"],
            summary=row["summary"], status=row["status"], created_at=row["created_at"],
        )

    def get_conversation_memory(self, conversation_id: str, ttl_hours: int = 24) -> ShoppingContext | None:
        """读取匿名选购记忆；过期或损坏的数据会被删除。"""
        normalized_id = conversation_id.strip()
        if not normalized_id:
            return None
        now = _now()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT context_json, expires_at FROM anonymous_conversation_memory WHERE conversation_id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= now:
                connection.execute("DELETE FROM anonymous_conversation_memory WHERE conversation_id = ?", (normalized_id,))
                return None
        try:
            return ShoppingContext.model_validate(json.loads(row["context_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            self.clear_conversation_memory(normalized_id)
            return None

    def save_conversation_memory(self, conversation_id: str, context: ShoppingContext, ttl_hours: int = 24) -> None:
        """保存匿名选购摘要，不保存订单验证码或原始聊天内容。"""
        normalized_id = conversation_id.strip()
        if not normalized_id:
            return
        now = _now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO anonymous_conversation_memory
                    (conversation_id, context_json, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    context_json = excluded.context_json,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (normalized_id, _json(context.model_dump(mode="json")), now, now, _expires_at(ttl_hours)),
            )

    def clear_conversation_memory(self, conversation_id: str) -> None:
        normalized_id = conversation_id.strip()
        if not normalized_id:
            return
        with self.connect() as connection:
            connection.execute("DELETE FROM anonymous_conversation_memory WHERE conversation_id = ?", (normalized_id,))

    @staticmethod
    def _product_from_row(connection: sqlite3.Connection, row: sqlite3.Row) -> Product:
        variants = [
            ProductVariant(
                id=variant["id"], title=variant["title"], price=variant["price"],
                available_for_sale=bool(variant["available_for_sale"]),
            )
            for variant in connection.execute("SELECT * FROM product_variants WHERE product_id = ?", (row["id"],)).fetchall()
        ]
        return Product(
            id=row["id"], handle=row["handle"], title=row["title"], description=row["description"],
            product_type=row["product_type"], vendor=row["vendor"], price=row["price"],
            currency=row["currency"], available_for_sale=bool(row["available_for_sale"]),
            layout=row["layout"], switch_type=row["switch_type"], noise_level=row["noise_level"],
            connectivity=json.loads(row["connectivity_json"]), compatibility=json.loads(row["compatibility_json"]),
            use_cases=json.loads(row["use_cases_json"]), tags=json.loads(row["tags_json"]), variants=variants,
            source_url=row["source_url"], fetched_at=row["fetched_at"],
        )
