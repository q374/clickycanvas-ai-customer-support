from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProductVariant(BaseModel):
    id: str
    title: str
    price: float
    available_for_sale: bool = True


class Product(BaseModel):
    id: str
    handle: str
    title: str
    description: str
    product_type: str
    vendor: str
    price: float
    currency: str = "CNY"
    available_for_sale: bool = True
    layout: str
    switch_type: str
    noise_level: str
    connectivity: list[str] = Field(default_factory=list)
    compatibility: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    variants: list[ProductVariant] = Field(default_factory=list)
    source_url: str | None = None
    fetched_at: str | None = None


class TrackingEvent(BaseModel):
    occurred_at: str
    status: str
    location: str | None = None


class OrderItem(BaseModel):
    product_id: str
    variant_id: str
    title: str
    quantity: int = 1


class Order(BaseModel):
    order_id: str
    demo_access_code: str
    created_at: str
    payment_status: str
    fulfillment_status: str
    tracking_number: str | None = None
    return_eligible_until: str | None = None
    issue_type: str | None = None
    line_items: list[OrderItem] = Field(default_factory=list)
    tracking_events: list[TrackingEvent] = Field(default_factory=list)


class KnowledgeDocument(BaseModel):
    document_id: str
    title: str
    section: str
    content: str
    source_path: str
    updated_at: str


class KnowledgeHit(BaseModel):
    """可解释的检索命中，保留证据片段与匹配信息供回答和评估复盘。"""

    document_id: str
    title: str
    section: str
    content: str
    excerpt: str
    source_path: str
    updated_at: str
    score: float
    semantic_score: float | None = None
    matched_terms: list[str] = Field(default_factory=list)
    retrieval_method: str = "lexical_chunk"


class EscalationCase(BaseModel):
    case_id: str
    order_id: str | None = None
    reason: str
    summary: str
    status: Literal["open", "closed"] = "open"
    created_at: str


class ProductSearchFilters(BaseModel):
    query: str = ""
    full_keyboards_only: bool = False
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = None
    use_case: str | None = None
    layout: str | None = None
    switch_type: str | None = None
    excluded_switch_types: list[str] = Field(default_factory=list)
    connectivity: str | None = None
    excluded_connectivity: list[str] = Field(default_factory=list)
    compatibility: str | None = None
    noise_level: str | None = None
    required_tags: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)


class ProductCompareRequest(BaseModel):
    product_ids: list[str] = Field(min_length=2, max_length=4)


class OrderLookupRequest(BaseModel):
    order_id: str
    demo_access_code: str


class ShoppingContext(BaseModel):
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    budget_unlimited: bool = False
    use_case: str | None = Field(default=None, max_length=30)
    layout: str | None = Field(default=None, max_length=20)
    switch_type: str | None = Field(default=None, max_length=30)
    excluded_switch_types: list[str] = Field(default_factory=list, max_length=5)
    connectivity: str | None = Field(default=None, max_length=30)
    excluded_connectivity: list[str] = Field(default_factory=list, max_length=3)
    compatibility: str | None = Field(default=None, max_length=30)
    noise_level: str | None = Field(default=None, max_length=10)
    required_tags: list[str] = Field(default_factory=list, max_length=5)
    preferences: list[str] = Field(default_factory=list, max_length=5)
    last_product_ids: list[str] = Field(default_factory=list, max_length=5)
    recommendation_pending: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    order_id: str | None = None
    demo_access_code: str | None = None
    shopping_context: ShoppingContext | None = None
    conversation_id: str | None = Field(default=None, max_length=80)


class ChatResponse(BaseModel):
    intent: str
    answer: str
    data: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    escalation_case: EscalationCase | None = None
