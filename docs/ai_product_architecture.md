# AI 产品能力闭环（P7.1）

更新时间：2026-10-11

## 当前实现

### 检索增强回答（Hybrid RAG 1.1）

政策和 FAQ 仍来自 `data/knowledge/*.md`。启动时每份文档会按标题和段落写入 SQLite 的 `knowledge_chunks`。`Database.retrieve_knowledge()` 先做本地别名扩展、中文双字片段和英文词匹配；配置独立的 OpenAI 兼容 embedding 后，会对候选片段计算余弦相似度，并按词法 65% + 语义 35% 融合排序，返回：

- 命中文档和片段；
- 可解释分数与匹配词；
- 原始来源路径和更新时间；
- `retrieval_method=lexical_chunk` 或 `hybrid_lexical_semantic`；语义分数单独保留。

回答只使用最高分文档的相关片段，`data.retrieval.hits` 保留本轮 Top 命中供评估复盘。商品、订单和售后事实不经过知识库检索，继续由结构化数据和本地代码决定。

未配置 `EMBEDDING_API_KEY` 时仍使用词法基线，不影响客服主流程；语义服务异常时记录告警并回退到词法结果。当前 5 份本地文档的基础词法集在 24 条政策探针上 Top-1/Top-2 均命中；30 条语义改写探针的词法基线为 Top-1 20/30、MRR 0.75。语义模式需通过 `python scripts/evaluate_retrieval.py --cases data/retrieval_semantic_cases.json --semantic --json` 单独验证，不能把配置代码当成已完成的线上 embedding 效果。

### 会话记忆

选购的 `ShoppingContext` 是匿名会话的短期记忆，保存预算、用途、必要条件、偏好和上一轮候选商品 ID。服务端按 `conversation_id` 保存摘要 24 小时，页面用本地存储复用会话 ID；每次条件覆盖或新增都会在 `shopping_memory.updates` 中记录前后值和解析来源。页面提供“清除会话记忆”按钮和 `DELETE /api/conversations/{conversation_id}` 接口。

这不是用户画像或永久长期记忆，不保存真实身份、订单验证码或原始聊天内容。记忆有 TTL，也支持按会话删除；当前仍不接入登录和跨设备同步。

### 链路追踪与评估

`/api/chat` 每轮生成 16 位追踪 ID，写入脱敏的 `conversation_traces`。追踪包含问法、意图、回答、来源、检索命中、选购上下文和模型状态，可通过 `/api/traces?conversation_id=...` 读取。演示验证码在写入前被替换为 `[验证码]`。

这让 badcase 可以按四步定位：需求理解（`shopping_memory`）、候选召回（`data.products` / `filters`）、排序（商品顺序和偏好）、回答（`answer` / `llm`）。`scripts/evaluate_retrieval.py` 额外输出 Top-1、Top-2、MRR 和匹配词。

### 边界清晰的工具型 Agent（P7.3）

每轮请求创建一个 `BoundedToolExecutor`，目前提供 `search_products`、`get_product_detail`、`compare_products`、`retrieve_policy`、`lookup_order` 和 `create_after_sale_case` 六类工具。工具仍由本地结构化数据和确定性代码执行，模型不能直接改写商品、订单或售后事实。

单轮最多执行 3 次工具调用，轨迹写入响应的 `data.agent`，包括工具名、脱敏参数、状态和是否全部完成。订单验证码在工具轨迹中也会脱敏；达到上限时返回可解释的拆分提示。高风险售后不交给模型自主决策，继续走确认和人工案件流程。

## 下一步演进

1. 收集脱敏真实问法，按查询、召回、排序、回答四层标注，避免只优化开发者设计的案例。
2. 为 Hybrid RAG 补充 30 条真实脱敏或人工复核的语义改写探针，对比词法、混合召回的 Top-k、MRR、事实引用和延迟。
3. 若要做跨设备或长期记忆，先补充登录/匿名会话边界、用户同意、查看删除和更严格的数据保留策略，再单独设计记忆存储。
4. 后续接入线上监控时只采集必要的脱敏字段，按追踪 ID 聚合失败率、人工转交率和用户改问率。

