# ClickyCanvas 智能电商客服 Demo 规格

版本：v0.1  
更新时间：2026-10-08  
当前阶段：MVP 设计确认

## 1. 产品定义

ClickyCanvas 是一个虚构的机械键盘商家。Demo 面向中文顾客，提供商品选择、政策咨询、模拟订单查询和售后人工转交。

第一版的目标不是做通用聊天机器人，而是验证一条可重复的电商客服闭环：

```text
顾客问题 -> 意图识别 -> 选择可信数据源 -> 生成带依据的回答或人工案件
```

## 2. 用户场景

### 2.1 售前商品问答

- “有没有适合办公室、声音比较小的键盘？”
- “这款支持 Mac 和 Windows 吗？”
- “Key A 和 Key B 有什么区别？”
- “预算 600 元，想要无线、75 配列，怎么选？”

系统需要提取约束条件，使用结构化商品字段筛选，再解释推荐理由。没有数据依据的属性必须明确标记为未知。

### 2.2 政策与使用问题

- “多久发货？”
- “可以退货吗？”
- “键盘进水了怎么办？”
- “轴体坏了是否保修？”

系统从政策和 FAQ 知识库检索，并显示来源文档或章节。

### 2.3 模拟订单与售后

- 顾客提供订单号和 4 位演示验证码，查询订单状态。
- 配送延迟时给出当前状态和下一步建议。
- 商品损坏、退款争议和政策例外创建人工转交案件。

系统不自动执行退款、改址、取消订单或库存修改。

## 3. 数据源与适配边界

### 3.1 商品目录

第一阶段从 mock.shop Storefront API 获取商品，保存本地快照。快照字段：

```text
product_id
handle
title
description
product_type
vendor
images
variants[]
  variant_id
  title
  price
  currency
  available_for_sale
tags
attributes
compatibility
source_url
fetched_at
```

`source_url` 和 `fetched_at` 用于回答溯源和调试。库存、价格等可能变化的字段，回答时应说明快照时间；以后可替换为实时查询工具。

### 3.2 商家知识库

首批 Markdown 文档：

```text
data/knowledge/shipping.md
data/knowledge/returns.md
data/knowledge/warranty.md
data/knowledge/keyboard_faq.md
data/knowledge/troubleshooting.md
```

每个检索片段需要携带：

```text
document_id
section
content
updated_at
```

### 3.3 模拟订单

订单仅供演示，建议使用 SQLite 表或等价 JSON 数据：

```text
order_id
demo_access_code
created_at
line_items[]
  product_id
  variant_id
  title
  quantity
payment_status
fulfillment_status
tracking_number
tracking_events[]
return_eligible_until
issue_type
```

首批固定案例：待处理、已发货、物流延迟、已送达且仍可退货、已过退货期、商品损坏、已发货后取消、订单不存在、验证码错误。

## 4. 意图路由

MVP 使用显式路由，不让模型直接决定业务写操作：

| 意图 | 可信来源 | 可执行动作 |
|---|---|---|
| `PRODUCT_SEARCH` | 商品快照 | 过滤、排序、推荐 |
| `PRODUCT_DETAIL` | 商品快照 | 返回规格和兼容性 |
| `PRODUCT_COMPARE` | 商品快照 | 对比共同字段 |
| `POLICY_FAQ` | Markdown 知识库 | 检索并引用政策 |
| `ORDER_STATUS` | 模拟订单工具 | 验证并查询状态 |
| `AFTER_SALES` | 政策 + 订单 | 给出流程或转人工 |
| `ESCALATE` | 对话上下文 | 创建案件并停止自动承诺 |
| `UNKNOWN` | 无 | 澄清或诚实拒答 |

## 5. 工具契约

工具只读或产生可审计的案件记录：

```text
search_products(filters, limit)
get_product(product_id)
compare_products(product_ids)
search_knowledge(query, document_types)
lookup_order(order_id, demo_access_code)
create_escalation_case(order_id, reason, summary, transcript_id)
```

所有工具返回 `source`、`retrieved_at` 或 `case_id`，方便在 UI 和日志中展示。

## 6. 回答规则

- 只使用路由选定来源中的事实。
- 商品推荐说明匹配了哪些顾客条件，以及哪些条件无法确认。
- 价格、库存和订单状态显示数据时间。
- 知识库没有答案时，说明缺少哪项资料，并提出澄清问题。
- 不承诺退款、赔偿、发货时间或库存锁定。
- 涉及严重损坏、退款争议、法律威胁或政策例外时转人工。
- 回答末尾显示来源名称，商品回答显示商品 ID 或快照时间。

## 7. 评估集

第一轮准备 40～60 条固定问题：

- 商品搜索、详情、对比：15 条。
- 政策与 FAQ：10 条。
- 订单状态和验证码：10 条。
- 售后与转人工：10 条。
- 无法回答、歧义和诱导编造：5～15 条。

验收目标：意图路由至少 90%；商品规格和订单状态至少 95%；政策回答有来源至少 90%；高风险售后案例全部转人工；未知信息不凭空补全。

## 8. 分阶段实现

1. **数据模型**：建立商品、知识文档和订单 fixture。
2. **数据拉取**：完成一次 mock.shop 拉取、规范化和快照保存。
3. **确定性工具**：先完成商品搜索、详情、对比和订单查询。
4. **路由与问答**：接入模型做意图分类和回答编排。
5. **知识检索**：先使用本地全文/关键词检索，保留后续替换 embedding 的接口。
6. **聊天界面**：显示回答、商品卡片、来源和案件编号。
7. **评估与修正**：运行固定问题集，记录错误路由、幻觉和遗漏转人工案例。

## 9. 明确暂缓

真实付款、真实退款、真实账户登录、订单写操作、实时库存承诺、多商家租户、全渠道接入、复杂向量数据库、生产部署和自动化营销均不属于本 MVP。

