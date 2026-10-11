# 用户问法收集与回归

`data/user_question_backlog.json` 保存待评审的问法。当前 `source: synthetic_seed` 是人工设计的回归种子，不应描述成真实用户反馈；真实问法进入项目之前必须脱敏并经过人工确认。

收集时只保留完成路由和检索评估所需的信息：

```json
{
  "id": "support-2026-001",
  "source": "support_transcript",
  "query": "脱敏后的原话",
  "intent": "POLICY_FAQ",
  "document": "shipping",
  "notes": "可选：为什么补充这个别名"
}
```

处理步骤：

1. 删除姓名、电话、地址、邮箱、订单号、验证码和任何可识别的店铺信息；订单场景使用固定的 `CC-1001` 等演示编号替代。
2. 保留顾客原本的口语表达，同时补一条稳定的意图和期望文档；无法确认时标记为 `needs_review`，不直接进入通过标准。
3. 通过评审后，把政策/FAQ 问法加入 `data/retrieval_cases.json`，把需要验证完整回答的问法加入 `data/eval_cases.json`；需要新同义词时同步更新 `app/db.py` 的本地别名并添加测试。
4. 运行 `python scripts/evaluate_demo.py`、`python scripts/evaluate_retrieval.py` 和测试集，确认新增问法没有改变既有边界行为。

每次更新记录来源、日期和脱敏状态。没有真实问法时可以继续使用 `synthetic_seed` 做回归，但不要把它当作用户研究结论。
