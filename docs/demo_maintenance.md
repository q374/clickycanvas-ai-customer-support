# 演示数据维护

ClickyCanvas 只使用本地演示数据。维护数据不会连接真实店铺、订单系统或客服系统。

## 商品目录

商品字段、价格和币种通过固定格式导入，不直接修改业务代码：

```powershell
python scripts/import_catalog.py --input data/catalog.example.json --output data/catalog.json
```

修改真实演示目录时，先复制 [`data/catalog.example.json`](../data/catalog.example.json)，补齐商品和变体字段，再运行导入命令。命令失败会保留旧目录；成功后重启应用。导入格式和校验规则见 [`docs/catalog_import.md`](catalog_import.md)。

商品回答读取 `data/catalog.json` 的结构化字段。新增配列、连接方式、兼容性或使用场景时，应同时补充 `tags`、`use_cases` 或对应字段，避免只写在描述里导致筛选无法命中。价格和币种必须保持目录校验通过。

## 政策与 FAQ

配送、退货、保修、键盘 FAQ 和故障排查资料位于 `data/knowledge/*.md`。每份文档使用一个一级标题作为显示名称，正文用短段落描述可确认的规则。修改后重启应用，数据库会重新载入文档。

政策资料只应写入已经确认的演示规则。不要在文档中加入真实地址、电话、个人信息、真实订单或“AI 可以直接退款”等承诺。对于资料没有覆盖的问题，客服应继续返回不确定并建议人工确认。

## 问法回归

人工设计的回归种子保存在 `data/user_question_backlog.json`，真实问法的脱敏和评审流程见 [`docs/user_question_collection.md`](user_question_collection.md)。通过评审的政策问法加入 `data/retrieval_cases.json`，需要检查完整回答的问法加入 `data/eval_cases.json`。

维护后运行：

```powershell
python -m unittest discover -s tests -v
python scripts/evaluate_demo.py
python scripts/evaluate_retrieval.py
```
