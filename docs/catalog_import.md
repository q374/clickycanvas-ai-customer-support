# 商品目录导入

应用启动时会读取 `data/catalog.json`，校验后把商品目录载入本地 SQLite。可以先准备一个 JSON 或 CSV 文件，再用导入命令替换它；命令不会连接真实店铺，也不会写入订单。

## 导入命令

在项目根目录运行：

```powershell
python scripts/import_catalog.py --input data/catalog.example.json
```

默认输出是 `data/catalog.json`。建议先写到临时目标检查结果，再替换正式目录：

```powershell
python scripts/import_catalog.py `
  --input data/catalog.example.csv `
  --output data/catalog.json
```

命令会先完整读取和校验输入，然后在目标文件所在目录写临时文件并原子替换目标文件。字段缺失、价格或币种错误、混合币种、重复商品/变体 ID 等问题都会直接失败，原目标文件保持不变。导入后重启 Demo，应用会在一次数据库事务中替换商品和变体；订单和售后案件不受影响。

## JSON 格式

顶层必须是对象，包含 `products` 非空数组；可选的 `store` 只能包含 `name`、`source`、`source_url`、`fetched_at` 字符串字段。商品必填字段如下：

```text
id, handle, title, description, product_type, vendor,
price, currency, layout, switch_type, noise_level
```

`available_for_sale` 默认为 `true`。`connectivity`、`compatibility`、`use_cases`、`tags` 和 `variants` 默认为空数组。每个变体需要 `id`、`title`、`price`，`available_for_sale` 默认为 `true`。`source_url` 和 `fetched_at` 可以放在商品上，也可以由 `store` 提供默认值。

价格必须是有限的非负数字；币种必须是三位大写 ISO 风格代码，例如 `CNY` 或 `USD`。一个目录只能使用一种币种。商品 `id`、`handle` 和变体 `id`（不区分大小写）都必须唯一。可直接复制的完整 JSON 例子见 [`data/catalog.example.json`](../data/catalog.example.json)。

## CSV 格式

CSV 每行一个商品，第一行是列名。必填列与 JSON 商品字段相同；可选列为 `available_for_sale`、四个列表字段、`variants`、`source_url` 和 `fetched_at`。列表字段用分号分隔，例如 `USB-C;Bluetooth 5.1`；`variants` 使用 JSON 数组字符串，因此包含逗号时要按 CSV 规则加引号。可直接复制的例子见 [`data/catalog.example.csv`](../data/catalog.example.csv)。

CSV 导入后会转换成相同的规范化 JSON，后续应用代码只读取这一种目录格式。
