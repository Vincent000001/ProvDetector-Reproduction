# 数据说明（Detection24）

本复现实验**仅使用 Detection24** 数据（不混用 ProvNinja），原因：
- 两者采集/特征工程链条不同（ProvNinja 的 `*-fv.csv` 是“已经抽好的 50 维特征”，而本复现需要从**图边 CSV**走完整 ProvDetector 流水线）。
- 直接混用会引入不可控偏差，不满足“严谨复现论文实验”的要求。

---

## 数据在仓库中的组织

```
 data/detection24/
 ├── raw_logs/G_log/          # Detection24 原始 *.log（ProvDetector-mysql 导出的事件/边）
 ├── benign/                  # 由 raw_logs 转换出的 benign 图 CSV（边列表）
 ├── malicious/               # 由 raw_logs 转换出的 malicious 图 CSV（边列表）
 └── benign_split10/          # 将 benign/ 中每个 CSV 按时间排序后按行数切分（用于扩充 benign 样本数）
```

**文件命名约定（与你截图一致）**
- `erinyes_n_*`：benign（normal）
- `erinyes_a_*`：malicious / anomaly

---

## 原始日志（raw_logs/G_log）是什么格式？

Detection24 的 `*.log` 每一行通常是一条 Python 列表/元组形态的记录（可以用 `ast.literal_eval` 安全解析），
字段包含：`src_id, src_type, dst_id, dst_type, action, timestamp, process`（不同版本可能略有差异）。

我们在 `scripts/convert_detection24_glog_to_csv.py` 中把它转换成 ProvDetector 期望的**边列表 CSV**：

CSV 列（固定输出）：
- `sourceId`
- `sourceType`
- `destinationId`
- `destinationType`
- `action`
- `timestamp`
- `processName`

---

## 为什么要做 benign_split10？会不会“作弊”？

Detection24 原始 benign 文件数较少（你的实验里是 12 个）。而 ProvDetector 论文通常在大量 benign 上训练 LOF。
为了让 LOF 的“正常分布”更稳定，你采用了“按时间排序后切片”来扩充 benign 样本数（例如 116 个片段）。

这本质上是**数据增强**，不是论文原文的一部分。

为了避免“同一条长日志切出来的片段同时出现在训练和测试”导致的数据泄漏，本仓库默认提供：
- **Group split**：将 `*_pN.csv` 视为同一 group，只能整体进入 train 或 test
- **Leak-free 表征学习**：频次库 + Doc2Vec 仅在 train-benign 上训练

这样做可以保证：
- 切分扩充的确提升样本量
- 但不会因为泄漏而“虚高”

---

## 复现用到的数据规模（参考）

运行 `scripts/convert_detection24_glog_to_csv.py` 后，你应看到类似数量级：
- benign CSV：≈ 12
- malicious CSV：≈ 156

运行 `scripts/split_csv_by_rows_sorted_time.py --parts 10` 后：
- benign_split10：≈ 100+（取决于每个 benign 文件行数与 `--min-rows`）

---

## 如果你想从 detection24-main 重新导入数据

Detection24-main 中数据通常位于：

`detection24-main/ProvDetector-mysql/G_log/`

你需要做两步：

1) 复制原始日志到本仓库：
```bash
mkdir -p data/detection24/raw_logs
cp -r /path/to/detection24-main/ProvDetector-mysql/G_log data/detection24/raw_logs/
```

2) 转换为 CSV + 切分 benign：
```bash
python scripts/convert_detection24_glog_to_csv.py \
  --benign-log-dir data/detection24/raw_logs/G_log \
  --malicious-log-dir data/detection24/raw_logs/G_log \
  --benign-out data/detection24/benign \
  --malicious-out data/detection24/malicious

python scripts/split_csv_by_rows_sorted_time.py \
  --in-dir data/detection24/benign \
  --out-dir data/detection24/benign_split10 \
  --parts 10 --min-rows 5
```

---

## 你不应该混用 ProvNinja 的 `sample-enterprise-data`

ProvNinja 的 `intrusion-detection-system/path-based/sample-enterprise-data` 里有两类文件：
- `*-fv.csv`：已经抽好的 50 维特征（适合你写的 `train_lof_from_fv.py`）
- `*-paragraph.csv`：文本段落特征/描述

它们**不等价于 ProvDetector 论文中用的原始图边**，也没有“同一套 Path→Sentence→Doc2Vec”链条。
所以严格复现时，不建议混用。

