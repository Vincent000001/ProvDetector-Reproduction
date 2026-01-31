# ProvDetector 论文复现实验（Detection24 数据集）

> 目标：**完整、严谨、可复现**地复现 ProvDetector 论文中“Path→Sentence → 频次库 → Rarest Paths → Doc2Vec(PV-DM) → LOF”这一条核心流水线，
> 并给出**可重复跑出的指标**与**避免数据泄漏**的实现细节。

本仓库已经内置 **Detection24** 的真实数据（你已获得原作者许可），做到**开箱即跑**。

---

## 1. 你现在算“复现成功”了吗？

以“能否复现论文核心方法 + 指标量级是否一致 + 实验设置是否严谨（尤其是避免泄漏）”为标准，你当前结果满足：

- ✅ **方法链条复现**：Path 提取 → 句子化 → 频次库 → 罕见路径选择 → Doc2Vec → LOF。
- ✅ **数据泄漏控制**：使用 `train_group_split_noleak_robust.py`，**先切分，再仅用训练集（且仅训练 benign）来构建频次库与训练 Doc2Vec**。
- ✅ **评估稳定**：你汇总的 20 个 seed，F1 平均约 **0.9648 ± 0.0125**，与论文报告的 ~0.974 同一量级。

注意两点“科研口径”的严谨表述：

1) 你现在复现的是**论文方法在 Detection24 数据上的可重复结果**，并非“论文原始数据集上的逐点复现”。
2) 你的指标非常高（甚至有 seed 接近 1.0），这可能来自 **Detection24 任务本身可分性较强**；因此你在 README 里应该明确：
   - 你做了 leak-free；
   - 你用多 seed 报告均值±方差；
   - 你给出复现脚本与固定随机性。

---

## 2. 数据集来自哪里？怎么判定 benign / malicious？

Detection24 仓库中，我们使用：

- `detection24-main/ProvDetector-mysql/G_log/` 下的 `*.log`

文件名规则（与你截图一致）：

- `erinyes_n_*.log`：**benign（normal）**
- `erinyes_a_*.log`：**malicious / anomaly（attack）**

本仓库把这些日志复制到了：

- `data/detection24/raw_logs/G_log/`

并额外提供了：

- `data/detection24/benign/*.csv`（转换后的 benign 图）
- `data/detection24/malicious/*.csv`（转换后的 malicious 图）
- `data/detection24/benign_split10/*.csv`（benign 切分扩充后的样本）

> 为什么需要 benign 切分扩充？
> Detection24 的 benign 原始文件数较少（你最初只有 12 个 benign CSV），
> 对 LOF 这种“仅在 benign 上拟合”的无监督方法不友好。
> 我们采用“按 timestamp 排序后按行切分”为一种工程化扩充手段。

---

## 3. 环境与依赖

推荐使用 Conda：

```bash
conda create -n provdetector python=3.11 -y
conda activate provdetector
pip install -r requirements.txt
```

关键依赖：
- `scikit-learn`（LOF + 评估）
- `gensim`（Doc2Vec）

可选：
- `imbalanced-learn`（SMOTE，默认不需要；LOF 按论文方法只用 benign 拟合）

---

## 4. 一键复现（推荐：Leak-Free + Group Split）

### 4.1 单次运行

```bash
python train_group_split_noleak_robust.py \
  --benign-dir data/detection24/benign_split10 \
  --malicious-dir data/detection24/malicious \
  --detector lof \
  --top-k-paths 20 --vector-size 100 --epochs 50 \
  --lof-contamination 0.04 --lof-n-neighbors 10 \
  --split-mode group \
  --group-regex '_p\\d+$' \
  --test-size 0.2 \
  --split-seed 42 \
  --group-split-tries 200 \
  --seed 42 \
  --d2v-workers 1 \
  --output-dir outputs/run_seed42
```

运行结束会生成：

- `outputs/run_seed42/test_metrics.json`
- `outputs/run_seed42/frequency_database.pkl`
- `outputs/run_seed42/doc2vec_model.model`
- `outputs/run_seed42/lof_model.joblib`

### 4.2 多 seed 稳健性复现

```bash
bash scripts/run_multiseed.sh \
  data/detection24/benign_split10 \
  data/detection24/malicious \
  outputs/multiseed

python scripts/summarize_metrics.py outputs/multiseed/models_detection24_group_robust_seed*/test_metrics.json
```

建议在论文/README 中报告：
- F1 mean ± std
- min / max
- 以及每个 seed 的 confusion matrix（或至少 TN/FP/FN/TP）

---

## 5. 如果你想从“原始 Detection24 logs”重新生成 CSV

仓库已内置转换结果；如果你想验证转换过程或更新数据，执行：

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

## 6. 复现严谨性清单（建议你在论文/仓库里写清楚）

- [x] Train/Test 切分：**Group split**，避免 `_pN.csv` 切分片段跨集合泄漏
- [x] 表征学习：频次库 + Doc2Vec **仅用训练集 benign**（对应论文“学习正常行为”）
- [x] 模型拟合：LOF **仅在训练集 benign 上 fit**
- [x] 随机性：提供 `--seed`、固定 `d2v-workers=1`（尽量减少非确定性）
- [x] 稳健性：多 seed 报告均值±方差

---

## 7. 引用与许可

- 代码：MIT（见 `LICENSE`）
- 数据：Detection24 原作者已允许你在本仓库中使用/分发；请在你公开 GitHub 前，补充：
  - Detection24 的引用信息（论文/仓库链接）
  - 数据许可说明（哪条邮件/issue/statement授权，最好截图或引用说明文本）

---

## 8. 目录结构

```
.
├── data/
│   └── detection24/
│       ├── raw_logs/G_log/          # 原始 *.log
│       ├── benign/                  # 转换后的 benign CSV
│       ├── malicious/               # 转换后的 malicious CSV
│       └── benign_split10/          # benign 切分扩充样本
├── scripts/
│   ├── convert_detection24_glog_to_csv.py
│   ├── split_csv_by_rows_sorted_time.py
│   ├── run_multiseed.sh
│   └── summarize_metrics.py
├── train_group_split_noleak_robust.py
└── ...
```

更多数据细节见：`data/README.md`。
