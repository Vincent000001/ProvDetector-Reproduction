# ProvDetector-Reproduction
本项目用于复现论文《You are what you do: hunting stealthy malware via data provenance analysis》的实验ProvDetector
# ProvDetector Reproduction (Leak-Free, Group-Split)

This repository provides a **scientific, reproducible, leak-free** reproduction of the method described in:

> *You Are What You Do: Hunting Stealthy Malware via Data Provenance Analysis* (ProvDetector)

It implements the core pipeline:

- **Path-to-Sentence** conversion of provenance paths
- **Frequency Database** of path sentences (to select rarest paths)
- **Doc2Vec (PV-DM)** representation learning
- **LOF anomaly detection** trained on **benign only** (paper-style)
- **Group-aware split** to prevent leakage when benign logs are split into fragments (e.g., `*_p7.csv`)
- **Leak-free representation learning**: frequency DB + Doc2Vec are trained on **train-only benign** data

> **Data note:** original paper datasets are often not publicly available.  
> This repo is designed to reproduce the *methodology* on an alternative provenance dataset (e.g., Detection24 converted to CSV),
> while enforcing **stronger no-leakage constraints** than many naive reproductions.

Project directory structure
```
ProvDetector-Reproduction/
├── data/
│   └── detection24/
│       ├── README.md              # 数据来源与说明（非常重要）
│       ├── benign/
│       │   ├── erinyes_n_1-.csv
│       │   ├── erinyes_n_1-unknown.csv
│       │   └── ...
│       ├── malicious/
│       │   ├── erinyes_a_1.csv
│       │   └── ...
│       └── LICENSE.txt            # 如果原作者给了许可文本
│
├── scripts/
│   ├── split_csv_by_rows_sorted_time.py
│   └── ...
│
├── train_group_split_noleak_robust.py
├── feature_extractor.py
├── data_loader.py
├── requirements.txt
├── README.md                      # 主 README
└── reproduction.md                # 论文复现说明（强烈建议）
```
---

## 1) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Data preparation (not included)

Place your CSV graphs under:

- `data/detection24/benign/` (or `data/detection24/benign_split10/`)
- `data/detection24/malicious/`

See `data/README.md` for expected schema.

### Optional: split benign into more samples (time-sorted)

If you only have a few benign sessions, you may split each benign CSV into time-sorted chunks:

```bash
python scripts/split_csv_by_rows_sorted_time.py   --in-dir data/detection24/benign   --out-dir data/detection24/benign_split10   --parts 10 --min-rows 5
```

---

## 3) Run a single leak-free training/eval (recommended)

```bash
python scripts/train_group_split_noleak_robust.py   --benign-dir data/detection24/benign_split10   --malicious-dir data/detection24/malicious   --detector lof   --top-k-paths 20 --vector-size 100 --epochs 50   --lof-contamination 0.04 --lof-n-neighbors 10   --split-mode group --group-regex '_p\d+$' --test-size 0.2   --split-seed 42 --group-split-tries 200   --seed 42 --d2v-workers 1   --output-dir results/seed42
```

Outputs (under `results/seed42/`):

- `frequency_database.pkl`
- `doc2vec_model.model`
- `lof_model.joblib`
- `test_metrics.json`

---

## 4) Multi-seed robustness (recommended for reporting)

```bash
bash scripts/run_multiseed.sh   data/detection24/benign_split10   data/detection24/malicious   results/multiseed   "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
```

Summarize:

```bash
python scripts/summarize_multiseed.py results/multiseed
```

---

## 5) Reproducibility checklist

- `--split-mode group` + `--group-regex '_p\d+$'` prevents fragment leakage
- Frequency DB + Doc2Vec trained on **train benign only**
- `--d2v-workers 1` reduces nondeterminism
- `--seed` and `--split-seed` allow controlled repeats

---

## Citation

If you use this reproduction in academic work, please cite the original paper and link this repository in your artifact section.

---

## License

MIT (see `LICENSE`).
