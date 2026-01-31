#!/usr/bin/env bash
set -euo pipefail

BENIGN_DIR=${1:-data/detection24/benign_split10}
MAL_DIR=${2:-data/detection24/malicious}
OUT_BASE=${3:-results/multiseed}
SEEDS=${4:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"}

mkdir -p "$OUT_BASE"

for s in $SEEDS; do
  echo "==== seed=$s ===="
  python scripts/train_group_split_noleak_robust.py     --benign-dir "$BENIGN_DIR"     --malicious-dir "$MAL_DIR"     --detector lof     --top-k-paths 20 --vector-size 100 --epochs 50     --lof-contamination 0.04 --lof-n-neighbors 10     --split-mode group --group-regex '_p\d+$' --test-size 0.2     --split-seed "$s" --group-split-tries 200     --seed 42 --d2v-workers 1     --output-dir "$OUT_BASE/seed${s}"
done

python scripts/summarize_multiseed.py "$OUT_BASE" > "$OUT_BASE/summary.txt"
echo "[OK] Wrote $OUT_BASE/summary.txt"
