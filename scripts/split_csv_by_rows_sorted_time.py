#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split each provenance CSV into multiple segments by row-count after sorting by timestamp.

Why this exists:
- Detection24 benign has very few files (e.g., 12). The original ProvDetector paper typically has many benign graphs.
- A simple (and admittedly imperfect) way to increase benign samples is to segment each benign graph into multiple
  time-ordered chunks. This matches the workflow you validated in your reproduction.

Input CSV must include a 'timestamp' column. We sort rows by timestamp (numeric when possible) and then cut into
`--parts` nearly-equal chunks, dropping chunks smaller than `--min-rows`.

Output files are named: <base>_p<idx>.csv
"""

import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--parts', type=int, default=10)
    ap.add_argument('--min-rows', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = [f for f in os.listdir(args.in_dir) if f.endswith('.csv')]
    total_out = 0

    for fn in sorted(files):
        in_path = os.path.join(args.in_dir, fn)
        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            print(f"{fn} -> read error: {e}")
            continue

        if 'timestamp' not in df.columns:
            print(f"{fn} -> no timestamp column, skip")
            continue

        if len(df) < args.min_rows:
            print(f"{fn} -> too few rows ({len(df)}), skip")
            continue

        # robust sort: try numeric
        ts = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.assign(_ts_num=ts)
        df = df.sort_values(by=['_ts_num', 'timestamp'], ascending=True, na_position='last').drop(columns=['_ts_num'])

        n = len(df)
        parts = max(1, int(args.parts))
        # compute boundaries
        idxs = [round(i * n / parts) for i in range(parts + 1)]

        wrote = 0
        base = os.path.splitext(fn)[0]
        for i in range(parts):
            a, b = idxs[i], idxs[i+1]
            seg = df.iloc[a:b].copy()
            if len(seg) < args.min_rows:
                continue
            out_fn = f"{base}_p{i+1}.csv"
            out_path = os.path.join(args.out_dir, out_fn)
            seg.to_csv(out_path, index=False)
            wrote += 1
            total_out += 1

        print(f"{fn} -> {wrote} segments")

    print(f"\nDone. Wrote {total_out} split CSV files to {args.out_dir}")


if __name__ == '__main__':
    main()
