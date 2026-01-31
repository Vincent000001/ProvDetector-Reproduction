#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split each provenance CSV into N time windows based on timestamp range.

Note: For some Detection24 logs, the timestamp distribution or missing values may cause zero valid segments.
This script is kept mainly for completeness; for Detection24 we recommend split_csv_by_rows_sorted_time.py.
"""

import argparse
import os
import pandas as pd


def split_one(path: str, out_dir: str, parts: int, min_rows: int):
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        print(f"{os.path.basename(path)} -> 0 segments (no timestamp)")
        return 0

    s = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.loc[s.notna()].copy()
    df['timestamp'] = s.loc[s.notna()].astype('int64')

    if len(df) < min_rows:
        print(f"{os.path.basename(path)} -> 0 segments (<min_rows)")
        return 0

    tmin, tmax = int(df['timestamp'].min()), int(df['timestamp'].max())
    if tmin == tmax:
        print(f"{os.path.basename(path)} -> 0 segments (tmin==tmax)")
        return 0

    edges = [tmin + (tmax - tmin) * i / parts for i in range(parts + 1)]
    count = 0
    base = os.path.splitext(os.path.basename(path))[0]
    for i in range(parts):
        lo, hi = edges[i], edges[i + 1]
        seg = df[(df['timestamp'] >= lo) & (df['timestamp'] < hi)]
        if len(seg) < min_rows:
            continue
        out_path = os.path.join(out_dir, f"{base}_p{i+1}.csv")
        seg.to_csv(out_path, index=False)
        count += 1
    print(f"{os.path.basename(path)} -> {count} segments")
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--parts', type=int, default=10)
    ap.add_argument('--min-rows', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    for fn in sorted(os.listdir(args.in_dir)):
        if not fn.endswith('.csv'):
            continue
        total += split_one(os.path.join(args.in_dir, fn), args.out_dir, args.parts, args.min_rows)

    print(f"\nDone. Wrote {total} split CSV files to {args.out_dir}")


if __name__ == '__main__':
    main()
