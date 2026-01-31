#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Detection24 ProvDetector-mysql G_log (*.log) into ProvDetector CSV graphs.

The Detection24 repository (ProvDetector-mysql/G_log) stores provenance edges as Python literals per line.
Each line is a list/tuple like:
  ['sourceId','sourceType','destinationId','destinationType','action','timestamp','processName?']

This script:
- selects benign logs:   erinyes_n_*.log
- selects malicious logs: erinyes_a_*.log
- parses each line with ast.literal_eval (safe) and writes a CSV per log.

Output CSV schema (header):
  sourceId,sourceType,destinationId,destinationType,action,timestamp,processName

Notes:
- Some logs may omit processName; we fill with ''
- Some fields may be missing/extra; we best-effort normalize to 7 columns.
"""

import argparse
import ast
import csv
import os
import re
from glob import glob

HEADER = [
    "sourceId",
    "sourceType",
    "destinationId",
    "destinationType",
    "action",
    "timestamp",
    "processName",
]


def _collect_logs(log_dir: str, pattern: str):
    p = os.path.join(os.path.expanduser(log_dir), pattern)
    return sorted(glob(p))


def _normalize_row(obj):
    """Return a 7-field row matching HEADER."""
    if obj is None:
        return None

    # Expect list/tuple
    if isinstance(obj, (list, tuple)):
        row = list(obj)
    else:
        return None

    # Some lines may contain nested objects; cast to str
    row = ["" if x is None else str(x) for x in row]

    # Ensure length
    if len(row) < 6:
        return None

    if len(row) == 6:
        row.append("")
    elif len(row) > 7:
        # Keep first 6 as defined and join the tail as processName-ish
        row = row[:6] + ["|".join(row[6:])]

    # Fix common typos
    if row[2] == "unknwon":
        row[2] = "unknown"

    return row


def _convert_one(log_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(log_path)
    out_csv = os.path.join(out_dir, os.path.splitext(base)[0] + ".csv")

    n_rows = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f_in, open(
        out_csv, "w", newline="", encoding="utf-8"
    ) as f_out:
        w = csv.writer(f_out)
        w.writerow(HEADER)

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
            except Exception:
                # skip malformed lines
                continue
            row = _normalize_row(obj)
            if row is None:
                continue
            w.writerow(row)
            n_rows += 1

    return out_csv, n_rows


def _count_edges(csv_path: str):
    # quick edge count excluding header
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign-log-dir", required=True)
    ap.add_argument("--malicious-log-dir", required=True)
    ap.add_argument("--benign-out", required=True)
    ap.add_argument("--malicious-out", required=True)
    ap.add_argument("--benign-pattern", default="erinyes_n_*.log")
    ap.add_argument("--malicious-pattern", default="erinyes_a_*.log")
    args = ap.parse_args()

    benign_logs = _collect_logs(args.benign_log_dir, args.benign_pattern)
    malicious_logs = _collect_logs(args.malicious_log_dir, args.malicious_pattern)

    print(f"[+] Input: {args.benign_log_dir}  ({len(benign_logs)} log files)")
    benign_csvs = []
    benign_edges = 0
    for p in benign_logs:
        out_csv, _ = _convert_one(p, args.benign_out)
        benign_csvs.append(out_csv)
        benign_edges += _count_edges(out_csv)

    print(f"[+] Output: {args.benign_out}  ({len(benign_csvs)} csv files, {benign_edges} edges)")

    print(f"[+] Input: {args.malicious_log_dir}  ({len(malicious_logs)} log files)")
    malicious_csvs = []
    malicious_edges = 0
    for p in malicious_logs:
        out_csv, _ = _convert_one(p, args.malicious_out)
        malicious_csvs.append(out_csv)
        malicious_edges += _count_edges(out_csv)

    print(f"[+] Output: {args.malicious_out}  ({len(malicious_csvs)} csv files, {malicious_edges} edges)")


if __name__ == "__main__":
    main()
