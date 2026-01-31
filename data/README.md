# Data (not included)

This repository does **not** ship the original datasets (size/licensing).
You should place your CSV graphs under:

- `data/detection24/benign/` or `data/detection24/benign_split10/`
- `data/detection24/malicious/`

## Expected CSV schema

Each CSV file should contain edge records with (at minimum) the following columns:

- `sourceId, sourceType, destinationId, destinationType, action`
- optional but recommended: `timestamp`, `processName`

Your conversion scripts (e.g., `convert_detection24_glog_to_csv.py`) should produce this schema.

## Splitting benign into more samples

To reproduce the paper-like setup when benign sessions are few, you may split
each benign CSV into multiple time-sorted segments:

```bash
python scripts/split_csv_by_rows_sorted_time.py   --in-dir data/detection24/benign   --out-dir data/detection24/benign_split10   --parts 10 --min-rows 5
```

This increases the number of benign samples without changing the underlying event order.
