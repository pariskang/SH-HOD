# Optional Parquet export

Binary files are intentionally not committed because the target pull-request system does not support binary diffs.
Generate a local Parquet copy from the canonical `records.csv` with:

```bash
python shanghai_hod_benchmark/scripts/generate_benchmarks.py \
  --profile standard \
  --export-parquet /tmp/shanghai-hod-records.parquet
```
