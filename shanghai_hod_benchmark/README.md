# Shanghai-HOD Benchmark

This directory contains reproducible generators and committed artifacts for two Shanghai municipal hospital operational dashboard benchmarks.

## Datasets

1. **Shanghai-HOD-Q37**: question-only natural-language interaction stress test for module routing, slot extraction, clarification, safe refusal, hallucination resistance, spoken/noisy questions, and management-style open questions.
2. **Shanghai-HOD-DataQA37**: data-grounded QA benchmark with structured synthetic/hybrid data, programmatically computed answers, evidence rows, calculations, anomaly labels, priority ranking, and grounded briefing tasks.

## Profiles

| Profile | Hospitals | Days | Half-hour windows | Use |
|---|---:|---:|---:|---|
| `mini` | 3 | 1 | 4 | Fast smoke tests |
| `standard` | 37 | 1 | 8 | Committed representative benchmark |
| `full` | 37 | 7 | 48/day | Production-scale local generation (~200k records) |

## Generate artifacts

```bash
python shanghai_hod_benchmark/scripts/generate_benchmarks.py --profile standard
```

Full-scale generation:

```bash
python shanghai_hod_benchmark/scripts/generate_benchmarks.py --profile full --q37-count 1000 --dataqa-questions 3000
```

## LiteLLM / Minimax through OpenAI-compatible relay

```bash
export MINIMAX_API_KEY=...
export MINIMAX_API_BASE=https://your-openai-compatible-relay/v1
export MINIMAX_MODEL=MiniMax-M1
python shanghai_hod_benchmark/scripts/generate_benchmarks.py --profile standard --use-litellm
```

LiteLLM is optional and only rewrites questions or polishes briefing prose. Numeric values, rankings, calculations, anomaly flags, and evidence rows are always computed by Python from `records.csv`.

## Current committed artifacts

The committed artifacts were generated with profile `standard`, 600 Q37 questions, and 1000 DataQA questions.

## Approved real-data and hybrid workflow

Pass an approved aggregate CSV with `--records-input`. The loader rejects patient-level columns and anonymizes hospital IDs by default. Use `--no-anonymize-input` only for already approved anonymous IDs.

```bash
python shanghai_hod_benchmark/scripts/generate_benchmarks.py --profile standard --records-input /secure/approved_aggregate_records.csv
```

## Validation and evaluation

```bash
python shanghai_hod_benchmark/scripts/validate_artifacts.py
pytest -q
```

The strict validator checks public/hidden Q37 separation, cross-file IDs, evidence integrity, answer contracts, and required task coverage. LiteLLM calls use caching, retries, JSON validation, and a factual-drift guard.


## Binary artifact policy

The canonical committed files are reviewable CSV/JSONL/Markdown. Binary `.parquet` and `.xlsx` outputs are intentionally excluded because the pull-request system does not support binary diffs. To export Parquet locally:

```bash
python shanghai_hod_benchmark/scripts/generate_benchmarks.py \
  --profile standard --export-parquet /tmp/shanghai-hod-records.parquet
```
