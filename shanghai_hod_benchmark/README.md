# Shanghai-HOD Benchmark

This directory contains reproducible generators and committed artifacts for two Shanghai municipal hospital operational dashboard benchmarks.

## Datasets

1. **Shanghai-HOD-Q37**: question-only natural-language interaction stress test for module routing, slot extraction, clarification, safe refusal, hallucination resistance, spoken/noisy questions, and management-style open questions. Questions span four difficulty levels: easy, medium, hard, extreme.
2. **Shanghai-HOD-DataQA37**: data-grounded QA benchmark with structured synthetic/hybrid data, programmatically computed answers, evidence rows, calculations, anomaly labels, priority ranking, and grounded briefing tasks.

## DataQA37 capability axes

Every DataQA question carries labels for three orthogonal capability axes:

- **Module selection**: `target_modules` (ground-truth dashboard modules, e.g. `M02`/`M07`) and `module_scope` (`single_module` vs `cross_module`). Predictions may report `selected_modules` so the scorer can measure routing before answering.
- **Difficulty**: `difficulty` in `easy` (direct lookup), `medium` (ranking / half-hour MoM), `hard` (sustained trend, composite explanation, anomaly detection), `extreme` (cross-module joint analysis, multi-window cross-module comparison, priority ranking, grounded briefing).
- **Context length**: `context_id`/`context_tier` reference `contexts.jsonl`, which holds the CSV-style source rows the model must answer from. Tiers: `short` (≤400 estimated tokens), `medium` (≤1900), `long` (strictly >2000). Every context is guaranteed to contain all evidence rows of its question.

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

