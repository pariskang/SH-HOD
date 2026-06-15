# Shanghai-HOD Benchmark

This directory contains reproducible generators and committed artifacts for two Shanghai municipal hospital operational dashboard benchmarks, **tuned to challenge top-tier LLMs (Claude / GPT / Gemini)**. The committed split keeps a hard-share of ≥40% across both datasets.

## Datasets

1. **Shanghai-HOD-Q37**: question-only natural-language stress test covering 11 question types:
   - `single_module`, `multi_module`, `cross_module_chain` (3+ modules), `temporal_compound` (cross-window conditions), `conflicting_signals` (contradictory indicator pairs), `boundary_precision` (threshold equality), `negative_enumeration` (no-violation cohorts), `management_open`, `ambiguous_boundary`, `hallucination_trap` (**18 distinct trap types**, no time-window repetition), and `spoken_noisy`.
2. **Shanghai-HOD-DataQA37**: data-grounded QA benchmark across 15 task types:
   - Baseline 8: `direct_lookup`, `cross_hospital_ranking`, `half_hour_mom`, `sustained_trend`, `composite_metric_explanation`, `anomaly_detection`, `priority_ranking`, `briefing`.
   - **7 new hard task types**: `cross_window_extremum`, `consecutive_threshold_streak`, `counterfactual_threshold`, `quality_filtered_aggregate`, `negative_enumeration`, `multi_criteria_ranking`, `precision_percentage_change`.
   - **5 briefing subtypes**: `standard_executive`, `risk_focused_targeted`, `conflicting_signals_briefing`, `exclusion_briefing`, `leadership_focus`.

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

