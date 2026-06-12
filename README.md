# SH-HOD

Shanghai-HOD benchmark workspace for constructing two municipal hospital operational dashboard evaluation datasets.

## Implemented benchmark sets

- **Shanghai-HOD-Q37** (question-only): natural-language stress test for module routing, intent recognition, slot extraction, clarification, safe refusal, hallucination resistance, spoken/noisy questions, and management-style open questions.
- **Shanghai-HOD-DataQA37** (question + verified answer): data-grounded QA benchmark with structured records, deterministic Python-computed answers, evidence rows, calculations, anomaly labels, priority ranking, and grounded briefing tasks.

## Capability axes (what the benchmark measures)

Both datasets share a four-level difficulty ladder: `easy` → `medium` → `hard` → `extreme`.

Every DataQA37 question additionally carries labels on three orthogonal axes so a model/agent can be scored separately on each capability:

| Axis | Field(s) | Values |
|---|---|---|
| Module selection | `target_modules`, `module_scope` | dashboard module codes (M01–M14); `single_module` / `cross_module` |
| Difficulty | `difficulty` | `easy` (direct lookup) / `medium` (ranking, half-hour MoM) / `hard` (sustained trend, composite explanation, anomaly detection) / `extreme` (cross-module joint analysis, multi-window cross-module comparison, priority ranking, grounded briefing) |
| Context length | `context_id`, `context_tier` → `contexts.jsonl` | `short` (≤400 est. tokens) / `medium` (≤1900) / `long` (strictly >2000 tokens of source rows) |

`dataset_2_data_qa/contexts.jsonl` holds the CSV-style source-data blocks the model must answer from; every context is guaranteed to contain all evidence rows of its question, and `long` contexts always exceed 2000 estimated tokens. The scorer (`evaluation/scorer.py`) reports `module_selection_accuracy`, `cross_module_selection_recall`, and accuracy breakdowns by difficulty, context tier, and module scope — predictions may include a `selected_modules` field to be scored on routing before answering.

## Generate datasets

Representative committed profile:

```bash
python shanghai_hod_benchmark/scripts/generate_benchmarks.py --profile standard
```

Production-scale local materialization for 37 hospitals, 7 days, 48 half-hour windows per day, and all configured indicators:

```bash
python shanghai_hod_benchmark/scripts/generate_benchmarks.py --profile full --q37-count 1000 --dataqa-questions 3000
```

## LiteLLM / Minimax configuration

LiteLLM is optional and is used only for safe question rewriting or briefing-language polishing. Numeric answers are always computed from `records.csv` by Python.

```bash
export MINIMAX_API_KEY=...
export MINIMAX_API_BASE=https://your-openai-compatible-relay/v1
export MINIMAX_MODEL=MiniMax-M1
python shanghai_hod_benchmark/scripts/generate_benchmarks.py --profile standard --use-litellm
```

## Validate committed artifacts

```bash
python shanghai_hod_benchmark/scripts/validate_artifacts.py
pytest -q
```

For approved aggregate real-data input, use `--records-input`. Patient-level columns are rejected and hospital identifiers are anonymized by default. See `shanghai_hod_benchmark/README.md` for the full workflow.

## Binary artifact policy

This repository commits reviewable text artifacts only because the PR system does not support binary diffs. Generate Parquet locally when needed:

```bash
python shanghai_hod_benchmark/scripts/generate_benchmarks.py \
  --profile standard --export-parquet /tmp/shanghai-hod-records.parquet
```
