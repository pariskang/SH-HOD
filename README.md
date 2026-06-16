# SH-HOD

Shanghai-HOD benchmark workspace for constructing two municipal hospital operational dashboard evaluation datasets.

## Reproduce in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/SH-HOD/blob/main/notebooks/Shanghai_HOD_Colab_Pipeline.ipynb)

[`notebooks/Shanghai_HOD_Colab_Pipeline.ipynb`](notebooks/Shanghai_HOD_Colab_Pipeline.ipynb) is an end-to-end Colab notebook that clones the repo, installs dependencies, configures the **MiniMax API** (OpenAI-compatible), runs a connectivity self-check, generates both datasets via `generate_benchmarks.py --use-litellm`, then strictly validates, unit-tests, visualizes distributions, cross-checks evidence rows against `records.csv`, and demonstrates the scoring harness. Numeric labels stay Python-deterministic; MiniMax only performs fact-guarded surface rewriting, so artifacts are identical with or without the LLM.

## Implemented benchmark sets

- **Shanghai-HOD-Q37**: question-only natural-language stress test for module routing, intent recognition, slot extraction, clarification, safe refusal, hallucination resistance, spoken/noisy questions, and management-style open questions.
- **Shanghai-HOD-DataQA37**: data-grounded QA benchmark with structured records, deterministic Python-computed answers, evidence rows, calculations, anomaly labels, priority ranking, and grounded briefing tasks.

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
