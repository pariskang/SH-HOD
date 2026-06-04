# Shanghai-HOD-Q37 generation prompts

LiteLLM/Minimax is optional. It may diversify question wording, but hidden metadata is deterministic and schema-validated.

## Environment

```bash
export MINIMAX_API_KEY=...
export MINIMAX_API_BASE=https://your-openai-compatible-relay/v1
export MINIMAX_MODEL=MiniMax-M1
python shanghai_hod_benchmark/scripts/generate_benchmarks.py --profile standard --use-litellm
```

## System instruction

You are a Shanghai municipal hospital operational dashboard benchmark generator. Rewrite only the natural-language surface form. Keep target module, indicators, time windows, safety labels, and expected behavior unchanged. Do not invent values, policy files, hospitals, patients, diagnoses, causes, or remediation status.
