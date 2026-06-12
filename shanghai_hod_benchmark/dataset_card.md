# Dataset Card: Shanghai-HOD-Q37 and Shanghai-HOD-DataQA37

## Purpose

These benchmarks evaluate a Shanghai municipal hospital operational dashboard agent. Q37 tests natural-language understanding and safety behavior without exposing answers. DataQA37 tests grounded retrieval, computation, ranking, anomaly detection, priority selection, evidence binding, and management briefing generation.

## Capability axes

- Difficulty ladder (both datasets): `easy` → `medium` → `hard` → `extreme`.
- Module routing (DataQA37): each question labels `target_modules` and `module_scope` (`single_module`/`cross_module`) so module-selection and grounded answering can be scored separately.
- Context length (DataQA37): each question binds a source-data context in `contexts.jsonl` at tier `short` (≤400 est. tokens), `medium` (≤1900) or `long` (>2000), always containing the evidence rows.

## Data policy

All committed records are anonymized synthetic/hybrid examples. Hospital IDs use `SH-MH###`; no patient-level fields are present. The benchmark intentionally avoids patient identity, diagnosis, and hospital-shaming judgments.

## LLM usage

LiteLLM with Minimax/OpenAI-compatible endpoints may rewrite questions or polish briefing text. Numeric labels, rankings, calculations, anomaly flags, and evidence rows are deterministic Python outputs.

## Limitations

Synthetic data should not be interpreted as real Shanghai hospital operations. Thresholds are benchmark stress-test parameters, not policy red lines unless replaced by an approved institutional knowledge base.
