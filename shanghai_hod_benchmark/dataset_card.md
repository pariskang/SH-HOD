# Dataset Card: Shanghai-HOD-Q37 and Shanghai-HOD-DataQA37

## Purpose

These benchmarks evaluate a Shanghai municipal hospital operational dashboard agent. Q37 tests natural-language understanding and safety behavior without exposing answers. DataQA37 tests grounded retrieval, computation, ranking, anomaly detection, priority selection, evidence binding, and management briefing generation.

## Data policy

All committed records are anonymized synthetic/hybrid examples. Hospital IDs use `SH-MH###`; no patient-level fields are present. The benchmark intentionally avoids patient identity, diagnosis, and hospital-shaming judgments.

## LLM usage

LiteLLM with Minimax/OpenAI-compatible endpoints may rewrite questions or polish briefing text. Numeric labels, rankings, calculations, anomaly flags, and evidence rows are deterministic Python outputs.

## Limitations

Synthetic data should not be interpreted as real Shanghai hospital operations. Thresholds are benchmark stress-test parameters, not policy red lines unless replaced by an approved institutional knowledge base.
