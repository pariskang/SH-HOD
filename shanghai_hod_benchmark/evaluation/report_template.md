# Shanghai-HOD Evaluation Report

## Run metadata

- Model:
- Date:
- Dataset version:
- Difficulty config: hard-tier (target hard share ≥ 40%, 18 distinct hallucination traps, 15 DataQA task types)

## Shanghai-HOD-Q37 — primary metrics

| Metric | Score | Notes |
|---|---:|---|
| Contract validity rate | | |
| Module routing accuracy | | |
| Hallucination resistance rate (18 trap types) | | |
| Cross-module chain accuracy (3+ modules) | | |
| Temporal compound accuracy (cross-window) | | |

## Shanghai-HOD-Q37 — secondary metrics

| Metric | Score | Notes |
|---|---:|---|
| Query type accuracy | | |
| Slot F1 | | |
| Multi-module recall | | |
| Clarification accuracy | | |
| Safe refusal accuracy | | |
| Noisy query robustness | | |
| Conflicting signals calibration | | |
| Boundary precision accuracy (threshold equality) | | |
| Negative enumeration accuracy | | |

## Shanghai-HOD-Q37 — penalty triggers (count incidents)

| Trigger | Count | Notes |
|---|---:|---|
| Incorrect refusal when answerable | | |
| Over-clarification on actionable question | | |
| Accepted authority-persona injection | | |
| Accepted fabricated threshold change | | |
| Confirmed partial truth without lookup | | |

## Shanghai-HOD-DataQA37 — primary metrics

| Metric | Score | Notes |
|---|---:|---|
| Exact / numeric accuracy | | tol: 0.0005 (ratios) / 0 (counts) / 0.1% (currency) |
| Evidence accuracy (set + Jaccard) | | |
| Cross-window extremum accuracy | | |
| Consecutive threshold streak accuracy | | |
| Counterfactual threshold accuracy | | |
| Negative enumeration F1 | | |
| Multi-criteria ranking NDCG@5 | | |
| Precision percentage change (2 d.p.) accuracy | | |

## Shanghai-HOD-DataQA37 — secondary metrics

| Metric | Score | Notes |
|---|---:|---|
| NDCG@5 | | |
| Top-K accuracy | | |
| Anomaly precision / recall / F1 | | |
| Spearman priority correlation | | |
| Briefing factual consistency | | |
| Briefing hallucination rate | | |
| Quality-filtered aggregate accuracy | | |
| Briefing subtype targeting precision | | risk_focused / conflicting / exclusion / leadership |

## Per-task-type breakdown (DataQA37)

| Task type | n | Avg score | Hard-task model gap vs baseline |
|---|---:|---:|---:|
| direct_lookup | | | |
| cross_hospital_ranking | | | |
| half_hour_mom | | | |
| sustained_trend | | | |
| composite_metric_explanation | | | |
| anomaly_detection | | | |
| priority_ranking | | | |
| briefing | | | |
| cross_window_extremum | | | |
| consecutive_threshold_streak | | | |
| counterfactual_threshold | | | |
| quality_filtered_aggregate | | | |
| negative_enumeration | | | |
| multi_criteria_ranking | | | |
| precision_percentage_change | | | |
