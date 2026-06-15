import csv, json
from pathlib import Path
import pytest

from shanghai_hod_benchmark.scripts.data_sources import ingest_csv
from shanghai_hod_benchmark.scripts.litellm_minimax import fact_guard
from shanghai_hod_benchmark.scripts.validate_artifacts import validate

ROOT=Path(__file__).resolve().parents[1]/'shanghai_hod_benchmark'

def test_committed_artifacts_are_cross_file_valid():
    report=validate(ROOT)
    assert report['q37'] >= 600
    assert report['dataqa'] >= 1000
    assert report['records'] >= 37*8*16
    # Hard-tier benchmark must include all 15 task types (8 baseline + 7 new hard).
    assert 'cross_window_extremum' in report['task_types']
    assert 'consecutive_threshold_streak' in report['task_types']
    assert 'counterfactual_threshold' in report['task_types']
    assert 'quality_filtered_aggregate' in report['task_types']
    assert 'negative_enumeration' in report['task_types']
    assert 'multi_criteria_ranking' in report['task_types']
    assert 'precision_percentage_change' in report['task_types']
    # Q37 must skew toward hard (validator threshold is 40%).
    assert report['q37_hard_share'] >= 0.40
    # Hallucination traps must be diversified across at least 12 distinct trap_type values.
    assert report['q37_trap_diversity'] >= 12
    # Briefing subtypes must include the three new sophisticated formats.
    assert 'risk_focused_targeted' in report['briefing_subtypes']
    assert 'conflicting_signals_briefing' in report['briefing_subtypes']
    assert 'exclusion_briefing' in report['briefing_subtypes']
    # Briefing factual grounding: zero ungrounded hospital citations.
    assert report['briefing_ungrounded_count'] == 0
    # Padding share must stay within the design budget.
    assert report['q37_padding_share'] <= 0.40

def test_question_public_split_has_no_hidden_labels():
    row=json.loads((ROOT/'dataset_1_question_only/questions_public.jsonl').open(encoding='utf-8').readline())
    assert set(row)=={'question'}

def test_fact_guard_rejects_factual_drift():
    source='2026-06-03 08:30，SH-MH002门急诊为182人次'
    assert fact_guard(source, '2026-06-03 08:30，SH-MH002门急诊共182人次')
    assert not fact_guard(source, '2026-06-03 08:30，SH-MH002门急诊共190人次')

def test_real_ingest_rejects_patient_columns(tmp_path):
    path=tmp_path/'bad.csv'; path.write_text('hospital_id,patient_name,timestamp_start,timestamp_end,indicator_code,value,unit\nA,张三,2026-06-03 08:00:00,2026-06-03 08:30:00,x,1,人次\n',encoding='utf-8')
    with pytest.raises(ValueError, match='patient-level'):
        ingest_csv(path)

def test_real_ingest_anonymizes_aggregate_rows(tmp_path):
    path=tmp_path/'ok.csv'; path.write_text('hospital_id,timestamp_start,timestamp_end,indicator_code,value,unit\n真实医院A,2026-06-03 08:00:00,2026-06-03 08:30:00,outpatient_emergency_visits,12,人次\n',encoding='utf-8')
    rows=ingest_csv(path)
    assert rows[0]['hospital_id'].startswith('SH-MH') and rows[0]['source_type']=='real'

def test_repository_does_not_commit_binary_dataset_artifacts():
    assert not (ROOT/'indicator_dictionary.xlsx').exists()
    assert not (ROOT/'dataset_2_data_qa/records.parquet').exists()
