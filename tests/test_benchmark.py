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
