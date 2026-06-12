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

def jsonl(path):
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]

def test_q37_covers_four_difficulty_levels():
    hidden=jsonl(ROOT/'dataset_1_question_only/questions_with_hidden_metadata.jsonl')
    assert {row['difficulty'] for row in hidden}=={'easy','medium','hard','extreme'}

def test_dataqa_has_module_routing_and_scope_labels():
    questions=jsonl(ROOT/'dataset_2_data_qa/questions.jsonl')
    assert all(q.get('target_modules') and q.get('module_scope') in {'single_module','cross_module'} for q in questions)
    scopes={q['module_scope'] for q in questions}
    assert scopes=={'single_module','cross_module'}

def test_dataqa_covers_four_difficulties_and_extreme_cross_module_tasks():
    questions=jsonl(ROOT/'dataset_2_data_qa/questions.jsonl')
    assert {q['difficulty'] for q in questions}=={'easy','medium','hard','extreme'}
    task_types={q['task_type'] for q in questions}
    assert {'cross_module_joint_analysis','multi_window_cross_module_compare'} <= task_types
    extreme={q['task_type'] for q in questions if q['difficulty']=='extreme'}
    assert {'cross_module_joint_analysis','multi_window_cross_module_compare','priority_ranking','briefing'} <= extreme

def test_dataqa_context_tiers_and_long_context_token_floor():
    questions=jsonl(ROOT/'dataset_2_data_qa/questions.jsonl')
    contexts={c['context_id']: c for c in jsonl(ROOT/'dataset_2_data_qa/contexts.jsonl')}
    assert {q['context_tier'] for q in questions}=={'short','medium','long'}
    longs=[c for c in contexts.values() if c['context_tier']=='long']
    assert longs and all(c['token_estimate']>2000 for c in longs)
    assert all(c['token_estimate']<=400 for c in contexts.values() if c['context_tier']=='short')
    assert all(c['token_estimate']<=1900 for c in contexts.values() if c['context_tier']=='medium')
    for q in questions[:200]:
        content=contexts[q['context_id']]['content']
        assert all(f"\n{rid}," in content for rid in q['evidence_rows'])

def test_dataqa_answers_carry_scoring_metadata():
    answers=jsonl(ROOT/'dataset_2_data_qa/answers.jsonl')
    assert all({'target_modules','module_scope','difficulty','context_tier'} <= set(a) for a in answers)

def test_repository_does_not_commit_binary_dataset_artifacts():
    assert not (ROOT/'indicator_dictionary.xlsx').exists()
    assert not (ROOT/'dataset_2_data_qa/records.parquet').exists()
