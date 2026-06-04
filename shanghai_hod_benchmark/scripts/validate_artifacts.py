"""Strict cross-file validation for generated Shanghai-HOD artifacts."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(root: Path) -> dict[str, Any]:
    d1=root/'dataset_1_question_only'; d2=root/'dataset_2_data_qa'
    public=jsonl(d1/'questions_public.jsonl'); hidden=jsonl(d1/'questions_with_hidden_metadata.jsonl')
    questions=jsonl(d2/'questions.jsonl'); answers=jsonl(d2/'answers.jsonl'); maps=jsonl(d2/'evidence_map.jsonl'); anomalies=jsonl(d2/'anomaly_labels.jsonl')
    with (d2/'records.csv').open(encoding='utf-8') as f: records=list(csv.DictReader(f))
    assert all(set(row)=={'question'} for row in public), 'public Q37 must contain question only'
    assert len(public)==len(hidden) and [x['question'] for x in public]==[x['question'] for x in hidden]
    assert len({x['question_id'] for x in hidden})==len(hidden)
    qids={x['question_id'] for x in questions}; assert qids=={x['question_id'] for x in answers}=={x['question_id'] for x in maps}
    row_ids={x['row_id'] for x in records}; assert len(row_ids)==len(records)
    for row in questions+answers+maps:
        assert set(row.get('evidence_rows', [])) <= row_ids
    assert all(x['row_id'] in row_ids for x in anomalies)
    assert all(a['calculation'] and a['confidence'] in {'high','medium','low'} for a in answers)
    task_types=Counter(x['task_type'] for x in questions)
    required={'direct_lookup','cross_hospital_ranking','half_hour_mom','sustained_trend','composite_metric_explanation','anomaly_detection','priority_ranking','briefing'}
    assert required <= set(task_types), f'missing task types: {required-set(task_types)}'
    return {'q37':len(public),'dataqa':len(questions),'records':len(records),'anomalies':len(anomalies),'task_types':dict(task_types)}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=p.parse_args()
    print(json.dumps(validate(args.root),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
