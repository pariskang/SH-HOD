"""Strict cross-file validation for generated Shanghai-HOD artifacts."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


DIFFICULTY_LEVELS={'easy','medium','hard','extreme'}
CONTEXT_TIERS={'short','medium','long'}
LONG_CONTEXT_MIN_TOKENS=2000


def validate(root: Path) -> dict[str, Any]:
    d1=root/'dataset_1_question_only'; d2=root/'dataset_2_data_qa'
    public=jsonl(d1/'questions_public.jsonl'); hidden=jsonl(d1/'questions_with_hidden_metadata.jsonl')
    questions=jsonl(d2/'questions.jsonl'); answers=jsonl(d2/'answers.jsonl'); maps=jsonl(d2/'evidence_map.jsonl'); anomalies=jsonl(d2/'anomaly_labels.jsonl')
    contexts=jsonl(d2/'contexts.jsonl')
    with (d2/'records.csv').open(encoding='utf-8') as f: records=list(csv.DictReader(f))
    assert all(set(row)=={'question'} for row in public), 'public Q37 must contain question only'
    assert len(public)==len(hidden) and [x['question'] for x in public]==[x['question'] for x in hidden]
    assert len({x['question_id'] for x in hidden})==len(hidden)
    assert {x['difficulty'] for x in hidden}==DIFFICULTY_LEVELS, 'Q37 must cover the 4 difficulty levels'
    qids={x['question_id'] for x in questions}; assert qids=={x['question_id'] for x in answers}=={x['question_id'] for x in maps}
    row_ids={x['row_id'] for x in records}; assert len(row_ids)==len(records)
    for row in questions+answers+maps:
        assert set(row.get('evidence_rows', [])) <= row_ids
    assert all(x['row_id'] in row_ids for x in anomalies)
    assert all(a['calculation'] and a['confidence'] in {'high','medium','low'} for a in answers)
    task_types=Counter(x['task_type'] for x in questions)
    required={'direct_lookup','cross_hospital_ranking','half_hour_mom','sustained_trend','composite_metric_explanation','anomaly_detection','cross_module_joint_analysis','multi_window_cross_module_compare','priority_ranking','briefing'}
    assert required <= set(task_types), f'missing task types: {required-set(task_types)}'
    # module routing, difficulty and context-tier contracts on DataQA
    ctx_by_id={x['context_id']: x for x in contexts}
    for q in questions:
        assert q.get('target_modules'), f"{q['question_id']} missing target_modules"
        assert q.get('module_scope') in {'single_module','cross_module'}
        assert q.get('difficulty') in DIFFICULTY_LEVELS
        assert q.get('context_tier') in CONTEXT_TIERS
        ctx=ctx_by_id[q['context_id']]
        assert all(f"\n{rid}," in ctx['content'] for rid in q['evidence_rows']), f"{q['question_id']} context missing evidence rows"
    assert {q['difficulty'] for q in questions}==DIFFICULTY_LEVELS, 'DataQA must cover the 4 difficulty levels'
    assert {q['context_tier'] for q in questions}==CONTEXT_TIERS, 'DataQA must cover the 3 context tiers'
    assert {q['module_scope'] for q in questions}=={'single_module','cross_module'}
    assert all(c['token_estimate']>LONG_CONTEXT_MIN_TOKENS for c in contexts if c['context_tier']=='long'), 'long contexts must exceed 2000 estimated tokens'
    assert all(c['token_estimate']<=400 for c in contexts if c['context_tier']=='short'), 'short contexts must stay within 400 estimated tokens'
    assert all(c['token_estimate']<=1900 for c in contexts if c['context_tier']=='medium'), 'medium contexts must stay within 1900 estimated tokens'
    return {'q37':len(public),'dataqa':len(questions),'records':len(records),'anomalies':len(anomalies),'contexts':len(contexts),
            'task_types':dict(task_types),
            'q37_difficulty':dict(Counter(x['difficulty'] for x in hidden)),
            'dataqa_difficulty':dict(Counter(x['difficulty'] for x in questions)),
            'dataqa_module_scope':dict(Counter(x['module_scope'] for x in questions)),
            'dataqa_context_tier':dict(Counter(x['context_tier'] for x in questions))}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=p.parse_args()
    print(json.dumps(validate(args.root),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
