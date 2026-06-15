"""Strict cross-file validation for generated Shanghai-HOD artifacts."""
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter
from pathlib import Path
from typing import Any

HOSPITAL_ID_PATTERN = re.compile(r"SH-MH\d{3}")


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

    # --- DataQA task type coverage (8 baseline + 7 new hard task types) ---
    task_types=Counter(x['task_type'] for x in questions)
    required={
        'direct_lookup','cross_hospital_ranking','half_hour_mom','sustained_trend',
        'composite_metric_explanation','anomaly_detection','priority_ranking','briefing',
        'cross_window_extremum','consecutive_threshold_streak','counterfactual_threshold',
        'quality_filtered_aggregate','negative_enumeration','multi_criteria_ranking',
        'precision_percentage_change',
    }
    missing=required-set(task_types)
    assert not missing, f'missing task types: {missing}'

    # --- DataQA briefing must include at least 3 distinct subtypes ---
    briefing_subtypes=Counter(
        x.get('briefing_subtype','unspecified')
        for x in questions
        if x['task_type']=='briefing'
    )
    required_briefing_subtypes={'risk_focused_targeted','conflicting_signals_briefing','exclusion_briefing'}
    missing_briefings=required_briefing_subtypes-set(briefing_subtypes)
    assert not missing_briefings, f'missing briefing subtypes: {missing_briefings}'

    # --- Q37 question_type coverage (must include the new hard reasoning families) ---
    q37_qtypes=Counter(x['question_type'] for x in hidden)
    required_q37={
        'single_module','multi_module','cross_module_chain','temporal_compound',
        'conflicting_signals','boundary_precision','negative_enumeration',
        'management_open','ambiguous_boundary','hallucination_trap','spoken_noisy',
    }
    missing_q37=required_q37-set(q37_qtypes)
    assert not missing_q37, f'missing q37 question types: {missing_q37}'

    # --- Q37 difficulty must skew toward hard reasoning (≥40% hard, ≤25% easy) ---
    diff=Counter(x['difficulty'] for x in hidden)
    total=len(hidden)
    hard_share=diff.get('hard',0)/total if total else 0
    easy_share=diff.get('easy',0)/total if total else 0
    assert hard_share>=0.40, f'hard share {hard_share:.2%} below 40% — benchmark is too easy'
    assert easy_share<=0.25, f'easy share {easy_share:.2%} above 25% — too many trivial questions'

    # --- Hallucination traps must be diversified (≥12 distinct trap_type values) ---
    trap_types={x['expected_slots'].get('trap_type') for x in hidden if x['question_type']=='hallucination_trap'}
    trap_types.discard(None)
    assert len(trap_types)>=12, f'hallucination traps too repetitive: {len(trap_types)} distinct trap_type values, need ≥12'

    # --- Briefing factual grounding: every SH-MH### cited in the briefing's final_answer must
    # appear in the briefing's evidence_rows hospital set. Prevents factual drift in narrative outputs. ---
    by_row_id={x['row_id']:x for x in records}
    ungrounded=[]
    answer_by_qid={a['question_id']:a for a in answers}
    for q in questions:
        if q['task_type']!='briefing':
            continue
        a=answer_by_qid.get(q['question_id'])
        if not a:
            continue
        ev_hospitals={by_row_id[rid]['hospital_id'] for rid in q.get('evidence_rows',[]) if rid in by_row_id}
        cited=set(HOSPITAL_ID_PATTERN.findall(a['final_answer']))
        if cited-ev_hospitals:
            ungrounded.append((q['question_id'],sorted(cited-ev_hospitals)))
    assert not ungrounded, f'briefings cite non-evidence hospitals: {ungrounded[:3]}'

    # --- Padding ratio: limit suffix-based variants to ≤40% so unique reasoning content dominates. ---
    suffixes=['请给出可追溯依据','按管理关注度排序','只基于大屏已有数据回答','如果信息不足请先澄清']
    padded=sum(1 for h in hidden if any(s in h['question'] for s in suffixes))
    pad_share=padded/len(hidden) if hidden else 0
    assert pad_share<=0.40, f'padding share {pad_share:.2%} above 40% — too many suffix variants'

    return {
        'q37':len(public),
        'dataqa':len(questions),
        'records':len(records),
        'anomalies':len(anomalies),
        'task_types':dict(task_types),
        'briefing_subtypes':dict(briefing_subtypes),
        'q37_question_types':dict(q37_qtypes),
        'q37_difficulty':dict(diff),
        'q37_hard_share':round(hard_share,3),
        'q37_trap_diversity':len(trap_types),
        'q37_padding_share':round(pad_share,3),
        'briefing_ungrounded_count':len(ungrounded),
    }


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=p.parse_args()
    print(json.dumps(validate(args.root),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
