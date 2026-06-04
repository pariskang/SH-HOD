"""Validated ingestion and anonymization for real/hybrid DataQA source records."""
from __future__ import annotations
import csv, hashlib
from pathlib import Path
from typing import Any

REQUIRED = {"hospital_id","timestamp_start","timestamp_end","indicator_code","value","unit"}


def anonymize_hospital(raw: str) -> str:
    number = int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16) % 999 + 1
    return f"SH-MH{number:03d}"


def ingest_csv(path: Path, anonymize: bool = True) -> list[dict[str, Any]]:
    """Load approved aggregate records; rejects patient-level or incomplete inputs."""
    with path.open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError("input records CSV is empty")
    missing=REQUIRED-set(rows[0]);
    if missing: raise ValueError(f"input records missing fields: {sorted(missing)}")
    forbidden={k for k in rows[0] if any(x in k.lower() for x in ("patient","name","id_card","身份证","患者"))}
    if forbidden: raise ValueError(f"patient-level fields are forbidden: {sorted(forbidden)}")
    output=[]
    for idx,row in enumerate(rows,1):
        try: value=float(row['value'])
        except ValueError as exc: raise ValueError(f"invalid numeric value at row {idx}") from exc
        output.append({"row_id":f"R{idx:06d}","hospital_id":anonymize_hospital(row['hospital_id']) if anonymize else row['hospital_id'],"hospital_group":row.get('hospital_group','unknown'),"timestamp_start":row['timestamp_start'],"timestamp_end":row['timestamp_end'],"indicator_code":row['indicator_code'],"indicator_name":row.get('indicator_name',row['indicator_code']),"value":value,"unit":row['unit'],"numerator":row.get('numerator',value),"denominator":row.get('denominator',''),"data_quality_flag":row.get('data_quality_flag','normal'),"source_type":"real","scenario_id":row.get('scenario_id','REAL_BASELINE')})
    return output
