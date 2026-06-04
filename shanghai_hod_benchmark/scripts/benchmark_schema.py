"""Shared taxonomy and schemas for Shanghai-HOD benchmark generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSpec:
    code: str
    name: str
    indicators: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class IndicatorSpec:
    code: str
    name: str
    unit: str
    module: str
    low: float
    high: float
    decimals: int
    threshold: float | None = None
    higher_is_risk: bool = False


MODULES: tuple[ModuleSpec, ...] = (
    ModuleSpec("M01", "预约挂号", ("appointment_registrations",), "实时预约挂号人次"),
    ModuleSpec("M02", "门急诊运行", ("outpatient_emergency_visits",), "门急诊人次"),
    ModuleSpec("M03", "出院运行", ("discharges",), "出院人次"),
    ModuleSpec("M04", "住院手术", ("inpatient_surgeries",), "住院手术人次"),
    ModuleSpec("M05", "重点病种", ("key_disease_cases",), "重点病种病例数、病种分析"),
    ModuleSpec("M06", "费用结构", ("avg_inpatient_cost",), "住院均次费用"),
    ModuleSpec("M07", "药耗结构", ("inpatient_drug_ratio", "inpatient_consumable_ratio"), "住院药占比、住院耗材比"),
    ModuleSpec("M08", "新优药械/国谈", ("innovative_drug_device_cases", "national_negotiation_drug_cases"), "新优药械、国谈相关指标"),
    ModuleSpec("M09", "特需国际医疗", ("special_international_outpatient_ratio",), "特需国际医疗门诊人次占比"),
    ModuleSpec("M10", "合理用药", ("rational_drug_alerts",), "合理用药分析"),
    ModuleSpec("M11", "质量安全", ("class_iii_incision_infection_rate",), "三类切口感染率"),
    ModuleSpec("M12", "重返分析", ("return_rate", "return_visits"), "重返率、重返人次"),
    ModuleSpec("M13", "综合播报", ("briefing_summary",), "今日/半日/专题播报"),
    ModuleSpec("M14", "政策解释", ("policy_interpretation",), "政策知识库、指标口径解释"),
)

INDICATORS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec("appointment_registrations", "预约挂号人次", "人次", "M01", 35, 220, 0),
    IndicatorSpec("outpatient_emergency_visits", "门急诊人次", "人次", "M02", 60, 320, 0),
    IndicatorSpec("discharges", "出院人次", "人次", "M03", 6, 65, 0),
    IndicatorSpec("inpatient_surgeries", "住院手术人次", "人次", "M04", 2, 40, 0),
    IndicatorSpec("key_disease_cases", "重点病种病例数", "例", "M05", 0, 35, 0),
    IndicatorSpec("avg_inpatient_cost", "住院均次费用", "元", "M06", 7200, 38000, 2, threshold=30000, higher_is_risk=True),
    IndicatorSpec("inpatient_drug_ratio", "住院药占比", "比例", "M07", 0.18, 0.42, 4, threshold=0.35, higher_is_risk=True),
    IndicatorSpec("inpatient_consumable_ratio", "住院耗材比", "比例", "M07", 0.12, 0.34, 4, threshold=0.28, higher_is_risk=True),
    IndicatorSpec("innovative_drug_device_cases", "新优药械使用例数", "例", "M08", 0, 22, 0),
    IndicatorSpec("national_negotiation_drug_cases", "国谈药品使用例数", "例", "M08", 0, 35, 0),
    IndicatorSpec("special_international_outpatient_ratio", "特需国际医疗门诊人次占比", "比例", "M09", 0.01, 0.16, 4),
    IndicatorSpec("rational_drug_alerts", "合理用药预警数", "条", "M10", 0, 18, 0, threshold=12, higher_is_risk=True),
    IndicatorSpec("class_iii_incision_infection_rate", "三类切口感染率", "比例", "M11", 0.001, 0.032, 4, threshold=0.02, higher_is_risk=True),
    IndicatorSpec("return_rate", "重返率", "比例", "M12", 0.004, 0.055, 4, threshold=0.04, higher_is_risk=True),
    IndicatorSpec("return_visits", "重返人次", "人次", "M12", 0, 18, 0, threshold=12, higher_is_risk=True),
    IndicatorSpec("bed_occupancy_rate", "床位使用率", "比例", "M03", 0.62, 0.98, 4, threshold=0.94, higher_is_risk=True),
)

QUESTION_TYPES = (
    "single_module",
    "multi_module",
    "management_open",
    "ambiguous_boundary",
    "hallucination_trap",
    "spoken_noisy",
)

QUERY_TYPES = (
    "DATA_LOOKUP",
    "DATA_RANKING",
    "DATA_TREND",
    "ANOMALY_DETECTION",
    "MANAGEMENT_BRIEFING",
    "POLICY_EXPLANATION",
    "CLARIFICATION_REQUIRED",
    "SAFE_REFUSAL_REQUIRED",
)

HOSPITAL_GROUPS = ("general", "specialty", "children", "traditional_chinese_medicine", "oncology", "infectious_disease")


def module_by_code(code: str) -> ModuleSpec:
    return next(module for module in MODULES if module.code == code)


def indicator_by_code(code: str) -> IndicatorSpec:
    return next(indicator for indicator in INDICATORS if indicator.code == code)
