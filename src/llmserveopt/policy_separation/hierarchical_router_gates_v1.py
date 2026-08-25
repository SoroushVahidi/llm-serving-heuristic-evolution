"""Hierarchical Regime Router v1 -- gate evaluator and mechanical verdict
logic (design doc SS M/O, `configs/hierarchical_regime_router_v1_gates.json`).

Thresholds are read from the machine-readable gate config, never
duplicated as separate hardcoded literals in this module (task
requirement S7) -- the only exception is the mechanical verdict-mapping
structure itself (SS O), which is control-flow, not a threshold.

IMPLEMENTATION + VALIDATION ONLY. This module can mechanically compute a
verdict from a `metrics` dict -- it does not itself decide when it is
scientifically appropriate to call it with real TEST-split numbers; that
decision belongs to a separately authorized future task.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GATES_CONFIG_PATH = ROOT / "configs/hierarchical_regime_router_v1_gates.json"

VERDICT_NO_GO = "HIERARCHICAL_ROUTER_NO_GO"
VERDICT_ROUTING_WORKS_SELECTION_NO_GAIN = "HIERARCHICAL_ROUTER_ROUTING_WORKS_SELECTION_NO_GAIN"
VERDICT_INCONCLUSIVE = "HIERARCHICAL_ROUTER_INCONCLUSIVE"
VERDICT_GO = "HIERARCHICAL_ROUTER_GO"

_OPERATORS = {
    "==": lambda v, t: v == t,
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t,
    "<": lambda v, t: v < t,
}


def load_gates_config(path: Path = DEFAULT_GATES_CONFIG_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


@dataclass
class GateResult:
    id: str
    name: str
    critical: bool
    passed: Optional[bool]
    value: Any
    threshold: Any
    note: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "critical": self.critical,
            "passed": self.passed, "value": self.value, "threshold": self.threshold,
            "note": self.note, "extra": self.extra,
        }


def _gate_spec(config: Dict[str, Any], gate_id: str) -> Dict[str, Any]:
    for g in config["gates"]:
        if g["id"] == gate_id:
            return g
    raise KeyError(f"gate {gate_id} not found in gates config")


def _simple_gate(config: Dict[str, Any], gate_id: str, value: Optional[float]) -> GateResult:
    spec = _gate_spec(config, gate_id)
    if value is None:
        return GateResult(gate_id, spec["name"], spec["critical"], None, None, spec["threshold"], "value not supplied")
    op = _OPERATORS[spec["operator"]]
    passed = bool(op(value, spec["threshold"]))
    return GateResult(gate_id, spec["name"], spec["critical"], passed, value, spec["threshold"])


def evaluate_g1(config: Dict[str, Any], stage1_input_validity_fraction: Optional[float]) -> GateResult:
    return _simple_gate(config, "G1", stage1_input_validity_fraction)


def evaluate_g2(config: Dict[str, Any], router_macro_f1: Optional[float]) -> GateResult:
    return _simple_gate(config, "G2", router_macro_f1)


def evaluate_g3(config: Dict[str, Any], catastrophic_misroute_rate: Optional[float]) -> GateResult:
    return _simple_gate(config, "G3", catastrophic_misroute_rate)


def evaluate_g4(config: Dict[str, Any], stage2_preservation_fraction_by_regime: Optional[Dict[str, float]]) -> GateResult:
    """Gate applies to the MINIMUM per-regime retained fraction (design
    doc: 'each regime's... retained' -- read as a per-regime requirement,
    not an average, matching the gate's `critical=True` strictness)."""
    spec = _gate_spec(config, "G4")
    if not stage2_preservation_fraction_by_regime:
        return GateResult("G4", spec["name"], spec["critical"], None, None, spec["threshold"], "value not supplied")
    min_regime = min(stage2_preservation_fraction_by_regime, key=stage2_preservation_fraction_by_regime.get)
    min_value = stage2_preservation_fraction_by_regime[min_regime]
    op = _OPERATORS[spec["operator"]]
    passed = bool(op(min_value, spec["threshold"]))
    return GateResult(
        "G4", spec["name"], spec["critical"], passed, min_value, spec["threshold"],
        note=f"binding regime={min_regime}",
        extra={"by_regime": dict(stage2_preservation_fraction_by_regime)},
    )


def evaluate_g5(
    config: Dict[str, Any], mean_delta_anwg: Optional[float], bootstrap_ci_lower: Optional[float]
) -> GateResult:
    spec = _gate_spec(config, "G5")
    if mean_delta_anwg is None:
        return GateResult("G5", spec["name"], spec["critical"], None, None, spec["threshold"], "value not supplied")
    op = _OPERATORS[spec["operator"]]
    mean_ok = bool(op(mean_delta_anwg, spec["threshold"]))
    ci_ok = True if bootstrap_ci_lower is None else bool(bootstrap_ci_lower > 0)
    passed = mean_ok and ci_ok
    return GateResult(
        "G5", spec["name"], spec["critical"], passed, mean_delta_anwg, spec["threshold"],
        note="CI lower bound not supplied (sample-size gated)" if bootstrap_ci_lower is None else "",
        extra={"bootstrap_ci_lower": bootstrap_ci_lower, "mean_criterion_passed": mean_ok, "ci_criterion_passed": ci_ok},
    )


def evaluate_g6(config: Dict[str, Any], oracle_gap_closure: Optional[float]) -> GateResult:
    return _simple_gate(config, "G6", oracle_gap_closure)


def evaluate_g7(config: Dict[str, Any], multi_regime_benefit_count: Optional[int]) -> GateResult:
    return _simple_gate(config, "G7", multi_regime_benefit_count)


def evaluate_g8(
    config: Dict[str, Any], leakage_instance_count: Optional[int], qualitative_all_clusters_attributable: Optional[bool]
) -> GateResult:
    spec = _gate_spec(config, "G8")
    if leakage_instance_count is None:
        return GateResult("G8", spec["name"], spec["critical"], None, None, spec["threshold_a"], "value not supplied")
    a_passed = leakage_instance_count == spec["threshold_a"]
    b_passed = qualitative_all_clusters_attributable
    passed = bool(a_passed) and (b_passed is True)
    return GateResult(
        "G8", spec["name"], spec["critical"], passed, leakage_instance_count, spec["threshold_a"],
        note="qualitative (b) not yet reviewed" if b_passed is None else "",
        extra={"a_leakage_count": leakage_instance_count, "a_passed": a_passed, "b_passed": b_passed},
    )


def evaluate_g9(
    config: Dict[str, Any],
    family_c_held_out_delta_anwg: Optional[float],
    blended_microcase_catastrophic_rate: Optional[float],
) -> GateResult:
    spec = _gate_spec(config, "G9")
    if family_c_held_out_delta_anwg is None:
        return GateResult("G9", spec["name"], spec["critical"], None, None, spec["threshold_a"], "value not supplied")
    a_passed = family_c_held_out_delta_anwg >= spec["threshold_a"]
    b_passed = (
        None if blended_microcase_catastrophic_rate is None
        else blended_microcase_catastrophic_rate <= spec["threshold_b"]
    )
    passed = bool(a_passed) and (b_passed is not False)
    return GateResult(
        "G9", spec["name"], spec["critical"], passed, family_c_held_out_delta_anwg, spec["threshold_a"],
        note="(b) blended microcase rate not supplied" if b_passed is None else "",
        extra={"a_passed": a_passed, "b_passed": b_passed, "blended_microcase_catastrophic_rate": blended_microcase_catastrophic_rate},
    )


def evaluate_all_gates(metrics: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, GateResult]:
    config = config or load_gates_config()
    return {
        "G1": evaluate_g1(config, metrics.get("stage1_input_validity_fraction")),
        "G2": evaluate_g2(config, metrics.get("router_macro_f1")),
        "G3": evaluate_g3(config, metrics.get("catastrophic_misroute_rate")),
        "G4": evaluate_g4(config, metrics.get("stage2_preservation_fraction_by_regime")),
        "G5": evaluate_g5(config, metrics.get("mean_delta_anwg"), metrics.get("bootstrap_ci_lower")),
        "G6": evaluate_g6(config, metrics.get("oracle_gap_closure")),
        "G7": evaluate_g7(config, metrics.get("multi_regime_benefit_count")),
        "G8": evaluate_g8(config, metrics.get("leakage_instance_count"), metrics.get("qualitative_all_clusters_attributable")),
        "G9": evaluate_g9(config, metrics.get("family_c_held_out_delta_anwg"), metrics.get("blended_microcase_catastrophic_rate")),
    }


def compute_verdict(
    gates: Dict[str, GateResult],
    blended_microcase_sample_too_small: bool = False,
    test_sample_insufficient_for_g5_ci: bool = False,
) -> str:
    """Exact mechanical mapping, design doc SS O. Only G1-G5 and G8(a) can
    force NO_GO; G6/G7/G9 never independently force NO_GO/INCONCLUSIVE."""

    def _fail(gate_id: str) -> bool:
        return gates[gate_id].passed is False

    g8_a_fail = gates["G8"].extra.get("a_passed") is False

    if _fail("G1") or g8_a_fail:
        return VERDICT_NO_GO
    if _fail("G2") or _fail("G3"):
        return VERDICT_NO_GO
    if _fail("G4"):
        return VERDICT_NO_GO
    if _fail("G5"):
        if gates["G2"].passed and gates["G3"].passed and gates["G4"].passed:
            return VERDICT_ROUTING_WORKS_SELECTION_NO_GAIN
        return VERDICT_NO_GO
    if blended_microcase_sample_too_small or test_sample_insufficient_for_g5_ci:
        return VERDICT_INCONCLUSIVE
    return VERDICT_GO
