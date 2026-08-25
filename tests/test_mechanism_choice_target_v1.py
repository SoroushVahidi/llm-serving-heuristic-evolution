"""Focused tests for the mechanism-choice target formulas (feasibility
investigation only -- see
docs/audits/mechanism_choice_target_feasibility_v1_20260817.md).

Verdict of that audit is MECHANISM_TARGET_NO_GO: these tests validate that
the formulas used to REACH that verdict are correct and reproducible, not
that the target is production-ready.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.policy_separation.mechanism_choice_target_v1 import (  # noqa: E402
    MECHANISM_POLICY_PAIRS,
    MECHANISMS,
    NATIVE_MECHANISM_BY_FAMILY,
    classify_target,
    classify_target_with_abstention,
    compute_mechanism_gains,
)

UNIFIED = REPO_ROOT / "experiments" / "unified_utility_matrix_v2" / "unified_utility_matrix_wide_v2.csv"


def test_mechanism_set_and_native_family_map_are_consistent():
    assert set(MECHANISMS) == set(MECHANISM_POLICY_PAIRS.keys()) == {"ranking", "chunk", "kv"}
    assert set(NATIVE_MECHANISM_BY_FAMILY.values()) == set(MECHANISMS)
    assert len(NATIVE_MECHANISM_BY_FAMILY) == 3


def test_compute_mechanism_gains_is_symmetric_and_nonnegative():
    row = {
        "anwg__weighted_fair_share": 0.7,
        "anwg__estimated_service_time_first": 0.5,
        "anwg__chunked_prefill_small": 0.3,
        "anwg__full_prefill": 0.3,
        "anwg__kv_constrained_online": 0.9,
        "anwg__least_laxity_first": 0.4,
    }
    gains = compute_mechanism_gains(row)
    assert gains["ranking"] == pytest.approx(0.2)
    assert gains["chunk"] == pytest.approx(0.0)
    assert gains["kv"] == pytest.approx(0.5)
    for v in gains.values():
        assert v >= 0.0

    # symmetry: swapping the two policies in the pair must not change |gap|
    row2 = dict(row)
    row2["anwg__weighted_fair_share"], row2["anwg__estimated_service_time_first"] = (
        row2["anwg__estimated_service_time_first"],
        row2["anwg__weighted_fair_share"],
    )
    assert compute_mechanism_gains(row2)["ranking"] == pytest.approx(gains["ranking"])


def test_compute_mechanism_gains_never_reads_family_or_scenario_fields():
    row = {
        "anwg__weighted_fair_share": 0.1,
        "anwg__estimated_service_time_first": 0.2,
        "anwg__chunked_prefill_small": 0.3,
        "anwg__full_prefill": 0.4,
        "anwg__kv_constrained_online": 0.5,
        "anwg__least_laxity_first": 0.6,
        "mechanism_family": "SHOULD_NEVER_BE_READ",
        "canonical_scenario_id": "SHOULD_NEVER_BE_READ",
    }
    # Must not raise, and result must not depend on the extra fields' values.
    gains_a = compute_mechanism_gains(row)
    row["mechanism_family"] = "SOMETHING_ELSE"
    row["canonical_scenario_id"] = "SOMETHING_ELSE_TOO"
    gains_b = compute_mechanism_gains(row)
    assert gains_a == gains_b


def test_classify_target_picks_max_and_correct_margin():
    gains = {"ranking": 0.05, "chunk": 0.30, "kv": 0.10}
    mech, top_gain, margin = classify_target(gains)
    assert mech == "chunk"
    assert top_gain == pytest.approx(0.30)
    assert margin == pytest.approx(0.30 - 0.10)


def test_classify_target_tie_break_is_alphabetical_and_deterministic():
    gains = {"ranking": 0.2, "chunk": 0.2, "kv": 0.05}
    mech, top_gain, margin = classify_target(gains)
    assert mech == "chunk"  # alphabetically first among the tied max
    assert margin == pytest.approx(0.0)


def test_classify_target_with_abstention_respects_eps():
    below_eps = {"ranking": 0.005, "chunk": 0.003, "kv": 0.001}
    assert classify_target_with_abstention(below_eps, eps=0.01) == "no_clear_mechanism"

    above_eps = {"ranking": 0.02, "chunk": 0.003, "kv": 0.001}
    assert classify_target_with_abstention(above_eps, eps=0.01) == "ranking"


def test_target_computation_deterministic_and_reproducible_on_frozen_matrix():
    if not UNIFIED.exists():
        pytest.skip("frozen unified utility matrix not present locally")
    with open(UNIFIED, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 176

    def _run() -> list:
        out = []
        for r in rows:
            anwg_row = {k: float(v) for k, v in r.items() if k.startswith("anwg__")}
            gains = compute_mechanism_gains(anwg_row)
            mech, top_gain, margin = classify_target(gains)
            out.append((r["canonical_scenario_id"], mech, round(top_gain, 12), round(margin, 12)))
        return out

    first = _run()
    second = _run()
    assert first == second


def test_gain_kv_confound_regression_family_a_exceeds_family_c():
    """Regression guard for the audit's central negative finding: gain_kv's
    mean on Family A (no real KV pressure -- see SHARED_CORE_V1's
    token_footprint_per_kv) is larger than on Family C (KV's own native
    family). If this ever flips, the confound finding in the mechanism-
    choice-target audit needs to be re-examined, not silently assumed."""
    if not UNIFIED.exists():
        pytest.skip("frozen unified utility matrix not present locally")
    with open(UNIFIED, newline="") as f:
        rows = list(csv.DictReader(f))
    gains_by_family: dict = {}
    for r in rows:
        anwg_row = {k: float(v) for k, v in r.items() if k.startswith("anwg__")}
        g = compute_mechanism_gains(anwg_row)["kv"]
        gains_by_family.setdefault(r["mechanism_family"], []).append(g)
    mean_a = sum(gains_by_family["FAMILY_A_FAIRNESS_STARVATION_V2"]) / len(
        gains_by_family["FAMILY_A_FAIRNESS_STARVATION_V2"]
    )
    mean_c = sum(gains_by_family["FAMILY_C_KV_PRESSURE_V2"]) / len(gains_by_family["FAMILY_C_KV_PRESSURE_V2"])
    assert mean_a > mean_c
