"""Lightweight tests for Family B v1 prefill/decode analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_policy_separation_prefill_decode_pilot_v1 import (
    analyze,
    near_tie,
    parse_scenario_id,
    shannon_entropy,
    winner_margin,
    write_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
B_CSV = ROOT / (
    "experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z"
    "/per_policy_results.csv"
)
B_FEATURES = B_CSV.parent / "scenario_features.csv"


def test_parse_scenario_id_v1_fields():
    sid = "pd1.psizelong.occhigh.slotbt_tight.loadmoderate.s20260818"
    meta = parse_scenario_id(sid)
    assert meta["prefill_size_class"] == "long"
    assert meta["decode_occupancy"] == "high"
    assert meta["slo_regime"] == "tbt_tight"
    assert meta["offered_load"] == "moderate"
    assert meta["seed"] == 20260818


def test_parse_scenario_id_rejects_family_a():
    with pytest.raises(ValueError):
        parse_scenario_id("fs2.util1.3000.skew5.0000.favlong.noise0.30.s20260816")


def test_winner_margin_and_entropy():
    winners, best, second, margin = winner_margin({"a": 0.9, "b": 0.8})
    assert best == "a"
    assert second == "b"
    assert margin == pytest.approx(0.1)
    assert near_tie({"a": 0.9, "b": 0.895}, 0.01) is True
    assert shannon_entropy({"x": 1, "y": 1}) == pytest.approx(1.0)


@pytest.mark.skipif(not B_CSV.is_file(), reason="Family B v1 CSV absent")
def test_family_b_corpus_deterministic_summary(tmp_path):
    import csv

    with B_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    features = {}
    if B_FEATURES.is_file():
        with B_FEATURES.open(newline="", encoding="utf-8") as f:
            features = {r["scenario_id"]: r for r in csv.DictReader(f)}
    result = analyze(rows, features=features)
    write_artifacts(tmp_path, result)
    summary = result["summary"]
    assert summary["integrity"]["n_scenarios"] == 144
    assert summary["integrity"]["n_rows"] == 720
    assert summary["integrity"]["failed_rows"] == 0
    assert summary["integrity"]["expected_720"] is True
    assert summary["integrity"]["burstgpt_consistent"] is True
    assert summary["integrity"]["nan_inf_primary"] == 0
    assert summary["exact_tie_count"] == 134
    assert summary["near_tie_counts"]["0.01"] == 138
    assert summary["unique_winner_counts_structural_eps_0.01"] == {"full_prefill": 6}
    assert summary["identity_collapse"][
        "chunked_prefill_small_eq_decode_priority_chunked"
    ] == 144
    assert summary["identity_collapse"]["chunked_prefill_small_eq_adaptive"] == 144
    assert summary["identity_collapse"]["spread_gt_0.01_structural"] == 58
    pair = next(
        p
        for p in summary["important_pairs"]
        if p["policy_i"] == "full_prefill" and p["policy_j"] == "chunked_prefill_small"
    )
    assert pair["i_beats_j_eps_0.01"] == 47
    assert pair["j_beats_i_eps_0.01"] == 11
    assert pair["bidirectional_eps_0.01"] is True
    assert summary["adaptive_diagnostic"]["envelope_expand_eps_0.01"] == 0
    assert summary["family_b_verdict"] == "USEFUL_BUT_NEEDS_REFINEMENT"
    assert summary["composition_decision"] == "PREFILL_COMPOSITION_NOT_YET_JUSTIFIED"
    hyp = {h["id"]: h["verdict"] for h in summary["hypotheses"]}
    assert hyp["H6"] == "CONTRADICT"
    assert hyp["H7"] == "CONTRADICT"
    assert hyp["H8"] == "CONFIRM"
    assert (tmp_path / "analysis_summary.json").is_file()
    loaded = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    assert loaded["unique_winner_counts"]["full_prefill"] == 10
