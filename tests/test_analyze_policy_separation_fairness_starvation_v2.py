"""Lightweight tests for Family A v2 fairness-vs-size analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_policy_separation_fairness_starvation_pilot_v2 import (
    analyze,
    near_tie,
    parse_scenario_id,
    shannon_entropy,
    winner_margin,
    write_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / (
    "experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377"
    "/per_policy_results.csv"
)
V2_FEATURES = V2_CSV.parent / "scenario_features.csv"


def test_parse_scenario_id_v2_fields():
    sid = "fs2.util1.3000.skew5.0000.favlong.noise0.30.s20260816"
    meta = parse_scenario_id(sid)
    assert meta["target_utilization"] == pytest.approx(1.3)
    assert meta["tenant_weight_skew"] == pytest.approx(5.0)
    assert meta["favored_tenant_size"] == "long"
    assert meta["prediction_noise_sigma"] == pytest.approx(0.3)
    assert meta["seed"] == 20260816


def test_parse_scenario_id_rejects_v1():
    with pytest.raises(ValueError):
        parse_scenario_id("fs.util1.2000.skew10.0000.vol0.2000.s20260815")


def test_winner_margin_and_entropy():
    winners, best, second, margin = winner_margin({"a": 0.9, "b": 0.8})
    assert best == "a"
    assert second == "b"
    assert margin == pytest.approx(0.1)
    assert near_tie({"a": 0.9, "b": 0.895}, 0.01) is True
    assert shannon_entropy({"x": 1, "y": 1}) == pytest.approx(1.0)


@pytest.mark.skipif(not V2_CSV.is_file(), reason="Family A v2 CSV absent")
def test_v2_corpus_deterministic_summary(tmp_path):
    import csv

    with V2_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    features = {}
    if V2_FEATURES.is_file():
        with V2_FEATURES.open(newline="") as f:
            features = {r["scenario_id"]: r for r in csv.DictReader(f)}
    result = analyze(rows, features=features)
    write_artifacts(tmp_path, result)
    summary = result["summary"]
    assert summary["integrity"]["n_scenarios"] == 72
    assert summary["integrity"]["n_rows"] == 288
    assert summary["integrity"]["failed_rows"] == 0
    assert summary["integrity"]["burstgpt_only"] is True
    assert summary["estf_vs_wfs"]["bidirectional_eps_0.01"] is True
    assert summary["estf_vs_wfs"]["i_beats_j_eps_0.01"] == 26
    assert summary["estf_vs_wfs"]["j_beats_i_eps_0.01"] == 29
    assert summary["near_tie_rates"]["0.01"] == pytest.approx(13 / 72)
    assert summary["aging_perfect_anwg_count"] == 6
    assert (tmp_path / "analysis_summary.json").is_file()
    loaded = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    assert loaded["unique_winner_counts"]["weighted_fair_share"] == 29
