"""Lightweight tests for Family A fairness/starvation analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_policy_separation_fairness_starvation_pilot import (
    analyze,
    near_tie,
    parse_scenario_id,
    pairwise_delta,
    resolve_primary_field,
    winner_margin,
    write_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / (
    "experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306"
    "/per_policy_results.csv"
)


def test_parse_scenario_id_roundtrip_fields():
    sid = "fs.util1.2000.skew10.0000.vol0.2000.s20260815"
    meta = parse_scenario_id(sid)
    assert meta["target_utilization"] == pytest.approx(1.2)
    assert meta["tenant_weight_skew"] == pytest.approx(10.0)
    assert meta["interactive_volume_fraction"] == pytest.approx(0.2)
    assert meta["seed"] == 20260815


def test_parse_scenario_id_rejects_malformed():
    with pytest.raises(ValueError):
        parse_scenario_id("not-a-family-a-id")


def test_resolve_primary_field_historical_and_clarified():
    col, label = resolve_primary_field(["scenario_id", "anwg", "status"])
    assert col == "anwg"
    assert "unweighted" in label
    col2, label2 = resolve_primary_field(
        ["unweighted_slo_success_rate", "arrival_normalized_weighted_goodput"]
    )
    assert col2 == "unweighted_slo_success_rate"
    assert label2 == "unweighted_slo_success_rate"


def test_winner_margin_and_near_tie():
    scores = {"a": 1.0, "b": 1.0, "c": 0.9}
    winners, best, second, margin = winner_margin(scores)
    assert set(winners) == {"a", "b"}
    assert best is None
    assert margin == pytest.approx(0.0)
    assert near_tie(scores, 0.0) is True
    assert near_tie({"a": 1.0, "b": 0.995}, 0.01) is True
    assert near_tie({"a": 1.0, "b": 0.98}, 0.005) is False

    winners2, best2, second2, margin2 = winner_margin({"a": 1.0, "b": 0.95})
    assert winners2 == ["a"]
    assert best2 == "a"
    assert second2 == "b"
    assert margin2 == pytest.approx(0.05)


def test_pairwise_delta_sign():
    scores = {"estf": 0.97, "wfs": 0.95}
    assert pairwise_delta(scores, "estf", "wfs") == pytest.approx(0.02)
    assert pairwise_delta(scores, "wfs", "estf") == pytest.approx(-0.02)


@pytest.mark.skipif(not HISTORICAL.is_file(), reason="historical Family A CSV absent")
def test_historical_corpus_deterministic_summary(tmp_path):
    import csv

    with HISTORICAL.open(newline="") as f:
        rows = list(csv.DictReader(f))
    bundle = analyze(
        rows,
        primary_col="anwg",
        primary_label="historical_unweighted_slo_success_rate",
        has_canonical_anwg=False,
    )
    s = bundle["summary"]
    assert s["n_scenarios"] == 120
    assert s["exact_tie_count"] == 68
    assert s["unique_winner_counts"] == {"aging_priority": 52}
    assert s["near_tie_rates"]["0.01"] == pytest.approx(0.6)
    assert s["headroom"]["mean"] == pytest.approx(0.0275)
    assert s["mechanism_flags"]["wfs_beats_estf_scenarios"] == 0
    assert s["mechanism_flags"]["aging_perfect_on_all_scenarios"] is True
    assert s["estf_vs_wfs"]["bidirectional_eps_0.01"] is False

    out = tmp_path / "analysis"
    written = write_artifacts(out, bundle)
    summary2 = json.loads(Path(written["summary_json"]).read_text(encoding="utf-8"))
    assert summary2["n_scenarios"] == 120
    assert (out / "pairwise_summary.csv").is_file()
