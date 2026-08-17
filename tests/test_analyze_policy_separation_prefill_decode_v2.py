"""Lightweight tests for Family B v2 analysis helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_policy_separation_prefill_decode_pilot_v2 import (
    analyze,
    parse_scenario_id,
    smoke_gate,
    winner_margin,
)


def test_parse_scenario_id_v2_fields():
    sid = "pd2.hog12.late40.slohog_ttft.s20260820"
    meta = parse_scenario_id(sid)
    assert meta["n_hog"] == 12
    assert meta["n_late"] == 40
    assert meta["slo_emphasis"] == "hog_ttft"
    assert meta["seed"] == 20260820
    assert meta["pair_id"] == "pd2.hog12.late40.slohog_ttft"


def test_winner_margin_two_policies():
    winners, best, second, margin = winner_margin({"full_prefill": 0.6, "chunked_prefill_small": 0.4})
    assert best == "full_prefill"
    assert second == "chunked_prefill_small"
    assert margin == pytest.approx(0.2)
    assert winners == ["full_prefill"]


def test_smoke_gate_requires_both_directions():
    grouped = {
        "pd2.hog12.late12.slohog_ttft.s7": {
            "full_prefill": {"arrival_normalized_weighted_goodput": "0.70"},
            "chunked_prefill_small": {"arrival_normalized_weighted_goodput": "0.50"},
        },
        "pd2.hog12.late40.slolate_ttft.s7": {
            "full_prefill": {"arrival_normalized_weighted_goodput": "0.40"},
            "chunked_prefill_small": {"arrival_normalized_weighted_goodput": "0.65"},
        },
    }
    g = smoke_gate(grouped)
    assert g["verdict"] == "SMOKE_GO"
    one_way = {
        "a": {
            "full_prefill": {"arrival_normalized_weighted_goodput": "0.70"},
            "chunked_prefill_small": {"arrival_normalized_weighted_goodput": "0.50"},
        }
    }
    assert smoke_gate(one_way)["verdict"] == "FAMILY_B_REFINEMENT_NO_GO"


def test_analyze_scores_gate_on_synthetic_grid():
    rows = []
    features = {}
    # 8 factor cells × 4 seeds. hog_ttft → full wins; late_ttft → small wins.
    hog_ns = (12, 24)
    late_ns = (12, 40)
    slos = ("hog_ttft", "late_ttft")
    seeds = (20260820, 20260821, 20260822, 20260823)
    for hog in hog_ns:
        for late in late_ns:
            for slo in slos:
                for seed in seeds:
                    sid = f"pd2.hog{hog}.late{late}.slo{slo}.s{seed}"
                    if slo == "hog_ttft":
                        full, small = 0.70, 0.50
                        hog_tf, hog_ts = 0.04, 0.10
                        late_tf, late_ts = 0.20, 0.12
                        slack_h, slack_l = 0.13, 2.08
                    else:
                        full, small = 0.45, 0.70
                        hog_tf, hog_ts = 0.04, 0.10
                        late_tf, late_ts = 0.25, 0.08
                        slack_h, slack_l = 2.08, 0.16
                    for name, anwg, h_ttft, l_ttft in (
                        ("full_prefill", full, hog_tf, late_tf),
                        ("chunked_prefill_small", small, hog_ts, late_ts),
                    ):
                        rows.append(
                            {
                                "scenario_id": sid,
                                "policy_name": name,
                                "arrival_normalized_weighted_goodput": str(anwg),
                                "status": "success",
                                "hog_mean_ttft": str(h_ttft),
                                "late_mean_ttft": str(l_ttft),
                                "decode_stalled_steps": "0",
                                "prefill_stalled_steps": "10",
                                "hog_slo_success": "0.5",
                                "late_slo_success": "0.5",
                            }
                        )
                    features[sid] = {
                        "mean_e2e_slack_hog": str(slack_h),
                        "mean_e2e_slack_late": str(slack_l),
                    }
    summary = analyze(rows, features)
    assert summary["n_comparable_cells"] == 32
    assert summary["family_b_verdict"] == "FAMILY_B_COMPOSITION_READY"
    hyp = {h["id"]: h["verdict"] for h in summary["hypotheses"]}
    assert hyp["H7"] == "NOT_APPLICABLE"
    assert hyp["H1"] == "CONFIRM"
    assert hyp["H2"] == "CONFIRM"
    assert hyp["H10"] == "CONFIRM"
