from __future__ import annotations

import pandas as pd

from llmserveopt.analysis import public_trace_replay_v1_analysis as ana


def _rows():
    rows = []
    for source in ("burstgpt", "azure_2023_conv"):
        sid_f = f"PUBLIC_TRACE::{source}::w0::faithful"
        rows.extend([
            {
                "canonical_scenario_id": sid_f,
                "canonical_policy_id": "full_prefill",
                "source_dataset": source,
                "scenario_evidence_class": ana.FAITHFUL,
                "primary_utility_anwg": 1.0,
                "secondary_completion_fraction": 1.0,
                "mean_latency": 1.0,
                "p95_latency": 1.0,
                "mean_queuing_delay": 0.0,
                "mean_ttft": 0.1,
                "slo_violation_rate": 0.0,
                "status": "success",
            },
            {
                "canonical_scenario_id": sid_f,
                "canonical_policy_id": "chunked_prefill_small",
                "source_dataset": source,
                "scenario_evidence_class": ana.FAITHFUL,
                "primary_utility_anwg": 0.9 if source == "azure_2023_conv" else 1.0,
                "secondary_completion_fraction": 1.0,
                "mean_latency": 1.2,
                "p95_latency": 1.2,
                "mean_queuing_delay": 0.0,
                "mean_ttft": 0.2,
                "slo_violation_rate": 0.0,
                "status": "success",
            },
        ])
        sid_a = f"PUBLIC_TRACE::{source}::w0::augmented"
        for policy, value in {
            "full_prefill": 0.5,
            "chunked_prefill_small": 0.5,
            "estimated_service_time_first": 0.7,
            "least_laxity_first": 0.6,
            "kv_constrained_online": 0.7,
            "weighted_fair_share": 0.4,
        }.items():
            rows.append({
                "canonical_scenario_id": sid_a,
                "canonical_policy_id": policy,
                "source_dataset": source,
                "scenario_evidence_class": ana.AUGMENTED,
                "primary_utility_anwg": value,
                "secondary_completion_fraction": 1.0,
                "status": "success",
            })
    return pd.DataFrame(rows)


def test_evidence_class_separation_and_faithful_pairing():
    s = ana.faithful_two_policy_summary(_rows())
    assert s["n_windows"] == 2
    assert s["wins_chunked_ties_losses"] == {"wins": 0, "ties": 1, "losses": 1}
    assert s["exact_tie_fraction"] == 0.5


def test_best_fixed_and_envelope_for_augmented_only():
    s = ana.best_fixed_and_envelope(_rows(), ana.AUGMENTED)
    assert s["best_fixed_policy"] == "estimated_service_time_first"
    assert s["best_fixed_mean"] == 0.7
    assert s["envelope_mean"] == 0.7
    assert s["positive_gain_fraction"] == 0.0


def test_tie_counting_with_multiple_winners():
    m = ana._metric_matrix(_rows(), ana.AUGMENTED)
    s = ana.winner_summary(m)
    assert s["n_tie_windows"] == 2
    assert s["tie_multiplicity_distribution"] == {"2": 2}
    assert s["n_distinct_winning_policies_fractional"] == 2


def test_pairwise_policy_separation_counts():
    s = ana.pairwise_policy_separation(_rows(), ana.AUGMENTED)
    pair = s["pairs"]["estimated_service_time_first__vs__weighted_fair_share"]
    assert pair["mean_abs_diff"] == 0.29999999999999993
    assert pair["estimated_service_time_first_wins"] == 2
    assert pair["weighted_fair_share_wins"] == 0
    assert pair["ties"] == 0


def test_parse_public_scenario_id():
    parsed = ana.parse_public_scenario_id("PUBLIC_TRACE::azure_2023_code::w38::augmented")
    assert parsed["source_dataset"] == "azure_2023_code"
    assert parsed["window_index"] == 38
    assert parsed["base_window_id"] == "azure_2023_code::w38"
