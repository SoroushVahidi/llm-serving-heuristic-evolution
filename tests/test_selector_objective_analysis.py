from __future__ import annotations

import pytest

from llmserveopt.selector.dataset_v2.objective_analysis import (
    ARRIVAL_NORMALIZED_OBJECTIVE,
    CONSTRAINED_ARRIVAL_NORMALIZED_OBJECTIVE,
    ConstrainedRankingConfig,
    constrained_winner,
    objective_summary,
    selective_service_advantages,
    sensitivity_grid,
)


def _row(
    scenario: str,
    window: int,
    policy: str,
    *,
    conditional: float,
    arrival_norm: float,
    completion: float,
    rejection: float,
    throughput: float = 1.0,
) -> dict:
    return {
        "scenario_id": scenario,
        "window_id": str(window),
        "policy_name": policy,
        "source_trace": "synthetic",
        "bottleneck_class": "unit",
        "metric_weighted_goodput": conditional,
        "metric_arrival_normalized_weighted_goodput": arrival_norm,
        "metric_completion_fraction": completion,
        "metric_rejection_fraction": rejection,
        "metric_rejection_rate": rejection,
        "metric_request_throughput": throughput,
    }


def test_constrained_ranking_filters_low_completion_high_rejection_policy():
    rows = [
        _row("s", 0, "scorpio", conditional=1.0, arrival_norm=0.4, completion=0.4, rejection=0.6),
        _row("s", 0, "edf", conditional=0.8, arrival_norm=0.8, completion=1.0, rejection=0.0),
    ]
    assert constrained_winner(rows, ConstrainedRankingConfig(0.5, 0.5)) == ("edf", pytest.approx(0.8))
    assert constrained_winner(rows, ConstrainedRankingConfig(0.1, 0.8)) == ("edf", pytest.approx(0.8))


def test_constrained_ranking_returns_none_when_no_policy_is_eligible():
    rows = [
        _row("s", 0, "a", conditional=1.0, arrival_norm=0.4, completion=0.4, rejection=0.6),
        _row("s", 0, "b", conditional=0.9, arrival_norm=0.3, completion=0.3, rejection=0.7),
    ]
    assert constrained_winner(rows, ConstrainedRankingConfig(0.8, 0.2)) == (None, None)


def test_sensitivity_grid_reports_policy_winner_changes():
    rows = [
        _row("s0", 0, "scorpio", conditional=1.0, arrival_norm=0.4, completion=0.4, rejection=0.6),
        _row("s0", 0, "edf", conditional=0.8, arrival_norm=0.8, completion=1.0, rejection=0.0),
        _row("s1", 0, "scorpio", conditional=0.95, arrival_norm=0.95, completion=1.0, rejection=0.0),
        _row("s1", 0, "edf", conditional=0.90, arrival_norm=0.90, completion=1.0, rejection=0.0),
    ]
    grid = sensitivity_grid(rows, completion_minimums=[0.5, 0.9], maximum_rejections=[0.5, 0.1])
    assert grid["cf_ge_0.50__rej_le_0.50"]["winner_counts"] == {"edf": 1, "scorpio": 1}
    assert grid["cf_ge_0.90__rej_le_0.10"]["windows_with_eligible_policy"] == 2


def test_selective_service_reversal_detection_counts_old_metric_reversal():
    rows = [
        _row("s", 0, "scorpio", conditional=1.0, arrival_norm=0.2, completion=0.2, rejection=0.8),
        _row("s", 0, "edf", conditional=0.8, arrival_norm=0.8, completion=1.0, rejection=0.0),
    ]
    report = selective_service_advantages(rows)
    assert report["SELECTIVE_SERVICE_ADVANTAGE"] == 1
    assert report["by_advantaged_policy"] == {"scorpio": 1}
    assert report["examples"][0]["historical_winner"] == "scorpio"


def test_objective_summary_supports_arrival_normalized_and_constrained_objectives():
    rows = [
        _row("s0", 0, "scorpio", conditional=1.0, arrival_norm=0.2, completion=0.2, rejection=0.8),
        _row("s0", 0, "edf", conditional=0.8, arrival_norm=0.8, completion=1.0, rejection=0.0),
        _row("s1", 0, "scorpio", conditional=0.9, arrival_norm=0.9, completion=1.0, rejection=0.0),
        _row("s1", 0, "edf", conditional=0.7, arrival_norm=0.7, completion=1.0, rejection=0.0),
    ]
    anwg = objective_summary(rows, ARRIVAL_NORMALIZED_OBJECTIVE)
    constrained = objective_summary(
        rows,
        CONSTRAINED_ARRIVAL_NORMALIZED_OBJECTIVE,
        config=ConstrainedRankingConfig(0.8, 0.2),
    )
    assert anwg["policy_win_distribution"] == {"edf": 1, "scorpio": 1}
    assert constrained["policy_win_distribution"] == {"edf": 1, "scorpio": 1}
    assert anwg["oracle_headroom"] == pytest.approx(0.1)
