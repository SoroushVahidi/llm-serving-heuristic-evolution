"""Tests for window label construction."""
import math
import pytest

from llmserveopt.core.metrics import RunMetrics
from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.labels import label_window, label_windows


def _metrics(policy: str, weighted_goodput: float, slo_rate: float = 0.1,
             p95_ttft: float = 1.0, p95_latency: float = 2.0,
             throughput: float = 10.0) -> RunMetrics:
    m = RunMetrics(policy_name=policy, workload_tag="test", seed=0)
    m.weighted_goodput = weighted_goodput
    m.slo_violation_rate = slo_rate
    m.p95_ttft = p95_ttft
    m.p95_latency = p95_latency
    m.request_throughput = throughput
    m.num_completed = 100
    return m


def _candidate_metrics(best: str, best_wg: float = 0.9, other_wg: float = 0.7) -> dict:
    """Build metrics dict with best having best_wg and others having other_wg."""
    result = {}
    for name in SELECTOR_CANDIDATES:
        wg = best_wg if name == best else other_wg
        result[name] = _metrics(name, weighted_goodput=wg)
    return result


# --- correct winner ---

def test_best_policy_is_highest_weighted_goodput():
    m = _candidate_metrics("fifo", best_wg=0.95, other_wg=0.5)
    lbl = label_window(m)
    assert lbl.best_policy == "fifo"
    assert lbl.best_weighted_goodput == pytest.approx(0.95)


def test_reward_vector_complete():
    m = _candidate_metrics("edf", best_wg=0.9)
    lbl = label_window(m)
    for name in SELECTOR_CANDIDATES:
        assert name in lbl.reward_vector, f"Missing reward for {name}"


def test_reward_vector_values_match():
    m = _candidate_metrics("slo_slack_score", best_wg=0.88, other_wg=0.6)
    lbl = label_window(m)
    assert lbl.reward_vector["slo_slack_score"] == pytest.approx(0.88)
    for name in SELECTOR_CANDIDATES:
        if name != "slo_slack_score":
            assert lbl.reward_vector[name] == pytest.approx(0.6)


# --- tie-breaking ---

def test_tie_breaking_by_slo_violation_rate():
    m = {}
    for name in SELECTOR_CANDIDATES:
        m[name] = _metrics(name, weighted_goodput=0.8, slo_rate=0.5)
    # Give "edf" a lower slo_violation_rate
    m["edf"] = _metrics("edf", weighted_goodput=0.8, slo_rate=0.1)
    lbl = label_window(m)
    assert lbl.best_policy == "edf"


def test_tie_breaking_by_alphabetical():
    m = {}
    for name in SELECTOR_CANDIDATES:
        m[name] = _metrics(name, weighted_goodput=0.8, slo_rate=0.2, p95_ttft=1.0,
                           p95_latency=2.0, throughput=10.0)
    lbl = label_window(m)
    # Alphabetically first among candidates
    alpha_first = sorted(SELECTOR_CANDIDATES)[0]
    assert lbl.best_policy == alpha_first


# --- oracle excluded ---

def test_oracle_not_in_best_policy():
    m = _candidate_metrics("fifo", best_wg=0.5)
    # Add oracle with very high weighted_goodput
    oracle_m = _metrics("oracle_srtf", weighted_goodput=0.99)
    m["oracle_srtf"] = oracle_m
    lbl = label_window(m)
    assert lbl.best_policy != "oracle_srtf"
    assert lbl.best_policy in SELECTOR_CANDIDATES


def test_oracle_appears_in_oracle_column():
    m = _candidate_metrics("fifo", best_wg=0.5)
    m["oracle_srtf"] = _metrics("oracle_srtf", weighted_goodput=0.99)
    lbl = label_window(m)
    assert lbl.oracle_weighted_goodput == pytest.approx(0.99)


# --- second best and margin ---

def test_second_best_policy():
    m = _candidate_metrics("edf", best_wg=0.9)
    # Set edf=0.9, best_fit=0.85, rest=0.5
    m["best_fit"] = _metrics("best_fit", weighted_goodput=0.85)
    lbl = label_window(m)
    assert lbl.best_policy == "edf"
    assert lbl.second_best_policy == "best_fit"
    assert lbl.policy_margin == pytest.approx(0.9 - 0.85, abs=1e-6)


# --- missing metrics error ---

def test_no_candidates_raises():
    with pytest.raises(ValueError, match="No deployable candidate"):
        label_window({"oracle_srtf": _metrics("oracle_srtf", 0.9)})


# --- label_windows: regret_to_best_fixed ---

def test_regret_to_best_fixed_nonnegative():
    windows_metrics = [_candidate_metrics("fifo", best_wg=0.9, other_wg=0.5) for _ in range(3)]
    labels = label_windows(windows_metrics)
    for lbl in labels:
        assert not math.isnan(lbl.regret_to_best_fixed)
        assert lbl.regret_to_best_fixed >= -1e-9   # best per-window >= best fixed


def test_regret_zero_when_same_winner():
    # All windows have same winner with same value → regret = 0
    windows_metrics = []
    for _ in range(5):
        m = {}
        for name in SELECTOR_CANDIDATES:
            m[name] = _metrics(name, weighted_goodput=0.9 if name == "fifo" else 0.5)
        windows_metrics.append(m)
    labels = label_windows(windows_metrics)
    for lbl in labels:
        assert lbl.regret_to_best_fixed == pytest.approx(0.0, abs=1e-6)
