"""Tests for Family A v2 fairness-vs-size generator and runner schema."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from llmserveopt.policies.aging_priority import AgingPriorityPolicy
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (
    BurstGPTUnavailableError,
    apply_prediction_noise,
    assert_size_priority_orthogonality,
    case_fairness_vs_size_v2,
)
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "policy_separation_fairness_starvation_pilot_v2.yaml"


def _scen(**kwargs):
    defaults = dict(
        target_utilization=1.15,
        tenant_weight_skew=5.0,
        favored_tenant_size="long",
        prediction_noise_sigma=0.3,
        seed=42,
        allow_synthetic_tokens=True,
    )
    defaults.update(kwargs)
    return case_fairness_vs_size_v2(**defaults)


def test_determinism_synthetic():
    a = _scen()
    b = _scen()
    assert a.scenario_id == b.scenario_id
    assert len(a.requests) == len(b.requests)
    for r1, r2 in zip(a.requests, b.requests):
        assert r1.arrival_time == r2.arrival_time
        assert r1.prompt_tokens == r2.prompt_tokens
        assert r1.actual_output_tokens == r2.actual_output_tokens
        assert r1.predicted_output_tokens == r2.predicted_output_tokens
        assert r1.priority == r2.priority
        assert r1.class_id == r2.class_id


def test_full_config_grid_unique_ids():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    grid = cfg["sweep_grid"]
    fixed = cfg["fixed"]
    ids = []
    for fav in grid["favored_tenant_size"]:
        for skew in grid["tenant_weight_skew"]:
            for util in grid["target_utilization"]:
                for noise in grid["prediction_noise_sigma"]:
                    for seed in grid["seeds"]:
                        s = case_fairness_vs_size_v2(
                            target_utilization=util,
                            tenant_weight_skew=skew,
                            favored_tenant_size=fav,
                            prediction_noise_sigma=noise,
                            seed=seed,
                            n_total_jobs=fixed["n_total_jobs"],
                            max_active_sequences=fixed["max_active_sequences"],
                            favored_slo_slack_s=fixed["favored_slo_slack_s"],
                            other_slo_slack_s=fixed["other_slo_slack_s"],
                            allow_synthetic_tokens=True,
                        )
                        ids.append(s.scenario_id)
    assert len(ids) == 72
    assert len(set(ids)) == 72


def test_size_priority_orthogonality_both_treatments():
    for fav in ("short", "long"):
        s = _scen(favored_tenant_size=fav, prediction_noise_sigma=0.0)
        assert_size_priority_orthogonality(s)
        fav_pri = {r.priority for r in s.requests if r.class_id == "tenant_favored"}
        oth_pri = {r.priority for r in s.requests if r.class_id == "tenant_other"}
        assert fav_pri == {5.0}
        assert oth_pri == {1.0}


def test_no_generator_label_leakage_in_class_id():
    s = _scen(favored_tenant_size="long", tenant_weight_skew=10.0)
    for r in s.requests:
        assert r.class_id in {"tenant_favored", "tenant_other"}
        assert "util" not in r.class_id
        assert "skew" not in r.class_id
        assert "long" not in r.class_id
        assert "short" not in r.class_id


def test_production_fails_without_burstgpt(tmp_path):
    with pytest.raises(BurstGPTUnavailableError):
        case_fairness_vs_size_v2(
            target_utilization=1.0,
            tenant_weight_skew=5.0,
            favored_tenant_size="long",
            prediction_noise_sigma=0.0,
            seed=1,
            allow_synthetic_tokens=False,
            datasets_root=tmp_path,
        )


def test_burstgpt_request_tokens_plural_header(tmp_path, monkeypatch):
    """Staged BurstGPT v2 uses 'Request tokens' / 'Response tokens' (plural)."""
    from llmserveopt.policy_separation import templates_fairness_starvation_v2 as v2

    raw = tmp_path / "burstgpt_v2" / "raw"
    raw.mkdir(parents=True)
    csv_path = raw / "BurstGPT_without_fails_1.csv"
    lines = ["Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"]
    for i in range(200):
        req = 100 + (i % 50)
        resp = 20 + (i % 30)
        lines.append(f"{i},ChatGPT,{req},{resp},{req + resp},Conversation log\n")
    csv_path.write_text("".join(lines), encoding="utf-8")

    v2._load_burstgpt_arrays.cache_clear()
    monkeypatch.setenv("LLM_SERVEOPT_BURSTGPT_CSV", str(csv_path))
    # Also point discovery at tmp datasets root.
    s = case_fairness_vs_size_v2(
        target_utilization=1.0,
        tenant_weight_skew=5.0,
        favored_tenant_size="long",
        prediction_noise_sigma=0.0,
        seed=1,
        allow_synthetic_tokens=False,
        datasets_root=tmp_path,
    )
    assert s.params["token_length_source"] == "burstgpt_staged"
    assert s.params.get("burstgpt_path")
    assert Path(s.params["burstgpt_path"]).name == "BurstGPT_without_fails_1.csv"
    assert all(r.prompt_tokens > 0 for r in s.requests)


def test_prediction_noise_determinism_and_accurate_control():
    rng = np.random.default_rng(0)
    actual = np.array([10, 20, 30, 40])
    assert list(apply_prediction_noise(rng, actual, 0.0)) == list(actual)
    a = apply_prediction_noise(np.random.default_rng(7), actual, 0.3)
    b = apply_prediction_noise(np.random.default_rng(7), actual, 0.3)
    assert list(a) == list(b)
    assert list(a) != list(actual)


def test_priority_direction_favored_higher_when_skewed():
    s = _scen(tenant_weight_skew=10.0)
    for r in s.requests:
        if r.class_id == "tenant_favored":
            assert r.priority == 10.0
        else:
            assert r.priority == 1.0


def test_canonical_anwg_present_in_simulation():
    s = _scen(favored_tenant_size="long", prediction_noise_sigma=0.3, seed=99)
    sim = Simulator(SimulatorConfig(gpu_configs=list(s.gpu_configs)))
    sim.load_trace(list(s.requests))
    metrics = sim.run(WeightedFairSharePolicy(), workload_tag="v2_metric")
    assert metrics.arrival_normalized_weighted_goodput == metrics.arrival_normalized_weighted_goodput
    assert 0.0 <= float(metrics.arrival_normalized_weighted_goodput) <= 1.0


def test_runner_dry_run_schema_synthetic(tmp_path):
    import sys
    from scripts.run_policy_separation_fairness_starvation_pilot_v2 import (
        RESULT_FIELDNAMES,
        main,
    )

    run_dir = tmp_path / "dry"
    argv = [
        "run_policy_separation_fairness_starvation_pilot_v2.py",
        "--config",
        str(CONFIG),
        "--run-dir",
        str(run_dir),
        "--workers",
        "1",
        "--dry-run",
        "--allow-synthetic-tokens",
        "--no-require-burstgpt",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old

    with (run_dir / "per_policy_results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16
    assert list(rows[0].keys()) == RESULT_FIELDNAMES
    assert "anwg" not in rows[0]
    assert "arrival_normalized_weighted_goodput" in rows[0]
    assert "unweighted_slo_success_rate" in rows[0]
    assert (run_dir / "scenarios.jsonl").is_file()
    assert (run_dir / "scenario_features.csv").is_file()


def test_smoke_conflict_cell_estf_wfs_differ():
    """Conflict cell (favored=long, high skew) should make ESTF and WFS diverge."""
    s = _scen(
        favored_tenant_size="long",
        tenant_weight_skew=10.0,
        target_utilization=1.5,
        prediction_noise_sigma=0.3,
        seed=20260816,
        max_active_sequences=1,
        favored_slo_slack_s=1.0,
        other_slo_slack_s=8.0,
    )
    assert_size_priority_orthogonality(s)
    results = {}
    for name, policy in {
        "fifo": FIFOPolicy(),
        "estf": EstimatedServiceTimeFirstPolicy(),
        "aging": AgingPriorityPolicy(),
        "wfs": WeightedFairSharePolicy(),
    }.items():
        sim = Simulator(SimulatorConfig(gpu_configs=list(s.gpu_configs)))
        sim.load_trace(list(s.requests))
        m = sim.run(policy, workload_tag=name)
        completed = sim._completed  # noqa: SLF001
        fav = [c for c in completed if c.request.class_id == "tenant_favored"]
        fav_v = sum(1 for c in fav if c.completion_time > c.request.slo_deadline)
        results[name] = {
            "anwg": float(m.arrival_normalized_weighted_goodput),
            "fav_v": fav_v,
            "total_v": sum(
                1 for c in completed if c.completion_time > c.request.slo_deadline
            ),
        }

    assert results["estf"]["anwg"] != results["wfs"]["anwg"] or results["estf"][
        "fav_v"
    ] != results["wfs"]["fav_v"]
    assert sum(r["total_v"] for r in results.values()) > 0
    # Calibrated: Aging should not be forced to perfect success in this cell.
    assert results["aging"]["anwg"] < 1.0 - 1e-12 or results["aging"]["total_v"] > 0
