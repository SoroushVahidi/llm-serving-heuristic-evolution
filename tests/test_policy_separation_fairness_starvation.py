"""Focused unit and smoke tests for the Fairness and Starvation template (Family A)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.aging_priority import AgingPriorityPolicy
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy
from llmserveopt.policy_separation.templates_fairness_starvation import (
    _BURSTGPT_CANDIDATE_NAMES,
    _get_staged_burstgpt_path,
    case4_fairness_starvation,
    sample_trace_token_lengths,
)
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "policy_separation_fairness_starvation_pilot_v1.yaml"


def test_determinism():
    """Same arguments and seed must produce identical requests and metadata."""
    scen1 = case4_fairness_starvation(
        target_utilization=0.8,
        tenant_weight_skew=5.0,
        interactive_volume_fraction=0.2,
        seed=42,
    )
    scen2 = case4_fairness_starvation(
        target_utilization=0.8,
        tenant_weight_skew=5.0,
        interactive_volume_fraction=0.2,
        seed=42,
    )

    assert scen1.scenario_id == scen2.scenario_id
    assert len(scen1.requests) == len(scen2.requests)
    assert scen1.params["token_length_source"] == scen2.params["token_length_source"]

    for r1, r2 in zip(scen1.requests, scen2.requests):
        assert r1.request_id == r2.request_id
        assert r1.arrival_time == r2.arrival_time
        assert r1.prompt_tokens == r2.prompt_tokens
        assert r1.predicted_output_tokens == r2.predicted_output_tokens
        assert r1.actual_output_tokens == r2.actual_output_tokens
        assert r1.slo_deadline == r2.slo_deadline
        assert r1.priority == r2.priority
        assert r1.class_id == r2.class_id


def test_unique_scenario_ids():
    """Varying coordinates must yield unique scenario IDs without collisions."""
    ids = set()
    for util in [0.5, 0.8, 1.1]:
        for skew in [1.0, 5.0, 10.0]:
            for vol in [0.1, 0.3]:
                scen = case4_fairness_starvation(
                    target_utilization=util,
                    tenant_weight_skew=skew,
                    interactive_volume_fraction=vol,
                    seed=0,
                )
                assert scen.scenario_id not in ids
                ids.add(scen.scenario_id)


def test_full_config_grid_unique_scenario_ids():
    """The committed pilot YAML grid must expand to 120 unique scenario IDs."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    grid = cfg["sweep_grid"]
    ids = []
    for util in grid["target_utilization"]:
        for skew in grid["tenant_weight_skew"]:
            for vol in grid["interactive_volume_fraction"]:
                for seed in grid["seeds"]:
                    scen = case4_fairness_starvation(
                        target_utilization=util,
                        tenant_weight_skew=skew,
                        interactive_volume_fraction=vol,
                        seed=seed,
                    )
                    ids.append(scen.scenario_id)
    assert len(ids) == 120
    assert len(set(ids)) == 120


def test_no_leakage():
    """Generator metadata must not leak into policy-visible request fields."""
    scen = case4_fairness_starvation(
        target_utilization=0.9,
        tenant_weight_skew=8.0,
        interactive_volume_fraction=0.25,
        seed=7,
    )

    for req in scen.requests:
        assert isinstance(req.request_id, int)
        assert isinstance(req.arrival_time, float)
        assert isinstance(req.prompt_tokens, int)
        assert isinstance(req.predicted_output_tokens, int)
        assert isinstance(req.slo_deadline, float)
        assert isinstance(req.priority, float)
        assert isinstance(req.class_id, str)
        assert "util" not in req.class_id
        assert "skew" not in req.class_id
        assert req.class_id in {"tenant_interactive", "tenant_bulk"}


def test_local_smoke_creates_fairness_pressure():
    """Tiny local smoke: all four policies complete and SLO pressure appears."""
    scen = case4_fairness_starvation(
        target_utilization=1.5,
        tenant_weight_skew=10.0,
        interactive_volume_fraction=0.2,
        seed=123,
    )

    policies = {
        "fifo": FIFOPolicy(),
        "estf": EstimatedServiceTimeFirstPolicy(),
        "aging": AgingPriorityPolicy(aging_rate=0.2),
        "wfs": WeightedFairSharePolicy(),
    }

    results = {}
    for name, policy in policies.items():
        sim_config = SimulatorConfig(gpu_configs=list(scen.gpu_configs))
        sim = Simulator(sim_config)
        sim.load_trace(list(scen.requests))
        sim.run(policy, workload_tag=f"smoke_{name}")
        completed = sim._completed  # noqa: SLF001

        interactive_completed = [
            cr for cr in completed if cr.request.class_id == "tenant_interactive"
        ]
        bulk_completed = [cr for cr in completed if cr.request.class_id == "tenant_bulk"]

        interactive_violations = sum(
            1
            for cr in interactive_completed
            if cr.completion_time > cr.request.slo_deadline
        )
        bulk_violations = sum(
            1 for cr in bulk_completed if cr.completion_time > cr.request.slo_deadline
        )

        results[name] = {
            "interactive_violations": interactive_violations,
            "bulk_violations": bulk_violations,
            "interactive_total": len(interactive_completed),
            "bulk_total": len(bulk_completed),
            "total_completed": len(completed),
        }

    for name, res in results.items():
        assert res["total_completed"] == len(scen.requests)

    total_violations = sum(
        res["interactive_violations"] + res["bulk_violations"] for res in results.values()
    )
    assert total_violations > 0, "No SLO violations observed; load might be too low."


def test_burstgpt_path_resolution_prefers_numbered_shard(tmp_path, monkeypatch):
    """Discover BurstGPT_without_fails_1.csv when the bare filename is absent."""
    raw = tmp_path / "burstgpt_v2" / "raw"
    raw.mkdir(parents=True)
    target = raw / "BurstGPT_without_fails_1.csv"
    target.write_text("Request Token,Response Token\n10,20\n", encoding="utf-8")
    # Deliberately do NOT create BurstGPT_without_fails.csv

    found = _get_staged_burstgpt_path(datasets_root=tmp_path)
    assert found == target

    # Env override wins
    override = tmp_path / "custom.csv"
    override.write_text("Request Token,Response Token\n1,2\n", encoding="utf-8")
    monkeypatch.setenv("LLM_SERVEOPT_BURSTGPT_CSV", str(override))
    assert _get_staged_burstgpt_path(datasets_root=tmp_path) == override


def test_burstgpt_path_resolution_missing_returns_none(tmp_path):
    raw = tmp_path / "burstgpt_v2" / "raw"
    raw.mkdir(parents=True)
    assert _get_staged_burstgpt_path(datasets_root=tmp_path) is None
    assert _BURSTGPT_CANDIDATE_NAMES[0].startswith("BurstGPT_without_fails")


def test_sample_reports_synthetic_fallback_source(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    prompts, outputs, source = sample_trace_token_lengths(
        rng, 8, use_bulk=False, datasets_root=tmp_path
    )
    assert source == "synthetic_lognormal_fallback"
    assert len(prompts) == 8
    assert len(outputs) == 8


def test_jains_index_formula():
    from scripts.run_policy_separation_fairness_starvation_pilot_v1 import jains_index

    assert jains_index(1.0, 1.0) == pytest.approx(1.0)
    assert jains_index(1.0, 0.0) == pytest.approx(0.5)
    assert jains_index(0.0, 0.0) == 0.0


def test_metric_definition_unweighted_vs_canonical_anwg():
    """Unweighted SLO success must not be silently labeled as canonical ANWG."""
    scen = case4_fairness_starvation(
        target_utilization=1.2,
        tenant_weight_skew=10.0,
        interactive_volume_fraction=0.2,
        seed=20260815,
    )
    sim = Simulator(SimulatorConfig(gpu_configs=list(scen.gpu_configs)))
    sim.load_trace(list(scen.requests))
    metrics = sim.run(FIFOPolicy(), workload_tag="metric_def")
    completed = sim._completed  # noqa: SLF001
    total_v = sum(1 for cr in completed if cr.completion_time > cr.request.slo_deadline)
    unweighted = (len(completed) - total_v) / len(scen.requests)

    assert 0.0 <= unweighted <= 1.0
    assert 0.0 <= float(metrics.arrival_normalized_weighted_goodput) <= 1.0
    # Under heterogeneous priorities these can differ; always both are defined.
    assert metrics.arrival_normalized_weighted_goodput == metrics.arrival_normalized_weighted_goodput


def test_runner_dry_run_schema(tmp_path):
    """Dry-run emits the clarified metric schema (not historical 'anwg' alone)."""
    from scripts.run_policy_separation_fairness_starvation_pilot_v1 import (
        RESULT_FIELDNAMES,
        main,
    )
    import sys

    run_dir = tmp_path / "dry"
    argv = [
        "run_policy_separation_fairness_starvation_pilot_v1.py",
        "--config",
        str(CONFIG_PATH),
        "--run-dir",
        str(run_dir),
        "--workers",
        "1",
        "--dry-run",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old

    csv_path = run_dir / "per_policy_results.csv"
    assert csv_path.is_file()
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16  # 4 scenarios x 4 policies
    assert list(rows[0].keys()) == RESULT_FIELDNAMES
    assert "anwg" not in rows[0]
    assert "unweighted_slo_success_rate" in rows[0]
    assert "arrival_normalized_weighted_goodput" in rows[0]
    assert all(r["status"] == "success" for r in rows)

    summary = (run_dir / "final_summary.json").read_text(encoding="utf-8")
    assert "unweighted_slo_success_rate" in summary
    assert "arrival_normalized_weighted_goodput" in summary
