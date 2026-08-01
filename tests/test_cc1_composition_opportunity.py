from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from llmserveopt.core.metrics import RunMetrics
from llmserveopt.experiments.cc1_composition_opportunity import (
    CC1Error,
    build_workload_windows,
    determine_verdict,
    load_config,
    normalize_weights,
    planned_runs,
    run_experiment,
    simplex_weight_grid,
    summarize_methods,
    summarize_per_window,
)


ROOT = Path(__file__).resolve().parents[1]


def tiny_config(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "mode": "smoke",
        "seed": 7,
        "policy_subset": ["weighted_shortest_processing", "edf"],
        "composition": {
            "operator": "weighted_borda_rank_aggregation",
            "implementation": "StaticRankEnsemblePolicy",
            "method": "borda",
            "normalization": "per_state_normalized_rank",
            "top_k": 2,
            "weight_grid_step": 0.5,
        },
        "metrics": {
            "primary": "arrival_normalized_weighted_goodput",
            "completion_fraction_tolerance": 0.005,
        },
        "thresholds": {
            "aggregate_anwg_gain": 0.005,
            "regime_specific_gain": 0.01,
            "stop_non_near_tie_gap": 0.002,
        },
        "near_tie_primary_threshold": 0.005,
        "near_tie_thresholds": [0.001, 0.005, 0.01],
        "development_splits": ["TRAIN"],
        "evaluation_splits": ["ID_TEST"],
        "safeguards": {"max_runs": 20, "forbid_live_api": True, "forbid_gpu": True, "forbid_real_vllm": True},
        "outputs": {"root": str(tmp_path), "decision_traces": False},
        "service_model": {"step_size": 0.001},
        "simulator": {"drain_steps": 1000},
        "gpus": [{"gpu_id": 0, "max_active_sequences": 2, "max_batch_tokens": 32, "max_kv_tokens": 512}],
        "workloads": [
            {
                "tag": "tiny_train",
                "kind": "synthetic",
                "split": "TRAIN",
                "regime": "tiny",
                "seed": 1,
                "max_requests": 4,
                "arrival_process": "poisson",
                "arrival_rate": 2.0,
                "duration": 2.0,
                "prompt_mean": 16.0,
                "prompt_sigma": 0.1,
                "output_mean": 6.0,
                "output_sigma": 0.1,
                "prediction_noise_rel": 0.0,
            },
            {
                "tag": "tiny_test",
                "kind": "synthetic",
                "split": "ID_TEST",
                "regime": "tiny",
                "seed": 2,
                "max_requests": 4,
                "arrival_process": "poisson",
                "arrival_rate": 2.0,
                "duration": 2.0,
                "prompt_mean": 16.0,
                "prompt_sigma": 0.1,
                "output_mean": 6.0,
                "output_sigma": 0.1,
                "prediction_noise_rel": 0.0,
            },
        ],
    }


def fake_metrics(policy_name: str, workload_tag: str, seed: int, *, anwg: float = 0.5, completion: float = 1.0) -> RunMetrics:
    m = RunMetrics(policy_name=policy_name, workload_tag=workload_tag, seed=seed)
    m.num_total = 4
    m.num_completed = int(round(4 * completion))
    m.completion_fraction = completion
    m.weighted_goodput = anwg
    m.arrival_normalized_weighted_goodput = anwg
    m.weighted_completion_fraction = completion
    return m


def test_weight_validation_and_normalization():
    assert normalize_weights({"b": 1, "a": 3}) == {"a": pytest.approx(0.75), "b": pytest.approx(0.25)}
    with pytest.raises(CC1Error):
        normalize_weights({"a": -1})
    with pytest.raises(CC1Error):
        normalize_weights({"a": 0})


def test_weight_grid_is_sparse_normalized_and_deterministic():
    grid1 = simplex_weight_grid(["a", "b", "c"], step=0.5, top_k=2)
    grid2 = simplex_weight_grid(["a", "b", "c"], step=0.5, top_k=2)
    assert [g.mixture_id for g in grid1] == [g.mixture_id for g in grid2]
    assert len(grid1) == 6
    assert all(sum(g.weights.values()) == pytest.approx(1.0) for g in grid1)
    assert all(len(g.weights) <= 2 for g in grid1)


def test_true_simulator_execution_and_no_reward_vector_interpolation(tmp_path):
    config = tiny_config(tmp_path)
    calls = []

    def runner(*, policy, requests, gpu_configs, service_model, workload_tag, seed, drain_steps):
        calls.append(getattr(policy, "name", "unknown"))
        return fake_metrics(getattr(policy, "name", "unknown"), workload_tag, seed)

    result = run_experiment(
        config,
        config_path="unit.yaml",
        timestamp="unit_true_exec",
        allow_dirty=True,
        runner=runner,
    )
    rows = list((result.output_dir / "policy_execution_rows.csv").read_text().splitlines())
    assert len(calls) == len(planned_runs(config))
    assert "true_simulator_executed" in rows[0]
    assert "reward_vector_interpolated" in rows[0]
    assert result.verdict in {"STOP_OR_REDESIGN", "INCONCLUSIVE", "PROCEED", "REGIME_SPECIFIC_ONLY"}


def test_actual_smoke_uses_simulator(tmp_path):
    config = tiny_config(tmp_path)
    result = run_experiment(config, config_path="unit.yaml", timestamp="unit_actual", allow_dirty=True)
    assert (result.output_dir / "manifest.json").exists()
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["planned_run_count"] == len(planned_runs(config))
    assert manifest["no_live_api"] is True


def test_oracle_fixed_and_mixture_calculation():
    rows = [
        row("w1", "ID_TEST", "r", "fixed__a", "fixed_policy", 0.7, 1.0),
        row("w1", "ID_TEST", "r", "fixed__b", "fixed_policy", 0.6, 1.0),
        row("w1", "ID_TEST", "r", "mix__x", "weighted_borda_mixture", 0.72, 1.0),
        row("w1", "ID_TEST", "r", "mix__y", "weighted_borda_mixture", 0.71, 1.0),
    ]
    summary = summarize_per_window(rows, {"near_tie_primary_threshold": 0.005})
    assert summary[0]["oracle_fixed_treatment_id"] == "fixed__a"
    assert summary[0]["oracle_mixture_treatment_id"] == "mix__x"
    assert summary[0]["composition_opportunity_gap"] == pytest.approx(0.02)


def test_near_tie_filtering():
    rows = [
        row("w1", "ID_TEST", "r", "fixed__a", "fixed_policy", 0.700, 1.0),
        row("w1", "ID_TEST", "r", "fixed__b", "fixed_policy", 0.697, 1.0),
        row("w1", "ID_TEST", "r", "mix__x", "weighted_borda_mixture", 0.702, 1.0),
    ]
    summary = summarize_per_window(rows, {"near_tie_primary_threshold": 0.005})
    assert summary[0]["near_tie"] is True


def test_completion_loss_gate_and_verdict_logic():
    per_window = [
        {
            "window_id": "w1",
            "split": "ID_TEST",
            "regime": "r",
            "source": "synthetic",
            "composition_opportunity_gap": 0.01,
            "near_tie": False,
            "oracle_fixed_anwg": 0.7,
            "oracle_mixture_anwg": 0.71,
            "oracle_fixed_completion_fraction": 1.0,
            "oracle_mixture_completion_fraction": 0.8,
        }
    ]
    method_rows = [
        {"method_id": "best_fixed_policy", "mean_completion_fraction": 1.0},
        {"method_id": "oracle_best_fixed_per_window", "mean_completion_fraction": 1.0, "mean_anwg": 0.7},
        {"method_id": "oracle_best_mixture_per_window", "mean_completion_fraction": 0.8, "mean_anwg": 0.71},
    ]
    verdict = determine_verdict(per_window, method_rows, [], tiny_config(Path("/tmp")))
    assert verdict["verdict"] == "STOP_OR_REDESIGN"
    assert verdict["reason"] == "completion-fraction constraint failed"


def test_proceed_and_regime_specific_verdicts():
    config = tiny_config(Path("/tmp"))
    per_window = [
        {
            "window_id": "w1",
            "split": "ID_TEST",
            "regime": "r",
            "source": "synthetic",
            "composition_opportunity_gap": 0.006,
            "near_tie": False,
            "oracle_fixed_anwg": 0.7,
            "oracle_mixture_anwg": 0.706,
            "oracle_fixed_completion_fraction": 1.0,
            "oracle_mixture_completion_fraction": 1.0,
        }
    ]
    methods = [
        {"method_id": "best_fixed_policy", "mean_completion_fraction": 1.0},
        {"method_id": "oracle_best_fixed_per_window", "mean_completion_fraction": 1.0, "mean_anwg": 0.7},
        {"method_id": "oracle_best_mixture_per_window", "mean_completion_fraction": 1.0, "mean_anwg": 0.706},
    ]
    assert determine_verdict(per_window, methods, [], config)["verdict"] == "PROCEED"
    per_window[0]["composition_opportunity_gap"] = 0.003
    subset = [{"subset": "regime:r", "mean_opportunity_gap": 0.011}]
    assert determine_verdict(per_window, methods, subset, config)["verdict"] == "REGIME_SPECIFIC_ONLY"


def test_smoke_cli_dry_run():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_cc1_composition_opportunity.py",
            "--config",
            "configs/cc1_composition_opportunity_smoke.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["manifest"]["dry_run"] is True
    assert payload["manifest"]["planned_run_count"] <= 50


def test_missing_trace_behavior_skips_when_optional(tmp_path):
    config = tiny_config(tmp_path)
    config["workloads"].append({
        "tag": "missing_trace",
        "kind": "real_trace",
        "split": "OOD_TEST",
        "regime": "missing",
        "seed": 3,
        "max_requests": 4,
        "path": "data/processed/does_not_exist.jsonl",
    })
    windows, skipped = build_workload_windows(config)
    assert windows
    assert skipped[0]["reason"] == "missing local trace data"


def test_reproducibility_with_fixed_seed_and_timestamp(tmp_path):
    c1 = tiny_config(tmp_path / "a")
    c2 = tiny_config(tmp_path / "b")
    r1 = run_experiment(c1, config_path="unit.yaml", timestamp="same", allow_dirty=True)
    r2 = run_experiment(c2, config_path="unit.yaml", timestamp="same", allow_dirty=True)
    assert r1.verdict == r2.verdict
    assert (r1.output_dir / "method_comparison.csv").read_text() == (r2.output_dir / "method_comparison.csv").read_text()


def row(window_id: str, split: str, regime: str, treatment_id: str, kind: str, anwg: float, completion: float) -> dict:
    return {
        "window_id": window_id,
        "split": split,
        "regime": regime,
        "source": "synthetic",
        "treatment_id": treatment_id,
        "treatment_kind": kind,
        "metric_arrival_normalized_weighted_goodput": anwg,
        "metric_completion_fraction": completion,
        "metric_weighted_goodput": anwg,
        "metric_num_total": 4,
    }

