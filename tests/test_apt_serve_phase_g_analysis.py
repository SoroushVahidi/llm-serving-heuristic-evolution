from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "analyze_apt_serve_phase_g",
    Path(__file__).parent.parent / "scripts" / "analyze_apt_serve_phase_g.py",
)
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules["analyze_apt_serve_phase_g"] = analysis
_SPEC.loader.exec_module(analysis)


def _cell(policy_label: str, policy_kind: str, transition_cost: str, anwg: float, *, completed: int = 4) -> dict:
    return {
        "policy_label": policy_label,
        "policy_kind": policy_kind,
        "transition_cost": transition_cost,
        "num_completed": completed,
        "num_dropped": 0,
        "num_total": completed,
        "completion_fraction": 1.0,
        "mean_latency": 1.0,
        "p95_latency": 1.0,
        "mean_ttft": 1.0,
        "p95_ttft": 1.0,
        "slo_violation_rate": 0.25,
        "weighted_goodput_completed_only": anwg,
        "arrival_normalized_weighted_goodput": anwg,
        "request_throughput": 1.0,
        "token_throughput": 10.0,
    }


def _record(stage: str, regime_id: str, seed: int, apt_anwg: float, best_baseline_anwg: float) -> dict:
    cells = []
    for idx, policy in enumerate(analysis.EXPECTED_BASELINES):
        value = best_baseline_anwg if idx == 0 else max(0.0, best_baseline_anwg - 0.1)
        cells.append(_cell(policy, "baseline", "na", value))
    for transition_cost in analysis.EXPECTED_TRANSITION_COSTS:
        cell = _cell("apt_serve_faithful", "apt_serve", transition_cost, apt_anwg)
        cell["apt_stats"] = {
            "kv_to_hidden_transitions": 2,
            "hidden_to_kv_transitions": 1,
            "evictions": 0,
            "recomputations": 0,
            "switch_latency_paid": 0.01,
            "restore_latency_paid": 0.02,
        }
        cells.append(cell)
    return {
        "stage": stage,
        "regime_id": regime_id,
        "seed": seed,
        "regime": {
            "regime_id": regime_id,
            "kv_pressure": "high",
            "slo_pattern": "bimodal",
            "length_pattern": "bimodal",
            "arrival_pattern": "steady",
            "cache_use_structure": "none",
            "n_requests": 4,
        },
        "n_requests": 4,
        "cells": cells,
        "failures": [],
        "critical_failure": None,
        "wall_time_sec": 0.1,
        "completed_at": 1.0,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_validate_dataset_accepts_complete_phase_g_shape():
    records = [
        _record("screening", "r1", 1001, 0.7, 0.6),
        _record("screening", "r1", 1002, 0.5, 0.6),
        _record("confirmation", "r1", 2001, 0.8, 0.6),
    ]
    result = analysis.validate_dataset(records)
    assert result["valid"] is True
    assert result["classification"] == "STRUCTURALLY_VALID"
    assert result["total_units"] == 3
    assert result["malformed_units"] == 0


def test_unit_summaries_compute_gap_and_marginal_contribution():
    records = [_record("screening", "r1", 1001, 0.7, 0.6)]
    summaries = analysis.build_unit_summaries(records)
    assert len(summaries) == 1
    assert summaries[0].best_baseline_policy == analysis.EXPECTED_BASELINES[0]
    assert summaries[0].gap_vs_best_baseline == 0.09999999999999998
    assert summaries[0].marginal_contribution == 0.09999999999999998
    assert summaries[0].total_transitions == 3


def test_grouped_bootstrap_is_deterministic_for_fixed_seed():
    records = [
        _record("screening", "r1", 1001, 0.7, 0.6),
        _record("screening", "r2", 1001, 0.4, 0.6),
    ]
    units = analysis.build_unit_summaries(records)
    a = analysis.grouped_bootstrap_results(units, n_bootstrap=25, seed=123)
    b = analysis.grouped_bootstrap_results(units, n_bootstrap=25, seed=123)
    assert a == b
    assert {row["metric"] for row in a} == {
        "apt_gap_vs_best_baseline",
        "marginal_contribution",
        "apt_primary_anwg",
        "best_baseline_anwg",
    }


def test_run_analysis_writes_incremental_outputs(tmp_path):
    dataset = tmp_path / "dataset"
    out = tmp_path / "analysis"
    dataset.mkdir()
    _write_jsonl(
        dataset / "results.jsonl",
        [
            _record("screening", "r1", 1001, 0.7, 0.6),
            _record("screening", "r1", 1002, 0.5, 0.6),
            _record("confirmation", "r1", 2001, 0.8, 0.6),
        ],
    )
    rc = analysis.main([
        "--dataset", str(dataset),
        "--output-dir", str(out),
        "--n-bootstrap", "10",
        "--seed", "7",
    ])
    assert rc == 0
    assert json.loads((out / "dataset_validation.json").read_text())["valid"] is True
    assert (out / "global_policy_summary.csv").exists()
    assert (out / "grouped_bootstrap_results.csv").exists()
    assert json.loads((out / "final_summary.json").read_text())["status"] == "COMPLETE"
    progress = json.loads((out / "progress.json").read_text())
    assert progress["current_stage"] == "complete"
