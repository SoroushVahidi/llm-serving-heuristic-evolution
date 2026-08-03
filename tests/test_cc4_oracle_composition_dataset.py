"""CC4: oracle composition dataset build tests.

Covers candidate generation determinism, DSL verification-before-execution,
resumable checkpointing, no reward-vector interpolation, oracle-label/
regret/near-tie/completion-constraint correctness, search-summary counts,
CloudRift clean-skip, split integrity (workload-window leakage check), and
reproducible shard merging across a real (small) simulator run.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from llmserveopt.core.metrics import RunMetrics
from llmserveopt.experiments import cc4_oracle_composition_dataset as cc4
from llmserveopt.experiments.cc1_composition_opportunity import CC1Error


def _tiny_config(tmp_root: str) -> dict:
    return {
        "schema_version": 1,
        "mode": "cc4",
        "seed": 1,
        "policy_subset": ["fifo", "edf"],
        "candidate_search": {
            "primitive_pool": [
                {"name": "laxity_urgency", "higher_is_preferred": True},
                {"name": "priority", "higher_is_preferred": True},
            ],
            "weight_grid_step": 0.5,
            "weight_grid_top_k": 2,
            "topk_mixture_k_values": [1],
            "admission_gate_laxity_thresholds": [0.0],
            "placement_key_variants": [[], ["projected_gpu_load"]],
        },
        "metrics": {"primary": cc4.PRIMARY, "completion_fraction_tolerance": 0.005},
        "near_tie_primary_threshold": 0.005,
        "near_tie_thresholds": [0.001, 0.005, 0.01],
        "development_splits": ["TRAIN"],
        "evaluation_splits": ["ID_TEST"],
        "safeguards": {"max_runs": 2000, "require_clean_git_for_full": True},
        "outputs": {"root": tmp_root},
        "service_model": {"step_size": 0.001},
        "simulator": {"drain_steps": 500},
        "gpus": [{"gpu_id": 0, "max_active_sequences": 4, "max_batch_tokens": 64, "max_kv_tokens": 1200}],
        "workloads": [
            {
                "tag": "tiny_train", "kind": "synthetic", "split": "TRAIN", "regime": "underloaded",
                "seed": 1, "max_requests": 12, "arrival_process": "poisson", "arrival_rate": 10.0,
                "duration": 1.0, "prompt_mean": 64.0, "prompt_sigma": 0.5, "prompt_low": 16, "prompt_high": 256,
                "output_mean": 32.0, "output_sigma": 0.5, "output_low": 8, "output_high": 128,
                "prediction_noise_rel": 0.1,
                "slo_classes": [{"class_id": "a", "slo_slack": 0.2, "priority": 1.0, "weight": 1.0}],
            },
            {
                "tag": "tiny_eval", "kind": "synthetic", "split": "ID_TEST", "regime": "underloaded",
                "seed": 2, "max_requests": 12, "arrival_process": "poisson", "arrival_rate": 10.0,
                "duration": 1.0, "prompt_mean": 64.0, "prompt_sigma": 0.5, "prompt_low": 16, "prompt_high": 256,
                "output_mean": 32.0, "output_sigma": 0.5, "output_low": 8, "output_high": 128,
                "prediction_noise_rel": 0.1,
                "slo_classes": [{"class_id": "a", "slo_slack": 0.2, "priority": 1.0, "weight": 1.0}],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def test_generate_all_candidates_deterministic_and_unique(tmp_path):
    config = _tiny_config(str(tmp_path))
    c1 = cc4.generate_all_candidates(config)
    c2 = cc4.generate_all_candidates(config)
    ids1 = [c.candidate_id for c in c1]
    ids2 = [c.candidate_id for c in c2]
    assert ids1 == ids2  # deterministic order and content
    assert len(ids1) == len(set(ids1))  # no duplicates


def test_candidate_families_present(tmp_path):
    config = _tiny_config(str(tmp_path))
    candidates = cc4.generate_all_candidates(config)
    families = {c.family for c in candidates}
    assert families == {
        "fixed_policy", "weighted_primitive_mixture", "sparse_topk_mixture",
        "admission_gate_variant", "placement_variant",
    }


def test_verify_candidate_accepts_valid_and_rejects_invalid(tmp_path):
    config = _tiny_config(str(tmp_path))
    candidates = cc4.generate_all_candidates(config)
    for c in candidates:
        ok, errors, _ = cc4.verify_candidate(c)
        assert ok, (c.candidate_id, errors)

    bad = cc4.Candidate(
        "bad_candidate", "weighted_primitive_mixture",
        heuristic_doc={
            "name": "bad_candidate", "tie_breaker": "arrival_order",
            "default": {"request_score": {"primitive": "does_not_exist"}},
        },
    )
    ok, errors, dsl_hash = cc4.verify_candidate(bad)
    assert not ok
    assert "PRIMITIVE_UNKNOWN" in errors
    assert dsl_hash == ""


def test_execute_candidate_row_skips_invalid_candidate_before_simulator_call(tmp_path):
    config = _tiny_config(str(tmp_path))
    windows, _ = cc4.build_workload_windows(config)
    calls = {"n": 0}

    def counting_runner(**kwargs):
        calls["n"] += 1
        return RunMetrics()

    bad = cc4.Candidate(
        "bad_candidate", "weighted_primitive_mixture",
        heuristic_doc={
            "name": "bad_candidate", "tie_breaker": "arrival_order",
            "default": {"request_score": {"primitive": "does_not_exist"}},
        },
    )
    row = cc4.execute_candidate_row(
        counting_runner, bad, windows[0], cc4.build_gpu_configs(config), cc4.build_service_model(config),
        drain_steps=100, git_sha="deadbeef",
    )
    assert calls["n"] == 0  # never reached the simulator
    assert row["true_simulator_executed"] is False
    assert row["verification_outcome"].startswith("invalid:")
    assert row["reward_vector_interpolated"] is False


# ---------------------------------------------------------------------------
# No reward-vector interpolation
# ---------------------------------------------------------------------------


def test_execute_candidate_row_uses_true_simulator_execution(tmp_path):
    config = _tiny_config(str(tmp_path))
    windows, _ = cc4.build_workload_windows(config)
    candidates = cc4.generate_all_candidates(config)
    fixed = next(c for c in candidates if c.family == "fixed_policy")
    row = cc4.execute_candidate_row(
        cc4.run_policy, fixed, windows[0], cc4.build_gpu_configs(config), cc4.build_service_model(config),
        drain_steps=200, git_sha="deadbeef",
    )
    assert row["true_simulator_executed"] is True
    assert row["reward_vector_interpolated"] is False
    assert cc4.PRIMARY_COL in row
    assert row["git_sha"] == "deadbeef"


# ---------------------------------------------------------------------------
# Split integrity (reuses CC1's build_workload_windows leakage check)
# ---------------------------------------------------------------------------


def test_cc4_config_windows_have_no_split_leakage(tmp_path):
    config = _tiny_config(str(tmp_path))
    windows, skipped = cc4.build_workload_windows(config)
    assert not skipped
    assert {w.window_id: w.split for w in windows} == {"tiny_train": "TRAIN", "tiny_eval": "ID_TEST"}


def test_full_cc4_config_builds_without_leakage():
    config = cc4.load_config("configs/cc4_oracle_composition_dataset.yaml")
    windows, skipped = cc4.build_workload_windows(config)
    assert len(windows) == 12
    assert not skipped  # both azure and burstgpt trace files are present locally
    splits = {w.split for w in windows}
    assert splits == {"TRAIN", "VALIDATION", "ID_TEST", "OOD_TEST"}


# ---------------------------------------------------------------------------
# Resumable trial store
# ---------------------------------------------------------------------------


def test_trial_store_resumes_and_skips_completed_keys(tmp_path):
    store1 = cc4.CC4TrialStore(tmp_path)
    assert store1.completed_keys == set()
    store1.append({"window_id": "w1", "candidate_id": "c1", "value": 1})
    store1.append({"window_id": "w1", "candidate_id": "c2", "value": 2})
    assert store1.completed_keys == {"w1::c1", "w1::c2"}

    store2 = cc4.CC4TrialStore(tmp_path)  # simulates a fresh process resuming
    assert store2.completed_keys == {"w1::c1", "w1::c2"}
    rows = store2.load_all_rows()
    assert len(rows) == 2
    assert {r["value"] for r in rows} == {1, 2}


def test_reproducible_shard_merging_via_resume(tmp_path):
    """Run a tiny search, interrupt conceptually by re-invoking run_search
    with resume_dir pointed at the same output_dir, and confirm the second
    call performs zero additional simulator executions (full resumability)."""
    config = _tiny_config(str(tmp_path / "out"))
    call_count = {"n": 0}

    def counting_runner(**kwargs):
        call_count["n"] += 1
        return cc4.run_policy(**kwargs)

    out_dir = tmp_path / "out" / "run1"
    result1 = cc4.run_search(
        config, config_path="tiny.yaml", full_run=True, allow_dirty=True,
        timestamp="run1", runner=counting_runner,
    )
    first_calls = call_count["n"]
    assert first_calls > 0
    n_rows_1 = len(pd.read_parquet(result1.output_dir / "per_window_results.parquet"))

    result2 = cc4.run_search(
        config, config_path="tiny.yaml", full_run=True, allow_dirty=True,
        resume_dir=result1.output_dir, runner=counting_runner,
    )
    assert call_count["n"] == first_calls  # no new simulator executions on resume
    n_rows_2 = len(pd.read_parquet(result2.output_dir / "per_window_results.parquet"))
    assert n_rows_2 == n_rows_1  # merged rows identical, not duplicated


# ---------------------------------------------------------------------------
# Oracle labels / regret / near-tie / completion constraints
# ---------------------------------------------------------------------------


def _rows_df(records):
    return pd.DataFrame(records)


def test_compute_oracle_labels_correctness():
    rows = _rows_df([
        {"window_id": "w1", "split": "ID_TEST", "regime": "r", "source": "synthetic",
         "candidate_id": "a", "family": "fixed_policy", "true_simulator_executed": True,
         cc4.PRIMARY_COL: 0.5, cc4.COMPLETION_COL: 0.9},
        {"window_id": "w1", "split": "ID_TEST", "regime": "r", "source": "synthetic",
         "candidate_id": "b", "family": "weighted_primitive_mixture", "true_simulator_executed": True,
         cc4.PRIMARY_COL: 0.8, cc4.COMPLETION_COL: 0.95},
        {"window_id": "w1", "split": "ID_TEST", "regime": "r", "source": "synthetic",
         "candidate_id": "c", "family": "fixed_policy", "true_simulator_executed": True,
         cc4.PRIMARY_COL: 0.3, cc4.COMPLETION_COL: 0.85},
    ])
    oracle = cc4.compute_oracle_labels(rows)
    assert len(oracle) == 1
    row = oracle.iloc[0]
    assert row["oracle_candidate_id"] == "b"
    assert row["oracle_anwg"] == pytest.approx(0.8)
    assert row["top2_margin"] == pytest.approx(0.3)  # 0.8 - 0.5


def test_compute_regret_matrix_correctness():
    rows = _rows_df([
        {"window_id": "w1", "candidate_id": "a", "family": "fixed_policy", "true_simulator_executed": True, cc4.PRIMARY_COL: 0.5, cc4.COMPLETION_COL: 0.9, "split": "ID_TEST", "regime": "r", "source": "synthetic"},
        {"window_id": "w1", "candidate_id": "b", "family": "weighted_primitive_mixture", "true_simulator_executed": True, cc4.PRIMARY_COL: 0.8, cc4.COMPLETION_COL: 0.95, "split": "ID_TEST", "regime": "r", "source": "synthetic"},
    ])
    oracle = cc4.compute_oracle_labels(rows)
    regret = cc4.compute_regret_matrix(rows, oracle)
    by_candidate = regret.set_index("candidate_id")["regret"].to_dict()
    assert by_candidate["b"] == pytest.approx(0.0)
    assert by_candidate["a"] == pytest.approx(0.3)


def test_near_tie_flags_thresholds():
    oracle = pd.DataFrame([{"window_id": "w1", "top2_margin": 0.003}, {"window_id": "w2", "top2_margin": 0.02}])
    flags = cc4.compute_near_tie_flags(oracle, [0.001, 0.005, 0.01])
    w1_flags = flags[flags["window_id"] == "w1"].set_index("threshold")["near_tie"].to_dict()
    assert w1_flags[0.001] is False  # 0.003 >= 0.001
    assert w1_flags[0.005] is True   # 0.003 < 0.005
    w2_flags = flags[flags["window_id"] == "w2"].set_index("threshold")["near_tie"].to_dict()
    assert w2_flags[0.01] is False  # 0.02 >= 0.01


def test_completion_constraints_tolerance_boundary():
    rows = _rows_df([
        {"window_id": "w1", "candidate_id": "fixed__fifo", "family": "fixed_policy", "true_simulator_executed": True, cc4.PRIMARY_COL: 0.5, cc4.COMPLETION_COL: 0.90, "split": "ID_TEST", "regime": "r", "source": "synthetic"},
        {"window_id": "w1", "candidate_id": "mix", "family": "weighted_primitive_mixture", "true_simulator_executed": True, cc4.PRIMARY_COL: 0.8, cc4.COMPLETION_COL: 0.894, "split": "ID_TEST", "regime": "r", "source": "synthetic"},
    ])
    oracle = cc4.compute_oracle_labels(rows)
    constraints = cc4.compute_completion_constraints(rows, oracle, tolerance=0.005)
    row = constraints.iloc[0]
    assert row["completion_impact"] == pytest.approx(0.894 - 0.90, abs=1e-9)
    assert bool(row["completion_ok"]) is False  # impact (-0.006) worse than -tolerance (-0.005)


# ---------------------------------------------------------------------------
# Search summary / primitive usage
# ---------------------------------------------------------------------------


def test_search_summary_counts(tmp_path):
    config = _tiny_config(str(tmp_path))
    candidates = cc4.generate_all_candidates(config)
    windows, skipped = cc4.build_workload_windows(config)
    rows = _rows_df([
        {"window_id": w.window_id, "candidate_id": c.candidate_id, "family": c.family,
         "true_simulator_executed": True, "composition_hash": "h" if c.heuristic_doc else "",
         cc4.PRIMARY_COL: 0.5, cc4.COMPLETION_COL: 0.9, "split": w.split, "regime": w.regime, "source": w.source}
        for w in windows for c in candidates
    ])
    summary = cc4.compute_search_summary(candidates, [], rows, windows, skipped)
    row = summary.iloc[0]
    assert row["n_windows"] == len(windows)
    assert row["n_candidates_total"] == len(candidates)
    assert row["n_simulator_executions"] == len(windows) * len(candidates)


def test_primitive_usage_statistics(tmp_path):
    config = _tiny_config(str(tmp_path))
    candidates = cc4.generate_all_candidates(config)
    oracle = pd.DataFrame([{"oracle_candidate_id": candidates[0].candidate_id}])
    stats = cc4.compute_primitive_usage_statistics(candidates, oracle)
    assert (stats["n_candidates_referencing"] >= 0).all()
    assert set(stats["primitive_name"]) <= {"laxity_urgency", "priority"}


# ---------------------------------------------------------------------------
# CloudRift clean skip
# ---------------------------------------------------------------------------


def test_cloudrift_skips_cleanly_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOUDRIFT_API_KEY", raising=False)
    config = _tiny_config(str(tmp_path))
    candidates, info = cc4.maybe_generate_cloudrift_candidates(config)
    assert candidates == []
    assert info["used"] is False
    assert info["skip_reason"] == "cloudrift.enabled is false in config"


def test_cloudrift_skips_cleanly_when_enabled_but_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOUDRIFT_API_KEY", raising=False)
    config = _tiny_config(str(tmp_path))
    config["cloudrift"] = {"enabled": True, "model": "some-model"}
    candidates, info = cc4.maybe_generate_cloudrift_candidates(config)
    assert candidates == []
    assert info["used"] is False
    assert info["api_key_present"] is False
    assert info["skip_reason"] == "CLOUDRIFT_API_KEY not set"


# ---------------------------------------------------------------------------
# Verdict determination
# ---------------------------------------------------------------------------


def test_dataset_verdict_complete_when_signal_and_completion_ok():
    oracle = pd.DataFrame([
        {"window_id": "w1", "split": "ID_TEST", "oracle_family": "weighted_primitive_mixture", "oracle_anwg": 0.8, "top2_margin": 0.02},
        {"window_id": "w2", "split": "ID_TEST", "oracle_family": "fixed_policy", "oracle_anwg": 0.5, "top2_margin": 0.02},
    ])
    near_tie = pd.DataFrame([
        {"window_id": "w1", "threshold": 0.005, "near_tie": False},
        {"window_id": "w2", "threshold": 0.005, "near_tie": False},
    ])
    completion = pd.DataFrame([{"window_id": "w1", "completion_ok": True}, {"window_id": "w2", "completion_ok": True}])
    config = {"near_tie_primary_threshold": 0.005, "evaluation_splits": ["ID_TEST"]}
    verdict = cc4.determine_dataset_verdict(oracle, near_tie, completion, config)
    assert verdict["status"] == "COMPLETE"


def test_dataset_verdict_in_progress_when_completion_fails():
    oracle = pd.DataFrame([{"window_id": "w1", "split": "ID_TEST", "oracle_family": "weighted_primitive_mixture", "oracle_anwg": 0.8, "top2_margin": 0.02}])
    near_tie = pd.DataFrame([{"window_id": "w1", "threshold": 0.005, "near_tie": False}])
    completion = pd.DataFrame([{"window_id": "w1", "completion_ok": False}])
    config = {"near_tie_primary_threshold": 0.005, "evaluation_splits": ["ID_TEST"]}
    verdict = cc4.determine_dataset_verdict(oracle, near_tie, completion, config)
    assert verdict["status"] == "IN_PROGRESS"


def test_dataset_verdict_ignores_development_split_windows():
    """A TRAIN-split window's oracle result must not certify the verdict --
    only evaluation-split windows count, mirroring CC1's dev/eval separation."""
    oracle = pd.DataFrame([
        {"window_id": "w_train", "split": "TRAIN", "oracle_family": "weighted_primitive_mixture", "oracle_anwg": 0.9, "top2_margin": 0.5},
    ])
    near_tie = pd.DataFrame([{"window_id": "w_train", "threshold": 0.005, "near_tie": False}])
    completion = pd.DataFrame([{"window_id": "w_train", "completion_ok": True}])
    config = {"near_tie_primary_threshold": 0.005, "evaluation_splits": ["ID_TEST"]}
    verdict = cc4.determine_dataset_verdict(oracle, near_tie, completion, config)
    assert verdict["status"] == "INCONCLUSIVE"
    assert verdict["reason"] == "no executed evaluation-split rows"
