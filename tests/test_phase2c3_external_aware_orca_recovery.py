from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
RUNNER_PATH = ROOT / "scripts" / "run_phase2c3_external_aware_orca_recovery.py"
CONFIG_PATH = ROOT / "configs" / "phase2c3_external_aware_orca_recovery.yaml"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_phase2c3", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_external_aware_pool_includes_orca():
    mod = _load_runner()
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    policies = [
        "fifo",
        "edf",
        "scorpio_style_slo_guard",
        "orca_style",
        "vllm_style_token_budget",
        "sarathi_style",
        "splitfuse_style",
        "multi_bin_batching",
        "estimated_service_time_first",
    ]
    pools = mod.resolve_target_pools(cfg, policies)
    assert "orca_style" in pools["external_aware_non_oracle"].allowed_policies
    assert pools["orca_vs_scorpio_gate"].allowed_policies == ("orca_style", "scorpio_style_slo_guard")


def test_oracle_assisted_policies_excluded_from_pools():
    mod = _load_runner()
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    policies = ["fifo", "orca_style", "safe_fallback_wsp_margin0.005", "scorpio_style_slo_guard"]
    pools = mod.resolve_target_pools(cfg, policies)
    for pool in pools.values():
        assert all(not policy.startswith("safe_fallback_wsp_margin") for policy in pool.allowed_policies)


def test_feature_columns_exclude_leaky_fields():
    mod = _load_runner()
    df = pd.DataFrame(
        [
            {
                "feat_arrival_rate_est": 1.0,
                "feat_mean_prompt_tokens": 2.0,
                "reward_fifo": 0.5,
                "completion_fifo": 1.0,
                "best_policy": "fifo",
                "sel_dt_anwg_policy": "fifo",
            }
        ]
    )
    cols = mod.select_feature_columns(df)
    assert cols == ["feat_arrival_rate_est", "feat_mean_prompt_tokens"]


def test_corrected_anwg_reconstruction_small_example():
    mod = _load_runner()
    df = pd.DataFrame(
        [
            {
                "reward_fifo": 0.8,
                "completion_fifo": 0.5,
                "reward_edf": 0.6,
                "completion_edf": 1.0,
            }
        ]
    )
    out = mod.reconstruct_corrected_anwg(df, ["fifo", "edf"])
    assert out.loc[0, "anwg_fifo"] == 0.4
    assert out.loc[0, "anwg_edf"] == 0.6


def test_evaluate_predictions_uses_predicted_policy_not_eval_best():
    mod = _load_runner()
    df = pd.DataFrame(
        [
            {
                "native_non_oracle_label": "edf",
                "anwg_fifo": 0.9,
                "completion_fifo": 1.0,
                "reward_fifo": 0.9,
                "slo_violation_fifo": 0.1,
                "anwg_edf": 0.1,
                "completion_edf": 1.0,
                "reward_edf": 0.1,
                "slo_violation_edf": 0.9,
                "anwg_scorpio_style_slo_guard": 0.2,
                "external_best_anwg": 0.2,
                "external_best_policy": "scorpio_style_slo_guard",
                "workload": "w0",
                "workload_group": "burstgpt",
            }
        ]
    )
    selector = mod.TrainedSelector(
        key="test_selector",
        target_pool="native_non_oracle",
        model_family="unit",
        predictor=None,
        label_col="native_non_oracle_label",
        allowed_policies=("fifo", "edf"),
        train_rows_used=1,
        near_tie_epsilon=0.005,
    )
    result = mod.evaluate_predictions(
        df,
        selector,
        ["fifo"],
        external_policies=["scorpio_style_slo_guard"],
        view_name="all_workloads",
    )
    assert result["mean_arrival_normalized_wg"] == 0.9
    assert result["label_accuracy"] == 0.0


def test_realistic_subset_filter_excludes_exact_and_overlap_windows():
    mod = _load_runner()
    df = pd.DataFrame(
        [
            {"trace_id": "burstgpt_moderate_exact_prediction_s0", "window_id": 0},
            {"trace_id": "burstgpt_scaled_high_s0", "window_id": 0},
            {"trace_id": "burstgpt_scaled_high_s0", "window_id": 1},
            {"trace_id": "burstgpt_scaled_high_s0", "window_id": 2},
            {"trace_id": "azure_2023_conv_s0", "window_id": 0},
        ]
    )
    flagged = mod.add_workload_flags(df)
    views = mod.build_eval_views(flagged)
    realistic = views["realistic_subset"]
    assert set(realistic["workload"]) == {"burstgpt_scaled_high", "azure_2023_conv"}
    assert set(realistic["window_id"]) == {2, 0}
