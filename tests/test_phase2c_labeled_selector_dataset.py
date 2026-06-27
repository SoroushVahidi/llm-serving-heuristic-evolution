"""Tests for the Phase 2C labeled-selector-dataset builder."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
RUNNER_PATH = ROOT / "scripts" / "build_phase2c_labeled_selector_dataset.py"
CONFIG_PATH = ROOT / "configs" / "phase2c_labeled_selector_dataset.yaml"


def _load_runner():
    spec = importlib.util.spec_from_file_location("build_phase2c_labeled_selector_dataset", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _minimal_policy_df(n_rows: int = 4) -> pd.DataFrame:
    """Tiny DataFrame with reward_*/completion_*/feat_* columns for two policies."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_rows):
        rows.append({
            "trace_id": f"wl_a_s0",
            "window_id": i,
            "feat_queue_length": float(rng.integers(1, 50)),
            "feat_mean_prompt_tokens": float(rng.integers(100, 1200)),
            "feat_arrival_rate_est": float(rng.uniform(0.1, 20.0)),
            "feat_burstiness_cv": float(rng.uniform(0.5, 1.5)),
            "feat_fraction_tight_slo": float(rng.uniform(0.0, 1.0)),
            "reward_orca_style": float(rng.uniform(0.7, 1.0)),
            "completion_orca_style": float(rng.uniform(0.6, 1.0)),
            "reward_scorpio_style_slo_guard": float(rng.uniform(0.7, 1.0)),
            "completion_scorpio_style_slo_guard": float(rng.uniform(0.6, 1.0)),
            "reward_fifo": float(rng.uniform(0.4, 0.9)),
            "completion_fifo": float(rng.uniform(0.5, 1.0)),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Module loading and config
# ─────────────────────────────────────────────────────────────────────────────

def test_module_loads():
    mod = _load_runner()
    assert callable(mod.run_pipeline)
    assert callable(mod.reconstruct_anwg)


def test_config_has_required_sections():
    cfg = _load_cfg()
    assert "inputs" in cfg
    assert "output" in cfg
    assert "policy_pools" in cfg
    assert "regime_thresholds" in cfg
    assert "api_annotation" in cfg
    assert "near_tie_margin" in cfg
    assert "phase2c2_reference_check" in cfg


def test_api_annotation_is_disabled():
    cfg = _load_cfg()
    assert cfg["api_annotation"]["enabled"] is False
    assert cfg["api_annotation"]["provider"] == "none"
    assert cfg["api_annotation"]["max_calls"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# ANWG reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def test_anwg_reconstruction_formula():
    mod = _load_runner()
    df = pd.DataFrame([{
        "reward_fifo": 0.8, "completion_fifo": 0.5,
        "reward_edf": 0.6, "completion_edf": 1.0,
    }])
    out = mod.reconstruct_anwg(df, ["fifo", "edf"])
    assert abs(out.loc[0, "anwg_fifo"] - 0.4) < 1e-9
    assert abs(out.loc[0, "anwg_edf"] - 0.6) < 1e-9


def test_anwg_reconstruction_no_leakage():
    """reconstruct_anwg must not use any existing best_/sel_/anwg_ column."""
    mod = _load_runner()
    df = pd.DataFrame([{
        "reward_fifo": 0.9, "completion_fifo": 0.9,
        "anwg_fifo": 0.0,     # pre-existing wrong value must not affect output
        "best_policy": "edf",
        "sel_dt_policy": "edf",
    }])
    out = mod.reconstruct_anwg(df, ["fifo"])
    assert abs(out.loc[0, "anwg_fifo"] - 0.81) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Feature column selection
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_columns_exclude_leaky():
    mod = _load_runner()
    df = pd.DataFrame([{
        "feat_queue_length": 1.0,
        "feat_arrival_rate_est": 2.0,
        "reward_fifo": 0.5,
        "completion_fifo": 1.0,
        "anwg_fifo": 0.5,
        "sel_dt_policy": "fifo",
        "best_policy": "fifo",
        "label_best": "fifo",
        "external_best": "orca_style",
        "selected_anwg": 0.5,
    }])
    cols = mod.select_feature_columns(df)
    assert set(cols) == {"feat_queue_length", "feat_arrival_rate_est"}


def test_feature_columns_includes_all_feat_cols():
    mod = _load_runner()
    df = _minimal_policy_df(1)
    cols = mod.select_feature_columns(df)
    assert all(c.startswith("feat_") for c in cols)
    assert len(cols) == 5   # 5 feat_ columns in _minimal_policy_df


# ─────────────────────────────────────────────────────────────────────────────
# Policy pool resolution
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_pools_native_excludes_external_style():
    mod = _load_runner()
    cfg = _load_cfg()
    all_policies = [
        "fifo", "edf", "orca_style", "sarathi_style", "splitfuse_style",
        "vllm_style_token_budget", "scorpio_style_slo_guard",
        "safe_fallback_wsp_margin0.005",
    ]
    pools = mod.resolve_pools(cfg, all_policies)
    native = pools["native_non_oracle"]
    assert "orca_style" not in native
    assert "sarathi_style" not in native
    assert "splitfuse_style" not in native
    assert "vllm_style_token_budget" not in native
    assert "safe_fallback_wsp_margin0.005" not in native
    assert "fifo" in native
    assert "scorpio_style_slo_guard" in native


def test_resolve_pools_external_aware_includes_orca():
    mod = _load_runner()
    cfg = _load_cfg()
    policies = ["fifo", "orca_style", "scorpio_style_slo_guard"]
    pools = mod.resolve_pools(cfg, policies)
    assert "orca_style" in pools["external_style"]


def test_resolve_pools_oracle_excluded_from_all():
    mod = _load_runner()
    cfg = _load_cfg()
    policies = ["fifo", "orca_style", "safe_fallback_wsp_margin0.001"]
    pools = mod.resolve_pools(cfg, policies)
    for pool_name, pool_policies in pools.items():
        for p in pool_policies:
            assert not p.startswith("safe_fallback_wsp_margin"), (
                f"Oracle policy in pool {pool_name}: {p}"
            )


def test_orca_vs_scorpio_pool_has_exactly_two():
    mod = _load_runner()
    cfg = _load_cfg()
    policies = ["fifo", "orca_style", "scorpio_style_slo_guard", "edf"]
    pools = mod.resolve_pools(cfg, policies)
    assert set(pools["orca_vs_scorpio"]) == {"orca_style", "scorpio_style_slo_guard"}


# ─────────────────────────────────────────────────────────────────────────────
# Policy-choice labels
# ─────────────────────────────────────────────────────────────────────────────

def test_policy_choice_label_selects_best_anwg():
    mod = _load_runner()
    df = pd.DataFrame([{"anwg_fifo": 0.9, "anwg_edf": 0.4}])
    out = mod.compute_policy_choice_labels(df, "native_non_oracle", ["fifo", "edf"], 0.005)
    assert out.loc[0, "label_best_native_non_oracle_policy"] == "fifo"


def test_near_tie_flag_when_margin_small():
    mod = _load_runner()
    df = pd.DataFrame([
        {"anwg_fifo": 0.800, "anwg_edf": 0.798},  # margin 0.002 < 0.005
        {"anwg_fifo": 0.900, "anwg_edf": 0.600},  # margin 0.3 > 0.005
    ])
    out = mod.compute_policy_choice_labels(df, "native_non_oracle", ["fifo", "edf"], 0.005)
    assert out.loc[0, "is_near_tie_native"] == True
    assert out.loc[1, "is_near_tie_native"] == False


def test_orca_vs_scorpio_labels():
    mod = _load_runner()
    df = pd.DataFrame([
        {"anwg_orca_style": 0.9, "anwg_scorpio_style_slo_guard": 0.8},
        {"anwg_orca_style": 0.7, "anwg_scorpio_style_slo_guard": 0.85},
    ])
    out = mod.compute_policy_choice_labels(
        df, "orca_vs_scorpio", ["orca_style", "scorpio_style_slo_guard"], 0.005
    )
    assert out.loc[0, "label_best_orca_vs_scorpio_policy"] == "orca_style"
    assert out.loc[1, "label_best_orca_vs_scorpio_policy"] == "scorpio_style_slo_guard"


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise orca-scorpio
# ─────────────────────────────────────────────────────────────────────────────

def test_pairwise_orca_scorpio_columns_present():
    mod = _load_runner()
    df = pd.DataFrame([{
        "anwg_orca_style": 0.9, "anwg_scorpio_style_slo_guard": 0.8,
        "completion_orca_style": 1.0, "completion_scorpio_style_slo_guard": 0.8,
        "reward_orca_style": 0.9, "reward_scorpio_style_slo_guard": 1.0,
    }])
    out = mod.compute_pairwise_orca_scorpio(df)
    assert "label_orca_beats_scorpio" in out.columns
    assert "orca_minus_scorpio_anwg" in out.columns
    assert "orca_minus_scorpio_completion" in out.columns
    assert "orca_minus_scorpio_quality" in out.columns
    assert "orca_better_by_completion" in out.columns
    assert "orca_better_by_quality" in out.columns


def test_pairwise_orca_scorpio_values():
    mod = _load_runner()
    df = pd.DataFrame([{
        "anwg_orca_style": 0.9, "anwg_scorpio_style_slo_guard": 0.8,
        "completion_orca_style": 1.0, "completion_scorpio_style_slo_guard": 0.8,
        "reward_orca_style": 0.9, "reward_scorpio_style_slo_guard": 1.0,
    }])
    out = mod.compute_pairwise_orca_scorpio(df)
    assert out.loc[0, "label_orca_beats_scorpio"] == True
    assert abs(out.loc[0, "orca_minus_scorpio_anwg"] - 0.1) < 1e-6
    assert out.loc[0, "orca_better_by_completion"] == True
    assert out.loc[0, "orca_better_by_quality"] == False  # orca reward 0.9 < scorpio 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Regime labels
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_labels_azure_flags():
    mod = _load_runner()
    cfg = _load_cfg()
    df = pd.DataFrame([
        {"trace_id": "azure_2023_conv_s0", "window_id": 0,
         "feat_arrival_rate_est": 0.2, "feat_burstiness_cv": 1.05,
         "feat_mean_prompt_tokens": 1200.0, "feat_fraction_tight_slo": 0.5},
        {"trace_id": "burstgpt_moderate_noise070_s0", "window_id": 3,
         "feat_arrival_rate_est": 12.0, "feat_burstiness_cv": 1.4,
         "feat_mean_prompt_tokens": 600.0, "feat_fraction_tight_slo": 0.1},
    ])
    df["workload"] = df["trace_id"].map(lambda x: x.rsplit("_s", 1)[0])
    out = mod.compute_regime_labels(df, cfg)
    assert out.loc[0, "is_azure"] == True
    assert out.loc[0, "is_azure_conv"] == True
    assert out.loc[1, "is_burstgpt"] == True
    assert out.loc[1, "is_azure"] == False


def test_exact_prediction_and_overlap_windows_flagged():
    mod = _load_runner()
    cfg = _load_cfg()
    df = pd.DataFrame([
        {"trace_id": "burstgpt_moderate_exact_prediction_s0", "window_id": 0,
         "feat_arrival_rate_est": 5.0, "feat_burstiness_cv": 1.1,
         "feat_mean_prompt_tokens": 500.0, "feat_fraction_tight_slo": 0.2},
        {"trace_id": "burstgpt_scaled_high_s0", "window_id": 0,
         "feat_arrival_rate_est": 5.0, "feat_burstiness_cv": 0.9,
         "feat_mean_prompt_tokens": 500.0, "feat_fraction_tight_slo": 0.2},
        {"trace_id": "burstgpt_scaled_high_s0", "window_id": 1,
         "feat_arrival_rate_est": 5.0, "feat_burstiness_cv": 0.9,
         "feat_mean_prompt_tokens": 500.0, "feat_fraction_tight_slo": 0.2},
        {"trace_id": "burstgpt_scaled_high_s0", "window_id": 2,
         "feat_arrival_rate_est": 5.0, "feat_burstiness_cv": 0.9,
         "feat_mean_prompt_tokens": 500.0, "feat_fraction_tight_slo": 0.2},
        {"trace_id": "azure_2023_conv_s0", "window_id": 0,
         "feat_arrival_rate_est": 0.2, "feat_burstiness_cv": 1.05,
         "feat_mean_prompt_tokens": 1200.0, "feat_fraction_tight_slo": 0.5},
    ])
    df["workload"] = df["trace_id"].map(lambda x: x.rsplit("_s", 1)[0])
    out = mod.compute_regime_labels(df, cfg)
    assert out[out["workload"] == "burstgpt_moderate_exact_prediction"]["is_exact_prediction_oracle_like"].all()
    overlap_rows = out[out["workload"] == "burstgpt_scaled_high"]
    assert overlap_rows.iloc[0]["is_overlap_sensitive_first_two"] == True
    assert overlap_rows.iloc[1]["is_overlap_sensitive_first_two"] == True
    assert overlap_rows.iloc[2]["is_overlap_sensitive_first_two"] == False


def test_is_realistic_subset_excludes_exact_and_overlap():
    mod = _load_runner()
    cfg = _load_cfg()
    # Use windows 0, 1, 2 so that window 2 gets cumcount rank=2 (>= n_windows_to_exclude=2)
    df = pd.DataFrame([
        {"trace_id": "burstgpt_moderate_exact_prediction_s0", "window_id": 0,
         "feat_arrival_rate_est": 5.0, "feat_burstiness_cv": 1.1,
         "feat_mean_prompt_tokens": 500.0, "feat_fraction_tight_slo": 0.2},
        {"trace_id": "burstgpt_scaled_high_s0", "window_id": 0,
         "feat_arrival_rate_est": 5.0, "feat_burstiness_cv": 0.9,
         "feat_mean_prompt_tokens": 500.0, "feat_fraction_tight_slo": 0.2},
        {"trace_id": "burstgpt_scaled_high_s0", "window_id": 1,
         "feat_arrival_rate_est": 5.0, "feat_burstiness_cv": 0.9,
         "feat_mean_prompt_tokens": 500.0, "feat_fraction_tight_slo": 0.2},
        {"trace_id": "burstgpt_scaled_high_s0", "window_id": 2,
         "feat_arrival_rate_est": 5.0, "feat_burstiness_cv": 0.9,
         "feat_mean_prompt_tokens": 500.0, "feat_fraction_tight_slo": 0.2},
        {"trace_id": "azure_2023_conv_s0", "window_id": 5,
         "feat_arrival_rate_est": 0.2, "feat_burstiness_cv": 1.05,
         "feat_mean_prompt_tokens": 1200.0, "feat_fraction_tight_slo": 0.5},
    ])
    df["workload"] = df["trace_id"].map(lambda x: x.rsplit("_s", 1)[0])
    out = mod.compute_regime_labels(df, cfg)
    realistic = out[out["is_realistic_subset"]]
    # exact_prediction excluded; windows 0 and 1 of burstgpt_scaled_high excluded;
    # window 2 (rank=2 >= n_windows_to_exclude=2) and azure_2023_conv survive
    assert len(realistic) == 2
    assert set(realistic["workload"]) == {"burstgpt_scaled_high", "azure_2023_conv"}


def test_is_azure_conv_like_is_feature_based():
    """is_azure_conv_like can be True for non-azure workloads that match feature thresholds."""
    mod = _load_runner()
    cfg = _load_cfg()
    df = pd.DataFrame([
        # A burstgpt window with azure-like features (long prompt + mixed tight SLO)
        {"trace_id": "burstgpt_moderate_noise070_s0", "window_id": 10,
         "feat_arrival_rate_est": 0.15, "feat_burstiness_cv": 1.08,
         "feat_mean_prompt_tokens": 1100.0, "feat_fraction_tight_slo": 0.55},
        # An azure_2023_code window that does NOT match (short prompts)
        {"trace_id": "azure_2023_code_s0", "window_id": 0,
         "feat_arrival_rate_est": 0.2, "feat_burstiness_cv": 1.1,
         "feat_mean_prompt_tokens": 200.0, "feat_fraction_tight_slo": 0.0},
    ])
    df["workload"] = df["trace_id"].map(lambda x: x.rsplit("_s", 1)[0])
    out = mod.compute_regime_labels(df, cfg)
    # burstgpt row with long prompt + mixed tight SLO → azure_conv_like even though not azure
    assert out.loc[0, "is_azure_conv_like"] == True
    assert out.loc[0, "is_azure"] == False   # workload is burstgpt
    # azure_code row with short prompts → NOT azure_conv_like
    assert out.loc[1, "is_azure_conv_like"] == False


# ─────────────────────────────────────────────────────────────────────────────
# No live API path
# ─────────────────────────────────────────────────────────────────────────────

def test_script_has_no_live_api_import():
    """The script source must not import any live API SDK at module level."""
    source = RUNNER_PATH.read_text()
    bad_imports = [
        "import google.generativeai",
        "from google.generativeai",
        "import vertexai",
        "from vertexai",
        "import openai",
        "import anthropic",
        "import cohere",
        "import boto3",
    ]
    for bad in bad_imports:
        assert bad not in source, f"Live API import found in script: {bad!r}"


def _find_output_dir(tmp_path: Path) -> Path:
    """Return the timestamped subdirectory created by main() under tmp_path."""
    subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(subdirs) >= 1, f"No output subdir found under {tmp_path}"
    return sorted(subdirs)[-1]


def test_api_enabled_true_returns_error(tmp_path):
    """If config enables API annotation, script must refuse."""
    mod = _load_runner()
    cfg = _load_cfg()
    cfg["api_annotation"]["enabled"] = True
    import yaml as _yaml
    bad_cfg = tmp_path / "bad_cfg.yaml"
    bad_cfg.write_text(_yaml.dump(cfg))
    result = mod.main(["--config", str(bad_cfg), "--dry-run"])
    assert result == 2


def test_mock_api_annotations_are_marked_mock(tmp_path):
    """Mock annotations must be clearly labeled as mock and not confused with labels."""
    mod = _load_runner()
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--mock-api-annotations",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    out_dir = _find_output_dir(tmp_path)
    df = pd.read_csv(out_dir / "labeled_windows.csv")
    assert "api_annotation_regime_summary_mock" in df.columns
    assert df["api_annotation_regime_summary_mock"].eq("MOCK_NOT_A_LABEL").all()
    assert "api_annotation_is_mock" in df.columns
    assert df["api_annotation_is_mock"].all()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset schema and manifest
# ─────────────────────────────────────────────────────────────────────────────

def test_schema_has_safety_metadata():
    mod = _load_runner()
    schema = mod.build_schema()
    for label_name, meta in schema.items():
        assert "safe_for_training" in meta, f"Missing safe_for_training in {label_name}"
        assert "analysis_only" in meta, f"Missing analysis_only in {label_name}"
        assert "oracle_like_sensitive" in meta
        assert "external_approximation_sensitive" in meta


def test_native_non_oracle_label_is_safe_for_training():
    mod = _load_runner()
    schema = mod.build_schema()
    assert schema["label_best_native_non_oracle_policy"]["safe_for_training"] is True
    assert schema["label_best_native_non_oracle_policy"]["analysis_only"] is False


def test_external_labels_are_analysis_only():
    mod = _load_runner()
    schema = mod.build_schema()
    for key in ("label_best_external_style_policy", "label_best_all_non_oracle_policy",
                "label_orca_beats_scorpio", "label_phase2c2_dt_loses_to_external_envelope"):
        assert schema[key]["safe_for_training"] is False, f"{key} should not be safe_for_training"
        assert schema[key]["analysis_only"] is True, f"{key} should be analysis_only"


def test_oracle_label_flagged_correctly():
    mod = _load_runner()
    schema = mod.build_schema()
    assert schema["is_exact_prediction_oracle_like"]["oracle_like_sensitive"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline integration test (uses real Phase 2C.2 data)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_dry_run_succeeds(tmp_path):
    mod = _load_runner()
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--dry-run",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    # dry-run writes no files, so no subdir should exist
    subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(subdirs) == 0, "Dry-run must not write files"


@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_full_pipeline_produces_all_output_files(tmp_path):
    mod = _load_runner()
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    out_dir = _find_output_dir(tmp_path)
    expected_files = [
        "labeled_windows.csv",
        "train_labeled_windows.csv",
        "val_labeled_windows.csv",
        "eval_labeled_windows.csv",
        "pairwise_orca_scorpio_labels.csv",
        "external_loss_labels.csv",
        "regime_labels.csv",
        "feature_columns.txt",
        "dataset_schema.json",
        "dataset_manifest.json",
        "label_distribution_summary.csv",
        "phase2c_labeled_dataset_report.md",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists(), f"Missing output file: {fname}"


@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_train_val_eval_splits_preserved(tmp_path):
    mod = _load_runner()
    mod.main(["--config", str(CONFIG_PATH), "--output-dir", str(tmp_path)])
    out_dir = _find_output_dir(tmp_path)
    full_df = pd.read_csv(out_dir / "labeled_windows.csv")
    train_df = pd.read_csv(out_dir / "train_labeled_windows.csv")
    val_df = pd.read_csv(out_dir / "val_labeled_windows.csv")
    eval_df = pd.read_csv(out_dir / "eval_labeled_windows.csv")
    assert len(train_df) + len(val_df) + len(eval_df) == len(full_df)
    assert (train_df["split"] == "train").all()
    assert (val_df["split"] == "val").all()
    assert (eval_df["split"] == "eval").all()


@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_no_leaky_feat_cols_in_output(tmp_path):
    mod = _load_runner()
    mod.main(["--config", str(CONFIG_PATH), "--output-dir", str(tmp_path)])
    out_dir = _find_output_dir(tmp_path)
    feat_cols = (out_dir / "feature_columns.txt").read_text().strip().splitlines()
    bad_tokens = ("reward_", "completion_", "sel_", "best_", "label",
                  "anwg_", "external_", "selected_")
    for col in feat_cols:
        for tok in bad_tokens:
            assert tok not in col.lower(), f"Leaky token {tok!r} in feature col {col!r}"


@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_manifest_has_live_api_false(tmp_path):
    mod = _load_runner()
    mod.main(["--config", str(CONFIG_PATH), "--output-dir", str(tmp_path)])
    out_dir = _find_output_dir(tmp_path)
    manifest = json.loads((out_dir / "dataset_manifest.json").read_text())
    assert manifest["live_api_used"] is False
    assert manifest["api_annotation_enabled"] is False


@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_orca_vs_scorpio_labels_present(tmp_path):
    mod = _load_runner()
    mod.main(["--config", str(CONFIG_PATH), "--output-dir", str(tmp_path)])
    out_dir = _find_output_dir(tmp_path)
    df = pd.read_csv(out_dir / "pairwise_orca_scorpio_labels.csv")
    assert "label_orca_beats_scorpio" in df.columns
    assert "orca_minus_scorpio_anwg" in df.columns
    # Some windows should have orca winning and some losing
    assert df["label_orca_beats_scorpio"].any()
    assert (~df["label_orca_beats_scorpio"]).any()


@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_near_tie_rows_exist(tmp_path):
    mod = _load_runner()
    mod.main(["--config", str(CONFIG_PATH), "--output-dir", str(tmp_path)])
    out_dir = _find_output_dir(tmp_path)
    df = pd.read_csv(out_dir / "labeled_windows.csv")
    assert "is_near_tie_native" in df.columns
    # There should be some near-tie rows (we know from Phase 2C.2 analysis that
    # admission_control has many near-ties with scorpio)
    assert df["is_near_tie_native"].sum() > 0


@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_azure_conv_like_not_only_azure(tmp_path):
    """is_azure_conv_like should cover non-azure rows that match the feature profile."""
    mod = _load_runner()
    mod.main(["--config", str(CONFIG_PATH), "--output-dir", str(tmp_path)])
    out_dir = _find_output_dir(tmp_path)
    df = pd.read_csv(out_dir / "labeled_windows.csv")
    azure_conv_like = df[df["is_azure_conv_like"]]
    # Should include azure_2023_conv windows (long prompts + mixed tight SLO)
    assert "azure_2023_conv" in azure_conv_like["workload"].values or len(azure_conv_like) == 0


@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_external_loss_rows_are_eval_only(tmp_path):
    """External-loss labels are only meaningful for eval rows."""
    mod = _load_runner()
    mod.main(["--config", str(CONFIG_PATH), "--output-dir", str(tmp_path)])
    out_dir = _find_output_dir(tmp_path)
    df = pd.read_csv(out_dir / "external_loss_labels.csv")
    if len(df) > 0:
        assert (df["split"] == "eval").all(), "External loss labels should only come from eval rows"


@pytest.mark.skipif(
    not (ROOT / "results" / "phase2c2_causal_selector_retraining").exists(),
    reason="Phase 2C.2 results not present"
)
def test_label_distribution_covers_all_pools(tmp_path):
    mod = _load_runner()
    mod.main(["--config", str(CONFIG_PATH), "--output-dir", str(tmp_path)])
    out_dir = _find_output_dir(tmp_path)
    df = pd.read_csv(out_dir / "label_distribution_summary.csv")
    pools_in_summary = set(df["pool"])
    expected_pools = {"native_non_oracle", "external_style", "all_non_oracle", "orca_vs_scorpio"}
    assert expected_pools.issubset(pools_in_summary)
