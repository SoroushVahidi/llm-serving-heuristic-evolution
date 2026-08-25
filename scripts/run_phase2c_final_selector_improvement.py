#!/usr/bin/env python3
"""Final serious selector-improvement phase over corrected-objective ANWG.

The runner uses Phase 2C's labeled causal dataset, optionally augments the
training/validation split with new targeted synthetic long-prompt mixed-SLO
windows, freezes model selection on validation, and evaluates once on held-out
real-trace eval plus a separately generated fresh synthetic targeted split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.selector.advanced import (  # noqa: E402
    FixedPolicySelector,
    PairwisePolicyRanker,
    PolicyClassifierSelector,
    PolicyRewardRegressorSelector,
    RegimeGatedSelector,
    UncertaintyFallbackSelector,
    all_pair_combinations,
    anwg_column,
    anwg_value,
    azure_conv_like_gate,
    validate_feature_columns,
)
from llmserveopt.selector.features import FeatureMode  # noqa: E402
from llmserveopt.simulator.service_model_factory import build_service_model_from_config  # noqa: E402

from build_phase2c_labeled_selector_dataset import (  # noqa: E402
    compute_pairwise_orca_scorpio,
    compute_policy_choice_labels,
    compute_regime_labels,
)
from run_phase2b12_workload_diversity_selector_labels import build_rows_for_group  # noqa: E402
from run_phase2b9_selector_robustness import build_gpu_configs  # noqa: E402


DEFAULT_DATASET_DIR = "results/phase2c_labeled_selector_dataset/20260627_142404"
DEFAULT_OUTPUT_ROOT = "results/phase2c_final_selector_improvement"
RANDOM_STATE = 42
NEAR_TIE = 0.005


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    estimator: str = "random_forest"
    weight_scheme: str = "uniform"
    n_estimators: int = 200
    max_depth: Optional[int] = 10
    min_pair_margin: float = 0.001


TARGETED_WORKLOADS: List[Dict[str, Any]] = [
    {
        "tag": "target_azure_conv_long_prompt_mixed_slo",
        "group": "targeted",
        "arrival_process": "poisson",
        "arrival_rate": 44.0,
        "duration": 16.0,
        "prompt_dist": "lognormal",
        "prompt_mean": 1800.0,
        "prompt_sigma": 0.7,
        "prompt_low": 512,
        "prompt_high": 8192,
        "output_dist": "lognormal",
        "output_mean": 32.0,
        "output_sigma": 0.7,
        "output_low": 4,
        "output_high": 256,
        "prediction_noise_rel": 0.25,
        "slo_classes": [
            {"class_id": "tight", "slo_slack": 2.0, "priority": 3.0, "weight": 0.50},
            {"class_id": "medium", "slo_slack": 8.0, "priority": 2.0, "weight": 0.30},
            {"class_id": "loose", "slo_slack": 24.0, "priority": 1.0, "weight": 0.20},
        ],
    },
    {
        "tag": "target_azure_conv_high_load",
        "group": "targeted",
        "arrival_process": "poisson",
        "arrival_rate": 62.0,
        "duration": 14.0,
        "prompt_dist": "lognormal",
        "prompt_mean": 2200.0,
        "prompt_sigma": 0.8,
        "prompt_low": 512,
        "prompt_high": 8192,
        "output_dist": "lognormal",
        "output_mean": 48.0,
        "output_sigma": 0.8,
        "output_low": 4,
        "output_high": 384,
        "prediction_noise_rel": 0.35,
        "slo_classes": [
            {"class_id": "tight", "slo_slack": 1.5, "priority": 3.0, "weight": 0.55},
            {"class_id": "medium", "slo_slack": 6.0, "priority": 2.0, "weight": 0.25},
            {"class_id": "loose", "slo_slack": 20.0, "priority": 1.0, "weight": 0.20},
        ],
    },
    {
        "tag": "target_azure_conv_bursty",
        "group": "targeted",
        "arrival_process": "bursty",
        "arrival_rate": 48.0,
        "duration": 16.0,
        "burst_factor": 4.0,
        "burst_fraction": 0.25,
        "prompt_dist": "lognormal",
        "prompt_mean": 1600.0,
        "prompt_sigma": 0.75,
        "prompt_low": 512,
        "prompt_high": 8192,
        "output_dist": "lognormal",
        "output_mean": 36.0,
        "output_sigma": 0.9,
        "output_low": 4,
        "output_high": 384,
        "prediction_noise_rel": 0.4,
        "slo_classes": [
            {"class_id": "tight", "slo_slack": 2.0, "priority": 3.0, "weight": 0.45},
            {"class_id": "medium", "slo_slack": 7.0, "priority": 2.0, "weight": 0.35},
            {"class_id": "loose", "slo_slack": 20.0, "priority": 1.0, "weight": 0.20},
        ],
    },
    {
        "tag": "target_long_prompt_noise",
        "group": "targeted",
        "arrival_process": "poisson",
        "arrival_rate": 54.0,
        "duration": 15.0,
        "prompt_dist": "lognormal",
        "prompt_mean": 1500.0,
        "prompt_sigma": 0.9,
        "prompt_low": 512,
        "prompt_high": 8192,
        "output_dist": "lognormal",
        "output_mean": 64.0,
        "output_sigma": 1.0,
        "output_low": 4,
        "output_high": 512,
        "prediction_noise_rel": 0.7,
        "slo_classes": [
            {"class_id": "tight", "slo_slack": 2.5, "priority": 3.0, "weight": 0.50},
            {"class_id": "medium", "slo_slack": 7.0, "priority": 2.0, "weight": 0.30},
            {"class_id": "loose", "slo_slack": 18.0, "priority": 1.0, "weight": 0.20},
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2C final selector improvement")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-targeted-augmentation", action="store_true")
    parser.add_argument("--include-all-pairs", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use fewer trees/seeds for a quick development run")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False))


def load_dataset(dataset_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    df = pd.read_csv(dataset_dir / "labeled_windows.csv")
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    feature_cols = [line.strip() for line in (dataset_dir / "feature_columns.txt").read_text().splitlines() if line.strip()]
    feature_cols = validate_feature_columns(feature_cols)
    return df, manifest, feature_cols


def policies_for_pool(manifest: Mapping[str, Any], pool: str) -> List[str]:
    return list(manifest["pools"][pool])


def workload_from_trace(trace_id: str) -> str:
    return trace_id.rsplit("_s", 1)[0] if "_s" in trace_id else trace_id


def ensure_anwg_and_labels(df: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for policy in manifest["policies"]:
        if anwg_column(policy) not in out.columns and f"reward_{policy}" in out.columns:
            out[anwg_column(policy)] = out[f"reward_{policy}"].astype(float) * out[f"completion_{policy}"].astype(float)
    if "workload" not in out.columns:
        out["workload"] = out["trace_id"].map(workload_from_trace)
    for pool_name, policies in manifest["pools"].items():
        label_col = f"label_best_{pool_name}_policy"
        if label_col not in out.columns:
            out = compute_policy_choice_labels(out, pool_name, policies, NEAR_TIE)
    if "label_orca_beats_scorpio" not in out.columns and {"orca_style", "scorpio_style_slo_guard"}.issubset(set(manifest["policies"])):
        out = compute_pairwise_orca_scorpio(out)
    if "is_azure_conv_like" not in out.columns:
        cfg = {
            "regime_thresholds": {
                "high_arrival_rate": 10.0,
                "high_burstiness_cv": 1.0,
                "long_prompt_tokens": 1000,
                "mixed_tight_slo_low": 0.4,
                "mixed_tight_slo_high": 0.7,
            }
        }
        out = compute_regime_labels(out, cfg)
    return out


def build_generation_cfg() -> Dict[str, Any]:
    return {
        "simulator": {"step_size": 0.001, "drain_steps": 20000},
        "service_model": {
            "enable_prefill_modeling": True,
            "prefill_cost_per_token": 1.0,
            "max_prefill_chunk_tokens": 512,
            "step_token_budget": 4096,
            "decode_first": False,
        },
        "gpus": [
            {
                "gpu_id": 0,
                "max_active_sequences": 4,
                "max_batch_tokens": 4096,
                "max_kv_tokens": 32768,
            }
        ],
        "window_size": 200,
        "min_partial_window": 50,
    }


def generate_targeted_rows(
    *,
    manifest: Mapping[str, Any],
    split: str,
    seeds: Sequence[int],
    quick: bool,
) -> pd.DataFrame:
    cfg = build_generation_cfg()
    workloads = TARGETED_WORKLOADS[:2] if quick else TARGETED_WORKLOADS
    gpu_configs = build_gpu_configs(cfg)
    service_model = build_service_model_from_config(cfg)
    rows = build_rows_for_group(
        workloads,
        list(seeds),
        gpu_configs,
        service_model,
        cfg["simulator"]["drain_steps"],
        cfg["window_size"],
        cfg["min_partial_window"],
        FeatureMode.CAUSAL,
        False,
    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["split"] = split
    out["workload"] = out["trace_id"].map(workload_from_trace)
    out = ensure_anwg_and_labels(out, manifest)
    out["is_train"] = split == "train"
    out["is_val"] = split == "val"
    out["is_eval"] = split in {"eval", "synthetic_fresh"}
    return out


def align_features(frames: Sequence[pd.DataFrame], feature_cols: Sequence[str]) -> Tuple[List[pd.DataFrame], List[str]]:
    all_feature_cols = sorted({c for frame in frames for c in frame.columns if c.startswith("feat_")} | set(feature_cols))
    all_feature_cols = validate_feature_columns(all_feature_cols)
    aligned = []
    for frame in frames:
        out = frame.copy()
        for col in all_feature_cols:
            if col not in out.columns:
                out[col] = 0.0
        aligned.append(out)
    return aligned, all_feature_cols


def selector_specs(quick: bool, include_all_pairs: bool) -> List[ModelSpec]:
    trees = 20 if quick else 120
    hgb_iter = 40 if quick else 120
    specs = [
        ModelSpec("rf_classifier", "classifier", "random_forest", "uniform", trees, 10),
        ModelSpec("rf_classifier_margin_weighted", "classifier", "random_forest", "margin_plus_epsilon", trees, 10),
        ModelSpec("extra_classifier_margin_weighted", "classifier", "extra_trees", "margin_plus_epsilon", trees, 10),
        ModelSpec("rf_reward_regression", "regression", "random_forest", "uniform", trees, 10),
        ModelSpec("rf_reward_regression_weighted", "regression", "random_forest", "margin_plus_epsilon", trees, 10),
        ModelSpec("extra_reward_regression", "regression", "extra_trees", "uniform", trees, 10),
        ModelSpec("extra_reward_regression_weighted", "regression", "extra_trees", "margin_plus_epsilon", trees, 10),
        ModelSpec("hgb_reward_regression", "regression", "hist_gradient_boosting", "uniform", hgb_iter, 31),
        ModelSpec("pairwise_core_rf", "pairwise_core", "random_forest", "uniform", trees, 8),
    ]
    if include_all_pairs:
        specs.append(ModelSpec("pairwise_all_rf", "pairwise_all", "random_forest", "uniform", max(10, trees // 2), 8))
    return specs


def pairwise_core_pairs(allowed: Sequence[str]) -> List[Tuple[str, str]]:
    requested = [
        ("orca_style", "scorpio_style_slo_guard"),
        ("scorpio_style_slo_guard", "admission_control"),
        ("scorpio_style_slo_guard", "weighted_shortest_processing"),
        ("scorpio_style_slo_guard", "multi_bin_batching"),
        ("admission_control", "weighted_shortest_processing"),
        ("edf", "scorpio_style_slo_guard"),
    ]
    allowed_set = set(allowed)
    return [(a, b) for a, b in requested if a in allowed_set and b in allowed_set]


def fit_spec(spec: ModelSpec, train: pd.DataFrame, *, allowed: Sequence[str], feature_cols: Sequence[str], label_col: str):
    if spec.family == "classifier":
        return PolicyClassifierSelector(
            name=spec.name,
            allowed_policies=allowed,
            feature_cols=feature_cols,
            label_col=label_col,
            estimator=spec.estimator,
            n_estimators=spec.n_estimators,
            max_depth=spec.max_depth,
            random_state=RANDOM_STATE,
            weight_scheme=spec.weight_scheme,
        ).fit(train)
    if spec.family == "regression":
        return PolicyRewardRegressorSelector(
            name=spec.name,
            allowed_policies=allowed,
            feature_cols=feature_cols,
            estimator=spec.estimator,
            n_estimators=spec.n_estimators,
            max_depth=spec.max_depth,
            random_state=RANDOM_STATE,
            weight_scheme=spec.weight_scheme,
        ).fit(train)
    if spec.family.startswith("pairwise"):
        pairs = pairwise_core_pairs(allowed) if spec.family == "pairwise_core" else all_pair_combinations(allowed)
        return PairwisePolicyRanker(
            name=spec.name,
            allowed_policies=allowed,
            feature_cols=feature_cols,
            pairs=pairs,
            estimator=spec.estimator,
            n_estimators=spec.n_estimators,
            max_depth=spec.max_depth,
            random_state=RANDOM_STATE,
            min_pair_margin=spec.min_pair_margin,
        ).fit(train)
    raise ValueError(spec.family)


def selected_values(df: pd.DataFrame, preds: Sequence[str], suffix: str) -> Dict[str, float]:
    anwgs: List[float] = []
    qualities: List[float] = []
    completions: List[float] = []
    for (_, row), policy in zip(df.iterrows(), preds):
        anwgs.append(anwg_value(row, policy))
        q_col = f"reward_{policy}"
        c_col = f"completion_{policy}"
        qualities.append(float(row[q_col]) if q_col in row else np.nan)
        completions.append(float(row[c_col]) if c_col in row else np.nan)
    return {
        f"mean_anwg{suffix}": float(np.nanmean(anwgs)) if anwgs else float("nan"),
        f"mean_completed_quality{suffix}": float(np.nanmean(qualities)) if qualities else float("nan"),
        f"mean_completion_fraction{suffix}": float(np.nanmean(completions)) if completions else float("nan"),
    }


def oracle_scores(df: pd.DataFrame, policies: Sequence[str]) -> Tuple[pd.Series, pd.Series]:
    matrix = df[[anwg_column(p) for p in policies]].astype(float)
    best_vals = matrix.max(axis=1)
    winners = matrix.idxmax(axis=1).str.replace("anwg_", "", regex=False)
    return best_vals, winners


def fixed_policy_table(df: pd.DataFrame, policies: Sequence[str], *, split_name: str) -> pd.DataFrame:
    rows = []
    for policy in policies:
        vals = selected_values(df, [policy] * len(df), "")
        rows.append({
            "split": split_name,
            "selector": f"fixed_{policy}",
            "selector_type": "fixed_policy",
            "policy": policy,
            **vals,
        })
    return pd.DataFrame(rows)


def evaluate_predictions(
    df: pd.DataFrame,
    preds: Sequence[str],
    *,
    selector_name: str,
    selector_type: str,
    allowed_policies: Sequence[str],
    oracle_name: str,
    split_name: str,
    best_fixed_anwg: float,
) -> Dict[str, Any]:
    chosen_anwg = np.asarray([anwg_value(row, p) for (_, row), p in zip(df.iterrows(), preds)], dtype=float)
    oracle_vals, oracle_policies = oracle_scores(df, allowed_policies)
    oracle_arr = oracle_vals.to_numpy(dtype=float)
    regret = np.maximum(oracle_arr - chosen_anwg, 0.0)
    mean_anwg = float(np.mean(chosen_anwg)) if len(chosen_anwg) else float("nan")
    oracle_mean = float(np.mean(oracle_arr)) if len(oracle_arr) else float("nan")
    denom = oracle_mean - best_fixed_anwg
    gap_closed = (mean_anwg - best_fixed_anwg) / denom if denom > 0 else float("nan")
    quality_comp = selected_values(df, preds, "")
    return {
        "split": split_name,
        "selector": selector_name,
        "selector_type": selector_type,
        "n_windows": int(len(df)),
        "oracle_name": oracle_name,
        **quality_comp,
        "absolute_gain_over_best_fixed": mean_anwg - best_fixed_anwg,
        "gap_to_oracle": oracle_mean - mean_anwg,
        "gap_closed_fraction": gap_closed,
        "mean_oracle_regret": float(np.mean(regret)) if len(regret) else float("nan"),
        "p95_oracle_regret": float(np.percentile(regret, 95)) if len(regret) else float("nan"),
        "worst_oracle_regret": float(np.max(regret)) if len(regret) else float("nan"),
        "within_0.001_oracle_fraction": float(np.mean(regret <= 0.001)) if len(regret) else float("nan"),
        "within_0.005_oracle_fraction": float(np.mean(regret <= 0.005)) if len(regret) else float("nan"),
        "within_0.010_oracle_fraction": float(np.mean(regret <= 0.010)) if len(regret) else float("nan"),
        "chosen_policy_distribution": dict(Counter(preds)),
        "oracle_policy_distribution": dict(Counter(oracle_policies)),
    }


def bootstrap_summary(
    df: pd.DataFrame,
    preds_by_selector: Mapping[str, Sequence[str]],
    *,
    allowed_policies: Sequence[str],
    best_fixed_anwg: float,
    n_bootstrap: int,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    if n_bootstrap <= 0 or df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    oracle_vals, _ = oracle_scores(df, allowed_policies)
    oracle_arr = oracle_vals.to_numpy(dtype=float)
    rows = []
    n = len(df)
    selected = {
        name: np.asarray([anwg_value(row, p) for (_, row), p in zip(df.iterrows(), preds)], dtype=float)
        for name, preds in preds_by_selector.items()
    }
    for name, arr in selected.items():
        mean_vals = []
        gap_fixed_vals = []
        regret_vals = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            mean_vals.append(float(np.mean(arr[idx])))
            gap_fixed_vals.append(float(np.mean(arr[idx]) - best_fixed_anwg))
            regret_vals.append(float(np.mean(oracle_arr[idx] - arr[idx])))
        rows.append({
            "selector": name,
            "mean_anwg_ci_low": float(np.percentile(mean_vals, 2.5)),
            "mean_anwg_ci_high": float(np.percentile(mean_vals, 97.5)),
            "gap_vs_best_fixed_ci_low": float(np.percentile(gap_fixed_vals, 2.5)),
            "gap_vs_best_fixed_ci_high": float(np.percentile(gap_fixed_vals, 97.5)),
            "mean_oracle_regret_ci_low": float(np.percentile(regret_vals, 2.5)),
            "mean_oracle_regret_ci_high": float(np.percentile(regret_vals, 97.5)),
        })
    return pd.DataFrame(rows)


def subgroup_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    masks = {
        "all": pd.Series(True, index=df.index),
        "azure_derived": df["workload"].astype(str).str.startswith("azure_"),
        "burstgpt_derived": df["workload"].astype(str).str.startswith("burstgpt_"),
        "azure_2023_conv": df["workload"].eq("azure_2023_conv"),
        "azure_conv_like": df.get("is_azure_conv_like", False).astype(bool),
        "meaningful_margin_ge_0.005": ~df.get("is_near_tie_all_non_oracle", False).astype(bool),
    }
    if "external_envelope_anwg" in df.columns:
        selected = df.get("selected_policy_anwg")
        if selected is not None:
            masks["phase2c_external_loss_analysis_only"] = (df["external_envelope_anwg"].astype(float) > selected.astype(float) + 1e-12)
    return masks


def failure_analysis(df: pd.DataFrame, preds: Sequence[str], allowed_policies: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oracle_vals, oracle_policies = oracle_scores(df, allowed_policies)
    rows = []
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    for idx, ((_, row), pred) in enumerate(zip(df.iterrows(), preds)):
        selected = anwg_value(row, pred)
        oracle = float(oracle_vals.iloc[idx])
        regret = max(0.0, oracle - selected)
        if regret <= 1e-12:
            continue
        best = oracle_policies.iloc[idx]
        scored = sorted(((p, anwg_value(row, p)) for p in allowed_policies), key=lambda x: x[1], reverse=True)
        rows.append({
            "trace_id": row.get("trace_id"),
            "workload": row.get("workload"),
            "window_id": int(row.get("window_id", idx)),
            "selected_policy": pred,
            "selected_anwg": selected,
            "oracle_policy": best,
            "oracle_anwg": oracle,
            "regret": regret,
            "policy_margin": scored[0][1] - scored[1][1] if len(scored) > 1 else 0.0,
            "is_near_tie": (scored[0][1] - scored[1][1]) < NEAR_TIE if len(scored) > 1 else True,
            "source_domain": "azure" if str(row.get("workload", "")).startswith("azure_") else "burstgpt" if str(row.get("workload", "")).startswith("burstgpt_") else "synthetic",
            "oracle_minus_scorpio": anwg_value(row, best) - anwg_value(row, "scorpio_style_slo_guard") if "scorpio_style_slo_guard" in allowed_policies else np.nan,
            **{c: row[c] for c in feature_cols},
        })
    failures = pd.DataFrame(rows).sort_values("regret", ascending=False) if rows else pd.DataFrame()
    by_workload = (
        failures.groupby("workload", dropna=False)["regret"].agg(["count", "sum", "mean", "max"]).reset_index().sort_values("sum", ascending=False)
        if not failures.empty else pd.DataFrame()
    )
    by_policy = (
        failures.groupby("oracle_policy", dropna=False)["regret"].agg(["count", "sum", "mean", "max"]).reset_index().sort_values("sum", ascending=False)
        if not failures.empty else pd.DataFrame()
    )
    return failures, by_workload, by_policy


def prior_selector_predictions(df: pd.DataFrame) -> Dict[str, List[str]]:
    out = {}
    for key in ["rule_based", "rf_anwg", "rf_anwg_regret", "dt_anwg", "dt_anwg_regret", "knn_anwg", "regression_anwg"]:
        col = f"sel_{key}_policy"
        if col in df.columns:
            out[f"prior_{key}"] = df[col].astype(str).tolist()
    if "phase2c3_selected_policy" in df.columns:
        out["prior_phase2c3_native_non_oracle_dt"] = df["phase2c3_selected_policy"].astype(str).tolist()
    return out


def choose_best_fixed(df: pd.DataFrame, allowed: Sequence[str]) -> Tuple[str, float]:
    means = {p: float(df[anwg_column(p)].mean()) for p in allowed}
    policy = max(means, key=means.__getitem__)
    return policy, means[policy]


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    dataset_dir = Path(args.dataset_dir)
    dataset_dir = dataset_dir if dataset_dir.is_absolute() else ROOT / dataset_dir
    out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_OUTPUT_ROOT) / utc_stamp()
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, manifest, feature_cols = load_dataset(dataset_dir)
    df = ensure_anwg_and_labels(df, manifest)
    train = df[df["split"].eq("train")].copy()
    val = df[df["split"].eq("val")].copy()
    real_eval = df[df["split"].eq("eval")].copy()

    generated_meta: Dict[str, Any] = {"targeted_augmentation": not args.skip_targeted_augmentation}
    synthetic_fresh = pd.DataFrame()
    if not args.skip_targeted_augmentation:
        train_aug = generate_targeted_rows(manifest=manifest, split="train", seeds=[100, 101] if args.quick else [100, 101, 102], quick=args.quick)
        val_aug = generate_targeted_rows(manifest=manifest, split="val", seeds=[103], quick=args.quick)
        synthetic_fresh = generate_targeted_rows(manifest=manifest, split="synthetic_fresh", seeds=[200] if args.quick else [200, 201, 202], quick=args.quick)
        train = pd.concat([train, train_aug], ignore_index=True, sort=False)
        val = pd.concat([val, val_aug], ignore_index=True, sort=False)
        generated_meta.update({
            "targeted_train_rows": int(len(train_aug)),
            "targeted_val_rows": int(len(val_aug)),
            "targeted_fresh_rows": int(len(synthetic_fresh)),
            "targeted_train_azure_conv_like": int(train_aug.get("is_azure_conv_like", pd.Series(dtype=bool)).sum()) if not train_aug.empty else 0,
            "targeted_val_azure_conv_like": int(val_aug.get("is_azure_conv_like", pd.Series(dtype=bool)).sum()) if not val_aug.empty else 0,
            "targeted_fresh_azure_conv_like": int(synthetic_fresh.get("is_azure_conv_like", pd.Series(dtype=bool)).sum()) if not synthetic_fresh.empty else 0,
        })

    frames = [train, val, real_eval] + ([synthetic_fresh] if not synthetic_fresh.empty else [])
    aligned, feature_cols = align_features(frames, feature_cols)
    train, val, real_eval = aligned[:3]
    synthetic_fresh = aligned[3] if len(aligned) > 3 else synthetic_fresh

    validation_rows = []
    final_tables = []
    final_predictions: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    selected_specs: Dict[str, str] = {}
    pools_to_model = ["native_non_oracle", "all_non_oracle"]

    for pool in pools_to_model:
        allowed = policies_for_pool(manifest, pool)
        label_col = f"label_best_{pool}_policy"
        best_fixed_policy, best_fixed_val = choose_best_fixed(val, allowed)
        specs = selector_specs(args.quick, args.include_all_pairs)
        candidates = {}
        for spec in specs:
            try:
                selector = fit_spec(spec, train, allowed=allowed, feature_cols=feature_cols, label_col=label_col)
            except Exception as exc:
                validation_rows.append({
                    "pool": pool,
                    "selector": spec.name,
                    "status": "fit_failed",
                    "error": repr(exc),
                })
                continue
            candidates[spec.name] = (spec, selector)

        val_scores: Dict[str, float] = {}
        best_fixed_policy_val, best_fixed_anwg_val = choose_best_fixed(val, allowed)

        def evaluate_candidate_on_val(name: str, selector, selector_type: str = "learned_validation") -> None:
            preds = selector.predict(val)
            row = evaluate_predictions(
                val,
                preds,
                selector_name=name,
                selector_type=selector_type,
                allowed_policies=allowed,
                oracle_name=f"{pool}_oracle",
                split_name="validation",
                best_fixed_anwg=best_fixed_anwg_val,
            )
            row["pool"] = pool
            row["status"] = "ok"
            row["validation_best_fixed_policy"] = best_fixed_policy_val
            validation_rows.append(row)
            val_scores[name] = float(row["mean_anwg"])

        for name, (_spec, selector) in list(candidates.items()):
            evaluate_candidate_on_val(name, selector)

        # Add deployable uncertainty fallbacks only around the best validation
        # regression base. This avoids a large wrapper grid while still testing
        # the formulation requested for this phase.
        regression_bases = [
            name for name, (_spec, selector) in candidates.items()
            if "regression" in name and hasattr(selector, "predict_scores") and name in val_scores
        ]
        if regression_bases:
            best_reg_base = max(regression_bases, key=lambda n: val_scores[n])
            fallback_choices = []
            for p in [best_fixed_policy, "scorpio_style_slo_guard", "weighted_shortest_processing", "admission_control"]:
                if p in allowed and p not in fallback_choices:
                    fallback_choices.append(p)
            for fallback in fallback_choices:
                for threshold in [0.001, 0.005, 0.010]:
                    name = f"{best_reg_base}_uncertain_fallback_{fallback}_m{threshold:.3f}"
                    selector = UncertaintyFallbackSelector(
                        name=name,
                        base_selector=candidates[best_reg_base][1],
                        fallback_policy=fallback,
                        margin_threshold=threshold,
                    )
                    candidates[name] = (ModelSpec(name, "uncertainty_fallback"), selector)
                    evaluate_candidate_on_val(name, selector)

        # Regime gate: validate only if the training/validation augmentation created gate-positive examples.
        gate_train = train[train.apply(lambda r: azure_conv_like_gate(r), axis=1)]
        if len(gate_train) >= 5 and candidates:
            global_name = max(val_scores, key=val_scores.__getitem__)
            specialist_spec = ModelSpec("azure_conv_like_specialist_extra_regression", "regression", "extra_trees", "margin_plus_epsilon", 120 if args.quick else 240, 8)
            try:
                specialist = fit_spec(specialist_spec, gate_train, allowed=allowed, feature_cols=feature_cols, label_col=label_col)
                gated = RegimeGatedSelector(
                    name=f"regime_gated_{global_name}_azure_conv_like",
                    gate=azure_conv_like_gate,
                    specialist_selector=specialist,
                    default_selector=candidates[global_name][1],
                )
                candidates[gated.name] = (ModelSpec(gated.name, "regime_gated"), gated)
                evaluate_candidate_on_val(gated.name, gated)
            except Exception as exc:
                validation_rows.append({"pool": pool, "selector": "regime_gated", "status": "fit_failed", "error": repr(exc)})

        ok_val = [r for r in validation_rows if r.get("pool") == pool and r.get("status") == "ok"]
        best_name = max(ok_val, key=lambda r: (r["mean_anwg"], r["gap_closed_fraction"]))["selector"]
        selected_specs[pool] = best_name

        # Refit all non-failed specs on train+val after model choice is frozen, then evaluate.
        train_final = pd.concat([train, val], ignore_index=True, sort=False)
        final_candidates = {}
        for name, (spec, _old_selector) in candidates.items():
            if spec.family in {"uncertainty_fallback", "regime_gated"}:
                # Reuse already-fitted wrappers; base models were validation-fit. This is conservative
                # and avoids accidentally changing the frozen validation-selected structure.
                final_candidates[name] = _old_selector
                continue
            try:
                final_candidates[name] = fit_spec(spec, train_final, allowed=allowed, feature_cols=feature_cols, label_col=label_col)
            except Exception:
                continue

        eval_splits = {"phase2c_real_eval": real_eval}
        if not synthetic_fresh.empty:
            eval_splits["targeted_synthetic_fresh"] = synthetic_fresh

        for split_name, eval_df in eval_splits.items():
            if eval_df.empty:
                continue
            best_fixed_policy_eval, best_fixed_anwg = choose_best_fixed(eval_df, allowed)
            preds_by_selector: Dict[str, List[str]] = {}

            # Fixed policies and prior selector columns for the real eval split.
            fixed_table = fixed_policy_table(eval_df, allowed, split_name=split_name)
            fixed_table["pool"] = pool
            final_tables.append(fixed_table)
            if split_name == "phase2c_real_eval":
                preds_by_selector.update(prior_selector_predictions(eval_df))

            for name, selector in final_candidates.items():
                preds_by_selector[f"new_{name}"] = selector.predict(eval_df)

            rows = []
            for name, preds in preds_by_selector.items():
                rows.append(evaluate_predictions(
                    eval_df,
                    preds,
                    selector_name=name,
                    selector_type="prior_selector" if name.startswith("prior_") else "new_selector",
                    allowed_policies=allowed,
                    oracle_name=f"{pool}_oracle",
                    split_name=split_name,
                    best_fixed_anwg=best_fixed_anwg,
                ))
            table = pd.DataFrame(rows)
            table["pool"] = pool
            table["best_fixed_policy"] = best_fixed_policy_eval
            table["best_fixed_anwg"] = best_fixed_anwg
            oracle_vals, _ = oracle_scores(eval_df, allowed)
            table["oracle_envelope_anwg"] = float(oracle_vals.mean())
            final_tables.append(table)
            final_predictions[(pool, split_name)] = preds_by_selector

            ci = bootstrap_summary(
                eval_df,
                preds_by_selector,
                allowed_policies=allowed,
                best_fixed_anwg=best_fixed_anwg,
                n_bootstrap=args.bootstrap,
            )
            if not ci.empty:
                ci["pool"] = pool
                ci["split"] = split_name
                ci.to_csv(out_dir / f"bootstrap_ci_{pool}_{split_name}.csv", index=False)

            subgroup_rows = []
            for subgroup, mask in subgroup_masks(eval_df).items():
                sub = eval_df[mask].copy()
                if sub.empty:
                    continue
                sub_best_policy, sub_best_anwg = choose_best_fixed(sub, allowed)
                for name, preds_all in preds_by_selector.items():
                    preds_sub = [p for p, keep in zip(preds_all, mask.to_numpy()) if bool(keep)]
                    subgroup_rows.append(evaluate_predictions(
                        sub,
                        preds_sub,
                        selector_name=name,
                        selector_type="prior_selector" if name.startswith("prior_") else "new_selector",
                        allowed_policies=allowed,
                        oracle_name=f"{pool}_oracle",
                        split_name=f"{split_name}:{subgroup}",
                        best_fixed_anwg=sub_best_anwg,
                    ) | {
                        "pool": pool,
                        "subgroup": subgroup,
                        "best_fixed_policy": sub_best_policy,
                        "best_fixed_anwg": sub_best_anwg,
                    })
            pd.DataFrame(subgroup_rows).to_csv(out_dir / f"subgroup_summary_{pool}_{split_name}.csv", index=False)

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(out_dir / "validation_model_selection.csv", index=False)
    final_df = pd.concat(final_tables, ignore_index=True, sort=False) if final_tables else pd.DataFrame()
    final_df.to_csv(out_dir / "final_evaluation_summary.csv", index=False)

    # Choose final headline from pre-specified candidates only. Do not select
    # the best final-eval row across the whole exploratory grid; that would be
    # test-set tuning. The frozen new selector is validation-selected. The
    # current baseline is the prior Phase 2C.3 selector when present.
    headline_pool = "all_non_oracle" if "all_non_oracle" in selected_specs else "native_non_oracle"
    headline_split = "phase2c_real_eval"
    headline_rows = final_df[
        final_df.get("pool", pd.Series(dtype=str)).eq(headline_pool)
        & final_df.get("split", pd.Series(dtype=str)).eq(headline_split)
        & final_df.get("selector_type", pd.Series(dtype=str)).isin(["new_selector", "prior_selector"])
    ].copy()
    frozen_new_selector_name = f"new_{selected_specs.get(headline_pool, '')}"
    current_baseline_candidates = [
        "prior_phase2c3_native_non_oracle_dt",
        "prior_dt_anwg",
        "prior_rf_anwg",
        "prior_regression_anwg",
    ]
    eligible_names = [frozen_new_selector_name] if frozen_new_selector_name in set(headline_rows.get("selector", [])) else []
    eligible_names.extend(name for name in current_baseline_candidates if name in set(headline_rows.get("selector", [])))
    eligible = headline_rows[headline_rows["selector"].isin(eligible_names)].copy()
    best_selector_name = (
        str(eligible.sort_values(["mean_anwg", "gap_closed_fraction"], ascending=False).iloc[0]["selector"])
        if not eligible.empty else ""
    )
    diagnostic_best_final_selector = (
        str(headline_rows.sort_values(["mean_anwg", "gap_closed_fraction"], ascending=False).iloc[0]["selector"])
        if not headline_rows.empty else ""
    )

    if best_selector_name:
        preds = final_predictions[(headline_pool, headline_split)][best_selector_name]
        failures, by_workload, by_policy = failure_analysis(real_eval, preds, policies_for_pool(manifest, headline_pool))
        failures.to_csv(out_dir / "oracle_regret_failures.csv", index=False)
        failures.head(20).to_csv(out_dir / "top20_oracle_regret_failures.csv", index=False)
        by_workload.to_csv(out_dir / "regret_by_workload.csv", index=False)
        by_policy.to_csv(out_dir / "regret_by_oracle_policy.csv", index=False)

    # External-style envelope diagnostics on real eval.
    external_policies = policies_for_pool(manifest, "external_style")
    all_policies = policies_for_pool(manifest, "all_non_oracle")
    native_policies = policies_for_pool(manifest, "native_non_oracle")
    external_vals, external_winners = oracle_scores(real_eval, external_policies)
    all_vals, all_winners = oracle_scores(real_eval, all_policies)
    native_vals, native_winners = oracle_scores(real_eval, native_policies)
    envelope_diag = pd.DataFrame({
        "trace_id": real_eval["trace_id"],
        "workload": real_eval["workload"],
        "window_id": real_eval["window_id"],
        "native_envelope_anwg": native_vals,
        "native_envelope_policy": native_winners,
        "external_style_envelope_anwg": external_vals,
        "external_style_envelope_policy": external_winners,
        "all_non_oracle_envelope_anwg": all_vals,
        "all_non_oracle_envelope_policy": all_winners,
        "external_minus_native": external_vals.to_numpy() - native_vals.to_numpy(),
        "all_minus_native": all_vals.to_numpy() - native_vals.to_numpy(),
    })
    envelope_diag.to_csv(out_dir / "external_envelope_diagnostics.csv", index=False)

    runtime = time.perf_counter() - start
    best_summary = {}
    if best_selector_name:
        best_row = headline_rows[headline_rows["selector"].eq(best_selector_name)].iloc[0].to_dict()
        best_summary = {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in best_row.items()}
    run_manifest = {
        "experiment": "phase2c_final_selector_improvement",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(out_dir),
        "git_note": "record exact commit via git rev-parse HEAD in final audit",
        "feature_columns": feature_cols,
        "near_tie_threshold": NEAR_TIE,
        "bootstrap_replicates": args.bootstrap,
        "quick": args.quick,
        "selected_specs": selected_specs,
        "headline_pool": headline_pool,
        "headline_split": headline_split,
        "frozen_new_selector_name": frozen_new_selector_name,
        "diagnostic_best_final_selector_not_selection_valid": diagnostic_best_final_selector,
        "best_selector_name": best_selector_name,
        "best_summary": best_summary,
        "generated_meta": generated_meta,
        "runtime_seconds": runtime,
    }
    write_json(out_dir / "run_manifest.json", run_manifest)

    print(f"Output: {out_dir}")
    print(f"Runtime seconds: {runtime:.1f}")
    print(f"Headline pool: {headline_pool}")
    print(f"Best selector on real eval: {best_selector_name}")
    if best_summary:
        print(f"Best ANWG: {best_summary.get('mean_anwg'):.6f}")
        print(f"Best fixed: {best_summary.get('best_fixed_policy')} {best_summary.get('best_fixed_anwg'):.6f}")
        print(f"Oracle envelope: {best_summary.get('oracle_envelope_anwg'):.6f}")
        print(f"Gap closed: {best_summary.get('gap_closed_fraction'):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
