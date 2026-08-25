#!/usr/bin/env python3
"""
Phase 2C labeled-dataset builder.

Generates a reproducible labeled dataset for selector training and failure
analysis from Phase 2C.2 / 2C.3 simulator outputs.

Labels are derived exclusively from ANWG = reward_* * completion_* computed
from simulator measurements.  No live API is called.  API annotation is
disabled; only mock fields may be added via --mock-api-annotations.

Usage:
    python scripts/build_phase2c_labeled_selector_dataset.py \\
        --config configs/phase2c_labeled_selector_dataset.yaml \\
        --dry-run

    python scripts/build_phase2c_labeled_selector_dataset.py \\
        --config configs/phase2c_labeled_selector_dataset.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = "configs/phase2c_labeled_selector_dataset.yaml"
DEFAULT_OUTPUT_ROOT = "results/phase2c_labeled_selector_dataset"

EXTERNAL_STYLE_POLICIES = [
    "orca_style",
    "vllm_style_token_budget",
    "sarathi_style",
    "splitfuse_style",
    "multi_bin_batching",
    "estimated_service_time_first",
    "scorpio_style_slo_guard",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _repo_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ── policy discovery & reconstruction ────────────────────────────────────────

def discover_policies(df: pd.DataFrame) -> List[str]:
    reward = {c[len("reward_"):] for c in df.columns if c.startswith("reward_")}
    completion = {c[len("completion_"):] for c in df.columns if c.startswith("completion_")}
    return sorted(reward & completion)


def reconstruct_anwg(df: pd.DataFrame, policies: Sequence[str]) -> pd.DataFrame:
    """Add anwg_<policy> = reward_<policy> * completion_<policy> for each policy."""
    out = df.copy()
    for p in policies:
        out[f"anwg_{p}"] = out[f"reward_{p}"].astype(float) * out[f"completion_{p}"].astype(float)
    return out


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    _bad = ("reward_", "completion_", "sel_", "best_", "label", "anwg_",
            "external_", "selected_", "slo_violation_")
    return [c for c in df.columns
            if c.startswith("feat_") and not any(t in c.lower() for t in _bad)]


# ── policy pools ──────────────────────────────────────────────────────────────

def resolve_pools(cfg: dict, candidate_policies: Sequence[str]) -> Dict[str, List[str]]:
    candidate_set = set(candidate_policies)
    pool_cfg: dict = cfg.get("policy_pools", {})

    def _filter_oracle(policies, oracle_prefixes):
        prefixes = tuple(oracle_prefixes)
        return [p for p in policies if not p.startswith(prefixes)]

    pools: Dict[str, List[str]] = {}

    # native_non_oracle
    p_cfg = pool_cfg.get("native_non_oracle", {})
    excluded = set(p_cfg.get("excluded_policies", []))
    oracle_pfx = p_cfg.get("oracle_prefixes", ["safe_fallback_wsp_margin"])
    native = _filter_oracle([p for p in candidate_policies if p not in excluded], oracle_pfx)
    pools["native_non_oracle"] = native

    # external_style
    ext_cfg = pool_cfg.get("external_style", {})
    ext_explicit = ext_cfg.get("policies", EXTERNAL_STYLE_POLICIES)
    pools["external_style"] = [p for p in ext_explicit if p in candidate_set]

    # all_non_oracle
    all_cfg = pool_cfg.get("all_non_oracle", {})
    oracle_pfx2 = all_cfg.get("oracle_prefixes", ["safe_fallback_wsp_margin"])
    pools["all_non_oracle"] = _filter_oracle(list(candidate_policies), oracle_pfx2)

    # orca_vs_scorpio
    ovs_cfg = pool_cfg.get("orca_vs_scorpio", {})
    ovs_explicit = ovs_cfg.get("policies", ["orca_style", "scorpio_style_slo_guard"])
    pools["orca_vs_scorpio"] = [p for p in ovs_explicit if p in candidate_set]

    return pools


# ── label generation ──────────────────────────────────────────────────────────

_POOL_NEAR_TIE_SHORT = {
    "native_non_oracle": "native",
    "external_style": "external",
    "all_non_oracle": "all_non_oracle",
    "orca_vs_scorpio": "orca_vs_scorpio",
}


def compute_policy_choice_labels(
    df: pd.DataFrame,
    pool_name: str,
    policies: Sequence[str],
    near_tie_margin: float,
) -> pd.DataFrame:
    """
    Add columns:
      label_best_{pool_name}_policy
      best_{pool_name}_anwg          (analysis-only; carry-over from anwg_ reconstruction)
      margin_best_{pool_name}
      is_near_tie_{short_name}
    """
    short = _POOL_NEAR_TIE_SHORT.get(pool_name, pool_name)
    allowed = list(policies)
    label_vals, best_anwg_vals, margin_vals, near_tie_vals = [], [], [], []
    for _, row in df.iterrows():
        scored = sorted(
            ((p, float(row[f"anwg_{p}"])) for p in allowed),
            key=lambda x: -x[1],
        )
        best_p, best_a = scored[0]
        second_a = scored[1][1] if len(scored) > 1 else best_a
        margin = best_a - second_a
        label_vals.append(best_p)
        best_anwg_vals.append(best_a)
        margin_vals.append(round(margin, 8))
        near_tie_vals.append(bool(margin < near_tie_margin))
    out = df.copy()
    out[f"label_best_{pool_name}_policy"] = label_vals
    out[f"best_{pool_name}_anwg"] = best_anwg_vals
    out[f"margin_best_{pool_name}"] = margin_vals
    out[f"is_near_tie_{short}"] = near_tie_vals
    return out


def compute_pairwise_orca_scorpio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns comparing orca_style vs scorpio_style_slo_guard per window.
    Returns a DataFrame with orca-vs-scorpio pairwise columns.
    """
    out = df.copy()
    orca_anwg = out["anwg_orca_style"].astype(float)
    scorp_anwg = out["anwg_scorpio_style_slo_guard"].astype(float)
    orca_comp = out["completion_orca_style"].astype(float)
    scorp_comp = out["completion_scorpio_style_slo_guard"].astype(float)
    orca_rwd = out["reward_orca_style"].astype(float)
    scorp_rwd = out["reward_scorpio_style_slo_guard"].astype(float)
    diff = orca_anwg - scorp_anwg
    out["label_orca_beats_scorpio"] = (diff > 0).astype(bool)
    out["orca_minus_scorpio_anwg"] = (diff).round(8)
    out["orca_minus_scorpio_completion"] = (orca_comp - scorp_comp).round(8)
    out["orca_minus_scorpio_quality"] = (orca_rwd - scorp_rwd).round(8)
    out["orca_better_by_completion"] = (orca_comp > scorp_comp + 1e-8).astype(bool)
    out["orca_better_by_quality"] = (orca_rwd > scorp_rwd + 1e-8).astype(bool)
    return out


def compute_regime_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add regime-membership columns based on causal feat_* features and workload names.
    is_azure_conv_like is feature-driven (threshold-based), NOT just workload name.
    """
    thresholds: dict = cfg.get("regime_thresholds", {})
    high_arr = float(thresholds.get("high_arrival_rate", 10.0))
    high_burst = float(thresholds.get("high_burstiness_cv", 1.0))
    long_prompt = float(thresholds.get("long_prompt_tokens", 1000))
    slo_low = float(thresholds.get("mixed_tight_slo_low", 0.4))
    slo_high = float(thresholds.get("mixed_tight_slo_high", 0.7))

    rsr = cfg.get("realistic_subset_rules", {})
    exact_workload = str(rsr.get("exclude_exact_prediction_workload", "burstgpt_moderate_exact_prediction"))
    overlap_workload = str(rsr.get("exclude_first_n_windows_of", "burstgpt_scaled_high"))
    n_overlap = int(rsr.get("n_windows_to_exclude", 2))

    out = df.copy()
    workload = out["workload"]

    # Workload-name membership
    out["is_azure"] = workload.str.startswith("azure_").astype(bool)
    out["is_burstgpt"] = workload.str.startswith("burstgpt_").astype(bool)
    out["is_azure_conv"] = workload.eq("azure_2023_conv").astype(bool)
    out["is_azure_code"] = workload.eq("azure_2023_code").astype(bool)
    out["is_exact_prediction_oracle_like"] = workload.eq(exact_workload).astype(bool)

    # Overlap-sensitive windows (first N of burstgpt_scaled_high)
    out = out.sort_values(["workload", "window_id"]).copy()
    out["_window_rank"] = out.groupby("workload").cumcount()
    out["is_overlap_sensitive_first_two"] = (
        workload.eq(overlap_workload) & (out["_window_rank"] < n_overlap)
    ).astype(bool)
    out["is_realistic_subset"] = ~(
        out["is_exact_prediction_oracle_like"] | out["is_overlap_sensitive_first_two"]
    )
    out = out.drop(columns=["_window_rank"])

    # Feature-based regime labels
    arr = out["feat_arrival_rate_est"].astype(float)
    burst = out["feat_burstiness_cv"].astype(float)
    prompt = out["feat_mean_prompt_tokens"].astype(float)
    tight_slo = out["feat_fraction_tight_slo"].astype(float)

    out["is_high_arrival_rate"] = (arr > high_arr).astype(bool)
    out["is_high_burstiness"] = (burst > high_burst).astype(bool)
    out["is_long_prompt"] = (prompt > long_prompt).astype(bool)
    out["is_mixed_tight_slo"] = ((tight_slo >= slo_low) & (tight_slo <= slo_high)).astype(bool)

    # is_azure_conv_like: feature-based (NOT just workload name)
    # Characteristic of azure_2023_conv: long prompts AND mixed tight SLO
    out["is_azure_conv_like"] = (out["is_long_prompt"] & out["is_mixed_tight_slo"]).astype(bool)

    return out


def compute_failure_labels(
    eval_df: pd.DataFrame,
    phase2c3_predictions: pd.DataFrame,
    external_policies: Sequence[str],
    phase2c3_best_selector: str,
) -> pd.DataFrame:
    """
    Add failure labels to eval rows only.
    Compares Phase 2C.2 dt_anwg selector and Phase 2C.3 best selector
    against per-window external-style envelope.
    """
    out = eval_df.copy()

    # External-style envelope per window
    ext_anwg_matrix = out[[f"anwg_{p}" for p in external_policies]].astype(float)
    out["external_envelope_anwg"] = ext_anwg_matrix.max(axis=1)
    out["external_winner_policy"] = ext_anwg_matrix.idxmax(axis=1).str.replace("anwg_", "", regex=False)

    # Best fixed external policy overall
    mean_ext = {p: float(out[f"anwg_{p}"].mean()) for p in external_policies}
    best_fixed_ext_policy = max(mean_ext, key=mean_ext.__getitem__)
    out["best_fixed_external_policy"] = best_fixed_ext_policy
    out["best_fixed_external_anwg"] = round(mean_ext[best_fixed_ext_policy], 8)

    # Phase 2C.2 dt_anwg selector failure label
    if "sel_dt_anwg_policy" in out.columns:
        dt_sel_anwg = out.apply(
            lambda row: float(row[f"anwg_{row['sel_dt_anwg_policy']}"]), axis=1
        )
        out["selected_policy"] = out["sel_dt_anwg_policy"]
        out["selected_policy_anwg"] = dt_sel_anwg.round(8)
        out["label_phase2c2_dt_loses_to_external_envelope"] = (
            out["external_envelope_anwg"] > dt_sel_anwg + 1e-12
        ).astype(bool)
    else:
        out["selected_policy"] = pd.NA
        out["selected_policy_anwg"] = pd.NA
        out["label_phase2c2_dt_loses_to_external_envelope"] = pd.NA

    # Phase 2C.3 best selector failure label
    # Join from per_window_predictions filtered to best selector
    c3_sub = phase2c3_predictions[
        phase2c3_predictions["selector"].eq(phase2c3_best_selector)
    ][["workload", "window_id", "selected_anwg", "predicted_policy"]].copy()
    c3_sub = c3_sub.rename(columns={
        "selected_anwg": "phase2c3_selected_anwg",
        "predicted_policy": "phase2c3_selected_policy",
    })

    # Add workload column to eval_df for join
    if "workload" not in out.columns:
        out["workload"] = out["trace_id"].map(lambda x: x.rsplit("_s", 1)[0] if "_s" in x else x)

    out = out.merge(
        c3_sub,
        on=["workload", "window_id"],
        how="left",
    )
    out["label_phase2c3_best_loses_to_external_envelope"] = (
        out["external_envelope_anwg"] > out["phase2c3_selected_anwg"].astype(float) + 1e-12
    ).astype(bool)
    out["external_loss_magnitude"] = (
        out["external_envelope_anwg"] - out["phase2c3_selected_anwg"].astype(float)
    ).clip(lower=0).round(8)

    return out


# ── Phase 2C.2 reference verification ────────────────────────────────────────

def verify_phase2c2_references(
    eval_df: pd.DataFrame,
    policies: Sequence[str],
    external_policies: Sequence[str],
    cfg: dict,
) -> Dict[str, Any]:
    ref_cfg = cfg.get("phase2c2_reference_check", {})
    tolerance = float(ref_cfg.get("tolerance", 5e-4))
    expected_loss_windows = int(ref_cfg.get("external_loss_windows", 62))

    dt_sel_anwg = eval_df.apply(
        lambda row: float(row[f"anwg_{row['sel_dt_anwg_policy']}"]), axis=1
    )
    dt_anwg = float(dt_sel_anwg.mean())
    always_scorpio = float(eval_df["anwg_scorpio_style_slo_guard"].mean())
    ext_anwg_matrix = eval_df[[f"anwg_{p}" for p in external_policies]].astype(float)
    external_envelope = float(ext_anwg_matrix.max(axis=1).mean())
    external_loss_windows = int((ext_anwg_matrix.max(axis=1) > dt_sel_anwg + 1e-12).sum())

    actual = {
        "dt_anwg": dt_anwg,
        "always_scorpio": always_scorpio,
        "external_style_envelope": external_envelope,
        "external_loss_windows": external_loss_windows,
    }
    failures = []
    for key in ("dt_anwg", "always_scorpio", "external_style_envelope"):
        if abs(actual[key] - ref_cfg.get(key, actual[key])) > tolerance:
            failures.append(f"{key}: got {actual[key]:.6f}, expected {ref_cfg.get(key):.4f}")
    if external_loss_windows != expected_loss_windows:
        failures.append(
            f"external_loss_windows: got {external_loss_windows}, expected {expected_loss_windows}"
        )
    return {"actual": actual, "failures": failures}


# ── dataset schema ────────────────────────────────────────────────────────────

def build_schema() -> Dict[str, Any]:
    """Label schema with safety metadata for each label column."""
    def _s(desc, safe=True, analysis=False, oracle=False, ext=False) -> dict:
        return {
            "description": desc,
            "safe_for_training": safe,
            "analysis_only": analysis,
            "oracle_like_sensitive": oracle,
            "external_approximation_sensitive": ext,
        }

    return {
        # Policy-choice labels
        "label_best_native_non_oracle_policy": _s(
            "Best policy in the native non-oracle pool by ANWG.",
            safe=True, analysis=False,
        ),
        "margin_best_native_non_oracle": _s(
            "ANWG gap between best and second-best native non-oracle policy.",
            safe=True,
        ),
        "is_near_tie_native": _s(
            "True when native non-oracle margin < near_tie_margin.",
            safe=True,
        ),
        "label_best_external_style_policy": _s(
            "Best policy among external-style approximations by ANWG.",
            safe=False, analysis=True, ext=True,
        ),
        "margin_best_external_style": _s(
            "ANWG gap between best and second-best external-style policy.",
            safe=False, analysis=True, ext=True,
        ),
        "is_near_tie_external": _s(
            "True when external-style margin < near_tie_margin.",
            safe=False, analysis=True, ext=True,
        ),
        "label_best_all_non_oracle_policy": _s(
            "Best policy across all non-oracle candidates by ANWG.",
            safe=False, analysis=True, ext=True,
        ),
        "margin_best_all_non_oracle": _s(
            "ANWG gap between best and second-best across all non-oracle candidates.",
            safe=False, analysis=True, ext=True,
        ),
        "is_near_tie_all_non_oracle": _s(
            "True when all-non-oracle margin < near_tie_margin.",
            safe=False, analysis=True, ext=True,
        ),
        "label_best_orca_vs_scorpio_policy": _s(
            "Better policy between orca_style and scorpio_style_slo_guard by ANWG.",
            safe=False, analysis=True, ext=True,
        ),
        "margin_orca_vs_scorpio": _s(
            "ANWG gap between orca_style and scorpio_style_slo_guard.",
            safe=False, analysis=True, ext=True,
        ),
        "is_near_tie_orca_vs_scorpio": _s(
            "True when orca_vs_scorpio margin < near_tie_margin.",
            safe=False, analysis=True, ext=True,
        ),
        # Pairwise orca-scorpio
        "label_orca_beats_scorpio": _s(
            "True when orca_style ANWG > scorpio_style_slo_guard ANWG.",
            safe=False, analysis=True, ext=True,
        ),
        "orca_minus_scorpio_anwg": _s(
            "ANWG difference: orca_style - scorpio_style_slo_guard.",
            safe=False, analysis=True, ext=True,
        ),
        "orca_minus_scorpio_completion": _s(
            "Completion fraction difference: orca - scorpio.",
            safe=False, analysis=True, ext=True,
        ),
        "orca_minus_scorpio_quality": _s(
            "Completed-request quality difference: orca - scorpio.",
            safe=False, analysis=True, ext=True,
        ),
        "orca_better_by_completion": _s(
            "True when orca achieves higher completion fraction than scorpio.",
            safe=False, analysis=True, ext=True,
        ),
        "orca_better_by_quality": _s(
            "True when orca achieves higher completed-request quality than scorpio.",
            safe=False, analysis=True, ext=True,
        ),
        # Failure labels (eval only)
        "label_phase2c2_dt_loses_to_external_envelope": _s(
            "True when Phase 2C.2 dt_anwg selector ANWG < per-window external envelope.",
            safe=False, analysis=True, ext=True,
        ),
        "label_phase2c3_best_loses_to_external_envelope": _s(
            "True when Phase 2C.3 best selector ANWG < per-window external envelope.",
            safe=False, analysis=True, ext=True,
        ),
        "external_loss_magnitude": _s(
            "ANWG shortfall of Phase 2C.3 best selector vs external envelope (≥0).",
            safe=False, analysis=True, ext=True,
        ),
        "external_winner_policy": _s(
            "The external-style policy that achieved the highest ANWG on this window.",
            safe=False, analysis=True, ext=True,
        ),
        "selected_policy": _s(
            "Policy chosen by Phase 2C.2 dt_anwg selector on this eval window.",
            safe=False, analysis=True,
        ),
        "selected_policy_anwg": _s(
            "ANWG achieved by Phase 2C.2 dt_anwg selector on this eval window.",
            safe=False, analysis=True,
        ),
        "external_envelope_anwg": _s(
            "Per-window maximum ANWG across external-style policies.",
            safe=False, analysis=True, ext=True,
        ),
        "best_fixed_external_policy": _s(
            "Best fixed (not per-window) external-style policy by mean ANWG.",
            safe=False, analysis=True, ext=True,
        ),
        "best_fixed_external_anwg": _s(
            "Mean ANWG of the best fixed external-style policy.",
            safe=False, analysis=True, ext=True,
        ),
        # Regime labels
        "is_train": _s("Row belongs to the training split.", safe=True),
        "is_val": _s("Row belongs to the validation split.", safe=True),
        "is_eval": _s(
            "Row belongs to the held-out evaluation split (Phase 2C.1 real traces).",
            safe=False, analysis=True,
        ),
        "is_azure": _s("Workload name starts with 'azure_'.", safe=True),
        "is_burstgpt": _s("Workload name starts with 'burstgpt_'.", safe=True),
        "is_azure_conv": _s("Workload is azure_2023_conv.", safe=True),
        "is_azure_code": _s("Workload is azure_2023_code.", safe=True),
        "is_exact_prediction_oracle_like": _s(
            "Workload is burstgpt_moderate_exact_prediction (oracle-like output prediction).",
            safe=False, oracle=True,
        ),
        "is_overlap_sensitive_first_two": _s(
            "One of the first two windows of burstgpt_scaled_high (potential overlap).",
            safe=False, analysis=True,
        ),
        "is_realistic_subset": _s(
            "Row survives both the exact-prediction and first-two-overlap exclusions.",
            safe=True,
        ),
        "is_high_arrival_rate": _s(
            "feat_arrival_rate_est > regime_thresholds.high_arrival_rate.", safe=True,
        ),
        "is_high_burstiness": _s(
            "feat_burstiness_cv > regime_thresholds.high_burstiness_cv.", safe=True,
        ),
        "is_long_prompt": _s(
            "feat_mean_prompt_tokens > regime_thresholds.long_prompt_tokens.", safe=True,
        ),
        "is_mixed_tight_slo": _s(
            "feat_fraction_tight_slo in [mixed_tight_slo_low, mixed_tight_slo_high].", safe=True,
        ),
        "is_azure_conv_like": _s(
            "Feature-based azure_2023_conv proxy: is_long_prompt AND is_mixed_tight_slo. "
            "Can be true for any workload that matches the feature profile.",
            safe=True,
        ),
    }


# ── assemble unified dataset ──────────────────────────────────────────────────

def workload_from_trace_id(trace_id: str) -> str:
    return trace_id.rsplit("_s", 1)[0] if "_s" in trace_id else trace_id


def assemble_rows(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: List[str],
    policies: List[str],
) -> pd.DataFrame:
    """
    Produce a single DataFrame with the minimal common columns needed for labeling.
    Adds: workload, split, and all anwg_* columns.
    """
    def _prep(df: pd.DataFrame, split: str) -> pd.DataFrame:
        out = df.copy()
        out["workload"] = out["trace_id"].map(workload_from_trace_id)
        out["split"] = split
        return out

    t = reconstruct_anwg(_prep(train_df, "train"), policies)
    v = reconstruct_anwg(_prep(val_df, "val"), policies)
    e = reconstruct_anwg(_prep(eval_df, "eval"), policies)

    keep_base = ["trace_id", "window_id", "workload", "split"] + feature_cols
    reward_comp_slo = (
        [f"reward_{p}" for p in policies]
        + [f"completion_{p}" for p in policies]
        + [f"slo_violation_{p}" for p in policies if f"slo_violation_{p}" in t.columns]
    )
    anwg_cols = [f"anwg_{p}" for p in policies]

    def _keep(df: pd.DataFrame) -> pd.DataFrame:
        cols = keep_base + [c for c in reward_comp_slo + anwg_cols if c in df.columns]
        # Add sel_* columns if they exist (eval only)
        sel_cols = [c for c in df.columns if c.startswith("sel_")]
        return df[[c for c in cols + sel_cols if c in df.columns]].copy()

    combined = pd.concat([_keep(t), _keep(v), _keep(e)], ignore_index=True)
    return combined


# ── label distribution summary ────────────────────────────────────────────────

def build_label_distribution(labeled_df: pd.DataFrame) -> List[Dict[str, Any]]:
    pool_label_cols = [
        ("native_non_oracle", "label_best_native_non_oracle_policy"),
        ("external_style", "label_best_external_style_policy"),
        ("all_non_oracle", "label_best_all_non_oracle_policy"),
        ("orca_vs_scorpio", "label_best_orca_vs_scorpio_policy"),
    ]
    rows = []
    for pool_name, col in pool_label_cols:
        if col not in labeled_df.columns:
            continue
        for split in ("train", "val", "eval", "all"):
            sub = labeled_df if split == "all" else labeled_df[labeled_df["split"].eq(split)]
            if sub.empty:
                continue
            counts = Counter(sub[col].dropna())
            for label, count in sorted(counts.items()):
                rows.append({
                    "pool": pool_name,
                    "split": split,
                    "label": label,
                    "count": count,
                    "fraction": round(count / max(len(sub), 1), 4),
                })
    return rows


# ── report renderer ───────────────────────────────────────────────────────────

def render_report(
    cfg: dict,
    labeled_df: pd.DataFrame,
    reproduction: Dict[str, Any],
    out_dir: Path,
    feature_cols: List[str],
    pools: Dict[str, List[str]],
    external_policies: List[str],
    mock_api: bool,
) -> str:
    by_split = labeled_df.groupby("split").size().to_dict()
    n_train = by_split.get("train", 0)
    n_val = by_split.get("val", 0)
    n_eval = by_split.get("eval", 0)
    n_total = len(labeled_df)

    ac_like = labeled_df["is_azure_conv_like"].sum() if "is_azure_conv_like" in labeled_df.columns else "n/a"
    near_tie_native = labeled_df["is_near_tie_native"].sum() if "is_near_tie_native" in labeled_df.columns else "n/a"
    ext_loss = labeled_df["label_phase2c3_best_loses_to_external_envelope"].sum() if "label_phase2c3_best_loses_to_external_envelope" in labeled_df.columns else "n/a"

    actual = reproduction.get("actual", {})
    failures = reproduction.get("failures", [])
    repro_status = "PASS" if not failures else f"FAIL: {'; '.join(failures)}"

    lines = [
        "# Phase 2C Labeled Selector Dataset Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Label version:** `{cfg.get('label_version', 'unknown')}`",
        f"**Output:** `{out_dir}`",
        "",
        "## Phase 2C.2 Reference Reproduction",
        f"- Status: **{repro_status}**",
        f"- dt_anwg: {actual.get('dt_anwg', 'n/a'):.6f} (expected 0.8021)",
        f"- always_scorpio: {actual.get('always_scorpio', 'n/a'):.6f} (expected 0.7963)",
        f"- external_style_envelope: {actual.get('external_style_envelope', 'n/a'):.6f} (expected 0.8297)",
        f"- external_loss_windows: {actual.get('external_loss_windows', 'n/a')} (expected 62)",
        "",
        "## Dataset Size",
        f"| Split | Rows |",
        f"|---|---|",
        f"| train | {n_train} |",
        f"| val | {n_val} |",
        f"| eval | {n_eval} |",
        f"| **total** | **{n_total}** |",
        "",
        "## Features",
        f"- Causal feat_* columns: **{len(feature_cols)}**",
        f"- No reward_*, completion_*, sel_*, best_*, anwg_*, external_*, selected_* columns in features.",
        "",
        "## Policy Pools",
    ]
    for pool_name, pool_policies in pools.items():
        lines.append(f"- **{pool_name}**: {len(pool_policies)} policies")
    lines += [
        "",
        "## Key Counts",
        f"- azure_conv_like rows (feature-based): **{ac_like}**",
        f"- near-tie rows (native pool): **{near_tie_native}**",
        f"- external-loss rows (Phase 2C.3 best vs envelope): **{ext_loss}**",
        "",
        "## Label Safety",
        "Labels are classified by safety tier:",
        "- **safe_for_training**: derived from simulator ANWG only, no eval leakage",
        "- **analysis_only**: derived from eval selectors or external-style approximations",
        "- **oracle_like_sensitive**: from oracle-inflated workload (exact prediction)",
        "- **external_approximation_sensitive**: from external-style policy approximations",
        "",
        "Safe-for-training labels include:",
        "  `label_best_native_non_oracle_policy`, `margin_best_native_non_oracle`,",
        "  `is_near_tie_native`, `is_realistic_subset`, all `is_*` regime flags,",
        "  `is_azure_conv_like` (feature-derived, not name-based).",
        "",
        "## API Annotations",
        "- **No live API call was made.**",
        "- API annotation is disabled (`api_annotation.enabled: false` in config).",
        "- LLM/API annotations are NOT used as ground-truth policy labels.",
    ]
    if mock_api:
        lines += [
            "- Mock API annotation fields were added (marked `_mock`). These are NOT labels.",
        ]
    lines += [
        "",
        "## Data Integrity Notes",
        "- ANWG reconstructed from `reward_* * completion_*` (not from pre-computed columns).",
        "- Train/val rows do not have `sel_*` failure labels; those are eval-only.",
        "- `is_azure_conv_like` is feature-based: `is_long_prompt AND is_mixed_tight_slo`.",
        "- `is_exact_prediction_oracle_like` marks burstgpt_moderate_exact_prediction rows.",
        "- Do not use analysis_only or oracle_like_sensitive labels as primary training targets.",
    ]
    return "\n".join(lines) + "\n"


# ── main pipeline ─────────────────────────────────────────────────────────────

def load_inputs(cfg: dict) -> Dict[str, Any]:
    inp = cfg["inputs"]
    pc2 = inp["phase2c2"]
    pc3 = inp["phase2c3"]
    fd = inp["failure_diagnosis"]

    def _csv(key: str) -> pd.DataFrame:
        return pd.read_csv(_repo_path(key))

    def _json(key: str) -> Any:
        with open(_repo_path(key)) as f:
            return json.load(f)

    result: Dict[str, Any] = {
        "training_rows": _csv(pc2["training_rows"]),
        "train_split": _csv(pc2["train_split"]),
        "val_split": _csv(pc2["val_split"]),
        "eval_per_window": _csv(pc2["eval_per_window"]),
        "phase2c3_predictions": _csv(pc3["per_window_predictions"]),
        "phase2c3_selector_summary": _csv(pc3["selector_summary"]),
        "phase2c3_external_loss_cases": _csv(pc3["external_loss_cases"]),
        "top_external_losses": _csv(fd["top_external_losses"]),
        "azure_conv_loss_summary": _json(fd["azure_conv_loss_summary"]),
    }
    return result


def run_pipeline(
    cfg: dict,
    out_dir: Path,
    *,
    dry_run: bool,
    mock_api: bool,
) -> Dict[str, Any]:
    inputs = load_inputs(cfg)

    # ── policy discovery ─────────────────────────────────────────────────────
    train_df = inputs["train_split"]
    val_df = inputs["val_split"]
    eval_df = inputs["eval_per_window"]
    phase2c3_predictions = inputs["phase2c3_predictions"]

    policies = discover_policies(train_df)
    ext_policies = cfg.get("policy_pools", {}).get("external_style", {}).get(
        "policies", EXTERNAL_STYLE_POLICIES
    )
    ext_policies = [p for p in ext_policies if p in policies]
    pools = resolve_pools(cfg, policies)
    near_tie = float(cfg.get("near_tie_margin", 0.005))
    best_selector = str(cfg.get("phase2c3_best_selector", "native_non_oracle_dt"))

    # ── ANWG reconstruction on eval for verification ─────────────────────────
    eval_with_anwg = reconstruct_anwg(eval_df.copy(), policies)
    eval_with_anwg["workload"] = eval_with_anwg["trace_id"].map(workload_from_trace_id)

    # ── Phase 2C.2 reference verification (halt on failure) ──────────────────
    reproduction = verify_phase2c2_references(eval_with_anwg, policies, ext_policies, cfg)
    if reproduction["failures"]:
        raise RuntimeError(
            "Phase 2C.2 reference check FAILED — halting.\n"
            + "\n".join(reproduction["failures"])
        )
    logging.info("Phase 2C.2 reference check passed: %s", reproduction["actual"])

    # ── feature columns ───────────────────────────────────────────────────────
    feature_cols = select_feature_columns(train_df)
    logging.info("Feature columns: %d", len(feature_cols))

    if dry_run:
        return {
            "policies": policies,
            "feature_cols": feature_cols,
            "pools": {k: len(v) for k, v in pools.items()},
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_eval": len(eval_df),
            "reproduction": reproduction["actual"],
            "dry_run": True,
        }

    # ── assemble unified rows ─────────────────────────────────────────────────
    labeled_df = assemble_rows(train_df, val_df, eval_df, feature_cols, policies)

    # ── policy-choice labels for each pool ────────────────────────────────────
    for pool_name, pool_policies in pools.items():
        if not pool_policies:
            continue
        labeled_df = compute_policy_choice_labels(labeled_df, pool_name, pool_policies, near_tie)
    # Rename near-tie columns to match spec
    labeled_df = labeled_df.rename(columns={
        "is_near_tie_native_non_oracle": "is_near_tie_native",
        "is_near_tie_external_style": "is_near_tie_external",
    })

    # ── pairwise orca/scorpio ─────────────────────────────────────────────────
    labeled_df = compute_pairwise_orca_scorpio(labeled_df)

    # ── regime labels ─────────────────────────────────────────────────────────
    labeled_df = compute_regime_labels(labeled_df, cfg)
    labeled_df["is_train"] = labeled_df["split"].eq("train")
    labeled_df["is_val"] = labeled_df["split"].eq("val")
    labeled_df["is_eval"] = labeled_df["split"].eq("eval")

    # ── failure labels (eval rows only) ──────────────────────────────────────
    eval_labeled = eval_with_anwg.copy()
    eval_labeled = compute_regime_labels(eval_labeled, cfg)
    eval_labeled["is_train"] = False
    eval_labeled["is_val"] = False
    eval_labeled["is_eval"] = True

    failure_labeled_eval = compute_failure_labels(
        eval_labeled, phase2c3_predictions, ext_policies, best_selector
    )

    # Merge failure labels back into labeled_df on (workload, window_id) for eval rows
    fail_cols = [
        "workload", "window_id",
        "label_phase2c2_dt_loses_to_external_envelope",
        "label_phase2c3_best_loses_to_external_envelope",
        "external_loss_magnitude",
        "external_winner_policy",
        "selected_policy",
        "selected_policy_anwg",
        "external_envelope_anwg",
        "best_fixed_external_policy",
        "best_fixed_external_anwg",
        "phase2c3_selected_policy",
        "phase2c3_selected_anwg",
    ]
    fail_df = failure_labeled_eval[[c for c in fail_cols if c in failure_labeled_eval.columns]].copy()
    labeled_df = labeled_df.merge(fail_df, on=["workload", "window_id"], how="left")

    # ── mock API annotations ──────────────────────────────────────────────────
    if mock_api:
        labeled_df["api_annotation_regime_summary_mock"] = "MOCK_NOT_A_LABEL"
        labeled_df["api_annotation_is_mock"] = True

    # ── write outputs ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_df.to_csv(out_dir / "labeled_windows.csv", index=False)
    labeled_df[labeled_df["split"].eq("train")].to_csv(out_dir / "train_labeled_windows.csv", index=False)
    labeled_df[labeled_df["split"].eq("val")].to_csv(out_dir / "val_labeled_windows.csv", index=False)
    labeled_df[labeled_df["split"].eq("eval")].to_csv(out_dir / "eval_labeled_windows.csv", index=False)

    # pairwise orca-scorpio table
    pairwise_cols = [
        "trace_id", "window_id", "workload", "split",
        "label_orca_beats_scorpio", "orca_minus_scorpio_anwg",
        "orca_minus_scorpio_completion", "orca_minus_scorpio_quality",
        "orca_better_by_completion", "orca_better_by_quality",
        "anwg_orca_style", "anwg_scorpio_style_slo_guard",
    ]
    labeled_df[[c for c in pairwise_cols if c in labeled_df.columns]].to_csv(
        out_dir / "pairwise_orca_scorpio_labels.csv", index=False
    )

    # external loss table (rows where Phase 2C.3 best loses)
    loss_mask = labeled_df.get("label_phase2c3_best_loses_to_external_envelope", pd.Series(False, index=labeled_df.index))
    labeled_df[loss_mask.fillna(False).astype(bool)].to_csv(
        out_dir / "external_loss_labels.csv", index=False
    )

    # regime labels table
    regime_cols = [
        "trace_id", "window_id", "workload", "split",
        "is_train", "is_val", "is_eval",
        "is_azure", "is_burstgpt", "is_azure_conv", "is_azure_code",
        "is_exact_prediction_oracle_like", "is_overlap_sensitive_first_two",
        "is_realistic_subset", "is_high_arrival_rate", "is_high_burstiness",
        "is_long_prompt", "is_mixed_tight_slo", "is_azure_conv_like",
    ]
    labeled_df[[c for c in regime_cols if c in labeled_df.columns]].to_csv(
        out_dir / "regime_labels.csv", index=False
    )

    _write_text(out_dir / "feature_columns.txt", "\n".join(feature_cols) + "\n")

    schema = build_schema()
    _write_json(out_dir / "dataset_schema.json", schema)

    manifest = {
        "experiment": cfg.get("experiment", "phase2c_labeled_selector_dataset"),
        "label_version": cfg.get("label_version", "phase2c_labels_v1"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": int((labeled_df["split"] == "train").sum()),
        "n_val": int((labeled_df["split"] == "val").sum()),
        "n_eval": int((labeled_df["split"] == "eval").sum()),
        "n_total": len(labeled_df),
        "n_feature_cols": len(feature_cols),
        "policies": policies,
        "pools": {k: v for k, v in pools.items()},
        "external_policies": ext_policies,
        "near_tie_margin": near_tie,
        "phase2c3_best_selector": best_selector,
        "reproduced_phase2c2_metrics": reproduction["actual"],
        "api_annotation_enabled": False,
        "live_api_used": False,
        "mock_api_annotations": mock_api,
    }
    _write_json(out_dir / "dataset_manifest.json", manifest)

    dist_rows = build_label_distribution(labeled_df)
    pd.DataFrame(dist_rows).to_csv(out_dir / "label_distribution_summary.csv", index=False)

    report = render_report(cfg, labeled_df, reproduction, out_dir, feature_cols, pools, ext_policies, mock_api)
    _write_text(out_dir / "phase2c_labeled_dataset_report.md", report)

    logging.info("Dataset written to %s", out_dir)
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 2C labeled selector dataset from simulator outputs."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print plan; do not write outputs.")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory.")
    parser.add_argument("--mock-api-annotations", action="store_true",
                        help="Add mock annotation fields (clearly marked as mock; not labels).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg_path = _repo_path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)

    # Guard: this script has no --allow-live-api-annotation flag and must NEVER call live APIs.
    api_cfg = cfg.get("api_annotation", {})
    if api_cfg.get("enabled", False):
        print(
            "ERROR: api_annotation.enabled is true in config. "
            "This script does not support live API annotations. "
            "Set enabled: false.",
            file=sys.stderr,
        )
        return 2

    out_root = _repo_path(
        args.output_dir or cfg.get("output", {}).get("root", DEFAULT_OUTPUT_ROOT)
    )
    out_dir = out_root / _timestamp()

    try:
        result = run_pipeline(
            cfg, out_dir,
            dry_run=args.dry_run,
            mock_api=args.mock_api_annotations,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if args.dry_run:
        print("Dry-run complete (no files written).")
        print(f"  Policies detected:    {len(result['policies'])}")
        print(f"  Feature columns:      {len(result['feature_cols'])}")
        print(f"  Pools:")
        for name, n in result["pools"].items():
            print(f"    {name}: {n} policies")
        print(f"  Rows: train={result['n_train']} val={result['n_val']} eval={result['n_eval']}")
        print(f"  Phase 2C.2 reference: {result['reproduction']}")
        print(f"  No live API call was made.")
    else:
        print(f"Dataset written to: {out_dir}")
        print(f"  Rows: train={result['n_train']} val={result['n_val']} eval={result['n_eval']} total={result['n_total']}")
        print(f"  No live API call was made.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
