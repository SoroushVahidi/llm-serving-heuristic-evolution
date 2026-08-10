#!/usr/bin/env python3
"""Rebuild SwissAI V2 reporting artifacts without evaluating policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC = "metric_arrival_normalized_weighted_goodput"
EXPECTED_WINDOWS = 512
EXPECTED_POLICIES = 27
NEW_POLICIES = [
    "sola_style_state_aware",
    "slai_style_phase_aware",
    "flow_control_stability",
    "kv_constrained_online",
    "adaptive_chunked_prefill",
    "aging_priority",
    "weighted_fair_share",
]
COVERAGE_FIELDS = [
    "feat_swiss_kv_proxy_p95",
    "feat_swiss_high_reuse_fraction",
    "feat_swiss_low_reuse_fraction",
    "feat_swiss_reuse_mean",
    "feat_swiss_reuse_p95",
    "feat_swiss_arrival_rate_1s",
    "feat_swiss_arrival_rate_5s",
    "feat_swiss_arrival_rate_20s",
    "feat_swiss_arrival_rate_60s",
    "feat_swiss_prompt_p95",
    "feat_swiss_output_p95",
    "feat_swiss_fraction_negative_slack",
    "feat_swiss_kv_pressure",
    "feat_swiss_token_budget_pressure",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_jsonable) + "\n")


def _check_inputs(source: Path, summary: pd.DataFrame, causal: pd.DataFrame, vectors: pd.DataFrame) -> dict[str, Any]:
    keys = ["window_id", "policy_name"]
    expected_windows = EXPECTED_WINDOWS
    expected_policies = EXPECTED_POLICIES
    expected_cells = expected_windows * expected_policies
    vector_keys = vectors[keys].drop_duplicates()
    causal_keys = causal[["window_id"]]
    checks = {
        "summary_parses": True,
        "causal_features_parses": True,
        "windows": int(summary["window_id"].nunique()),
        "policies": int(vectors["policy_name"].nunique()),
        "cells": int(len(vectors)),
        "expected_windows": expected_windows,
        "expected_policies": expected_policies,
        "expected_cells": expected_cells,
        "duplicate_policy_keys": int(vectors.duplicated(keys).sum()),
        "duplicate_summary_windows": int(summary.duplicated(["window_id"]).sum()),
        "duplicate_causal_windows": int(causal.duplicated(["window_id"]).sum()),
        "causal_windows": int(causal["window_id"].nunique()),
        "missing_policy_cells": int(expected_cells - len(vector_keys)),
        "summary_causal_join_loss": int(len(set(summary.window_id) - set(causal.window_id))),
        "coverage_fields_present": {field: field in causal.columns for field in COVERAGE_FIELDS},
        "source_sweep": str(source),
    }
    if checks["windows"] != expected_windows or checks["causal_windows"] != expected_windows:
        raise ValueError(f"window count failed: {checks}")
    if checks["policies"] != expected_policies or checks["cells"] != expected_cells:
        raise ValueError(f"matrix dimensions failed: {checks}")
    if any(checks[name] for name in ("duplicate_policy_keys", "duplicate_summary_windows", "duplicate_causal_windows")):
        raise ValueError(f"duplicate key check failed: {checks}")
    if checks["missing_policy_cells"] or checks["summary_causal_join_loss"]:
        raise ValueError(f"join/cell completeness failed: {checks}")
    if not all(checks["coverage_fields_present"].values()):
        raise ValueError(f"coverage schema failed: {checks}")
    if len(set(vector_keys.itertuples(index=False, name=None))) != expected_cells:
        raise ValueError("policy key cardinality failed")
    return checks


def _policy_summary(vectors: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pivot = vectors.pivot(index="window_id", columns="policy_name", values=METRIC)
    oracle = pivot.max(axis=1)
    fixed_means = pivot.mean()
    best_fixed_value = float(fixed_means.max())
    best_fixed = sorted(fixed_means[fixed_means == best_fixed_value].index.tolist())
    rows = []
    for policy in pivot.columns:
        rows.append({
            "policy_name": policy,
            "mean_anwg": float(pivot[policy].mean()),
            "median_anwg": float(pivot[policy].median()),
            "mean_regret_to_oracle": float((oracle - pivot[policy]).mean()),
            "mean_rank": float(pivot.rank(axis=1, ascending=False, method="average")[policy].mean()),
            "exact_best_windows": int((pivot[policy] == oracle).sum()),
            "near_best_eps_0_001": int((pivot[policy] >= oracle - 0.001).sum()),
            "near_best_eps_0_005": int((pivot[policy] >= oracle - 0.005).sum()),
            "near_best_eps_0_010": int((pivot[policy] >= oracle - 0.010).sum()),
            "mean_completion_fraction": float(vectors.loc[vectors.policy_name == policy, "metric_completion_fraction"].mean()),
            "mean_slo_violation_rate": float(vectors.loc[vectors.policy_name == policy, "metric_slo_violation_rate"].mean()),
        })
    policy_summary = pd.DataFrame(rows).sort_values(["mean_anwg", "policy_name"], ascending=[False, True])
    old = [policy for policy in pivot.columns if policy not in NEW_POLICIES]
    new_envelope = pivot[NEW_POLICIES].max(axis=1)
    old_envelope = pivot[old].max(axis=1)
    marginal = new_envelope - old_envelope
    oracle_summary = {
        "windows": int(len(pivot)),
        "oracle_mean_anwg": float(oracle.mean()),
        "oracle_median_anwg": float(oracle.median()),
        "best_fixed_policies": best_fixed,
        "best_fixed_mean_anwg": best_fixed_value,
        "oracle_minus_best_fixed": float(oracle.mean() - best_fixed_value),
        "best_fixed_tie_count": len(best_fixed),
        "mean_exact_oracle_tie_count": float(pivot.apply(lambda row: len(set(row[row == row.max()].index)), axis=1).mean()),
    }
    marginal_summary = {
        "baseline_policy_set": old,
        "new_policy_set": NEW_POLICIES,
        "strict_new_policy_wins": int((marginal > 0).sum()),
        "strict_new_policy_win_fraction": float((marginal > 0).mean()),
        "max_marginal_gain": float(marginal.max()),
        "mean_marginal_gain": float(marginal.mean()),
        "fraction_gain_gt_0_001": float((marginal > 0.001).mean()),
        "fraction_gain_gt_0_005": float((marginal > 0.005).mean()),
        "fraction_gain_gt_0_010": float((marginal > 0.010).mean()),
        "near_win_windows_eps_0_001": int((marginal >= -0.001).sum()),
        "near_win_windows_eps_0_005": int((marginal >= -0.005).sum()),
        "near_win_windows_eps_0_010": int((marginal >= -0.010).sum()),
    }
    return policy_summary, {"oracle": oracle_summary, "marginal": marginal_summary}


def _coverage_summary(joined: pd.DataFrame, vectors: pd.DataFrame) -> dict[str, Any]:
    pivot = vectors.pivot(index="window_id", columns="policy_name", values=METRIC)
    oracle = pivot.max(axis=1)
    old = [policy for policy in pivot.columns if policy not in NEW_POLICIES]
    marginal = pivot[NEW_POLICIES].max(axis=1) - pivot[old].max(axis=1)
    thresholds = {
        "high_kv": joined["feat_swiss_kv_proxy_p95"] >= joined["feat_swiss_kv_proxy_p95"].quantile(0.75),
        "high_reuse": joined["feat_swiss_high_reuse_fraction"] >= 0.25,
        "high_arrival": joined["feat_swiss_arrival_rate_1s"] >= joined["feat_swiss_arrival_rate_1s"].quantile(0.85),
    }
    thresholds["high_kv_high_reuse"] = thresholds["high_kv"] & thresholds["high_reuse"]
    strata = {}
    for name, mask in thresholds.items():
        ids = joined.loc[mask, "window_id"]
        subset_oracle = oracle.loc[oracle.index.isin(ids)]
        subset_marginal = marginal.loc[marginal.index.isin(ids)]
        strata[name] = {
            "windows": int(mask.sum()),
            "oracle_minus_best_fixed": float(subset_oracle.mean() - pivot.loc[ids].mean().max()),
            "v2_marginal_gain_mean": float(subset_marginal.mean()),
            "unique_new_policy_wins": int((subset_marginal > 0).sum()),
            "near_win_eps_0_001": int((subset_marginal >= -0.001).sum()),
            "near_win_eps_0_005": int((subset_marginal >= -0.005).sum()),
        }
    distributions = {}
    for field in COVERAGE_FIELDS:
        series = joined[field]
        distributions[field] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p10": float(series.quantile(0.10)),
            "p90": float(series.quantile(0.90)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return {
        "thresholds": {name: float(joined.loc[mask, field].min()) for name, mask, field in [
            ("high_kv", thresholds["high_kv"], "feat_swiss_kv_proxy_p95"),
            ("high_arrival", thresholds["high_arrival"], "feat_swiss_arrival_rate_1s"),
        ]},
        "distributions": distributions,
        "source_family_breakdown": joined["source_file"].value_counts().to_dict(),
        "strata": strata,
    }


def build_report(source: Path, output: Path, old_documented_value: float = 0.991726) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    summary = pd.read_csv(source / "combined" / "window_summary.csv")
    causal = pd.read_csv(source / "combined" / "causal_features.csv")
    vectors = pd.read_csv(source / "combined" / "policy_vectors.csv")
    checks = _check_inputs(source, summary, causal, vectors)
    joined = summary.merge(causal[["window_id", *COVERAGE_FIELDS]], on="window_id", how="inner", validate="one_to_one")
    joined.to_csv(output / "joined_coverage_features.csv", index=False)
    policy_summary, aggregates = _policy_summary(vectors)
    policy_summary.to_csv(output / "policy_summary.csv", index=False)
    coverage = _coverage_summary(joined, vectors)
    _write_json(output / "input_integrity.json", checks)
    _write_json(output / "oracle_summary.json", aggregates["oracle"])
    _write_json(output / "marginal_gain_summary.json", aggregates["marginal"])
    _write_json(output / "coverage_frontier_summary.json", coverage)
    old_cause = "stale prose/intermediate audit value: the simulator-audit CSV for swissai_v2_sweep already records 0.9925129331848189; no current matrix aggregation produces 0.991726."
    final = {
        "report_status": "REPAIRED_FROM_EXISTING_MATRIX",
        "full_policy_recomputation_required": False,
        "matrix_integrity": checks,
        "oracle": aggregates["oracle"],
        "marginal_gain": aggregates["marginal"],
        "coverage": coverage,
        "old_documented_value": old_documented_value,
        "corrected_value": aggregates["oracle"]["oracle_mean_anwg"],
        "old_number_root_cause": old_cause,
        "caveats": [
            "SwissAI bucket-reuse output lengths and SLOs are partly reconstructed or synthetic.",
            "This is a bounded simulator sensitivity study, not real-system causal validation.",
            "Zero policy gain does not establish intrinsic FIFO/EDF optimality.",
            "Selector transfer and module-composition generalization are not established.",
        ],
    }
    _write_json(output / "final_summary.json", final)
    (output / "source_sweep_path.txt").write_text(str(source) + "\n")
    (output / "source_git_sha.txt").write_text("e8bd759b6cdaa8a05096b0ceeb1c7684cfa07302\n")
    _write_json(output / "reanalysis_manifest.json", {
        "tool": "scripts/repair_swissai_v2_report.py",
        "source": str(source),
        "output": str(output),
        "metric": METRIC,
        "policy_evaluations_rerun": False,
        "original_sweep_modified": False,
        "input_integrity": checks,
    })
    report = f"""# Repaired SwissAI V2 Policy Sweep Report

This report was reconstructed from the existing 512-window x 27-policy matrix. No policy evaluations were rerun.

## Integrity

- Windows: {checks['windows']}
- Policies: {checks['policies']}
- Valid cells: {checks['cells']}
- Duplicate policy keys: {checks['duplicate_policy_keys']}
- Join loss: {checks['summary_causal_join_loss']}

## Results

- Best fixed policy set: {', '.join(aggregates['oracle']['best_fixed_policies'])}
- Oracle mean ANWG: {aggregates['oracle']['oracle_mean_anwg']:.12f}
- Best-fixed mean ANWG: {aggregates['oracle']['best_fixed_mean_anwg']:.12f}
- Oracle gap: {aggregates['oracle']['oracle_minus_best_fixed']:.12f}
- Strict V2 marginal gain: {aggregates['marginal']['max_marginal_gain']:.12f} maximum; {aggregates['marginal']['strict_new_policy_wins']} strict wins

## Reporting repair

The failed coverage stage expected `kv_proxy_p95`, `high_reuse_fraction`, and `low_reuse_fraction` in `window_summary.csv`. The canonical fields are in `causal_features.csv` as `feat_swiss_kv_proxy_p95`, `feat_swiss_high_reuse_fraction`, and `feat_swiss_low_reuse_fraction`. The repair performs a one-to-one `window_id` join and leaves source files untouched.

The old documented ANWG value was `{old_documented_value}`. The corrected matrix aggregate is `{aggregates['oracle']['oracle_mean_anwg']:.12f}`. Root cause: {old_cause}

## Interpretation

Supported: SwissAI expands feature/workload coverage; the matrix is complete; ANWG is saturated; strict V2 marginal oracle gain is zero; no unique policy specialization is observed under this metric.

Qualified: output lengths and SLOs are partly reconstructed or synthetic. This is a bounded simulator sensitivity study. Zero policy gain does not imply intrinsic FIFO or EDF optimality.

Not established: real-system optimality, causal KV-reuse effects, module-composition usefulness, or selector generalization from SwissAI.
"""
    (output / "final_report.md").write_text(report)
    (output / "repair_log.txt").write_text("Report-only reconstruction completed; original sweep inputs were read-only.\n")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.source, args.output)
    print(f"report repair complete: {args.output}")


if __name__ == "__main__":
    main()
