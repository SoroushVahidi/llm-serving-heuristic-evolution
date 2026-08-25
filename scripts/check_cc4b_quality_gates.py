#!/usr/bin/env python3
"""Quality gates CC4b must pass before CC5 may be retrained against it.

Exits non-zero with an exact diagnosis if any gate fails, per instruction
("if these gates fail, stop with an exact diagnosis rather than forcing
retraining").
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from llmserveopt.experiments.cc5_contextual_predictor import CC5Error, load_cc4_dataset, validate_cc4_dataset

MIN_HELD_OUT = 50
MIN_NON_NEAR_TIE_HELD_OUT = 20
MAX_SINGLE_FAMILY_SHARE = 0.70


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_cc4b_quality_gates.py <dataset_dir>", file=sys.stderr)
        return 2
    dataset_dir = Path(sys.argv[1])

    findings: list[str] = []
    failures: list[str] = []

    try:
        ds = load_cc4_dataset(dataset_dir)
    except CC5Error as exc:
        print(f"FAIL: could not load dataset: {exc}")
        return 1

    try:
        audit_findings = validate_cc4_dataset(ds)
        findings.extend(audit_findings)
    except CC5Error as exc:
        print(f"FAIL: dataset validation (split integrity / leakage / manifest completeness): {exc}")
        return 1

    total_windows = len(ds.causal_features)
    split_counts = ds.causal_features["split"].value_counts().to_dict()
    findings.append(f"total_windows={total_windows}, by_split={split_counts}")

    eval_windows = ds.causal_features[ds.causal_features["split"].isin(ds.evaluation_splits)]["window_id"]
    n_held_out = len(eval_windows)
    findings.append(f"held_out_windows={n_held_out} (gate: >= {MIN_HELD_OUT})")
    if n_held_out < MIN_HELD_OUT:
        failures.append(f"held_out_windows={n_held_out} < required {MIN_HELD_OUT}")

    # Fixed-policy spread: per held-out window, best-fixed minus worst-fixed
    # mean ANWG (mirrors CC1's own fixed_policy_spread discriminativeness
    # signal) -- reported, not gated, since CC4b's search space is fixed
    # by design (identical candidate_search to CC4).
    fixed_rows = ds.per_window_results[
        (ds.per_window_results["family"] == "fixed_policy") & (ds.per_window_results["window_id"].isin(eval_windows))
    ]
    if not fixed_rows.empty:
        spreads = fixed_rows.groupby("window_id")["metric_arrival_normalized_weighted_goodput"].agg(lambda s: s.max() - s.min())
        findings.append(
            f"fixed_policy_spread(held-out): mean={spreads.mean():.4f}, min={spreads.min():.4f}, max={spreads.max():.4f}, n_windows={len(spreads)}"
        )

    near_tie = ds.near_tie_flags[ds.near_tie_flags["threshold"] == 0.005]
    near_tie_eval = near_tie[near_tie["window_id"].isin(eval_windows)]
    non_near_tie_count = int((~near_tie_eval["near_tie"]).sum())
    findings.append(f"non_near_tie_held_out_windows={non_near_tie_count} (gate: >= {MIN_NON_NEAR_TIE_HELD_OUT})")
    if non_near_tie_count < MIN_NON_NEAR_TIE_HELD_OUT:
        failures.append(f"non_near_tie_held_out_windows={non_near_tie_count} < required {MIN_NON_NEAR_TIE_HELD_OUT}")

    oracle_eval = ds.oracle_labels[ds.oracle_labels["window_id"].isin(eval_windows)]
    family_counts = oracle_eval["oracle_family"].value_counts()
    if len(oracle_eval) > 0:
        top_share = family_counts.iloc[0] / len(oracle_eval)
        findings.append(f"oracle_family_distribution(held-out)={family_counts.to_dict()}, top_share={top_share:.2f} (gate: <= {MAX_SINGLE_FAMILY_SHARE})")
        if top_share > MAX_SINGLE_FAMILY_SHARE:
            failures.append(f"single oracle family '{family_counts.index[0]}' dominates held-out set at {top_share:.2f} > {MAX_SINGLE_FAMILY_SHARE}")

    completion = ds.completion_constraints
    inconsistent = completion[(completion["oracle_completion_fraction"] < 0) | (completion["oracle_completion_fraction"] > 1)]
    if not inconsistent.empty:
        failures.append(f"completion accounting inconsistent for {len(inconsistent)} window(s): {list(inconsistent['window_id'])}")
    else:
        findings.append("completion accounting consistent (all completion fractions in [0, 1])")

    replay_path = dataset_dir / "replay_commands.sh"
    if not replay_path.exists():
        failures.append("replay_commands.sh missing")
    else:
        findings.append("replay_commands.sh present")

    print("=== CC4b quality gate findings ===")
    for f in findings:
        print(f"  - {f}")

    if failures:
        print("\n=== FAILURES (exact diagnosis) ===")
        for f in failures:
            print(f"  - {f}")
        print("\nVerdict: GATES NOT PASSED. Do not retrain CC5 against this dataset.")
        return 1

    print("\nVerdict: ALL GATES PASSED. CC5 retraining may proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
