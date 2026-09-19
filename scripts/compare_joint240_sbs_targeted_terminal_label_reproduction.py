#!/usr/bin/env python3
"""Compare targeted terminal-label outputs under the frozen numeric contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ABS_TOL = 1e-12
REL_TOL = 1e-12


def _is_numeric(a: pd.Series, b: pd.Series) -> bool:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    return bool(
        aa.notna().all()
        and bb.notna().all()
        and a.dropna().map(type).ne(bool).all()
        and b.dropna().map(type).ne(bool).all()
    )


def _numeric_diff(a: pd.Series, b: pd.Series) -> Dict[str, object]:
    aa = pd.to_numeric(a, errors="coerce").astype(float)
    bb = pd.to_numeric(b, errors="coerce").astype(float)
    abs_diff = (aa - bb).abs()
    denom = pd.concat([aa.abs(), bb.abs(), pd.Series(np.full(len(aa), 1e-300))], axis=1).max(axis=1)
    rel_diff = abs_diff / denom
    ok = (abs_diff <= ABS_TOL) | (rel_diff <= REL_TOL)
    return {
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "ok": ok,
        "max_abs": float(abs_diff.max()) if len(abs_diff) else 0.0,
        "p50_abs": float(abs_diff.quantile(0.50)) if len(abs_diff) else 0.0,
        "p90_abs": float(abs_diff.quantile(0.90)) if len(abs_diff) else 0.0,
        "p99_abs": float(abs_diff.quantile(0.99)) if len(abs_diff) else 0.0,
        "max_rel": float(rel_diff.max()) if len(rel_diff) else 0.0,
        "n_gt_1e_15": int((abs_diff > 1e-15).sum()),
        "n_gt_1e_14": int((abs_diff > 1e-14).sum()),
        "n_gt_1e_13": int((abs_diff > 1e-13).sum()),
        "n_gt_1e_12": int((abs_diff > 1e-12).sum()),
        "n_tolerance_violations": int((~ok).sum()),
    }


def compare_rows(candidate: pd.DataFrame, reference: pd.DataFrame) -> Dict[str, object]:
    keys = ["state_id", "canonical_action_id"]
    merged = candidate.merge(reference, on=keys, how="outer", suffixes=("_candidate", "_reference"), indicator=True)
    exact_mismatches: List[Dict[str, object]] = []
    numeric_summaries: List[Dict[str, object]] = []
    numeric_abs_values: List[float] = []
    numeric_rel_values: List[float] = []
    sign_mismatches: List[Dict[str, object]] = []

    if not (merged["_merge"] == "both").all():
        exact_mismatches.append({"field": "row_keys", "counts": merged["_merge"].value_counts().to_dict()})
        return {
            "exact_mismatches": exact_mismatches,
            "numeric": {},
            "numeric_columns": [],
            "sign_mismatches": sign_mismatches,
        }

    exact_fields = [
        "scenario_id",
        "fold",
        "step",
        "canonical_action_full",
        "canonical_action_admit",
        "sbs_canonical_action_id",
        "sbs_canonical_action_full",
        "is_sbs_action",
        "policies_generating_action",
        "n_policies_generating_action",
        "n_distinct_canonical_actions",
        "policy_signature",
        "disagreement_episode_id",
        "disagreement_episode_type",
        "disagreement_episode_length",
        "trajectory_position_bin",
    ]
    for c in exact_fields:
        if c in candidate.columns and c in reference.columns:
            neq = merged[f"{c}_candidate"].astype(str) != merged[f"{c}_reference"].astype(str)
            if neq.any():
                exact_mismatches.append({"field": c, "n_mismatch": int(neq.sum())})

    compare_cols = [
        c
        for c in candidate.columns
        if c not in keys
        and c in reference.columns
        and (c.startswith("state__") or c.startswith("action__") or c.startswith("q_") or c.startswith("q0_") or c.startswith("a_sbs_"))
    ]
    for c in compare_cols:
        a = merged[f"{c}_candidate"]
        b = merged[f"{c}_reference"]
        if _is_numeric(a, b):
            d = _numeric_diff(a, b)
            numeric_abs_values.extend(float(x) for x in d["abs_diff"].tolist())
            numeric_rel_values.extend(float(x) for x in d["rel_diff"].tolist())
            if d["max_abs"] > 0 or d["n_tolerance_violations"]:
                numeric_summaries.append(
                    {
                        "field": c,
                        "max_abs": d["max_abs"],
                        "max_rel": d["max_rel"],
                        "n_gt_1e_15": d["n_gt_1e_15"],
                        "n_gt_1e_14": d["n_gt_1e_14"],
                        "n_gt_1e_13": d["n_gt_1e_13"],
                        "n_gt_1e_12": d["n_gt_1e_12"],
                        "n_tolerance_violations": d["n_tolerance_violations"],
                    }
                )
        else:
            neq = a.astype(str) != b.astype(str)
            if neq.any():
                exact_mismatches.append({"field": c, "n_mismatch": int(neq.sum())})

    for c in [x for x in ("a_sbs_anwg", "a_sbs_soft", "a_sbs_wcg", "a_sbs_wmt_improvement", "a_sbs_wnt_improvement") if x in candidate.columns and x in reference.columns]:
        a = pd.to_numeric(merged[f"{c}_candidate"], errors="coerce")
        b = pd.to_numeric(merged[f"{c}_reference"], errors="coerce")
        material = (a.abs() > ABS_TOL) | (b.abs() > ABS_TOL)
        neq = np.sign(a) != np.sign(b)
        if (material & neq).any():
            sign_mismatches.append({"field": c, "n_mismatch": int((material & neq).sum())})

    abs_arr = np.asarray(numeric_abs_values, dtype=float)
    rel_arr = np.asarray(numeric_rel_values, dtype=float)
    numeric = {
        "n_numeric_comparisons": int(len(abs_arr)),
        "max_abs_diff": float(abs_arr.max()) if len(abs_arr) else 0.0,
        "p50_abs_diff": float(np.quantile(abs_arr, 0.50)) if len(abs_arr) else 0.0,
        "p90_abs_diff": float(np.quantile(abs_arr, 0.90)) if len(abs_arr) else 0.0,
        "p99_abs_diff": float(np.quantile(abs_arr, 0.99)) if len(abs_arr) else 0.0,
        "max_rel_diff": float(rel_arr.max()) if len(rel_arr) else 0.0,
        "count_abs_gt_1e_15": int((abs_arr > 1e-15).sum()) if len(abs_arr) else 0,
        "count_abs_gt_1e_14": int((abs_arr > 1e-14).sum()) if len(abs_arr) else 0,
        "count_abs_gt_1e_13": int((abs_arr > 1e-13).sum()) if len(abs_arr) else 0,
        "count_abs_gt_1e_12": int((abs_arr > 1e-12).sum()) if len(abs_arr) else 0,
        "tolerance_violations": int(sum(x["n_tolerance_violations"] for x in numeric_summaries)),
    }
    return {
        "exact_mismatches": exact_mismatches,
        "numeric": numeric,
        "numeric_columns": sorted(numeric_summaries, key=lambda x: float(x["max_abs"]), reverse=True),
        "sign_mismatches": sign_mismatches,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-rows", required=True)
    ap.add_argument("--candidate-map", required=True)
    ap.add_argument("--reference-rows", required=True)
    ap.add_argument("--reference-map", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cand = pd.read_csv(args.candidate_rows)
    ref = pd.read_csv(args.reference_rows)
    rows = compare_rows(cand, ref)

    cand_map = pd.read_csv(args.candidate_map)
    ref_map = pd.read_csv(args.reference_map)
    mm = cand_map.merge(ref_map, on=["state_id", "policy_id"], how="outer", suffixes=("_candidate", "_reference"), indicator=True)
    map_mismatches: List[Dict[str, object]] = []
    if not (mm["_merge"] == "both").all():
        map_mismatches.append({"field": "policy_map_keys", "counts": mm["_merge"].value_counts().to_dict()})
    else:
        for c in ["scenario_id", "fold", "step", "policy_index", "canonical_action_id", "canonical_action_full", "canonical_action_admit", "sbs_canonical_action_id", "action_equal_to_sbs", "is_sbs_policy"]:
            if c in cand_map.columns and c in ref_map.columns:
                neq = mm[f"{c}_candidate"].astype(str) != mm[f"{c}_reference"].astype(str)
                if neq.any():
                    map_mismatches.append({"field": c, "n_mismatch": int(neq.sum())})

    summary = {
        "abs_tolerance": ABS_TOL,
        "rel_tolerance": REL_TOL,
        "states_checked": int(cand["state_id"].nunique()),
        "state_action_rows_checked": int(len(cand)),
        "policy_map_rows_checked": int(len(cand_map)),
        "exact_invariant_mismatches": rows["exact_mismatches"],
        "policy_map_mismatches": map_mismatches,
        "numeric": rows["numeric"],
        "numeric_columns_with_differences": rows["numeric_columns"],
        "sign_mismatches": rows["sign_mismatches"],
    }
    summary["verdict"] = (
        "CROSS_PLATFORM_REPRODUCTION_PASS"
        if not summary["exact_invariant_mismatches"]
        and not summary["policy_map_mismatches"]
        and int(summary["numeric"].get("tolerance_violations", 0)) == 0
        and not summary["sign_mismatches"]
        else "CROSS_PLATFORM_REPRODUCTION_FAIL"
    )
    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["verdict"] != "CROSS_PLATFORM_REPRODUCTION_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
