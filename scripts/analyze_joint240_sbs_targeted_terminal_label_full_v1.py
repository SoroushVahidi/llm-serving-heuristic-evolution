#!/usr/bin/env python3
"""Post-completion analysis for JOINT240_SBS_TARGETED_TERMINAL_LABEL_FULL_V1.

This script aggregates completed shard CSVs and computes integrity gates,
target-support summaries, descriptive feature summaries, and a learning
readiness verdict.  It never runs simulator branches and never trains a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


EXPECTED_STATES = 8888
EXPECTED_ACTION_ROWS = 30746
EXPECTED_POLICY_MAP_ROWS = 53328
EXPECTED_SCENARIOS = 236
EXPECTED_FOLDS = 5
EXPECTED_SHARDS = 80
P6 = [
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
]
SBS_POLICY = "kv_constrained_online"
PRIMARY_ADV = "a_sbs_anwg"
TOL = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return None


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(x) for x in values if pd.notna(x)], dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def count_thresholds(values: pd.Series, thresholds: Sequence[float], *, positive: bool = False) -> Dict[str, int]:
    out: Dict[str, int] = {}
    numeric = pd.to_numeric(values, errors="coerce")
    for t in thresholds:
        key = f">={t:g}" if positive else f"abs>={t:g}"
        out[key] = int((numeric >= t).sum() if positive else (numeric.abs() >= t).sum())
    return out


def gini(values: Sequence[int]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2.0 * np.sum((np.arange(1, n + 1) * arr))) / (n * np.sum(arr)) - (n + 1) / n)


def classify_feature_group(col: str) -> str:
    name = col
    if name.startswith("state__"):
        name = name[len("state__") :]
    if name.startswith("action__"):
        name = name[len("action__") :]
    if any(k in name for k in ["waiting_count", "active_count", "total_in_system", "admissible_count", "queued_token", "waiting_prompt", "waiting_predicted", "active_predicted", "new_request_count", "completed_count"]):
        return "queue_load"
    if any(k in name for k in ["kv_", "kv_tokens", "kv_util", "projected_kv", "large_req"]):
        return "kv_pressure"
    if any(k in name for k in ["prefill", "decode", "hold_decode"]):
        return "prefill_decode"
    if any(k in name for k in ["prompt_tokens", "predicted_output", "predicted_total", "remaining_tokens"]):
        return "service_request_size"
    if any(k in name for k in ["slack", "urgent", "slo"]):
        return "slo_slack"
    if any(k in name for k in ["priority", "class_", "fair"]):
        return "priority_fairness"
    if "hist_" in name:
        return "recent_dynamics"
    if col.startswith("action__"):
        return "action_difference"
    return "other"


def read_shards(run_dir: Path, num_shards: int) -> tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    missing: Dict[str, List[int]] = {"done": [], "rows": [], "maps": [], "summary": []}
    summaries: List[Dict[str, Any]] = []
    row_parts: List[pd.DataFrame] = []
    map_parts: List[pd.DataFrame] = []
    for i in range(num_shards):
        done = run_dir / f"DONE.shard{i:04d}-of-{num_shards:04d}"
        rows_path = run_dir / f"state_action_rows.shard{i:04d}-of-{num_shards:04d}.csv"
        map_path = run_dir / f"state_policy_action_map.shard{i:04d}-of-{num_shards:04d}.csv"
        summary_path = run_dir / f"summary.shard{i:04d}-of-{num_shards:04d}.json"
        if not done.exists():
            missing["done"].append(i)
        if not rows_path.exists():
            missing["rows"].append(i)
        if not map_path.exists():
            missing["maps"].append(i)
        if not summary_path.exists():
            missing["summary"].append(i)
        if rows_path.exists() and rows_path.stat().st_size:
            row_parts.append(pd.read_csv(rows_path))
        if map_path.exists() and map_path.stat().st_size:
            map_parts.append(pd.read_csv(map_path))
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text()))
    if any(missing.values()):
        raise FileNotFoundError({"missing_shard_artifacts": missing})
    rows = pd.concat(row_parts, ignore_index=True)
    maps = pd.concat(map_parts, ignore_index=True)
    return rows, maps, summaries


def aggregate(run_dir: Path, out_dir: Path, num_shards: int, repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows, maps, summaries = read_shards(run_dir, num_shards)
    rows = rows.sort_values(["scenario_id", "step", "canonical_action_id"]).reset_index(drop=True)
    maps = maps.sort_values(["scenario_id", "step", "policy_index"]).reset_index(drop=True)
    rows_path = out_dir / "state_action_rows_full.csv"
    maps_path = out_dir / "state_policy_action_map_full.csv"
    rows.to_csv(rows_path, index=False)
    maps.to_csv(maps_path, index=False)
    shard_checksums = {
        f"state_action_rows.shard{i:04d}-of-{num_shards:04d}.csv": sha256_file(
            run_dir / f"state_action_rows.shard{i:04d}-of-{num_shards:04d}.csv"
        )
        for i in range(num_shards)
    }
    shard_checksums.update(
        {
            f"state_policy_action_map.shard{i:04d}-of-{num_shards:04d}.csv": sha256_file(
                run_dir / f"state_policy_action_map.shard{i:04d}-of-{num_shards:04d}.csv"
            )
            for i in range(num_shards)
        }
    )
    if (run_dir / "full_manifest_all_disagreements.csv").exists():
        shard_checksums["full_manifest_all_disagreements.csv"] = sha256_file(
            run_dir / "full_manifest_all_disagreements.csv"
        )
    campaign_commits: List[str] = []
    for prov in sorted((run_dir / "manifests").glob("provenance.*.txt")):
        lines = [line.strip() for line in prov.read_text().splitlines()]
        for line in lines:
            if len(line) == 40 and all(ch in "0123456789abcdef" for ch in line):
                campaign_commits.append(line)
                break
    provenance = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "analysis_code_commit": git(["rev-parse", "HEAD"], repo_root),
        "analysis_code_branch": git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "campaign_source_commits_from_shard_provenance": sorted(set(campaign_commits)),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "num_shards": int(num_shards),
        "aggregate_command": " ".join(["scripts/analyze_joint240_sbs_targeted_terminal_label_full_v1.py", "--run-dir", str(run_dir), "--output-dir", str(out_dir), "analyze"]),
        "input_shard_sha256": shard_checksums,
        "aggregate_sha256": {
            "state_action_rows_full.csv": sha256_file(rows_path),
            "state_policy_action_map_full.csv": sha256_file(maps_path),
        },
        "shard_summary_totals": {
            "n_input_states": int(sum(int(s.get("n_input_states", 0)) for s in summaries)),
            "n_hit_states": int(sum(int(s.get("n_hit_states", 0)) for s in summaries)),
            "n_state_action_rows": int(sum(int(s.get("n_state_action_rows", 0)) for s in summaries)),
            "n_state_policy_action_map_rows": int(sum(int(s.get("n_state_policy_action_map_rows", 0)) for s in summaries)),
            "n_failed_scenarios": int(sum(int(s.get("n_failed_scenarios", 0)) for s in summaries)),
            "wall_seconds_sum": float(sum(float(s.get("wall_seconds", 0.0)) for s in summaries)),
            "wall_seconds_max": float(max(float(s.get("wall_seconds", 0.0)) for s in summaries)),
        },
    }
    write_json(out_dir / "aggregation_provenance_summary.json", provenance)
    return rows, maps, provenance


def integrity_gates(rows: pd.DataFrame, maps: pd.DataFrame, out_dir: Path, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    state_cols = [c for c in rows.columns if c.startswith("state__")]
    action_cols = [c for c in rows.columns if c.startswith("action__")]
    numeric_cols = rows.select_dtypes(include=[np.number]).columns.tolist()
    required_numeric = [
        c
        for c in numeric_cols
        if c.startswith("state__")
        or c.startswith("action__")
        or c.startswith("q_sbs")
        or c.startswith("q0_sbs")
        or c.startswith("a_sbs")
    ]
    forbidden_learnable_markers = [
        "future",
        "terminal",
        "actual_output",
        "post_decision",
        "target",
        "scenario_id",
        "state_id",
        "vbs",
        "q_sbs",
        "a_sbs",
        "num_completed",
        "num_dropped",
        "sim_duration",
    ]
    learnable_cols = state_cols + action_cols
    forbidden_learnable = [
        c
        for c in learnable_cols
        if any(marker in c.lower() for marker in forbidden_learnable_markers)
    ]
    map_counts = maps.groupby("state_id")["policy_id"].nunique()
    sbs_rows = rows[rows["is_sbs_action"].astype(int) == 1]
    branches_for_maps = maps.merge(
        rows[["state_id", "canonical_action_id", "q_sbs_anwg", PRIMARY_ADV]],
        on=["state_id", "canonical_action_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    scenario_folds = rows[["scenario_id", "fold"]].drop_duplicates()
    scenarios_multi_fold = scenario_folds.groupby("scenario_id")["fold"].nunique()
    q_cols = [c for c in rows.columns if c.startswith("q_sbs") or c.startswith("q0_sbs")]
    q_ranges = {
        c: {"min": float(pd.to_numeric(rows[c]).min()), "max": float(pd.to_numeric(rows[c]).max())}
        for c in q_cols
    }
    missing_counts = rows[required_numeric].isna().sum()
    inf_counts = {
        c: int(np.isinf(pd.to_numeric(rows[c], errors="coerce")).sum())
        for c in required_numeric
    }
    experiment_root = out_dir.parent.parent
    validation_summary_path = experiment_root / "validation50_v1" / "cross_platform_validation_summary.json"
    validation_summary = json.loads(validation_summary_path.read_text()) if validation_summary_path.exists() else None
    validation_numeric = validation_summary.get("numeric", {}) if validation_summary else {}
    gates = {
        "G1_COMPLETENESS": {
            "pass": bool(
                rows["state_id"].nunique() == EXPECTED_STATES
                and len(rows) == EXPECTED_ACTION_ROWS
                and len(maps) == EXPECTED_POLICY_MAP_ROWS
                and map_counts.eq(len(P6)).all()
            ),
            "states": int(rows["state_id"].nunique()),
            "state_action_rows": int(len(rows)),
            "policy_map_rows": int(len(maps)),
            "map_rows_per_state_min": int(map_counts.min()),
            "map_rows_per_state_max": int(map_counts.max()),
        },
        "G2_UNIQUENESS": {
            "pass": bool(
                rows.duplicated(["state_id", "canonical_action_id"]).sum() == 0
                and maps.duplicated(["state_id", "policy_id"]).sum() == 0
            ),
            "duplicate_state_action_rows": int(rows.duplicated(["state_id", "canonical_action_id"]).sum()),
            "duplicate_state_policy_rows": int(maps.duplicated(["state_id", "policy_id"]).sum()),
        },
        "G3_SBS_BASELINE": {
            "pass": bool(
                len(sbs_rows) == EXPECTED_STATES
                and sbs_rows.groupby("state_id").size().eq(1).all()
                and (pd.to_numeric(sbs_rows[PRIMARY_ADV]).abs() <= 0.0).all()
            ),
            "sbs_rows": int(len(sbs_rows)),
            "states_with_one_sbs_branch": int(sbs_rows.groupby("state_id").size().eq(1).sum()),
            "max_abs_sbs_advantage": float(pd.to_numeric(sbs_rows[PRIMARY_ADV]).abs().max()),
        },
        "G4_ACTION_DEDUPLICATION": {
            "pass": bool(
                branches_for_maps["_merge"].eq("both").all()
                and rows.groupby(["state_id", "canonical_action_id"]).size().eq(1).all()
            ),
            "policy_map_rows_joined_to_one_causal_branch": int(branches_for_maps["_merge"].eq("both").sum()),
            "missing_policy_map_branch_links": int((branches_for_maps["_merge"] != "both").sum()),
            "actions_generated_by_1_policy": int((rows["n_policies_generating_action"].astype(int) == 1).sum()),
            "actions_generated_by_2_policies": int((rows["n_policies_generating_action"].astype(int) == 2).sum()),
            "actions_generated_by_3plus_policies": int((rows["n_policies_generating_action"].astype(int) >= 3).sum()),
        },
        "G5_NUMERIC_VALIDITY": {
            "pass": bool(missing_counts.sum() == 0 and sum(inf_counts.values()) == 0),
            "required_numeric_columns": int(len(required_numeric)),
            "nan_count_total": int(missing_counts.sum()),
            "inf_count_total": int(sum(inf_counts.values())),
            "columns_with_nan": {k: int(v) for k, v in missing_counts[missing_counts > 0].items()},
            "columns_with_inf": {k: int(v) for k, v in inf_counts.items() if v > 0},
            "q_ranges": q_ranges,
        },
        "G6_CROSS_PLATFORM_REPRODUCIBILITY": {
            "pass": bool(
                validation_summary is not None
                and validation_summary.get("verdict") == "CROSS_PLATFORM_REPRODUCTION_PASS"
                and validation_numeric.get("tolerance_violations", 0) == 0
                and len(validation_summary.get("sign_mismatches", [])) == 0
            ),
            "validation_summary_path": str(validation_summary_path),
            "validation_verdict": validation_summary.get("verdict") if validation_summary else None,
            "max_abs_diff": validation_numeric.get("max_abs_diff") if validation_summary else None,
            "tolerance_violations": validation_numeric.get("tolerance_violations") if validation_summary else None,
            "sign_mismatches": validation_summary.get("sign_mismatches") if validation_summary else None,
        },
        "G7_FEATURE_LEAKAGE": {
            "pass": bool(len(state_cols) == 95 and len(action_cols) == 70 and not forbidden_learnable),
            "state_feature_columns": int(len(state_cols)),
            "action_diff_columns": int(len(action_cols)),
            "forbidden_learnable_columns": forbidden_learnable,
            "learnable_column_count": int(len(learnable_cols)),
        },
        "G8_FOLD_SCENARIO_CONSERVATION": {
            "pass": bool(
                rows["fold"].nunique() == EXPECTED_FOLDS
                and rows["scenario_id"].nunique() == EXPECTED_SCENARIOS
                and scenarios_multi_fold.max() == 1
            ),
            "folds": int(rows["fold"].nunique()),
            "scenarios": int(rows["scenario_id"].nunique()),
            "scenarios_assigned_to_multiple_folds": int((scenarios_multi_fold > 1).sum()),
        },
        "G9_PROVENANCE": {
            "pass": bool(
                provenance.get("campaign_source_commits_from_shard_provenance") == [
                    "ff34f6fa0303b6f21e13277553f2d5da789b01d4"
                ]
                and "full_manifest_all_disagreements.csv" in provenance.get("input_shard_sha256", {})
            ),
            "analysis_code_commit": provenance.get("analysis_code_commit"),
            "campaign_source_commits_from_shard_provenance": provenance.get(
                "campaign_source_commits_from_shard_provenance"
            ),
            "aggregate_sha256": provenance.get("aggregate_sha256"),
        },
    }
    all_pass = all(v["pass"] for v in gates.values())
    report = {"all_integrity_gates_pass": bool(all_pass), "gates": gates}
    write_json(out_dir / "full_integrity_report.json", report)
    return report


def target_distribution(rows: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    non_sbs = rows[rows["is_sbs_action"].astype(int) == 0].copy()
    adv = pd.to_numeric(non_sbs[PRIMARY_ADV])
    thresholds = [0.001, 0.005, 0.010, 0.020, 0.050]
    summary = {
        "non_sbs_unique_actions": int(len(non_sbs)),
        "positive": int((adv > 0).sum()),
        "negative": int((adv < 0).sum()),
        "zero": int((adv == 0).sum()),
        "nonzero": int((adv != 0).sum()),
        "positive_prevalence": float((adv > 0).mean()),
        "negative_prevalence": float((adv < 0).mean()),
        "nonzero_prevalence": float((adv != 0).mean()),
        "positive_advantage": stats(adv[adv > 0]),
        "negative_advantage": stats(adv[adv < 0]),
        "absolute_effect": stats(adv.abs()),
        "absolute_threshold_counts": count_thresholds(adv, thresholds),
        "positive_threshold_counts": count_thresholds(adv, thresholds, positive=True),
    }
    write_json(out_dir / "target_distribution_summary.json", summary)
    return non_sbs, summary


def state_level(rows: pd.DataFrame, non_sbs: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    grouped = non_sbs.groupby("state_id")[PRIMARY_ADV]
    best = grouped.max()
    worst = grouped.min()
    meta_cols = [
        "state_id",
        "scenario_id",
        "fold",
        "step",
        "sim_time",
        "n_distinct_canonical_actions",
        "policy_signature",
        "disagreement_episode_id",
        "disagreement_episode_type",
        "disagreement_episode_length",
        "trajectory_position_bin",
    ]
    meta = rows[meta_cols].drop_duplicates("state_id").set_index("state_id")
    opp = meta.join(best.rename("best_advantage")).join(worst.rename("worst_advantage"))
    opp["positive_action_count"] = non_sbs.groupby("state_id")[PRIMARY_ADV].apply(lambda s: int((s > 0).sum()))
    opp["negative_action_count"] = non_sbs.groupby("state_id")[PRIMARY_ADV].apply(lambda s: int((s < 0).sum()))
    opp["zero_action_count"] = non_sbs.groupby("state_id")[PRIMARY_ADV].apply(lambda s: int((s == 0).sum()))
    opp["state_class"] = np.select(
        [
            (opp["best_advantage"] > 0) & (opp["worst_advantage"] < 0),
            opp["best_advantage"] > 0,
            (opp["best_advantage"] <= 0) & (opp["worst_advantage"] < 0),
        ],
        [
            "MIXED_GOOD_AND_BAD",
            "BENEFICIAL_AVAILABLE",
            "CAUSALLY_DIFFERENT_BUT_NO_BENEFIT",
        ],
        default="ALL_DIFFERENT_ACTIONS_CAUSALLY_NULL",
    )
    opp = opp.reset_index().sort_values(["scenario_id", "step"])
    opp.to_csv(out_dir / "state_level_opportunity.csv", index=False)
    beneficial = opp[opp["best_advantage"] > 0]
    thresholds = [0.001, 0.005, 0.010, 0.020, 0.050]
    summary = {
        "states": int(len(opp)),
        "state_class_counts": {str(k): int(v) for k, v in opp["state_class"].value_counts().items()},
        "state_class_fractions": {str(k): float(v) for k, v in opp["state_class"].value_counts(normalize=True).items()},
        "beneficial_available_states": int((opp["best_advantage"] > 0).sum()),
        "causally_different_no_benefit_states": int(((opp["best_advantage"] <= 0) & (opp["worst_advantage"] < 0)).sum()),
        "all_null_states": int(((opp["best_advantage"] == 0) & (opp["worst_advantage"] == 0)).sum()),
        "mixed_good_and_bad_states": int(((opp["best_advantage"] > 0) & (opp["worst_advantage"] < 0)).sum()),
        "best_advantage_beneficial_distribution": stats(beneficial["best_advantage"]),
        "best_advantage_threshold_counts": count_thresholds(beneficial["best_advantage"], thresholds, positive=True),
    }
    write_json(out_dir / "state_level_opportunity_summary.json", summary)
    return opp, summary


def support_summaries(rows: pd.DataFrame, maps: pd.DataFrame, non_sbs: pd.DataFrame, opp: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    action_key = rows[["state_id", "canonical_action_id", PRIMARY_ADV]].copy()
    policy = maps.merge(action_key, on=["state_id", "canonical_action_id"], how="left", validate="many_to_one")
    policy_non_sbs = policy[(policy["policy_id"] != SBS_POLICY) & (policy["action_equal_to_sbs"].astype(int) == 0)].copy()
    policy_rows = []
    for pid, g in policy_non_sbs.groupby("policy_id", sort=True):
        adv = pd.to_numeric(g[PRIMARY_ADV])
        policy_rows.append(
            {
                "policy_id": pid,
                "states_action_differs": int(g["state_id"].nunique()),
                "states_positive": int(g.loc[adv > 0, "state_id"].nunique()),
                "states_negative": int(g.loc[adv < 0, "state_id"].nunique()),
                "states_zero": int(g.loc[adv == 0, "state_id"].nunique()),
                "mean_advantage_conditional_on_differing": float(adv.mean()) if len(adv) else None,
                "median_advantage_conditional_on_differing": float(adv.median()) if len(adv) else None,
                "beneficial_scenarios": int(g.loc[adv > 0, "scenario_id"].nunique()),
                "beneficial_folds": int(g.loc[adv > 0, "fold"].nunique()),
            }
        )
    policy_df = pd.DataFrame(policy_rows)
    policy_df.to_csv(out_dir / "policy_support_summary.csv", index=False)

    fold_rows = []
    non_sbs_state = non_sbs[["state_id", "scenario_id", "fold", PRIMARY_ADV]]
    for fold, g in opp.groupby("fold", sort=True):
        action_g = non_sbs_state[non_sbs_state["fold"] == fold]
        adv = pd.to_numeric(action_g[PRIMARY_ADV])
        best = pd.to_numeric(g["best_advantage"])
        fold_rows.append(
            {
                "fold": int(fold),
                "disagreement_states": int(g["state_id"].nunique()),
                "beneficial_states": int((g["best_advantage"] > 0).sum()),
                "harmful_only_states": int(((g["best_advantage"] <= 0) & (g["worst_advantage"] < 0)).sum()),
                "null_states": int(((g["best_advantage"] == 0) & (g["worst_advantage"] == 0)).sum()),
                "positive_action_rows": int((adv > 0).sum()),
                "negative_action_rows": int((adv < 0).sum()),
                "positive_prevalence": float((adv > 0).mean()) if len(adv) else None,
                "best_advantage_mean": float(best.mean()),
                "best_advantage_median": float(best.median()),
                "best_advantage_p90": float(best.quantile(0.9)),
                "best_advantage_max": float(best.max()),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_support_summary.csv", index=False)

    scen = opp.groupby("scenario_id").agg(
        fold=("fold", "first"),
        states=("state_id", "nunique"),
        beneficial_states=("best_advantage", lambda s: int((s > 0).sum())),
        harmful_states=("worst_advantage", lambda s: int((s < 0).sum())),
        null_states=("best_advantage", lambda s: int((s == 0).sum())),
        max_best_advantage=("best_advantage", "max"),
    ).reset_index()
    scen.to_csv(out_dir / "scenario_support_summary.csv", index=False)
    pos_counts = scen["beneficial_states"].astype(int).sort_values(ascending=False).to_numpy()
    total_pos_states = int(pos_counts.sum())
    concentration = {}
    for pct in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(math.ceil(len(pos_counts) * pct)))
        concentration[f"top_{int(pct*100)}pct_scenarios_fraction_positive_states"] = (
            float(pos_counts[:k].sum() / total_pos_states) if total_pos_states else 0.0
        )
        concentration[f"top_{int(pct*100)}pct_scenarios_count"] = int(k)
    concentration["beneficial_state_gini_by_scenario"] = gini(pos_counts.tolist())
    scen_summary = {
        "scenarios": int(len(scen)),
        "scenarios_with_beneficial_state": int((scen["beneficial_states"] > 0).sum()),
        "scenarios_with_harmful_state": int((scen["harmful_states"] > 0).sum()),
        "scenarios_only_null_effects": int(((scen["beneficial_states"] == 0) & (scen["harmful_states"] == 0)).sum()),
        "beneficial_states_per_scenario": stats(scen["beneficial_states"]),
        "concentration": concentration,
    }
    write_json(out_dir / "scenario_support_summary.json", scen_summary)

    multiplicity = rows[(rows["is_sbs_action"].astype(int) == 0) & (rows[PRIMARY_ADV] > 0)]["n_policies_generating_action"].astype(int)
    policy_summary = {
        "non_sbs_policy_support_rows": policy_rows,
        "beneficial_action_policy_multiplicity": {
            "exactly_1_policy": int((multiplicity == 1).sum()),
            "exactly_2_policies": int((multiplicity == 2).sum()),
            "three_or_more_policies": int((multiplicity >= 3).sum()),
        },
    }
    write_json(out_dir / "policy_support_summary.json", policy_summary)
    return {
        "fold_support": fold_rows,
        "scenario_support": scen_summary,
        "policy_support": policy_summary,
    }


def episodic_temporal(opp: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    rows = []
    for group_col in ["disagreement_episode_type", "trajectory_position_bin"]:
        for value, g in opp.groupby(group_col, sort=True):
            rows.append(
                {
                    "group": group_col,
                    "value": value,
                    "states": int(len(g)),
                    "beneficial_states": int((g["best_advantage"] > 0).sum()),
                    "harmful_or_mixed_states": int((g["worst_advantage"] < 0).sum()),
                    "null_states": int(((g["best_advantage"] == 0) & (g["worst_advantage"] == 0)).sum()),
                    "beneficial_fraction": float((g["best_advantage"] > 0).mean()),
                    "best_advantage_mean": float(g["best_advantage"].mean()),
                    "best_advantage_median": float(g["best_advantage"].median()),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "episodic_temporal_summary.csv", index=False)
    summary = {
        "episode_and_position_rows": rows,
        "beneficial_states_in_isolated_episodes": int(((opp["best_advantage"] > 0) & (opp["disagreement_episode_type"] == "isolated")).sum()),
        "beneficial_states_in_multistep_episodes": int(((opp["best_advantage"] > 0) & (opp["disagreement_episode_type"] == "multi_step")).sum()),
        "max_beneficial_states_in_one_scenario": int(opp[opp["best_advantage"] > 0].groupby("scenario_id")["state_id"].nunique().max()),
    }
    write_json(out_dir / "episodic_temporal_summary.json", summary)
    return summary


def feature_descriptive(rows: pd.DataFrame, opp: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    feature_cols = [c for c in rows.columns if c.startswith("state__") or c.startswith("action__")]
    non_sbs = rows[rows["is_sbs_action"].astype(int) == 0].copy()
    non_sbs["target_sign"] = np.select(
        [non_sbs[PRIMARY_ADV] > 0, non_sbs[PRIMARY_ADV] < 0],
        ["positive", "negative"],
        default="zero",
    )
    records = []
    for col in feature_cols:
        s = pd.to_numeric(non_sbs[col], errors="coerce")
        pos = s[non_sbs["target_sign"] == "positive"]
        neg = s[non_sbs["target_sign"] == "negative"]
        zero = s[non_sbs["target_sign"] == "zero"]
        rest = s[non_sbs["target_sign"] != "positive"]
        pooled = float(s.std(ddof=0)) if len(s) else 0.0
        smd = (float(pos.mean()) - float(rest.mean())) / pooled if pooled and len(pos) and len(rest) else 0.0
        corr = float(s.corr(pd.to_numeric(non_sbs[PRIMARY_ADV]), method="spearman")) if s.nunique(dropna=True) > 1 else 0.0
        records.append(
            {
                "feature": col,
                "feature_group": classify_feature_group(col),
                "positive_mean": float(pos.mean()) if len(pos) else None,
                "negative_mean": float(neg.mean()) if len(neg) else None,
                "zero_mean": float(zero.mean()) if len(zero) else None,
                "positive_median": float(pos.median()) if len(pos) else None,
                "negative_median": float(neg.median()) if len(neg) else None,
                "zero_median": float(zero.median()) if len(zero) else None,
                "positive_vs_nonpositive_smd": float(smd),
                "spearman_with_advantage": corr if not math.isnan(corr) else 0.0,
                "missing_count": int(s.isna().sum()),
            }
        )
    feature_df = pd.DataFrame(records)
    feature_df.to_csv(out_dir / "feature_descriptive_summary.csv", index=False)
    state_feature_cols = [c for c in rows.columns if c.startswith("state__")]
    state_rows = rows.drop_duplicates("state_id").set_index("state_id")
    state_join = opp.set_index("state_id").join(state_rows[state_feature_cols], how="left")
    state_records = []
    for col in state_feature_cols:
        s = pd.to_numeric(state_join[col], errors="coerce")
        ben = s[state_join["best_advantage"] > 0]
        non = s[state_join["best_advantage"] <= 0]
        pooled = float(s.std(ddof=0)) if len(s) else 0.0
        smd = (float(ben.mean()) - float(non.mean())) / pooled if pooled and len(ben) and len(non) else 0.0
        state_records.append(
            {
                "feature": col,
                "feature_group": classify_feature_group(col),
                "beneficial_state_mean": float(ben.mean()) if len(ben) else None,
                "nonbeneficial_state_mean": float(non.mean()) if len(non) else None,
                "beneficial_state_median": float(ben.median()) if len(ben) else None,
                "nonbeneficial_state_median": float(non.median()) if len(non) else None,
                "beneficial_vs_nonbeneficial_smd": float(smd),
            }
        )
    state_feature_df = pd.DataFrame(state_records)
    state_feature_df.to_csv(out_dir / "state_feature_descriptive_summary.csv", index=False)
    group_summary = (
        feature_df.assign(abs_smd=feature_df["positive_vs_nonpositive_smd"].abs(), abs_corr=feature_df["spearman_with_advantage"].abs())
        .groupby("feature_group")
        .agg(
            features=("feature", "count"),
            max_abs_smd=("abs_smd", "max"),
            median_abs_smd=("abs_smd", "median"),
            max_abs_spearman=("abs_corr", "max"),
            median_abs_spearman=("abs_corr", "median"),
        )
        .reset_index()
    )
    group_summary.to_csv(out_dir / "feature_group_descriptive_summary.csv", index=False)
    top = feature_df.reindex(feature_df["positive_vs_nonpositive_smd"].abs().sort_values(ascending=False).index).head(20)
    summary = {
        "feature_columns": int(len(feature_cols)),
        "state_feature_columns": int(len(state_feature_cols)),
        "action_feature_columns": int(len([c for c in feature_cols if c.startswith("action__")])),
        "feature_group_summary": group_summary.to_dict(orient="records"),
        "top_20_action_level_abs_smd": top.to_dict(orient="records"),
    }
    write_json(out_dir / "feature_descriptive_summary.json", summary)
    return summary


def pilot_vs_full(full_summary: Mapping[str, Any], state_summary: Mapping[str, Any], pilot_dir: Optional[Path], out_dir: Path) -> Dict[str, Any]:
    pilot_expected = {
        "states": 300,
        "positive_unique_action_rate": 0.1258,
        "negative_unique_action_rate": 0.1813,
        "nonzero_unique_action_rate": 0.3072,
        "beneficial_states": 68,
        "beneficial_state_rate": 68 / 300,
    }
    pilot_actual: Dict[str, Any] = {}
    if pilot_dir and (pilot_dir / "state_action_rows.csv").exists():
        pilot_rows = pd.read_csv(pilot_dir / "state_action_rows.csv")
        p_non = pilot_rows[pilot_rows["is_sbs_action"].astype(int) == 0]
        p_adv = pd.to_numeric(p_non[PRIMARY_ADV])
        p_best = p_non.groupby("state_id")[PRIMARY_ADV].max()
        pilot_actual = {
            "states": int(pilot_rows["state_id"].nunique()),
            "non_sbs_unique_actions": int(len(p_non)),
            "positive_unique_action_rate": float((p_adv > 0).mean()),
            "negative_unique_action_rate": float((p_adv < 0).mean()),
            "nonzero_unique_action_rate": float((p_adv != 0).mean()),
            "beneficial_states": int((p_best > 0).sum()),
            "beneficial_state_rate": float((p_best > 0).mean()),
        }
    full_actual = {
        "states": int(state_summary["states"]),
        "positive_unique_action_rate": float(full_summary["positive_prevalence"]),
        "negative_unique_action_rate": float(full_summary["negative_prevalence"]),
        "nonzero_unique_action_rate": float(full_summary["nonzero_prevalence"]),
        "beneficial_states": int(state_summary["beneficial_available_states"]),
        "beneficial_state_rate": float(state_summary["beneficial_available_states"] / state_summary["states"]),
    }
    baseline = pilot_actual or pilot_expected
    deltas = {
        k: (float(full_actual[k]) - float(baseline[k]))
        for k in [
            "positive_unique_action_rate",
            "negative_unique_action_rate",
            "nonzero_unique_action_rate",
            "beneficial_state_rate",
        ]
    }
    comparison = {
        "pilot_preregistered_reference": pilot_expected,
        "pilot_actual_from_artifacts": pilot_actual,
        "full_actual": full_actual,
        "full_minus_pilot": deltas,
        "interpretation": "Pilot preserved the qualitative signal if signs and fold/scenario support agree; full rates supersede pilot rates for planning.",
    }
    write_json(out_dir / "pilot_vs_full_comparison.json", comparison)
    return comparison


def readiness(
    target_summary: Mapping[str, Any],
    state_summary: Mapping[str, Any],
    support: Mapping[str, Any],
    rows: pd.DataFrame,
    non_sbs: pd.DataFrame,
    opp: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Any]:
    pos_rows = non_sbs[non_sbs[PRIMARY_ADV] > 0]
    neg_rows = non_sbs[non_sbs[PRIMARY_ADV] < 0]
    positive_states = set(pos_rows["state_id"])
    negative_states = set(neg_rows["state_id"])
    pos_scen = int(pos_rows["scenario_id"].nunique())
    neg_scen = int(neg_rows["scenario_id"].nunique())
    pos_folds = int(pos_rows["fold"].nunique())
    neg_folds = int(neg_rows["fold"].nunique())
    scen_conc = support["scenario_support"]["concentration"]
    broad = (
        len(positive_states) >= 100
        and pos_scen >= 50
        and pos_folds == EXPECTED_FOLDS
        and len(negative_states) >= 100
        and neg_scen >= 50
        and neg_folds == EXPECTED_FOLDS
        and scen_conc["top_5pct_scenarios_fraction_positive_states"] < 0.75
    )
    if broad:
        verdict = "READY_FOR_HELD_OUT_LEARNABILITY_EXPERIMENT"
    elif len(positive_states) > 0 and (pos_scen < 30 or scen_conc["top_5pct_scenarios_fraction_positive_states"] >= 0.75):
        verdict = "TARGET_SUPPORT_TOO_SPARSE_FOR_RELIABLE_LEARNING"
    elif int(target_summary["positive"]) > 0 and any(r["states_positive"] == 0 for r in support["policy_support"]["non_sbs_policy_support_rows"]):
        verdict = "CAUSAL_EFFECT_PRESENT_BUT_POLICY_SUPPORT_INSUFFICIENT"
    else:
        verdict = "TARGET_SUPPORT_TOO_SPARSE_FOR_RELIABLE_LEARNING"
    summary = {
        "verdict": verdict,
        "positive_unique_action_rows": int(len(pos_rows)),
        "positive_states": int(len(positive_states)),
        "positive_scenarios": pos_scen,
        "positive_folds": pos_folds,
        "negative_unique_action_rows": int(len(neg_rows)),
        "negative_states": int(len(negative_states)),
        "negative_scenarios": neg_scen,
        "negative_folds": neg_folds,
        "null_unique_action_rows": int((non_sbs[PRIMARY_ADV] == 0).sum()),
        "null_states": int(((opp["best_advantage"] == 0) & (opp["worst_advantage"] == 0)).sum()),
        "state_count": int(opp["state_id"].nunique()),
        "scenario_count": int(opp["scenario_id"].nunique()),
        "correlated_action_rows_per_state_mean": float(non_sbs.groupby("state_id").size().mean()),
        "learnable_feature_count_state_action_v1": int(len([c for c in rows.columns if c.startswith("state__") or c.startswith("action__")])),
        "scenario_concentration": scen_conc,
        "quantitative_reason": (
            "Positive and negative effects span all folds and many scenarios, with thousands of state/action rows; "
            "splits must group by scenario/state because action rows within a state are correlated."
        )
        if broad
        else (
            "Effects are present but support/concentration does not yet justify scenario-held-out learning."
        ),
    }
    write_json(out_dir / "learning_readiness_verdict.json", summary)
    return summary


def write_next_protocol(out_dir: Path, readiness_summary: Mapping[str, Any]) -> None:
    if readiness_summary["verdict"] != "READY_FOR_HELD_OUT_LEARNABILITY_EXPERIMENT":
        return
    text = f"""# JOINT240_SBS_OVERRIDE_LEARNABILITY_V1 Preregistered Sketch

This follow-on design is frozen after the full terminal-label dataset integrity
analysis and before any learner is trained.

Primary target: `A_SBS(s,a)` from `JOINT240_SBS_TARGETED_TERMINAL_LABEL_FULL_V1`.

Primary representation: `STATE_ACTION_V1`, the 95 online-safe physical
state features plus the 70 candidate-action-vs-SBS difference features.  Do
not include `state_id`, `scenario_id`, terminal utility columns, policy-map
aliases, or outcome-derived metadata as learnable inputs.

Recommended first formulation: direct advantage regression with an abstaining
SBS fallback.  Also report sign classification and a two-stage nonzero gate
only as preregistered secondary baselines.  The first learner should not use a
large neural network.

Initial model families:
- ridge/elastic-net style linear regression and logistic baselines;
- HistGradientBoosting;
- ExtraTrees or random-forest style ensemble if reported with grouped
  validation and calibration diagnostics.

Splitting:
- use scenario-group-safe held-out evaluation;
- preserve or nest the frozen 5-fold structure where practical;
- never randomly split rows from the same state across train/test;
- keep all unique actions from a state in the same split.

Evaluation:
- compare against always-SBS, which corresponds to choosing no override;
- report decision value under abstention, not just predictive error;
- calibrate/threshold only inside training folds;
- report support by fold, scenario, state, and policy alias.

Readiness basis: `{readiness_summary['verdict']}` with
{readiness_summary['positive_states']} positive states across
{readiness_summary['positive_scenarios']} scenarios and all
{readiness_summary['positive_folds']} folds, plus
{readiness_summary['negative_states']} negative states across
{readiness_summary['negative_scenarios']} scenarios and all
{readiness_summary['negative_folds']} folds.
"""
    (out_dir / "NEXT_LEARNING_PROTOCOL_PREREGISTERED_SKETCH.md").write_text(text)


def analyze(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, maps, provenance = aggregate(run_dir, out_dir, int(args.num_shards), repo_root)
    integrity = integrity_gates(rows, maps, out_dir, provenance)
    if not integrity["all_integrity_gates_pass"]:
        print(json.dumps({"verdict": "FULL_DATASET_INTEGRITY_FAIL", "integrity": integrity}, indent=2, sort_keys=True))
        return
    non_sbs, target_summary = target_distribution(rows, out_dir)
    opp, state_summary = state_level(rows, non_sbs, out_dir)
    support = support_summaries(rows, maps, non_sbs, opp, out_dir)
    episode_summary = episodic_temporal(opp, out_dir)
    feature_summary = feature_descriptive(rows, opp, out_dir)
    pilot_comparison = pilot_vs_full(
        target_summary,
        state_summary,
        Path(args.pilot_dir).resolve() if args.pilot_dir else None,
        out_dir,
    )
    readiness_summary = readiness(target_summary, state_summary, support, rows, non_sbs, opp, out_dir)
    write_next_protocol(out_dir, readiness_summary)
    final = {
        "duplication_gate": "NOT_PREVIOUSLY_DONE",
        "aggregation": provenance,
        "integrity": integrity,
        "target_distribution": target_summary,
        "state_level_opportunity": state_summary,
        "support": support,
        "episodic_temporal": episode_summary,
        "feature_descriptive": feature_summary,
        "pilot_vs_full": pilot_comparison,
        "learning_readiness": readiness_summary,
        "artifact_paths": {
            "state_action_rows_full": str(out_dir / "state_action_rows_full.csv"),
            "state_policy_action_map_full": str(out_dir / "state_policy_action_map_full.csv"),
            "integrity_report": str(out_dir / "full_integrity_report.json"),
            "target_distribution": str(out_dir / "target_distribution_summary.json"),
            "state_level_opportunity": str(out_dir / "state_level_opportunity.csv"),
            "fold_support": str(out_dir / "fold_support_summary.csv"),
            "scenario_support": str(out_dir / "scenario_support_summary.csv"),
            "policy_support": str(out_dir / "policy_support_summary.csv"),
            "learning_readiness": str(out_dir / "learning_readiness_verdict.json"),
        },
    }
    write_json(out_dir / "post_completion_analysis_summary.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument(
        "--run-dir",
        default="experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1",
    )
    p.add_argument("--pilot-dir", default="")
    p.add_argument("--num-shards", type=int, default=EXPECTED_SHARDS)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("analyze")
    sp.set_defaults(func=analyze)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
