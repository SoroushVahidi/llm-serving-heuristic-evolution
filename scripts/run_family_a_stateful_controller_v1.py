#!/usr/bin/env python3
"""Family-A stateful controller V1 TRAIN/VAL evaluation.

This script follows docs/design/FAMILY_A_STATEFUL_CONTROLLER_V1.md. It never
loads TEST rows and stops before simulation if the frozen offline gate fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.analysis import family_a_observability_continuation_v1 as family_a_obs
from llmserveopt.core.metrics import metrics_to_dict
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.family_a_stateful_controller_v1 import (
    ESTF_MODE,
    WFS_MODE,
    FamilyAStatefulControllerV1,
    FamilyAStatelessTreeControllerV1,
    FrozenTreeModeModel,
    STATEFUL_CONTROLLER_FEATURES,
    validate_feature_names,
)
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


DESIGN_PATH = REPO_ROOT / "docs/design/FAMILY_A_STATEFUL_CONTROLLER_V1.md"
REPAIRED_EVENTS_PATH = (
    REPO_ROOT
    / "experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv"
)
OUTPUT_DIR = REPO_ROOT / "experiments/family_a_stateful_controller_v1"
ANALYSIS_PATH = REPO_ROOT / "docs/current/family_a_stateful_controller_v1_analysis_20260820.md"

PRIMARY_DWELL = 20
ESTF_ENTER_THRESHOLD = 0.65
WFS_ENTER_THRESHOLD = 0.35
TREE_RANDOM_STATE = 20260820


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def finite_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [json_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return finite_float(value)
    if isinstance(value, float):
        return finite_float(value)
    return value


def load_events() -> pd.DataFrame:
    events = pd.read_csv(REPAIRED_EVENTS_PATH)
    required = {"canonical_scenario_id", "split", "step", "delta_native", *STATEFUL_CONTROLLER_FEATURES}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise RuntimeError(f"repaired event table missing required columns: {missing}")
    if (events["split"].str.lower() == "test").any():
        raise RuntimeError("TEST row leaked into repaired event table")
    if len(events) != 91:
        raise RuntimeError(f"expected 91 repaired events, found {len(events)}")
    validate_feature_names(STATEFUL_CONTROLLER_FEATURES)
    return events.copy()


def event_target(events: pd.DataFrame) -> np.ndarray:
    return (events["delta_native"].astype(float).to_numpy() > 0.0).astype(int)


def event_matrix(events: pd.DataFrame) -> np.ndarray:
    X = events.loc[:, STATEFUL_CONTROLLER_FEATURES].astype(float)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def fit_tree(events: pd.DataFrame) -> DecisionTreeClassifier:
    X = event_matrix(events)
    y = event_target(events)
    tree = DecisionTreeClassifier(
        max_depth=3,
        class_weight="balanced",
        random_state=TREE_RANDOM_STATE,
    )
    tree.fit(X, y)
    return tree


def _auc_or_none(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def grouped_cv(events: pd.DataFrame) -> Dict[str, Any]:
    X = event_matrix(events)
    y = event_target(events)
    groups = events["canonical_scenario_id"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    validate_grouped_splits(groups, splitter.split(X, y, groups))

    rows: List[Dict[str, Any]] = []
    all_tree_pred: List[int] = []
    all_tree_prob: List[float] = []
    all_base_pred: List[int] = []
    all_base_prob: List[float] = []
    all_true: List[int] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        tree = DecisionTreeClassifier(
            max_depth=3,
            class_weight="balanced",
            random_state=TREE_RANDOM_STATE,
        )
        tree.fit(X[train_idx], y[train_idx])
        tree_pred = tree.predict(X[val_idx]).astype(int)
        tree_prob = tree.predict_proba(X[val_idx])[:, list(tree.classes_).index(1)]

        train_positive_rate = float(np.mean(y[train_idx]))
        majority_label = int(train_positive_rate >= 0.5)
        base_pred = np.full(len(val_idx), majority_label, dtype=int)
        base_prob = np.full(len(val_idx), train_positive_rate, dtype=float)

        y_val = y[val_idx]
        rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "train_scenarios": int(len(np.unique(groups[train_idx]))),
                "val_scenarios": int(len(np.unique(groups[val_idx]))),
                "tree_balanced_accuracy": float(balanced_accuracy_score(y_val, tree_pred)),
                "tree_auc": _auc_or_none(y_val, tree_prob),
                "tree_macro_f1": float(f1_score(y_val, tree_pred, average="macro", zero_division=0)),
                "baseline_balanced_accuracy": float(balanced_accuracy_score(y_val, base_pred)),
                "baseline_auc": _auc_or_none(y_val, base_prob),
                "baseline_macro_f1": float(f1_score(y_val, base_pred, average="macro", zero_division=0)),
                "confusion_matrix_tree": confusion_matrix(y_val, tree_pred, labels=[0, 1]).tolist(),
                "confusion_matrix_baseline": confusion_matrix(y_val, base_pred, labels=[0, 1]).tolist(),
            }
        )
        all_tree_pred.extend(tree_pred.tolist())
        all_tree_prob.extend(tree_prob.tolist())
        all_base_pred.extend(base_pred.tolist())
        all_base_prob.extend(base_prob.tolist())
        all_true.extend(y_val.tolist())

    y_all = np.asarray(all_true, dtype=int)
    tree_pred_all = np.asarray(all_tree_pred, dtype=int)
    tree_prob_all = np.asarray(all_tree_prob, dtype=float)
    base_pred_all = np.asarray(all_base_pred, dtype=int)
    base_prob_all = np.asarray(all_base_prob, dtype=float)

    return {
        "target": "ESTF_MODE if delta_native > 0 else WFS_MODE",
        "n_events": int(len(events)),
        "n_scenarios": int(events["canonical_scenario_id"].nunique()),
        "class_counts": {
            "wfs_or_zero": int(np.sum(y == 0)),
            "estf": int(np.sum(y == 1)),
        },
        "folds": rows,
        "tree": {
            "balanced_accuracy": float(balanced_accuracy_score(y_all, tree_pred_all)),
            "auc": _auc_or_none(y_all, tree_prob_all),
            "macro_f1": float(f1_score(y_all, tree_pred_all, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_all, tree_pred_all, labels=[0, 1]).tolist(),
            "fold_mean_balanced_accuracy": float(np.mean([r["tree_balanced_accuracy"] for r in rows])),
            "fold_std_balanced_accuracy": float(np.std([r["tree_balanced_accuracy"] for r in rows])),
            "fold_mean_macro_f1": float(np.mean([r["tree_macro_f1"] for r in rows])),
            "fold_std_macro_f1": float(np.std([r["tree_macro_f1"] for r in rows])),
        },
        "baseline": {
            "balanced_accuracy": float(balanced_accuracy_score(y_all, base_pred_all)),
            "auc": _auc_or_none(y_all, base_prob_all),
            "macro_f1": float(f1_score(y_all, base_pred_all, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_all, base_pred_all, labels=[0, 1]).tolist(),
            "fold_mean_balanced_accuracy": float(np.mean([r["baseline_balanced_accuracy"] for r in rows])),
            "fold_std_balanced_accuracy": float(np.std([r["baseline_balanced_accuracy"] for r in rows])),
            "fold_mean_macro_f1": float(np.mean([r["baseline_macro_f1"] for r in rows])),
            "fold_std_macro_f1": float(np.std([r["baseline_macro_f1"] for r in rows])),
        },
    }


def validate_grouped_splits(groups: np.ndarray, splits: Iterable[Tuple[np.ndarray, np.ndarray]]) -> None:
    for train_idx, val_idx in splits:
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise RuntimeError(f"grouped split leakage: {sorted(overlap)[:3]}")


def predict_tree_probability(tree: DecisionTreeClassifier, rows: pd.DataFrame) -> np.ndarray:
    X = rows.loc[:, STATEFUL_CONTROLLER_FEATURES].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return tree.predict_proba(X)[:, list(tree.classes_).index(1)]


def offline_replay(events: pd.DataFrame, tree: DecisionTreeClassifier) -> Dict[str, Any]:
    rows = events.copy()
    rows["_prob_estf"] = predict_tree_probability(tree, rows)
    rows = rows.sort_values(["canonical_scenario_id", "step"]).reset_index(drop=True)

    mode_counts = {ESTF_MODE: 0, WFS_MODE: 0}
    switches: List[Dict[str, Any]] = []
    abstentions = 0
    dwell_violations = 0
    segment_lengths: List[int] = []

    for scenario_id, group in rows.groupby("canonical_scenario_id", sort=False):
        mode = WFS_MODE
        last_step = 0
        last_switch_step = 0
        current_segment_start = int(group["step"].iloc[0])
        for _, row in group.iterrows():
            step = int(row["step"])
            dwell_elapsed = step - last_switch_step
            probability = float(row["_prob_estf"])
            switched = False
            if dwell_elapsed >= PRIMARY_DWELL and mode == WFS_MODE and probability >= ESTF_ENTER_THRESHOLD:
                segment_lengths.append(max(0, step - current_segment_start))
                mode_before = mode
                mode = ESTF_MODE
                switches.append(
                    {
                        "scenario_id": scenario_id,
                        "step": step,
                        "direction": f"{mode_before}->{mode}",
                        "prob_estf": probability,
                    }
                )
                switched = True
                last_switch_step = step
                current_segment_start = step
            elif dwell_elapsed >= PRIMARY_DWELL and mode == ESTF_MODE and probability <= WFS_ENTER_THRESHOLD:
                segment_lengths.append(max(0, step - current_segment_start))
                mode_before = mode
                mode = WFS_MODE
                switches.append(
                    {
                        "scenario_id": scenario_id,
                        "step": step,
                        "direction": f"{mode_before}->{mode}",
                        "prob_estf": probability,
                    }
                )
                switched = True
                last_switch_step = step
                current_segment_start = step
            if not switched:
                abstentions += 1
            mode_counts[mode] += 1
            last_step = step
        if len(group):
            segment_lengths.append(max(0, last_step - current_segment_start))

    total = int(sum(mode_counts.values()))
    short_segments = [seg for seg in segment_lengths if 0 < seg < PRIMARY_DWELL]
    dwell_violations = len(short_segments)
    return {
        "n_candidate_events": total,
        "abstention_count": int(abstentions),
        "abstention_rate": float(abstentions / total) if total else 1.0,
        "mode_counts": mode_counts,
        "estf_event_share": float(mode_counts[ESTF_MODE] / total) if total else 0.0,
        "wfs_event_share": float(mode_counts[WFS_MODE] / total) if total else 0.0,
        "switch_count": int(len(switches)),
        "switch_directions": {
            direction: int(sum(1 for sw in switches if sw["direction"] == direction))
            for direction in sorted({sw["direction"] for sw in switches})
        },
        "dwell_segment_lengths": segment_lengths,
        "dwell_violations": int(dwell_violations),
        "short_segments": short_segments,
        "switches_preview": switches[:20],
        "prob_estf_summary": numeric_summary(rows["_prob_estf"].to_numpy(dtype=float)),
    }


def offline_gate(cv: Mapping[str, Any], replay: Mapping[str, Any]) -> Dict[str, Any]:
    tree = cv["tree"]
    baseline = cv["baseline"]
    reasons: List[str] = []
    if tree["balanced_accuracy"] <= baseline["balanced_accuracy"]:
        reasons.append("tree balanced accuracy did not exceed majority baseline")
    if tree["macro_f1"] <= baseline["macro_f1"]:
        reasons.append("tree macro F1 did not exceed majority baseline")
    if tree["auc"] is not None and tree["auc"] <= 0.50:
        reasons.append("tree ROC-AUC was not above 0.50")
    if replay["estf_event_share"] < 0.10 or replay["estf_event_share"] > 0.90:
        reasons.append("offline replay did not use ESTF/WFS modes nontrivially")
    if replay["wfs_event_share"] < 0.10 or replay["wfs_event_share"] > 0.90:
        reasons.append("offline replay did not use WFS/ESTF modes nontrivially")
    if replay["abstention_rate"] > 0.90:
        reasons.append("offline replay abstention rate exceeded 90%")
    if replay["dwell_violations"] > 0:
        reasons.append("offline replay violated 20-step dwell in event step-space")
    return {"go": not reasons, "reasons": reasons}


def numeric_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if finite_float(v) is not None], dtype=float)
    if arr.size == 0:
        return {k: None for k in ("mean", "median", "p25", "p75", "min", "max")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_policy_on_scenario(policy: Any, row: pd.Series, policy_id: str) -> Dict[str, Any]:
    family_a_obs.assert_trainval_only(row["split"])
    scenario = family_a_obs.rebuild_scenario_from_row(row)
    service_model = ServiceModel(**scenario.service_model_kwargs)
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=service_model,
        )
    )
    if hasattr(policy, "reset"):
        policy.reset()
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(policy, workload_tag=str(row["canonical_scenario_id"]), seed=int(scenario.seed))
    result = metrics_to_dict(metrics)
    result.update(
        {
            "canonical_scenario_id": str(row["canonical_scenario_id"]),
            "split": str(row["split"]),
            "policy_id": policy_id,
        }
    )
    if hasattr(policy, "diagnostics"):
        result["controller_diagnostics"] = policy.diagnostics()
    else:
        result["controller_diagnostics"] = None
    return result


def run_trainval_simulation(events: pd.DataFrame, tree: DecisionTreeClassifier) -> Dict[str, Any]:
    table = family_a_obs.load_family_a_trainval_scenario_table()
    if (table["split"].str.lower() == "test").any():
        raise RuntimeError("TEST row leaked into Family-A TRAIN/VAL scenario table")
    frozen_model = FrozenTreeModeModel.from_sklearn(tree, STATEFUL_CONTROLLER_FEATURES)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    start = utc_now()
    t0 = time.perf_counter()

    for _, row in table.iterrows():
        scenario_id = str(row["canonical_scenario_id"])
        try:
            policies = {
                "estimated_service_time_first": EstimatedServiceTimeFirstPolicy(),
                "weighted_fair_share": WeightedFairSharePolicy(),
                "family_a_stateless_tree_controller_v1": FamilyAStatelessTreeControllerV1(
                    mode_model=frozen_model,
                    step_size=float(ServiceModel(**family_a_obs.rebuild_scenario_from_row(row).service_model_kwargs).step_size),
                ),
                "family_a_stateful_controller_v1": FamilyAStatefulControllerV1(
                    mode_model=frozen_model,
                    step_size=float(ServiceModel(**family_a_obs.rebuild_scenario_from_row(row).service_model_kwargs).step_size),
                    min_dwell_steps=PRIMARY_DWELL,
                    estf_enter_threshold=ESTF_ENTER_THRESHOLD,
                    wfs_enter_threshold=WFS_ENTER_THRESHOLD,
                ),
            }
            for policy_id, policy in policies.items():
                rows.append(run_policy_on_scenario(policy, row, policy_id))
        except Exception as exc:
            failures.append({"canonical_scenario_id": scenario_id, "error": repr(exc)})

    elapsed = time.perf_counter() - t0
    results_df = pd.DataFrame(rows)
    per_scenario_path = OUTPUT_DIR / "family_a_stateful_controller_v1_per_scenario_results.csv"
    results_df.to_csv(per_scenario_path, index=False)

    summary = summarize_simulation(results_df)
    return {
        "start_time_utc": start,
        "end_time_utc": utc_now(),
        "wall_clock_s": elapsed,
        "scenario_count": int(table["canonical_scenario_id"].nunique()),
        "split_counts": {str(k): int(v) for k, v in table["split"].value_counts().to_dict().items()},
        "failures": failures,
        "per_scenario_results_path": str(per_scenario_path.relative_to(REPO_ROOT)),
        "summary": summary,
    }


def summarize_simulation(results_df: pd.DataFrame) -> Dict[str, Any]:
    if results_df.empty:
        return {}
    metric = "arrival_normalized_weighted_goodput"
    policy_means = {
        policy: finite_float(group[metric].mean())
        for policy, group in results_df.groupby("policy_id")
    }
    pivot = results_df.pivot_table(
        index="canonical_scenario_id",
        columns="policy_id",
        values=metric,
        aggfunc="first",
    )
    estf = pivot.get("estimated_service_time_first")
    wfs = pivot.get("weighted_fair_share")
    stateful = pivot.get("family_a_stateful_controller_v1")
    stateless = pivot.get("family_a_stateless_tree_controller_v1")
    native_envelope = pd.concat([estf, wfs], axis=1).max(axis=1)
    best_fixed_mean = max(policy_means.get("estimated_service_time_first") or -math.inf,
                          policy_means.get("weighted_fair_share") or -math.inf)
    paired = {}
    if stateful is not None:
        for name, series in {
            "estimated_service_time_first": estf,
            "weighted_fair_share": wfs,
            "best_fixed_parent_by_scenario": native_envelope,
            "stateless_tree": stateless,
        }.items():
            if series is None:
                continue
            diff = stateful - series
            paired[name] = {
                "mean_diff": finite_float(diff.mean()),
                "median_diff": finite_float(diff.median()),
                "wins": int((diff > 1e-12).sum()),
                "ties": int((diff.abs() <= 1e-12).sum()),
                "losses": int((diff < -1e-12).sum()),
            }
    controller_rows = results_df[results_df["policy_id"] == "family_a_stateful_controller_v1"]
    switch_counts: List[int] = []
    estf_occ: List[float] = []
    wfs_occ: List[float] = []
    for diag in controller_rows["controller_diagnostics"].dropna():
        if isinstance(diag, dict):
            switch_counts.append(int(diag.get("switch_count", 0)))
            estf_occ.append(float(diag.get("estf_occupancy_fraction", 0.0)))
            wfs_occ.append(float(diag.get("wfs_occupancy_fraction", 0.0)))
    safety_metrics = [
        "completion_fraction",
        "weighted_completion_fraction",
        "p95_latency",
        "p95_queuing_delay",
        "slo_violation_rate",
    ]
    safety = {
        metric_name: {
            policy: finite_float(group[metric_name].mean())
            for policy, group in results_df.groupby("policy_id")
        }
        for metric_name in safety_metrics
        if metric_name in results_df.columns
    }
    return {
        "policy_mean_anwg": policy_means,
        "best_fixed_parent_mean_anwg": finite_float(best_fixed_mean),
        "native_pair_envelope_mean_anwg": finite_float(native_envelope.mean()),
        "paired_vs_stateful": paired,
        "controller_switch_count_summary": numeric_summary(switch_counts),
        "controller_estf_occupancy_summary": numeric_summary(estf_occ),
        "controller_wfs_occupancy_summary": numeric_summary(wfs_occ),
        "safety_metric_means": safety,
        "six_policy_portfolio": "not_computed_in_v1_first_internal_run",
    }


def classify_result(gate: Mapping[str, Any], simulation: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    if not gate["go"]:
        return "STATEFUL_CONTROLLER_OFFLINE_NO_GO", "STOP_FAMILY_A_CONSTRUCTION"
    if simulation is None or simulation.get("failures"):
        return "NEED_CONTROLLER_INTEGRITY_REPAIR", "NEED_CONTROLLER_INTEGRITY_REPAIR"
    summary = simulation.get("summary", {})
    means = summary.get("policy_mean_anwg", {})
    paired = summary.get("paired_vs_stateful", {})
    stateful_mean = means.get("family_a_stateful_controller_v1")
    best_fixed = summary.get("best_fixed_parent_mean_anwg")
    occupancy = summary.get("controller_estf_occupancy_summary", {})
    estf_occ = occupancy.get("mean") if isinstance(occupancy, dict) else None
    both_modes = estf_occ is not None and 0.05 < estf_occ < 0.95
    best_pair = paired.get("best_fixed_parent_by_scenario", {})
    if (
        stateful_mean is not None
        and best_fixed is not None
        and stateful_mean > best_fixed
        and best_pair.get("wins", 0) > best_pair.get("losses", 0)
        and both_modes
    ):
        return "STATEFUL_CONTROLLER_POSITIVE_SIGNAL", "FREEZE_CONTROLLER_AND_PREPARE_TEST"
    if stateful_mean is not None and best_fixed is not None and stateful_mean >= best_fixed and both_modes:
        return "STATEFUL_CONTROLLER_MIXED_SIGNAL", "REFINE_CONTROLLER_ON_TRAINVAL"
    return "STATEFUL_CONTROLLER_NO_GO", "STOP_FAMILY_A_CONSTRUCTION"


def write_analysis_report(
    *,
    cv: Mapping[str, Any],
    replay: Mapping[str, Any],
    gate: Mapping[str, Any],
    simulation: Optional[Mapping[str, Any]],
    classification: str,
    next_step: str,
    command: str,
) -> None:
    tree = cv["tree"]
    baseline = cv["baseline"]
    lines = [
        "# Family-A Stateful Controller V1 Analysis",
        "",
        f"Date: 2026-08-20",
        "",
        "## Executive Verdict",
        "",
        f"Classification: `{classification}`.",
        "",
        f"Next step: `{next_step}`.",
        "",
        "The V1 controller was evaluated strictly on repaired Family-A TRAIN/VAL inputs. TEST was not loaded.",
        "",
        "## Controller",
        "",
        "- Representation: `STATEFUL_TREE` shallow decision tree.",
        "- Initial mode: `WFS_MODE`.",
        f"- Minimum dwell: `{PRIMARY_DWELL}` scheduler steps.",
        f"- Hysteresis: WFS to ESTF at `P(ESTF_MODE) >= {ESTF_ENTER_THRESHOLD}`; ESTF to WFS at `P(ESTF_MODE) <= {WFS_ENTER_THRESHOLD}`.",
        "- Candidate gate: switching evidence is evaluated only when ESTF and WFS produce different canonical actions on the current observable state.",
        "",
        "## Offline Feasibility",
        "",
        f"- Events: {cv['n_events']}",
        f"- Scenarios with events: {cv['n_scenarios']}",
        f"- Class counts: {cv['class_counts']}",
        f"- Majority baseline balanced accuracy: {baseline['balanced_accuracy']:.6f}",
        f"- Tree balanced accuracy: {tree['balanced_accuracy']:.6f}",
        f"- Majority baseline AUC: {baseline['auc']}",
        f"- Tree AUC: {tree['auc']}",
        f"- Majority baseline macro F1: {baseline['macro_f1']:.6f}",
        f"- Tree macro F1: {tree['macro_f1']:.6f}",
        f"- Confusion matrix tree [WFS/zero, ESTF]: `{tree['confusion_matrix']}`",
        "",
        "## Offline Dwell Replay",
        "",
        f"- Abstention rate: {replay['abstention_rate']:.6f}",
        f"- ESTF event share: {replay['estf_event_share']:.6f}",
        f"- WFS event share: {replay['wfs_event_share']:.6f}",
        f"- Switch count: {replay['switch_count']}",
        f"- Switch directions: {replay['switch_directions']}",
        f"- Dwell violations: {replay['dwell_violations']}",
        "",
        "## Offline Gate",
        "",
        f"- GO: `{gate['go']}`",
        f"- Reasons: {gate['reasons'] if gate['reasons'] else 'none'}",
        "",
    ]
    if simulation is None:
        lines.extend(
            [
                "## Full Simulation",
                "",
                "Full TRAIN/VAL simulation was not launched because the offline gate failed.",
                "",
            ]
        )
    else:
        summary = simulation["summary"]
        lines.extend(
            [
                "## Full Simulation",
                "",
                f"- Scenario count: {simulation['scenario_count']}",
                f"- Failures: {len(simulation['failures'])}",
                f"- Wall clock seconds: {simulation['wall_clock_s']:.3f}",
                f"- Mean ANWG by policy: `{summary.get('policy_mean_anwg', {})}`",
                f"- Paired differences vs stateful: `{summary.get('paired_vs_stateful', {})}`",
                f"- Switch-count summary: `{summary.get('controller_switch_count_summary', {})}`",
                f"- ESTF occupancy summary: `{summary.get('controller_estf_occupancy_summary', {})}`",
                f"- WFS occupancy summary: `{summary.get('controller_wfs_occupancy_summary', {})}`",
                f"- Safety metric means: `{summary.get('safety_metric_means', {})}`",
                f"- Six-policy portfolio: `{summary.get('six_policy_portfolio')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Novelty Guard",
            "",
            "This V1 is deliberately close to known state-dependent scheduling ideas. If it succeeds, novelty still remains at risk against FSP-style fairness/SRPT hybrids, T-SRPT-style state-dependent switching, VTC-style fairness baselines, vLLM-LTR service-time ranking, and PARS-like learned service-aware ranking. Those external baselines are not integrated in this first internal feasibility run.",
            "",
            "## Limitations",
            "",
            "- Only 91 repaired diagnostic events supervise the offline scorer.",
            "- Only 32/64 Family-A scenarios have repaired events.",
            "- TRAIN/VAL only.",
            "- Grouped CV uncertainty remains high.",
            "- Event-only labels are guarded by a candidate region but still differ from ordinary online state distribution.",
            "- No TEST, public-trace, or real-serving validation has been run.",
            "- Six-policy portfolio marginal contribution was not computed unless reported above.",
            "",
            "## Artifacts And Reproducibility",
            "",
            f"- Design: `docs/design/FAMILY_A_STATEFUL_CONTROLLER_V1.md`",
            f"- Analysis: `docs/current/family_a_stateful_controller_v1_analysis_20260820.md`",
            f"- Experiment dir: `experiments/family_a_stateful_controller_v1/`",
            f"- Command: `{command}`",
        ]
    )
    ANALYSIS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-only", action="store_true", help="stop after offline gate regardless of outcome")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = " ".join([str(Path(sys.argv[0]).as_posix()), *sys.argv[1:]])
    events = load_events()
    cv = grouped_cv(events)
    tree = fit_tree(events)
    frozen_model = FrozenTreeModeModel.from_sklearn(tree, STATEFUL_CONTROLLER_FEATURES)
    replay = offline_replay(events, tree)
    gate = offline_gate(cv, replay)

    provenance = {
        "schema_version": "family_a_stateful_controller_v1.0",
        "start_time_utc": utc_now(),
        "command": command,
        "git_head": git_text(["rev-parse", "HEAD"]),
        "git_status": git_text(["status", "--short"]),
        "design_sha256": sha256_file(DESIGN_PATH),
        "repaired_events_sha256": sha256_file(REPAIRED_EVENTS_PATH),
        "feature_names": list(STATEFUL_CONTROLLER_FEATURES),
        "tree_model": frozen_model.to_json_dict(),
    }

    offline_payload = {
        "provenance": provenance,
        "grouped_cv": cv,
        "offline_replay": replay,
        "offline_gate": gate,
    }
    offline_path = OUTPUT_DIR / "family_a_stateful_controller_v1_offline_feasibility.json"
    offline_path.write_text(json.dumps(json_clean(offline_payload), indent=2, sort_keys=True), encoding="utf-8")

    simulation: Optional[Dict[str, Any]] = None
    if gate["go"] and not args.offline_only:
        simulation = run_trainval_simulation(events, tree)
        simulation_path = OUTPUT_DIR / "family_a_stateful_controller_v1_results.json"
        simulation_path.write_text(json.dumps(json_clean(simulation), indent=2, sort_keys=True), encoding="utf-8")

    classification, next_step = classify_result(gate, simulation)
    final_payload = {
        **offline_payload,
        "simulation": simulation,
        "classification": classification,
        "next_step": next_step,
        "analysis_path": str(ANALYSIS_PATH.relative_to(REPO_ROOT)),
    }
    final_path = OUTPUT_DIR / "family_a_stateful_controller_v1_summary.json"
    final_path.write_text(json.dumps(json_clean(final_payload), indent=2, sort_keys=True), encoding="utf-8")

    write_analysis_report(
        cv=cv,
        replay=replay,
        gate=gate,
        simulation=simulation,
        classification=classification,
        next_step=next_step,
        command=command,
    )

    print(json.dumps(json_clean({"classification": classification, "next_step": next_step, "gate": gate}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
