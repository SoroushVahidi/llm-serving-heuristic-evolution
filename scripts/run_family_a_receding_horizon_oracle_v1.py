#!/usr/bin/env python3
"""Family-A receding-horizon oracle feasibility V1 TRAIN/VAL evaluation.

Follows docs/design/FAMILY_A_RECEDING_HORIZON_ORACLE_V1.md. TRAIN/VAL only;
never loads a TEST scenario. Supports --pilot N for the SS12 integrity-gate
pilot (a handful of TRAIN scenarios, all horizons) before committing to the
full 64-scenario x 3-horizon run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.analysis import family_a_observability_continuation_v1 as family_a_obs
from llmserveopt.core.metrics import metrics_to_dict
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.family_a_receding_horizon_oracle_v1 import (
    COMMON_CONTINUATION_BUDGET,
    ESTF_ID,
    WFS_ID,
    FamilyARecedingHorizonOracleV1,
)
from llmserveopt.policies.family_a_stateful_controller_v1 import (
    FamilyAStatefulControllerV1,
    FrozenTreeModeModel,
    STATEFUL_CONTROLLER_FEATURES,
)
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

DESIGN_PATH = REPO_ROOT / "docs/design/FAMILY_A_RECEDING_HORIZON_ORACLE_V1.md"
STATEFUL_DESIGN_PATH = REPO_ROOT / "docs/design/FAMILY_A_STATEFUL_CONTROLLER_V1.md"
REPAIRED_EVENTS_PATH = (
    REPO_ROOT
    / "experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv"
)
OUTPUT_DIR = REPO_ROOT / "experiments/family_a_receding_horizon_oracle_v1"
ANALYSIS_PATH = REPO_ROOT / "docs/current/family_a_receding_horizon_oracle_v1_analysis_20260820.md"

HORIZONS: Sequence[int] = (1, 5, 20)
STATEFUL_PRIMARY_DWELL = 20
STATEFUL_ESTF_ENTER = 0.65
STATEFUL_WFS_ENTER = 0.35
TREE_RANDOM_STATE = 20260820

#: Compute-safety bound only (design doc SS15) -- not a scientific tuning
#: parameter. Sized from a pilot measurement (~0.5s/planning-call at H=20,
#: worst observed eligible_count ~60/scenario in a 12-scenario probe): 150
#: bounds worst-case single-scenario planning wall-clock to a few minutes
#: even if a VAL scenario has substantially more eligible decisions than
#: anything seen in the pilot.
DEFAULT_MAX_PLANNING_CALLS_PER_SCENARIO = 150

EXPECTED_FAMILY_A_TOTAL = 64


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_text(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc!r}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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


# ---------------------------------------------------------------------------
# Baseline #3: refit family_a_stateful_controller_v1 identically (byte-
# identical recipe to scripts/run_family_a_stateful_controller_v1.py).
# ---------------------------------------------------------------------------

def load_repaired_events() -> pd.DataFrame:
    events = pd.read_csv(REPAIRED_EVENTS_PATH)
    if (events["split"].str.lower() == "test").any():
        raise RuntimeError("TEST row leaked into repaired event table")
    if len(events) != 91:
        raise RuntimeError(f"expected 91 repaired events, found {len(events)}")
    return events


def fit_stateful_controller_tree(events: pd.DataFrame) -> FrozenTreeModeModel:
    X = events.loc[:, STATEFUL_CONTROLLER_FEATURES].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = (events["delta_native"].astype(float).to_numpy() > 0.0).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=TREE_RANDOM_STATE)
    tree.fit(X, y)
    return FrozenTreeModeModel.from_sklearn(tree, STATEFUL_CONTROLLER_FEATURES)


# ---------------------------------------------------------------------------
# Per-scenario / per-policy execution
# ---------------------------------------------------------------------------

def build_sim(row: pd.Series) -> Simulator:
    scenario = family_a_obs.rebuild_scenario_from_row(row)
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    return sim, scenario


def run_one(row: pd.Series, policy_id: str, policy_factory, max_planning_calls: int) -> Dict[str, Any]:
    sim, scenario = build_sim(row)
    if policy_id.startswith("family_a_receding_horizon_oracle_v1"):
        horizon = int(policy_id.rsplit("_h", 1)[1])
        policy = FamilyARecedingHorizonOracleV1(
            sim_ref=sim, horizon=horizon, continuation_budget=COMMON_CONTINUATION_BUDGET,
            max_planning_calls_per_scenario=max_planning_calls,
        )
    else:
        policy = policy_factory()
    if hasattr(policy, "reset"):
        policy.reset()
    t0 = time.perf_counter()
    metrics = sim.run(policy, workload_tag=str(row["canonical_scenario_id"]), seed=int(scenario.seed))
    wall = time.perf_counter() - t0
    result = metrics_to_dict(metrics)
    result.update({
        "canonical_scenario_id": str(row["canonical_scenario_id"]),
        "split": str(row["split"]),
        "policy_id": policy_id,
        "scenario_wall_clock_s": wall,
    })
    result["controller_diagnostics"] = policy.diagnostics() if hasattr(policy, "diagnostics") else None
    return result


def run_trainval(table: pd.DataFrame, tree_model: FrozenTreeModeModel, max_planning_calls: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    start = utc_now()
    t0 = time.perf_counter()

    n = len(table)
    for i, (_, row) in enumerate(table.iterrows()):
        sid = str(row["canonical_scenario_id"])
        step_size = float(ServiceModel(**family_a_obs.rebuild_scenario_from_row(row).service_model_kwargs).step_size)
        policy_specs = {
            ESTF_ID: lambda: EstimatedServiceTimeFirstPolicy(),
            WFS_ID: lambda: WeightedFairSharePolicy(),
            "family_a_stateful_controller_v1": lambda: FamilyAStatefulControllerV1(
                mode_model=tree_model, step_size=step_size, min_dwell_steps=STATEFUL_PRIMARY_DWELL,
                estf_enter_threshold=STATEFUL_ESTF_ENTER, wfs_enter_threshold=STATEFUL_WFS_ENTER,
            ),
        }
        for h in HORIZONS:
            policy_specs[f"family_a_receding_horizon_oracle_v1_h{h}"] = None  # built in run_one
        try:
            for policy_id, factory in policy_specs.items():
                rows.append(run_one(row, policy_id, factory, max_planning_calls))
        except Exception as exc:  # noqa: BLE001 -- long unattended run
            failures.append({"canonical_scenario_id": sid, "error": repr(exc)})
            print(f"[{i + 1}/{n}] FAILED {sid}: {exc!r}", flush=True)
            continue
        print(f"[{i + 1}/{n}] {sid} ({row['split']}) done", flush=True)

    elapsed = time.perf_counter() - t0
    results_df = pd.DataFrame(rows)
    per_scenario_path = OUTPUT_DIR / "family_a_receding_horizon_oracle_v1_per_scenario_results.csv"
    results_df.to_csv(per_scenario_path, index=False)

    return {
        "start_time_utc": start,
        "end_time_utc": utc_now(),
        "wall_clock_s": elapsed,
        "scenario_count": int(table["canonical_scenario_id"].nunique()),
        "split_counts": {str(k): int(v) for k, v in table["split"].value_counts().to_dict().items()},
        "failures": failures,
        "per_scenario_results_path": str(per_scenario_path.relative_to(REPO_ROOT)),
        "results_df": results_df,
    }


# ---------------------------------------------------------------------------
# Summary / GO-NO_GO
# ---------------------------------------------------------------------------

SAFETY_METRICS = [
    "completion_fraction", "weighted_completion_fraction",
    "p95_latency", "p95_queuing_delay", "slo_violation_rate",
]


def summarize(results_df: pd.DataFrame) -> Dict[str, Any]:
    metric = "arrival_normalized_weighted_goodput"
    policy_ids = sorted(results_df["policy_id"].unique())
    policy_means = {p: finite_float(results_df.loc[results_df.policy_id == p, metric].mean()) for p in policy_ids}

    pivot = results_df.pivot_table(index="canonical_scenario_id", columns="policy_id", values=metric, aggfunc="first")
    estf = pivot.get(ESTF_ID)
    wfs = pivot.get(WFS_ID)
    native_envelope = pd.concat([estf, wfs], axis=1).max(axis=1)
    best_fixed_series = native_envelope  # per-scenario best of the two fixed parents
    best_fixed_mean = max(policy_means.get(ESTF_ID) or -math.inf, policy_means.get(WFS_ID) or -math.inf)

    paired: Dict[str, Dict[str, Any]] = {}
    for h in HORIZONS:
        pid = f"family_a_receding_horizon_oracle_v1_h{h}"
        series = pivot.get(pid)
        if series is None:
            continue
        entry: Dict[str, Any] = {}
        for name, other in {
            "estimated_service_time_first": estf, "weighted_fair_share": wfs,
            "best_fixed_parent_by_scenario": best_fixed_series,
            "stateful_controller_v1": pivot.get("family_a_stateful_controller_v1"),
        }.items():
            if other is None:
                continue
            diff = series - other
            entry[name] = {
                "mean_diff": finite_float(diff.mean()), "median_diff": finite_float(diff.median()),
                "wins": int((diff > 1e-12).sum()), "ties": int((diff.abs() <= 1e-12).sum()),
                "losses": int((diff < -1e-12).sum()),
            }
        for other_h in HORIZONS:
            if other_h == h:
                continue
            other_series = pivot.get(f"family_a_receding_horizon_oracle_v1_h{other_h}")
            if other_series is None:
                continue
            diff = series - other_series
            entry[f"vs_h{other_h}"] = {
                "mean_diff": finite_float(diff.mean()), "median_diff": finite_float(diff.median()),
                "wins": int((diff > 1e-12).sum()), "ties": int((diff.abs() <= 1e-12).sum()),
                "losses": int((diff < -1e-12).sum()),
            }
        paired[pid] = entry

    concentration: Dict[str, Any] = {}
    for h in HORIZONS:
        pid = f"family_a_receding_horizon_oracle_v1_h{h}"
        series = pivot.get(pid)
        if series is None:
            continue
        diff = (series - best_fixed_series).dropna()
        positive = diff[diff > 1e-12].sort_values(ascending=False)
        total_pos = float(positive.sum())
        top1_share = float(positive.iloc[0] / total_pos) if len(positive) and total_pos > 0 else None
        concentration[pid] = {
            "n_positive_scenarios": int(len(positive)),
            "total_positive_mass": total_pos if total_pos else 0.0,
            "top1_scenario_share_of_positive_mass": top1_share,
        }

    safety = {
        m: {p: finite_float(results_df.loc[results_df.policy_id == p, m].mean()) for p in policy_ids}
        for m in SAFETY_METRICS if m in results_df.columns
    }

    cost: Dict[str, Any] = {}
    for h in HORIZONS:
        pid = f"family_a_receding_horizon_oracle_v1_h{h}"
        sub = results_df[results_df.policy_id == pid]
        diags = [d for d in sub["controller_diagnostics"].tolist() if isinstance(d, dict)]
        eligible = [d.get("eligible_count", 0) for d in diags]
        planning = [d.get("planning_calls_used", 0) for d in diags]
        cap_hits = sum(1 for d in diags if d.get("planning_cap_hit"))
        wall = sub["scenario_wall_clock_s"].tolist()
        estf_wins = [d.get("estf_win_count", 0) for d in diags]
        wfs_wins = [d.get("wfs_win_count", 0) for d in diags]
        cost[pid] = {
            "eligible_count_summary": numeric_summary(eligible),
            "planning_calls_summary": numeric_summary(planning),
            "total_planning_calls": int(sum(planning)),
            "scenarios_with_cap_hit": int(cap_hits),
            "scenario_wall_clock_summary": numeric_summary(wall),
            "total_wall_clock_s": float(sum(wall)),
            "estf_choice_fraction": (
                float(sum(estf_wins) / (sum(estf_wins) + sum(wfs_wins)))
                if (sum(estf_wins) + sum(wfs_wins)) else None
            ),
            "wfs_choice_fraction": (
                float(sum(wfs_wins) / (sum(estf_wins) + sum(wfs_wins)))
                if (sum(estf_wins) + sum(wfs_wins)) else None
            ),
        }

    oracle_gap = None
    native_mean = finite_float(native_envelope.mean())
    if native_mean is not None and best_fixed_mean is not None and best_fixed_mean != -math.inf:
        oracle_gap = native_mean - best_fixed_mean
    recovered_fraction = {}
    for h in HORIZONS:
        pid = f"family_a_receding_horizon_oracle_v1_h{h}"
        controller_mean = policy_means.get(pid)
        if oracle_gap is not None and oracle_gap > 0 and controller_mean is not None and best_fixed_mean != -math.inf:
            recovered_fraction[pid] = (controller_mean - best_fixed_mean) / oracle_gap
        else:
            recovered_fraction[pid] = None

    return {
        "policy_mean_anwg": policy_means,
        "best_fixed_parent_mean_anwg": finite_float(best_fixed_mean) if best_fixed_mean != -math.inf else None,
        "native_pair_envelope_mean_anwg": native_mean,
        "oracle_gap": oracle_gap,
        "recovered_fraction": recovered_fraction,
        "paired_diffs": paired,
        "concentration": concentration,
        "safety_metric_means": safety,
        "computational_cost": cost,
    }


def classify(summary: Mapping[str, Any]) -> tuple[str, str]:
    means = summary["policy_mean_anwg"]
    best_fixed = summary["best_fixed_parent_mean_anwg"]
    if best_fixed is None:
        return "RECEDING_HORIZON_INTEGRITY_NO_GO", "REPAIR_RECEDING_HORIZON_INSTRUMENTATION"

    h1_mean = means.get("family_a_receding_horizon_oracle_v1_h1")
    short_horizon_ids = [f"family_a_receding_horizon_oracle_v1_h{h}" for h in HORIZONS if h != 1]
    best_short = max(((pid, means.get(pid)) for pid in short_horizon_ids if means.get(pid) is not None),
                      key=lambda kv: kv[1], default=(None, None))
    best_short_pid, best_short_mean = best_short
    if best_short_pid is None or h1_mean is None:
        return "RECEDING_HORIZON_INTEGRITY_NO_GO", "REPAIR_RECEDING_HORIZON_INSTRUMENTATION"

    paired = summary["paired_diffs"].get(best_short_pid, {})
    vs_h1_key = f"vs_h1"
    vs_h1 = paired.get(vs_h1_key, {})
    improves_over_h1 = (best_short_mean > h1_mean) and (vs_h1.get("wins", 0) > vs_h1.get("losses", 0))

    vs_best_fixed = paired.get("best_fixed_parent_by_scenario", {})
    beats_best_fixed = (
        best_short_mean > best_fixed
        and vs_best_fixed.get("wins", 0) > vs_best_fixed.get("losses", 0)
    )

    conc = summary["concentration"].get(best_short_pid, {})
    top1_share = conc.get("top1_scenario_share_of_positive_mass")
    not_concentrated = top1_share is None or top1_share <= 0.50

    safety_ok = True
    safety = summary["safety_metric_means"]
    for m in ("completion_fraction", "weighted_completion_fraction"):
        vals = safety.get(m, {})
        parents_min = min(
            (v for k, v in vals.items() if k in (ESTF_ID, WFS_ID) and v is not None), default=None,
        )
        controller_val = vals.get(best_short_pid)
        if parents_min is not None and controller_val is not None and controller_val < parents_min - 0.02:
            safety_ok = False
    slo_vals = safety.get("slo_violation_rate", {})
    parents_max = max((v for k, v in slo_vals.items() if k in (ESTF_ID, WFS_ID) and v is not None), default=None)
    controller_slo = slo_vals.get(best_short_pid)
    if parents_max is not None and controller_slo is not None and controller_slo > parents_max + 0.02:
        safety_ok = False

    oracle_gap = summary.get("oracle_gap")
    recovered = summary.get("recovered_fraction", {}).get(best_short_pid)

    if (
        improves_over_h1 and beats_best_fixed and not_concentrated and safety_ok
        and oracle_gap is not None and oracle_gap > 0 and recovered is not None and recovered > 0
    ):
        return "RECEDING_HORIZON_POSITIVE_SIGNAL", "INVESTIGATE_APPROXIMATE_ONLINE_FUTURE_MODEL"
    if improves_over_h1 and safety_ok:
        return "RECEDING_HORIZON_MIXED_SIGNAL", "DIAGNOSE_ROLLOUT_VALUE_LIMIT"
    return "RECEDING_HORIZON_NO_GO", "STOP_FUTURE_AWARE_FAMILY_A_CONSTRUCTION"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=0, help="run only the first N TRAIN scenarios, all horizons")
    parser.add_argument("--max-planning-calls-per-scenario", type=int, default=DEFAULT_MAX_PLANNING_CALLS_PER_SCENARIO)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = " ".join([str(Path(sys.argv[0]).as_posix()), *sys.argv[1:]])

    print(f"start_utc={utc_now()}", flush=True)
    print(f"git_head_sha={git_text(['rev-parse', 'HEAD'])}", flush=True)
    print(f"git_tree_dirty={bool(git_text(['status', '--short']))}", flush=True)
    print(f"python={sys.executable}", flush=True)

    events = load_repaired_events()
    tree_model = fit_stateful_controller_tree(events)

    table = family_a_obs.load_family_a_trainval_scenario_table()
    assert not (table["split"].str.lower() == "test").any(), "internal error: TEST row leaked"
    if len(table) != EXPECTED_FAMILY_A_TOTAL:
        print(f"FATAL: expected {EXPECTED_FAMILY_A_TOTAL} Family-A TRAIN/VAL scenarios, got {len(table)}", flush=True)
        return 1

    if args.pilot:
        table = table[table["split"] == "train"].head(args.pilot).reset_index(drop=True)
        print(f"PILOT MODE: {len(table)} TRAIN scenarios", flush=True)

    provenance = {
        "schema_version": "family_a_receding_horizon_oracle_v1.0",
        "command": command,
        "git_head": git_text(["rev-parse", "HEAD"]),
        "git_status": git_text(["status", "--short"]),
        "design_sha256": sha256_file(DESIGN_PATH),
        "repaired_events_sha256": sha256_file(REPAIRED_EVENTS_PATH),
        "horizons": list(HORIZONS),
        "common_continuation_budget": COMMON_CONTINUATION_BUDGET,
        "max_planning_calls_per_scenario": args.max_planning_calls_per_scenario,
        "python_executable": sys.executable,
        "pilot": args.pilot,
    }

    run_result = run_trainval(table, tree_model, args.max_planning_calls_per_scenario)
    results_df = run_result.pop("results_df")
    summary = summarize(results_df) if not run_result["failures"] else None
    classification, next_step = ("NEED_INTEGRITY_REPAIR", "REPAIR_RECEDING_HORIZON_INSTRUMENTATION") \
        if run_result["failures"] else classify(summary)

    final_payload = {
        "provenance": provenance,
        "run": run_result,
        "summary": summary,
        "classification": classification,
        "next_step": next_step,
    }
    suffix = f"_pilot{args.pilot}" if args.pilot else ""
    out_path = OUTPUT_DIR / f"family_a_receding_horizon_oracle_v1_results{suffix}.json"
    out_path.write_text(json.dumps(json_clean(final_payload), indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(json_clean({
        "classification": classification, "next_step": next_step,
        "failures": run_result["failures"], "wall_clock_s": run_result["wall_clock_s"],
    }), indent=2), flush=True)
    print(f"Results written to {out_path}", flush=True)
    if run_result["failures"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
