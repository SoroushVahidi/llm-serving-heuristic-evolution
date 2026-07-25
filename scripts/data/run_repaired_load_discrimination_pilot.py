#!/usr/bin/env python3
"""
Repaired bounded load-discrimination pilot.

Fixes the first pilot's global 200-window cap (which omitted Mooncake) by using
deterministic per-dataset / per-origin quotas (~50 × 5 = 250 windows).

Run-specific absolute paths are CLI arguments (--run-root, --pilot-root).
Slurm job IDs are read from the environment (SLURM_JOB_ID), never hardcoded.

Metric definitions intentionally match the first pilot:
  * primary objective: arrival_normalized_weighted_goodput (ANWG)
  * exact_tie: abs(best - second) <= 1e-12
  * near_tie: best_second_margin <= 0.01
  * saturated: all policies completion_fraction >= 0.999
    (or all ANWG are NaN)

Diagnostic limitation (retained explicitly):
  "behavioral disagreement" / tie-cause labels use *outcome signatures*
  (completion/drop counts, rounded ANWG/SLO/batch metrics), not true
  scheduler action / decision traces. Do not interpret those labels as
  verified action-level explanations.

Does not train a selector, run composition/synthesis, or submit a full sweep.
Does not authorize a full 27-policy fingerprint sweep by itself.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import GPUConfig, ObservableRequest
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.registry import (
    _POLICY_LIBRARY_V2_REGISTRY,
    _REGISTRY,
    make_policy,
    make_policy_library_v2,
)
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.real_window_construction import load_window_jsonl
from llmserveopt.workloads.repaired_discrimination_selection import (
    DATASETS,
    EXACT_TIE_EPS,
    NEAR_TIE_MARGIN,
    OUTCOME_SIGNATURE_FIELDS,
    QUOTA_BUSY,
    QUOTA_NATURAL,
    QUOTA_SCALED_PER_FACTOR,
    QUOTA_SYNTHETIC,
    SCALED_FACTORS,
    DEFAULT_SAMPLING_SEED as SAMPLING_SEED,
    SATURATION_COMPLETION,
    outcome_signature,
    select_windows_stratified,
)

# ---------------------------------------------------------------------------
# Stable evaluation constants (selection quotas live in the shared module)
# ---------------------------------------------------------------------------
EVAL_SEED = 17

POLICY_SPECS: List[Dict[str, str]] = [
    {"name": "fifo", "mechanism": "fifo_control", "registry": "main"},
    {"name": "edf", "mechanism": "edf_deadline_aware", "registry": "main"},
    {
        "name": "estimated_service_time_first",
        "mechanism": "estimated_service_processing_time",
        "registry": "main",
    },
    {
        "name": "weighted_shortest_processing",
        "mechanism": "priority_weighted_service_wsp",
        "registry": "main",
    },
    {
        "name": "scorpio_style_slo_guard",
        "mechanism": "scorpio_style_admission_slo_guard",
        "registry": "main",
    },
    {
        "name": "vllm_style_token_budget",
        "mechanism": "vllm_style_batching_kv_aware",
        "registry": "main",
    },
    {"name": "aging_priority", "mechanism": "aging_fairness", "registry": "v2"},
    {
        "name": "adaptive_chunked_prefill",
        "mechanism": "adaptive_chunked_prefill",
        "registry": "v2",
    },
]
PILOT_POLICIES = [p["name"] for p in POLICY_SPECS]

MARGIN_THRESHOLDS = (0.005, 0.01, 0.02, 0.05)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def instantiate_policy(name: str, seed: int = EVAL_SEED):
    if name in _REGISTRY:
        return make_policy(name, seed=seed)
    if name in _POLICY_LIBRARY_V2_REGISTRY:
        return make_policy_library_v2(name, seed=seed)
    raise KeyError(f"policy not in live registries: {name}")


def assert_no_actual_output_leakage(reqs: Sequence) -> None:
    for r in reqs[: min(32, len(reqs))]:
        obs = ObservableRequest.from_request(r) if hasattr(ObservableRequest, "from_request") else None
        if obs is None:
            continue
        if hasattr(obs, "actual_output_tokens"):
            raise AssertionError("actual_output_leakage: ObservableRequest exposes actual_output_tokens")


def _metrics_dict(m) -> Dict[str, Any]:
    return {
        "anwg": float(m.arrival_normalized_weighted_goodput),
        "completion_fraction": float(m.completion_fraction),
        "num_completed": int(m.num_completed),
        "num_dropped": int(m.num_dropped),
        "num_total": int(m.num_total),
        "slo_violation_rate": float(m.slo_violation_rate),
        "mean_latency": float(m.mean_latency),
        "mean_ttft": float(m.mean_ttft),
        "mean_tpot": float(m.mean_tpot),
        "mean_queuing_delay": float(m.mean_queuing_delay),
        "mean_active_batch_size": float(m.mean_active_batch_size),
        "sim_duration": float(m.sim_duration),
        "token_throughput": float(m.token_throughput),
    }


def evaluate_window_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker for ProcessPoolExecutor."""
    path = Path(payload["path"])
    policies: List[str] = payload["policies"]
    seed = int(payload["seed"])
    meta, reqs = load_window_jsonl(path)
    # Leakage guard: ObservableRequest must not expose actual outputs
    if reqs:
        obs = ObservableRequest.from_request(reqs[0])
        if hasattr(obs, "actual_output_tokens"):
            raise AssertionError("actual_output_leakage: ObservableRequest exposes actual_output_tokens")

    gpus = [
        GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=65536),
        GPUConfig(gpu_id=1, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=65536),
    ]
    sm = ServiceModel(step_size=0.01)
    results: Dict[str, Any] = {}
    behavior_sigs: Dict[str, Tuple[Any, ...]] = {}
    for pname in policies:
        policy = instantiate_policy(pname, seed=seed)
        metrics = run_policy(
            policy,
            reqs,
            gpus,
            service_model=sm,
            workload_tag=meta.get("window_id", path.stem),
            seed=seed,
            drain_steps=20_000,
        )
        md = _metrics_dict(metrics)
        results[pname] = md
        # Outcome signature (NOT a true action/decision trace).
        behavior_sigs[pname] = outcome_signature(md)

    anwgs = {p: results[p]["anwg"] for p in policies}
    ordered = sorted(
        anwgs.items(),
        key=lambda kv: (
            -(-1.0 if math.isnan(kv[1]) else kv[1]),
            kv[0],
        ),
    )
    # Prefer non-NaN: re-sort with NaN last
    ordered = sorted(
        anwgs.items(),
        key=lambda kv: (
            1 if (isinstance(kv[1], float) and math.isnan(kv[1])) else 0,
            -(kv[1] if not (isinstance(kv[1], float) and math.isnan(kv[1])) else -1e300),
            kv[0],
        ),
    )
    best, best_v = ordered[0]
    second_v = ordered[1][1] if len(ordered) > 1 else best_v
    if isinstance(best_v, float) and math.isnan(best_v):
        margin = 0.0
        exact_tie = True
        near_tie = True
    else:
        bv = float(best_v)
        sv = float(second_v) if not (isinstance(second_v, float) and math.isnan(second_v)) else bv
        margin = bv - sv
        exact_tie = abs(bv - sv) <= EXACT_TIE_EPS
        near_tie = margin <= NEAR_TIE_MARGIN
    saturated = all(results[p]["completion_fraction"] >= SATURATION_COMPLETION for p in policies) or all(
        math.isnan(results[p]["anwg"]) for p in policies
    )
    unique_behaviors = len(set(behavior_sigs.values()))
    behavioral_disagreement = unique_behaviors >= 2

    return {
        "window_meta": {
            "dataset": payload["dataset"],
            "window_id": meta.get("window_id") or payload.get("window_id"),
            "window_origin": payload.get("window_origin") or meta.get("window_origin"),
            "chronological_split": payload.get("chronological_split")
            or meta.get("chronological_split"),
            "source_family": payload.get("source_family") or meta.get("source_family"),
            "load_factor": payload.get("load_factor") or meta.get("load_factor"),
            "n_requests": len(reqs),
            "path": str(path),
            "evaluation_role": payload.get("evaluation_role"),
            "redistribution": payload.get("redistribution"),
            "duration_s": float((meta.get("fingerprint") or {}).get("duration_s") or 0.0)
            if isinstance(meta, dict)
            else 0.0,
        },
        "policy_results": results,
        "behavior_signatures": {k: list(v) for k, v in behavior_sigs.items()},
        "best_policy": best,
        "best_anwg": float(best_v) if not (isinstance(best_v, float) and math.isnan(best_v)) else float("nan"),
        "second_anwg": float(second_v)
        if not (isinstance(second_v, float) and math.isnan(second_v))
        else float("nan"),
        "best_second_margin": float(margin),
        "exact_tie": bool(exact_tie),
        "near_tie": bool(near_tie),
        "saturated": bool(saturated),
        "behavioral_disagreement": bool(behavioral_disagreement),
        "n_unique_behavior_signatures": int(unique_behaviors),
    }


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(xs, p))


def summarize_group(rows: List[Dict[str, Any]], policies: List[str]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    winners = Counter(r["best_policy"] for r in rows)
    margins = [float(r["best_second_margin"]) for r in rows]
    anwg_best = [float(r["best_anwg"]) for r in rows]
    # Oracle envelope = mean of per-window best ANWG; best fixed = best mean policy ANWG
    policy_means = {}
    for p in policies:
        vals = [float(r["policy_results"][p]["anwg"]) for r in rows]
        policy_means[p] = float(np.nanmean(vals))
    best_fixed = max(policy_means.items(), key=lambda kv: (kv[1] if not math.isnan(kv[1]) else -1e300, kv[0]))
    oracle_mean = float(np.nanmean(anwg_best))
    oracle_gain = oracle_mean - float(best_fixed[1])
    margin_pct = {
        f"pct_margin_gt_{ thr}": float(np.mean([m > thr for m in margins])) for thr in MARGIN_THRESHOLDS
    }
    return {
        "n": len(rows),
        "mean_best_anwg": float(np.nanmean(anwg_best)),
        "anwg_by_policy_mean": policy_means,
        "saturated_rate": float(np.mean([r["saturated"] for r in rows])),
        "exact_tie_rate": float(np.mean([r["exact_tie"] for r in rows])),
        "near_tie_rate": float(np.mean([r["near_tie"] for r in rows])),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "p75_margin": percentile(margins, 75),
        "p90_margin": percentile(margins, 90),
        "n_effective_winner_classes": len(winners),
        "winner_counts": dict(winners),
        "behavioral_disagreement_rate": float(np.mean([r["behavioral_disagreement"] for r in rows])),
        "oracle_envelope_mean_anwg": oracle_mean,
        "best_fixed_policy": best_fixed[0],
        "best_fixed_mean_anwg": float(best_fixed[1]),
        "oracle_gain_over_best_fixed": float(oracle_gain),
        **margin_pct,
        "pct_behavioral_disagreement": float(np.mean([r["behavioral_disagreement"] for r in rows])),
    }


def pairwise_wins(rows: List[Dict[str, Any]], policies: List[str]) -> Dict[str, Dict[str, int]]:
    mat = {a: {b: 0 for b in policies} for a in policies}
    for r in rows:
        pr = r["policy_results"]
        for i, a in enumerate(policies):
            for b in policies[i + 1 :]:
                va = pr[a]["anwg"]
                vb = pr[b]["anwg"]
                if math.isnan(va) and math.isnan(vb):
                    continue
                if math.isnan(va):
                    mat[b][a] += 1
                elif math.isnan(vb):
                    mat[a][b] += 1
                elif abs(va - vb) <= EXACT_TIE_EPS:
                    continue
                elif va > vb:
                    mat[a][b] += 1
                else:
                    mat[b][a] += 1
    return mat


def classify_tie_cause(row: Dict[str, Any]) -> str:
    """Heuristic tie-cause labels using recorded metrics (no simulator change).

    Labels that mention "actions" or "behavior" refer to *outcome signatures*
    (see OUTCOME_SIGNATURE_FIELDS), not true scheduler action traces.
    """
    pr = row["policy_results"]
    policies = list(pr.keys())
    all_complete = all(pr[p]["completion_fraction"] >= SATURATION_COMPLETION for p in policies)
    all_slo_zero = all(
        (not math.isnan(pr[p]["slo_violation_rate"]) and pr[p]["slo_violation_rate"] <= 1e-12)
        or pr[p]["completion_fraction"] < SATURATION_COMPLETION
        for p in policies
    )
    # Refine: all finished with zero SLO violations
    perfect = all(
        pr[p]["completion_fraction"] >= SATURATION_COMPLETION
        and (not math.isnan(pr[p]["slo_violation_rate"]))
        and pr[p]["slo_violation_rate"] <= 1e-12
        for p in policies
    )
    identical_actions = row["n_unique_behavior_signatures"] <= 1
    anwgs = [pr[p]["anwg"] for p in policies]
    anwg_spread = (
        float(np.nanmax(anwgs) - np.nanmin(anwgs))
        if any(not math.isnan(x) for x in anwgs)
        else 0.0
    )
    n_req = int(row["window_meta"].get("n_requests") or 0)
    dur = float(row["window_meta"].get("duration_s") or 0.0)
    drops = [pr[p]["num_dropped"] for p in policies]
    any_drop_diff = len(set(drops)) > 1

    if row["exact_tie"]:
        if perfect and identical_actions:
            return "all_complete_on_time_identical_actions"
        if perfect and not identical_actions:
            return "all_complete_on_time_anwg_insensitive_despite_behavior_diff"
        if identical_actions and not perfect:
            return "identical_actions_incomplete_or_slo_miss"
        if not identical_actions and anwg_spread <= EXACT_TIE_EPS:
            return "different_actions_anwg_insensitive"
        if n_req < 50 or (dur > 0 and dur < 60):
            return "short_window_possible"
        if any_drop_diff:
            return "admission_drop_diff_but_anwg_tied"
        return "exact_tie_other"
    if row["near_tie"]:
        if identical_actions:
            return "near_tie_identical_behavior"
        if perfect:
            return "near_tie_loose_slo_or_capacity_headroom"
        return "near_tie_small_margin"
    if identical_actions:
        return "disagree_on_anwg_but_identical_behavior_sig"
    return "discriminating_no_tie"


def evidence_role(origin: str) -> str:
    if origin in ("natural_replay", "natural_busy_period"):
        return "primary_real_evidence"
    if origin == "trace_derived_time_scaled":
        return "trace_derived_stress_evidence"
    if origin == "trace_calibrated_synthetic":
        return "supporting_synthetic_evidence"
    return "other"


def decide_readiness(summary: Dict[str, Any], by_dataset: Dict[str, Any], by_origin: Dict[str, Any], selection_counts: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    gates: Dict[str, Any] = {}
    # Mandatory
    gates["every_dataset_represented"] = all(
        selection_counts.get("by_dataset", {}).get(ds, 0) > 0 for ds in DATASETS
    )
    origins_needed = {
        "natural_replay",
        "natural_busy_period",
        "trace_derived_time_scaled",
        "trace_calibrated_synthetic",
    }
    gates["every_origin_family_represented"] = origins_needed.issubset(
        set(selection_counts.get("by_origin", {}).keys())
    )
    gates["window_validation_passed"] = True  # enforced at selection
    gates["no_actual_output_leakage"] = True  # enforced at eval (job fails otherwise)
    gates["at_least_four_winner_classes_overall"] = (
        summary.get("n_effective_winner_classes", 0) >= 4
    )
    multi_winner_datasets = sum(
        1 for _ds, s in by_dataset.items() if s.get("n_effective_winner_classes", 0) > 1
    )
    gates["at_least_three_datasets_multi_winner"] = multi_winner_datasets >= 3
    natural_rows = by_origin.get("natural_replay", {})
    busy_rows = by_origin.get("natural_busy_period", {})
    nat_diff = (
        (natural_rows.get("n_effective_winner_classes", 0) > 1)
        or (natural_rows.get("behavioral_disagreement_rate", 0) > 0)
        or (busy_rows.get("n_effective_winner_classes", 0) > 1)
        or (busy_rows.get("behavioral_disagreement_rate", 0) > 0)
        or (natural_rows.get("exact_tie_rate", 1) < 1.0)
        or (busy_rows.get("exact_tie_rate", 1) < 1.0)
    )
    gates["natural_busy_show_differentiation"] = bool(nat_diff)

    # Signal-quality targets
    gates["sat_rate_below_0_25"] = summary.get("saturated_rate", 1.0) < 0.25
    gates["exact_tie_below_0_70"] = summary.get("exact_tie_rate", 1.0) < 0.70
    gates["near_tie_below_0_80"] = summary.get("near_tie_rate", 1.0) < 0.80
    gates["mean_margin_above_0_01"] = summary.get("mean_margin", 0.0) > 0.01
    gates["pct_margin_gt_0_02_above_0_20"] = summary.get("pct_margin_gt_0.02", 0.0) >= 0.20
    gates["meaningful_behavioral_disagreement"] = (
        summary.get("behavioral_disagreement_rate", 0.0) >= 0.10
    )
    gates["oracle_above_best_fixed"] = summary.get("oracle_gain_over_best_fixed", 0.0) > 1e-6

    mandatory_keys = [
        "every_dataset_represented",
        "every_origin_family_represented",
        "window_validation_passed",
        "no_actual_output_leakage",
        "at_least_four_winner_classes_overall",
        "at_least_three_datasets_multi_winner",
        "natural_busy_show_differentiation",
    ]
    signal_keys = [
        "sat_rate_below_0_25",
        "exact_tie_below_0_70",
        "near_tie_below_0_80",
        "mean_margin_above_0_01",
        "pct_margin_gt_0_02_above_0_20",
        "meaningful_behavioral_disagreement",
        "oracle_above_best_fixed",
    ]
    mandatory_ok = all(gates[k] for k in mandatory_keys)
    signal_ok = all(gates[k] for k in signal_keys)
    gates["mandatory_pass_count"] = sum(1 for k in mandatory_keys if gates[k])
    gates["signal_pass_count"] = sum(1 for k in signal_keys if gates[k])
    gates["failed_mandatory"] = [k for k in mandatory_keys if not gates[k]]
    gates["failed_signal"] = [k for k in signal_keys if not gates[k]]

    if summary.get("n", 0) < 50 or not gates["every_dataset_represented"]:
        decision = "INVALID_WINDOWS"
    elif summary.get("saturated_rate", 1.0) >= 0.85 and summary.get("exact_tie_rate", 1.0) >= 0.85:
        decision = "STILL_SATURATED"
    elif mandatory_ok and signal_ok:
        decision = "READY_FOR_FULL_FINGERPRINT_SWEEP"
    elif mandatory_ok or gates["signal_pass_count"] >= 3:
        decision = "PARTIALLY_READY"
    elif summary.get("saturated_rate", 1.0) >= 0.5:
        decision = "STILL_SATURATED"
    else:
        decision = "PARTIALLY_READY"
    return decision, gates


def write_selected_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "window_id",
        "window_origin",
        "load_factor",
        "chronological_split",
        "source_family",
        "n_requests",
        "total_token_arrival_rate",
        "path",
        "evaluation_role",
        "redistribution",
        "evidence_role",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            fp = r.get("fingerprint") or {}
            w.writerow(
                {
                    "dataset": r.get("dataset"),
                    "window_id": r.get("window_id"),
                    "window_origin": r.get("window_origin"),
                    "load_factor": r.get("load_factor"),
                    "chronological_split": r.get("chronological_split"),
                    "source_family": r.get("source_family"),
                    "n_requests": fp.get("n_requests"),
                    "total_token_arrival_rate": fp.get("total_token_arrival_rate"),
                    "path": r.get("path"),
                    "evaluation_role": r.get("evaluation_role"),
                    "redistribution": r.get("redistribution"),
                    "evidence_role": evidence_role(r.get("window_origin", "")),
                }
            )
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--seed", type=int, default=SAMPLING_SEED)
    parser.add_argument("--workers", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    pilot_root = Path(args.pilot_root)
    for sub in ("jobs", "logs", "manifests", "results", "reports"):
        (pilot_root / sub).mkdir(parents=True, exist_ok=True)

    start = utc_now()
    t0 = time.time()
    status_path = pilot_root / "reports" / "PILOT_STATUS.txt"
    atomic_write_text(status_path, f"RUNNING\nstart_utc={start}\n")

    selected, selection_meta = select_windows_stratified(run_root, seed=args.seed)
    write_selected_csv(pilot_root / "manifests" / "selected_windows.csv", selected)
    atomic_write_json(pilot_root / "manifests" / "selected_window_counts.json", selection_meta)
    atomic_write_json(
        pilot_root / "manifests" / "policies.json",
        {
            "policies": POLICY_SPECS,
            "names": PILOT_POLICIES,
            "topology": "monolithic_two_gpu_compatible",
            "excluded": "faithful_disaggregated_and_migratory_baselines",
        },
    )
    metric_defs = {
        "primary_objective": "arrival_normalized_weighted_goodput",
        "exact_tie": f"abs(best_anwg - second_anwg) <= {EXACT_TIE_EPS}",
        "near_tie": f"best_second_margin <= {NEAR_TIE_MARGIN}",
        "saturated": f"all(completion_fraction >= {SATURATION_COMPLETION}) or all(anwg is NaN)",
        "matched_prior_pilot": True,
        "eval_seed": EVAL_SEED,
        "sampling_seed": args.seed,
        "simulator": {
            "gpus": 2,
            "max_active_sequences": 16,
            "max_batch_tokens": 2048,
            "max_kv_tokens": 65536,
            "step_size": 0.01,
            "drain_steps": 20000,
        },
    }
    atomic_write_json(
        pilot_root / "manifests" / "pilot_manifest.json",
        {
            "git_sha": args.git_sha,
            "source_run_root": str(run_root),
            "source_window_run_id": run_root.name,
            "pilot_root": str(pilot_root),
            "start_utc": start,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "policies": PILOT_POLICIES,
            "metric_definitions": metric_defs,
            "selection": selection_meta["counts"],
            "mooncake_note": {
                "evaluation_role": "internal_ood_only",
                "redistribution": "prohibited_until_license_clarified",
            },
        },
    )

    if args.select_only:
        atomic_write_text(status_path, "SELECT_ONLY_DONE\n")
        return

    payloads = []
    for w in selected:
        payloads.append(
            {
                "path": w["path"],
                "policies": PILOT_POLICIES,
                "seed": EVAL_SEED,
                "dataset": w["dataset"],
                "window_id": w["window_id"],
                "window_origin": w["window_origin"],
                "chronological_split": w.get("chronological_split"),
                "source_family": w.get("source_family"),
                "load_factor": w.get("load_factor"),
                "evaluation_role": w.get("evaluation_role"),
                "redistribution": w.get("redistribution"),
            }
        )

    rows: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for i, p in enumerate(payloads):
            rows.append(evaluate_window_payload(p))
            if (i + 1) % 10 == 0:
                print(f"progress {i+1}/{len(payloads)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(evaluate_window_payload, p): i for i, p in enumerate(payloads)}
            done = 0
            for fut in as_completed(futs):
                rows.append(fut.result())
                done += 1
                if done % 10 == 0:
                    print(f"progress {done}/{len(payloads)}", flush=True)

    # Stable order
    rows.sort(
        key=lambda r: (
            r["window_meta"]["dataset"],
            r["window_meta"]["window_origin"],
            str(r["window_meta"]["window_id"]),
        )
    )

    # Tie-cause diagnostics on deterministic sample (up to 20 windows)
    rng = random.Random(args.seed)
    diag_idx = list(range(len(rows)))
    rng.shuffle(diag_idx)
    diag_idx = sorted(diag_idx[: min(20, len(rows))])
    tie_diags = []
    for i in diag_idx:
        r = rows[i]
        tie_diags.append(
            {
                "dataset": r["window_meta"]["dataset"],
                "window_id": r["window_meta"]["window_id"],
                "window_origin": r["window_meta"]["window_origin"],
                "exact_tie": r["exact_tie"],
                "near_tie": r["near_tie"],
                "saturated": r["saturated"],
                "margin": r["best_second_margin"],
                "n_unique_behavior_signatures": r["n_unique_behavior_signatures"],
                "tie_cause_class": classify_tie_cause(r),
                "behavior_signatures": r["behavior_signatures"],
                "policy_completion_fraction": {
                    p: r["policy_results"][p]["completion_fraction"] for p in PILOT_POLICIES
                },
                "policy_slo_violation_rate": {
                    p: r["policy_results"][p]["slo_violation_rate"] for p in PILOT_POLICIES
                },
                "policy_num_dropped": {
                    p: r["policy_results"][p]["num_dropped"] for p in PILOT_POLICIES
                },
            }
        )
    # Full-row tie-cause histogram
    tie_hist = Counter(classify_tie_cause(r) for r in rows)

    summary_all = summarize_group(rows, PILOT_POLICIES)
    by_dataset = {
        ds: summarize_group([r for r in rows if r["window_meta"]["dataset"] == ds], PILOT_POLICIES)
        for ds in DATASETS
    }
    origins = sorted({r["window_meta"]["window_origin"] for r in rows})
    by_origin = {
        o: summarize_group([r for r in rows if r["window_meta"]["window_origin"] == o], PILOT_POLICIES)
        for o in origins
    }
    by_scale = {}
    for f in SCALED_FACTORS:
        key = f"{f}x"
        by_scale[key] = summarize_group(
            [
                r
                for r in rows
                if r["window_meta"]["window_origin"] == "trace_derived_time_scaled"
                and int(r["window_meta"].get("load_factor") or 1) == f
            ],
            PILOT_POLICIES,
        )
    splits = sorted({r["window_meta"].get("chronological_split") or "?" for r in rows})
    by_split = {
        s: summarize_group(
            [r for r in rows if (r["window_meta"].get("chronological_split") or "?") == s],
            PILOT_POLICIES,
        )
        for s in splits
    }
    # Evidence roles (never only aggregate)
    by_evidence = {
        role: summarize_group(
            [r for r in rows if evidence_role(r["window_meta"]["window_origin"]) == role],
            PILOT_POLICIES,
        )
        for role in (
            "primary_real_evidence",
            "trace_derived_stress_evidence",
            "supporting_synthetic_evidence",
        )
    }
    # Mooncake separate
    mooncake_summary = by_dataset.get("mooncake", {"n": 0})

    win_mat = pairwise_wins(rows, PILOT_POLICIES)
    action_disagreement = {
        "overall_rate": summary_all.get("behavioral_disagreement_rate"),
        "by_dataset": {ds: by_dataset[ds].get("behavioral_disagreement_rate") for ds in DATASETS},
        "by_origin": {o: by_origin[o].get("behavioral_disagreement_rate") for o in origins},
    }

    decision, gates = decide_readiness(
        summary_all, by_dataset, by_origin, selection_meta["counts"]
    )

    # Persist results
    atomic_write_json(
        pilot_root / "results" / "per_window_policy_results.json",
        {"n": len(rows), "rows": rows},
    )
    # CSV compact
    csv_path = pilot_root / "results" / "per_window_policy_results.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        fields = [
            "dataset",
            "window_id",
            "window_origin",
            "load_factor",
            "chronological_split",
            "source_family",
            "evidence_role",
            "best_policy",
            "best_anwg",
            "second_anwg",
            "best_second_margin",
            "exact_tie",
            "near_tie",
            "saturated",
            "behavioral_disagreement",
        ] + [f"anwg__{p}" for p in PILOT_POLICIES]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            wm = r["window_meta"]
            row = {
                "dataset": wm["dataset"],
                "window_id": wm["window_id"],
                "window_origin": wm["window_origin"],
                "load_factor": wm.get("load_factor"),
                "chronological_split": wm.get("chronological_split"),
                "source_family": wm.get("source_family"),
                "evidence_role": evidence_role(wm["window_origin"]),
                "best_policy": r["best_policy"],
                "best_anwg": r["best_anwg"],
                "second_anwg": r["second_anwg"],
                "best_second_margin": r["best_second_margin"],
                "exact_tie": r["exact_tie"],
                "near_tie": r["near_tie"],
                "saturated": r["saturated"],
                "behavioral_disagreement": r["behavioral_disagreement"],
            }
            for p in PILOT_POLICIES:
                row[f"anwg__{p}"] = r["policy_results"][p]["anwg"]
            w.writerow(row)
    tmp.replace(csv_path)

    atomic_write_json(pilot_root / "results" / "pairwise_policy_wins.json", win_mat)
    atomic_write_json(pilot_root / "results" / "action_disagreement.json", action_disagreement)
    atomic_write_json(
        pilot_root / "results" / "tie_cause_diagnostics.json",
        {"histogram": dict(tie_hist), "sample": tie_diags},
    )

    summary_obj = {
        "utc_end": utc_now(),
        "elapsed_s": time.time() - t0,
        "git_sha": args.git_sha,
        "source_run_root": str(run_root),
        "pilot_root": str(pilot_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "policies": PILOT_POLICIES,
        "metric_definitions": metric_defs,
        "selection_counts": selection_meta["counts"],
        "overall": summary_all,
        "by_dataset": by_dataset,
        "by_origin": by_origin,
        "by_scale_factor": by_scale,
        "by_chronological_split": by_split,
        "by_evidence_role": by_evidence,
        "mooncake_internal_ood": {
            "evaluation_role": "internal_ood_only",
            "redistribution": "prohibited_until_license_clarified",
            "summary": mooncake_summary,
        },
        "readiness_gates": gates,
        "decision": decision,
        "tie_cause_histogram": dict(tie_hist),
    }
    atomic_write_json(pilot_root / "reports" / "repaired_pilot_summary.json", summary_obj)

    # Markdown report
    lines = [
        "# Repaired load-discrimination pilot",
        "",
        f"- utc_end: {summary_obj['utc_end']}",
        f"- git_sha: `{args.git_sha}`",
        f"- source_run_root: `{run_root}`",
        f"- pilot_root: `{pilot_root}`",
        f"- n_windows: {summary_all.get('n')}",
        f"- policies: {', '.join(PILOT_POLICIES)}",
        f"- LOAD_DISCRIMINATION_PILOT = {decision}",
        "",
        "## Metric definitions (matched to prior pilot)",
        f"- primary: `{metric_defs['primary_objective']}`",
        f"- exact_tie: `{metric_defs['exact_tie']}`",
        f"- near_tie: `{metric_defs['near_tie']}`",
        f"- saturated: `{metric_defs['saturated']}`",
        "",
        "## Selection",
        f"- mooncake_included: {selection_meta['counts']['mooncake_included']}",
        f"- by_dataset: `{json.dumps(selection_meta['counts']['by_dataset'])}`",
        f"- by_origin: `{json.dumps(selection_meta['counts']['by_origin'])}`",
        f"- deficits: `{json.dumps(selection_meta.get('deficits', []))}`",
        "",
        "## Overall (do not use alone)",
        f"- saturated_rate: {summary_all.get('saturated_rate')}",
        f"- exact_tie_rate: {summary_all.get('exact_tie_rate')}",
        f"- near_tie_rate: {summary_all.get('near_tie_rate')}",
        f"- mean_margin: {summary_all.get('mean_margin')}",
        f"- median/p75/p90 margin: {summary_all.get('median_margin')} / {summary_all.get('p75_margin')} / {summary_all.get('p90_margin')}",
        f"- winner_classes: {summary_all.get('n_effective_winner_classes')}",
        f"- winners: `{summary_all.get('winner_counts')}`",
        f"- behavioral_disagreement_rate: {summary_all.get('behavioral_disagreement_rate')}",
        f"- best_fixed: {summary_all.get('best_fixed_policy')} ({summary_all.get('best_fixed_mean_anwg')})",
        f"- oracle_envelope: {summary_all.get('oracle_envelope_mean_anwg')} (gain {summary_all.get('oracle_gain_over_best_fixed')})",
        "",
        "## By evidence role",
    ]
    for role, s in by_evidence.items():
        lines.append(
            f"- {role}: n={s.get('n')} sat={s.get('saturated_rate')} exact_tie={s.get('exact_tie_rate')} "
            f"margin={s.get('mean_margin')} winners=`{s.get('winner_counts')}`"
        )
    lines += ["", "## By dataset"]
    for ds in DATASETS:
        s = by_dataset[ds]
        note = " (internal_ood_only; redistribution prohibited until license clarified)" if ds == "mooncake" else ""
        lines.append(
            f"- {ds}{note}: n={s.get('n')} sat={s.get('saturated_rate')} exact_tie={s.get('exact_tie_rate')} "
            f"winners=`{s.get('winner_counts')}`"
        )
    lines += ["", "## By origin"]
    for o in origins:
        s = by_origin[o]
        lines.append(
            f"- {o}: n={s.get('n')} sat={s.get('saturated_rate')} exact_tie={s.get('exact_tie_rate')} "
            f"winners=`{s.get('winner_counts')}`"
        )
    lines += ["", "## By scale factor (time-scaled only)"]
    for k, s in by_scale.items():
        lines.append(
            f"- {k}: n={s.get('n')} sat={s.get('saturated_rate')} exact_tie={s.get('exact_tie_rate')} "
            f"winners=`{s.get('winner_counts')}`"
        )
    lines += ["", "## By chronological split"]
    for sname, s in by_split.items():
        lines.append(
            f"- {sname}: n={s.get('n')} sat={s.get('saturated_rate')} winners=`{s.get('winner_counts')}`"
        )
    lines += ["", "## Tie-cause histogram", f"`{json.dumps(dict(tie_hist))}`", "", "## Readiness gates"]
    for k, v in gates.items():
        if k in ("failed_mandatory", "failed_signal"):
            continue
        lines.append(f"- {k}: {v}")
    lines += [
        f"- failed_mandatory: {gates.get('failed_mandatory')}",
        f"- failed_signal: {gates.get('failed_signal')}",
        "",
        f"LOAD_DISCRIMINATION_PILOT = {decision}",
        "",
        "This job does not submit a full fingerprint sweep.",
        "",
    ]
    atomic_write_text(pilot_root / "reports" / "REPAIRED_PILOT_REPORT.md", "\n".join(lines))
    atomic_write_text(
        status_path,
        f"COMPLETED\nexit_code=0\ndecision={decision}\nend_utc={summary_obj['utc_end']}\n",
    )
    print(f"LOAD_DISCRIMINATION_PILOT = {decision}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Best-effort status update when pilot_root known
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        raise
