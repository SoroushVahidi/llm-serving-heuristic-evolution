#!/usr/bin/env python3
"""Apt-Serve Phase G comparative sweep: large-scale, two-stage, resumable.

Why this exists: Phase F's headroom sweep (3 regimes x 3 seeds, FIFO/EDF
only) produced exact ties in every regime -- a null result, not evidence
Apt-Serve lacks headroom (docs/PROJECT_MAP.md SS5/SS8). This script tests a
much larger regime x seed x baseline matrix (docs/PROJECT_MAP.md SS10 near
term item 2), computes Apt-Serve's leave-one-out marginal contribution to
the policy-library envelope, and separates a broad screening stage from a
targeted confirmation stage so an adaptively-selected "best niche" is never
reported as a pristine held-out result.

Resumable: every (regime, seed) work unit is a stable, hashable key. On
restart, units already present in results.jsonl are skipped. Every
completed unit is appended immediately (flush + fsync) -- a crash loses at
most the in-flight unit(s), not the whole run.

Scope note on parallelism/atomicity: work is parallelized at (regime, seed)
granularity (one process computes all policy-cells for one regime/seed
pair, then the single writer process appends the whole batch). A crash can
therefore lose an in-flight (regime, seed) unit (all its policy-cells,
sub-minute in the observed pilot), not literally one policy-cell. This is a
deliberate simplification for correctness (a single writer avoids
concurrent-append races); it is stated explicitly here rather than implied
to be finer-grained than it is.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from llmserveopt.core.types import GPUConfig  # noqa: E402
from llmserveopt.evaluation.run_policy import run_policy  # noqa: E402
from llmserveopt.policies.registry import make_policy  # noqa: E402
from llmserveopt.policies.apt_serve_faithful import (  # noqa: E402
    AptServeSchedulerPolicy,
    AptServeAdapterConfig,
    AptServeAdapterError,
    AptServeCapacityViolation,
)
from llmserveopt.simulator.hybrid_cache_manager import HybridCacheInvariantError  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.workloads.apt_serve_stress import generate_apt_serve_regime_workload  # noqa: E402
from llmserveopt.workloads.apt_serve_phase_g_regimes import REGIME_CATALOG  # noqa: E402

CRITICAL_EXCEPTION_TYPES = (AptServeCapacityViolation, HybridCacheInvariantError, AptServeAdapterError)


STRONG_BASELINE_NAMES: List[str] = [
    "fifo", "edf", "weighted_shortest_processing", "least_laxity_first",
    "estimated_service_time_first", "scorpio_style_slo_guard",
    "vllm_style_token_budget", "sarathi_style", "orca_style",
    "shortest_output_first", "slo_slack_score", "admission_control",
]

# Multipliers applied to Phase F's calibrated base latencies. "0x" is an
# explicit idealized diagnostic (never used as the primary/headline
# transition-cost setting) -- see docs/PROJECT_MAP.md SS10.
TRANSITION_COST_LABELS: List[str] = ["0x_idealized", "0.5x", "1x", "2x", "4x"]
TRANSITION_COST_MULTIPLIERS: Dict[str, float] = {
    "0x_idealized": 0.0, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0,
}
PRIMARY_TRANSITION_COST = "1x"
BASE_CACHE_SWITCH_LATENCY = 0.005
BASE_HIDDEN_RESTORE_LATENCY = 0.01

GPU_CONFIGS = [GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024)]
SERVICE_MODEL_STEP_SIZE = 0.05

# Exceptions/messages that indicate a systematic Apt-Serve correctness
# problem (not an ordinary metric edge case) -- these hard-stop the run
# rather than being logged and skipped, per docs instructions SS15.
CRITICAL_MESSAGE_MARKERS = (
    "rolled back", "capacity violation", "invariant", "hybridcacheinvarianterror",
)


def strong_baseline_policy_names() -> List[str]:
    """Fail loudly if the curated baseline set drifts from the registry."""
    from llmserveopt.policies.registry import BASELINE_NAMES
    missing = [n for n in STRONG_BASELINE_NAMES if n not in BASELINE_NAMES]
    if missing:
        raise RuntimeError(f"Strong baseline names not found in registry.BASELINE_NAMES: {missing}")
    return list(STRONG_BASELINE_NAMES)


def build_apt_policy(tc_label: str) -> AptServeSchedulerPolicy:
    mult = TRANSITION_COST_MULTIPLIERS[tc_label]
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    return AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=32,
        hidden_to_kv_memory_ratio=0.5,
        cache_switch_latency=BASE_CACHE_SWITCH_LATENCY * mult,
        hidden_restore_latency=BASE_HIDDEN_RESTORE_LATENCY * mult,
        recomputation_cost_model="hidden_restore",
    )


def unit_key(stage: str, regime_id: str, seed: int) -> str:
    return f"{stage}|{regime_id}|seed={seed}"


def metrics_to_dict(m: Any) -> Dict[str, Any]:
    def f(x: float) -> Optional[float]:
        return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

    return {
        "num_completed": m.num_completed,
        "num_dropped": m.num_dropped,
        "num_total": m.num_total,
        "completion_fraction": f(m.completion_fraction),
        "mean_latency": f(m.mean_latency),
        "p95_latency": f(m.p95_latency),
        "mean_ttft": f(m.mean_ttft),
        "p95_ttft": f(m.p95_ttft),
        "slo_violation_rate": f(m.slo_violation_rate),
        "weighted_goodput_completed_only": f(m.weighted_goodput),
        "arrival_normalized_weighted_goodput": f(m.arrival_normalized_weighted_goodput),
        "request_throughput": f(m.request_throughput),
        "token_throughput": f(m.token_throughput),
    }


def run_one_unit(regime: Dict[str, Any], seed: int, stage: str) -> Dict[str, Any]:
    """Run all policy-cells (strong baselines + Apt-Serve x transition costs)
    for one (regime, seed) pair. Executes in a worker process; must be
    self-contained (re-imports are cheap and avoid pickling live policy
    objects across the process boundary).
    """
    t0 = time.time()
    n_requests = regime["n_requests"]
    requests = generate_apt_serve_regime_workload(
        seed=seed,
        n_requests=n_requests,
        arrival_pattern=regime["arrival_pattern"],
        kv_pressure=regime["kv_pressure"],
        slo_pattern=regime["slo_pattern"],
        length_pattern=regime["length_pattern"],
        cache_use_structure=regime["cache_use_structure"],
    )
    service_model = ServiceModel(step_size=SERVICE_MODEL_STEP_SIZE)

    cells: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    critical_failure: Optional[Dict[str, Any]] = None

    for name in strong_baseline_policy_names():
        try:
            policy = make_policy(name)
            m = run_policy(
                policy=policy, requests=requests, gpu_configs=GPU_CONFIGS,
                service_model=service_model, workload_tag=regime["regime_id"], seed=seed,
            )
            cells.append({
                "policy_label": name, "policy_kind": "baseline", "transition_cost": "na",
                **metrics_to_dict(m),
            })
        except Exception as e:  # noqa: BLE001
            failures.append({
                "policy_label": name, "exception_type": type(e).__name__,
                "message": str(e), "traceback": traceback.format_exc(),
            })

    for tc_label in TRANSITION_COST_LABELS:
        try:
            policy = build_apt_policy(tc_label)
            m = run_policy(
                policy=policy, requests=requests, gpu_configs=GPU_CONFIGS,
                service_model=service_model, workload_tag=regime["regime_id"], seed=seed,
            )
            apt_stats = dict(policy.stats)
            policy.terminate()
            cells.append({
                "policy_label": "apt_serve_faithful", "policy_kind": "apt_serve",
                "transition_cost": tc_label, **metrics_to_dict(m), "apt_stats": apt_stats,
            })
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            rec = {
                "policy_label": "apt_serve_faithful", "transition_cost": tc_label,
                "exception_type": type(e).__name__, "message": str(e),
                "traceback": traceback.format_exc(),
            }
            is_critical = (
                isinstance(e, CRITICAL_EXCEPTION_TYPES)
                or any(marker in msg for marker in CRITICAL_MESSAGE_MARKERS)
            )
            if is_critical:
                critical_failure = rec
            else:
                failures.append(rec)

    return {
        "stage": stage, "regime_id": regime["regime_id"], "seed": seed,
        "regime": regime, "n_requests": n_requests, "cells": cells,
        "failures": failures, "critical_failure": critical_failure,
        "wall_time_sec": time.time() - t0,
        "completed_at": time.time(),
    }


# ----------------------------------------------------------------------
# Run-directory I/O (single-writer; workers only compute and return dicts)
# ----------------------------------------------------------------------

def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed_units(results_path: Path) -> set:
    done = set()
    if not results_path.exists():
        return done
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(unit_key(rec["stage"], rec["regime_id"], rec["seed"]))
    return done


def git_state_text() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT).decode()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
        return f"branch={branch}\nHEAD={sha}\nstatus:\n{status}"
    except Exception as e:  # noqa: BLE001
        return f"git_state unavailable: {e}"


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def select_stage2_regimes(results_path: Path, top_k: int = 8) -> List[Tuple[str, float]]:
    """Pick the strongest apparent Apt-Serve niches AND strongest
    counterexamples from stage-1 results, ranked by mean (apt_primary_anwg -
    best_baseline_anwg) per regime_id. Selection is based only on stage-1
    data (never stage-2 data feeding back into its own selection)."""
    per_regime: Dict[str, List[float]] = {}
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["stage"] != "screening":
                continue
            baselines = [c["arrival_normalized_weighted_goodput"] for c in rec["cells"]
                         if c["policy_kind"] == "baseline" and c["arrival_normalized_weighted_goodput"] is not None]
            apt_primary = [c["arrival_normalized_weighted_goodput"] for c in rec["cells"]
                           if c["policy_kind"] == "apt_serve" and c["transition_cost"] == PRIMARY_TRANSITION_COST
                           and c["arrival_normalized_weighted_goodput"] is not None]
            if not baselines or not apt_primary:
                continue
            gap = apt_primary[0] - max(baselines)
            per_regime.setdefault(rec["regime_id"], []).append(gap)

    regime_mean_gap = [(rid, float(np.mean(gaps))) for rid, gaps in per_regime.items()]
    regime_mean_gap.sort(key=lambda x: x[1])
    losses = regime_mean_gap[:top_k]
    wins = regime_mean_gap[-top_k:]
    selected = {rid: gap for rid, gap in (losses + wins)}
    return sorted(selected.items(), key=lambda x: -x[1])


def run_stage(
    stage: str,
    regimes: List[Dict[str, Any]],
    seeds: List[int],
    run_dir: Path,
    results_path: Path,
    progress_path: Path,
    workers: int,
    max_end_time: Optional[float],
) -> bool:
    """Returns True if the stage ran to completion, False if it stopped
    early due to a critical failure or the wall-clock budget."""
    completed = load_completed_units(results_path)
    all_units = [(regime, seed) for regime in regimes for seed in seeds]
    pending = [(r, s) for (r, s) in all_units if unit_key(stage, r["regime_id"], s) not in completed]

    total = len(all_units)
    done_count = total - len(pending)
    remaining = len(pending)
    start = time.time()

    def write_progress(extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "stage": stage, "total_units": total, "completed_units": done_count,
            "pending_units": remaining, "elapsed_sec": time.time() - start,
            "updated_at": time.time(),
        }
        if extra:
            payload.update(extra)
        atomic_write_json(progress_path, payload)

    write_progress()
    if not pending:
        print(f"[{stage}] all {total} units already complete (resumed).")
        return True

    print(f"[{stage}] {done_count}/{total} units already complete; {len(pending)} remaining. workers={workers}")

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one_unit, r, s, stage): (r, s) for (r, s) in pending}
        for fut in cf.as_completed(futures):
            regime, seed = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001
                result = {
                    "stage": stage, "regime_id": regime["regime_id"], "seed": seed,
                    "regime": regime, "n_requests": regime["n_requests"], "cells": [],
                    "failures": [{"policy_label": "unit", "exception_type": type(e).__name__,
                                  "message": str(e), "traceback": traceback.format_exc()}],
                    "critical_failure": None, "wall_time_sec": 0.0, "completed_at": time.time(),
                }
            append_jsonl(results_path, result)
            for fail in result.get("failures", []):
                append_jsonl(run_dir / "failures.jsonl", {**fail, "regime_id": regime["regime_id"], "seed": seed, "stage": stage})
            done_count += 1
            remaining -= 1
            write_progress({"last_unit": unit_key(stage, regime["regime_id"], seed)})

            if result.get("critical_failure"):
                append_jsonl(run_dir / "failures.jsonl", {
                    **result["critical_failure"], "regime_id": regime["regime_id"], "seed": seed,
                    "stage": stage, "SEVERITY": "CRITICAL_INVARIANT_VIOLATION",
                })
                print(f"[{stage}] CRITICAL Apt-Serve invariant failure in "
                      f"{regime['regime_id']} seed={seed}: {result['critical_failure']['message']}")
                print("Terminating run per SS15 (systematic invariant violation) rather than "
                      "continuing to produce data downstream of a possibly-corrupted mechanism.")
                atomic_write_json(run_dir / "final_summary.json", {
                    "status": "FAILED_CRITICAL_INVARIANT_VIOLATION",
                    "stage": stage, "failing_unit": unit_key(stage, regime["regime_id"], seed),
                })
                ex.shutdown(wait=False, cancel_futures=True)
                return False

            if max_end_time is not None and time.time() > max_end_time:
                print(f"[{stage}] wall-clock budget reached ({remaining} "
                      f"units remaining in this stage); stopping stage cleanly. Resume later to continue.")
                ex.shutdown(wait=False, cancel_futures=True)
                return False

    return True


def finalize(run_dir: Path, results_path: Path) -> None:
    """Derive CSV/summary artifacts from results.jsonl. Safe to re-run at
    any time (e.g. mid-run, or standalone tomorrow via --analyze-only)."""
    import csv

    records = []
    with open(results_path) as f:
        for line in f:
            records.append(json.loads(line))

    per_policy_path = run_dir / "per_policy_results.csv"
    with open(per_policy_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "stage", "regime_id", "seed", "policy_label", "policy_kind", "transition_cost",
            "kv_pressure", "slo_pattern", "length_pattern", "arrival_pattern", "cache_use_structure",
            "num_completed", "num_dropped", "num_total", "completion_fraction",
            "arrival_normalized_weighted_goodput", "weighted_goodput_completed_only",
            "slo_violation_rate", "mean_latency", "p95_latency", "mean_ttft", "p95_ttft",
            "request_throughput", "token_throughput",
        ])
        for rec in records:
            regime = rec["regime"]
            for c in rec["cells"]:
                w.writerow([
                    rec["stage"], rec["regime_id"], rec["seed"], c["policy_label"], c["policy_kind"],
                    c["transition_cost"], regime["kv_pressure"], regime["slo_pattern"],
                    regime["length_pattern"], regime["arrival_pattern"], regime["cache_use_structure"],
                    c["num_completed"], c["num_dropped"], c["num_total"], c["completion_fraction"],
                    c["arrival_normalized_weighted_goodput"], c["weighted_goodput_completed_only"],
                    c["slo_violation_rate"], c["mean_latency"], c["p95_latency"], c["mean_ttft"],
                    c["p95_ttft"], c["request_throughput"], c["token_throughput"],
                ])

    transition_path = run_dir / "transition_diagnostics.csv"
    with open(transition_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "stage", "regime_id", "seed", "transition_cost", "arrival_normalized_weighted_goodput",
            "kv_to_hidden_transitions", "hidden_to_kv_transitions", "evictions", "recomputations",
            "switch_latency_paid", "restore_latency_paid", "transitions_per_completed_request",
        ])
        for rec in records:
            for c in rec["cells"]:
                if c["policy_kind"] != "apt_serve":
                    continue
                stats = c.get("apt_stats", {}) or {}
                total_transitions = stats.get("kv_to_hidden_transitions", 0) + stats.get("hidden_to_kv_transitions", 0)
                per_completed = (total_transitions / c["num_completed"]) if c["num_completed"] else None
                w.writerow([
                    rec["stage"], rec["regime_id"], rec["seed"], c["transition_cost"],
                    c["arrival_normalized_weighted_goodput"],
                    stats.get("kv_to_hidden_transitions", 0), stats.get("hidden_to_kv_transitions", 0),
                    stats.get("evictions", 0), stats.get("recomputations", 0),
                    stats.get("switch_latency_paid", 0.0), stats.get("restore_latency_paid", 0.0),
                    per_completed,
                ])

    # Leave-one-out marginal contribution of Apt-Serve (primary transition
    # cost) to the strong-baseline-library envelope, per (regime, seed).
    mc_path = run_dir / "apt_marginal_contribution.csv"
    mc_rows = []
    with open(mc_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "regime_id", "seed", "best_baseline_anwg", "apt_primary_anwg",
                    "envelope_with_apt", "marginal_contribution", "apt_is_unique_winner"])
        for rec in records:
            baselines = [c["arrival_normalized_weighted_goodput"] for c in rec["cells"]
                         if c["policy_kind"] == "baseline" and c["arrival_normalized_weighted_goodput"] is not None]
            apt_primary = [c["arrival_normalized_weighted_goodput"] for c in rec["cells"]
                           if c["policy_kind"] == "apt_serve" and c["transition_cost"] == PRIMARY_TRANSITION_COST
                           and c["arrival_normalized_weighted_goodput"] is not None]
            if not baselines or not apt_primary:
                continue
            best_baseline = max(baselines)
            apt_val = apt_primary[0]
            envelope = max(best_baseline, apt_val)
            mc = envelope - best_baseline
            row = [rec["stage"], rec["regime_id"], rec["seed"], best_baseline, apt_val,
                   envelope, mc, apt_val > best_baseline]
            w.writerow(row)
            mc_rows.append(mc)

    per_regime_path = run_dir / "per_regime_rankings.csv"
    regime_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rec in records:
        regime_groups.setdefault((rec["stage"], rec["regime_id"]), []).append(rec)
    with open(per_regime_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "regime_id", "n_seeds", "mean_apt_gap_vs_best_baseline",
                     "win_count_eps005", "tie_count_eps005", "loss_count_eps005",
                     "win_count_eps01", "tie_count_eps01", "loss_count_eps01"])
        for (stage, rid), recs in regime_groups.items():
            gaps = []
            for rec in recs:
                baselines = [c["arrival_normalized_weighted_goodput"] for c in rec["cells"]
                             if c["policy_kind"] == "baseline" and c["arrival_normalized_weighted_goodput"] is not None]
                apt_primary = [c["arrival_normalized_weighted_goodput"] for c in rec["cells"]
                               if c["policy_kind"] == "apt_serve" and c["transition_cost"] == PRIMARY_TRANSITION_COST
                               and c["arrival_normalized_weighted_goodput"] is not None]
                if baselines and apt_primary:
                    gaps.append(apt_primary[0] - max(baselines))
            if not gaps:
                continue

            def classify(eps):
                w_ = sum(1 for g in gaps if g > eps)
                l_ = sum(1 for g in gaps if g < -eps)
                t_ = len(gaps) - w_ - l_
                return w_, t_, l_

            w5 = classify(0.005)
            w10 = classify(0.01)
            w.writerow([stage, rid, len(gaps), float(np.mean(gaps)), w5[0], w5[1], w5[2], w10[0], w10[1], w10[2]])

    summary = {
        "generated_at": time.time(),
        "total_units": len(records),
        "mean_marginal_contribution": float(np.mean(mc_rows)) if mc_rows else None,
        "median_marginal_contribution": float(np.median(mc_rows)) if mc_rows else None,
        "fraction_apt_unique_winner_eps005": (
            float(np.mean([1.0 if mc > 0.005 else 0.0 for mc in mc_rows])) if mc_rows else None
        ),
        "n_marginal_contribution_samples": len(mc_rows),
        "note": (
            "This summary is descriptive of raw collected data only. Statistical "
            "significance (bootstrap CIs, grouped-by-regime) and mechanism "
            "correlation (headroom vs KV pressure / transition counts) are "
            "intentionally deferred to the morning analysis pass over these "
            "CSVs, per docs/PROJECT_MAP.md's standing rule against asserting "
            "conclusions the collecting run itself did not verify."
        ),
    }
    atomic_write_json(run_dir / "final_summary.json", summary)
    print(f"[finalize] wrote {per_policy_path}, {transition_path}, {mc_path}, {per_regime_path}, "
          f"{run_dir / 'final_summary.json'}")


def pilot(seeds_to_try: int, workers: int) -> float:
    """Run a tiny pilot (3 regimes x seeds_to_try seeds) and return the
    measured mean wall-clock seconds per (regime, seed) unit."""
    sample_regimes = REGIME_CATALOG[:3]
    seeds = list(range(90001, 90001 + seeds_to_try))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one_unit, r, s, "pilot") for r in sample_regimes for s in seeds]
        for fut in cf.as_completed(futures):
            fut.result()
    elapsed = time.time() - t0
    n_units = len(sample_regimes) * len(seeds)
    per_unit_wall = elapsed / n_units
    # per_unit_wall already reflects `workers`-way parallelism; convert to
    # single-unit compute cost for matrix planning.
    per_unit_compute = per_unit_wall * min(workers, n_units)
    print(f"[pilot] {n_units} units in {elapsed:.1f}s wall ({workers} workers) "
          f"=> ~{per_unit_compute:.2f}s compute per unit")
    return per_unit_compute


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="Output directory (default: results/apt_serve_phase_g_overnight_<ts>)")
    ap.add_argument("--stage1-seeds", type=int, default=None, help="Explicit stage-1 seed count (overrides auto-budget)")
    ap.add_argument("--stage2-seeds", type=int, default=None, help="Explicit stage-2 additional seed count")
    ap.add_argument("--stage2-top-k", type=int, default=8, help="Regimes selected per direction (win/loss) for stage 2")
    ap.add_argument("--max-hours", type=float, default=8.5, help="Target wall-clock budget")
    ap.add_argument("--workers", type=int, default=8, help="Parallel worker processes")
    ap.add_argument("--pilot-only", action="store_true", help="Run pilot timing only and exit")
    ap.add_argument("--plan-only", action="store_true", help="Run pilot, print the computed matrix plan, and exit without launching")
    ap.add_argument("--analyze-only", action="store_true", help="Regenerate CSV/summary from an existing run-dir and exit")
    ap.add_argument("--stage2-n-requests-multiplier", type=float, default=1.5)
    args = ap.parse_args()

    if args.analyze_only:
        if not args.run_dir:
            print("--analyze-only requires --run-dir")
            return 1
        run_dir = Path(args.run_dir)
        finalize(run_dir, run_dir / "results.jsonl")
        return 0

    strong_baseline_policy_names()  # fail fast if registry drifted

    per_unit_compute = pilot(seeds_to_try=2, workers=args.workers)
    if args.pilot_only:
        return 0

    n_regimes = len(REGIME_CATALOG)
    budget_sec = args.max_hours * 3600.0
    # Reserve ~15% of the budget for stage 2 by default.
    stage1_budget = budget_sec * 0.70
    stage2_budget = budget_sec * 0.30

    if args.stage1_seeds is not None:
        stage1_seed_count = args.stage1_seeds
    else:
        # Units that fit in the wall-clock budget with `workers`-way
        # parallelism: (budget_seconds * workers) / per_unit_compute_seconds.
        stage1_seed_count = max(3, int((stage1_budget * args.workers) / (per_unit_compute * n_regimes)))
    if args.stage2_seeds is not None:
        stage2_seed_count = args.stage2_seeds
    else:
        stage2_seed_count = max(4, int(
            (stage2_budget * args.workers) / (per_unit_compute * max(1, 2 * args.stage2_top_k) * args.stage2_n_requests_multiplier)
        ))

    stage1_seeds = list(range(1001, 1001 + stage1_seed_count))
    est_stage1_hours = (n_regimes * stage1_seed_count * per_unit_compute) / (3600.0 * args.workers)
    print(f"[plan] regimes={n_regimes} stage1_seeds={stage1_seed_count} "
          f"est_stage1_hours={est_stage1_hours:.2f} "
          f"stage2_seeds_per_selected_regime={stage2_seed_count}")

    if args.plan_only:
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else (REPO_ROOT / "results" / f"apt_serve_phase_g_overnight_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    progress_path = run_dir / "progress.json"

    start_time = time.time()
    end_time = start_time + budget_sec

    if not (run_dir / "run_manifest.json").exists():
        atomic_write_json(run_dir / "run_manifest.json", {
            "started_at": start_time, "max_hours": args.max_hours, "workers": args.workers,
            "n_regimes": n_regimes, "stage1_seed_count": stage1_seed_count,
            "stage2_top_k": args.stage2_top_k, "per_unit_compute_sec_pilot": per_unit_compute,
            "strong_baseline_names": strong_baseline_policy_names(),
            "transition_cost_labels": TRANSITION_COST_LABELS,
            "primary_transition_cost": PRIMARY_TRANSITION_COST,
            "gpu_configs": [gc.__dict__ if hasattr(gc, "__dict__") else str(gc) for gc in GPU_CONFIGS],
        })
        (run_dir / "git_state.txt").write_text(git_state_text())
        atomic_write_json(run_dir / "config_snapshot.json", {
            "regime_catalog": REGIME_CATALOG,
            "transition_cost_multipliers": TRANSITION_COST_MULTIPLIERS,
            "base_cache_switch_latency": BASE_CACHE_SWITCH_LATENCY,
            "base_hidden_restore_latency": BASE_HIDDEN_RESTORE_LATENCY,
        })

    ok = run_stage("screening", REGIME_CATALOG, stage1_seeds, run_dir, results_path,
                    progress_path, args.workers, end_time)
    finalize(run_dir, results_path)
    if not ok:
        return 1

    selection = select_stage2_regimes(results_path, top_k=args.stage2_top_k)
    atomic_write_json(run_dir / "stage2_selection.json", {
        "selected_regime_gaps": selection,
        "note": "Selected using stage-1 (screening) results only.",
    })
    selected_ids = {rid for rid, _ in selection}
    stage2_regimes = []
    for r in REGIME_CATALOG:
        if r["regime_id"] in selected_ids:
            r2 = dict(r)
            r2["n_requests"] = int(round(r["n_requests"] * args.stage2_n_requests_multiplier))
            stage2_regimes.append(r2)

    stage2_seeds = list(range(2001, 2001 + stage2_seed_count))
    ok2 = run_stage("confirmation", stage2_regimes, stage2_seeds, run_dir, results_path,
                     progress_path, args.workers, end_time)
    finalize(run_dir, results_path)

    atomic_write_json(run_dir / "final_status.json", {
        "screening_complete": ok, "confirmation_complete": ok2,
        "finished_at": time.time(),
    })
    return 0 if (ok and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
