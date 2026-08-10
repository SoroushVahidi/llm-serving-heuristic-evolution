#!/usr/bin/env python3
"""Policy Separation Sobol Pilot v1 -- landscape-characterization runner.

DESIGN/VALIDATION ONLY as of this script's authorship: implements the
scenario generation, evaluation plumbing, and output schema for the first
Sobol/space-filling stage of the Policy Separation roadmap (built strictly
from dimensions validated by jobs 1170116 and 1171116 -- see
docs/design/POLICY_SEPARATION_SOBOL_PILOT_V1.md). The scientific sweep at
full config scale has NOT been run; only a tiny --dry-run smoke (a handful
of Sobol points, 1 seed) has been used to validate this script end-to-end.

Two independent Sobol subspaces (Family B: prediction-sensitive scheduling;
Family C: deadline/admission scheduling) plus one small non-Sobol
categorical FCFS add-on -- see src/llmserveopt/policy_separation/sobol_pilot.py
for why these are not merged into one hybrid space.

Deterministic, CLI-configurable, resumable, multiprocessing-safe,
Slurm-safe. Writes only to --run-dir (scratch space, never the git
checkout). Never retries a task recorded in failures.jsonl automatically.

Usage:
  python scripts/run_policy_separation_sobol_pilot_v1.py \\
      --config configs/policy_separation_sobol_pilot_v1.yaml \\
      --run-dir <RUN_DIR> --workers 8 --resume

  # Tiny local smoke, no Slurm, few points only:
  python scripts/run_policy_separation_sobol_pilot_v1.py \\
      --config configs/policy_separation_sobol_pilot_v1.yaml \\
      --run-dir <RUN_DIR> --workers 2 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.evaluation.run_policy import run_policy  # noqa: E402
from llmserveopt.policies.registry import make_policy_library_v2  # noqa: E402
from llmserveopt.policy_separation.metrics_three_case import (  # noqa: E402
    PolicyResultRow, pairwise_rows,
)
from llmserveopt.policy_separation.schema import PolicySeparationScenario  # noqa: E402
from llmserveopt.policy_separation.sobol_pilot import (  # noqa: E402
    FAMILY_B_RANGES, FAMILY_C_RANGES, generate_family_b_sobol_scenarios,
    generate_family_c_sobol_scenarios, generate_fcfs_categorical_add_on,
    validate_scenario,
)
from llmserveopt.selector.dataset_v2.discriminativeness import (  # noqa: E402
    PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN,
)
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402

GENERATOR_FAMILY_ORDER = [
    "sobol_family_b_prediction_sensitive",
    "sobol_family_c_deadline_admission",
    "fcfs_categorical_add_on",
]

PROGRESS_FLUSH_EVERY = 25


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def _run_git(args: List[str], cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"<git call failed: {exc}>"


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def build_all_scenarios(cfg: Dict, dry_run: bool) -> List[PolicySeparationScenario]:
    sobol_cfg = cfg["sobol"]
    b_cfg = cfg["family_b_prediction_sensitive"]
    c_cfg = cfg["family_c_deadline_admission"]
    a_cfg = cfg["fcfs_categorical_add_on"]

    if dry_run:
        # A handful of Sobol points, 1 seed, both heterogeneity levels for B
        # (so the categorical crossing logic is still exercised) -- plumbing
        # validation only, never scientific interpretation.
        s_b = generate_family_b_sobol_scenarios(
            m=2, scramble_seed=sobol_cfg["family_b_scramble_seed"],
            heterogeneity_levels=b_cfg["heterogeneity"], seeds=b_cfg["seeds"][:1],
        )
        s_c = generate_family_c_sobol_scenarios(
            m=2, scramble_seed=sobol_cfg["family_c_scramble_seed"], seeds=c_cfg["seeds"][:1],
        )
        s_a = generate_fcfs_categorical_add_on(
            a_cfg["a1_ratios"][:1], a_cfg["a1_short_counts"][:1], a_cfg["a1_seeds"][:1],
            a_cfg["a2_ratio"], a_cfg["a2_short_count"], a_cfg["a2_offsets"][:1],
            a_cfg["a2_max_active_sequences"], a_cfg["a2_seeds"][:1],
        )
    else:
        s_b = generate_family_b_sobol_scenarios(
            m=sobol_cfg["family_b_m"], scramble_seed=sobol_cfg["family_b_scramble_seed"],
            heterogeneity_levels=b_cfg["heterogeneity"], seeds=b_cfg["seeds"],
        )
        s_c = generate_family_c_sobol_scenarios(
            m=sobol_cfg["family_c_m"], scramble_seed=sobol_cfg["family_c_scramble_seed"], seeds=c_cfg["seeds"],
        )
        s_a = generate_fcfs_categorical_add_on(
            a_cfg["a1_ratios"], a_cfg["a1_short_counts"], a_cfg["a1_seeds"],
            a_cfg["a2_ratio"], a_cfg["a2_short_count"], a_cfg["a2_offsets"],
            a_cfg["a2_max_active_sequences"], a_cfg["a2_seeds"],
        )

    all_scenarios = s_b + s_c + s_a
    ids = [s.scenario_id for s in all_scenarios]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate scenario_id(s) generated: {sorted(dupes)[:10]}")
    return all_scenarios


def _family_of(scenario: PolicySeparationScenario) -> str:
    fam = scenario.params.get("generator_family")
    return fam if fam is not None else "fcfs_categorical_add_on"


def policies_for_scenario(cfg: Dict, scenario: PolicySeparationScenario) -> List[str]:
    fam = _family_of(scenario)
    key = {
        "sobol_family_b_prediction_sensitive": "family_b_prediction_sensitive",
        "sobol_family_c_deadline_admission": "family_c_deadline_admission",
        "fcfs_categorical_add_on": "fcfs_categorical_add_on",
    }[fam]
    return list(cfg[key]["policies"])


def build_sobol_design_manifest(cfg: Dict) -> Dict:
    sobol_cfg = cfg["sobol"]
    return {
        "family_b": {
            "scramble_seed": sobol_cfg["family_b_scramble_seed"],
            "m": sobol_cfg["family_b_m"],
            "n_points": 2 ** sobol_cfg["family_b_m"],
            "dimensions": ["target_utilization", "inversion_fraction"],
            "ranges": FAMILY_B_RANGES,
            "categorical": {"heterogeneity": cfg["family_b_prediction_sensitive"]["heterogeneity"]},
            "seeds": cfg["family_b_prediction_sensitive"]["seeds"],
        },
        "family_c": {
            "scramble_seed": sobol_cfg["family_c_scramble_seed"],
            "m": sobol_cfg["family_c_m"],
            "n_points": 2 ** sobol_cfg["family_c_m"],
            "dimensions": ["overload_factor", "fraction_impossible"],
            "ranges": FAMILY_C_RANGES,
            "categorical": {},
            "seeds": cfg["family_c_deadline_admission"]["seeds"],
        },
        "fcfs_categorical_add_on": {
            "sobol": False,
            "reason": "arrival_offset proven discontinuous (job 1171116); fixed small grid instead",
            "a1": {k: cfg["fcfs_categorical_add_on"][k] for k in ("a1_ratios", "a1_short_counts", "a1_seeds")},
            "a2": {k: cfg["fcfs_categorical_add_on"][k] for k in
                   ("a2_ratio", "a2_short_count", "a2_offsets", "a2_max_active_sequences", "a2_seeds")},
        },
    }


# ---------------------------------------------------------------------------
# Worker (must be top-level/picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_one_task(args: Tuple[str, str, PolicySeparationScenario, int]) -> Dict:
    scenario_id, policy_name, scenario, drain_steps = args
    try:
        policy = make_policy_library_v2(policy_name)
        service_model = ServiceModel()
        metrics = run_policy(
            policy=policy,
            requests=list(scenario.requests),
            gpu_configs=list(scenario.gpu_configs),
            service_model=service_model,
            workload_tag=scenario_id,
            seed=scenario.seed,
            drain_steps=drain_steps,
        )
        row = asdict(metrics)
        row["scenario_id"] = scenario_id
        row["generator_family"] = _family_of(scenario)
        row["template_name"] = scenario.template_name
        row["seed"] = scenario.seed
        for nan_field in (
            "arrival_normalized_weighted_goodput", "weighted_goodput", "completion_fraction",
            "slo_violation_rate", "mean_latency", "mean_ttft", "mean_tpot",
        ):
            v = row.get(nan_field)
            if v is not None and isinstance(v, float) and (v != v):
                row[nan_field] = None
        return {"status": "ok", "scenario_id": scenario_id, "policy_name": policy_name, "row": row}
    except Exception as exc:
        return {
            "status": "error", "scenario_id": scenario_id, "policy_name": policy_name,
            "error": str(exc), "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# CSV/resume plumbing
# ---------------------------------------------------------------------------

RESULT_FIELDNAMES = [
    "scenario_id", "policy_name", "generator_family", "template_name", "seed",
    "workload_tag",
    "num_completed", "num_dropped", "num_slo_violated", "num_total", "completion_fraction",
    "mean_latency", "median_latency", "p95_latency", "p99_latency", "max_latency",
    "mean_queuing_delay", "p95_queuing_delay",
    "mean_ttft", "p95_ttft", "p99_ttft", "mean_tpot", "p95_tpot",
    "mean_prefill_delay", "p95_prefill_delay",
    "slo_violation_rate", "weighted_goodput", "arrival_normalized_weighted_goodput",
    "weighted_completion_fraction", "request_throughput",
]


def _load_existing(run_dir: Path) -> Tuple[set, set]:
    done = set()
    results_path = run_dir / "per_policy_results.csv"
    if results_path.exists():
        with open(results_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["scenario_id"], row["policy_name"]))
    failed = set()
    failures_path = run_dir / "failures.jsonl"
    if failures_path.exists():
        with open(failures_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                failed.add((d["scenario_id"], d["policy_name"]))
    return done, failed


def _append_result(run_dir: Path, policy_name: str, row: Dict, write_header: bool) -> None:
    path = run_dir / "per_policy_results.csv"
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        out = dict(row)
        out["policy_name"] = policy_name
        w.writerow(out)


def _append_failure(run_dir: Path, entry: Dict) -> None:
    with open(run_dir / "failures.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def _write_progress(run_dir: Path, total: int, completed: int, failed: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    done = completed + failed
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = total - done
    eta_s = remaining / rate if rate > 0 else None
    payload = {
        "total_tasks": total, "completed": completed, "failed": failed,
        "remaining": remaining, "elapsed_s": round(elapsed, 1),
        "tasks_per_s": round(rate, 3), "eta_s": round(eta_s, 1) if eta_s is not None else None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = run_dir / "progress.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(run_dir / "progress.json")


def _read_results(run_dir: Path) -> List[Dict]:
    path = run_dir / "per_policy_results.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stat_block(values: List[float]) -> Dict:
    n = len(values)
    if n == 0:
        return dict(n=0, mean=None, median=None, std=None, sign_consistency=None)
    mean = sum(values) / n
    sv = sorted(values)
    median = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    wins = sum(1 for v in values if v > PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return dict(n=n, mean=mean, median=median, std=std, sign_consistency=wins / n)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_pairwise_separation(run_dir: Path) -> Dict[str, List[PolicyResultRow]]:
    rows = _read_results(run_dir)
    by_scenario: Dict[str, List[PolicyResultRow]] = {}
    for r in rows:
        prr = PolicyResultRow(
            scenario_id=r["scenario_id"], policy_name=r["policy_name"],
            arrival_normalized_weighted_goodput=_to_float(r.get("arrival_normalized_weighted_goodput")),
            weighted_goodput=_to_float(r.get("weighted_goodput")),
            completion_fraction=_to_float(r.get("completion_fraction")),
            slo_violation_rate=_to_float(r.get("slo_violation_rate")),
            mean_latency=_to_float(r.get("mean_latency")),
            mean_ttft=_to_float(r.get("mean_ttft")),
            mean_tpot=_to_float(r.get("mean_tpot")),
            num_completed=int(float(r.get("num_completed") or 0)),
            num_dropped=int(float(r.get("num_dropped") or 0)),
            num_total=int(float(r.get("num_total") or 0)),
        )
        by_scenario.setdefault(r["scenario_id"], []).append(prr)

    out_path = run_dir / "pairwise_separation.csv"
    fieldnames = [
        "scenario_id", "policy_i", "policy_j", "anwg_i", "anwg_j",
        "signed_advantage_i_minus_j", "abs_separation", "latency_log_ratio", "practically_equivalent",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for scenario_id, policy_rows in sorted(by_scenario.items()):
            for pr in pairwise_rows(scenario_id, policy_rows):
                w.writerow(pr)
    return by_scenario


def compute_policy_winner_summary(
    run_dir: Path, scenario_by_id: Dict[str, PolicySeparationScenario],
    by_scenario: Dict[str, List[PolicyResultRow]],
) -> None:
    out_rows = []
    for family in GENERATOR_FAMILY_ORDER:
        sids = [sid for sid, s in scenario_by_id.items() if _family_of(s) == family]
        winners = []
        for sid in sids:
            vals = {r.policy_name: r.arrival_normalized_weighted_goodput for r in by_scenario.get(sid, [])
                    if r.arrival_normalized_weighted_goodput is not None}
            if vals:
                winners.append(max(vals, key=vals.get))
        counts = Counter(winners)
        total = sum(counts.values())
        entropy = 0.0
        for c in counts.values():
            p = c / total if total else 0
            if p > 0:
                entropy -= p * math.log2(p)
        for policy, count in sorted(counts.items()):
            out_rows.append({
                "generator_family": family, "policy_name": policy, "win_count": count,
                "win_fraction": count / total if total else None,
                "n_scenarios": total, "winner_entropy_bits": round(entropy, 4),
            })
        if not counts:
            out_rows.append({
                "generator_family": family, "policy_name": None, "win_count": 0,
                "win_fraction": None, "n_scenarios": 0, "winner_entropy_bits": None,
            })

    out_path = run_dir / "policy_winner_summary.csv"
    fieldnames = ["generator_family", "policy_name", "win_count", "win_fraction", "n_scenarios", "winner_entropy_bits"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


def compute_oracle_headroom(
    run_dir: Path, scenario_by_id: Dict[str, PolicySeparationScenario],
    by_scenario: Dict[str, List[PolicyResultRow]],
) -> None:
    out_rows = []
    for family in GENERATOR_FAMILY_ORDER:
        sids = [sid for sid, s in scenario_by_id.items() if _family_of(s) == family]
        policy_sum: Dict[str, float] = {}
        policy_n: Dict[str, int] = {}
        per_scenario_vals: Dict[str, Dict[str, float]] = {}
        for sid in sids:
            vals = {r.policy_name: r.arrival_normalized_weighted_goodput for r in by_scenario.get(sid, [])
                    if r.arrival_normalized_weighted_goodput is not None}
            if not vals:
                continue
            per_scenario_vals[sid] = vals
            for p, v in vals.items():
                policy_sum[p] = policy_sum.get(p, 0.0) + v
                policy_n[p] = policy_n.get(p, 0) + 1

        if not policy_sum:
            out_rows.append({"generator_family": family, "n": 0, "best_fixed_policy": None,
                              "mean_headroom": None, "fraction_positive": None,
                              "fraction_gt_0005": None, "fraction_gt_001": None,
                              "unique_winners": 0, "near_tie_rate": None})
            continue

        policy_mean = {p: policy_sum[p] / policy_n[p] for p in policy_sum}
        best_fixed = max(policy_mean, key=policy_mean.get)
        headrooms, winners, near_ties = [], set(), 0
        for sid, vals in per_scenario_vals.items():
            oracle = max(vals.values())
            winners.add(max(vals, key=vals.get))
            ranked = sorted(vals.values(), reverse=True)
            if len(ranked) >= 2 and (ranked[0] - ranked[1]) <= PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN:
                near_ties += 1
            if best_fixed in vals:
                headrooms.append(oracle - vals[best_fixed])
        n = len(headrooms)
        out_rows.append({
            "generator_family": family, "n": n, "best_fixed_policy": best_fixed,
            "mean_headroom": sum(headrooms) / n if n else None,
            "fraction_positive": sum(1 for h in headrooms if h > 0) / n if n else None,
            "fraction_gt_0005": sum(1 for h in headrooms if h > 0.005) / n if n else None,
            "fraction_gt_001": sum(1 for h in headrooms if h > 0.01) / n if n else None,
            "unique_winners": len(winners),
            "near_tie_rate": near_ties / len(per_scenario_vals) if per_scenario_vals else None,
        })

    out_path = run_dir / "oracle_headroom.csv"
    fieldnames = ["generator_family", "n", "best_fixed_policy", "mean_headroom",
                  "fraction_positive", "fraction_gt_0005", "fraction_gt_001",
                  "unique_winners", "near_tie_rate"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


def compute_family_summary(
    run_dir: Path, scenario_by_id: Dict[str, PolicySeparationScenario],
    by_scenario: Dict[str, List[PolicyResultRow]],
) -> None:
    out_rows = []
    for family in GENERATOR_FAMILY_ORDER:
        sids = [sid for sid, s in scenario_by_id.items() if _family_of(s) == family]
        by_policy: Dict[str, List[float]] = {}
        for sid in sids:
            for r in by_scenario.get(sid, []):
                if r.arrival_normalized_weighted_goodput is not None:
                    by_policy.setdefault(r.policy_name, []).append(r.arrival_normalized_weighted_goodput)
        for policy, vals in sorted(by_policy.items()):
            stat = _stat_block(vals)
            out_rows.append({"generator_family": family, "policy_name": policy, **stat})
        if not by_policy:
            out_rows.append({"generator_family": family, "policy_name": None,
                              "n": 0, "mean": None, "median": None, "std": None, "sign_consistency": None})

    out_path = run_dir / "family_summary.csv"
    fieldnames = ["generator_family", "policy_name", "n", "mean", "median", "std", "sign_consistency"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


def compute_coverage_summary(run_dir: Path, scenario_by_id: Dict[str, PolicySeparationScenario], n_bins: int = 10) -> None:
    """Bins each Sobol family's continuous coordinates into an n_bins x
    n_bins grid and reports how many of the n_bins**2 cells received at
    least one point -- a simple, descriptive space-filling coverage
    check, not a statistical test."""
    out_rows = []
    for family, dims in (
        ("sobol_family_b_prediction_sensitive", ("target_utilization", "inversion_fraction")),
        ("sobol_family_c_deadline_admission", ("overload_factor", "fraction_impossible")),
    ):
        ranges = FAMILY_B_RANGES if family.endswith("prediction_sensitive") else FAMILY_C_RANGES
        d0, d1 = dims
        lo0, hi0 = ranges[d0]
        lo1, hi1 = ranges[d1]
        cells = set()
        n_points = 0
        for sid, s in scenario_by_id.items():
            if _family_of(s) != family:
                continue
            n_points += 1
            v0, v1 = s.params[d0], s.params[d1]
            b0 = min(n_bins - 1, int((v0 - lo0) / (hi0 - lo0) * n_bins)) if hi0 > lo0 else 0
            b1 = min(n_bins - 1, int((v1 - lo1) / (hi1 - lo1) * n_bins)) if hi1 > lo1 else 0
            cells.add((b0, b1))
        out_rows.append({
            "generator_family": family, "dim_0": d0, "dim_1": d1, "n_bins_per_dim": n_bins,
            "n_points": n_points, "n_cells_covered": len(cells), "n_cells_total": n_bins * n_bins,
            "coverage_fraction": len(cells) / (n_bins * n_bins),
        })

    out_path = run_dir / "coverage_summary.csv"
    fieldnames = ["generator_family", "dim_0", "dim_1", "n_bins_per_dim", "n_points",
                  "n_cells_covered", "n_cells_total", "coverage_fraction"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


def write_scenario_features(run_dir: Path, scenarios: List[PolicySeparationScenario]) -> None:
    all_keys = set()
    for s in scenarios:
        all_keys.update(s.params.keys())
    base_fields = ["scenario_id", "family", "template_name", "generator_version", "seed"]
    fieldnames = base_fields + sorted(all_keys - set(base_fields))
    out_path = run_dir / "scenario_features.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for s in scenarios:
            row = {"scenario_id": s.scenario_id, "family": s.family, "template_name": s.template_name,
                   "generator_version": s.generator_version, "seed": s.seed, **s.params}
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                         help="tiny plumbing-validation run: a few Sobol points, 1 seed -- NOT scientific")
    parser.add_argument("--drain-steps", type=int, default=20_000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    if not args.resume:
        for name in ("run.log", "per_policy_results.csv", "failures.jsonl", "progress.json"):
            p = run_dir / name
            if p.exists():
                raise RuntimeError(
                    f"{p} already exists and --resume was not passed -- refusing to silently "
                    "overwrite a prior run. Pass --resume to continue it, or use a new --run-dir."
                )

    (run_dir / "config_snapshot.yaml").write_text(config_path.read_text())
    git_state = (
        f"head={_run_git(['rev-parse', 'HEAD'], ROOT)}\n"
        f"branch={_run_git(['rev-parse', '--abbrev-ref', 'HEAD'], ROOT)}\n"
        f"status_short=\n{_run_git(['status', '--short'], ROOT)}\n"
    )
    (run_dir / "git_state.txt").write_text(git_state)
    (run_dir / "sobol_design.json").write_text(json.dumps(build_sobol_design_manifest(cfg), indent=2))

    _log(run_dir, f"starting run_dir={run_dir} dry_run={args.dry_run} resume={args.resume} workers={args.workers}")

    scenarios = build_all_scenarios(cfg, dry_run=args.dry_run)
    scenario_by_id: Dict[str, PolicySeparationScenario] = {s.scenario_id: s for s in scenarios}
    _log(run_dir, f"built {len(scenarios)} scenarios across {len(GENERATOR_FAMILY_ORDER)} generator families")

    validity_problems = 0
    with open(run_dir / "scenarios.jsonl", "w") as f:
        for s in scenarios:
            problems = validate_scenario(s)
            if problems:
                validity_problems += 1
                _log(run_dir, f"VALIDITY WARNING scenario_id={s.scenario_id}: {problems}")
            manifest = s.to_manifest_dict()
            manifest["generator_family"] = _family_of(s)
            manifest["policy_roster"] = policies_for_scenario(cfg, s)
            manifest["validity_problems"] = problems
            f.write(json.dumps(manifest) + "\n")
    _log(run_dir, f"validity check: {validity_problems} scenario(s) with problems out of {len(scenarios)}")
    write_scenario_features(run_dir, scenarios)

    tasks: List[Tuple[str, str, PolicySeparationScenario, int]] = []
    for s in scenarios:
        for policy_name in policies_for_scenario(cfg, s):
            tasks.append((s.scenario_id, policy_name, s, args.drain_steps))

    done_keys, failed_keys = _load_existing(run_dir) if args.resume else (set(), set())
    remaining_tasks = [t for t in tasks if (t[0], t[1]) not in done_keys and (t[0], t[1]) not in failed_keys]
    _log(
        run_dir,
        f"{len(tasks)} total (scenario, policy) tasks; {len(done_keys)} already done, "
        f"{len(failed_keys)} previously failed (not retried), {len(remaining_tasks)} to run",
    )

    manifest = {
        "config_path": str(config_path), "run_dir": str(run_dir),
        "git_head": _run_git(["rev-parse", "HEAD"], ROOT),
        "git_branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], ROOT),
        "python_version": sys.version, "dry_run": args.dry_run, "workers": args.workers,
        "n_scenarios": len(scenarios), "n_tasks": len(tasks),
        "n_tasks_remaining_at_start": len(remaining_tasks),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator_families": GENERATOR_FAMILY_ORDER,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    write_header = not (run_dir / "per_policy_results.csv").exists()
    start_time = time.time()
    completed, failed = len(done_keys), len(failed_keys)
    total = len(tasks)
    _write_progress(run_dir, total, completed, failed, start_time)

    if remaining_tasks:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(_run_one_task, t) for t in remaining_tasks]
            since_flush = 0
            for fut in as_completed(futures):
                result = fut.result()
                if result["status"] == "ok":
                    _append_result(run_dir, result["policy_name"], result["row"], write_header)
                    write_header = False
                    completed += 1
                else:
                    _append_failure(run_dir, {
                        "scenario_id": result["scenario_id"], "policy_name": result["policy_name"],
                        "error": result["error"], "traceback": result["traceback"],
                        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                    failed += 1
                since_flush += 1
                if since_flush >= PROGRESS_FLUSH_EVERY:
                    _write_progress(run_dir, total, completed, failed, start_time)
                    since_flush = 0
        _write_progress(run_dir, total, completed, failed, start_time)
    else:
        _log(run_dir, "nothing to run (all tasks already done or previously failed)")

    _log(run_dir, f"execution phase done: completed={completed} failed={failed} total={total}")

    by_scenario = compute_pairwise_separation(run_dir)
    compute_policy_winner_summary(run_dir, scenario_by_id, by_scenario)
    compute_oracle_headroom(run_dir, scenario_by_id, by_scenario)
    compute_family_summary(run_dir, scenario_by_id, by_scenario)
    compute_coverage_summary(run_dir, scenario_by_id)
    _log(run_dir, "aggregation done: pairwise_separation.csv, policy_winner_summary.csv, "
                   "oracle_headroom.csv, family_summary.csv, coverage_summary.csv written")

    final_summary = {
        "run_dir": str(run_dir), "n_scenarios": len(scenarios), "n_tasks": total,
        "n_completed": completed, "n_failed": failed, "n_validity_warnings": validity_problems,
        "generator_families": GENERATOR_FAMILY_ORDER, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": manifest["git_head"], "dry_run": args.dry_run,
        "scientific_result": False if args.dry_run else True,
    }
    (run_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2))
    _log(run_dir, f"DONE: {json.dumps(final_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
