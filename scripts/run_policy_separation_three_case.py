#!/usr/bin/env python3
"""Policy Separation Dataset v1 -- first compute experiment.

Runs exactly three theory-grounded scenario families (see
docs/design/POLICY_SEPARATION_DATASET_V1.md for the broader v1 design this
extends, and this script's own docstring-level scope note below) to answer
one narrow question: can these three synthetic-workload mechanisms produce
reproducible, interpretable separation among scheduling policies in the
current simulator?

  CASE 1: FCFS convoy / head-of-line blocking
  CASE 2: SJF / estimated-size prediction inversion
  CASE 3: EDF unsalvageable overload

This is deliberately NOT the full 5-family / 25-template Policy Separation
Dataset v1 corpus. No Sobol search, no MAP-Elites, no selector training.

Deterministic, CLI-configurable, resumable, multiprocessing-safe, Slurm-safe.
Writes only to --run-dir (intended to be scratch space, never the git
checkout). Never retries a task recorded in failures.jsonl automatically --
a scientifically invalid cell should be diagnosed, not silently re-run.

Usage:
  python scripts/run_policy_separation_three_case.py \\
      --config configs/policy_separation_three_case_v1.yaml \\
      --run-dir <RUN_DIR> --workers 8 --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
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
    PolicyResultRow, pairwise_rows, scenario_summary,
)
from llmserveopt.policy_separation.schema import PolicySeparationScenario  # noqa: E402
from llmserveopt.policy_separation.templates_three_case import (  # noqa: E402
    generate_case1_grid, generate_case2_grid, generate_case3_grid,
)
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402

FAMILY_ORDER = ["fcfs_convoy", "sjf_prediction_inversion", "edf_unsalvageable_overload"]

# Per-family candidate ("expected to win under stress") / baseline
# ("expected to be comparatively insensitive") policy sets used only by the
# hypothesis-validation aggregation -- NOT a restriction on which policies
# are actually evaluated (that is config-driven, see CASE_POLICIES keys in
# the YAML config).
FAMILY_CANDIDATE_POLICIES = {
    "fcfs_convoy": ["estimated_service_time_first", "weighted_shortest_processing"],
    "sjf_prediction_inversion": ["estimated_service_time_first", "weighted_shortest_processing", "shortest_output_first"],
    "edf_unsalvageable_overload": ["scorpio_style_slo_guard", "admission_control"],
}
FAMILY_BASELINE_POLICY = {
    "fcfs_convoy": "fifo",
    "sjf_prediction_inversion": "fifo",
    "edf_unsalvageable_overload": "edf",
}

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

def build_all_scenarios(cfg: Dict, smoke: bool) -> List[PolicySeparationScenario]:
    seeds = list(cfg["seeds"])
    if smoke:
        seeds = seeds[:2]

    c1 = cfg["case1_fcfs_convoy"]
    c2 = cfg["case2_sjf_inversion"]
    c3 = cfg["case3_edf_overload"]

    if smoke:
        s1 = generate_case1_grid(c1["ratios"][:1], c1["short_counts"][:1], c1["offsets"][:1], seeds[:1])
        s2 = generate_case2_grid(c2["inversion_fractions"][:2], c2["heterogeneity"][:1], c2["load"][:1], seeds[:1])
        s3 = generate_case3_grid(c3["overload_factors"][:1], c3["fraction_impossible"][:1], seeds[:1])
    else:
        s1 = generate_case1_grid(c1["ratios"], c1["short_counts"], c1["offsets"], seeds)
        s2 = generate_case2_grid(c2["inversion_fractions"], c2["heterogeneity"], c2["load"], seeds)
        s3 = generate_case3_grid(c3["overload_factors"], c3["fraction_impossible"], seeds)

    all_scenarios = s1 + s2 + s3
    ids = [s.scenario_id for s in all_scenarios]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate scenario_id(s) generated: {sorted(dupes)[:10]}")
    return all_scenarios


def policies_for_family(cfg: Dict, family: str) -> List[str]:
    key = {
        "fcfs_convoy": "case1_fcfs_convoy",
        "sjf_prediction_inversion": "case2_sjf_inversion",
        "edf_unsalvageable_overload": "case3_edf_overload",
    }[family]
    return list(cfg[key]["policies"])


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
        row["family"] = scenario.family
        row["template_name"] = scenario.template_name
        row["pair_id"] = scenario.pair_id or ""
        row["stress_control_relationship"] = scenario.stress_control_relationship or ""
        row["seed"] = scenario.seed
        for nan_field in (
            "arrival_normalized_weighted_goodput", "weighted_goodput", "completion_fraction",
            "slo_violation_rate", "mean_latency", "mean_ttft", "mean_tpot",
        ):
            v = row.get(nan_field)
            if v is not None and isinstance(v, float) and (v != v):  # NaN check w/o importing math in worker path
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
    "scenario_id", "policy_name", "family", "template_name", "pair_id",
    "stress_control_relationship", "seed",
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
    """Returns (done_keys, failed_keys) -- (scenario_id, policy_name) pairs
    already recorded, so resume never repeats work and never silently
    retries a recorded failure."""
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


# ---------------------------------------------------------------------------
# Aggregation (pairwise / stress-control / hypothesis validation)
# ---------------------------------------------------------------------------

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
    if f != f:  # NaN
        return None
    return f


def compute_pairwise_and_summaries(run_dir: Path) -> None:
    rows = _read_results(run_dir)
    by_scenario: Dict[str, List[PolicyResultRow]] = {}
    scenario_meta: Dict[str, Dict] = {}
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
        scenario_meta[r["scenario_id"]] = {
            "family": r.get("family", ""), "pair_id": r.get("pair_id", ""),
            "stress_control_relationship": r.get("stress_control_relationship", ""),
            "seed": r.get("seed", ""),
        }

    pairwise_out = run_dir / "pairwise_separation.csv"
    equiv_out = run_dir / "policy_equivalence.csv"
    pairwise_fieldnames = [
        "scenario_id", "policy_i", "policy_j", "anwg_i", "anwg_j",
        "signed_advantage_i_minus_j", "abs_separation", "latency_log_ratio", "practically_equivalent",
    ]
    equiv_fieldnames = [
        "scenario_id", "family", "pair_id", "stress_control_relationship", "seed",
        "n_valid_policies", "winner_policy", "unique_winner", "tie_set",
        "top_two_margin", "ranking", "dispersion_std", "dispersion_mad", "classification",
    ]
    with open(pairwise_out, "w", newline="") as fp, open(equiv_out, "w", newline="") as fe:
        wp = csv.DictWriter(fp, fieldnames=pairwise_fieldnames)
        wp.writeheader()
        we = csv.DictWriter(fe, fieldnames=equiv_fieldnames)
        we.writeheader()
        for scenario_id, policy_rows in sorted(by_scenario.items()):
            for pr in pairwise_rows(scenario_id, policy_rows):
                wp.writerow(pr)
            summary = scenario_summary(scenario_id, policy_rows)
            meta = scenario_meta[scenario_id]
            we.writerow({**meta, **summary})


def compute_stress_control_summary(run_dir: Path, cfg: Dict) -> None:
    rows = _read_results(run_dir)
    # anwg[(scenario_id, policy_name)] = value
    anwg: Dict[Tuple[str, str], Optional[float]] = {}
    scenario_by_id: Dict[str, Dict] = {}
    for r in rows:
        anwg[(r["scenario_id"], r["policy_name"])] = _to_float(r.get("arrival_normalized_weighted_goodput"))
        scenario_by_id[r["scenario_id"]] = r

    # group scenario_ids by (family, pair_id, seed, role)
    by_pair_seed_role: Dict[Tuple[str, str, str, str], List[str]] = {}
    for sid, meta in scenario_by_id.items():
        key = (meta["family"], meta["pair_id"], meta["seed"], meta["stress_control_relationship"])
        by_pair_seed_role.setdefault(key, []).append(sid)

    out_rows = []
    for family in FAMILY_ORDER:
        baseline_policy = FAMILY_BASELINE_POLICY[family]
        candidate_policies = FAMILY_CANDIDATE_POLICIES[family]
        pair_ids = sorted({meta["pair_id"] for meta in scenario_by_id.values() if meta["family"] == family})
        for pair_id in pair_ids:
            seeds = sorted({meta["seed"] for meta in scenario_by_id.values()
                             if meta["family"] == family and meta["pair_id"] == pair_id})
            for seed in seeds:
                stress_ids = by_pair_seed_role.get((family, pair_id, seed, "stress"), [])
                control_ids = by_pair_seed_role.get((family, pair_id, seed, "control"), [])
                for stress_sid in stress_ids:
                    # case2 pairs one stress scenario_id per (pair_id, seed) against
                    # the single control scenario_id sharing that pair_id/seed;
                    # case1/case3 have exactly one stress and one control per
                    # (pair_id, seed) as well, so this loop naturally covers both.
                    for control_sid in control_ids or [None]:
                        best_candidate = None
                        best_candidate_val_stress = None
                        for cand in candidate_policies:
                            v = anwg.get((stress_sid, cand))
                            if v is not None and (best_candidate_val_stress is None or v > best_candidate_val_stress):
                                best_candidate, best_candidate_val_stress = cand, v
                        if best_candidate is None:
                            continue
                        baseline_val_stress = anwg.get((stress_sid, baseline_policy))
                        if baseline_val_stress is None or best_candidate_val_stress is None:
                            continue
                        advantage_stress = best_candidate_val_stress - baseline_val_stress

                        advantage_control = None
                        if control_sid is not None:
                            cand_val_control = anwg.get((control_sid, best_candidate))
                            base_val_control = anwg.get((control_sid, baseline_policy))
                            if cand_val_control is not None and base_val_control is not None:
                                advantage_control = cand_val_control - base_val_control

                        out_rows.append({
                            "family": family, "pair_id": pair_id, "seed": seed,
                            "stress_scenario_id": stress_sid, "control_scenario_id": control_sid or "",
                            "candidate_policy": best_candidate, "baseline_policy": baseline_policy,
                            "advantage_stress": advantage_stress, "advantage_control": advantage_control,
                            "margin_change": (
                                advantage_stress - advantage_control if advantage_control is not None else None
                            ),
                        })

    out_path = run_dir / "stress_control_summary.csv"
    fieldnames = [
        "family", "pair_id", "seed", "stress_scenario_id", "control_scenario_id",
        "candidate_policy", "baseline_policy", "advantage_stress", "advantage_control", "margin_change",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def compute_hypothesis_validation(run_dir: Path, stress_control_rows: List[Dict]) -> None:
    out_rows = []
    for family in FAMILY_ORDER:
        family_rows = [r for r in stress_control_rows if r["family"] == family]
        if not family_rows:
            out_rows.append({
                "family": family, "n_comparisons": 0, "classification": "UNSUPPORTED_BY_SIMULATOR",
                "mean_advantage_stress": None, "median_advantage_stress": None,
                "mean_advantage_control": None, "mean_margin_change": None,
                "fraction_direction_confirmed": None,
                "diagnosis": "no valid (scenario, policy) results available for this family",
            })
            continue

        adv_stress = [r["advantage_stress"] for r in family_rows if r["advantage_stress"] is not None]
        adv_control = [r["advantage_control"] for r in family_rows if r["advantage_control"] is not None]
        margin_change = [r["margin_change"] for r in family_rows if r["margin_change"] is not None]
        confirmed = sum(1 for r in family_rows if r["margin_change"] is not None and r["margin_change"] > 0)
        n_with_control = len(margin_change)
        fraction_confirmed = (confirmed / n_with_control) if n_with_control else None

        mean_stress = sum(adv_stress) / len(adv_stress) if adv_stress else None
        sorted_stress = sorted(adv_stress)
        median_stress = (
            sorted_stress[len(sorted_stress) // 2] if sorted_stress and len(sorted_stress) % 2
            else (
                (sorted_stress[len(sorted_stress) // 2 - 1] + sorted_stress[len(sorted_stress) // 2]) / 2
                if len(sorted_stress) >= 2 else (sorted_stress[0] if sorted_stress else None)
            )
        )
        mean_control = sum(adv_control) / len(adv_control) if adv_control else None
        mean_margin_change = sum(margin_change) / len(margin_change) if margin_change else None

        diagnosis = ""
        if fraction_confirmed is None:
            classification = "UNSUPPORTED_BY_SIMULATOR"
            diagnosis = "no paired stress/control comparisons with valid ANWG on both sides"
        elif mean_stress is not None and mean_stress <= 0:
            classification = "CONTRADICTED"
            diagnosis = "candidate policy does not beat baseline even under stress on average"
        elif fraction_confirmed >= 0.7 and (mean_margin_change or 0) > 0:
            classification = "CONFIRMED"
        elif fraction_confirmed >= 0.4:
            classification = "PARTIALLY_CONFIRMED"
        elif abs(mean_margin_change or 0) < 0.005:
            classification = "AMBIGUOUS"
            diagnosis = "stress and control margins are statistically indistinguishable at this scale"
        else:
            classification = "CONTRADICTED"
            diagnosis = "stress-control margin change is negative or opposite the expected direction"

        out_rows.append({
            "family": family, "n_comparisons": len(family_rows), "classification": classification,
            "mean_advantage_stress": mean_stress, "median_advantage_stress": median_stress,
            "mean_advantage_control": mean_control, "mean_margin_change": mean_margin_change,
            "fraction_direction_confirmed": fraction_confirmed,
            "diagnosis": diagnosis,
        })

    out_path = run_dir / "hypothesis_validation.csv"
    fieldnames = [
        "family", "n_comparisons", "classification", "mean_advantage_stress", "median_advantage_stress",
        "mean_advantage_control", "mean_margin_change", "fraction_direction_confirmed", "diagnosis",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny run: 1-2 cells/seeds per family, for validation only")
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

    _log(run_dir, f"starting run_dir={run_dir} smoke={args.smoke} resume={args.resume} workers={args.workers}")

    scenarios = build_all_scenarios(cfg, smoke=args.smoke)
    _log(run_dir, f"built {len(scenarios)} scenarios across {len(FAMILY_ORDER)} families")

    with open(run_dir / "scenarios.jsonl", "w") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_manifest_dict()) + "\n")

    tasks: List[Tuple[str, str, PolicySeparationScenario, int]] = []
    for s in scenarios:
        for policy_name in policies_for_family(cfg, s.family):
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
        "python_version": sys.version, "smoke": args.smoke, "workers": args.workers,
        "n_scenarios": len(scenarios), "n_tasks": len(tasks),
        "n_tasks_remaining_at_start": len(remaining_tasks),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "families": FAMILY_ORDER,
        "case_policies": {family: policies_for_family(cfg, family) for family in FAMILY_ORDER},
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

    compute_pairwise_and_summaries(run_dir)
    stress_control_rows = compute_stress_control_summary(run_dir, cfg)
    compute_hypothesis_validation(run_dir, stress_control_rows)
    _log(run_dir, "aggregation done: pairwise_separation.csv, policy_equivalence.csv, "
                   "stress_control_summary.csv, hypothesis_validation.csv written")

    final_summary = {
        "run_dir": str(run_dir), "n_scenarios": len(scenarios), "n_tasks": total,
        "n_completed": completed, "n_failed": failed,
        "families": FAMILY_ORDER, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": manifest["git_head"], "smoke": args.smoke,
    }
    (run_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2))
    _log(run_dir, f"DONE: {json.dumps(final_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
