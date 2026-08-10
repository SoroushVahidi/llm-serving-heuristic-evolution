#!/usr/bin/env python3
"""Policy Separation Boundary Refinement v1 -- SECOND compute experiment.

Follows job 1170116 (docs/audits/policy_separation_three_case_v1_20260810.md),
which established that (1) FCFS convoy separation is a razor-sharp
offset==0.0 phenomenon that goes structurally uninformative for ANY
positive offset under max_active_sequences=1, (2) prediction-inversion
advantage declines monotonically with inversion under strong
heterogeneity/high load and can reverse, and (3) EDF/admission_control are
almost behaviorally identical while scorpio_style_slo_guard beats EDF under
unsalvageable overload. This experiment does NOT repeat that broad
three-case sweep; it runs three narrower, finer-grained studies to locate
the actual decision boundaries those findings only bracketed:

  Study A: FCFS convoy boundary refinement -- a fine arrival-offset grid
  (in units of the simulator's step_size) around zero, to find exactly
  where the "genuine choice" between fifo and size-aware admission order
  disappears, plus an optional max_active_sequences=1-vs->1 comparison.

  Study B (primary): prediction-inversion decision-boundary refinement --
  a calibrated load grid x a fine inversion-fraction grid x heterogeneity,
  to map advantage = f(load, prediction-ranking-quality, heterogeneity) as
  an actual decision surface rather than two anchor points.

  Study C: EDF / admission-policy mechanism audit -- after inspecting the
  edf/admission_control/scorpio_style_slo_guard implementations (see
  docs/audits/policy_separation_edf_admission_mechanism_20260810.md), a
  small targeted matrix over overload_factor x fraction_impossible to
  quantify the admission_control-vs-edf near-identity and the
  scorpio-vs-edf margin under unsalvageable overload.

NOT MAP-Elites, NOT CMA-ES, NOT Bayesian optimization, NOT selector
training, NOT module synthesis -- see docs/audits/
policy_separation_three_case_v1_20260810.md's readiness section. This is
generator-calibration data for those methods to use later, not a
replacement for them.

Deterministic, CLI-configurable, resumable, multiprocessing-safe,
Slurm-safe. Writes only to --run-dir (scratch space, never the git
checkout). Never retries a task recorded in failures.jsonl automatically.

Usage:
  python scripts/run_policy_separation_boundary_refinement.py \\
      --config configs/policy_separation_boundary_refinement_v1.yaml \\
      --run-dir <RUN_DIR> --workers 8 --resume
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
from llmserveopt.policy_separation.templates_boundary_refinement import (  # noqa: E402
    generate_case1_boundary_grid, generate_case2_boundary_grid,
)
from llmserveopt.policy_separation.templates_three_case import generate_case3_grid  # noqa: E402
from llmserveopt.selector.dataset_v2.discriminativeness import (  # noqa: E402
    PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN,
)
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402

FAMILY_ORDER = ["fcfs_convoy", "prediction_inversion_boundary", "edf_unsalvageable_overload"]
STUDY_OF_FAMILY = {
    "fcfs_convoy": "A",
    "prediction_inversion_boundary": "B",
    "edf_unsalvageable_overload": "C",
}
CONFIG_KEY_OF_FAMILY = {
    "fcfs_convoy": "study_a_fcfs_boundary",
    "prediction_inversion_boundary": "study_b_prediction_inversion_boundary",
    "edf_unsalvageable_overload": "study_c_edf_admission_mechanism",
}

# Per-family candidate/baseline policy sets -- used only by hypothesis
# validation and the decision-boundary aggregations, not a restriction on
# which policies are actually run (that is config-driven).
FAMILY_CANDIDATE_POLICIES = {
    "fcfs_convoy": ["estimated_service_time_first", "weighted_shortest_processing"],
    "prediction_inversion_boundary": [
        "estimated_service_time_first", "weighted_shortest_processing", "shortest_output_first",
    ],
    "edf_unsalvageable_overload": ["scorpio_style_slo_guard", "admission_control"],
}
FAMILY_BASELINE_POLICY = {
    "fcfs_convoy": "fifo",
    "prediction_inversion_boundary": "fifo",
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
    seeds_a = list(cfg["seeds_study_a"])
    seeds_b = list(cfg["seeds_study_b"])
    seeds_c = list(cfg["seeds_study_c"])
    if smoke:
        seeds_a, seeds_b, seeds_c = seeds_a[:1], seeds_b[:1], seeds_c[:1]

    a = cfg["study_a_fcfs_boundary"]
    b = cfg["study_b_prediction_inversion_boundary"]
    c = cfg["study_c_edf_admission_mechanism"]

    if smoke:
        s_a_main = generate_case1_boundary_grid(
            a["ratios"][:1], a["short_counts"][:1], a["offsets"][:2], seeds_a,
            max_active_sequences_values=[1],
        )
        mas = a["mas_substudy"]
        s_a_sub = generate_case1_boundary_grid(
            [mas["ratio"]], [mas["short_count"]], a["offsets"][:2], seeds_a,
            max_active_sequences_values=[mas["max_active_sequences"]],
        )
        s_b = generate_case2_boundary_grid(
            b["target_utilizations"][:2], b["heterogeneity"][:1], b["inversion_fractions"][:2], seeds_b,
        )
        s_c = generate_case3_grid(c["overload_factors"][:1], c["fraction_impossible"][:2], seeds_c)
    else:
        s_a_main = generate_case1_boundary_grid(
            a["ratios"], a["short_counts"], a["offsets"], seeds_a, max_active_sequences_values=[1],
        )
        mas = a["mas_substudy"]
        s_a_sub = generate_case1_boundary_grid(
            [mas["ratio"]], [mas["short_count"]], a["offsets"], seeds_a,
            max_active_sequences_values=[mas["max_active_sequences"]],
        )
        s_b = generate_case2_boundary_grid(
            b["target_utilizations"], b["heterogeneity"], b["inversion_fractions"], seeds_b,
        )
        s_c = generate_case3_grid(c["overload_factors"], c["fraction_impossible"], seeds_c)

    all_scenarios = s_a_main + s_a_sub + s_b + s_c
    ids = [s.scenario_id for s in all_scenarios]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate scenario_id(s) generated: {sorted(dupes)[:10]}")
    return all_scenarios


def policies_for_family(cfg: Dict, family: str) -> List[str]:
    return list(cfg[CONFIG_KEY_OF_FAMILY[family]]["policies"])


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
# CSV/resume plumbing (identical conventions to
# run_policy_separation_three_case.py, so tooling built against that run's
# output shape works unchanged against this one)
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
    if f != f:  # NaN
        return None
    return f


# ---------------------------------------------------------------------------
# Statistics helpers shared by every grouped-summary output
# ---------------------------------------------------------------------------

def _stat_block(values: List[float]) -> Dict:
    """n, mean, median, sample std, 95% normal-approx CI, sign consistency
    (fraction strictly > tie margin), and wins/ties/losses vs 0 at the same
    tie margin `pairwise_rows`/`scenario_summary` already use
    (PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN), so "tie" here means the same
    thing it means everywhere else in this dataset."""
    n = len(values)
    if n == 0:
        return dict(n=0, mean=None, median=None, std=None, ci95_lo=None, ci95_hi=None,
                    sign_consistency=None, wins=0, ties=0, losses=0)
    mean = sum(values) / n
    sv = sorted(values)
    median = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    wins = sum(1 for v in values if v > PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN)
    losses = sum(1 for v in values if v < -PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN)
    ties = n - wins - losses
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
        ci_h = 1.96 * std / math.sqrt(n)
        ci_lo, ci_hi = mean - ci_h, mean + ci_h
    else:
        std, ci_lo, ci_hi = 0.0, None, None
    return dict(n=n, mean=mean, median=median, std=std, ci95_lo=ci_lo, ci95_hi=ci_hi,
                sign_consistency=wins / n, wins=wins, ties=ties, losses=losses)


def _linear_zero_crossing(xs: List[float], ys: List[float]) -> Optional[float]:
    """First x at which y crosses 0 via linear interpolation between
    consecutive (x, mean_y) grid points, xs assumed sorted ascending. None
    if there is no sign change anywhere in the grid (mechanism is
    one-sided across the whole swept range at this resolution)."""
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 is None or y1 is None:
            continue
        if y0 == 0.0:
            return xs[i]
        if (y0 < 0.0) != (y1 < 0.0):
            x0, x1 = xs[i], xs[i + 1]
            return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    return None


# ---------------------------------------------------------------------------
# Aggregation: pairwise / stress-control / hypothesis validation (identical
# logic to run_policy_separation_three_case.py -- generic over `family` via
# scenario_meta, so it is reused verbatim rather than re-derived)
# ---------------------------------------------------------------------------

def compute_pairwise_and_summaries(run_dir: Path) -> Tuple[Dict[str, List[PolicyResultRow]], Dict[str, Dict]]:
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

    return by_scenario, scenario_meta


def compute_stress_control_summary(run_dir: Path) -> List[Dict]:
    rows = _read_results(run_dir)
    anwg: Dict[Tuple[str, str], Optional[float]] = {}
    scenario_by_id: Dict[str, Dict] = {}
    for r in rows:
        anwg[(r["scenario_id"], r["policy_name"])] = _to_float(r.get("arrival_normalized_weighted_goodput"))
        scenario_by_id[r["scenario_id"]] = r

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
# Aggregation: family-level oracle / best-fixed headroom (shared helper --
# see docs/audits/policy_separation_three_case_v1_20260810.md's "family
# oracle headroom" figure, which this refines with real per-cell breakdown)
# ---------------------------------------------------------------------------

def _family_oracle_headroom(scenario_ids: List[str], by_scenario: Dict[str, List[PolicyResultRow]]) -> Dict:
    policy_sum: Dict[str, float] = {}
    policy_n: Dict[str, int] = {}
    per_scenario_vals: Dict[str, Dict[str, float]] = {}
    for sid in scenario_ids:
        vals = {r.policy_name: r.arrival_normalized_weighted_goodput for r in by_scenario.get(sid, [])
                if r.arrival_normalized_weighted_goodput is not None}
        if not vals:
            continue
        per_scenario_vals[sid] = vals
        for p, v in vals.items():
            policy_sum[p] = policy_sum.get(p, 0.0) + v
            policy_n[p] = policy_n.get(p, 0) + 1

    if not policy_sum:
        return dict(n=0, best_fixed_policy=None, mean_oracle=None, mean_best_fixed=None,
                    mean_headroom=None, fraction_positive=None, fraction_gt_0005=None,
                    fraction_gt_001=None, unique_winners=None, near_tie_rate=None)

    policy_mean = {p: policy_sum[p] / policy_n[p] for p in policy_sum}
    best_fixed_policy = max(policy_mean, key=policy_mean.get)

    headrooms = []
    winners = set()
    near_ties = 0
    for sid, vals in per_scenario_vals.items():
        oracle = max(vals.values())
        winners.add(max(vals, key=vals.get))
        ranked = sorted(vals.values(), reverse=True)
        if len(ranked) >= 2 and (ranked[0] - ranked[1]) <= PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN:
            near_ties += 1
        if best_fixed_policy in vals:
            headrooms.append(oracle - vals[best_fixed_policy])

    n = len(headrooms)
    if n == 0:
        return dict(n=0, best_fixed_policy=best_fixed_policy, mean_oracle=None, mean_best_fixed=None,
                    mean_headroom=None, fraction_positive=None, fraction_gt_0005=None,
                    fraction_gt_001=None, unique_winners=len(winners), near_tie_rate=None)

    return dict(
        n=n,
        best_fixed_policy=best_fixed_policy,
        mean_oracle=sum(max(v.values()) for v in per_scenario_vals.values()) / len(per_scenario_vals),
        mean_best_fixed=policy_mean[best_fixed_policy],
        mean_headroom=sum(headrooms) / n,
        fraction_positive=sum(1 for h in headrooms if h > 0.0) / n,
        fraction_gt_0005=sum(1 for h in headrooms if h > 0.005) / n,
        fraction_gt_001=sum(1 for h in headrooms if h > 0.01) / n,
        unique_winners=len(winners),
        near_tie_rate=near_ties / len(per_scenario_vals),
    )


# ---------------------------------------------------------------------------
# Study A: FCFS offset boundary
# ---------------------------------------------------------------------------

def compute_fcfs_offset_boundary(
    run_dir: Path,
    scenario_by_id: Dict[str, PolicySeparationScenario],
    by_scenario: Dict[str, List[PolicyResultRow]],
) -> None:
    anwg: Dict[Tuple[str, str], float] = {}
    for sid, rows in by_scenario.items():
        for r in rows:
            if r.arrival_normalized_weighted_goodput is not None:
                anwg[(sid, r.policy_name)] = r.arrival_normalized_weighted_goodput

    cells: Dict[Tuple, List[str]] = {}
    for sid, s in scenario_by_id.items():
        if s.family != "fcfs_convoy":
            continue
        p = s.params
        key = (p["ratio"], p["n_short"], p["offset"], p["max_active_sequences"], p["role"])
        cells.setdefault(key, []).append(sid)

    baseline = FAMILY_BASELINE_POLICY["fcfs_convoy"]
    candidates = FAMILY_CANDIDATE_POLICIES["fcfs_convoy"] + ["shortest_output_first"]
    out_rows = []
    for (ratio, n_short, offset, mas, role), sids in sorted(cells.items()):
        for cand in candidates:
            vals = []
            for sid in sids:
                a = anwg.get((sid, cand))
                f = anwg.get((sid, baseline))
                if a is not None and f is not None:
                    vals.append(a - f)
            stat = _stat_block(vals)
            if stat["n"] == 0:
                classification = "no_data"
            elif stat["mean"] is not None and stat["mean"] > PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN and (
                stat["sign_consistency"] is not None and stat["sign_consistency"] >= 0.7
            ):
                classification = "genuine_choice"
            elif stat["mean"] is not None and abs(stat["mean"]) <= PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN:
                classification = "structurally_uninformative"
            else:
                classification = "boundary_transition"
            out_rows.append({
                "ratio": ratio, "n_short": n_short, "offset": offset, "max_active_sequences": mas,
                "role": role, "candidate_policy": cand, "baseline_policy": baseline,
                **stat, "classification": classification,
            })

    out_path = run_dir / "fcfs_offset_boundary.csv"
    fieldnames = [
        "ratio", "n_short", "offset", "max_active_sequences", "role", "candidate_policy", "baseline_policy",
        "n", "mean", "median", "std", "ci95_lo", "ci95_hi", "sign_consistency", "wins", "ties", "losses",
        "classification",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


# ---------------------------------------------------------------------------
# Study B: prediction-inversion decision surface
# ---------------------------------------------------------------------------

def compute_prediction_inversion_surface(
    run_dir: Path,
    scenario_by_id: Dict[str, PolicySeparationScenario],
    by_scenario: Dict[str, List[PolicyResultRow]],
) -> None:
    anwg: Dict[Tuple[str, str], float] = {}
    for sid, rows in by_scenario.items():
        for r in rows:
            if r.arrival_normalized_weighted_goodput is not None:
                anwg[(sid, r.policy_name)] = r.arrival_normalized_weighted_goodput

    cells: Dict[Tuple, List[str]] = {}
    for sid, s in scenario_by_id.items():
        if s.family != "prediction_inversion_boundary":
            continue
        p = s.params
        key = (p["heterogeneity"], p["target_utilization"], p["inversion_fraction"])
        cells.setdefault(key, []).append(sid)

    family_sids = [sid for sid, s in scenario_by_id.items() if s.family == "prediction_inversion_boundary"]
    headroom = _family_oracle_headroom(family_sids, by_scenario)

    baseline = FAMILY_BASELINE_POLICY["prediction_inversion_boundary"]
    candidates = FAMILY_CANDIDATE_POLICIES["prediction_inversion_boundary"] + ["aging_priority"]
    out_rows = []
    for (het, util, inv), sids in sorted(cells.items()):
        params0 = scenario_by_id[sids[0]].params
        kendall = params0.get("rank_agreement_kendall_tau")
        spearman = params0.get("rank_agreement_spearman")
        for cand in candidates:
            vals = []
            for sid in sids:
                a = anwg.get((sid, cand))
                f = anwg.get((sid, baseline))
                if a is not None and f is not None:
                    vals.append(a - f)
            stat = _stat_block(vals)
            out_rows.append({
                "heterogeneity": het, "target_utilization": util, "inversion_fraction": inv,
                "rank_agreement_kendall_tau": kendall, "rank_agreement_spearman": spearman,
                "candidate_policy": cand, "baseline_policy": baseline,
                **stat,
                "family_best_fixed_policy": headroom["best_fixed_policy"],
                "family_mean_oracle_headroom": headroom["mean_headroom"],
            })

    out_path = run_dir / "prediction_inversion_surface.csv"
    fieldnames = [
        "heterogeneity", "target_utilization", "inversion_fraction",
        "rank_agreement_kendall_tau", "rank_agreement_spearman",
        "candidate_policy", "baseline_policy",
        "n", "mean", "median", "std", "ci95_lo", "ci95_hi", "sign_consistency", "wins", "ties", "losses",
        "family_best_fixed_policy", "family_mean_oracle_headroom",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


# ---------------------------------------------------------------------------
# Study C: EDF / admission-policy mechanism summary
# ---------------------------------------------------------------------------

def _slack_tier(overload_factor: float, bounds: Dict) -> str:
    if overload_factor <= bounds["feasible_max"]:
        return "feasible"
    if overload_factor <= bounds["borderline_max"]:
        return "borderline"
    return "tight"


def compute_edf_admission_mechanism_summary(
    run_dir: Path,
    scenario_by_id: Dict[str, PolicySeparationScenario],
    by_scenario: Dict[str, List[PolicyResultRow]],
    cfg: Dict,
) -> None:
    bounds = cfg["study_c_edf_admission_mechanism"]["slack_tier_bounds"]
    policies = cfg["study_c_edf_admission_mechanism"]["policies"]

    rows_by_sid: Dict[str, Dict[str, PolicyResultRow]] = {}
    for sid, rows in by_scenario.items():
        rows_by_sid[sid] = {r.policy_name: r for r in rows}

    cells: Dict[Tuple, List[str]] = {}
    for sid, s in scenario_by_id.items():
        if s.family != "edf_unsalvageable_overload":
            continue
        p = s.params
        key = (p["overload_factor"], p["fraction_impossible"], p["role"])
        cells.setdefault(key, []).append(sid)

    out_rows = []
    for (of, fi, role), sids in sorted(cells.items()):
        tier = _slack_tier(of, bounds)
        edf_vals = {sid: rows_by_sid.get(sid, {}).get("edf") for sid in sids}
        for policy in policies:
            anwg_vals, slo_vals, comp_vals, dropped_vals, margin_vals = [], [], [], [], []
            for sid in sids:
                r = rows_by_sid.get(sid, {}).get(policy)
                if r is None:
                    continue
                if r.arrival_normalized_weighted_goodput is not None:
                    anwg_vals.append(r.arrival_normalized_weighted_goodput)
                if r.slo_violation_rate is not None:
                    slo_vals.append(r.slo_violation_rate)
                if r.completion_fraction is not None:
                    comp_vals.append(r.completion_fraction)
                dropped_vals.append(r.num_dropped)
                edf_r = edf_vals.get(sid)
                if (edf_r is not None and edf_r.arrival_normalized_weighted_goodput is not None
                        and r.arrival_normalized_weighted_goodput is not None):
                    margin_vals.append(r.arrival_normalized_weighted_goodput - edf_r.arrival_normalized_weighted_goodput)

            anwg_stat = _stat_block(anwg_vals)
            margin_stat = _stat_block(margin_vals)
            out_rows.append({
                "overload_factor": of, "fraction_impossible": fi, "role": role, "slack_tier": tier,
                "policy": policy,
                "n": anwg_stat["n"],
                "mean_anwg": anwg_stat["mean"], "median_anwg": anwg_stat["median"], "std_anwg": anwg_stat["std"],
                "mean_slo_violation_rate": (sum(slo_vals) / len(slo_vals)) if slo_vals else None,
                "mean_completion_fraction": (sum(comp_vals) / len(comp_vals)) if comp_vals else None,
                "mean_num_dropped": (sum(dropped_vals) / len(dropped_vals)) if dropped_vals else None,
                "mean_margin_vs_edf": margin_stat["mean"],
                "margin_vs_edf_sign_consistency": margin_stat["sign_consistency"],
                "margin_vs_edf_wins": margin_stat["wins"], "margin_vs_edf_ties": margin_stat["ties"],
                "margin_vs_edf_losses": margin_stat["losses"],
            })

    out_path = run_dir / "edf_admission_mechanism_summary.csv"
    fieldnames = [
        "overload_factor", "fraction_impossible", "role", "slack_tier", "policy",
        "n", "mean_anwg", "median_anwg", "std_anwg",
        "mean_slo_violation_rate", "mean_completion_fraction", "mean_num_dropped",
        "mean_margin_vs_edf", "margin_vs_edf_sign_consistency",
        "margin_vs_edf_wins", "margin_vs_edf_ties", "margin_vs_edf_losses",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


# ---------------------------------------------------------------------------
# Cross-study decision boundary synthesis
# ---------------------------------------------------------------------------

def compute_decision_boundary_summary(
    run_dir: Path,
    scenario_by_id: Dict[str, PolicySeparationScenario],
    by_scenario: Dict[str, List[PolicyResultRow]],
    cfg: Dict,
) -> None:
    out_rows: List[Dict] = []

    def emit(study, family, metric, group_desc, value, n, detail):
        out_rows.append({
            "study": study, "family": family, "metric": metric, "group_desc": group_desc,
            "value": value, "n": n, "detail": detail,
        })

    # --- Study A: offset boundary + family headroom -----------------------
    a_sids = [sid for sid, s in scenario_by_id.items() if s.family == "fcfs_convoy"]
    headroom_a = _family_oracle_headroom(a_sids, by_scenario)
    emit("A", "fcfs_convoy", "oracle_headroom_mean", "all fcfs_convoy scenarios",
         headroom_a["mean_headroom"], headroom_a["n"],
         f"best_fixed_policy={headroom_a['best_fixed_policy']}; "
         f"fraction_positive={headroom_a['fraction_positive']}; "
         f"fraction_gt_0005={headroom_a['fraction_gt_0005']}; "
         f"fraction_gt_001={headroom_a['fraction_gt_001']}; "
         f"unique_winners={headroom_a['unique_winners']}; near_tie_rate={headroom_a['near_tie_rate']}")

    a_cfg = cfg["study_a_fcfs_boundary"]
    offsets = sorted(a_cfg["offsets"])
    anwg: Dict[Tuple[str, str], float] = {}
    for sid, rows in by_scenario.items():
        for r in rows:
            if r.arrival_normalized_weighted_goodput is not None:
                anwg[(sid, r.policy_name)] = r.arrival_normalized_weighted_goodput
    for mas, ratio, n_short, label in (
        (1, a_cfg["ratios"][0], a_cfg["short_counts"][len(a_cfg["short_counts"]) // 2], "mas1_representative_cell"),
        (a_cfg["mas_substudy"]["max_active_sequences"], a_cfg["mas_substudy"]["ratio"], a_cfg["mas_substudy"]["short_count"], "mas_substudy_cell"),
    ):
        means = []
        for off in offsets:
            sids = [sid for sid, s in scenario_by_id.items()
                    if s.family == "fcfs_convoy" and s.params.get("role") == "stress"
                    and s.params.get("max_active_sequences") == mas and s.params.get("ratio") == ratio
                    and s.params.get("n_short") == n_short and s.params.get("offset") == off]
            vals = []
            for sid in sids:
                a_v = anwg.get((sid, "estimated_service_time_first"))
                f_v = anwg.get((sid, "fifo"))
                if a_v is not None and f_v is not None:
                    vals.append(a_v - f_v)
            means.append(sum(vals) / len(vals) if vals else None)
        last_genuine = None
        for off, m in zip(offsets, means):
            if m is not None and m > PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN:
                last_genuine = off
        emit("A", "fcfs_convoy", "last_genuine_choice_offset", f"{label} (mas={mas}, ratio={ratio}, n_short={n_short})",
             last_genuine, len(offsets),
             f"offset grid={offsets}; mean(ESTF-FIFO) per offset={means}")

    # --- Study B: critical inversion threshold per (heterogeneity, load) --
    b_sids = [sid for sid, s in scenario_by_id.items() if s.family == "prediction_inversion_boundary"]
    headroom_b = _family_oracle_headroom(b_sids, by_scenario)
    emit("B", "prediction_inversion_boundary", "oracle_headroom_mean", "all prediction_inversion_boundary scenarios",
         headroom_b["mean_headroom"], headroom_b["n"],
         f"best_fixed_policy={headroom_b['best_fixed_policy']}; "
         f"fraction_positive={headroom_b['fraction_positive']}; "
         f"fraction_gt_0005={headroom_b['fraction_gt_0005']}; "
         f"fraction_gt_001={headroom_b['fraction_gt_001']}; "
         f"unique_winners={headroom_b['unique_winners']}; near_tie_rate={headroom_b['near_tie_rate']}")

    b_cfg = cfg["study_b_prediction_inversion_boundary"]
    inv_fracs = sorted(b_cfg["inversion_fractions"])
    for het in b_cfg["heterogeneity"]:
        for util in b_cfg["target_utilizations"]:
            means = []
            for inv in inv_fracs:
                sids = [sid for sid, s in scenario_by_id.items()
                        if s.family == "prediction_inversion_boundary" and s.params.get("heterogeneity") == het
                        and s.params.get("target_utilization") == util and s.params.get("inversion_fraction") == inv]
                vals = []
                for sid in sids:
                    a_v = anwg.get((sid, "estimated_service_time_first"))
                    f_v = anwg.get((sid, "fifo"))
                    if a_v is not None and f_v is not None:
                        vals.append(a_v - f_v)
                means.append(sum(vals) / len(vals) if vals else None)
            crossing = _linear_zero_crossing(inv_fracs, means)
            emit("B", "prediction_inversion_boundary", "critical_inversion_threshold",
                 f"heterogeneity={het}, target_utilization={util}",
                 crossing, len(inv_fracs),
                 f"inversion grid={inv_fracs}; mean(ESTF-FIFO) per inversion={means}")

    # --- Study C: crossover overload_factor per fraction_impossible + -----
    #     admission_control-vs-edf near-identity check
    c_sids = [sid for sid, s in scenario_by_id.items() if s.family == "edf_unsalvageable_overload"]
    headroom_c = _family_oracle_headroom(c_sids, by_scenario)
    emit("C", "edf_unsalvageable_overload", "oracle_headroom_mean", "all edf_unsalvageable_overload scenarios",
         headroom_c["mean_headroom"], headroom_c["n"],
         f"best_fixed_policy={headroom_c['best_fixed_policy']}; "
         f"fraction_positive={headroom_c['fraction_positive']}; "
         f"fraction_gt_0005={headroom_c['fraction_gt_0005']}; "
         f"fraction_gt_001={headroom_c['fraction_gt_001']}; "
         f"unique_winners={headroom_c['unique_winners']}; near_tie_rate={headroom_c['near_tie_rate']}")

    c_cfg = cfg["study_c_edf_admission_mechanism"]
    overloads = sorted(c_cfg["overload_factors"])
    for fi in c_cfg["fraction_impossible"]:
        means = []
        for of in overloads:
            sids = [sid for sid, s in scenario_by_id.items()
                    if s.family == "edf_unsalvageable_overload" and s.params.get("role") == "stress"
                    and s.params.get("overload_factor") == of and s.params.get("fraction_impossible") == fi]
            vals = []
            for sid in sids:
                s_v = anwg.get((sid, "scorpio_style_slo_guard"))
                e_v = anwg.get((sid, "edf"))
                if s_v is not None and e_v is not None:
                    vals.append(s_v - e_v)
            means.append(sum(vals) / len(vals) if vals else None)
        crossing = _linear_zero_crossing(overloads, means)
        emit("C", "edf_unsalvageable_overload", "scorpio_vs_edf_crossover_overload_factor",
             f"fraction_impossible={fi}", crossing, len(overloads),
             f"overload grid={overloads}; mean(SCORPIO-EDF) per overload={means}")

    ac_edf_margins = []
    for sid in c_sids:
        if scenario_by_id[sid].params.get("role") != "stress":
            continue
        a_v = anwg.get((sid, "admission_control"))
        e_v = anwg.get((sid, "edf"))
        if a_v is not None and e_v is not None:
            ac_edf_margins.append(a_v - e_v)
    ac_stat = _stat_block(ac_edf_margins)
    emit("C", "edf_unsalvageable_overload", "admission_control_vs_edf_mean_abs_margin",
         "all stress cells", (abs(ac_stat["mean"]) if ac_stat["mean"] is not None else None), ac_stat["n"],
         f"admission_control default laxity_threshold=inf performs no rejection filtering (see "
         f"docs/audits/policy_separation_edf_admission_mechanism_20260810.md) -- mean signed margin="
         f"{ac_stat['mean']}, sign_consistency={ac_stat['sign_consistency']}")

    out_path = run_dir / "decision_boundary_summary.csv"
    fieldnames = ["study", "family", "metric", "group_desc", "value", "n", "detail"]
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
    parser.add_argument("--smoke", action="store_true", help="tiny run: 1-2 cells/seeds per study, for validation only")
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
    scenario_by_id: Dict[str, PolicySeparationScenario] = {s.scenario_id: s for s in scenarios}
    _log(run_dir, f"built {len(scenarios)} scenarios across {len(FAMILY_ORDER)} families (3 studies)")

    with open(run_dir / "scenarios.jsonl", "w") as f:
        for s in scenarios:
            manifest = s.to_manifest_dict()
            manifest["study"] = STUDY_OF_FAMILY[s.family]
            manifest["policy_roster"] = policies_for_family(cfg, s.family)
            f.write(json.dumps(manifest) + "\n")

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
        "studies": STUDY_OF_FAMILY,
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

    by_scenario, _ = compute_pairwise_and_summaries(run_dir)
    stress_control_rows = compute_stress_control_summary(run_dir)
    compute_hypothesis_validation(run_dir, stress_control_rows)
    compute_fcfs_offset_boundary(run_dir, scenario_by_id, by_scenario)
    compute_prediction_inversion_surface(run_dir, scenario_by_id, by_scenario)
    compute_edf_admission_mechanism_summary(run_dir, scenario_by_id, by_scenario, cfg)
    compute_decision_boundary_summary(run_dir, scenario_by_id, by_scenario, cfg)
    _log(run_dir, "aggregation done: pairwise_separation.csv, policy_equivalence.csv, "
                   "stress_control_summary.csv, hypothesis_validation.csv, fcfs_offset_boundary.csv, "
                   "prediction_inversion_surface.csv, edf_admission_mechanism_summary.csv, "
                   "decision_boundary_summary.csv written")

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
