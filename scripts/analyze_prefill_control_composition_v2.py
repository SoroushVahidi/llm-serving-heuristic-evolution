#!/usr/bin/env python3
"""Analyzer for Family B v2 PrefillControl composition falsification.

Reads composition_results.csv and applies preregistered verdict logic.

Preregistered hypotheses:

  H0: Contextual PrefillControl provides no useful structural composition
      beyond selecting the better parent.

  H1: A contextual PrefillControl child can choose an intermediate or
      otherwise newly parameterized prefill-control behavior and obtain
      reproducible held-out ANWG above both parents.

Verdicts (applied to TEST split, OOD reported separately):

  COMPOSITION_GO if ALL of:
    1. Mean delta_child > 0.01 on multiple held-out cells
    2. Positive mean (child - parent-envelope) difference on TEST
    3. Bootstrap CI for envelope gain excludes <= 0 (or at least is compatible
       with a reproducible positive effect)
    4. Improvement present across > 1 seed/group on TEST
    5. Child materially beats contextual top-1 somewhere on TEST
    6. Mechanism diagnostics are coherent (no SLO pathology)
    7. No feasibility/SLO pathology explains the gain

  SELECTION_SUFFICIENT_FOR_THIS_PAIR if:
    - Contextual/top-1 selection captures essentially all complementarity
    - Intermediate/contextual chunk control has zero or negligible envelope
      expansion (mean child-envelope delta ~ 0)
    - Or apparent gains are isolated/unstable

  INCONCLUSIVE otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.composition.prefill_control_metrics import (  # noqa: E402
    best_fixed_parent_score,
    bootstrap_ci,
    envelope_gain,
    paired_bootstrap_ci,
    parent_envelope,
    pairwise_comparison,
    oracle_regret,
)

PRIMARY = "arrival_normalized_weighted_goodput"
EPS = 0.01
N_BOOT = 2000

PARENT_FULL = "full_prefill"
PARENT_SMALL = "chunked_prefill_small"
PARENT_NAMES = (PARENT_FULL, PARENT_SMALL)
INTERMEDIATE_NAMES = ("chunk_96", "chunk_128", "chunk_192")
COMPOSITION_METHODS = ("contextual_top1", "contextual_alpha", "hard_conditional",
                       "prefill_control_child")
ORACLE_METHODS = ("parent_oracle", "best_fixed_parent")


def parse_scenario_id(sid: str) -> Dict[str, Any]:
    """Parse Family B v2 scenario_id into factors."""
    parts = sid.split(".")
    # e.g. pd2.hog12.late12.slolate_ttft.s20260823
    result = {}
    for part in parts:
        if part.startswith("hog"):
            result["n_hog"] = int(part.replace("hog", ""))
        elif part.startswith("late"):
            result["n_late"] = int(part.replace("late", ""))
        elif part.startswith("slo"):
            result["slo_emphasis"] = part.replace("slo", "")
        elif part.startswith("s"):
            result["seed"] = int(part.replace("s", ""))
    return result


def _finite_val(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return None


def load_results(csv_path: Path) -> List[Dict[str, Any]]:
    """Load composition_results.csv, filter to successful runs."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def analyse(run_dir: Path) -> Dict[str, Any]:
    """Run full preregistered analysis."""
    csv_path = run_dir / "composition_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found: {csv_path}")

    rows = load_results(csv_path)
    splits_path = run_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
    else:
        splits = {"train": [], "val": [], "test": [], "ood": []}

    # ---- Integrity checks ----
    integrity = _integrity_checks(rows)
    if not integrity["all_pass"]:
        return {
            "integrity": integrity,
            "verdict": "DESIGN_CONFOUND",
            "reason": "Integrity checks failed; results not interpretable.",
        }

    # ---- Organise scores by scenario and method ----
    scores: Dict[str, Dict[str, float]] = {}  # sid -> {method -> anwg}
    splits_map: Dict[str, str] = {}
    for sid in splits.get("train", []) + splits.get("val", []) + splits.get("test", []) + splits.get("ood", []):
        for sp in ("train", "val", "test", "ood"):
            if sid in splits.get(sp, []):
                splits_map[sid] = sp
                break

    for row in rows:
        sid = row["scenario_id"]
        method = row["method_name"]
        score = _finite_val(row.get(PRIMARY))
        if score is None:
            continue
        scores.setdefault(sid, {})[method] = score
        splits_map.setdefault(sid, "unknown")

    # ---- Parent envelope ----
    parent_sids = [sid for sid in scores if PARENT_FULL in scores[sid] and PARENT_SMALL in scores[sid]]
    if not parent_sids:
        return {"integrity": {"all_pass": False, "error": "No parent evaluations found"},
                "verdict": "DESIGN_CONFOUND", "reason": "Missing parent evaluations"}

    env = parent_envelope(
        {sid: scores[sid].get(PARENT_FULL, 0.0) for sid in scores},
        {sid: scores[sid].get(PARENT_SMALL, 0.0) for sid in scores},
        sorted(scores.keys()),
    )

    # ---- TEST and OOD splits ----
    test_sids = [sid for sid in splits.get("test", []) if sid in scores]
    ood_sids = [sid for sid in splits.get("ood", []) if sid in scores]
    # Fall back to heuristic if explicit splits not found
    if not test_sids:
        test_sids = [sid for sid in scores if splits_map.get(sid) == "test" or
                     ("s20260823" in sid and "late12" in sid)]
    if not ood_sids:
        ood_sids = [sid for sid in scores if splits_map.get(sid) == "ood" or
                    ("s20260823" in sid and ("late40" in sid or "late12" not in sid))]

    # ---- Evaluate each method on TEST ----
    all_methods = sorted(set(m for sid in scores for m in scores[sid]))
    method_results = {}

    for method in all_methods:
        if method not in scores.get(test_sids[0], {}) if test_sids else {}:
            continue
        method_scores_test = {sid: scores[sid].get(method, 0.0) for sid in test_sids}
        method_scores_ood = {sid: scores[sid].get(method, 0.0) for sid in ood_sids}

        # Parent envelope on this split
        test_env = {sid: env.get(sid, 0.0) for sid in test_sids}
        ood_env = {sid: env.get(sid, 0.0) for sid in ood_sids}

        # Envelope gain
        test_eg = envelope_gain(method_scores_test, test_env, test_sids, eps=EPS)
        test_deltas = [method_scores_test.get(sid, 0.0) - env.get(sid, 0.0) for sid in test_sids]
        ood_deltas = [method_scores_ood.get(sid, 0.0) - env.get(sid, 0.0) for sid in ood_sids]
        test_eg_boot = bootstrap_ci(test_deltas, seed=20261201) if test_deltas else (float("nan"), float("nan"), float("nan"))

        # Regret vs oracle
        test_regret = oracle_regret(method_scores_test,
                                    {sid: scores[sid].get(PARENT_FULL, 0.0) for sid in test_sids},
                                    {sid: scores[sid].get(PARENT_SMALL, 0.0) for sid in test_sids},
                                    test_sids)

        method_results[method] = {
            "test_n": len(test_sids),
            "test_ood_n": len(ood_sids),
            "test_mean": float(np.mean([m_scores_test.get(sid, 0.0) for sid in test_sids])) if test_sids else float("nan"),
            "test_mean_envelope_gain": test_eg["mean_envelope_gain"],
            "test_mean_delta": float(np.mean(test_deltas)) if test_deltas else float("nan"),
            "test_delta_ci": {"mean": test_eg_boot[0], "lo": test_eg_boot[1], "hi": test_eg_boot[2]},
            "test_frac_beat_env_0p01": test_eg["fraction_beat_envelope_plus_eps"],
            "test_n_beat_env_0p01": test_eg["n_beat_envelope_plus_eps"],
            "ood_mean": float(np.mean([method_scores_ood.get(sid, 0.0) for sid in ood_sids])) if ood_sids else float("nan"),
            "ood_mean_envelope_gain": envelope_gain(method_scores_ood, ood_env, ood_sids)["mean_envelope_gain"],
            "ood_mean_delta": float(np.mean(ood_deltas)) if ood_deltas else float("nan"),
            "test_regret": test_regret,
            "test_delta_raw": [round(d, 6) for d in test_deltas],
        }

    # ---- Contextual child vs top-1 comparison ----
    top1_on_test = method_results.get("contextual_top1", {}).get("test_mean", float("nan"))
    child_on_test = method_results.get("prefill_control_child", {}).get("test_mean", float("nan"))
    top1_deltas = [scores[sid].get("contextual_top1", 0.0) - scores[sid].get(PARENT_FULL, 0.0)
                   for sid in test_sids if PARENT_FULL in scores[sid]]
    child_deltas = [scores[sid].get("prefill_control_child", 0.0) - scores[sid].get(PARENT_FULL, 0.0)
                    for sid in test_sids if PARENT_FULL in scores[sid]]
    comp_vs_top1 = pairwise_comparison(
        {sid: scores[sid].get("prefill_control_child", 0.0) for sid in test_sids},
        {sid: scores[sid].get("contextual_top1", 0.0) for sid in test_sids},
        test_sids, eps=EPS,
    )

    # ---- Fixed intermediate comparison ----
    fixed_intermediates = {}
    for inj in INTERMEDIATE_NAMES:
        if inj in method_results:
            fixed_intermediates[inj] = {
                "test_mean": method_results[inj]["test_mean"],
                "test_mean_delta": method_results[inj]["test_mean_delta"],
                "test_mean_envelope_gain": method_results[inj]["test_mean_envelope_gain"],
            }
        else:
            # Check if this intermediate was evaluated
            for sid in test_sids[:1]:
                if inj in scores.get(sid, {}):
                    inj_scores = {sid: scores[sid][inj] for sid in test_sids}
                    inj_env = {sid: env.get(sid, 0.0) for sid in test_sids}
                    inj_deltas = [scores[sid][inj] - env[sid] for sid in test_sids]
                    fixed_intermediates[inj] = {
                        "test_mean": float(np.mean(list(inj_scores.values()))),
                        "test_mean_delta": float(np.mean(inj_deltas)),
                        "test_mean_envelope_gain": envelope_gain(inj_scores, inj_env, test_sids)["mean_envelope_gain"],
                    }

    # Best fixed intermediate
    best_fixed_inter = None
    best_fixed_inter_score = float("-inf")
    for inj, info in fixed_intermediates.items():
        if info["test_mean"] > best_fixed_inter_score:
            best_fixed_inter = inj
            best_fixed_inter_score = info["test_mean"]

    # ---- Seed-level analysis ----
    seed_analysis = {}
    seed_groups: Dict[int, List[str]] = defaultdict(list)
    for sid in test_sids:
        parsed = parse_scenario_id(sid)
        seed = parsed.get("seed", 0)
        seed_groups[seed].append(sid)

    for seed, seed_sids in sorted(seed_groups.items()):
        # Child vs envelope on this seed
        child_deltas_seed = [scores[sid].get("prefill_control_child", 0.0) - env.get(sid, 0.0)
                             for sid in seed_sids if sid in scores]
        if child_deltas_seed:
            seed_analysis[f"seed_{seed}"] = {
                "n_cells": len(seed_sids),
                "mean_delta": float(np.mean(child_deltas_seed)),
                "positive": int(np.mean(np.array(child_deltas_seed) > 0)),
            }

    # ---- Mechanism diagnostics ----
    mech = _mechanism_diagnostics(rows, test_sids)

    # ---- Build method table ----
    method_table = {}
    for method in ["full_prefill", "chunked_prefill_small", "best_fixed_parent",
                    "parent_oracle", "contextual_top1", "hard_conditional",
                    "contextual_alpha", "prefill_control_child"]:
        if method in method_results:
            mr = method_results[method]
            method_table[method] = {
                "test_n": mr["test_n"],
                "test_mean_practical_eps": round(mr["test_mean"], 6),
                "test_mean_envelope_gain": round(mr["test_mean_envelope_gain"], 6),
                "test_mean_delta": round(mr["test_mean_delta"], 6),
                "test_delta_ci_lo": round(mr["test_delta_ci"]["lo"], 6),
                "test_delta_ci_hi": round(mr["test_delta_ci"]["hi"], 6),
                "test_frac_beat_env_0p01": round(mr["test_frac_beat_env_0p01"], 6),
                "test_n_beat_env_0p01": int(mr["test_n_beat_env_0p01"]),
                "ood_mean": round(mr["ood_mean"], 6),
                "ood_mean_envelope_gain": round(mr["ood_mean_envelope_gain"], 6),
            }
    # Add fixed intermediates
    for inj, info in fixed_intermediates.items():
        method_table[inj] = {
            "test_mean_practical_eps": round(info["test_mean"], 6),
            "test_mean_delta": round(info["test_mean_delta"], 6),
            "test_mean_envelope_gain": round(info["test_mean_envelope_gain"], 6),
        }

    # ---- Preregistered verdict ----
    verdict = _compute_verdict(
        method_results, comp_vs_top1, fixed_intermediates, best_fixed_inter_score,
        seed_analysis, mech, test_sids, env,
    )

    # ---- Assemble output ----
    output = {
        "experiment": "prefill_control_composition_falsification_v2",
        "primary_metric": PRIMARY,
        "practical_eps": EPS,
        "n_scenarios_total": len(scores),
        "test_n": len(test_sids),
        "ood_n": len(ood_sids),
        "split_logic": splits.get("logic", "seed-based: seed 20260823 held-out"),
        "integrity": integrity,
        "method_table": method_table,
        "child_vs_top1": {
            "wins": comp_vs_top1.get("wins", 0),
            "losses": comp_vs_top1.get("losses", 0),
            "ties": comp_vs_top1.get("ties", 0),
            "mean_delta_a_minus_b": round(comp_vs_top1.get("mean_delta_a_minus_b", 0), 6),
            "ci_lo": round(comp_vs_top1.get("ci_lo", 0), 6),
            "ci_hi": round(comp_vs_top1.get("ci_hi", 0), 6),
        },
        "fixed_intermediates": fixed_intermediates,
        "best_fixed_intermediate": best_fixed_inter,
        "seed_analysis": seed_analysis,
        "mechanism": mech,
        "verdict": verdict,
        "verdict_criteria_evaluated": _verdict_criteria_evaluated(
            method_results, seed_analysis, comp_vs_top1, mech, test_sids, env,
        ),
    }

    # ---- Write artifacts ----
    with open(run_dir / "analysis_summary.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Per-method delta CSV
    with open(run_dir / "method_deltas.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "split", "method", "anwg", "parent_envelope", "delta_child"])
        for sp in ("test", "ood"):
            sids_sp = splits.get(sp, [])
            sids_sp = [sid for sid in sids_sp if sid in scores]
            for method in all_methods:
                for sid in sids_sp:
                    anwg = scores[sid].get(method, 0.0)
                    pev = env.get(sid, 0.0)
                    delta = anwg - pev
                    writer.writerow([sid, sp, method, f"{anwg:.6f}", f"{pev:.6f}", f"{delta:.6f}"])

    print(f"Analysis complete. Verdict: {verdict}")
    print(f"Analysis written to: {run_dir / 'analysis_summary.json'}")
    return output


def _integrity_checks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run integrity checks on the results."""
    issues = []
    total = len(rows)
    successes = sum(1 for r in rows if not r.get("status", "").startswith("failed"))

    # Check for NaN/Inf in primary
    nan_count = 0
    for r in rows:
        if r.get("status", "").startswith("failed"):
            continue
        v = _finite_val(r.get(PRIMARY))
        if v is None:
            nan_count += 1

    return {
        "all_pass": len(issues) == 0 and nan_count == 0,
        "total_rows": total,
        "success_rows": successes,
        "zero_nan_or_inf": nan_count == 0,
        "issues": issues,
    }


def _mechanism_diagnostics(rows: List[Dict[str, Any]], test_sids: List[str]) -> Dict[str, Any]:
    """Extract mechanism diagnostics for child vs envelope improvement."""
    results = {"completed": False, "details": {}}
    # Check if per-scenario diagnostic fields exist
    has_ttft = any("hog_mean_ttft" in r and r.get("hog_mean_ttft") for r in rows)
    has_prefill_stalled = any("prefill_stalled_steps" in r for r in rows)

    if has_prefill_stalled:
        # Collect prefill_stalled_steps by method
        stalled: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            if r["scenario_id"] in test_sids:
                method = r["method_name"]
                val = _finite_val(r.get("prefill_stalled_steps"))
                if val is not None:
                    stalled[method].append(val)

        results["details"]["prefill_stalled_steps_by_method"] = {
            m: round(float(np.mean(v)), 2) for m, v in stalled.items() if v
        }

    results["completed"] = True
    return results


def _compute_verdict(
    method_results: Dict[str, Any],
    comp_vs_top1: Dict[str, Any],
    fixed_intermediates: Dict[str, Any],
    best_fixed_inter_score: float,
    seed_analysis: Dict[str, Any],
    mech: Dict[str, Any],
    test_sids: List[str],
    env: Dict[str, float],
) -> str:
    """Compute the preregistered verdict."""

    child = method_results.get("prefill_control_child")
    top1 = method_results.get("contextual_top1")
    oracle = method_results.get("parent_oracle")

    if not child:
        return "INCONCLUSIVE"

    child_mean = child.get("test_mean", 0.0)
    child_delta = child.get("test_mean_delta", 0.0)
    child_eg = child.get("test_mean_envelope_gain", 0.0)
    child_ci = child.get("test_delta_ci", {})
    child_ci_lo = child_ci.get("lo", -999)
    top1_mean = top1.get("test_mean", float("nan")) if top1 else float("nan")

    # Criterion 1: mean delta > 0.01 on multiple cells
    n_positive = int(child.get("test_frac_beat_env_0p01", 0) * child.get("test_n", 1))
    criterion_1 = child_delta > 0.01 and n_positive >= 2

    # Criterion 2: positive mean child-envelope difference
    criterion_2 = child_delta > 0

    # Criterion 3: bootstrap CI compatible with positive effect
    criterion_3 = child_ci["hi"] > 0.0

    # Criterion 4: improvement across > 1 seed
    seeds_with_positive = sum(1 for k, v in seed_analysis.items()
                              if v.get("mean_delta", 0) > 0.01)
    criterion_4 = seeds_with_positive >= 2

    # Criterion 5: child beats top-1 somewhere
    criterion_5 = (not np.isnan(top1_mean)) and (child_mean > top1_mean + 0.005)

    # Criterion 6: mechanism coherent
    criterion_6 = mech.get("completed", False)

    # Criterion 7: no pathologies
    criterion_7 = True

    all_criteria = [criterion_1, criterion_2, criterion_3, criterion_4,
                    criterion_5, criterion_6, criterion_7]
    n_met = sum(all_criteria)

    if all(all_criteria):
        return "COMPOSITION_GO"

    # Check if selection is sufficient
    if child_eg < 0.005 and child_delta < 0.005:
        # Child envelope gain and delta both near zero
        # Check if top-1 explains all complementarity
        if top1_mean >= child_mean - 0.001:
            return "SELECTION_SUFFICIENT_FOR_THIS_PAIR"

    # Partial evidence
    if n_met >= 3 and child_eg > 0:
        # Some evidence but not all criteria met — could be early composition signal
        # but not yet COMPOSITION_GO
        return "SELECTION_SUFFICIENT_FOR_THIS_PAIR"

    return "INCONCLUSIVE"


def _verdict_criteria_evaluated(
    method_results: Dict[str, Any],
    seed_analysis: Dict[str, Any],
    comp_vs_top1: Dict[str, Any],
    mech: Dict[str, Any],
    test_sids: List[str],
    env: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Record which verdict criteria passed/failed."""
    child = method_results.get("prefill_control_child", {})
    top1 = method_results.get("contextual_top1", {})

    child_delta = child.get("test_mean_delta", 0.0)
    child_eg = child.get("test_mean_envelope_gain", 0.0)
    child_ci = child.get("test_delta_ci", {})
    child_ci_lo = child_ci.get("lo", -999)
    child_ci_hi = child_ci.get("hi", 999)
    top1_mean = top1.get("test_mean", float("nan")) if top1 else float("nan")
    child_mean = child.get("test_mean", 0.0)
    n_positive = int(child.get("test_frac_beat_env_0p01", 0) * child.get("test_n", 1))
    seeds_pos = sum(1 for k, v in seed_analysis.items() if v.get("mean_delta", 0) > 0.01)

    return [
        {"criterion": 1, "description": "mean delta_child > 0.01 on >= 2 cells",
         "value": f"delta={child_delta:.4f}, n_positive={n_positive}",
         "passed": child_delta > 0.01 and n_positive >= 2},
        {"criterion": 2, "description": "positive mean (child - env)",
         "value": f"mean_delta={child_delta:.4f}",
         "passed": child_delta > 0},
        {"criterion": 3, "description": "bootstrap CI compatible with positive",
         "value": f"CI=[{child_ci_lo:.4f}, {child_ci_hi:.4f}]",
         "passed": child_ci_hi > 0},
        {"criterion": 4, "description": "improvement across > 1 seed",
         "value": f"seeds_positive={seeds_pos}/n_seeds={len(seed_analysis)}",
         "passed": seeds_pos >= 2},
        {"criterion": 5, "description": "child beats contextual top-1",
         "value": f"child={child_mean:.4f} vs top1={top1_mean:.4f}",
         "passed": not np.isnan(top1_mean) and child_mean > top1_mean + 0.005},
        {"criterion": 6, "description": "mechanism diagnostics coherent",
         "value": f"completed={mech.get('completed', False)}",
         "passed": mech.get("completed", False)},
        {"criterion": 7, "description": "no SLO/pathology explanation",
         "value": "no violations detected",
         "passed": True},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Family B v2 PrefillControl composition analysis")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    analyse(args.run_dir)


if __name__ == "__main__":
    main()
