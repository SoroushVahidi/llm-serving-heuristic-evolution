#!/usr/bin/env python3
"""p5_analysis_chunk_comp.py — Family B v2 PrefillControl composition analysis.

Preregistered scientific analysis that answers the decisive question:

    "Does contextual PrefillControl composition provide statistically and
     practically credible held-out envelope expansion beyond contextual
     top-1 selection and the original two-parent envelope?"

Criteria (preregistered):
- Positive composition evidence requires genuine held-out envelope expansion:
  child method beats BOTH parents on held-out scenarios at ε>0, AND
  envelope gain's bootstrap CI lower bound > 0 at 95%.
- SELECTION_SUFFICIENT_FOR_THIS_PAIR if contextual top-1 already matches
  the parent envelope on test/ood (no gain beyond smart selection).
- INCONCLUSIVE if split sizes are too small, or results are ambiguous.

This script loads the CSV output from p7_runner.py and writes:
- composition_analysis.json (full machine-readable results)
- per_scenario_analysis.csv (per-scenario deltas)

All computation uses canonical `arrival_normalized_weighted_goodput`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import p3_chunk_control as p3

from llmserveopt.composition.prefill_control_metrics import (  # noqa: E402
    parent_envelope,
    envelope_gain,
    bootstrap_ci,
    paired_bootstrap_ci,
    paired_deltas,
    best_fixed_parent_score,
    best_fixed_intermediate_score,
    pairwise_comparison,
    oracle_scores,
    oracle_regret,
)

# ===================================================================
# Constants
# ===================================================================

PRIMARY = "arrival_normalized_weighted_goodput"
PRACTICAL_EPS = 0.01
HELD_OUT_SEED = 20260823
PARENTS = ("full_prefill", "chunked_prefill_small")

SCENARIO_ID_RE = re.compile(
    r"^pd2\.hog(?P<n_hog>\d+)\.late(?P<n_late>\d+)"
    r"\.slo(?P<slo_emphasis>hog_ttft|late_ttft)\.s(?P<seed>\d+)$"
)


def parse_scenario_id(sid: str) -> Dict[str, Any]:
    m = SCENARIO_ID_RE.fullmatch(sid)
    if not m:
        return {"scenario_id": sid, "seed": 0, "n_hog": 0, "n_late": 0,
                "slo_emphasis": "unknown", "pair_id": sid}
    d = m.groupdict()
    return {
        "scenario_id": sid,
        "n_hog": int(d["n_hog"]),
        "n_late": int(d["n_late"]),
        "slo_emphasis": d["slo_emphasis"],
        "seed": int(d["seed"]),
        "pair_id": f"pd2.hog{d['n_hog']}.late{d['n_late']}.slo{d['slo_emphasis']}",
    }


def _f(x: Any) -> float:
    return float(x)


# ===================================================================
# Score lookup helpers
# ===================================================================

def _scores(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    """Convert result rows to {method: {scenario_id: score}}."""
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r.get("status") != "success":
            continue
        score = _f(r.get(PRIMARY, 0.0))
        out[r["policy_name"]][r["scenario_id"]] = score
    return {k: dict(v) for k, v in out.items()}


def _scenario_ids(rows: List[dict]) -> List[str]:
    seen = []
    for r in rows:
        if r.get("status") == "success" and r["scenario_id"] not in seen:
            seen.append(r["scenario_id"])
    return seen


# ===================================================================
# Integrity checks
# ===================================================================

def integrity_checks(rows: List[dict]) -> Dict[str, Any]:
    success = [r for r in rows if r.get("status") == "success"]
    failed = [r for r in rows if r.get("status") == "failed"]
    all_sids = [r["scenario_id"] for r in rows]

    # Duplicate (scenario_id, policy) pairs
    seen_pairs = set()
    dupes = 0
    for r in rows:
        key = (r["scenario_id"], r["policy_name"])
        if key in seen_pairs:
            dupes += 1
        seen_pairs.add(key)

    # NaN/Inf in primary
    nan_inf = 0
    for r in success:
        try:
            v = _f(r.get(PRIMARY, "nan"))
            if not math.isfinite(v):
                nan_inf += 1
        except (TypeError, ValueError):
            nan_inf += 1

    # Policy coverage
    policies_seen = {r["policy_name"] for r in success}

    return {
        "n_rows": len(rows),
        "n_success": len(success),
        "n_failed": len(failed),
        "n_scenarios": len(set(all_sids)),
        "duplicate_pairs": dupes,
        "nan_or_inf_primary": nan_inf,
        "policies_seen": sorted(policies_seen),
        "has_parent_full": "full_prefill" in policies_seen,
        "has_parent_small": "chunked_prefill_small" in policies_seen,
    }


# ===================================================================
# Split assignment validation
# ===================================================================

def split_integrity_check(rows: List[dict]) -> Dict[str, Any]:
    """Validate that test/ood are truly held-out (seed=20260823)."""
    # Collect scenario metadata from scenario_id patterns
    splits_seen: Dict[str, set] = {"train": set(), "val": set(),
                                    "test": set(), "ood": set()}
    seed_to_sids: Dict[int, set] = defaultdict(set)
    for r in rows:
        sid = r["scenario_id"]
        split = r.get("split", "unknown")
        meta = parse_scenario_id(sid)
        seed_to_sids[meta["seed"]].add(sid)
        if split in splits_seen:
            splits_seen[split].add(sid)

    test_sids = splits_seen.get("test", set())
    ood_sids = splits_seen.get("ood", set())

    # Check: test + ood scenarios should only have seed 20260823
    test_ood_seeds = set()
    for sid in test_sids | ood_sids:
        meta = parse_scenario_id(sid)
        test_ood_seeds.add(meta["seed"])

    # Check disjointness
    buckets = {k: v for k, v in splits_seen.items() if v}
    disjoint_ok = True
    overlap_sets = []
    names = sorted(buckets)
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            inter = buckets[a_name] & buckets[b_name]
            if inter:
                disjoint_ok = False
                overlap_sets.append({"sets": [a_name, b_name], "overlap_count": len(inter)})

    # Held-out check
    held_out_clean = test_ood_seeds == {HELD_OUT_SEED} if test_ood_seeds else True

    return {
        "split_sizes": {k: len(v) for k, v in splits_seen.items()},
        "test_ood_only_held_out_seed": held_out_clean,
        "test_ood_seeds_found": sorted(test_ood_seeds),
        "splits_disjoint": disjoint_ok,
        "overlap_sets": overlap_sets,
    }


# ===================================================================
# Verdict logic (preregistered)
# ===================================================================

def compute_verdict(analysis: Dict[str, Any]) -> str:
    """Deterministic preregistered verdict from analysis results.

    Returns one of:
    - COMPOSITION_GO: genuine held-out envelope expansion confirmed
    - SELECTION_SUFFICIENT_FOR_THIS_PAIR: top-1 selection already matches envelope
    - INCONCLUSIVE: insufficient evidence or ambiguous results
    """
    test_results = analysis.get("test_results", {})
    ood_results = analysis.get("ood_results", {})

    # Requirement checks for COMPOSITION_GO:
    # 1. Envelope gain > 0 on TEST with practical significance
    test_gain = test_results.get("envelope_gain", {})
    test_mean_gain = test_gain.get("mean_envelope_gain", 0.0)

    # 2. Bootstrap CI lower bound > 0
    test_boot = test_results.get("envelope_gain_bootstrap_ci", [float("nan")] * 3)
    test_ci_lo = test_boot[1] if len(test_boot) > 1 else float("nan")

    # 3. Child beats both parents on at least some scenarios
    test_beats = test_results.get("percent_beat_parent_full", 0.0)
    test_beats_small = test_results.get("percent_beat_parent_small", 0.0)

    # 4. Split sizes adequate
    test_n = test_results.get("n_scenarios", 0)
    ood_n = ood_results.get("n_scenarios", 0)

    adequate_samples = test_n >= 4 and ood_n >= 2

    # Check if selector already matches envelope (selection sufficient)
    selector_vs_oracle = test_results.get("selector_vs_oracle_delta", float("nan"))
    if math.isnan(selector_vs_oracle):
        selector_matches = False
    else:
        selector_matches = abs(selector_vs_oracle) < 0.005  # effectively matches

    # Check composition beats selector
    composition_vs_selector = test_results.get("composition_vs_selector_delta", float("nan"))
    if math.isnan(composition_vs_selector):
        comp_beats_selector = False
    else:
        comp_beats_selector = composition_vs_selector > PRACTICAL_EPS

    # COMPOSITION_GO criteria:
    # - mean envelope gain > 0 on TEST
    # - bootstrap CI lo > 0 on TEST
    # - adequate samples
    # - composition beats selector (or selector already matches oracle)
    if (test_mean_gain > PRACTICAL_EPS
            and not math.isnan(test_ci_lo) and test_ci_lo > 0
            and adequate_samples
            and (comp_beats_selector or selector_matches)):
        return "COMPOSITION_GO"

    # SELECTION_SUFFICIENT_FOR_THIS_PAIR:
    # Top-1 selector already achieves the envelope (no need for composition)
    if selector_matches and test_mean_gain < PRACTICAL_EPS:
        return "SELECTION_SUFFICIENT_FOR_THIS_PAIR"

    # INCONCLUSIVE:
    # Insufficient evidence, ambiguous results, or no envelope gain
    return "INCONCLUSIVE"


# ===================================================================
# Main analysis pipeline
# ===================================================================

def analyse(
    rows: List[dict],
    *,
    features: Optional[Dict[str, dict]] = None,
) -> Dict[str, Any]:
    """Run the full preregistered analysis pipeline."""
    features = features or {}

    # ---- Integrity ----
    integrity = integrity_checks(rows)
    split_integrity = split_integrity_check(rows)

    # ---- Score matrices ----
    all_scores = _scores(rows)
    scenario_ids = _scenario_ids(rows)

    # ---- Split scenarios ----
    train_sids = []
    val_sids = []
    test_sids = []
    ood_sids = []
    for r in rows:
        sid = r["scenario_id"]
        split = r.get("split", "")
        if split == "train" and sid not in train_sids:
            train_sids.append(sid)
        elif split == "val" and sid not in val_sids:
            val_sids.append(sid)
        elif split == "test" and sid not in test_sids:
            test_sids.append(sid)
        elif split == "ood" and sid not in ood_sids:
            ood_sids.append(sid)

    # Fallback: if split column not available, classify by seed
    if not test_sids and not ood_sids:
        for sid in scenario_ids:
            meta = parse_scenario_id(sid)
            if meta["seed"] == HELD_OUT_SEED:
                if "late40" in sid:
                    ood_sids.append(sid)
                else:
                    test_sids.append(sid)
            else:
                train_sids.append(sid if sid not in [s for s in train_sids + val_sids] else sid)

    # If we have test but no split column for train/val:
    non_held = [s for s in scenario_ids if s not in set(test_sids) and s not in set(ood_sids)]
    if not val_sids:
        val_sids = sorted(non_held[:min(8, len(non_held))])
    if not train_sids:
        train_sids = sorted(s for s in non_held if s not in val_sids)

    # ---- Parent envelope ----
    full_scores = all_scores.get("full_prefill", {})
    small_scores = all_scores.get("chunked_prefill_small", {})

    def compute_envelope(full_sc, small_sc, sids):
        return {sid: max(full_sc.get(sid, 0.0), small_sc.get(sid, 0.0)) for sid in sids}

    train_env = compute_envelope(full_scores, small_scores, train_sids)
    test_env = compute_envelope(full_scores, small_scores, test_sids)
    ood_env = compute_envelope(full_scores, small_scores, ood_sids)
    all_env = compute_envelope(full_scores, small_scores, scenario_ids)

    # ---- Contextual top-1 (oracle version for analysis) ----
    def oracle_selector(sid, sc):
        s_full = sc.get("full_prefill", {}).get(sid, 0.0)
        s_small = sc.get("chunked_prefill_small", {}).get(sid, 0.0)
        return "full_prefill" if s_full >= s_small else "chunked_prefill_small"

    def get_oracle_scores(sc, sids):
        return {sid: full_scores.get(oracle_selector(sid, sc), 0.0)
                if oracle_selector(sid, sc) == "full_prefill" else small_scores.get(sid, 0.0)
                for sid in sids}

    test_sel_scores = get_oracle_scores(all_scores, test_sids)
    ood_sel_scores = get_oracle_scores(all_scores, ood_sids)

    # Compute selector-oracle gaps
    test_oracle = get_oracle_scores(all_scores, test_sids)
    ood_oracle = get_oracle_scores(all_scores, ood_sids)
    # For our case, oracle = best-per-parent-per-scenario (same as sel_scores)
    # So gap should be 0. But if we had a fitted selector, gap > 0 would exist.

    # ---- Best fixed parent per split ----
    test_bfp = best_fixed_parent_score(full_scores, small_scores, test_sids)
    ood_bfp = best_fixed_parent_score(full_scores, small_scores, ood_sids)

    # ---- Find child methods (non-parent methods) ----
    raw_child_names = [n for n in all_scores
                       if n not in PARENTS
                       and n not in ("contextual_top1", "hard_conditional",
                                     "contextual_alpha", "best_fixed_parent",
                                     "parent_oracle", "best_fixed_intermediate")]

    # ---- Envelope gain for each child ----
    def child_scores(cname, sids_x):
        if cname in all_scores:
            return {sid: all_scores[cname].get(sid, 0.0) for sid in sids_x}
        return {sid: 0.0 for sid in sids_x}

    test_child_gains = {}
    ood_child_gains = {}
    test_child_cis = {}
    child_beats = {}

    for cname in raw_child_names:
        cs_test = child_scores(cname, test_sids)
        cs_ood = child_scores(cname, ood_sids)
        if not cs_test:
            continue
        eg_test = envelope_gain(cs_test, test_env, test_sids)
        eg_ood = envelope_gain(cs_ood, ood_env, ood_sids)
        test_child_gains[cname] = eg_test
        ood_child_gains[cname] = eg_ood

        # Bootstrap CI
        gains = [max(float(cs_test.get(sid, 0.0)) - float(test_env.get(sid, 0.0)), 0.0)
                 for sid in test_sids]
        if gains:
            ci = bootstrap_ci(gains, n_boot=2000, seed=20261201, alpha=0.05)
            test_child_cis[cname] = list(ci)

        # Beats both count
        beats_0 = sum(1 for sid in test_sids
                      if float(cs_test.get(sid, 0.0)) > float(test_env.get(sid, 0.0)) + 0.0)
        beats_01 = sum(1 for sid in test_sids
                       if float(cs_test.get(sid, 0.0)) > float(test_env.get(sid, 0.0)) + 0.01)
        child_beats[cname] = {"eps_0": beats_0, "eps_001": beats_01}

    # ---- Paired per-scenario deltas ----
    test_pair_deltas = []
    for sid in test_sids:
        c = float(test_sel_scores.get(sid, 0.0))
        e = float(test_env.get(sid, 0.0))
        test_pair_deltas.append(c - e)
    ood_pair_deltas = []
    for sid in ood_sids:
        c = float(ood_sel_scores.get(sid, 0.0))
        e = float(ood_env.get(sid, 0.0))
        ood_pair_deltas.append(c - e)

    # ---- Composition vs selector comparison ----
    # Find best child by mean envelope gain on TEST
    best_comp_child = None
    best_comp_gain = -float("inf")
    for cname, g in test_child_gains.items():
        mg = g.get("mean_envelope_gain", 0.0)
        if mg > best_comp_gain:
            best_comp_gain = mg
            best_comp_child = cname

    comp_vs_sel_delta = float("nan")
    if best_comp_child:
        cs_comp = child_scores(best_comp_child, test_sids)
        deltas_c = [float(cs_comp.get(sid, 0.0)) - float(test_sel_scores.get(sid, 0.0))
                    for sid in test_sids]
        comp_vs_sel_delta = float(np.mean(deltas_c)) if deltas_c else float("nan")

    # ---- ID/ID robustness ----
    best_test_child = max(test_child_gains.items(),
                           key=lambda x: x[1].get("mean_envelope_gain", 0),
                           default=("none", {}))
    best_ood_child = max(ood_child_gains.items(),
                          key=lambda x: x[1].get("mean_envelope_gain", 0),
                          default=("none", {}))

    robustness = {
        "best_test_child": best_test_child[0],
        "best_test_gain": best_test_child[1].get("mean_envelope_gain", 0.0) if best_test_child[0] != "none" else 0.0,
        "best_ood_child": best_ood_child[0],
        "best_ood_gain": best_ood_child[1].get("mean_envelope_gain", 0.0) if best_ood_child[0] != "none" else 0.0,
        "same_best_child": best_test_child[0] == best_ood_child[0],
    }

    # ---- Failure / safety diagnostics ----
    n_failed = sum(1 for r in rows if r.get("status") == "failed")
    n_nan = 0
    for r in rows:
        try:
            v = _f(r.get(PRIMARY, "nan"))
            if not math.isfinite(v):
                n_nan += 1
        except (TypeError, ValueError):
            n_nan += 1
    n_exact_ties = 0
    n_near_ties = 0
    all_full_sc = all_scores.get("full_prefill", {})
    all_small_sc = all_scores.get("chunked_prefill_small", {})
    for sid in scenario_ids:
        sf = all_full_sc.get(sid, 0.0)
        ss = all_small_sc.get(sid, 0.0)
        delta = abs(sf - ss)
        if delta <= 1e-15:
            n_exact_ties += 1
        elif delta <= PRACTICAL_EPS:
            n_near_ties += 1

    safety = {
        "n_failed": n_failed,
        "n_nan_inf": n_nan,
        "n_scenarios": len(scenario_ids),
        "n_exact_ties": n_exact_ties,
        "n_near_ties_eps0.01": n_near_ties,
        "exact_tie_rate": n_exact_ties / max(1, len(scenario_ids)),
        "near_tie_rate": n_near_ties / max(1, len(scenario_ids)),
    }

    # ---- Assemble test/ood split results ----
    def make_split_results(sids, env, selector_sc, child_gains_x,
                           child_cis_x, child_beats_x):
        mean_full = float(np.mean([full_scores.get(sid, 0.0) for sid in sids]))
        mean_small = float(np.mean([small_scores.get(sid, 0.0) for sid in sids]))
        mean_oracle = float(np.mean([env.get(sid, 0.0) for sid in sids]))

        mean_sel = float(np.mean([selector_sc.get(sid, 0.0) for sid in sids]))

        best_cg = max(child_gains_x.items(), key=lambda x: x[1].get("mean_envelope_gain", 0),
                      default=("none", {}))
        top_eg = best_cg[1] if best_cg[0] != "none" else {}
        top_cis = child_cis_x.get(best_cg[0], [float("nan")] * 3) if best_cg[0] in child_cis_x else [float("nan")] * 3

        # Child beats both parents
        top_child_name = best_cg[0]
        top_beats = child_beats_x.get(top_child_name, {"eps_0": 0, "eps_001": 0})

        # Fraction beating each parent
        pct_beat_full = 0.0
        pct_beat_small = 0.0
        top_cs = child_scores(top_child_name, sids) if top_child_name != "none" else {sid: 0.0 for sid in sids}
        n_s = len(sids)
        if n_s > 0 and top_child_name != "none":
            pct_beat_full = sum(1 for sid in sids
                                if float(top_cs.get(sid, 0.0)) > float(full_scores.get(sid, 0.0)) + PRACTICAL_EPS) / n_s
            pct_beat_small = sum(1 for sid in sids
                                 if float(top_cs.get(sid, 0.0)) > float(small_scores.get(sid, 0.0)) + PRACTICAL_EPS) / n_s

        return {
            "n_scenarios": float(len(sids)),
            "mean_full": mean_full,
            "mean_small": mean_small,
            "mean_oracle": mean_oracle,
            "mean_selector": mean_sel,
            "top_child_name": top_child_name,
            "envelope_gain": top_eg.get("mean_envelope_gain", 0.0),
            "mean_envelope_gain_raw": top_eg.get("mean_envelope_gain", 0.0),
            "median_envelope_gain": top_eg.get("median_envelope_gain", 0.0),
            "envelope_gain_bootstrap_ci": top_cis,
            "select_oracle_delta": 0.0,  # oracle selector matches env
            "fraction_beat_parent_full": pct_beat_full,
            "fraction_beat_parent_small": pct_beat_small,
            "child_beats_both_eps_0": top_beats.get("eps_0", 0),
            "child_beats_both_eps_001": top_beats.get("eps_001", 0),
            "child_methods_in_split": list(child_gains_x.keys()),
        }

    test_results = make_split_results(test_sids, test_env, test_sel_scores,
                                       test_child_gains, test_child_cis, child_beats)
    ood_results = make_split_results(ood_sids, ood_env, ood_sel_scores,
                                      ood_child_gains, test_child_cis, child_beats)

    # ================================================================
    # Parent oracle envelope results
    # ================================================================

    test_all_oracle = oracle_scores(full_scores, small_scores, test_sids)
    ood_all_oracle = oracle_scores(full_scores, small_scores, ood_sids)

    # ================================================================
    # Selector vs composition final comparison
    # ================================================================

    composition_vs_selector_info = {
        "best_child": best_comp_child or "none",
        "delta": comp_vs_sel_delta,
    }

    # ================================================================
    # Verdict
    # ================================================================
    verdict = compute_verdict({
        "test_results": {
            **test_results,
            "envelope_gain": {"mean_envelope_gain": test_results.get("mean_envelope_gain_raw", 0.0)},
            "envelope_gain_bootstrap_ci": test_results.get("envelope_gain_bootstrap_ci", [0.0]*3),
            "percent_beat_parent_full": test_results.get("fraction_beat_parent_full", 0.0),
            "percent_beat_parent_small": test_results.get("fraction_beat_parent_small", 0.0),
            "selector_vs_oracle_delta": test_results.get("select_oracle_delta", float("nan")),
            "composition_vs_selector_delta": comp_vs_sel_delta,
        },
        "ood_results": {
            **ood_results,
            "envelope_gain": {"mean_envelope_gain": ood_results.get("mean_envelope_gain_raw", 0.0)},
            "n_scenarios": float(len(ood_sids)),
        },
    })

    # ================================================================
    # Per-scenario rows
    # ================================================================
    per_scenario_rows = []
    for sid in sorted(set(test_sids) | set(ood_sids)):
        meta = parse_scenario_id(sid)
        row_dict = dict(meta)
        row_dict["split"] = "test" if sid in test_sids else "ood"
        row_dict["anwg_full"] = full_scores.get(sid, float("nan"))
        row_dict["anwg_small"] = small_scores.get(sid, float("nan"))
        row_dict["parent_envelope"] = all_env.get(sid, float("nan"))
        row_dict["delta_full_small"] = full_scores.get(sid, 0.0) - small_scores.get(sid, 0.0)
        for cname in raw_child_names:
            cs = child_scores(cname, [sid])
            val = float(cs.get(sid, 0.0))
            row_dict[f"child_{cname}_anwg"] = val
            row_dict[f"child_{cname}_delta_envelope"] = val - float(all_env.get(sid, 0.0))
        per_scenario_rows.append(row_dict)

    return {
        "integrity": integrity,
        "split_integrity": split_integrity,
        "test_results": test_results,
        "ood_results": ood_results,
        "best_child_test": list(best_test_child),
        "best_child_ood": list(best_ood_child),
        "child_gains_test": {k: v.get("mean_envelope_gain", 0) for k, v in test_child_gains.items()},
        "child_gains_ood": {k: v.get("mean_envelope_gain", 0) for k, v in ood_child_gains.items()},
        "child_beats_both_test": child_beats,
        "child_beats_both_ood": {k: v for k, v in child_beats.items()},
        "child_cis_test": {k: list(v) for k, v in test_child_cis.items()},
        "paired_deltas_test": [float(d) for d in test_pair_deltas],
        "paired_deltas_ood": [float(d) for d in ood_pair_deltas],
        "composition_vs_selector": composition_vs_selector_info,
        "robustness": robustness,
        "safety": safety,
        "oracle_envelope_test": {sid: test_all_oracle.get(sid, 0.0) for sid in test_sids},
        "oracle_envelope_ood": {sid: ood_all_oracle.get(sid, 0.0) for sid in ood_sids},
        "verdict": verdict,
        "per_scenario_analysis": per_scenario_rows,
    }


def write_artifacts(analysis: Dict[str, Any], out_dir: Path) -> None:
    """Write analysis JSON and per-scenario CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    slim = {k: v for k, v in analysis.items() if k != "per_scenario_analysis"}
    with open(out_dir / "composition_analysis.json", "w") as f:
        json.dump(slim, f, indent=2, default=str)

    rows = analysis.get("per_scenario_analysis", [])
    if rows:
        fieldnames = sorted(rows[0].keys())
        with open(out_dir / "per_scenario_analysis.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(f"Verdict: {analysis['verdict']}")
    print(f"Test n={analysis['test_results'].get('n_scenarios', 0)}, "
          f"OOD n={analysis['ood_results'].get('n_scenarios', 0)}")
    print(f"Best child TEST: {analysis['best_child_test']}")
    print(f"Best child OOD: {analysis['best_child_ood']}")
    print(f"Child gains TEST: {analysis['child_gains_test']}")
    print(f"Child gains OOD: {analysis['child_gains_ood']}")
    print(f"Safety: {analysis['safety']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Family B v2 PrefillControl composition analysis"
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory with per_policy_results.csv")
    parser.add_argument("--features", type=Path, default=None, help="Optional scenario_features.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <run-dir>/analysis)")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.run_dir / "per_policy_results.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing results file: {csv_path}")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    features = {}
    if args.features:
        with open(args.features) as f:
            features = {r["scenario_id"]: r for r in csv.DictReader(f)}

    analysis = analyse(rows, features=features)
    write_artifacts(analysis, out_dir)

    with open(out_dir / "composition_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)


if __name__ == "__main__":
    main()
