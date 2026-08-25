#!/usr/bin/env python3
"""Analyze Policy Separation Family A Fairness/Starvation pilot results.

Supports:
- Historical Job 1182306 schema where column ``anwg`` is *unweighted SLO-success*
  (NOT canonical RunMetrics.arrival_normalized_weighted_goodput).
- Future clarified schema with ``unweighted_slo_success_rate`` and optional
  ``arrival_normalized_weighted_goodput``.

Scenario coordinates are parsed from scenario_id strings of the form::

    fs.util{U}.skew{S}.vol{V}.s{SEED}

This is documented and unit-tested; Family A v1 did not write a separate
scenario_features.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_DIR = ROOT / (
    "experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306"
)

SCENARIO_ID_RE = re.compile(
    r"^fs\.util(?P<util>[0-9.]+)\.skew(?P<skew>[0-9.]+)\.vol(?P<vol>[0-9.]+)\.s(?P<seed>\d+)$"
)

HISTORICAL_PRIMARY_ALIASES = ("anwg",)  # Job 1182306 only
PRIMARY_CANDIDATES = (
    "unweighted_slo_success_rate",
    "anwg",  # historical alias for unweighted SLO-success
)
CANONICAL_ANWG_FIELD = "arrival_normalized_weighted_goodput"

EPSILONS = (0.0, 0.001, 0.005, 0.01)

POLICIES_EXPECTED = (
    "fifo",
    "estimated_service_time_first",
    "aging_priority",
    "weighted_fair_share",
)


def parse_scenario_id(scenario_id: str) -> Dict[str, Any]:
    """Parse Family A scenario_id into utilization/skew/vol/seed floats/ints."""
    m = SCENARIO_ID_RE.fullmatch(scenario_id)
    if m is None:
        raise ValueError(f"Unrecognized Family A scenario_id: {scenario_id!r}")
    return {
        "target_utilization": float(m.group("util")),
        "tenant_weight_skew": float(m.group("skew")),
        "interactive_volume_fraction": float(m.group("vol")),
        "seed": int(m.group("seed")),
    }


def resolve_primary_field(fieldnames: Sequence[str]) -> Tuple[str, str]:
    """Return (column_name, semantic_label) for the primary scalar.

    Semantic label is always explicit about unweighted SLO-success when the
    historical ``anwg`` column is used.
    """
    names = set(fieldnames)
    if "unweighted_slo_success_rate" in names:
        return "unweighted_slo_success_rate", "unweighted_slo_success_rate"
    if "anwg" in names:
        return "anwg", "historical_unweighted_slo_success_rate"
    raise ValueError(
        "No primary scalar column found; expected "
        "'unweighted_slo_success_rate' or historical 'anwg'"
    )


def load_rows(csv_path: Path) -> Tuple[List[dict], str, str, bool]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    primary_col, primary_label = resolve_primary_field(fieldnames)
    has_canonical = CANONICAL_ANWG_FIELD in fieldnames
    return rows, primary_col, primary_label, has_canonical


def _f(x: Any) -> float:
    return float(x)


def pivot_primary(
    rows: Sequence[Mapping[str, Any]], primary_col: str
) -> Dict[str, Dict[str, float]]:
    by: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        by[r["scenario_id"]][r["policy_name"]] = _f(r[primary_col])
    return dict(by)


def winner_margin(
    scores: Mapping[str, float],
) -> Tuple[List[str], Optional[str], Optional[str], float]:
    """Return (exact_winners, best_unique_or_None, second_label, best_vs_second_margin).

    ``best_vs_second_margin`` is always ``max - second_ranked`` on the sorted
    score list (0 under an exact top tie). ``best_unique_or_None`` is set only
    when there is a unique exact winner.
    """
    if not scores:
        return [], None, None, float("nan")
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_name, best_score = ranked[0]
    winners = [p for p, v in ranked if abs(v - best_score) <= 1e-12]
    if len(ranked) == 1:
        return winners, best_name, None, 0.0
    second_name, second_score = ranked[1]
    margin = best_score - second_score
    best_unique = best_name if len(winners) == 1 else None
    return winners, best_unique, second_name, margin


def near_tie(scores: Mapping[str, float], eps: float) -> bool:
    if not scores:
        return True
    best = max(scores.values())
    within = sum(1 for v in scores.values() if best - v <= eps + 1e-15)
    return within >= 2


def pairwise_delta(
    scores: Mapping[str, float], a: str, b: str
) -> float:
    return scores[a] - scores[b]


def analyze(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_col: str,
    primary_label: str,
    has_canonical_anwg: bool,
) -> Dict[str, Any]:
    policies = sorted({r["policy_name"] for r in rows})
    by_primary = pivot_primary(rows, primary_col)

    # Attach metadata + tenant metrics keyed by (scenario, policy)
    row_index: Dict[Tuple[str, str], Mapping[str, Any]] = {
        (r["scenario_id"], r["policy_name"]): r for r in rows
    }
    meta = {sid: parse_scenario_id(sid) for sid in by_primary}

    scenario_rows: List[dict] = []
    for sid, scores in sorted(by_primary.items()):
        winners, best, second, margin = winner_margin(scores)
        m = meta[sid]
        entry = {
            "scenario_id": sid,
            **m,
            "best_policy": best if best is not None else "|".join(winners),
            "second_policy": second,
            "best_vs_second_margin": margin,
            "exact_tie": len(winners) > 1,
            "exact_winners": "|".join(winners),
            "n_exact_winners": len(winners),
        }
        for eps in EPSILONS:
            entry[f"near_tie_eps_{eps}"] = near_tie(scores, eps)
        for p in policies:
            entry[f"score__{p}"] = scores[p]
        scenario_rows.append(entry)

    n_scen = len(scenario_rows)
    unique_wins = Counter(
        r["best_policy"] for r in scenario_rows if not r["exact_tie"]
    )
    exact_tie_n = sum(1 for r in scenario_rows if r["exact_tie"])
    near_tie_counts = {
        str(eps): sum(1 for r in scenario_rows if r[f"near_tie_eps_{eps}"])
        for eps in EPSILONS
    }
    margins = [r["best_vs_second_margin"] for r in scenario_rows]
    headroom_summary = {
        "mean": statistics.mean(margins) if margins else float("nan"),
        "median": statistics.median(margins) if margins else float("nan"),
        "frac_gt_0": (sum(1 for x in margins if x > 0) / n_scen) if n_scen else 0.0,
        "frac_gt_0.01": (
            sum(1 for x in margins if x > 0.01) / n_scen if n_scen else 0.0
        ),
    }

    # Winner distribution conditioned on axes
    def _cond_counts(key: str) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Counter] = defaultdict(Counter)
        for r in scenario_rows:
            label = str(r[key])
            if r["exact_tie"]:
                out[label]["EXACT_TIE"] += 1
            else:
                out[label][r["best_policy"]] += 1
        return {k: dict(v) for k, v in sorted(out.items(), key=lambda kv: float(kv[0]) if kv[0].replace(".", "", 1).isdigit() else kv[0])}

    winner_by_axis = {
        "target_utilization": _cond_counts("target_utilization"),
        "tenant_weight_skew": _cond_counts("tenant_weight_skew"),
        "interactive_volume_fraction": _cond_counts("interactive_volume_fraction"),
        "seed": _cond_counts("seed"),
    }

    # Pairwise
    pair_rows: List[dict] = []
    pair_summary: List[dict] = []
    for a, b in combinations(policies, 2):
        deltas = []
        for sid, scores in by_primary.items():
            d = pairwise_delta(scores, a, b)
            m = meta[sid]
            deltas.append(d)
            pair_rows.append(
                {
                    "scenario_id": sid,
                    **m,
                    "policy_i": a,
                    "policy_j": b,
                    "delta_ij": d,
                }
            )
        summary = {
            "policy_i": a,
            "policy_j": b,
            "mean_delta_ij": statistics.mean(deltas) if deltas else float("nan"),
            "median_delta_ij": statistics.median(deltas) if deltas else float("nan"),
        }
        for eps in EPSILONS:
            i_beats = sum(1 for d in deltas if d > eps)
            j_beats = sum(1 for d in deltas if d < -eps)
            ties = sum(1 for d in deltas if abs(d) <= eps)
            summary[f"i_beats_j_eps_{eps}"] = i_beats
            summary[f"j_beats_i_eps_{eps}"] = j_beats
            summary[f"near_ties_eps_{eps}"] = ties
            summary[f"bidirectional_eps_{eps}"] = i_beats > 0 and j_beats > 0
        pair_summary.append(summary)

    # Seed stability over (util, skew, vol)
    cells: Dict[Tuple[float, float, float], List[dict]] = defaultdict(list)
    for r in scenario_rows:
        key = (
            r["target_utilization"],
            r["tenant_weight_skew"],
            r["interactive_volume_fraction"],
        )
        cells[key].append(r)

    seed_rows: List[dict] = []
    important_pairs = [
        ("estimated_service_time_first", "weighted_fair_share"),
        ("aging_priority", "estimated_service_time_first"),
        ("aging_priority", "fifo"),
        ("weighted_fair_share", "fifo"),
        ("estimated_service_time_first", "fifo"),
    ]
    agree_winner_set = 0
    agree_best = 0
    n_cells = 0
    unstable_cells: List[dict] = []
    for (util, skew, vol), group in sorted(cells.items()):
        if len(group) != 2:
            # unexpected; still record
            pass
        n_cells += 1
        g_sorted = sorted(group, key=lambda r: r["seed"])
        sets = [tuple(sorted(r["exact_winners"].split("|"))) for r in g_sorted]
        bests = [r["best_policy"] for r in g_sorted]
        winner_set_agree = len(set(sets)) == 1
        best_agree = len(set(bests)) == 1
        if winner_set_agree:
            agree_winner_set += 1
        if best_agree:
            agree_best += 1
        margins_g = [r["best_vs_second_margin"] for r in g_sorted]
        margin_spread = max(margins_g) - min(margins_g) if margins_g else float("nan")
        pair_sign_agree = {}
        for a, b in important_pairs:
            signs = []
            for r in g_sorted:
                d = r[f"score__{a}"] - r[f"score__{b}"]
                if abs(d) <= 1e-12:
                    signs.append(0)
                else:
                    signs.append(1 if d > 0 else -1)
            pair_sign_agree[f"{a}_vs_{b}"] = len(set(signs)) == 1
        row = {
            "target_utilization": util,
            "tenant_weight_skew": skew,
            "interactive_volume_fraction": vol,
            "n_seeds_observed": len(group),
            "winner_set_agree": winner_set_agree,
            "best_policy_agree": best_agree,
            "margin_spread": margin_spread,
            "winner_sets": ";".join("|".join(s) for s in sets),
            "best_policies": ";".join(bests),
            **{f"sign_agree__{k}": v for k, v in pair_sign_agree.items()},
        }
        seed_rows.append(row)
        if (not winner_set_agree) or (not best_agree) or (
            not all(pair_sign_agree.values())
        ):
            unstable_cells.append(row)

    # Policy-level tenant metrics
    policy_fairness: List[dict] = []
    for p in policies:
        inter_rates = []
        bulk_rates = []
        jfis = []
        ttfts = []
        primaries = []
        for sid in by_primary:
            r = row_index[(sid, p)]
            inter_tot = max(1.0, _f(r["inter_total"]))
            bulk_tot = max(1.0, _f(r["bulk_total"]))
            inter_rates.append(_f(r["inter_violations"]) / inter_tot)
            bulk_rates.append(_f(r["bulk_violations"]) / bulk_tot)
            jfis.append(_f(r["jains_fairness_index"]))
            ttfts.append(_f(r["mean_ttft"]))
            primaries.append(_f(r[primary_col]))
        policy_fairness.append(
            {
                "policy_name": p,
                "mean_primary": statistics.mean(primaries),
                "median_primary": statistics.median(primaries),
                "mean_interactive_violation_rate": statistics.mean(inter_rates),
                "mean_bulk_violation_rate": statistics.mean(bulk_rates),
                "mean_jains_fairness_index": statistics.mean(jfis),
                "mean_ttft": statistics.mean(ttfts),
                "scenarios_with_inter_violations": sum(1 for x in inter_rates if x > 0),
                "scenarios_with_bulk_violations": sum(1 for x in bulk_rates if x > 0),
                "n_scenarios": n_scen,
            }
        )

    # Conditioned fairness surfaces (policy x util, skew, vol)
    fairness_surfaces: List[dict] = []
    for p in policies:
        for axis, axis_key in (
            ("target_utilization", "target_utilization"),
            ("tenant_weight_skew", "tenant_weight_skew"),
            ("interactive_volume_fraction", "interactive_volume_fraction"),
        ):
            buckets: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
            for sid, m in meta.items():
                buckets[m[axis_key]].append(row_index[(sid, p)])
            for val, group in sorted(buckets.items()):
                inter_rates = [
                    _f(r["inter_violations"]) / max(1.0, _f(r["inter_total"]))
                    for r in group
                ]
                bulk_rates = [
                    _f(r["bulk_violations"]) / max(1.0, _f(r["bulk_total"]))
                    for r in group
                ]
                primaries = [_f(r[primary_col]) for r in group]
                jfis = [_f(r["jains_fairness_index"]) for r in group]
                fairness_surfaces.append(
                    {
                        "policy_name": p,
                        "axis": axis,
                        "axis_value": val,
                        "n": len(group),
                        "mean_primary": statistics.mean(primaries),
                        "mean_interactive_violation_rate": statistics.mean(inter_rates),
                        "mean_bulk_violation_rate": statistics.mean(bulk_rates),
                        "mean_jains_fairness_index": statistics.mean(jfis),
                    }
                )

    # ESTF vs WFS explicit check
    estf_wfs = next(
        s
        for s in pair_summary
        if {s["policy_i"], s["policy_j"]}
        == {"estimated_service_time_first", "weighted_fair_share"}
    )
    # orient so i=ESTF, j=WFS
    if estf_wfs["policy_i"] != "estimated_service_time_first":
        # flip counts conceptually for report helper
        flipped = {
            **estf_wfs,
            "policy_i": "estimated_service_time_first",
            "policy_j": "weighted_fair_share",
            "mean_delta_ij": -estf_wfs["mean_delta_ij"],
            "median_delta_ij": -estf_wfs["median_delta_ij"],
        }
        for eps in EPSILONS:
            flipped[f"i_beats_j_eps_{eps}"] = estf_wfs[f"j_beats_i_eps_{eps}"]
            flipped[f"j_beats_i_eps_{eps}"] = estf_wfs[f"i_beats_j_eps_{eps}"]
            flipped[f"near_ties_eps_{eps}"] = estf_wfs[f"near_ties_eps_{eps}"]
            flipped[f"bidirectional_eps_{eps}"] = estf_wfs[f"bidirectional_eps_{eps}"]
        estf_wfs_oriented = flipped
    else:
        estf_wfs_oriented = estf_wfs

    # Integrity
    integrity = {
        "n_rows": len(rows),
        "n_scenarios": n_scen,
        "n_policies": len(policies),
        "policies": policies,
        "duplicate_scenario_policy_keys": len(rows)
        - len({(r["scenario_id"], r["policy_name"]) for r in rows}),
        "primary_column": primary_col,
        "primary_semantic_label": primary_label,
        "has_canonical_anwg_column": has_canonical_anwg,
        "expected_policies_present": all(p in policies for p in POLICIES_EXPECTED),
        "grid_product": None,
    }
    utils = sorted({m["target_utilization"] for m in meta.values()})
    skews = sorted({m["tenant_weight_skew"] for m in meta.values()})
    vols = sorted({m["interactive_volume_fraction"] for m in meta.values()})
    seeds = sorted({m["seed"] for m in meta.values()})
    integrity["grid"] = {
        "target_utilization": utils,
        "tenant_weight_skew": skews,
        "interactive_volume_fraction": vols,
        "seeds": seeds,
        "product": len(utils) * len(skews) * len(vols) * len(seeds),
    }
    integrity["grid_product_matches_n_scenarios"] = (
        integrity["grid"]["product"] == n_scen
    )

    # Aging saturation / ESTF interactive
    aging_all_perfect = all(
        abs(by_primary[sid]["aging_priority"] - 1.0) <= 1e-12 for sid in by_primary
    )
    estf_inter_any = sum(
        1
        for sid in by_primary
        if _f(row_index[(sid, "estimated_service_time_first")]["inter_violations"]) > 0
    )
    wfs_beats_estf = sum(
        1
        for sid, scores in by_primary.items()
        if scores["weighted_fair_share"] > scores["estimated_service_time_first"] + 1e-12
    )
    estf_beats_wfs = sum(
        1
        for sid, scores in by_primary.items()
        if scores["estimated_service_time_first"] > scores["weighted_fair_share"] + 1e-12
    )
    wfs_eq_estf = n_scen - wfs_beats_estf - estf_beats_wfs

    mechanism = {
        "aging_perfect_on_all_scenarios": aging_all_perfect,
        "estf_scenarios_with_interactive_violations": estf_inter_any,
        "wfs_beats_estf_scenarios": wfs_beats_estf,
        "estf_beats_wfs_scenarios": estf_beats_wfs,
        "wfs_equals_estf_scenarios": wfs_eq_estf,
        "estf_wfs_bidirectional_eps_0.01": bool(
            estf_wfs_oriented["bidirectional_eps_0.01"]
        ),
    }

    summary = {
        "experiment": "policy_separation_fairness_starvation_pilot_v1",
        "primary_metric_caveat": (
            "Historical Job 1182306 column 'anwg' is unweighted SLO-success, "
            "NOT canonical RunMetrics.arrival_normalized_weighted_goodput."
        ),
        "integrity": integrity,
        "n_scenarios": n_scen,
        "unique_winner_counts": dict(unique_wins),
        "exact_tie_count": exact_tie_n,
        "exact_tie_rate": exact_tie_n / n_scen if n_scen else 0.0,
        "near_tie_counts": near_tie_counts,
        "near_tie_rates": {
            k: (v / n_scen if n_scen else 0.0) for k, v in near_tie_counts.items()
        },
        "headroom": headroom_summary,
        "winner_by_axis": winner_by_axis,
        "seed_stability": {
            "n_cells": n_cells,
            "winner_set_agree": agree_winner_set,
            "best_policy_agree": agree_best,
            "winner_set_agree_rate": agree_winner_set / n_cells if n_cells else 0.0,
            "best_policy_agree_rate": agree_best / n_cells if n_cells else 0.0,
            "n_unstable_cells": len(unstable_cells),
        },
        "estf_vs_wfs": estf_wfs_oriented,
        "mechanism_flags": mechanism,
        "policy_fairness_overall": policy_fairness,
        "scientific_verdict_hint": "USEFUL_DIAGNOSTIC_ONLY",
    }

    return {
        "summary": summary,
        "per_scenario": scenario_rows,
        "pairwise_deltas": pair_rows,
        "pairwise_summary": pair_summary,
        "seed_stability": seed_rows,
        "unstable_cells": unstable_cells,
        "policy_fairness": policy_fairness,
        "fairness_surfaces": fairness_surfaces,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Stable union of keys
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_artifacts(out_dir: Path, bundle: Mapping[str, Any]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": out_dir / "analysis_summary.json",
        "per_scenario_csv": out_dir / "per_scenario_winners.csv",
        "pairwise_summary_csv": out_dir / "pairwise_summary.csv",
        "pairwise_deltas_csv": out_dir / "pairwise_deltas.csv",
        "seed_stability_csv": out_dir / "seed_stability.csv",
        "unstable_cells_csv": out_dir / "unstable_cells.csv",
        "policy_fairness_csv": out_dir / "policy_fairness_overall.csv",
        "fairness_surfaces_csv": out_dir / "fairness_surfaces.csv",
    }
    paths["summary_json"].write_text(
        json.dumps(bundle["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(paths["per_scenario_csv"], bundle["per_scenario"])
    _write_csv(paths["pairwise_summary_csv"], bundle["pairwise_summary"])
    _write_csv(paths["pairwise_deltas_csv"], bundle["pairwise_deltas"])
    _write_csv(paths["seed_stability_csv"], bundle["seed_stability"])
    _write_csv(paths["unstable_cells_csv"], bundle["unstable_cells"])
    _write_csv(paths["policy_fairness_csv"], bundle["policy_fairness"])
    _write_csv(paths["fairness_surfaces_csv"], bundle["fairness_surfaces"])
    return {k: str(v.relative_to(ROOT) if v.is_relative_to(ROOT) else v) for k, v in paths.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Experiment directory containing per_policy_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for analysis artifacts (default: <run-dir>/analysis)",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    csv_path = run_dir / "per_policy_results.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing results CSV: {csv_path}")

    rows, primary_col, primary_label, has_canonical = load_rows(csv_path)
    bundle = analyze(
        rows,
        primary_col=primary_col,
        primary_label=primary_label,
        has_canonical_anwg=has_canonical,
    )
    out_dir = args.out_dir if args.out_dir is not None else run_dir / "analysis"
    written = write_artifacts(out_dir, bundle)

    print("Family A analysis complete")
    print(f"  run_dir={run_dir}")
    print(f"  primary_column={primary_col} ({primary_label})")
    print(f"  n_scenarios={bundle['summary']['n_scenarios']}")
    print(f"  exact_tie_rate={bundle['summary']['exact_tie_rate']:.4f}")
    print(
        "  near_tie_rate_eps_0.01="
        f"{bundle['summary']['near_tie_rates']['0.01']:.4f}"
    )
    print(f"  unique_winners={bundle['summary']['unique_winner_counts']}")
    print(
        "  estf_wfs_bidirectional_eps_0.01="
        f"{bundle['summary']['estf_vs_wfs']['bidirectional_eps_0.01']}"
    )
    print("  artifacts:")
    for k, v in written.items():
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
