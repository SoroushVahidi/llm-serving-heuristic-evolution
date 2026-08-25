#!/usr/bin/env python3
"""Analyze Policy Separation Family A v2 fairness-vs-size pilot results.

Primary metric: canonical ``arrival_normalized_weighted_goodput``.
Scenario IDs::

    fs2.util{U}.skew{S}.fav{short|long}.noise{N}.s{SEED}

Optional ``scenario_features.csv`` supplies the same factors plus BurstGPT
provenance fields.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_DIR = ROOT / (
    "experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377"
)

SCENARIO_ID_RE = re.compile(
    r"^fs2\.util(?P<util>[0-9.]+)\.skew(?P<skew>[0-9.]+)"
    r"\.fav(?P<fav>short|long)\.noise(?P<noise>[0-9.]+)\.s(?P<seed>\d+)$"
)

PRIMARY = "arrival_normalized_weighted_goodput"
EPSILONS = (0.0, 0.001, 0.005, 0.01)
POLICIES_EXPECTED = (
    "fifo",
    "estimated_service_time_first",
    "aging_priority",
    "weighted_fair_share",
)


def parse_scenario_id(scenario_id: str) -> Dict[str, Any]:
    m = SCENARIO_ID_RE.fullmatch(scenario_id)
    if m is None:
        raise ValueError(f"Unrecognized Family A v2 scenario_id: {scenario_id!r}")
    return {
        "target_utilization": float(m.group("util")),
        "tenant_weight_skew": float(m.group("skew")),
        "favored_tenant_size": m.group("fav"),
        "prediction_noise_sigma": float(m.group("noise")),
        "seed": int(m.group("seed")),
    }


def _f(x: Any) -> float:
    return float(x)


def winner_margin(
    scores: Mapping[str, float],
) -> Tuple[List[str], Optional[str], Optional[str], float]:
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
    return sum(1 for v in scores.values() if best - v <= eps + 1e-15) >= 2


def shannon_entropy(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p, 2)
    return h


def load_features(path: Path) -> Dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="") as f:
        return {r["scenario_id"]: r for r in csv.DictReader(f)}


def analyze(
    rows: Sequence[Mapping[str, Any]],
    *,
    features: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    features = features or {}
    fieldnames = list(rows[0].keys()) if rows else []
    if PRIMARY not in fieldnames:
        raise ValueError(f"Missing required primary column {PRIMARY!r}")
    if "anwg" in fieldnames:
        raise ValueError("v2 results must not use ambiguous column name 'anwg'")

    policies = sorted({r["policy_name"] for r in rows})
    by_primary: Dict[str, Dict[str, float]] = defaultdict(dict)
    row_index: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for r in rows:
        sid, p = r["scenario_id"], r["policy_name"]
        by_primary[sid][p] = _f(r[PRIMARY])
        row_index[(sid, p)] = r
    by_primary = dict(by_primary)
    meta = {sid: parse_scenario_id(sid) for sid in by_primary}

    scenario_rows: List[dict] = []
    for sid, scores in sorted(by_primary.items()):
        winners, best, second, margin = winner_margin(scores)
        m = meta[sid]
        feat = features.get(sid, {})
        entry = {
            "scenario_id": sid,
            **m,
            "token_length_source": feat.get("token_length_source", ""),
            "size_priority_alignment": feat.get("size_priority_alignment", ""),
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
    unique_wins = Counter(r["best_policy"] for r in scenario_rows if not r["exact_tie"])
    exact_tie_n = sum(1 for r in scenario_rows if r["exact_tie"])
    near_tie_counts = {
        str(eps): sum(1 for r in scenario_rows if r[f"near_tie_eps_{eps}"])
        for eps in EPSILONS
    }
    margins = [r["best_vs_second_margin"] for r in scenario_rows]
    headroom = {
        "mean": statistics.mean(margins) if margins else float("nan"),
        "median": statistics.median(margins) if margins else float("nan"),
        "frac_gt_0": (sum(1 for x in margins if x > 0) / n_scen) if n_scen else 0.0,
        "frac_gt_0.01": (
            sum(1 for x in margins if x > 0.01) / n_scen if n_scen else 0.0
        ),
    }

    def _cond_counts(key: str) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Counter] = defaultdict(Counter)
        for r in scenario_rows:
            label = str(r[key])
            if r["exact_tie"]:
                out[label]["EXACT_TIE"] += 1
            else:
                out[label][r["best_policy"]] += 1
        return {k: dict(v) for k, v in sorted(out.items())}

    winner_by_axis = {
        "target_utilization": _cond_counts("target_utilization"),
        "tenant_weight_skew": _cond_counts("tenant_weight_skew"),
        "favored_tenant_size": _cond_counts("favored_tenant_size"),
        "prediction_noise_sigma": _cond_counts("prediction_noise_sigma"),
        "seed": _cond_counts("seed"),
    }

    pair_rows: List[dict] = []
    pair_summary: List[dict] = []
    for a, b in combinations(policies, 2):
        deltas = []
        for sid, scores in by_primary.items():
            d = scores[a] - scores[b]
            deltas.append(d)
            pair_rows.append({"scenario_id": sid, **meta[sid], "policy_i": a, "policy_j": b, "delta_ij": d})
        summary: Dict[str, Any] = {
            "policy_i": a,
            "policy_j": b,
            "mean_delta_ij": statistics.mean(deltas),
            "median_delta_ij": statistics.median(deltas),
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

    # ESTF vs WFS oriented + by favored size
    estf, wfs = "estimated_service_time_first", "weighted_fair_share"
    estf_wfs_by_fav: Dict[str, Dict[str, Any]] = {}
    for fav in ("short", "long"):
        deltas = [
            by_primary[sid][estf] - by_primary[sid][wfs]
            for sid, m in meta.items()
            if m["favored_tenant_size"] == fav
        ]
        cell: Dict[str, Any] = {"favored_tenant_size": fav, "n": len(deltas)}
        for eps in EPSILONS:
            cell[f"estf_beats_wfs_eps_{eps}"] = sum(1 for d in deltas if d > eps)
            cell[f"wfs_beats_estf_eps_{eps}"] = sum(1 for d in deltas if d < -eps)
            cell[f"near_ties_eps_{eps}"] = sum(1 for d in deltas if abs(d) <= eps)
            cell[f"bidirectional_eps_{eps}"] = (
                cell[f"estf_beats_wfs_eps_{eps}"] > 0
                and cell[f"wfs_beats_estf_eps_{eps}"] > 0
            )
        estf_wfs_by_fav[fav] = cell

    # Seed stability over (util, skew, fav, noise)
    cells: Dict[Tuple[Any, ...], List[dict]] = defaultdict(list)
    for r in scenario_rows:
        key = (
            r["target_utilization"],
            r["tenant_weight_skew"],
            r["favored_tenant_size"],
            r["prediction_noise_sigma"],
        )
        cells[key].append(r)

    seed_rows: List[dict] = []
    agree_winner_set = agree_best = 0
    unstable_cells: List[dict] = []
    for key, group in sorted(cells.items()):
        util, skew, fav, noise = key
        g_sorted = sorted(group, key=lambda r: r["seed"])
        sets = [tuple(sorted(r["exact_winners"].split("|"))) for r in g_sorted]
        bests = [r["best_policy"] for r in g_sorted]
        winner_set_agree = len(set(sets)) == 1
        best_agree = len(set(bests)) == 1
        if winner_set_agree:
            agree_winner_set += 1
        if best_agree:
            agree_best += 1
        signs = []
        for r in g_sorted:
            d = r[f"score__{estf}"] - r[f"score__{wfs}"]
            signs.append(0 if abs(d) <= 1e-12 else (1 if d > 0 else -1))
        row = {
            "target_utilization": util,
            "tenant_weight_skew": skew,
            "favored_tenant_size": fav,
            "prediction_noise_sigma": noise,
            "n_seeds_observed": len(group),
            "winner_set_agree": winner_set_agree,
            "best_policy_agree": best_agree,
            "estf_wfs_sign_agree": len(set(signs)) == 1,
            "winner_sets": ";".join("|".join(s) for s in sets),
            "best_policies": ";".join(bests),
        }
        seed_rows.append(row)
        if not (winner_set_agree and best_agree and row["estf_wfs_sign_agree"]):
            unstable_cells.append(row)

    n_cells = len(cells)

    # Policy fairness aggregates
    policy_fairness: List[dict] = []
    for p in policies:
        fav_rates, oth_rates, jfis, ttfts, primaries, unweighted = [], [], [], [], [], []
        perfect_anwg = 0
        for sid in by_primary:
            r = row_index[(sid, p)]
            fav_tot = max(1.0, _f(r["favored_total"]))
            oth_tot = max(1.0, _f(r["other_total"]))
            fav_rates.append(_f(r["favored_violations"]) / fav_tot)
            oth_rates.append(_f(r["other_violations"]) / oth_tot)
            jfis.append(_f(r["jains_fairness_index"]))
            ttfts.append(_f(r["mean_ttft"]))
            prim = _f(r[PRIMARY])
            primaries.append(prim)
            unweighted.append(_f(r["unweighted_slo_success_rate"]))
            if abs(prim - 1.0) <= 1e-12:
                perfect_anwg += 1
        policy_fairness.append(
            {
                "policy_name": p,
                "mean_canonical_anwg": statistics.mean(primaries),
                "median_canonical_anwg": statistics.median(primaries),
                "mean_unweighted_slo_success": statistics.mean(unweighted),
                "mean_favored_violation_rate": statistics.mean(fav_rates),
                "mean_other_violation_rate": statistics.mean(oth_rates),
                "mean_jains_fairness_index": statistics.mean(jfis),
                "mean_ttft": statistics.mean(ttfts),
                "n_perfect_anwg": perfect_anwg,
                "frac_perfect_anwg": perfect_anwg / n_scen if n_scen else 0.0,
                "scenarios_with_favored_violations": sum(1 for x in fav_rates if x > 0),
                "scenarios_with_other_violations": sum(1 for x in oth_rates if x > 0),
                "n_scenarios": n_scen,
            }
        )

    # Surfaces
    fairness_surfaces: List[dict] = []
    for p in policies:
        for axis in (
            "target_utilization",
            "tenant_weight_skew",
            "favored_tenant_size",
            "prediction_noise_sigma",
        ):
            buckets: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
            for sid, m in meta.items():
                buckets[m[axis]].append(row_index[(sid, p)])
            for val, group in sorted(buckets.items(), key=lambda kv: str(kv[0])):
                fairness_surfaces.append(
                    {
                        "policy_name": p,
                        "axis": axis,
                        "axis_value": val,
                        "n": len(group),
                        "mean_canonical_anwg": statistics.mean(
                            [_f(r[PRIMARY]) for r in group]
                        ),
                        "mean_favored_violation_rate": statistics.mean(
                            [
                                _f(r["favored_violations"])
                                / max(1.0, _f(r["favored_total"]))
                                for r in group
                            ]
                        ),
                        "mean_other_violation_rate": statistics.mean(
                            [
                                _f(r["other_violations"])
                                / max(1.0, _f(r["other_total"]))
                                for r in group
                            ]
                        ),
                        "mean_jains_fairness_index": statistics.mean(
                            [_f(r["jains_fairness_index"]) for r in group]
                        ),
                    }
                )

    # WFS skew response under conflict (favored=long)
    wfs_skew_conflict: List[dict] = []
    for skew in sorted({m["tenant_weight_skew"] for m in meta.values()}):
        sids = [
            sid
            for sid, m in meta.items()
            if m["favored_tenant_size"] == "long" and m["tenant_weight_skew"] == skew
        ]
        fav_v = [
            _f(row_index[(sid, wfs)]["favored_violations"])
            / max(1.0, _f(row_index[(sid, wfs)]["favored_total"]))
            for sid in sids
        ]
        anwgs = [_f(row_index[(sid, wfs)][PRIMARY]) for sid in sids]
        wfs_skew_conflict.append(
            {
                "tenant_weight_skew": skew,
                "n": len(sids),
                "mean_wfs_canonical_anwg": statistics.mean(anwgs) if anwgs else float("nan"),
                "mean_wfs_favored_violation_rate": (
                    statistics.mean(fav_v) if fav_v else float("nan")
                ),
            }
        )

    aging_perfect_n = next(
        p["n_perfect_anwg"] for p in policy_fairness if p["policy_name"] == "aging_priority"
    )

    # Orient ESTF-WFS pair summary
    raw_pair = next(
        s
        for s in pair_summary
        if {s["policy_i"], s["policy_j"]} == {estf, wfs}
    )
    if raw_pair["policy_i"] == estf:
        estf_wfs = raw_pair
    else:
        estf_wfs = {
            **raw_pair,
            "policy_i": estf,
            "policy_j": wfs,
            "mean_delta_ij": -raw_pair["mean_delta_ij"],
            "median_delta_ij": -raw_pair["median_delta_ij"],
        }
        for eps in EPSILONS:
            estf_wfs[f"i_beats_j_eps_{eps}"] = raw_pair[f"j_beats_i_eps_{eps}"]
            estf_wfs[f"j_beats_i_eps_{eps}"] = raw_pair[f"i_beats_j_eps_{eps}"]

    utils = sorted({m["target_utilization"] for m in meta.values()})
    skews = sorted({m["tenant_weight_skew"] for m in meta.values()})
    favs = sorted({m["favored_tenant_size"] for m in meta.values()})
    noises = sorted({m["prediction_noise_sigma"] for m in meta.values()})
    seeds = sorted({m["seed"] for m in meta.values()})
    token_sources = Counter(
        (features.get(sid, {}) or {}).get("token_length_source", "unknown")
        for sid in by_primary
    )

    integrity = {
        "n_rows": len(rows),
        "n_scenarios": n_scen,
        "n_policies": len(policies),
        "policies": policies,
        "duplicate_scenario_policy_keys": len(rows)
        - len({(r["scenario_id"], r["policy_name"]) for r in rows}),
        "failed_rows": sum(1 for r in rows if r.get("status") != "success"),
        "primary_column": PRIMARY,
        "has_ambiguous_anwg_column": False,
        "expected_policies_present": all(p in policies for p in POLICIES_EXPECTED),
        "grid": {
            "target_utilization": utils,
            "tenant_weight_skew": skews,
            "favored_tenant_size": favs,
            "prediction_noise_sigma": noises,
            "seeds": seeds,
            "product": len(utils) * len(skews) * len(favs) * len(noises) * len(seeds),
        },
        "token_length_sources": dict(token_sources),
        "burstgpt_only": set(token_sources) == {"burstgpt_staged"},
    }
    integrity["grid_product_matches_n_scenarios"] = (
        integrity["grid"]["product"] == n_scen
    )

    summary = {
        "integrity": integrity,
        "unique_winner_counts": dict(unique_wins),
        "exact_tie_count": exact_tie_n,
        "exact_tie_rate": exact_tie_n / n_scen if n_scen else 0.0,
        "near_tie_counts": near_tie_counts,
        "near_tie_rates": {k: v / n_scen for k, v in near_tie_counts.items()},
        "headroom": headroom,
        "winner_entropy_bits": shannon_entropy(unique_wins),
        "winner_by_axis": winner_by_axis,
        "pairwise_summary": pair_summary,
        "estf_vs_wfs": estf_wfs,
        "estf_vs_wfs_by_favored_size": estf_wfs_by_fav,
        "seed_stability": {
            "n_cells": n_cells,
            "winner_set_agree_frac": agree_winner_set / n_cells if n_cells else 0.0,
            "best_policy_agree_frac": agree_best / n_cells if n_cells else 0.0,
            "n_unstable_cells": len(unstable_cells),
        },
        "policy_fairness_overall": policy_fairness,
        "wfs_skew_response_conflict_long": wfs_skew_conflict,
        "aging_perfect_anwg_count": aging_perfect_n,
        "aging_perfect_anwg_rate": aging_perfect_n / n_scen if n_scen else 0.0,
    }

    return {
        "summary": summary,
        "per_scenario_winners": scenario_rows,
        "pairwise_deltas": pair_rows,
        "pairwise_summary": pair_summary,
        "seed_stability": seed_rows,
        "unstable_cells": unstable_cells,
        "policy_fairness_overall": policy_fairness,
        "fairness_surfaces": fairness_surfaces,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_artifacts(out_dir: Path, result: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "per_scenario_winners.csv", result["per_scenario_winners"])
    _write_csv(out_dir / "pairwise_deltas.csv", result["pairwise_deltas"])
    _write_csv(out_dir / "pairwise_summary.csv", result["pairwise_summary"])
    _write_csv(out_dir / "seed_stability.csv", result["seed_stability"])
    _write_csv(out_dir / "unstable_cells.csv", result["unstable_cells"])
    _write_csv(out_dir / "policy_fairness_overall.csv", result["policy_fairness_overall"])
    _write_csv(out_dir / "fairness_surfaces.csv", result["fairness_surfaces"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/analysis",
    )
    args = ap.parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "analysis")
    csv_path = run_dir / "per_policy_results.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    features = load_features(run_dir / "scenario_features.csv")
    result = analyze(rows, features=features)
    write_artifacts(out_dir, result)
    s = result["summary"]
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "n_scenarios": s["integrity"]["n_scenarios"],
            "exact_tie_rate": s["exact_tie_rate"],
            "near_tie_rate_eps_0.01": s["near_tie_rates"]["0.01"],
            "unique_winners": s["unique_winner_counts"],
            "estf_vs_wfs_eps_0.01": {
                "estf_beats": s["estf_vs_wfs"]["i_beats_j_eps_0.01"],
                "wfs_beats": s["estf_vs_wfs"]["j_beats_i_eps_0.01"],
                "bidirectional": s["estf_vs_wfs"]["bidirectional_eps_0.01"],
            },
            "aging_perfect_rate": s["aging_perfect_anwg_rate"],
            "burstgpt_only": s["integrity"]["burstgpt_only"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
