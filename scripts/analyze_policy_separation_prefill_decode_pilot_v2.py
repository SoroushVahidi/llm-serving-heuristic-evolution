#!/usr/bin/env python3
"""Analyze Policy Separation Family B v2 TTFT-contention results.

Primary metric: canonical ``arrival_normalized_weighted_goodput``.
Scenario IDs::

    pd2.hog{12|24}.late{12|40}.slo{hog_ttft|late_ttft}.s{SEED}

Thresholds are preregistered in
docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md and must not be
changed after observing a full run. This script does not rewrite
``per_policy_results.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

SCENARIO_ID_RE = re.compile(
    r"^pd2\.hog(?P<n_hog>\d+)\.late(?P<n_late>\d+)"
    r"\.slo(?P<slo_emphasis>hog_ttft|late_ttft)\.s(?P<seed>\d+)$"
)

PRIMARY = "arrival_normalized_weighted_goodput"
PRACTICAL_EPS = 0.01
POLICIES = ("full_prefill", "chunked_prefill_small")
HELD_OUT_SEED = 20260823

THRESHOLDS = {
    "practical_eps": PRACTICAL_EPS,
    "h1_unique_wins": 8,
    "h1_seeds_with_win": 3,
    "h2_unique_wins": 8,
    "h3_observable_accuracy": 0.80,
    "h4_seeds_with_reverse": 2,
    "h4_sign_agree": 0.75,
    "h5_exact_tie_max": 0.25,
    "h5_near_tie_max": 0.35,
    "h6_mechanism_frac": 0.80,
    "h8_heldout_sign_match": 0.75,
    "h9_min_losses": 4,
    "g1_min_each_direction": 8,
    "g3_seed_agree": 0.75,
    "g4_near_tie_max": 0.35,
    "g5_mean_abs_delta": 0.02,
    "g9_exact_match_max": 0.10,
}


def parse_scenario_id(scenario_id: str) -> Dict[str, Any]:
    m = SCENARIO_ID_RE.fullmatch(scenario_id)
    if not m:
        raise ValueError(f"unrecognized Family B v2 scenario_id: {scenario_id!r}")
    d = m.groupdict()
    return {
        "n_hog": int(d["n_hog"]),
        "n_late": int(d["n_late"]),
        "slo_emphasis": d["slo_emphasis"],
        "seed": int(d["seed"]),
        "pair_id": f"pd2.hog{d['n_hog']}.late{d['n_late']}.slo{d['slo_emphasis']}",
    }


def _f(x: Any) -> float:
    return float(x)


def winner_margin(scores: Mapping[str, float]) -> Tuple[List[str], str, str, float]:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_name, best = ordered[0]
    second_name, second = ordered[1] if len(ordered) > 1 else (best_name, best)
    winners = [n for n, v in ordered if abs(v - best) <= 1e-15]
    return winners, best_name, second_name, best - second


def near_tie(scores: Mapping[str, float], eps: float) -> bool:
    vals = list(scores.values())
    return (max(vals) - min(vals)) <= eps


def load_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_features(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    with open(path, newline="") as f:
        return {r["scenario_id"]: r for r in csv.DictReader(f)}


def group_by_scenario(rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    out: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for r in rows:
        out[r["scenario_id"]][r["policy_name"]] = dict(r)
    return out


def smoke_gate(grouped: Mapping[str, Mapping[str, Mapping[str, str]]]) -> Dict[str, Any]:
    full_wins = 0
    small_wins = 0
    n = 0
    policies_seen = set()
    for sid, prow in grouped.items():
        policies_seen.update(prow)
        if set(POLICIES) - set(prow):
            continue
        n += 1
        d = _f(prow["full_prefill"][PRIMARY]) - _f(prow["chunked_prefill_small"][PRIMARY])
        if d > PRACTICAL_EPS:
            full_wins += 1
        elif d < -PRACTICAL_EPS:
            small_wins += 1
    s2 = full_wins >= 1
    s3 = small_wins >= 1
    s4 = policies_seen == set(POLICIES)
    go = bool(s2 and s3 and s4 and n > 0)
    return {
        "n_comparable_cells": n,
        "full_practical_wins": full_wins,
        "small_practical_wins": small_wins,
        "policies_seen": sorted(policies_seen),
        "S2_full_win": s2,
        "S3_small_win": s3,
        "S4_two_anchors_only": s4,
        "verdict": "SMOKE_GO" if go else "FAMILY_B_REFINEMENT_NO_GO",
    }


def analyze(
    rows: Sequence[Mapping[str, str]],
    features: Optional[Mapping[str, Mapping[str, str]]] = None,
    *,
    held_out_seed: int = HELD_OUT_SEED,
) -> Dict[str, Any]:
    features = features or {}
    success = [r for r in rows if r.get("status", "success") == "success"]
    failed = [r for r in rows if r.get("status") == "failed"]
    grouped = group_by_scenario(success)

    integrity = {
        "n_result_rows": len(rows),
        "n_success": len(success),
        "n_failed": len(failed),
        "n_scenarios": len(grouped),
        "duplicate_pairs": 0,
        "nan_primary": 0,
        "policies": sorted({r["policy_name"] for r in success}),
    }
    seen_pairs = set()
    for r in success:
        key = (r["scenario_id"], r["policy_name"])
        if key in seen_pairs:
            integrity["duplicate_pairs"] += 1
        seen_pairs.add(key)
        val = r.get(PRIMARY, "")
        try:
            fv = float(val)
            if not math.isfinite(fv):
                integrity["nan_primary"] += 1
        except (TypeError, ValueError):
            integrity["nan_primary"] += 1

    confound = (
        integrity["duplicate_pairs"] > 0
        or integrity["nan_primary"] > 0
        or integrity["n_failed"] > 0
        or set(integrity["policies"]) != set(POLICIES)
    )

    cells = []
    unique_counts: Counter[str] = Counter()
    exact_ties = 0
    near_ties = 0
    full_beats = 0
    small_beats = 0
    abs_deltas = []
    exact_matches = 0
    hog_ttft_ok_when_full = []
    late_ttft_ok_when_small = []
    h3_correct = []
    by_seed: Dict[int, Dict[str, str]] = defaultdict(dict)
    sign_by_pair: Dict[str, Dict[int, int]] = defaultdict(dict)

    for sid, prow in sorted(grouped.items()):
        meta = parse_scenario_id(sid)
        if set(POLICIES) - set(prow):
            continue
        scores = {p: _f(prow[p][PRIMARY]) for p in POLICIES}
        winners, best, second, margin = winner_margin(scores)
        delta = scores["full_prefill"] - scores["chunked_prefill_small"]
        abs_deltas.append(abs(delta))
        is_exact = abs(delta) <= 1e-15
        is_near = abs(delta) <= PRACTICAL_EPS
        if is_exact:
            exact_ties += 1
            exact_matches += 1
        if is_near:
            near_ties += 1
        if delta > PRACTICAL_EPS:
            full_beats += 1
            unique_counts["full_prefill"] += 1
        elif delta < -PRACTICAL_EPS:
            small_beats += 1
            unique_counts["chunked_prefill_small"] += 1

        hog_full = _f(prow["full_prefill"].get("hog_mean_ttft", "nan"))
        hog_small = _f(prow["chunked_prefill_small"].get("hog_mean_ttft", "nan"))
        late_full = _f(prow["full_prefill"].get("late_mean_ttft", "nan"))
        late_small = _f(prow["chunked_prefill_small"].get("late_mean_ttft", "nan"))
        if delta > PRACTICAL_EPS and math.isfinite(hog_full) and math.isfinite(hog_small):
            hog_ttft_ok_when_full.append(hog_full < hog_small)
        if delta < -PRACTICAL_EPS and math.isfinite(late_full) and math.isfinite(late_small):
            late_ttft_ok_when_small.append(late_small < late_full)

        feat = features.get(sid, {})
        slack_hog = feat.get("mean_e2e_slack_hog")
        slack_late = feat.get("mean_e2e_slack_late")
        if slack_hog not in (None, "") and slack_late not in (None, ""):
            pred = math.copysign(1.0, _f(slack_late) - _f(slack_hog))
            # Tight hog slack (hog < late) → predict full wins (pred > 0).
            if not is_near:
                obs = math.copysign(1.0, delta) if delta != 0 else 0.0
                h3_correct.append(pred == obs)

        sign = 0 if is_near else (1 if delta > 0 else -1)
        by_seed[meta["seed"]][meta["pair_id"]] = (
            frozenset(winners) if is_near else frozenset([best])
        )
        sign_by_pair[meta["pair_id"]][meta["seed"]] = 0 if is_near else (1 if delta > 0 else -1)

        cells.append(
            {
                "scenario_id": sid,
                **meta,
                "anwg_full": scores["full_prefill"],
                "anwg_small": scores["chunked_prefill_small"],
                "delta_full_minus_small": delta,
                "winners": winners,
                "best": best,
                "margin": margin,
                "near_tie": is_near,
                "hog_mean_ttft_full": hog_full,
                "hog_mean_ttft_small": hog_small,
                "late_mean_ttft_full": late_full,
                "late_mean_ttft_small": late_small,
                "decode_stalled_full": _f(
                    prow["full_prefill"].get("decode_stalled_steps", 0)
                ),
                "decode_stalled_small": _f(
                    prow["chunked_prefill_small"].get("decode_stalled_steps", 0)
                ),
                "prefill_stalled_full": _f(
                    prow["full_prefill"].get("prefill_stalled_steps", 0)
                ),
                "prefill_stalled_small": _f(
                    prow["chunked_prefill_small"].get("prefill_stalled_steps", 0)
                ),
                "hog_slo_full": _f(prow["full_prefill"].get("hog_slo_success", "nan")),
                "hog_slo_small": _f(
                    prow["chunked_prefill_small"].get("hog_slo_success", "nan")
                ),
                "late_slo_full": _f(prow["full_prefill"].get("late_slo_success", "nan")),
                "late_slo_small": _f(
                    prow["chunked_prefill_small"].get("late_slo_success", "nan")
                ),
            }
        )

    n = len(cells)
    seeds_full = {
        c["seed"] for c in cells if c["delta_full_minus_small"] > PRACTICAL_EPS
    }
    seeds_small = {
        c["seed"] for c in cells if c["delta_full_minus_small"] < -PRACTICAL_EPS
    }

    # Seed winner-set agreement over pair_ids present in all seeds.
    pair_ids = sorted({c["pair_id"] for c in cells})
    seed_list = sorted({c["seed"] for c in cells})
    agree = 0
    n_pair_cells = 0
    sign_agree = 0
    for pid in pair_ids:
        sets = [by_seed[s].get(pid) for s in seed_list]
        if any(x is None for x in sets):
            continue
        n_pair_cells += 1
        if len({frozenset(x) for x in sets}) == 1:
            agree += 1
        signs = [sign_by_pair[pid].get(s) for s in seed_list]
        if len(set(signs)) == 1:
            sign_agree += 1
    seed_agree = agree / n_pair_cells if n_pair_cells else 0.0
    sign_agree_frac = sign_agree / n_pair_cells if n_pair_cells else 0.0

    h3_acc = (sum(h3_correct) / len(h3_correct)) if h3_correct else 0.0
    h6_full = (
        sum(hog_ttft_ok_when_full) / len(hog_ttft_ok_when_full)
        if hog_ttft_ok_when_full
        else 0.0
    )
    h6_small = (
        sum(late_ttft_ok_when_small) / len(late_ttft_ok_when_small)
        if late_ttft_ok_when_small
        else 0.0
    )

    # H8 held-out
    train_seeds = [s for s in seed_list if s != held_out_seed]
    held_cells = [c for c in cells if c["seed"] == held_out_seed]
    held_full = sum(1 for c in held_cells if c["delta_full_minus_small"] > PRACTICAL_EPS)
    held_small = sum(1 for c in held_cells if c["delta_full_minus_small"] < -PRACTICAL_EPS)
    held_match = []
    for pid in pair_ids:
        if held_out_seed not in sign_by_pair[pid]:
            continue
        train_signs = [sign_by_pair[pid][s] for s in train_seeds if s in sign_by_pair[pid]]
        if not train_signs:
            continue
        majority = Counter(train_signs).most_common(1)[0][0]
        held_match.append(sign_by_pair[pid][held_out_seed] == majority)
    h8_match = (sum(held_match) / len(held_match)) if held_match else 0.0

    exact_tie_rate = exact_ties / n if n else 1.0
    near_tie_rate = near_ties / n if n else 1.0
    mean_abs = float(statistics.mean(abs_deltas)) if abs_deltas else 0.0
    exact_match_rate = exact_matches / n if n else 1.0

    def _v(ok: bool) -> str:
        return "CONFIRM" if ok else "CONTRADICT"

    h1 = unique_counts["full_prefill"] >= THRESHOLDS["h1_unique_wins"] and len(
        seeds_full
    ) >= THRESHOLDS["h1_seeds_with_win"]
    h2 = unique_counts["chunked_prefill_small"] >= THRESHOLDS["h2_unique_wins"] and len(
        seeds_small
    ) >= THRESHOLDS["h1_seeds_with_win"]
    h3 = h3_acc >= THRESHOLDS["h3_observable_accuracy"] and len(h3_correct) >= 4
    h4 = len(seeds_small) >= THRESHOLDS["h4_seeds_with_reverse"] and sign_agree_frac >= THRESHOLDS[
        "h4_sign_agree"
    ]
    h5 = (
        exact_tie_rate <= THRESHOLDS["h5_exact_tie_max"]
        and near_tie_rate <= THRESHOLDS["h5_near_tie_max"]
    )
    h6 = h6_full >= THRESHOLDS["h6_mechanism_frac"] and h6_small >= THRESHOLDS[
        "h6_mechanism_frac"
    ]
    h8 = held_full >= 1 and held_small >= 1 and h8_match >= THRESHOLDS["h8_heldout_sign_match"]
    h9 = (
        unique_counts["chunked_prefill_small"] >= THRESHOLDS["h9_min_losses"]
        and unique_counts["full_prefill"] >= THRESHOLDS["h9_min_losses"]
    )

    g = {
        "G1": full_beats >= THRESHOLDS["g1_min_each_direction"]
        and small_beats >= THRESHOLDS["g1_min_each_direction"],
        "G2": unique_counts["full_prefill"] >= 8
        and unique_counts["chunked_prefill_small"] >= 8,
        "G3": seed_agree >= THRESHOLDS["g3_seed_agree"],
        "G4": near_tie_rate <= THRESHOLDS["g4_near_tie_max"],
        "G5": mean_abs >= THRESHOLDS["g5_mean_abs_delta"],
        "G6": h6,
        "G7": h3,
        "G8": h8,
        "G9": exact_match_rate < THRESHOLDS["g9_exact_match_max"],
        "G10": set(integrity["policies"]) == set(POLICIES),
    }
    gate_all = all(g.values()) and not confound

    if confound:
        family_verdict = "DESIGN_CONFOUND"
    elif gate_all:
        family_verdict = "FAMILY_B_COMPOSITION_READY"
    elif g["G1"]:
        family_verdict = "USEFUL_BUT_NEEDS_REFINEMENT"
    else:
        family_verdict = "FAMILY_B_REFINEMENT_NO_GO"

    hypotheses = [
        {
            "id": "H1",
            "verdict": _v(h1),
            "detail": {
                "unique_wins_full": unique_counts["full_prefill"],
                "seeds_with_full_win": sorted(seeds_full),
            },
        },
        {
            "id": "H2",
            "verdict": _v(h2),
            "detail": {
                "unique_wins_small": unique_counts["chunked_prefill_small"],
                "seeds_with_small_win": sorted(seeds_small),
            },
        },
        {
            "id": "H3",
            "verdict": _v(h3),
            "detail": {"accuracy": h3_acc, "n_non_neartie": len(h3_correct)},
        },
        {
            "id": "H4",
            "verdict": _v(h4),
            "detail": {
                "seeds_with_reverse": sorted(seeds_small),
                "sign_agree": sign_agree_frac,
            },
        },
        {
            "id": "H5",
            "verdict": _v(h5),
            "detail": {
                "exact_tie_rate": exact_tie_rate,
                "near_tie_rate": near_tie_rate,
            },
        },
        {
            "id": "H6",
            "verdict": _v(h6),
            "detail": {
                "frac_full_win_hog_ttft_lower": h6_full,
                "frac_small_win_late_ttft_lower": h6_small,
                "n_full_win": len(hog_ttft_ok_when_full),
                "n_small_win": len(late_ttft_ok_when_small),
            },
        },
        {
            "id": "H7",
            "verdict": "NOT_APPLICABLE",
            "detail": "third decode-priority policy not retained",
        },
        {
            "id": "H8",
            "verdict": _v(h8),
            "detail": {
                "held_out_seed": held_out_seed,
                "held_full_wins": held_full,
                "held_small_wins": held_small,
                "sign_match_frac": h8_match,
            },
        },
        {
            "id": "H9",
            "verdict": _v(h9),
            "detail": {
                "full_unique_wins": unique_counts["full_prefill"],
                "small_unique_wins": unique_counts["chunked_prefill_small"],
            },
        },
        {
            "id": "H10",
            "verdict": _v(gate_all),
            "detail": g,
        },
    ]

    pairwise = {
        "full_beats_small_eps_0.01": full_beats,
        "small_beats_full_eps_0.01": small_beats,
        "near_ties_eps_0.01": near_ties,
        "exact_ties": exact_ties,
        "bidirectional": full_beats >= 1 and small_beats >= 1,
        "mean_abs_delta": mean_abs,
        "mean_delta_full_minus_small": (
            float(statistics.mean(c["delta_full_minus_small"] for c in cells))
            if cells
            else 0.0
        ),
    }

    decode_stall_zero = all(
        c["decode_stalled_full"] == 0 and c["decode_stalled_small"] == 0 for c in cells
    )

    return {
        "thresholds": THRESHOLDS,
        "integrity": integrity,
        "n_comparable_cells": n,
        "family_b_verdict": family_verdict,
        "composition_gate": g,
        "hypotheses": hypotheses,
        "unique_wins_eps_0.01": dict(unique_counts),
        "exact_tie_rate": exact_tie_rate,
        "near_tie_rate_eps_0.01": near_tie_rate,
        "exact_match_rate": exact_match_rate,
        "seed_winner_set_agree": seed_agree,
        "seed_sign_agree": sign_agree_frac,
        "pairwise": pairwise,
        "decode_stalled_steps_identically_zero": decode_stall_zero,
        "cells": cells,
        "smoke": smoke_gate(grouped),
        "confound": confound,
    }


def write_artifacts(summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in summary.items() if k != "cells"}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(slim, f, indent=2)
    cells = summary.get("cells") or []
    if cells:
        fieldnames = list(cells[0].keys())
        with open(out_dir / "per_cell.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in cells:
                out = dict(row)
                out["winners"] = "|".join(out["winners"])
                w.writerow(out)
    with open(out_dir / "hypotheses.json", "w") as f:
        json.dump(summary["hypotheses"], f, indent=2)
    with open(out_dir / "composition_gate.json", "w") as f:
        json.dump(
            {
                "verdict": summary["family_b_verdict"],
                "gate": summary["composition_gate"],
                "pairwise": summary["pairwise"],
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-gate", action="store_true")
    parser.add_argument("--held-out-seed", type=int, default=HELD_OUT_SEED)
    args = parser.parse_args()
    csv_path = args.run_dir / "per_policy_results.csv"
    feat_path = args.run_dir / "scenario_features.csv"
    rows = load_rows(csv_path)
    features = load_features(feat_path)
    grouped = group_by_scenario([r for r in rows if r.get("status", "success") == "success"])
    if args.smoke_gate:
        gate = smoke_gate(grouped)
        out = args.run_dir / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "smoke_gate.json", "w") as f:
            json.dump(gate, f, indent=2)
        print(json.dumps(gate, indent=2))
        if gate["verdict"] != "SMOKE_GO":
            raise SystemExit(1)
        return
    summary = analyze(rows, features, held_out_seed=args.held_out_seed)
    write_artifacts(summary, args.run_dir / "analysis")
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
