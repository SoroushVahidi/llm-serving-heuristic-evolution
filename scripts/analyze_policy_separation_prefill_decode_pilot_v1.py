#!/usr/bin/env python3
"""Analyze Policy Separation Family B v1 prefill/decode chunk-control results.

Primary metric: canonical ``arrival_normalized_weighted_goodput``.
Scenario IDs::

    pd1.psize{short|medium|long|mixed}.occ{low|medium|high}
        .slo{ttft_tight|tbt_tight|balanced}.load{moderate|high}.s{SEED}

This script computes integrity checks, winner structure, pairwise Δ, factor
surfaces, mechanism-metric aggregates, seed stability, and preregistered
H1–H10 scores. It does not rewrite ``per_policy_results.csv``.
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
    "experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z"
)

SCENARIO_ID_RE = re.compile(
    r"^pd1\.psize(?P<prefill_size_class>short|medium|long|mixed)"
    r"\.occ(?P<decode_occupancy>low|medium|high)"
    r"\.slo(?P<slo_regime>ttft_tight|tbt_tight|balanced)"
    r"\.load(?P<offered_load>moderate|high)"
    r"\.s(?P<seed>\d+)$"
)

PRIMARY = "arrival_normalized_weighted_goodput"
EPSILONS = (0.0, 0.001, 0.005, 0.01)
PRACTICAL_EPS = 0.01

STRUCTURAL_POLICIES = (
    "full_prefill",
    "chunked_prefill_small",
    "chunked_prefill_large",
    "decode_priority_chunked",
)
DIAGNOSTIC_POLICY = "adaptive_prefill_control"
POLICIES_EXPECTED = STRUCTURAL_POLICIES + (DIAGNOSTIC_POLICY,)

IMPORTANT_PAIRS = (
    ("full_prefill", "chunked_prefill_small"),
    ("full_prefill", "chunked_prefill_large"),
    ("full_prefill", "decode_priority_chunked"),
    ("chunked_prefill_small", "decode_priority_chunked"),
    ("chunked_prefill_large", "decode_priority_chunked"),
    ("chunked_prefill_small", "chunked_prefill_large"),
)

MECHANISM_METRICS = (
    "mean_ttft",
    "p95_ttft",
    "p99_ttft",
    "mean_tpot",
    "p95_tpot",
    "mean_queuing_delay",
    "mean_prefill_delay_s",
    "ttft_attainment",
    "tbt_attainment",
    "request_throughput",
    "token_throughput",
    "completion_fraction",
    "decode_stalled_steps",
    "cumulative_decode_tokens_deferred",
    "steps_with_prefill_while_decode_deferred",
    "prefill_stalled_steps",
    "cumulative_prefill_requests_stalled",
    "budget_saturation_fraction",
    "mean_num_decoding",
    "mean_num_prefilling",
    "fraction_prefill_tokens_while_decodes_active",
    PRIMARY,
    "unweighted_slo_success_rate",
)

FACTORS = (
    "prefill_size_class",
    "decode_occupancy",
    "slo_regime",
    "offered_load",
    "seed",
)


def parse_scenario_id(scenario_id: str) -> Dict[str, Any]:
    m = SCENARIO_ID_RE.fullmatch(scenario_id)
    if m is None:
        raise ValueError(f"Unrecognized Family B v1 scenario_id: {scenario_id!r}")
    return {
        "prefill_size_class": m.group("prefill_size_class"),
        "decode_occupancy": m.group("decode_occupancy"),
        "slo_regime": m.group("slo_regime"),
        "offered_load": m.group("offered_load"),
        "seed": int(m.group("seed")),
    }


def _f(x: Any) -> float:
    return float(x)


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


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


def unique_winner_at_eps(scores: Mapping[str, float], eps: float) -> Optional[str]:
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_name, best_score = ranked[0]
    if len(ranked) == 1:
        return best_name
    second_score = ranked[1][1]
    if best_score - second_score > eps + 1e-15:
        return best_name
    return None


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
    with path.open(newline="", encoding="utf-8") as f:
        return {r["scenario_id"]: r for r in csv.DictReader(f)}


def _burstgpt_ok(token_sources_raw: str) -> Tuple[bool, Dict[str, str]]:
    if not token_sources_raw:
        return False, {}
    try:
        parsed = json.loads(token_sources_raw)
    except json.JSONDecodeError:
        return False, {}
    if not isinstance(parsed, dict):
        return False, {}
    kinds = {
        k: str(v)
        for k, v in parsed.items()
        if k in ("prefill_prompt", "prefill_output", "decode_prompt", "decode_output")
    }
    allowed = {"burstgpt_staged", "burstgpt_anchored"}
    ok = bool(kinds) and all(v in allowed for v in kinds.values())
    return ok, kinds


def _mean(xs: Sequence[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _subset(
    scenario_rows: Sequence[Mapping[str, Any]], **filters: Any
) -> List[Mapping[str, Any]]:
    out = []
    for r in scenario_rows:
        if all(str(r.get(k)) == str(v) for k, v in filters.items()):
            out.append(r)
    return out


def _pairwise_block(
    by_primary: Mapping[str, Mapping[str, float]],
    meta: Mapping[str, Mapping[str, Any]],
    a: str,
    b: str,
    sids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    if sids is None:
        sids = list(by_primary)
    deltas = []
    regime_i: Dict[str, Counter] = {k: Counter() for k in FACTORS}
    regime_j: Dict[str, Counter] = {k: Counter() for k in FACTORS}
    for sid in sids:
        scores = by_primary[sid]
        d = scores[a] - scores[b]
        deltas.append(d)
        m = meta[sid]
        if d > PRACTICAL_EPS:
            for k in FACTORS:
                regime_i[k][str(m[k])] += 1
        elif d < -PRACTICAL_EPS:
            for k in FACTORS:
                regime_j[k][str(m[k])] += 1
    summary: Dict[str, Any] = {
        "policy_i": a,
        "policy_j": b,
        "n": len(deltas),
        "mean_delta_ij": _mean(deltas),
        "median_delta_ij": _median(deltas),
        "mean_abs_delta": _mean([abs(d) for d in deltas]) if deltas else float("nan"),
        "i_beats_regimes_eps_0.01": {k: dict(v) for k, v in regime_i.items()},
        "j_beats_regimes_eps_0.01": {k: dict(v) for k, v in regime_j.items()},
    }
    for eps in EPSILONS:
        i_beats = sum(1 for d in deltas if d > eps)
        j_beats = sum(1 for d in deltas if d < -eps)
        ties = sum(1 for d in deltas if abs(d) <= eps)
        summary[f"i_beats_j_eps_{eps}"] = i_beats
        summary[f"j_beats_i_eps_{eps}"] = j_beats
        summary[f"near_ties_eps_{eps}"] = ties
        summary[f"bidirectional_eps_{eps}"] = i_beats > 0 and j_beats > 0
        summary[f"practical_margin_i_gt_{eps}"] = i_beats
        summary[f"practical_margin_j_gt_{eps}"] = j_beats
    return summary


def _surface(
    scenario_rows: Sequence[Mapping[str, Any]],
    axis_a: str,
    axis_b: str,
    policies: Sequence[str],
) -> List[dict]:
    buckets: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in scenario_rows:
        buckets[(str(r[axis_a]), str(r[axis_b]))].append(r)
    out: List[dict] = []
    for (va, vb), group in sorted(buckets.items()):
        wins = Counter(
            r["best_policy_structural"]
            for r in group
            if r.get("unique_structural_eps_0.01")
        )
        exact_wins = Counter(
            r["best_policy_structural"]
            for r in group
            if not r["exact_tie_structural"]
        )
        dominant = None
        if wins:
            dominant = wins.most_common(1)[0][0]
        elif exact_wins:
            dominant = exact_wins.most_common(1)[0][0]
        row: Dict[str, Any] = {
            axis_a: va,
            axis_b: vb,
            "n": len(group),
            "dominant_unique_eps_0.01": dominant,
            "unique_winner_counts_eps_0.01": dict(wins),
            "exact_unique_winner_counts": dict(exact_wins),
            "n_unique_eps_0.01": sum(wins.values()),
            "n_exact_ties_structural": sum(
                1 for r in group if r["exact_tie_structural"]
            ),
            "mean_margin_structural": _mean(
                [r["best_vs_second_margin_structural"] for r in group]
            ),
            "frac_margin_gt_0.01_structural": (
                sum(1 for r in group if r["best_vs_second_margin_structural"] > 0.01)
                / len(group)
                if group
                else 0.0
            ),
        }
        for p in policies:
            row[f"mean_anwg__{p}"] = _mean([r[f"score__{p}"] for r in group])
        out.append(row)
    return out


def _policy_metric_means(
    row_index: Mapping[Tuple[str, str], Mapping[str, Any]],
    sids: Sequence[str],
    policies: Sequence[str],
) -> List[dict]:
    out = []
    for p in policies:
        entry: Dict[str, Any] = {"policy_name": p, "n": len(sids)}
        for metric in MECHANISM_METRICS:
            vals = []
            for sid in sids:
                r = row_index.get((sid, p))
                if r is None or metric not in r:
                    continue
                if _finite(r[metric]):
                    vals.append(_f(r[metric]))
            entry[f"mean_{metric}"] = _mean(vals)
            entry[f"median_{metric}"] = _median(vals)
        out.append(entry)
    return out


def score_hypotheses(
    *,
    n_scen: int,
    structural_unique_eps: Mapping[str, int],
    near_tie_rate_all_001: float,
    near_tie_rate_struct_001: float,
    pair_lookup: Mapping[Tuple[str, str], Mapping[str, Any]],
    seed_winner_set_agree: float,
    h1: Mapping[str, Any],
    h2: Mapping[str, Any],
    h3: Mapping[str, Any],
    h4_h5: Mapping[str, Any],
    h10: Mapping[str, Any],
) -> List[dict]:
    """Score H1–H10 from preregistered wording; rules are quantitative."""

    def _verdict_h1() -> Tuple[str, str]:
        full_large_wins = int(h1["unique_wins_full_or_large_eps_0.01"])
        small_dp_wins = int(h1["unique_wins_small_or_decode_priority_eps_0.01"])
        mean_full_large = float(h1["mean_anwg_full_or_large_best"])
        mean_small_dp = float(h1["mean_anwg_small_or_decode_priority_best"])
        if h1["n"] == 0:
            return "DESIGN_CONFOUND", "empty low-occupancy TTFT-tight subset"
        if full_large_wins == 0 and small_dp_wins == 0:
            if mean_full_large + 1e-12 >= mean_small_dp:
                return (
                    "AMBIGUOUS",
                    "no unique ε=0.01 wins; full/large mean ANWG still competitive",
                )
            return (
                "CONTRADICT",
                "no unique ε=0.01 wins and small/decode-priority mean ANWG higher",
            )
        if full_large_wins >= small_dp_wins and mean_full_large + 0.005 >= mean_small_dp:
            return (
                "CONFIRM",
                "full/large unique-win count and mean ANWG competitive/superior",
            )
        if small_dp_wins > full_large_wins and mean_small_dp > mean_full_large + 0.005:
            return (
                "CONTRADICT",
                "small/decode-priority dominate low-occupancy TTFT-tight cells",
            )
        return "AMBIGUOUS", "mixed unique-win / mean-ANWG pattern in H1 subset"

    def _verdict_h2() -> Tuple[str, str]:
        if h2["n"] == 0:
            return "DESIGN_CONFOUND", "empty high-occupancy TBT-tight subset"
        anwg_protect = float(h2["mean_anwg_small_or_decode_priority_best"])
        anwg_full = float(h2["mean_anwg_full"])
        tbt_protect = float(h2["mean_tbt_attainment_small_or_decode_priority"])
        tbt_full = float(h2["mean_tbt_attainment_full"])
        tpot_protect = float(h2["mean_tpot_small_or_decode_priority"])
        tpot_full = float(h2["mean_tpot_full"])
        unique_protect = int(h2["unique_wins_small_or_decode_priority_eps_0.01"])
        unique_full = int(h2["unique_wins_full_eps_0.01"])
        delta = anwg_protect - anwg_full
        better_anwg = delta > PRACTICAL_EPS
        tbt_saturated = min(tbt_protect, tbt_full) >= 1.0 - 1e-12
        better_tbt = tbt_protect > tbt_full + 1e-12
        better_tpot = tpot_protect < tpot_full - 1e-12
        if unique_protect > unique_full and (better_anwg or better_tbt or better_tpot):
            return (
                "CONFIRM",
                "chunked/decode-priority uniquely win and decode-SLO/ANWG improve",
            )
        if better_anwg and (better_tbt or better_tpot or not tbt_saturated):
            return (
                "CONFIRM",
                "chunked/decode-priority improve ANWG by >0.01 under high overlap + TBT-tight",
            )
        if unique_full > unique_protect and anwg_full > anwg_protect + PRACTICAL_EPS:
            return (
                "CONTRADICT",
                "full_prefill wins high-overlap TBT-tight cells on ANWG",
            )
        if delta > 0 and tbt_saturated:
            return (
                "AMBIGUOUS",
                "chunked mean ANWG is higher but Δ≤0.01 and TBT/TPOT are saturated",
            )
        return "AMBIGUOUS", "high-overlap TBT-tight contrast is mixed"

    def _verdict_h3() -> Tuple[str, str]:
        m_lh = float(h3["mean_margin_long_high"])
        m_all = float(h3["mean_margin_all"])
        m_sl = float(h3["mean_margin_short_low"])
        frac_lh = float(h3["frac_margin_gt_0.01_long_high"])
        frac_all = float(h3["frac_margin_gt_0.01_all"])
        if m_lh > m_all and m_lh > m_sl and frac_lh >= frac_all:
            return (
                "CONFIRM",
                "long×high mean margin and >0.01 fraction exceed overall and short×low",
            )
        if m_lh <= m_sl and frac_lh <= frac_all:
            return (
                "CONTRADICT",
                "long×high separation is not stronger than short×low / overall",
            )
        return "AMBIGUOUS", "long×high is stronger on some but not all separation stats"

    def _verdict_h4() -> Tuple[str, str]:
        chunk_universal = bool(h4_h5["chunking_universal_at_eps_0.01"])
        other_wins = int(h4_h5["unique_wins_non_chunk_structural_eps_0.01"])
        if other_wins > 0 and not chunk_universal:
            return "CONFIRM", "non-chunk structural policies uniquely win some cells"
        if chunk_universal:
            return "CONTRADICT", "a chunked policy is in every ε=0.01 winner set"
        return "AMBIGUOUS", "chunking is frequent but not a clean universal optimum"

    def _verdict_h5() -> Tuple[str, str]:
        full_universal = bool(h4_h5["full_universal_at_eps_0.01"])
        other_wins = int(h4_h5["unique_wins_non_full_structural_eps_0.01"])
        pairwise_beats = int(h4_h5["n_cells_some_structural_beats_full_eps_0.01"])
        if pairwise_beats > 0 and not full_universal:
            return (
                "CONFIRM",
                f"{pairwise_beats} cells have a structural policy beating full by >0.01; "
                f"unique non-full ε=0.01 wins={other_wins}",
            )
        if full_universal:
            return "CONTRADICT", "full_prefill is in every ε=0.01 winner set"
        return "AMBIGUOUS", "full_prefill is frequent but not a clean universal optimum"

    def _verdict_h6() -> Tuple[str, str]:
        n_pols = sum(1 for c in structural_unique_eps.values() if c >= 1)
        if n_pols >= 2:
            return (
                "CONFIRM",
                f"{n_pols} structural policies each uniquely win ≥1 cell at ε=0.01",
            )
        if n_pols <= 1:
            return (
                "CONTRADICT",
                f"only {n_pols} structural policy uniquely wins at ε=0.01",
            )
        return "AMBIGUOUS", "winner-identity criterion inconclusive"

    def _verdict_h7() -> Tuple[str, str]:
        # Preregistered bar is ≤ ~0.45. Use the evaluated 5-policy near-tie rate.
        if near_tie_rate_all_001 <= 0.45:
            return (
                "CONFIRM",
                f"near-tie rate {near_tie_rate_all_001:.3f} ≤ 0.45 (all policies)",
            )
        if near_tie_rate_struct_001 <= 0.45 < near_tie_rate_all_001:
            return (
                "AMBIGUOUS",
                "structural-only near-tie ≤0.45 but 5-policy rate exceeds bar",
            )
        return (
            "CONTRADICT",
            f"near-tie rate {near_tie_rate_all_001:.3f} exceeds ~0.45 bar",
        )

    def _verdict_h8() -> Tuple[str, str]:
        bidir = []
        for a, b in IMPORTANT_PAIRS:
            if a == "chunked_prefill_small" and b == "chunked_prefill_large":
                continue  # H8 asks for a structural pair; chunk-size pair is extra
            block = pair_lookup[(a, b)]
            if block["bidirectional_eps_0.01"]:
                bidir.append(f"{a}↔{b}")
        if bidir:
            return "CONFIRM", "bidirectional at ε=0.01: " + ", ".join(bidir)
        return "CONTRADICT", "no important structural pair is bidirectional at ε=0.01"

    def _verdict_h9() -> Tuple[str, str]:
        if seed_winner_set_agree >= 0.70:
            return (
                "CONFIRM",
                f"winner-set agreement {seed_winner_set_agree:.3f} ≥ 0.70",
            )
        if seed_winner_set_agree >= 0.60:
            return (
                "AMBIGUOUS",
                f"winner-set agreement {seed_winner_set_agree:.3f} is a near-miss of ~0.70",
            )
        return (
            "CONTRADICT",
            f"winner-set agreement {seed_winner_set_agree:.3f} < 0.60",
        )

    def _verdict_h10() -> Tuple[str, str]:
        decode_stall_dead = bool(h10["decode_stalled_steps_identically_zero"])
        ok_full = bool(h10["full_large_wins_have_lower_prefill_ttft_or_delay"])
        ok_chunk = bool(h10["small_beats_full_have_fewer_prefill_stalls"])
        n_full = int(h10["n_full_or_large_unique"])
        n_pair = int(h10["n_small_beats_full_eps_0.01"])
        if n_full == 0 and n_pair == 0:
            return "AMBIGUOUS", "no unique or pairwise niches to explain"
        if ok_full and ok_chunk and not decode_stall_dead:
            return "CONFIRM", "mechanism metrics match the expected win-direction pattern"
        if ok_full and ok_chunk and decode_stall_dead:
            return (
                "AMBIGUOUS",
                "TTFT/prefill-stall directions match, but decode_stalled_steps are identically 0",
            )
        if (ok_full or ok_chunk) and not h10["mechanism_reversed"]:
            return "AMBIGUOUS", "only one win-direction has the expected mechanism pattern"
        if h10["mechanism_reversed"]:
            return "CONTRADICT", "mechanism metrics reverse the expected win-direction pattern"
        return "AMBIGUOUS", "diagnostics do not cleanly explain winner transitions"

    rows = []
    for hid, fn, text in (
        ("H1", _verdict_h1, "Full/large prefill competitive/superior under low decode overlap and TTFT-tight"),
        ("H2", _verdict_h2, "Chunked/decode-priority reduces decode-tenant SLO harm under high overlap + decode-tight"),
        ("H3", _verdict_h3, "Long-prefill + high-overlap cells produce the strongest separation"),
        ("H4", _verdict_h4, "Fixed chunking is not universally optimal"),
        ("H5", _verdict_h5, "Full prefill is not universally optimal"),
        ("H6", _verdict_h6, "Winner identity changes across regimes (≥2 policies uniquely win at ε=0.01)"),
        ("H7", _verdict_h7, "Near-tie rate at ε=0.01 is low enough (≤ ~0.45)"),
        ("H8", _verdict_h8, "At least one structural pair is bidirectional at ε=0.01"),
        ("H9", _verdict_h9, "Seed-stable winner-set agreement ≥ ~0.7"),
        ("H10", _verdict_h10, "Diagnostics explain transitions"),
    ):
        verdict, evidence = fn()
        rows.append(
            {
                "id": hid,
                "hypothesis": text,
                "verdict": verdict,
                "evidence": evidence,
            }
        )
    return rows


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
        raise ValueError("Family B v1 results must not use ambiguous column name 'anwg'")

    policies = tuple(sorted({r["policy_name"] for r in rows}))
    by_primary: Dict[str, Dict[str, float]] = defaultdict(dict)
    row_index: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    nan_inf_primary = 0
    nan_inf_other = 0
    for r in rows:
        sid, p = r["scenario_id"], r["policy_name"]
        if not _finite(r.get(PRIMARY)):
            nan_inf_primary += 1
        else:
            by_primary[sid][p] = _f(r[PRIMARY])
        for m in MECHANISM_METRICS:
            if m in r and r[m] not in ("", None) and not _finite(r[m]):
                nan_inf_other += 1
        row_index[(sid, p)] = r
    by_primary = dict(by_primary)
    meta = {sid: parse_scenario_id(sid) for sid in by_primary}

    scenario_rows: List[dict] = []
    burst_kinds: Counter = Counter()
    burst_ok_n = 0
    for sid, scores in sorted(by_primary.items()):
        winners_all, best_all, second_all, margin_all = winner_margin(scores)
        struct_scores = {p: scores[p] for p in STRUCTURAL_POLICIES if p in scores}
        winners_s, best_s, second_s, margin_s = winner_margin(struct_scores)
        m = meta[sid]
        feat = features.get(sid, {})
        ok, kinds = _burstgpt_ok(str(feat.get("token_sources", "")))
        if ok:
            burst_ok_n += 1
        for k, v in kinds.items():
            burst_kinds[f"{k}:{v}"] += 1
        entry: Dict[str, Any] = {
            "scenario_id": sid,
            **m,
            "token_sources": feat.get("token_sources", ""),
            "best_policy": best_all if best_all is not None else "|".join(winners_all),
            "second_policy": second_all,
            "best_vs_second_margin": margin_all,
            "exact_tie": len(winners_all) > 1,
            "exact_winners": "|".join(winners_all),
            "n_exact_winners": len(winners_all),
            "best_policy_structural": (
                best_s if best_s is not None else "|".join(winners_s)
            ),
            "second_policy_structural": second_s,
            "best_vs_second_margin_structural": margin_s,
            "exact_tie_structural": len(winners_s) > 1,
            "exact_winners_structural": "|".join(winners_s),
            "unique_eps_0.01": unique_winner_at_eps(scores, PRACTICAL_EPS),
            "unique_structural_eps_0.01": unique_winner_at_eps(
                struct_scores, PRACTICAL_EPS
            ),
        }
        for eps in EPSILONS:
            entry[f"near_tie_eps_{eps}"] = near_tie(scores, eps)
            entry[f"near_tie_structural_eps_{eps}"] = near_tie(struct_scores, eps)
        for p in policies:
            entry[f"score__{p}"] = scores[p]
        adaptive = scores.get(DIAGNOSTIC_POLICY)
        max_struct = max(struct_scores.values()) if struct_scores else float("nan")
        if adaptive is None:
            entry["adaptive_vs_best_structural"] = float("nan")
            entry["adaptive_expands_eps_0.01"] = False
            entry["adaptive_matches_structural"] = ""
        else:
            entry["adaptive_vs_best_structural"] = adaptive - max_struct
            entry["adaptive_expands_eps_0.01"] = adaptive > max_struct + PRACTICAL_EPS
            matched = [
                p
                for p, v in struct_scores.items()
                if abs(adaptive - v) <= PRACTICAL_EPS
            ]
            entry["adaptive_matches_structural"] = "|".join(matched)
        scenario_rows.append(entry)

    n_scen = len(scenario_rows)
    exact_winner_sets = Counter(r["exact_winners_structural"] for r in scenario_rows)
    identity_collapse = {
        "chunked_prefill_small_eq_decode_priority_chunked": sum(
            1
            for r in scenario_rows
            if abs(r["score__chunked_prefill_small"] - r["score__decode_priority_chunked"])
            <= 1e-15
        ),
        "chunked_prefill_small_eq_adaptive": sum(
            1
            for r in scenario_rows
            if abs(r["score__chunked_prefill_small"] - r["score__adaptive_prefill_control"])
            <= 1e-15
        ),
        "full_prefill_eq_chunked_prefill_large": sum(
            1
            for r in scenario_rows
            if abs(r["score__full_prefill"] - r["score__chunked_prefill_large"]) <= 1e-15
        ),
        "full_prefill_near_chunked_prefill_large_eps_0.01": sum(
            1
            for r in scenario_rows
            if abs(r["score__full_prefill"] - r["score__chunked_prefill_large"])
            <= PRACTICAL_EPS
        ),
        "spread_gt_0.01_structural": sum(
            1
            for r in scenario_rows
            if max(r[f"score__{p}"] for p in STRUCTURAL_POLICIES)
            - min(r[f"score__{p}"] for p in STRUCTURAL_POLICIES)
            > PRACTICAL_EPS
        ),
    }
    unique_wins_all = Counter(
        r["best_policy"] for r in scenario_rows if not r["exact_tie"]
    )
    unique_wins_struct = Counter(
        r["best_policy_structural"]
        for r in scenario_rows
        if not r["exact_tie_structural"]
    )
    unique_eps_all = Counter(
        r["unique_eps_0.01"] for r in scenario_rows if r["unique_eps_0.01"]
    )
    unique_eps_struct = Counter(
        r["unique_structural_eps_0.01"]
        for r in scenario_rows
        if r["unique_structural_eps_0.01"]
    )
    exact_tie_n = sum(1 for r in scenario_rows if r["exact_tie"])
    exact_tie_s_n = sum(1 for r in scenario_rows if r["exact_tie_structural"])
    near_tie_counts = {
        str(eps): sum(1 for r in scenario_rows if r[f"near_tie_eps_{eps}"])
        for eps in EPSILONS
    }
    near_tie_struct_counts = {
        str(eps): sum(1 for r in scenario_rows if r[f"near_tie_structural_eps_{eps}"])
        for eps in EPSILONS
    }
    margins_all = [r["best_vs_second_margin"] for r in scenario_rows]
    margins_s = [r["best_vs_second_margin_structural"] for r in scenario_rows]

    def _headroom(margins: Sequence[float]) -> Dict[str, float]:
        return {
            "mean": _mean(list(margins)),
            "median": _median(list(margins)),
            "frac_gt_0": (sum(1 for x in margins if x > 0) / n_scen) if n_scen else 0.0,
            "frac_gt_0.01": (
                sum(1 for x in margins if x > 0.01) / n_scen if n_scen else 0.0
            ),
        }

    def _cond_counts(key: str, structural: bool = False) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Counter] = defaultdict(Counter)
        for r in scenario_rows:
            label = str(r[key])
            if structural:
                if r["exact_tie_structural"]:
                    out[label]["EXACT_TIE"] += 1
                else:
                    out[label][r["best_policy_structural"]] += 1
            else:
                if r["exact_tie"]:
                    out[label]["EXACT_TIE"] += 1
                else:
                    out[label][r["best_policy"]] += 1
        return {k: dict(v) for k, v in sorted(out.items())}

    winner_by_axis = {axis: _cond_counts(axis, structural=True) for axis in FACTORS}

    pair_rows: List[dict] = []
    pair_summary: List[dict] = []
    pair_lookup: Dict[Tuple[str, str], dict] = {}
    for a, b in combinations(policies, 2):
        block = _pairwise_block(by_primary, meta, a, b)
        pair_summary.append(block)
        pair_lookup[(a, b)] = block
        pair_lookup[(b, a)] = _pairwise_block(by_primary, meta, b, a)
        for sid, scores in by_primary.items():
            pair_rows.append(
                {
                    "scenario_id": sid,
                    **meta[sid],
                    "policy_i": a,
                    "policy_j": b,
                    "delta_ij": scores[a] - scores[b],
                }
            )

    important_pairs = []
    for a, b in IMPORTANT_PAIRS:
        important_pairs.append(pair_lookup[(a, b)])

    surfaces = {
        "prefill_size_x_decode_occupancy": _surface(
            scenario_rows, "prefill_size_class", "decode_occupancy", policies
        ),
        "slo_regime_x_decode_occupancy": _surface(
            scenario_rows, "slo_regime", "decode_occupancy", policies
        ),
        "prefill_size_x_slo_regime": _surface(
            scenario_rows, "prefill_size_class", "slo_regime", policies
        ),
    }

    # Low-load counterexample (preregistered case 1 + task-7 slice).
    low_prereg = _subset(
        scenario_rows,
        decode_occupancy="low",
        prefill_size_class="long",
        slo_regime="ttft_tight",
    )
    low_task7 = _subset(
        scenario_rows,
        decode_occupancy="low",
        offered_load="moderate",
        slo_regime="ttft_tight",
    )

    def _edge_stats(group: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Any]:
        sids = [r["scenario_id"] for r in group]
        wins = Counter(
            r["unique_structural_eps_0.01"]
            for r in group
            if r["unique_structural_eps_0.01"]
        )
        exact = Counter(
            r["best_policy_structural"]
            for r in group
            if not r["exact_tie_structural"]
        )
        return {
            "label": label,
            "n": len(group),
            "unique_wins_eps_0.01": dict(wins),
            "exact_unique_wins": dict(exact),
            "mean_margin_structural": _mean(
                [r["best_vs_second_margin_structural"] for r in group]
            ),
            "frac_margin_gt_0.01": (
                sum(1 for r in group if r["best_vs_second_margin_structural"] > 0.01)
                / len(group)
                if group
                else 0.0
            ),
            "mean_anwg_by_policy": {
                p: _mean([r[f"score__{p}"] for r in group]) for p in policies
            },
            "mechanism_means": _policy_metric_means(row_index, sids, policies),
        }

    low_load = {
        "preregistered_long_low_ttft_tight": _edge_stats(
            low_prereg, "occ=low, prefill=long, slo=ttft_tight"
        ),
        "task7_low_moderate_ttft_tight": _edge_stats(
            low_task7, "occ=low, load=moderate, slo=ttft_tight"
        ),
    }

    convoy_long = _subset(
        scenario_rows,
        decode_occupancy="high",
        prefill_size_class="long",
        slo_regime="tbt_tight",
    )
    convoy_mixed = _subset(
        scenario_rows,
        decode_occupancy="high",
        prefill_size_class="mixed",
        slo_regime="tbt_tight",
    )
    convoy = {
        "long_high_tbt_tight": _edge_stats(
            convoy_long, "prefill=long, occ=high, slo=tbt_tight"
        ),
        "mixed_high_tbt_tight": _edge_stats(
            convoy_mixed, "prefill=mixed, occ=high, slo=tbt_tight"
        ),
    }

    policy_overall = _policy_metric_means(
        row_index, [r["scenario_id"] for r in scenario_rows], policies
    )

    # Adaptive diagnostic.
    adaptive_unique = sum(
        1 for r in scenario_rows if r["unique_eps_0.01"] == DIAGNOSTIC_POLICY
    )
    adaptive_exact = sum(
        1
        for r in scenario_rows
        if (not r["exact_tie"]) and r["best_policy"] == DIAGNOSTIC_POLICY
    )
    adaptive_expand = sum(1 for r in scenario_rows if r["adaptive_expands_eps_0.01"])
    collapse_counts: Counter = Counter()
    for r in scenario_rows:
        matched = [
            p for p in r["adaptive_matches_structural"].split("|") if p
        ]
        if len(matched) == 1:
            collapse_counts[matched[0]] += 1
        elif len(matched) == 0:
            collapse_counts["matches_none"] += 1
        else:
            collapse_counts["matches_multiple"] += 1
    adaptive_win_regimes = [
        {
            "scenario_id": r["scenario_id"],
            "prefill_size_class": r["prefill_size_class"],
            "decode_occupancy": r["decode_occupancy"],
            "slo_regime": r["slo_regime"],
            "offered_load": r["offered_load"],
            "seed": r["seed"],
            "adaptive_vs_best_structural": r["adaptive_vs_best_structural"],
        }
        for r in scenario_rows
        if r["unique_eps_0.01"] == DIAGNOSTIC_POLICY or r["adaptive_expands_eps_0.01"]
    ]

    # Seed stability over non-seed factors.
    cells: Dict[Tuple[Any, ...], List[dict]] = defaultdict(list)
    for r in scenario_rows:
        key = (
            r["prefill_size_class"],
            r["decode_occupancy"],
            r["slo_regime"],
            r["offered_load"],
        )
        cells[key].append(r)

    seed_rows: List[dict] = []
    agree_winner_set = agree_best = 0
    agree_struct_set = agree_struct_best = 0
    pair_sign_agree = {f"{a}__{b}": 0 for a, b in IMPORTANT_PAIRS}
    unstable_cells: List[dict] = []
    for key, group in sorted(cells.items()):
        psize, occ, slo, load = key
        g_sorted = sorted(group, key=lambda r: r["seed"])
        sets = [tuple(sorted(r["exact_winners"].split("|"))) for r in g_sorted]
        bests = [r["best_policy"] for r in g_sorted]
        sets_s = [
            tuple(sorted(r["exact_winners_structural"].split("|"))) for r in g_sorted
        ]
        bests_s = [r["best_policy_structural"] for r in g_sorted]
        winner_set_agree = len(set(sets)) == 1
        best_agree = len(set(bests)) == 1
        struct_set_agree = len(set(sets_s)) == 1
        struct_best_agree = len(set(bests_s)) == 1
        if winner_set_agree:
            agree_winner_set += 1
        if best_agree:
            agree_best += 1
        if struct_set_agree:
            agree_struct_set += 1
        if struct_best_agree:
            agree_struct_best += 1
        row = {
            "prefill_size_class": psize,
            "decode_occupancy": occ,
            "slo_regime": slo,
            "offered_load": load,
            "n_seeds_observed": len(group),
            "winner_set_agree": winner_set_agree,
            "best_policy_agree": best_agree,
            "structural_winner_set_agree": struct_set_agree,
            "structural_best_policy_agree": struct_best_agree,
            "winner_sets": ";".join("|".join(s) for s in sets),
            "best_policies": ";".join(bests),
            "structural_winner_sets": ";".join("|".join(s) for s in sets_s),
            "structural_best_policies": ";".join(bests_s),
        }
        all_pair_ok = True
        for a, b in IMPORTANT_PAIRS:
            signs = []
            for r in g_sorted:
                d = r[f"score__{a}"] - r[f"score__{b}"]
                signs.append(0 if abs(d) <= PRACTICAL_EPS else (1 if d > 0 else -1))
            ok = len(set(signs)) == 1
            row[f"sign_agree_{a}__{b}"] = ok
            if ok:
                pair_sign_agree[f"{a}__{b}"] += 1
            else:
                all_pair_ok = False
        seed_rows.append(row)
        if not (struct_set_agree and struct_best_agree and all_pair_ok):
            unstable_cells.append(row)

    n_cells = len(cells)

    # H1–H5 / H10 quantitative blocks.
    h1_group = _subset(
        scenario_rows, decode_occupancy="low", slo_regime="ttft_tight"
    )
    h2_group = _subset(
        scenario_rows, decode_occupancy="high", slo_regime="tbt_tight"
    )

    def _unique_count(group: Sequence[Mapping[str, Any]], names: Sequence[str]) -> int:
        return sum(
            1
            for r in group
            if r["unique_structural_eps_0.01"] in names
        )

    h1_block = {
        "n": len(h1_group),
        "unique_wins_full_or_large_eps_0.01": _unique_count(
            h1_group, ("full_prefill", "chunked_prefill_large")
        ),
        "unique_wins_small_or_decode_priority_eps_0.01": _unique_count(
            h1_group, ("chunked_prefill_small", "decode_priority_chunked")
        ),
        "mean_anwg_full_or_large_best": _mean(
            [
                max(r["score__full_prefill"], r["score__chunked_prefill_large"])
                for r in h1_group
            ]
        ),
        "mean_anwg_small_or_decode_priority_best": _mean(
            [
                max(
                    r["score__chunked_prefill_small"],
                    r["score__decode_priority_chunked"],
                )
                for r in h1_group
            ]
        ),
        "mean_anwg_by_policy": {
            p: _mean([r[f"score__{p}"] for r in h1_group]) for p in STRUCTURAL_POLICIES
        },
    }

    def _metric_mean(group: Sequence[Mapping[str, Any]], policy: str, metric: str) -> float:
        vals = []
        for r in group:
            row = row_index[(r["scenario_id"], policy)]
            if metric in row and _finite(row[metric]):
                vals.append(_f(row[metric]))
        return _mean(vals)

    def _metric_mean_best_of(
        group: Sequence[Mapping[str, Any]], names: Sequence[str], metric: str
    ) -> float:
        vals = []
        for r in group:
            best_p = max(names, key=lambda p: r[f"score__{p}"])
            row = row_index[(r["scenario_id"], best_p)]
            if metric in row and _finite(row[metric]):
                vals.append(_f(row[metric]))
        return _mean(vals)

    h2_block = {
        "n": len(h2_group),
        "unique_wins_small_or_decode_priority_eps_0.01": _unique_count(
            h2_group, ("chunked_prefill_small", "decode_priority_chunked")
        ),
        "unique_wins_full_eps_0.01": _unique_count(h2_group, ("full_prefill",)),
        "mean_anwg_small_or_decode_priority_best": _mean(
            [
                max(
                    r["score__chunked_prefill_small"],
                    r["score__decode_priority_chunked"],
                )
                for r in h2_group
            ]
        ),
        "mean_anwg_full": _mean([r["score__full_prefill"] for r in h2_group]),
        "mean_tbt_attainment_small_or_decode_priority": _metric_mean_best_of(
            h2_group,
            ("chunked_prefill_small", "decode_priority_chunked"),
            "tbt_attainment",
        ),
        "mean_tbt_attainment_full": _metric_mean(h2_group, "full_prefill", "tbt_attainment"),
        "mean_tpot_small_or_decode_priority": _metric_mean_best_of(
            h2_group,
            ("chunked_prefill_small", "decode_priority_chunked"),
            "mean_tpot",
        ),
        "mean_tpot_full": _metric_mean(h2_group, "full_prefill", "mean_tpot"),
        "mean_anwg_by_policy": {
            p: _mean([r[f"score__{p}"] for r in h2_group]) for p in STRUCTURAL_POLICIES
        },
    }

    long_high = _subset(
        scenario_rows, prefill_size_class="long", decode_occupancy="high"
    )
    short_low = _subset(
        scenario_rows, prefill_size_class="short", decode_occupancy="low"
    )
    h3_block = {
        "n_long_high": len(long_high),
        "n_short_low": len(short_low),
        "mean_margin_long_high": _mean(
            [r["best_vs_second_margin_structural"] for r in long_high]
        ),
        "mean_margin_short_low": _mean(
            [r["best_vs_second_margin_structural"] for r in short_low]
        ),
        "mean_margin_all": _mean(margins_s),
        "frac_margin_gt_0.01_long_high": (
            sum(1 for r in long_high if r["best_vs_second_margin_structural"] > 0.01)
            / len(long_high)
            if long_high
            else 0.0
        ),
        "frac_margin_gt_0.01_short_low": (
            sum(1 for r in short_low if r["best_vs_second_margin_structural"] > 0.01)
            / len(short_low)
            if short_low
            else 0.0
        ),
        "frac_margin_gt_0.01_all": _headroom(margins_s)["frac_gt_0.01"],
    }

    chunk_in_all_winner_sets = all(
        (
            "chunked_prefill_small" in r["exact_winners_structural"].split("|")
            or "chunked_prefill_large" in r["exact_winners_structural"].split("|")
            or (
                r["unique_structural_eps_0.01"] is None
                and (
                    abs(
                        r["score__chunked_prefill_small"]
                        - max(r[f"score__{p}"] for p in STRUCTURAL_POLICIES)
                    )
                    <= PRACTICAL_EPS
                    or abs(
                        r["score__chunked_prefill_large"]
                        - max(r[f"score__{p}"] for p in STRUCTURAL_POLICIES)
                    )
                    <= PRACTICAL_EPS
                )
            )
        )
        for r in scenario_rows
    )
    full_in_all_eps_sets = all(
        (
            r["unique_structural_eps_0.01"] == "full_prefill"
            or (
                r["unique_structural_eps_0.01"] is None
                and abs(
                    r["score__full_prefill"]
                    - max(r[f"score__{p}"] for p in STRUCTURAL_POLICIES)
                )
                <= PRACTICAL_EPS
            )
        )
        for r in scenario_rows
    )
    n_beats_full = sum(
        1
        for r in scenario_rows
        if any(
            r[f"score__{p}"] - r["score__full_prefill"] > PRACTICAL_EPS
            for p in STRUCTURAL_POLICIES
            if p != "full_prefill"
        )
    )
    h4_h5_block = {
        "unique_wins_non_chunk_structural_eps_0.01": int(
            unique_eps_struct.get("full_prefill", 0)
            + unique_eps_struct.get("decode_priority_chunked", 0)
        ),
        "unique_wins_non_full_structural_eps_0.01": int(
            sum(v for k, v in unique_eps_struct.items() if k != "full_prefill")
        ),
        "n_cells_some_structural_beats_full_eps_0.01": n_beats_full,
        "chunking_universal_at_eps_0.01": chunk_in_all_winner_sets,
        "full_universal_at_eps_0.01": full_in_all_eps_sets,
        "structural_unique_eps_0.01": dict(unique_eps_struct),
    }

    full_large_sids = [
        r["scenario_id"]
        for r in scenario_rows
        if r["unique_structural_eps_0.01"] in {"full_prefill", "chunked_prefill_large"}
    ]
    small_dp_sids = [
        r["scenario_id"]
        for r in scenario_rows
        if r["unique_structural_eps_0.01"]
        in {"chunked_prefill_small", "decode_priority_chunked"}
    ]

    def _cmp_when_wins(sids: Sequence[str], winner_names: Sequence[str]) -> Dict[str, Any]:
        if not sids:
            return {
                "n": 0,
                "winner_mean_ttft": float("nan"),
                "full_mean_ttft": float("nan"),
                "winner_mean_prefill_delay": float("nan"),
                "full_mean_prefill_delay": float("nan"),
                "winner_mean_tpot": float("nan"),
                "full_mean_tpot": float("nan"),
                "winner_tbt_attainment": float("nan"),
                "full_tbt_attainment": float("nan"),
                "winner_prefill_stalled_steps": float("nan"),
                "full_prefill_stalled_steps": float("nan"),
                "winner_decode_stalled_steps": float("nan"),
                "full_decode_stalled_steps": float("nan"),
            }
        winner_ttft, full_ttft = [], []
        winner_delay, full_delay = [], []
        winner_tpot, full_tpot = [], []
        winner_tbt, full_tbt = [], []
        winner_ps, full_ps = [], []
        winner_ds, full_ds = [], []
        for sid in sids:
            scores = by_primary[sid]
            winner = max(winner_names, key=lambda p: scores[p])
            wr = row_index[(sid, winner)]
            fr = row_index[(sid, "full_prefill")]
            winner_ttft.append(_f(wr["mean_ttft"]))
            full_ttft.append(_f(fr["mean_ttft"]))
            winner_delay.append(_f(wr["mean_prefill_delay_s"]))
            full_delay.append(_f(fr["mean_prefill_delay_s"]))
            winner_tpot.append(_f(wr["mean_tpot"]))
            full_tpot.append(_f(fr["mean_tpot"]))
            winner_tbt.append(_f(wr["tbt_attainment"]))
            full_tbt.append(_f(fr["tbt_attainment"]))
            winner_ps.append(_f(wr["prefill_stalled_steps"]))
            full_ps.append(_f(fr["prefill_stalled_steps"]))
            winner_ds.append(_f(wr["decode_stalled_steps"]))
            full_ds.append(_f(fr["decode_stalled_steps"]))
        return {
            "n": len(sids),
            "winner_mean_ttft": _mean(winner_ttft),
            "full_mean_ttft": _mean(full_ttft),
            "winner_mean_prefill_delay": _mean(winner_delay),
            "full_mean_prefill_delay": _mean(full_delay),
            "winner_mean_tpot": _mean(winner_tpot),
            "full_mean_tpot": _mean(full_tpot),
            "winner_tbt_attainment": _mean(winner_tbt),
            "full_tbt_attainment": _mean(full_tbt),
            "winner_prefill_stalled_steps": _mean(winner_ps),
            "full_prefill_stalled_steps": _mean(full_ps),
            "winner_decode_stalled_steps": _mean(winner_ds),
            "full_decode_stalled_steps": _mean(full_ds),
        }

    fl_mech = _cmp_when_wins(
        full_large_sids, ("full_prefill", "chunked_prefill_large")
    )
    sdp_mech = _cmp_when_wins(
        small_dp_sids, ("chunked_prefill_small", "decode_priority_chunked")
    )
    full_large_better_ttft = (
        fl_mech["n"] == 0
        or fl_mech["winner_mean_ttft"] <= fl_mech["full_mean_ttft"] + 1e-12
        or fl_mech["winner_mean_prefill_delay"]
        <= fl_mech["full_mean_prefill_delay"] + 1e-12
    )
    # When full/large wins, the winner *is* full or large, so TTFT comparison vs
    # full is tautological if full wins. Compare vs small-chunk instead.
    if full_large_sids:
        win_ttft, small_ttft, win_delay, small_delay = [], [], [], []
        for sid in full_large_sids:
            scores = by_primary[sid]
            winner = max(
                ("full_prefill", "chunked_prefill_large"), key=lambda p: scores[p]
            )
            wr = row_index[(sid, winner)]
            sr = row_index[(sid, "chunked_prefill_small")]
            win_ttft.append(_f(wr["mean_ttft"]))
            small_ttft.append(_f(sr["mean_ttft"]))
            win_delay.append(_f(wr["mean_prefill_delay_s"]))
            small_delay.append(_f(sr["mean_prefill_delay_s"]))
        full_large_better_ttft = _mean(win_ttft) < _mean(small_ttft) or _mean(
            win_delay
        ) < _mean(small_delay)
        fl_mech["winner_vs_small_mean_ttft"] = _mean(win_ttft)
        fl_mech["small_mean_ttft"] = _mean(small_ttft)
        fl_mech["winner_vs_small_prefill_delay"] = _mean(win_delay)
        fl_mech["small_prefill_delay"] = _mean(small_delay)

    small_beats_full_sids = [
        r["scenario_id"]
        for r in scenario_rows
        if r["score__chunked_prefill_small"] - r["score__full_prefill"] > PRACTICAL_EPS
    ]
    small_beats_mech = _cmp_when_wins(
        small_beats_full_sids, ("chunked_prefill_small", "decode_priority_chunked")
    )
    small_beats_fewer_stalls = small_beats_mech["n"] > 0 and (
        small_beats_mech["winner_prefill_stalled_steps"]
        < small_beats_mech["full_prefill_stalled_steps"]
    )
    decode_stall_zero = all(
        abs(_metric_mean(scenario_rows, p, "decode_stalled_steps")) <= 1e-12
        for p in STRUCTURAL_POLICIES
    )
    small_dp_better_decode = small_beats_fewer_stalls or (
        sdp_mech["n"] > 0
        and (
            sdp_mech["winner_mean_tpot"] <= sdp_mech["full_mean_tpot"] + 1e-12
            or sdp_mech["winner_tbt_attainment"] + 1e-12
            >= sdp_mech["full_tbt_attainment"]
            or sdp_mech["winner_prefill_stalled_steps"]
            <= sdp_mech["full_prefill_stalled_steps"] + 1e-12
        )
    )
    reversed_mech = False
    if full_large_sids and not full_large_better_ttft:
        reversed_mech = True
    if small_beats_full_sids and not small_beats_fewer_stalls:
        if (
            small_beats_mech["winner_mean_tpot"] > small_beats_mech["full_mean_tpot"]
            and small_beats_mech["winner_tbt_attainment"]
            < small_beats_mech["full_tbt_attainment"]
            and small_beats_mech["winner_prefill_stalled_steps"]
            > small_beats_mech["full_prefill_stalled_steps"]
        ):
            reversed_mech = True

    h10_block = {
        "n_full_or_large_unique": len(full_large_sids),
        "n_small_or_dp_unique": len(small_dp_sids),
        "n_small_beats_full_eps_0.01": len(small_beats_full_sids),
        "full_large_when_unique": fl_mech,
        "small_dp_when_unique": sdp_mech,
        "small_beats_full_when_pairwise": small_beats_mech,
        "full_large_wins_have_lower_prefill_ttft_or_delay": full_large_better_ttft,
        "small_dp_wins_have_better_decode_or_prefill_stall": small_dp_better_decode,
        "small_beats_full_have_fewer_prefill_stalls": small_beats_fewer_stalls,
        "decode_stalled_steps_identically_zero": decode_stall_zero,
        "mechanism_reversed": reversed_mech,
        "mean_decode_stalled_steps_by_policy": {
            p: _metric_mean(scenario_rows, p, "decode_stalled_steps")
            for p in STRUCTURAL_POLICIES
        },
    }

    n_struct_with_unique = sum(1 for c in unique_eps_struct.values() if c >= 1)
    n_struct_meaningful = sum(1 for c in unique_eps_struct.values() if c >= 3)
    bidir_pairs = [
        f"{a}↔{b}"
        for a, b in IMPORTANT_PAIRS
        if not (a == "chunked_prefill_small" and b == "chunked_prefill_large")
        and pair_lookup[(a, b)]["bidirectional_eps_0.01"]
    ]
    max_unique = max(unique_eps_struct.values()) if unique_eps_struct else 0
    dominant_share = (max_unique / n_scen) if n_scen else 1.0
    near_tie_all_001 = near_tie_counts["0.01"] / n_scen if n_scen else 1.0
    seed_agree = agree_struct_set / n_cells if n_cells else 0.0

    full_small = pair_lookup[("full_prefill", "chunked_prefill_small")]
    pairwise_full_small_bidir = bool(full_small["bidirectional_eps_0.01"])
    if (
        n_struct_with_unique >= 2
        and bidir_pairs
        and near_tie_all_001 <= 0.45
        and dominant_share < 0.85
        and seed_agree >= 0.70
        and n_struct_meaningful >= 2
    ):
        family_verdict = "STRUCTURAL_SEPARATION_VALIDATED"
    elif pairwise_full_small_bidir or (
        n_struct_with_unique >= 2 and unique_eps_struct
    ):
        # Pairwise structural contrast exists, but unique-winner diversity,
        # near-tie rate, twins, or seed stability block immediate composition.
        family_verdict = "USEFUL_BUT_NEEDS_REFINEMENT"
    else:
        family_verdict = "REDESIGN_REQUIRED"

    # Composition candidates: rank important structural pairs (exclude diagnostic).
    candidates = []
    for a, b in IMPORTANT_PAIRS:
        block = pair_lookup[(a, b)]
        i_beats = block["i_beats_j_eps_0.01"]
        j_beats = block["j_beats_i_eps_0.01"]
        structural = {a, b} <= set(STRUCTURAL_POLICIES)
        score = 0.0
        if block["bidirectional_eps_0.01"]:
            score += 10.0
        score += min(i_beats, j_beats)  # balance
        score += 0.05 * (i_beats + j_beats)
        score += 5.0 * float(block["mean_abs_delta"])
        sign_frac = pair_sign_agree[f"{a}__{b}"] / n_cells if n_cells else 0.0
        score += 3.0 * sign_frac
        if not structural:
            score -= 20.0
        candidates.append(
            {
                "policy_i": a,
                "policy_j": b,
                "structural": structural,
                "i_beats_j_eps_0.01": i_beats,
                "j_beats_i_eps_0.01": j_beats,
                "near_ties_eps_0.01": block["near_ties_eps_0.01"],
                "bidirectional_eps_0.01": block["bidirectional_eps_0.01"],
                "mean_abs_delta": block["mean_abs_delta"],
                "mean_delta_ij": block["mean_delta_ij"],
                "seed_sign_agree_frac": sign_frac,
                "i_beats_regimes_eps_0.01": block["i_beats_regimes_eps_0.01"],
                "j_beats_regimes_eps_0.01": block["j_beats_regimes_eps_0.01"],
                "rank_score": score,
            }
        )
    candidates.sort(key=lambda r: (-r["rank_score"], r["policy_i"], r["policy_j"]))
    strongest = candidates[0] if candidates else None

    hyp_rows = score_hypotheses(
        n_scen=n_scen,
        structural_unique_eps=unique_eps_struct,
        near_tie_rate_all_001=near_tie_all_001,
        near_tie_rate_struct_001=(
            near_tie_struct_counts["0.01"] / n_scen if n_scen else 1.0
        ),
        pair_lookup=pair_lookup,
        seed_winner_set_agree=seed_agree,
        h1=h1_block,
        h2=h2_block,
        h3=h3_block,
        h4_h5=h4_h5_block,
        h10=h10_block,
    )

    # Composition go/no-go from Family B evidence only.
    hyp_map = {h["id"]: h["verdict"] for h in hyp_rows}
    composition_ok = (
        family_verdict in {"STRUCTURAL_SEPARATION_VALIDATED", "USEFUL_BUT_NEEDS_REFINEMENT"}
        and hyp_map.get("H8") == "CONFIRM"
        and hyp_map.get("H6") == "CONFIRM"
        and strongest is not None
        and strongest["bidirectional_eps_0.01"]
        and strongest["structural"]
    )
    composition_decision = (
        "PREFILL_COMPOSITION_JUSTIFIED"
        if composition_ok
        else "PREFILL_COMPOSITION_NOT_YET_JUSTIFIED"
    )

    psizes = sorted({m["prefill_size_class"] for m in meta.values()})
    occs = sorted({m["decode_occupancy"] for m in meta.values()})
    slos = sorted({m["slo_regime"] for m in meta.values()})
    loads = sorted({m["offered_load"] for m in meta.values()})
    seeds = sorted({m["seed"] for m in meta.values()})

    status_ok = sum(1 for r in rows if r.get("status") == "success")
    duplicate_n = len(rows) - len({(r["scenario_id"], r["policy_name"]) for r in rows})
    per_scen_counts = Counter(r["scenario_id"] for r in rows)

    integrity = {
        "n_rows": len(rows),
        "n_scenarios": n_scen,
        "n_policies": len(policies),
        "policies": list(policies),
        "expected_policies_present": all(p in policies for p in POLICIES_EXPECTED),
        "duplicate_scenario_policy_keys": duplicate_n,
        "failed_rows": len(rows) - status_ok,
        "primary_column": PRIMARY,
        "has_ambiguous_anwg_column": False,
        "nan_inf_primary": nan_inf_primary,
        "nan_inf_mechanism_metrics": nan_inf_other,
        "policies_per_scenario_ok": all(c == 5 for c in per_scen_counts.values())
        and n_scen == len(per_scen_counts),
        "grid": {
            "prefill_size_class": psizes,
            "decode_occupancy": occs,
            "slo_regime": slos,
            "offered_load": loads,
            "seeds": seeds,
            "product": len(psizes) * len(occs) * len(slos) * len(loads) * len(seeds),
        },
        "burstgpt_ok_scenarios": burst_ok_n,
        "burstgpt_kind_counts": dict(burst_kinds),
        "burstgpt_consistent": burst_ok_n == n_scen and n_scen > 0,
    }
    integrity["grid_product_matches_n_scenarios"] = (
        integrity["grid"]["product"] == n_scen
    )
    integrity["expected_720"] = len(rows) == 720 and n_scen == 144 and duplicate_n == 0

    composition_questions = {
        "A_structural_decision_boundary": pairwise_full_small_bidir,
        "A_unique_winner_diversity": hyp_map.get("H6") == "CONFIRM",
        "B_predictable_from_online_state": bool(bidir_pairs),
        "C_fixed_chunk_size_insufficient": pair_lookup[
            ("chunked_prefill_small", "chunked_prefill_large")
        ]["bidirectional_eps_0.01"]
        or unique_eps_struct.get("full_prefill", 0) > 0,
        "D_decode_priority_distinct_niche": unique_eps_struct.get(
            "decode_priority_chunked", 0
        )
        > 0
        or pair_lookup[("chunked_prefill_small", "decode_priority_chunked")][
            "mean_abs_delta"
        ]
        > PRACTICAL_EPS,
        "E_adaptive_suggests_state_dependent_control": adaptive_expand > 0
        or adaptive_unique > 0,
        "F_composition_experiment_justified": composition_decision
        == "PREFILL_COMPOSITION_JUSTIFIED",
    }

    summary = {
        "integrity": integrity,
        "unique_winner_counts": dict(unique_wins_all),
        "unique_winner_counts_structural": dict(unique_wins_struct),
        "exact_winner_set_counts_structural": dict(exact_winner_sets),
        "identity_collapse": identity_collapse,
        "unique_winner_counts_eps_0.01": dict(unique_eps_all),
        "unique_winner_counts_structural_eps_0.01": dict(unique_eps_struct),
        "exact_tie_count": exact_tie_n,
        "exact_tie_rate": exact_tie_n / n_scen if n_scen else 0.0,
        "exact_tie_count_structural": exact_tie_s_n,
        "exact_tie_rate_structural": exact_tie_s_n / n_scen if n_scen else 0.0,
        "near_tie_counts": near_tie_counts,
        "near_tie_rates": {k: v / n_scen for k, v in near_tie_counts.items()},
        "near_tie_counts_structural": near_tie_struct_counts,
        "near_tie_rates_structural": {
            k: v / n_scen for k, v in near_tie_struct_counts.items()
        },
        "headroom": _headroom(margins_all),
        "headroom_structural": _headroom(margins_s),
        "winner_entropy_bits": shannon_entropy(unique_wins_all),
        "winner_entropy_bits_structural": shannon_entropy(unique_wins_struct),
        "winner_entropy_bits_structural_eps_0.01": shannon_entropy(unique_eps_struct),
        "winner_by_axis_structural": winner_by_axis,
        "pairwise_summary": pair_summary,
        "important_pairs": important_pairs,
        "seed_stability": {
            "n_cells": n_cells,
            "winner_set_agree_frac": agree_winner_set / n_cells if n_cells else 0.0,
            "best_policy_agree_frac": agree_best / n_cells if n_cells else 0.0,
            "structural_winner_set_agree_frac": seed_agree,
            "structural_best_policy_agree_frac": (
                agree_struct_best / n_cells if n_cells else 0.0
            ),
            "pair_sign_agree_frac": {
                k: v / n_cells if n_cells else 0.0 for k, v in pair_sign_agree.items()
            },
            "n_unstable_cells": len(unstable_cells),
        },
        "adaptive_diagnostic": {
            "unique_wins_eps_0.01": adaptive_unique,
            "exact_unique_wins": adaptive_exact,
            "envelope_expand_eps_0.01": adaptive_expand,
            "collapse_or_match_counts": dict(collapse_counts),
            "mean_adaptive_vs_best_structural": _mean(
                [r["adaptive_vs_best_structural"] for r in scenario_rows]
            ),
            "n_win_or_expand_rows": len(adaptive_win_regimes),
        },
        "hypothesis_blocks": {
            "H1": h1_block,
            "H2": h2_block,
            "H3": h3_block,
            "H4_H5": h4_h5_block,
            "H10": h10_block,
        },
        "hypotheses": hyp_rows,
        "family_b_verdict": family_verdict,
        "composition_decision": composition_decision,
        "composition_questions": composition_questions,
        "strongest_candidate_pair": strongest,
        "composition_candidates": candidates,
        "low_load_counterexample": {
            k: {kk: vv for kk, vv in v.items() if kk != "mechanism_means"}
            for k, v in low_load.items()
        },
        "prefill_convoy_edge": {
            k: {kk: vv for kk, vv in v.items() if kk != "mechanism_means"}
            for k, v in convoy.items()
        },
        "policy_overall": policy_overall,
    }

    return {
        "summary": summary,
        "per_scenario_winners": scenario_rows,
        "pairwise_deltas": pair_rows,
        "pairwise_summary": pair_summary,
        "important_pairs": important_pairs,
        "seed_stability": seed_rows,
        "unstable_cells": unstable_cells,
        "policy_overall": policy_overall,
        "surfaces_prefill_x_occupancy": surfaces["prefill_size_x_decode_occupancy"],
        "surfaces_slo_x_occupancy": surfaces["slo_regime_x_decode_occupancy"],
        "surfaces_prefill_x_slo": surfaces["prefill_size_x_slo_regime"],
        "low_load_prereg": low_load["preregistered_long_low_ttft_tight"][
            "mechanism_means"
        ],
        "low_load_task7": low_load["task7_low_moderate_ttft_tight"]["mechanism_means"],
        "convoy_long": convoy["long_high_tbt_tight"]["mechanism_means"],
        "convoy_mixed": convoy["mixed_high_tbt_tight"]["mechanism_means"],
        "adaptive_win_regimes": adaptive_win_regimes,
        "hypotheses": hyp_rows,
        "composition_candidates": candidates,
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

    def _cell(v: Any) -> Any:
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, sort_keys=True)
        return v

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k, "")) for k in fields})


def write_artifacts(out_dir: Path, result: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "per_scenario_winners.csv", result["per_scenario_winners"])
    _write_csv(out_dir / "pairwise_deltas.csv", result["pairwise_deltas"])
    _write_csv(out_dir / "pairwise_summary.csv", result["pairwise_summary"])
    _write_csv(out_dir / "important_pairs.csv", result["important_pairs"])
    _write_csv(out_dir / "seed_stability.csv", result["seed_stability"])
    _write_csv(out_dir / "unstable_cells.csv", result["unstable_cells"])
    _write_csv(out_dir / "policy_overall.csv", result["policy_overall"])
    _write_csv(
        out_dir / "surface_prefill_x_occupancy.csv",
        result["surfaces_prefill_x_occupancy"],
    )
    _write_csv(
        out_dir / "surface_slo_x_occupancy.csv", result["surfaces_slo_x_occupancy"]
    )
    _write_csv(out_dir / "surface_prefill_x_slo.csv", result["surfaces_prefill_x_slo"])
    _write_csv(out_dir / "low_load_prereg_mechanism.csv", result["low_load_prereg"])
    _write_csv(out_dir / "low_load_task7_mechanism.csv", result["low_load_task7"])
    _write_csv(out_dir / "convoy_long_mechanism.csv", result["convoy_long"])
    _write_csv(out_dir / "convoy_mixed_mechanism.csv", result["convoy_mixed"])
    _write_csv(out_dir / "adaptive_win_regimes.csv", result["adaptive_win_regimes"])
    _write_csv(out_dir / "hypotheses.csv", result["hypotheses"])
    _write_csv(out_dir / "composition_candidates.csv", result["composition_candidates"])


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
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "integrity": {
                    "n_rows": s["integrity"]["n_rows"],
                    "n_scenarios": s["integrity"]["n_scenarios"],
                    "failed_rows": s["integrity"]["failed_rows"],
                    "expected_720": s["integrity"]["expected_720"],
                    "burstgpt_consistent": s["integrity"]["burstgpt_consistent"],
                    "primary_column": s["integrity"]["primary_column"],
                },
                "unique_winners_eps_0.01_structural": s[
                    "unique_winner_counts_structural_eps_0.01"
                ],
                "exact_tie_rate": s["exact_tie_rate"],
                "near_tie_rate_eps_0.01": s["near_tie_rates"]["0.01"],
                "headroom_frac_gt_0.01_structural": s["headroom_structural"][
                    "frac_gt_0.01"
                ],
                "hypotheses": {h["id"]: h["verdict"] for h in s["hypotheses"]},
                "identity_collapse": s["identity_collapse"],
                "family_b_verdict": s["family_b_verdict"],
                "composition_decision": s["composition_decision"],
                "strongest_candidate_pair": {
                    "policy_i": s["strongest_candidate_pair"]["policy_i"],
                    "policy_j": s["strongest_candidate_pair"]["policy_j"],
                    "i_beats": s["strongest_candidate_pair"]["i_beats_j_eps_0.01"],
                    "j_beats": s["strongest_candidate_pair"]["j_beats_i_eps_0.01"],
                    "bidirectional": s["strongest_candidate_pair"][
                        "bidirectional_eps_0.01"
                    ],
                }
                if s["strongest_candidate_pair"]
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
