"""
Search-phase ranking: rank candidates by validation performance, not training.

Ranking rule:
  1. higher val_mean_wg (primary)
  2. lower val_violation_rate
  3. lower val_p95_ttft
  4. lower train_val_gap (less overfit)
  5. higher n_regimes_beating_best_fixed
  NaN → worst-case
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

from .multi_regime_evaluation import AggregatedCandidateResult


def _nan_safe(v: float, worst: float) -> float:
    return worst if math.isnan(v) or math.isinf(v) else v


def _sort_key(r: AggregatedCandidateResult):
    val_wg = _nan_safe(r.val_mean_wg, -1.0)
    val_vr = _nan_safe(r.val_violation_rate, 2.0)
    val_ttft = _nan_safe(r.val_p95_ttft, 1e9)
    gap = _nan_safe(r.train_val_gap, -1.0)   # larger gap = worse (less negative = better)
    beats = r.regimes_beating_best_fixed
    return (-val_wg, val_vr, val_ttft, -gap, -beats)


def rank_search_results(
    aggregated: Dict[str, AggregatedCandidateResult],
    *,
    source_filter: Optional[str] = None,
) -> List[AggregatedCandidateResult]:
    """Return candidates sorted best-first by validation performance."""
    candidates = list(aggregated.values())
    if source_filter:
        candidates = [c for c in candidates if c.source == source_filter]
    return sorted(candidates, key=_sort_key)


def _fmt(v: float) -> object:
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, 6)


def save_search_ranking_csv(
    ranked: List[AggregatedCandidateResult],
    output_path: Path,
) -> None:
    fields = [
        "rank", "name", "source",
        "val_mean_wg", "train_mean_wg", "train_val_gap",
        "worst_regime_wg", "worst_regime_name",
        "val_violation_rate", "val_p95_ttft",
        "regimes_beating_best_fixed",
        "regimes_beating_slo_slack",
        "regimes_beating_estf",
        "n_train_regimes", "n_val_regimes",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, r in enumerate(ranked, 1):
            w.writerow({
                "rank": rank,
                "name": r.name,
                "source": r.source,
                "val_mean_wg": _fmt(r.val_mean_wg),
                "train_mean_wg": _fmt(r.train_mean_wg),
                "train_val_gap": _fmt(r.train_val_gap),
                "worst_regime_wg": _fmt(r.worst_regime_wg),
                "worst_regime_name": r.worst_regime_name,
                "val_violation_rate": _fmt(r.val_violation_rate),
                "val_p95_ttft": _fmt(r.val_p95_ttft),
                "regimes_beating_best_fixed": r.regimes_beating_best_fixed,
                "regimes_beating_slo_slack": r.regimes_beating_slo_slack,
                "regimes_beating_estf": r.regimes_beating_estf,
                "n_train_regimes": r.n_train_regimes,
                "n_val_regimes": r.n_val_regimes,
            })


def save_per_regime_csv(
    ranked: List[AggregatedCandidateResult],
    regime_names: List[str],
    output_path: Path,
) -> None:
    fields = ["rank", "name", "source"] + regime_names
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, r in enumerate(ranked, 1):
            row = {"rank": rank, "name": r.name, "source": r.source}
            for rn in regime_names:
                row[rn] = _fmt(r.per_regime.get(rn, float("nan")))
            w.writerow(row)


def build_search_summary_md(
    ranked_all: List[AggregatedCandidateResult],
    ranked_heuristics: List[AggregatedCandidateResult],
    ranked_baselines: List[AggregatedCandidateResult],
    n_generated: int,
    n_verified: int,
    n_repaired: int,
    n_failed: int,
    n_duplicates: int,
    regime_names: List[str],
    train_regime_names: List[str],
    val_regime_names: List[str],
) -> str:
    lines = [
        "# Phase 2B.3 Search Summary",
        "",
        "## Generation Stats",
        f"- Candidates requested: {n_generated + n_failed}",
        f"- Generated: {n_generated}",
        f"- Verified OK (first pass): {n_verified - n_repaired}",
        f"- Repaired OK: {n_repaired}",
        f"- Failed: {n_failed}",
        f"- Duplicates removed: {n_duplicates}",
        "",
        "## Evaluation Regimes",
        f"- Train: {', '.join(train_regime_names)}",
        f"- Validation: {', '.join(val_regime_names)}",
        "",
        "## Ranking (by validation priority_weighted_slo_goodput, heuristics only)",
        "",
        "| Rank | Name | Val-WG | Train-WG | Gap | Worst-WG | Beats-Fixed |",
        "|------|------|--------|----------|-----|----------|-------------|",
    ]
    for rank, r in enumerate(ranked_heuristics[:10], 1):
        lines.append(
            f"| {rank} | {r.name[:35]} | {_fmt(r.val_mean_wg)} | "
            f"{_fmt(r.train_mean_wg)} | {_fmt(r.train_val_gap)} | "
            f"{_fmt(r.worst_regime_wg)} | {r.regimes_beating_best_fixed} |"
        )
    if len(ranked_heuristics) > 10:
        lines.append(f"| ... | ({len(ranked_heuristics) - 10} more) | | | | | |")
    lines += [
        "",
        "## Best Baseline Comparison",
        "",
        "| Rank | Name | Val-WG | Train-WG |",
        "|------|------|--------|----------|",
    ]
    for rank, r in enumerate(ranked_baselines[:5], 1):
        lines.append(f"| {rank} | {r.name} | {_fmt(r.val_mean_wg)} | {_fmt(r.train_mean_wg)} |")
    if ranked_heuristics and ranked_baselines:
        best_h = ranked_heuristics[0]
        best_b = ranked_baselines[0]
        delta = (best_h.val_mean_wg - best_b.val_mean_wg
                 if not (math.isnan(best_h.val_mean_wg) or math.isnan(best_b.val_mean_wg))
                 else float("nan"))
        lines += [
            "",
            f"**Best heuristic:** {best_h.name} (val WG={_fmt(best_h.val_mean_wg)})",
            f"**Best baseline:** {best_b.name} (val WG={_fmt(best_b.val_mean_wg)})",
            f"**Delta:** {_fmt(delta)}",
        ]
    lines += [
        "",
        "## Overfitting Analysis",
        "",
        "Candidates with train_val_gap < -0.02 may be overfit to training regimes:",
        "",
    ]
    overfit = [r for r in ranked_heuristics if not math.isnan(r.train_val_gap) and r.train_val_gap < -0.02]
    if overfit:
        for r in overfit:
            lines.append(f"- {r.name}: gap={_fmt(r.train_val_gap):.4f}")
    else:
        lines.append("- None detected.")
    lines += [
        "",
        "## Notes",
        "- Ranking is by **validation** performance, not training.",
        "- Test regimes held out — not used in this phase.",
        "- oracle_srtf excluded from all deployable comparisons.",
        "- RF Selector (Phase 2A.3) trained on 16-policy set; rerun with 18 needed for paper.",
        "- estimated_service_time_first is PARS-inspired proxy, NOT a PARS reproduction.",
    ]
    return "\n".join(lines)
