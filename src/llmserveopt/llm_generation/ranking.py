"""
Rank candidate results by priority_weighted_slo_goodput with standard tie-breaks.

Ranking rule:
  1. higher priority_weighted_slo_goodput (primary)
  2. lower slo_violation_rate
  3. lower p95_ttft
  4. lower p95_latency
  5. higher request_throughput
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

from .evaluation import CandidateResult

_NAN = float("nan")


def _sort_key(r: CandidateResult):
    wg = r.priority_weighted_slo_goodput
    vr = r.slo_violation_rate
    ttft = r.p95_ttft
    lat = r.p95_latency
    tput = r.request_throughput
    # Replace NaN with worst-case values so failed candidates sort to the bottom
    wg = -1.0 if math.isnan(wg) else wg
    vr = 2.0 if math.isnan(vr) else vr
    ttft = 1e9 if math.isnan(ttft) else ttft
    lat = 1e9 if math.isnan(lat) else lat
    tput = 0.0 if math.isnan(tput) else tput
    return (-wg, vr, ttft, lat, -tput)


def rank_candidates(results: List[CandidateResult]) -> List[CandidateResult]:
    """Return results sorted best-first."""
    return sorted(results, key=_sort_key)


def save_ranking_csv(ranked: List[CandidateResult], output_path: Path) -> None:
    fields = [
        "rank", "name", "source", "priority_weighted_slo_goodput",
        "weighted_goodput", "slo_violation_rate", "p95_ttft", "p95_latency",
        "request_throughput", "num_completed", "error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, r in enumerate(ranked, 1):
            w.writerow({
                "rank": rank,
                "name": r.name,
                "source": r.source,
                "priority_weighted_slo_goodput": _fmt(r.priority_weighted_slo_goodput),
                "weighted_goodput": _fmt(r.weighted_goodput),
                "slo_violation_rate": _fmt(r.slo_violation_rate),
                "p95_ttft": _fmt(r.p95_ttft),
                "p95_latency": _fmt(r.p95_latency),
                "request_throughput": _fmt(r.request_throughput),
                "num_completed": r.num_completed,
                "error": r.error or "",
            })


def _fmt(v: float) -> object:
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, 6)


def build_summary_md(
    ranked_all: List[CandidateResult],
    n_generated: int,
    n_verified: int,
    n_repaired: int,
    n_failed: int,
) -> str:
    lines = [
        "# Phase 2B.2 Evaluation Summary",
        "",
        "## Generation Stats",
        f"- Generated: {n_generated}",
        f"- Verified OK (first pass): {n_verified - n_repaired}",
        f"- Repaired OK: {n_repaired}",
        f"- Failed: {n_failed}",
        "",
        "## Ranking (all candidates + baselines by priority_weighted_slo_goodput)",
        "",
        "| Rank | Name | Source | Pr-WG-SLO | VR | p95_lat |",
        "|------|------|--------|-----------|-----|---------|",
    ]
    for rank, r in enumerate(ranked_all, 1):
        wg = _fmt(r.priority_weighted_slo_goodput)
        vr = _fmt(r.slo_violation_rate)
        lat = _fmt(r.p95_latency)
        lines.append(
            f"| {rank} | {r.name} | {r.source} | {wg} | {vr} | {lat} |"
        )
    lines += [
        "",
        "## Note on oracle_srtf",
        "oracle_srtf is NOT included as a deployable baseline. It may be run",
        "separately as a hindsight upper bound and must always be labeled as such.",
        "",
        "## Note on RF Selector",
        "RF Selector (Phase 2A.4) trained on 18-policy candidate set (52 windows: 30 train / 13 val / 9 test).",
    ]
    return "\n".join(lines)
