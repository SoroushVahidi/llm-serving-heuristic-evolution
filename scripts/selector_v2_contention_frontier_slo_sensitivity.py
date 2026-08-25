#!/usr/bin/env python3
"""SLO-threshold sensitivity analysis for the mechanism-active windows
found by `selector_v2_contention_frontier_search.py`.

Does NOT re-run any simulation and does NOT change the historical
`arrival_normalized_weighted_goodput` definition or any Request's
`slo_deadline` field. Instead, post-hoc, for each already-saved
mechanism-active window's raw per-policy (arrival_time, admission_time,
completion_time) records, recomputes an ANWG-LIKE proxy
(`completed_and_within_synthetic_deadline / n_total_requests`, unweighted
since every generated request here has priority=1.0) under a grid of
SYNTHETIC deadlines derived from that window's OWN observed latency scale
(median completion latency across all policies, pooled) -- never tuned
per-window to manufacture a winner, the same multiplier grid is applied
to every window. This answers section 5's question: are the latency
differences the frontier search found (p95_latency NEAR_TIE, mostly from
`admission_reorder` windows) large enough to become utility differences
once SLO thresholds are stressed, and at what tightness does that happen.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "experiments" / "selector_v2_contention_frontier_search" / "raw_latencies"
OUT_CSV = ROOT / "experiments" / "selector_v2_contention_frontier_search" / "slo_sensitivity.csv"

# Multiplier grid applied uniformly to every window's own observed median
# latency -- k=1.0 means "deadline == the scale of what was actually
# observed", not an arbitrarily loose or tight external constant.
K_GRID = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0]
PRACTICAL_EQUIVALENCE_ABS = 0.002  # matches discriminativeness.py's own threshold


def _anwg_like(records, deadline_by_arrival, n_total: int) -> float:
    if n_total == 0:
        return float("nan")
    success = 0
    for arrival, _admission, completion in records:
        deadline = deadline_by_arrival(arrival)
        if completion <= deadline:
            success += 1
    return success / n_total


def main() -> int:
    rows_out = []
    n_windows_with_divergence_at = {k: 0 for k in K_GRID}
    tightest_k_for_divergence = []

    for path in sorted(RAW_DIR.glob("window_*.json")):
        window_idx = int(path.stem.split("_")[1])
        by_policy = json.loads(path.read_text())
        all_latencies = [
            (completion - arrival)
            for records in by_policy.values()
            for arrival, _admission, completion in records
        ]
        if not all_latencies:
            continue
        all_latencies.sort()
        median_latency = all_latencies[len(all_latencies) // 2]
        if median_latency <= 0:
            continue
        n_total = max(len(records) for records in by_policy.values())

        first_divergence_k = None
        for k in K_GRID:
            ref = k * median_latency

            def deadline_by_arrival(arrival, _ref=ref):
                return arrival + _ref

            values = {}
            for pname, records in by_policy.items():
                values[pname] = _anwg_like(records, deadline_by_arrival, n_total)
            spread = max(values.values()) - min(values.values())
            diverges = spread > PRACTICAL_EQUIVALENCE_ABS
            if diverges:
                n_windows_with_divergence_at[k] += 1
                if first_divergence_k is None:
                    first_divergence_k = k
            best_policy = max(values, key=lambda p: values[p])
            rows_out.append(dict(
                window_idx=window_idx, k=k, median_latency=round(median_latency, 6),
                anwg_like_spread=round(spread, 6), diverges=diverges, best_policy=best_policy,
            ))
        if first_divergence_k is not None:
            tightest_k_for_divergence.append(first_divergence_k)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = sorted({k for row in rows_out for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    n_windows = len(set(r["window_idx"] for r in rows_out))
    summary = {
        "n_windows_analyzed": n_windows,
        "k_grid": K_GRID,
        "n_windows_diverging_at_k": n_windows_with_divergence_at,
        "fraction_windows_diverging_at_k": {
            k: round(v / n_windows, 4) if n_windows else 0.0 for k, v in n_windows_with_divergence_at.items()
        },
        "n_windows_ever_diverging": len(tightest_k_for_divergence),
        "tightest_k_distribution": {
            str(k): tightest_k_for_divergence.count(k) for k in sorted(set(tightest_k_for_divergence))
        },
    }
    (OUT_CSV.parent / "slo_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
