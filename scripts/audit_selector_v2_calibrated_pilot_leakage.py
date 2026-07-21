#!/usr/bin/env python3
"""Independent leakage audit for a Selector Dataset v2 calibrated-targeted-pilot run.

The pilot's own quality_gates.json reports `no_leakage: passed=true`, but that
check only verifies that each GROUP (source_trace x transform x pool) is
wholly assigned to a single split -- it does not check whether two DIFFERENT
groups (e.g. two different transforms applied to the same underlying trace
rows) draw overlapping row ranges and land in different splits. This script
checks specifically for that cross-transform row-range overlap, which is a
real leakage vector the group-level check cannot catch: the underlying
request content (prompt lengths, base ordering) is identical across a
transform pair, only arrival timing/noise differs.

Usage:
    python scripts/audit_selector_v2_calibrated_pilot_leakage.py \
        --pilot-dir experiments/selector_v2_calibrated_pilot_20260720T163235Z

Writes leakage_audit.json (and prints a summary) into --pilot-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(pilot_dir: Path) -> dict:
    with open(pilot_dir / "retained_windows.csv") as f:
        rows = list(csv.DictReader(f))

    real_rows = [r for r in rows if r["dataset_family"] == "real_trace"]
    by_ancestor = defaultdict(list)
    for r in real_rows:
        by_ancestor[r["request_plan_ancestor_id"]].append(r)

    cross_split_pairs = []
    pool_violations = []
    affected_windows = set()

    for ancestor, group in by_ancestor.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                a_start, a_end = int(a["time_slice_row_start"]), int(a["time_slice_row_end"])
                b_start, b_end = int(b["time_slice_row_start"]), int(b["time_slice_row_end"])
                overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
                if overlap <= 0:
                    continue
                if a["time_slice_pool"] != b["time_slice_pool"]:
                    pool_violations.append(
                        {"a_window": a["window_idx"], "a_pool": a["time_slice_pool"],
                         "b_window": b["window_idx"], "b_pool": b["time_slice_pool"]}
                    )
                if a["split"] != b["split"]:
                    cross_split_pairs.append(
                        {
                            "ancestor": ancestor,
                            "a_window": a["window_idx"], "a_split": a["split"], "a_group": a["group_key"],
                            "b_window": b["window_idx"], "b_split": b["split"], "b_group": b["group_key"],
                            "row_range": [a_start, a_end],
                            "overlap_rows": overlap,
                        }
                    )
                    affected_windows.add(a["window_idx"])
                    affected_windows.add(b["window_idx"])

    split_pair_counts = Counter(
        tuple(sorted([p["a_split"], p["b_split"]])) for p in cross_split_pairs
    )
    n_historical_real_trace = sum(1 for r in real_rows if r["time_slice_pool"] == "historical")

    if cross_split_pairs:
        conclusion = (
            "FAIL: cross-transform row-range reuse crosses split boundaries. "
            "VALIDATION and ID_TEST results should not be treated as clean "
            "held-out evaluation until split construction is fixed."
        )
    else:
        conclusion = (
            "PASS: no cross-split raw row-range overlap was found among "
            "real-trace windows. OOD pool isolation also holds if "
            "historical_ood_pool_violations is zero."
        )

    result = {
        "pilot_dir": str(pilot_dir),
        "n_total_windows": len(rows),
        "n_real_trace_windows": len(real_rows),
        "n_real_trace_historical_windows": n_historical_real_trace,
        "cross_split_row_overlap_pairs": len(cross_split_pairs),
        "distinct_windows_involved": sorted(affected_windows, key=int),
        "n_distinct_windows_involved": len(affected_windows),
        "split_pair_breakdown": {"-".join(k): v for k, v in split_pair_counts.items()},
        "ood_test_involved": any("OOD_TEST" in (p["a_split"], p["b_split"]) for p in cross_split_pairs),
        "historical_ood_pool_violations": len(pool_violations),
        "conclusion": conclusion,
        "detail_pairs": cross_split_pairs,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-dir", required=True, type=Path)
    args = p.parse_args()

    result = audit(args.pilot_dir)
    out_path = args.pilot_dir / "leakage_audit.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"Cross-split row-overlap pairs: {result['cross_split_row_overlap_pairs']}")
    print(f"Distinct windows involved: {result['n_distinct_windows_involved']} "
          f"(of {result['n_real_trace_historical_windows']} real-trace historical windows)")
    print(f"Split-pair breakdown: {result['split_pair_breakdown']}")
    print(f"OOD_TEST involved: {result['ood_test_involved']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
