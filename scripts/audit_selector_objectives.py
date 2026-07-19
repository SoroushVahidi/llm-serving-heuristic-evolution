#!/usr/bin/env python3
"""Re-score existing Selector Dataset v2 CSVs under audited objectives."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.selector.dataset_v2.objective_analysis import (
    ARRIVAL_NORMALIZED_OBJECTIVE,
    COMPLETION_ADJUSTED_OBJECTIVE,
    CONSTRAINED_ARRIVAL_NORMALIZED_OBJECTIVE,
    HISTORICAL_CONDITIONAL_OBJECTIVE,
    SLO_SUCCESS_THROUGHPUT_OBJECTIVE,
    ConstrainedRankingConfig,
    objective_summary,
    selective_service_advantages,
    sensitivity_grid,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Flattened Dataset v2 CSV path(s).")
    parser.add_argument("--output", required=True, help="JSON report path.")
    parser.add_argument("--constraint-completion", type=float, default=0.8)
    parser.add_argument("--constraint-rejection", type=float, default=0.2)
    args = parser.parse_args()

    datasets = {}
    combined = []
    for path_s in args.input:
        path = Path(path_s)
        rows = list(csv.DictReader(open(path)))
        dataset_key = f"{path.parent.name}/{path.name}"
        for row in rows:
            combined_row = dict(row)
            combined_row["scenario_id"] = f"{dataset_key}::{combined_row['scenario_id']}"
            combined.append(combined_row)
        datasets[dataset_key] = _summarize_rows(rows, args)

    report = {
        "datasets": datasets,
        "combined": _summarize_rows(combined, args) if len(args.input) > 1 else None,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps({
        "output": str(out),
        "datasets": list(datasets),
        "combined_rows": len(combined),
    }, indent=2, sort_keys=True))
    return 0


def _summarize_rows(rows: list[dict], args: argparse.Namespace) -> dict:
    constrained = ConstrainedRankingConfig(args.constraint_completion, args.constraint_rejection)
    objectives = [
        HISTORICAL_CONDITIONAL_OBJECTIVE,
        ARRIVAL_NORMALIZED_OBJECTIVE,
        COMPLETION_ADJUSTED_OBJECTIVE,
        SLO_SUCCESS_THROUGHPUT_OBJECTIVE,
    ]
    summaries = {
        name: objective_summary(rows, name)
        for name in objectives
    }
    summaries[CONSTRAINED_ARRIVAL_NORMALIZED_OBJECTIVE] = objective_summary(
        rows,
        CONSTRAINED_ARRIVAL_NORMALIZED_OBJECTIVE,
        config=constrained,
    )
    return {
        "num_rows": len(rows),
        "num_windows": len({(r["scenario_id"], r["window_id"]) for r in rows}),
        "objective_summaries": summaries,
        "constraint_sensitivity": sensitivity_grid(rows),
        "selective_service_advantages": selective_service_advantages(rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
