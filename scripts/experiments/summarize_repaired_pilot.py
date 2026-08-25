#!/usr/bin/env python3
"""Summarize an existing repaired-pilot root into a Git-safe compact JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEEP_OVERALL = {
    "n",
    "saturated_rate",
    "exact_tie_rate",
    "near_tie_rate",
    "mean_margin",
    "median_margin",
    "p75_margin",
    "p90_margin",
    "n_effective_winner_classes",
    "winner_counts",
    "behavioral_disagreement_rate",
    "best_fixed_policy",
    "best_fixed_mean_anwg",
    "oracle_envelope_mean_anwg",
    "oracle_gain_over_best_fixed",
}


def compact(summary: dict) -> dict:
    overall = summary.get("overall") or {}
    return {
        "schema_version": "pause_repaired_pilot_summary_v1",
        "slurm_job_id": summary.get("slurm_job_id"),
        "git_sha": summary.get("git_sha"),
        "decision": summary.get("decision"),
        "policies": summary.get("policies"),
        "selection_counts": summary.get("selection_counts"),
        "overall": {k: overall[k] for k in KEEP_OVERALL if k in overall},
        "readiness_gates": summary.get("readiness_gates"),
        "tie_cause_histogram": summary.get("tie_cause_histogram"),
        "diagnostic_limitations": {
            "behavioral_disagreement_uses_outcome_signatures_not_action_traces": True,
            "tie_cause_labels_are_heuristic": True,
            "full_fingerprint_sweep_authorized": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    src = args.pilot_root / "reports" / "repaired_pilot_summary.json"
    summary = json.loads(src.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(compact(summary), indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
