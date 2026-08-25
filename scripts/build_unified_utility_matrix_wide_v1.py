#!/usr/bin/env python3
"""Derive the wide-form Step-2 utility matrix (one row per scenario, one
column per canonical anchor) from MF-PSD v1's frozen native cells plus
experiments/unified_utility_matrix_v1/unified_utility_matrix_long_v1.csv.

Pure, deterministic, read-only transform -- no new simulation, no scoring
decision. See docs/design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md S11.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation.unified_utility_matrix import CANONICAL_ANCHOR_IDS  # noqa: E402

MF_PSD_LONG = ROOT / "experiments/mf_psd_v1/mf_psd_long_v1.csv"
UUM_LONG = ROOT / "experiments/unified_utility_matrix_v1/unified_utility_matrix_long_v1.csv"
OUT = ROOT / "experiments/unified_utility_matrix_v1/unified_utility_matrix_wide_v1.csv"


def main() -> None:
    anchors = CANONICAL_ANCHOR_IDS
    cell_val: dict[tuple[str, str], str] = {}
    cell_src: dict[tuple[str, str], str] = {}

    with open(MF_PSD_LONG) as f:
        for r in csv.DictReader(f):
            if r["canonical_policy_id"] in anchors:
                key = (r["canonical_scenario_id"], r["canonical_policy_id"])
                cell_val[key] = r["primary_utility_anwg"]
                cell_src[key] = "SOURCE_NATIVE"

    with open(UUM_LONG) as f:
        for r in csv.DictReader(f):
            key = (r["canonical_scenario_id"], r["canonical_policy_id"])
            cell_val[key] = r["primary_utility_anwg"] if r["status"] == "success" else ""
            cell_src[key] = r["status"]

    scenario_ids = sorted({s for (s, _p) in cell_val.keys()})
    fieldnames = (
        ["canonical_scenario_id", "mechanism_family"]
        + [f"anwg__{p}" for p in anchors]
        + [f"source__{p}" for p in anchors]
        + ["n_anchors_populated"]
    )
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sid in scenario_ids:
            fam = sid.split("::", 1)[0]
            row = {"canonical_scenario_id": sid, "mechanism_family": fam}
            n_pop = 0
            for p in anchors:
                key = (sid, p)
                v = cell_val.get(key, "")
                row[f"anwg__{p}"] = v
                row[f"source__{p}"] = cell_src.get(key, "MISSING")
                if v != "":
                    n_pop += 1
            row["n_anchors_populated"] = n_pop
            w.writerow(row)

    print(f"Wrote {OUT} ({len(scenario_ids)} scenarios x {len(anchors)} anchors).")


if __name__ == "__main__":
    main()
