#!/usr/bin/env python3
"""Rebuild the Step-2 unified utility matrix as v2: Family A and Family B
keep their existing valid Step-2 layer (MF-PSD v1 native + Step-2 v1
cross-family cells, unchanged); Family C is replaced entirely by
CURRENT_RECONSTRUCTED_FAMILY_C_V1 (all 6 anchors, including the 2 native
ones re-evaluated fresh) -- historical Family-C native rows are NEVER
mixed row-wise into this matrix. See docs/design/FAMILY_C_RECONSTRUCTION_V1.md S4.

Pure, deterministic, read-only assembly -- no new simulation. Historical
KV v2 evidence and MF-PSD v1's own frozen Family-C rows are untouched.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation.unified_utility_matrix import CANONICAL_ANCHOR_IDS  # noqa: E402

MF_PSD_LONG = ROOT / "experiments/mf_psd_v1/mf_psd_long_v1.csv"
UUM_V1_LONG = ROOT / "experiments/unified_utility_matrix_v1/unified_utility_matrix_long_v1.csv"
FC_RECON_LONG = ROOT / "experiments/family_c_reconstruction_v1/family_c_reconstruction_v1_long.csv"
OUT_DIR = ROOT / "experiments/unified_utility_matrix_v2"
OUT_LONG = OUT_DIR / "unified_utility_matrix_long_v2.csv"
OUT_WIDE = OUT_DIR / "unified_utility_matrix_wide_v2.csv"

FIELDNAMES = [
    "canonical_scenario_id", "mechanism_family", "canonical_policy_id",
    "primary_utility_anwg", "cell_source", "status",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchors = CANONICAL_ANCHOR_IDS
    cells: dict[tuple[str, str], dict] = {}

    with open(MF_PSD_LONG) as f:
        for r in csv.DictReader(f):
            if r["canonical_policy_id"] not in anchors:
                continue
            if r["mechanism_family"] == "FAMILY_C_KV_PRESSURE_V2":
                continue  # Family C native rows excluded -- replaced by reconstruction v1
            key = (r["canonical_scenario_id"], r["canonical_policy_id"])
            cells[key] = {
                "canonical_scenario_id": r["canonical_scenario_id"],
                "mechanism_family": r["mechanism_family"],
                "canonical_policy_id": r["canonical_policy_id"],
                "primary_utility_anwg": r["primary_utility_anwg"],
                "cell_source": "SOURCE_NATIVE",
                "status": "success",
            }

    with open(UUM_V1_LONG) as f:
        for r in csv.DictReader(f):
            if r["mechanism_family"] == "FAMILY_C_KV_PRESSURE_V2":
                continue  # unsupported placeholders excluded -- replaced by reconstruction v1
            key = (r["canonical_scenario_id"], r["canonical_policy_id"])
            cells[key] = {
                "canonical_scenario_id": r["canonical_scenario_id"],
                "mechanism_family": r["mechanism_family"],
                "canonical_policy_id": r["canonical_policy_id"],
                "primary_utility_anwg": r["primary_utility_anwg"] if r["status"] == "success" else "",
                "cell_source": "STEP2_CROSS_FAMILY_EVALUATION",
                "status": r["status"],
            }

    with open(FC_RECON_LONG) as f:
        for r in csv.DictReader(f):
            sid = r["reconstruction_scenario_id"]  # already "FAMILY_C_KV_PRESSURE_V2::<scenario_id>"
            key = (sid, r["canonical_policy_id"])
            cells[key] = {
                "canonical_scenario_id": sid,
                "mechanism_family": "FAMILY_C_KV_PRESSURE_V2",
                "canonical_policy_id": r["canonical_policy_id"],
                "primary_utility_anwg": r["primary_utility_anwg"] if r["status"] == "success" else "",
                "cell_source": "FAMILY_C_RECONSTRUCTION_V1",
                "status": r["status"],
            }

    with open(OUT_LONG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for key in sorted(cells):
            w.writerow(cells[key])
    print(f"Wrote {OUT_LONG} ({len(cells)} rows).")

    scenario_ids = sorted({sid for (sid, _p) in cells})
    wide_fields = (
        ["canonical_scenario_id", "mechanism_family"]
        + [f"anwg__{p}" for p in anchors]
        + [f"source__{p}" for p in anchors]
        + ["n_anchors_populated"]
    )
    with open(OUT_WIDE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wide_fields)
        w.writeheader()
        for sid in scenario_ids:
            fam = sid.split("::", 1)[0]
            row = {"canonical_scenario_id": sid, "mechanism_family": fam}
            n_pop = 0
            for p in anchors:
                key = (sid, p)
                cell = cells.get(key)
                v = cell["primary_utility_anwg"] if cell else ""
                row[f"anwg__{p}"] = v
                row[f"source__{p}"] = cell["cell_source"] if cell else "MISSING"
                if v != "":
                    n_pop += 1
            row["n_anchors_populated"] = n_pop
            w.writerow(row)
    print(f"Wrote {OUT_WIDE} ({len(scenario_ids)} scenarios x {len(anchors)} anchors).")


if __name__ == "__main__":
    main()
