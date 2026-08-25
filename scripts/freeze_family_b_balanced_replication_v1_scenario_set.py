#!/usr/bin/env python3
"""Materialize and freeze the Family-B-Balanced Replication v1 36-scenario
selection (design doc SS 10 of HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1.md,
completed by docs/design/FAMILY_B_BALANCED_REPLICATION_V1.md).

METADATA-ONLY. Reads `mf_psd_scenarios_v1.csv`'s scenario metadata (family,
group_key, canonical_scenario_id, seed, split) and the frozen
`build_splits` split assignment. Never reads or reasons about any ANWG/
utility/policy-performance column. Does not fit models, does not invoke
the simulator or live harness, and computes no scientific result -- this
is preregistration bookkeeping only.

Idempotent and deterministic: running this script twice produces a
byte-identical output file.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from llmserveopt.policy_separation.hierarchical_regime_router_v1 import build_splits  # noqa: E402
from llmserveopt.policy_separation.family_b_balanced_replication_v1 import (  # noqa: E402
    FAMILY_A,
    FAMILY_B,
    FAMILY_C,
    FAMILY_C_PRIMARY_HELD_OUT_SEED,
    replication_family_counts,
    select_balanced_replication_set,
    verify_no_train_leakage,
)

MF_PSD_SCENARIOS = REPO_ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"
OUTPUT_DIR = REPO_ROOT / "experiments/family_b_balanced_replication_v1"
OUTPUT_PATH = OUTPUT_DIR / "frozen_scenario_selection_v1.json"


def _git_head_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _sha256_of_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    scen = pd.read_csv(MF_PSD_SCENARIOS)
    split_map = build_splits(scen)
    scen["split"] = scen["canonical_scenario_id"].map(split_map)

    replication_set = select_balanced_replication_set(scen)
    verify_no_train_leakage(scen, replication_set)

    counts = replication_family_counts(replication_set)
    assert counts == {FAMILY_A: 12, FAMILY_B: 12, FAMILY_C: 12}, counts

    out = {
        "schema_version": "family_b_balanced_replication_v1.1.0.0",
        "purpose": (
            "Frozen 36-scenario (12/12/12) held-out replication set for the "
            "Family-B-Balanced Replication, completing design doc SS 10 of "
            "HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1.md. Selection is "
            "purely metadata-driven (split assignment + canonical_scenario_id "
            "sort order); no ANWG/utility column was read to produce this "
            "selection."
        ),
        "freeze_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "freeze_git_head_sha": _git_head_sha(),
        "source_mf_psd_scenarios_sha256": _sha256_of_file(MF_PSD_SCENARIOS),
        "family_c_primary_held_out_seed": FAMILY_C_PRIMARY_HELD_OUT_SEED,
        "target_per_family": 12,
        "family_counts": counts,
        "scenario_ids_by_family": {
            fam: sorted(replication_set[replication_set["mechanism_family"] == fam]["canonical_scenario_id"].tolist())
            for fam in (FAMILY_A, FAMILY_B, FAMILY_C)
        },
        "split_provenance_by_family": {
            fam: sorted(replication_set[replication_set["mechanism_family"] == fam]["split"].value_counts().to_dict().items())
            for fam in (FAMILY_A, FAMILY_B, FAMILY_C)
        },
        "primary_test_overlap_count_by_family": {
            fam: int(
                (
                    (replication_set["mechanism_family"] == fam)
                    & (replication_set["canonical_scenario_id"].isin(scen[scen["split"] == "test"]["canonical_scenario_id"]))
                ).sum()
            )
            for fam in (FAMILY_A, FAMILY_B, FAMILY_C)
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    print(f"\nWrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
