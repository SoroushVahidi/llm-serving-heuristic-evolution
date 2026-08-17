#!/usr/bin/env python3
"""CLI for Step 2: unified cross-family policy-utility matrix (v1).

See docs/design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md for the frozen
preregistration. Reads MF-PSD v1 (read-only), regenerates Family A/B
scenarios (verified byte-exact reconstruction), evaluates the 4 non-native
canonical anchors on each, and emits explicit unsupported placeholder rows
for Family C (scenario reconstruction confirmed not byte-exact -- not
evaluated). Resume-safe: re-running with the same --out-dir skips any
uum_row_id already present in the output CSV.

Data generation only. No selector, no composition, no mechanism attribution.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation.unified_utility_matrix import (  # noqa: E402
    ALL_FAMILIES,
    BLOCKED_TARGET_FAMILIES,
    BUILDER_VERSION,
    CANONICAL_ANCHOR_IDS,
    NATIVE_FAMILY_OF_POLICY,
    RESULT_FIELDNAMES,
    _git_dirty,
    _git_sha,
    _sha256_file,
    regenerate_family_a_scenarios,
    regenerate_family_b_scenarios,
    run_cell,
    unsupported_row,
)

MF_PSD_DIR = ROOT / "experiments" / "mf_psd_v1"
DEFAULT_OUT_DIR = ROOT / "experiments" / "unified_utility_matrix_v1"


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def _load_mf_psd_scenarios() -> List[Dict[str, str]]:
    rows = []
    with open(MF_PSD_DIR / "mf_psd_scenarios_v1.csv") as f:
        for row in csv.DictReader(f):
            rows.append({
                "canonical_scenario_id": row["canonical_scenario_id"],
                "source_scenario_id": row["source_scenario_id"],
                "mechanism_family": row["mechanism_family"],
            })
    return rows


def _worker(args):
    scenario, canonical_policy_id, mechanism_family = args
    return run_cell(scenario, canonical_policy_id, mechanism_family)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Only run 1 scenario per non-native anchor per family (fast validation subset).")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "unified_utility_matrix_long_v1.csv"
    manifest_path = out_dir / "unified_utility_matrix_build_manifest_v1.json"

    _log(out_dir, f"Starting Step-2 unified utility matrix build (smoke={args.smoke}).")
    _log(out_dir, f"git_head={_git_sha()} dirty={_git_dirty()}")

    # ---- 0. MF-PSD integrity: read-only, checksums must be untouched ----
    mf_psd_rows = _load_mf_psd_scenarios()
    by_family: Dict[str, List[Dict[str, str]]] = {fam: [] for fam in ALL_FAMILIES}
    for r in mf_psd_rows:
        by_family[r["mechanism_family"]].append(r)
    _log(out_dir, f"MF-PSD v1 scenario table: {len(mf_psd_rows)} rows "
                  f"(A={len(by_family['FAMILY_A_FAIRNESS_STARVATION_V2'])}, "
                  f"B={len(by_family['FAMILY_B_PREFILL_DECODE_V2'])}, "
                  f"C={len(by_family['FAMILY_C_KV_PRESSURE_V2'])})")

    # ---- 1. Regenerate Family A / B scenarios, verify scenario_id sets match ----
    _log(out_dir, "Regenerating Family A v2 scenarios...")
    fa_scenarios = regenerate_family_a_scenarios()
    fa_ids = {s.scenario_id for s in fa_scenarios}
    expected_fa_ids = {r["source_scenario_id"] for r in by_family["FAMILY_A_FAIRNESS_STARVATION_V2"]}
    if fa_ids != expected_fa_ids:
        raise SystemExit(
            f"Family A scenario_id mismatch: regenerated {len(fa_ids)}, "
            f"expected {len(expected_fa_ids)}, symdiff={len(fa_ids ^ expected_fa_ids)}"
        )
    _log(out_dir, f"Family A: regenerated {len(fa_scenarios)} scenarios, IDs match MF-PSD exactly.")

    _log(out_dir, "Regenerating Family B v2 scenarios...")
    fb_scenarios = regenerate_family_b_scenarios()
    fb_ids = {s.scenario_id for s in fb_scenarios}
    expected_fb_ids = {r["source_scenario_id"] for r in by_family["FAMILY_B_PREFILL_DECODE_V2"]}
    if fb_ids != expected_fb_ids:
        raise SystemExit(
            f"Family B scenario_id mismatch: regenerated {len(fb_ids)}, "
            f"expected {len(expected_fb_ids)}, symdiff={len(fb_ids ^ expected_fb_ids)}"
        )
    _log(out_dir, f"Family B: regenerated {len(fb_scenarios)} scenarios, IDs match MF-PSD exactly.")

    if args.smoke:
        fa_scenarios = fa_scenarios[:1]
        fb_scenarios = fb_scenarios[:1]

    scenario_lookup = {"FAMILY_A_FAIRNESS_STARVATION_V2": fa_scenarios, "FAMILY_B_PREFILL_DECODE_V2": fb_scenarios}
    id_to_canonical = {
        (r["mechanism_family"], r["source_scenario_id"]): r["canonical_scenario_id"]
        for r in mf_psd_rows
    }

    # ---- 2. Build task list per the frozen executability audit ----
    real_tasks = []  # (scenario, canonical_policy_id, mechanism_family)
    unsupported_rows_list = []
    for family in ALL_FAMILIES:
        for policy_id in CANONICAL_ANCHOR_IDS:
            if NATIVE_FAMILY_OF_POLICY[policy_id] == family:
                continue
            if family in BLOCKED_TARGET_FAMILIES:
                for r in by_family[family]:
                    unsupported_rows_list.append(unsupported_row(
                        r["canonical_scenario_id"], r["source_scenario_id"], family, policy_id
                    ))
                continue
            for scen in scenario_lookup[family]:
                real_tasks.append((scen, policy_id, family))

    _log(out_dir, f"Planned: {len(real_tasks)} real cells, {len(unsupported_rows_list)} unsupported placeholder cells "
                  f"(expected 416/288 at full scale, smoke={args.smoke}).")

    # ---- 3. Resume: skip already-completed uum_row_ids ----
    done_ids: Set[str] = set()
    write_header = not out_csv.exists()
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                done_ids.add(row["uum_row_id"])
        _log(out_dir, f"Resuming: {len(done_ids)} rows already present in {out_csv.name}.")

    csv_f = open(out_csv, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=RESULT_FIELDNAMES)
    if write_header:
        writer.writeheader()
        csv_f.flush()

    # Unsupported rows are cheap; write any not already done, immediately.
    n_written_unsupported = 0
    for row in unsupported_rows_list:
        if row["uum_row_id"] in done_ids:
            continue
        writer.writerow(row)
        done_ids.add(row["uum_row_id"])
        n_written_unsupported += 1
    csv_f.flush()
    _log(out_dir, f"Wrote {n_written_unsupported} new unsupported placeholder rows.")

    pending_tasks = []
    for scen, policy_id, family in real_tasks:
        canonical_scenario_id = id_to_canonical[(family, scen.scenario_id)]
        uum_row_id = f"{canonical_scenario_id}::{policy_id}"
        if uum_row_id in done_ids:
            continue
        pending_tasks.append((scen, policy_id, family))

    _log(out_dir, f"{len(pending_tasks)} real cells pending (of {len(real_tasks)} planned).")

    n_success, n_failed = 0, 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, t): t for t in pending_tasks}
        n_done = 0
        for fut in as_completed(futures):
            row = fut.result()
            writer.writerow(row)
            csv_f.flush()
            n_done += 1
            if row["status"] == "success":
                n_success += 1
            else:
                n_failed += 1
                _log(out_dir, f"FAILED {row['uum_row_id']}: {row['error'][:200]}")
            if n_done % 20 == 0 or n_done == len(pending_tasks):
                elapsed = time.time() - t0
                _log(out_dir, f"progress: {n_done}/{len(pending_tasks)} cells "
                              f"({n_success} success, {n_failed} failed) elapsed={elapsed:.1f}s")

    csv_f.close()
    _log(out_dir, f"Done. This invocation: {n_success} success, {n_failed} failed "
                  f"(real cells), {n_written_unsupported} unsupported placeholders written.")

    manifest = {
        "builder_version": BUILDER_VERSION,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_git_head_sha": _git_sha(),
        "build_git_dirty": _git_dirty(),
        "command_line": " ".join(sys.argv),
        "smoke": args.smoke,
        "workers": args.workers,
        "mf_psd_provenance_sha256": _sha256_file(MF_PSD_DIR / "mf_psd_provenance_v1.json"),
        "mf_psd_scenarios_sha256": _sha256_file(MF_PSD_DIR / "mf_psd_scenarios_v1.csv"),
        "mf_psd_long_sha256": _sha256_file(MF_PSD_DIR / "mf_psd_long_v1.csv"),
        "config_sha256": {
            "policy_separation_fairness_starvation_pilot_v2.yaml": _sha256_file(
                ROOT / "configs/policy_separation_fairness_starvation_pilot_v2.yaml"),
            "policy_separation_prefill_decode_pilot_v2.yaml": _sha256_file(
                ROOT / "configs/policy_separation_prefill_decode_pilot_v2.yaml"),
        },
        "expected_real_cells": len(real_tasks),
        "expected_unsupported_cells": len(unsupported_rows_list),
        "this_invocation_success": n_success,
        "this_invocation_failed": n_failed,
        "this_invocation_unsupported_written": n_written_unsupported,
        "total_rows_in_output_after_this_run": len(done_ids) + n_success + n_failed,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    _log(out_dir, f"Manifest written to {manifest_path}.")


if __name__ == "__main__":
    main()
