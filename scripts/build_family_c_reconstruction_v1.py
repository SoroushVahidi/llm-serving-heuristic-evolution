#!/usr/bin/env python3
"""CLI for Family C Reconstruction v1.

See docs/design/FAMILY_C_RECONSTRUCTION_V1.md. Generates the 72 Family-C
scenarios ONCE, serializes full request-level content to disk, verifies the
serialization round-trips exactly, then evaluates all 6 canonical anchors
by replaying from the frozen serialization (never by re-calling the
generator). Resume-safe: re-running with the same --out-dir skips any
reconstruction_scenario_id::canonical_policy_id pair already present.

Data generation only. No selector, no composition, no mechanism attribution.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation import family_c_reconstruction_v1 as fc  # noqa: E402
from llmserveopt.policy_separation import unified_utility_matrix as uum  # noqa: E402

MF_PSD_DIR = ROOT / "experiments" / "mf_psd_v1"
DEFAULT_OUT_DIR = ROOT / "experiments" / "family_c_reconstruction_v1"


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def _worker(args):
    scenario, canonical_policy_id = args
    return fc.run_cell_reconstruction(scenario, canonical_policy_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Only run 3 scenarios x all 6 policies (fast validation subset).")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios_path = out_dir / "family_c_reconstruction_v1_scenarios.jsonl"
    out_csv = out_dir / "family_c_reconstruction_v1_long.csv"
    manifest_path = out_dir / "family_c_reconstruction_v1_build_manifest.json"

    _log(out_dir, f"Starting Family C Reconstruction v1 build (smoke={args.smoke}).")
    _log(out_dir, f"git_head={uum._git_sha()} dirty={uum._git_dirty()}")  # noqa: SLF001

    # ---- 1. Generate the 72 scenarios ONCE ----
    _log(out_dir, "Generating Family C v2 scenarios (single call)...")
    scenarios = fc.regenerate_family_c_scenarios()
    ids = {s.scenario_id for s in scenarios}
    with open(MF_PSD_DIR / "mf_psd_scenarios_v1.csv") as f:
        expected_ids = {
            row["source_scenario_id"] for row in csv.DictReader(f)
            if row["mechanism_family"] == "FAMILY_C_KV_PRESSURE_V2"
        }
    if len(scenarios) != 72 or ids != expected_ids:
        raise SystemExit(
            f"Family C scenario_id mismatch: generated {len(scenarios)}, "
            f"expected 72; symdiff vs MF-PSD={len(ids ^ expected_ids)}"
        )
    _log(out_dir, f"Generated {len(scenarios)} scenarios, IDs match MF-PSD v1's Family-C set exactly.")

    # ---- 2. Serialize full request-level content ----
    fc.serialize_scenarios(scenarios, scenarios_path)
    scenarios_sha256 = uum._sha256_file(scenarios_path)  # noqa: SLF001
    _log(out_dir, f"Serialized to {scenarios_path.name} (sha256={scenarios_sha256}).")

    # ---- 3. Verify deterministic replay: reload and check exact equality ----
    reloaded = fc.load_serialized_scenarios(scenarios_path)
    if len(reloaded) != len(scenarios):
        raise SystemExit(f"Reload count mismatch: {len(reloaded)} vs {len(scenarios)}")
    by_id_orig = {s.scenario_id: s for s in scenarios}
    for r in reloaded:
        orig = by_id_orig[r.scenario_id]
        if tuple(asdict(x) for x in r.requests) != tuple(asdict(x) for x in orig.requests):
            raise SystemExit(f"Replay mismatch (requests) on {r.scenario_id}")
        if tuple(asdict(x) for x in r.gpu_configs) != tuple(asdict(x) for x in orig.gpu_configs):
            raise SystemExit(f"Replay mismatch (gpu_configs) on {r.scenario_id}")
        if r.service_model_kwargs != orig.service_model_kwargs or r.seed != orig.seed:
            raise SystemExit(f"Replay mismatch (config) on {r.scenario_id}")
    _log(out_dir, f"Replay verified: {len(reloaded)}/{len(reloaded)} scenarios reconstruct exactly.")

    if args.smoke:
        reloaded = reloaded[:3]

    # ---- 4. Build task list: all 6 anchors x all reloaded scenarios ----
    tasks = [(s, p) for s in reloaded for p in uum.CANONICAL_ANCHOR_IDS]
    _log(out_dir, f"Planned: {len(tasks)} cells (expect 432 at full scale, smoke={args.smoke}).")

    # ---- 5. Resume-safe execution ----
    done_ids: Set[str] = set()
    write_header = not out_csv.exists()
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                done_ids.add(f"{row['reconstruction_scenario_id']}::{row['canonical_policy_id']}")
        _log(out_dir, f"Resuming: {len(done_ids)} rows already present.")

    csv_f = open(out_csv, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fc.RESULT_FIELDNAMES)
    if write_header:
        writer.writeheader()
        csv_f.flush()

    pending = []
    for s, p in tasks:
        key = f"{fc.SOURCE_FAMILY}::{s.scenario_id}::{p}"
        if key in done_ids:
            continue
        pending.append((s, p))
    _log(out_dir, f"{len(pending)} cells pending (of {len(tasks)} planned).")

    n_success, n_failed = 0, 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, t): t for t in pending}
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
                _log(out_dir, f"FAILED {row['reconstruction_scenario_id']}::{row['canonical_policy_id']}: {row['error'][:200]}")
            if n_done % 20 == 0 or n_done == len(pending):
                elapsed = time.time() - t0
                _log(out_dir, f"progress: {n_done}/{len(pending)} cells "
                              f"({n_success} success, {n_failed} failed) elapsed={elapsed:.1f}s")

    csv_f.close()
    _log(out_dir, f"Done. This invocation: {n_success} success, {n_failed} failed.")

    burstgpt_path = fc.uum.ROOT / ".local_data" / "burstgpt_v2" / "raw" / "BurstGPT_without_fails_1.csv"
    manifest = {
        "reconstruction_version": fc.RECONSTRUCTION_VERSION,
        "builder_version": uum.BUILDER_VERSION,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_git_head_sha": uum._git_sha(),  # noqa: SLF001
        "build_git_dirty": uum._git_dirty(),  # noqa: SLF001
        "command_line": " ".join(sys.argv),
        "smoke": args.smoke,
        "workers": args.workers,
        "config_sha256": uum._sha256_file(ROOT / "configs/kv_pressure_pilot_v2.yaml"),  # noqa: SLF001
        "burstgpt_dataset_path": str(burstgpt_path),
        "burstgpt_dataset_sha256": uum._sha256_file(burstgpt_path),  # noqa: SLF001
        "scenarios_jsonl_sha256": scenarios_sha256,
        "n_scenarios": len(scenarios),
        "expected_cells": len(tasks),
        "this_invocation_success": n_success,
        "this_invocation_failed": n_failed,
        "total_rows_after_this_run": len(done_ids) + n_success + n_failed,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    _log(out_dir, f"Manifest written to {manifest_path}.")


if __name__ == "__main__":
    main()
