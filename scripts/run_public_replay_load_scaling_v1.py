#!/usr/bin/env python3
"""Public Replay Load Scaling v1 -- canonical runner.

Executes the frozen preregistration in
`docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md` over the 60 canonical
augmented-view public-trace windows already produced by
`llmserveopt.policy_separation.public_trace_replay_v1.build_all_scenarios()`,
scaling ONLY inter-arrival timing by a preregistered load factor lambda in
{1,2,4,8,16,32,64,128}, holding GPU capacity fixed at the base replay's
512/512/8,000,000, and evaluating the frozen 8-policy Pext portfolio.

Designed for SLURM array execution: one task = one canonical window (60
tasks total), each producing 8 lambda x 8 policy = 64 cells. This keeps
per-task wall-clock small (<1 min observed in local smoke) while avoiding
thousands of tiny SLURM tasks.

No flag here can change window selection, the load-factor grid, the policy
portfolio, or GPU capacity -- those are frozen constants in
public_replay_load_scaling_v1.py. The only knobs are engineering ones:
which window(s) to run, where to write output, and whether this is a smoke
subset.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.policy_separation import public_replay_load_scaling_v1 as prl  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "public_replay_load_scaling_v1"


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True)
        return bool(out.strip())
    except Exception:
        return True


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not_installed"


def run_window(record, *, lambdas, policies, out_dir: Path, label: str) -> dict:
    cells_path = out_dir / f"cells_window_{record['window_index']:03d}_{record['source_dataset']}.jsonl"
    if cells_path.exists():
        cells_path.unlink()
    t0 = time.time()
    n_ok = n_fail = 0
    with cells_path.open("a") as jf:
        for lam in lambdas:
            for pid in policies:
                row = prl.evaluate_cell(record, lam, pid)
                jf.write(json.dumps(row, sort_keys=True) + "\n")
                if row["status"] == "success":
                    n_ok += 1
                else:
                    n_fail += 1
    elapsed = time.time() - t0
    print(
        f"[{label}] window={record['canonical_scenario_id']} n_ok={n_ok} n_fail={n_fail} "
        f"elapsed={elapsed:.2f}s -> {cells_path}",
        flush=True,
    )
    return {
        "canonical_scenario_id": record["canonical_scenario_id"],
        "source_dataset": record["source_dataset"],
        "window_index": record["window_index"],
        "n_ok": n_ok,
        "n_fail": n_fail,
        "elapsed_s": elapsed,
        "cells_path": str(cells_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task-index", type=int, default=None,
        help="SLURM-array style index into the 60 canonical windows (0-based, sorted by "
             "canonical_scenario_id). Mutually exclusive with --all.",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Run every canonical window sequentially in this process (local/engineering use "
             "only; the SLURM array should use --task-index instead).",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="Tiny engineering-validation subset: first 2 windows x lambda in {1,16} x "
             "2 policies (full_prefill, official_vtc_joint_token_budget_remap). Writes to "
             "a smoke/ subdirectory, never touching real cells_window_*.jsonl files.",
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    if args.task_index is None and not args.all and not args.smoke:
        print("FATAL: one of --task-index, --all, --smoke is required", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = prl.get_canonical_windows()

    provenance = {
        "schema_version": "public_replay_load_scaling_v1_provenance.1.0.0",
        "git_head_sha": _git_head_sha(),
        "git_tree_dirty": _git_dirty(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": _pkg_version("numpy"),
        "pandas_version": _pkg_version("pandas"),
        "hostname": platform.node(),
        "load_factors": list(prl.LOAD_FACTORS),
        "pext_policies": list(prl.PEXT_POLICIES),
        "expected_n_cells": prl.EXPECTED_N_CELLS,
        "run_timestamp_utc_start": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if args.smoke:
        smoke_dir = out_dir / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        provenance["mode"] = "smoke"
        provenance["exact_command"] = "python3 scripts/run_public_replay_load_scaling_v1.py --smoke"
        smoke_lambdas = [1.0, 16.0]
        smoke_policies = ["full_prefill", "official_vtc_joint_token_budget_remap"]
        results = []
        for record in windows[:2]:
            results.append(
                run_window(
                    record, lambdas=smoke_lambdas, policies=smoke_policies,
                    out_dir=smoke_dir, label="SMOKE",
                )
            )
        provenance["run_timestamp_utc_end"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (smoke_dir / "smoke_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        (smoke_dir / "smoke_summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        n_fail_total = sum(r["n_fail"] for r in results)
        print(f"SMOKE done: n_fail_total={n_fail_total}", flush=True)
        return 0 if n_fail_total == 0 else 1

    if args.all:
        provenance["mode"] = "all"
        provenance["exact_command"] = "python3 scripts/run_public_replay_load_scaling_v1.py --all"
        results = []
        for record in windows:
            results.append(
                run_window(
                    record, lambdas=prl.LOAD_FACTORS, policies=prl.PEXT_POLICIES,
                    out_dir=out_dir, label="ALL",
                )
            )
        provenance["run_timestamp_utc_end"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (out_dir / "provenance_all.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        n_fail_total = sum(r["n_fail"] for r in results)
        print(f"ALL done: n_fail_total={n_fail_total}", flush=True)
        return 0 if n_fail_total == 0 else 1

    # --task-index path (SLURM array)
    idx = args.task_index
    if idx < 0 or idx >= len(windows):
        print(f"FATAL: task-index {idx} out of range [0, {len(windows)})", file=sys.stderr)
        return 2
    record = windows[idx]
    provenance["mode"] = "task_index"
    provenance["task_index"] = idx
    provenance["exact_command"] = f"python3 scripts/run_public_replay_load_scaling_v1.py --task-index {idx}"
    result = run_window(
        record, lambdas=prl.LOAD_FACTORS, policies=prl.PEXT_POLICIES,
        out_dir=out_dir, label=f"TASK{idx}",
    )
    provenance["run_timestamp_utc_end"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    provenance["result"] = result
    prov_path = out_dir / f"provenance_task_{idx:03d}.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"provenance written to {prov_path}", flush=True)
    return 0 if result["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
