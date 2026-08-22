#!/usr/bin/env python3
"""Public Trace Replay Scenarios v1 -- Layer 3 (multi-policy outcomes) and
Layer 4 (step-level trajectories) canonical runner.

Executes the frozen preregistration in
`docs/design/PUBLIC_TRACE_REPLAY_SCENARIOS_V1.md` over every canonical Layer-2
scenario produced by
`llmserveopt.policy_separation.public_trace_replay_v1.build_all_scenarios()`
(120 scenario records; 480 total scenario-policy cells: 60 faithful-view
scenarios x 2 policies + 60 augmented-view scenarios x 6 policies).

This script deliberately exposes NO flags that could silently change the
frozen scientific design (window size/selection, GPU config, SLO/priority/
class/prediction-noise annotation rules, evidence-class definitions, or
policy applicability) -- those are fixed constants in
public_trace_replay_v1.py, not runtime options. The only knobs here are
engineering ones: where to write output, whether to run the full corpus or
a tiny smoke subset, and whether to resume an interrupted run.

Checkpointing: every completed cell is appended as one JSON line to
`layer3_checkpoint.jsonl` (fsync'd immediately -- an interrupted process
loses at most its one in-flight cell, never a previously-completed one).
Re-running requires --resume once a checkpoint already has content, to
guard against an accidental duplicate invocation; --resume skips only
well-formed `status: success` cells and recomputes everything else
(including previously-failed cells, in case the failure was transient).
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.policy_separation import public_trace_replay_v1 as ptr  # noqa: E402

DESIGN_DOC = REPO_ROOT / "docs/design/PUBLIC_TRACE_REPLAY_SCENARIOS_V1.md"
BUILDER_MODULE = REPO_ROOT / "src/llmserveopt/policy_separation/public_trace_replay_v1.py"
LAYER2_MANIFEST = REPO_ROOT / "experiments/public_trace_replay_v1/layer2_scenario_manifest.json"
LAYER1_MANIFEST = REPO_ROOT / "data/public_trace_corpus_v1/manifest.json"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/public_trace_replay_v1"

EXPECTED_N_SCENARIOS = 120
EXPECTED_N_FAITHFUL_SCENARIOS = 60
EXPECTED_N_AUGMENTED_SCENARIOS = 60
EXPECTED_N_CELLS = 480
EXPECTED_N_FAITHFUL_CELLS = 120
EXPECTED_N_AUGMENTED_CELLS = 360


def _git_head_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True)
    return bool(out.strip())


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not_installed"


def build_provenance(command: str) -> dict:
    return {
        "schema_version": "public_trace_replay_v1_layer3_provenance.1.0.0",
        "git_head_sha": _git_head_sha(),
        "git_tree_dirty": _git_dirty(),
        "design_doc_sha256": ptr.sha256_of_file(DESIGN_DOC),
        "builder_module_sha256": ptr.sha256_of_file(BUILDER_MODULE),
        "layer2_manifest_sha256": ptr.sha256_of_file(LAYER2_MANIFEST) if LAYER2_MANIFEST.exists() else None,
        "layer1_manifest_sha256": ptr.sha256_of_file(LAYER1_MANIFEST),
        "runner_sha256": ptr.sha256_of_file(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": _pkg_version("numpy"),
        "pandas_version": _pkg_version("pandas"),
        "exact_command": command,
        "window_size": ptr.WINDOW_SIZE,
        "windows_per_source": ptr.WINDOWS_PER_SOURCE,
        "seed": ptr.SEED,
        "prediction_noise_sigma": ptr.PREDICTION_NOISE_SIGMA,
        "slack_multiplier": ptr.SLACK_MULTIPLIER,
    }


def run_cells(records, *, out_dir: Path, resume: bool, checkpoint_name: str, label: str) -> int:
    """Shared engine for both --smoke and the full run. Returns process
    exit code (0 success, 1 integrity failure, 2 unexpected internal
    condition)."""
    checkpoint_path = out_dir / checkpoint_name
    traj_dir = out_dir / "trajectories"

    expected_keys = ptr.expected_cell_keys(records)
    n_faithful_scenarios = sum(1 for r in records if r["scenario_evidence_class"] == ptr.FAITHFUL)
    n_augmented_scenarios = sum(1 for r in records if r["scenario_evidence_class"] == ptr.AUGMENTED)
    n_faithful_cells = sum(
        len(r["applicable_policies"]) for r in records if r["scenario_evidence_class"] == ptr.FAITHFUL
    )
    n_augmented_cells = sum(
        len(r["applicable_policies"]) for r in records if r["scenario_evidence_class"] == ptr.AUGMENTED
    )
    print(f"=== {label}: {len(records)} scenarios ({n_faithful_scenarios} faithful, "
          f"{n_augmented_scenarios} augmented), {len(expected_keys)} expected cells "
          f"({n_faithful_cells} faithful, {n_augmented_cells} augmented) ===", flush=True)

    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0 and not resume:
        print(f"FATAL: {checkpoint_path} already has content. Pass --resume to continue "
              f"an interrupted run, or move/delete it first if you intend a fresh run.",
              file=sys.stderr, flush=True)
        return 2

    corrupt_lines = ptr.scan_checkpoint_corruption(checkpoint_path)
    if corrupt_lines:
        print(f"WARNING: {len(corrupt_lines)} corrupt/torn line(s) in existing checkpoint "
              f"at {corrupt_lines} -- these will be ignored and their cells recomputed.",
              flush=True)

    completed = ptr.load_checkpoint(checkpoint_path) if resume else {}
    n_skipped = 0
    n_run = 0
    t_start = time.time()

    for i, r in enumerate(records):
        for policy_id in r["applicable_policies"]:
            key = ptr.canonical_cell_key(r["canonical_scenario_id"], policy_id)
            existing = completed.get(key)
            if existing is not None and ptr.is_valid_success_row(existing):
                n_skipped += 1
                continue

            t0 = time.time()
            row, traj_rows = ptr.evaluate_scenario_policy(
                r["scenario"], policy_id, capture_trajectory=True,
                canonical_scenario_id=r["canonical_scenario_id"],
                source_dataset=r["source_dataset"],
                scenario_evidence_class=r["scenario_evidence_class"],
            )
            elapsed = time.time() - t0
            ptr.append_checkpoint_row(checkpoint_path, row)
            if row["status"] == "success":
                ptr.write_trajectory_parquet(traj_dir, r["canonical_scenario_id"], policy_id, traj_rows)
            n_run += 1
            print(f"[{n_run + n_skipped}/{len(expected_keys)}] {key} status={row['status']} "
                  f"anwg={row.get('primary_utility_anwg')} n_traj_rows={len(traj_rows)} "
                  f"elapsed={elapsed:.2f}s", flush=True)

    print(f"=== {label} done: {n_run} cells run, {n_skipped} skipped (already valid), "
          f"wall_clock={time.time() - t_start:.1f}s ===", flush=True)

    final_rows = ptr.load_checkpoint(checkpoint_path)
    report = ptr.validate_full_result_set(expected_keys, final_rows, checkpoint_path)
    report_path = out_dir / f"{Path(checkpoint_name).stem}_integrity_report.json"
    import json
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"integrity report: {report_path} ok={report['ok']} "
          f"n_success={report['n_success']} n_failed={report['n_failed']} "
          f"n_missing={report['n_missing']}", flush=True)

    return 0 if report["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run only a tiny deterministic engineering-validation subset (window 0 of each "
             "source, both evidence classes, all applicable policies -- 24 cells, the same "
             "subset already validated ad hoc in a prior task) instead of the full 480-cell "
             "corpus. Writes to smoke_checkpoint.jsonl, never touching the real "
             "layer3_checkpoint.jsonl.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an interrupted run: skip cells already recorded with status=success "
             "in the checkpoint; recompute everything else (including previously-failed "
             "cells). Required if the target checkpoint file already has content.",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help="Output directory (engineering parameter only -- does not affect scientific "
             f"design). Defaults to the canonical {DEFAULT_OUT_DIR}.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = ptr.build_all_scenarios()
    if len(all_records) != EXPECTED_N_SCENARIOS:
        print(f"FATAL: expected {EXPECTED_N_SCENARIOS} Layer-2 scenario records, "
              f"got {len(all_records)}", file=sys.stderr, flush=True)
        return 2
    n_faithful = sum(1 for r in all_records if r["scenario_evidence_class"] == ptr.FAITHFUL)
    n_augmented = sum(1 for r in all_records if r["scenario_evidence_class"] == ptr.AUGMENTED)
    if n_faithful != EXPECTED_N_FAITHFUL_SCENARIOS or n_augmented != EXPECTED_N_AUGMENTED_SCENARIOS:
        print(f"FATAL: expected {EXPECTED_N_FAITHFUL_SCENARIOS}/{EXPECTED_N_AUGMENTED_SCENARIOS} "
              f"faithful/augmented scenarios, got {n_faithful}/{n_augmented}",
              file=sys.stderr, flush=True)
        return 2
    total_cells = sum(len(r["applicable_policies"]) for r in all_records)
    if total_cells != EXPECTED_N_CELLS:
        print(f"FATAL: expected {EXPECTED_N_CELLS} total cells, got {total_cells}",
              file=sys.stderr, flush=True)
        return 2

    start_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    command = "python3 scripts/run_public_trace_replay_v1.py" + (" --smoke" if args.smoke else "")
    provenance = build_provenance(command)
    provenance["run_timestamp_utc_start"] = start_utc

    if args.smoke:
        smoke_records = [r for r in all_records if r["window_index"] == 0]
        exit_code = run_cells(
            smoke_records, out_dir=out_dir, resume=args.resume,
            checkpoint_name="smoke_checkpoint.jsonl", label="SMOKE (window 0 of each source)",
        )
        provenance_path = out_dir / "smoke_provenance.json"
    else:
        exit_code = run_cells(
            all_records, out_dir=out_dir, resume=args.resume,
            checkpoint_name="layer3_checkpoint.jsonl", label="FULL LAYER-3/4 REPLAY (480 cells)",
        )
        provenance_path = out_dir / "layer3_provenance.json"

    provenance["run_timestamp_utc_end"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    import json
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"provenance written to {provenance_path}", flush=True)
    print(f"exit_code={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
