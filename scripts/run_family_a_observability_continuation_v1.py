#!/usr/bin/env python3
"""Family-A Observability / Continuation-Dependence Diagnostic v1 -- TRAIN/VAL,
FAMILY-A-ONLY long-running execution.

Executes the diagnostic frozen by
`docs/design/FAMILY_A_OBSERVABILITY_CONTINUATION_DIAGNOSTIC_V1.md` over the
64 Family-A (`FAMILY_A_FAIRNESS_STARVATION_V2`) TRAIN/VAL scenarios only.
NEVER reads a TEST-split scenario or telemetry row, NEVER touches Family B/C,
NEVER reads or launches the Family-B held-out replication, and NEVER
modifies the completed decision-criticality or public-trace-replay canonical
result artifacts.

DIAGNOSTIC / METHODOLOGY ONLY. Computes no new project-level scientific
verdict beyond the interpretation category named in the design doc SS_J.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from llmserveopt.analysis import family_a_observability_continuation_v1 as fac
from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm

DESIGN_DOC = REPO_ROOT / "docs/design/FAMILY_A_OBSERVABILITY_CONTINUATION_DIAGNOSTIC_V1.md"
OUTPUT_DIR = REPO_ROOT / "experiments/family_a_observability_continuation_v1"
LOG_DIR = REPO_ROOT / "logs"

EXPECTED_FAMILY_A_TOTAL = 64


def _git_head_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True)
    return bool(out.strip())


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not_installed"


def main() -> int:
    t_start = time.time()
    start_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=== Family-A Observability / Continuation-Dependence Diagnostic v1 (TRAIN/VAL, FAMILY-A-ONLY) ===", flush=True)
    print(f"start_utc={start_utc}", flush=True)
    print(f"git_head_sha={_git_head_sha()}", flush=True)
    print(f"git_tree_dirty={_git_dirty()}", flush=True)
    print(f"python={sys.executable}", flush=True)

    report: dict = {
        "schema_version": fac.SCHEMA_VERSION,
        "run_timestamp_utc_start": start_utc,
    }
    report["provenance"] = {
        "git_head_sha": _git_head_sha(),
        "git_tree_dirty": _git_dirty(),
        "design_doc_sha256": _sha256_of_file(DESIGN_DOC),
        "mf_psd_scenarios_sha256": _sha256_of_file(dcm.MF_PSD_SCENARIOS_CSV),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": _pkg_version("numpy"),
        "pandas_version": _pkg_version("pandas"),
        "sklearn_version": _pkg_version("sklearn"),
        "family_a_diagnostic_max_extra_steps": fac.FAMILY_A_DIAGNOSTIC_MAX_EXTRA_STEPS,
        "full_trajectory_branches_per_scenario": fac.FULL_TRAJECTORY_BRANCHES_PER_SCENARIO,
        "history_window": fac.HISTORY_WINDOW,
        "estf_id": fac.ESTF_ID,
        "wfs_id": fac.WFS_ID,
        "exact_command": "python3 scripts/run_family_a_observability_continuation_v1.py",
    }

    fac.assert_no_replication_module_imported()

    print("Loading Family-A TRAIN/VAL scenario table...", flush=True)
    table = fac.load_family_a_trainval_scenario_table()
    print(f"Family-A TRAIN/VAL scenario count: {len(table)}", flush=True)
    if len(table) != EXPECTED_FAMILY_A_TOTAL:
        print(f"FATAL: expected {EXPECTED_FAMILY_A_TOTAL} Family-A TRAIN/VAL scenarios, got {len(table)}", flush=True)
        return 1
    assert not (table["split"] == "test").any(), "internal error: TEST row present"
    report["family_a_scenario_total"] = int(len(table))
    report["family_a_split_counts"] = {k: int(v) for k, v in table["split"].value_counts().to_dict().items()}

    print("Fitting frozen Stage-1/Stage-2 models (TRAIN only, exact frozen recipe)...", flush=True)
    stage1, stage2_selectors = fac.fit_frozen_models()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenario_summaries = []
    all_events_rows = []
    failures = []

    n = len(table)
    for i, (_, row) in enumerate(table.iterrows()):
        sid = row["canonical_scenario_id"]
        t0 = time.time()
        try:
            res = fac.run_family_a_row_diagnostic(row, stage1=stage1, stage2_selectors=stage2_selectors)
        except Exception as exc:  # noqa: BLE001 -- long unattended run; record and continue
            elapsed = time.time() - t0
            print(f"[{i + 1}/{n}] FAILED {sid} after {elapsed:.1f}s: {exc!r}", flush=True)
            failures.append({"canonical_scenario_id": sid, "error": repr(exc)})
            continue
        elapsed = time.time() - t0

        scenario_summaries.append({
            "canonical_scenario_id": sid,
            "split": row["split"],
            "n_steps": res.n_steps,
            "n_family_a_active_steps": res.n_family_a_active_steps,
            "n_events": len(res.events),
            "elapsed_s": elapsed,
        })
        for ev in res.events:
            all_events_rows.append(ev.to_row())
        print(
            f"[{i + 1}/{n}] {sid} ({row['split']}): n_steps={res.n_steps} "
            f"a_active_steps={res.n_family_a_active_steps} events={len(res.events)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    report["failures"] = failures
    report["n_scenarios_succeeded"] = len(scenario_summaries)
    report["n_scenarios_failed"] = len(failures)
    report["n_events_total"] = len(all_events_rows)

    summary_df = pd.DataFrame(scenario_summaries)
    summary_df.to_csv(OUTPUT_DIR / "family_a_scenario_summaries.csv", index=False)

    events_df = pd.DataFrame(all_events_rows)
    if len(events_df):
        events_df.to_csv(OUTPUT_DIR / "family_a_observability_continuation_events.csv", index=False)

        report["n_events_materially_nonzero_delta_same"] = int((events_df["delta_same"] != 0).sum())
        report["n_events_sign_same_eq_native"] = int(events_df["sign_same_eq_native"].sum())
        report["mean_delta_same"] = float(events_df["delta_same"].mean())
        report["mean_delta_native"] = float(events_df["delta_native"].mean())
        report["mean_continuation_dependence"] = float(events_df["continuation_dependence"].mean())

    n_scenarios_with_events = int((summary_df["n_events"] > 0).sum()) if len(summary_df) else 0

    duplicate_join_keys = (
        int(events_df.duplicated(subset=["canonical_scenario_id", "step"]).sum())
        if len(events_df) else 0
    )

    # 2026-08-20 repair (see
    # docs/current/family_a_observability_continuation_v1_repair_audit_20260820.md):
    # the prior integrity gate only checked for crashes/scenario-count/
    # duplicate-keys and could report `ok=true` even when zero disagreement
    # events were ever captured (exactly what happened in the invalid
    # pre-repair run, preserved at
    # experiments/family_a_observability_continuation_v1_invalid_pre_snapshot_fix_20260820/).
    # A scientifically usable run must find at least one event, in at least
    # one scenario. This is NOT a claim that the repaired instrumentation
    # must reproduce the parent decision-criticality study's exact event
    # count or definition (the two diagnostics sample disagreements
    # differently -- see design doc SS_G and the repair audit's parent-study
    # cross-check) -- only that literally zero is never a valid outcome
    # given this diagnostic re-derives disagreement over the identical,
    # already-verified-nonzero-disagreement Family-A population.
    zero_events_detected = len(all_events_rows) == 0
    zero_scenarios_with_events_detected = n_scenarios_with_events == 0

    integrity = {
        "n_scenarios_expected": EXPECTED_FAMILY_A_TOTAL,
        "n_scenarios_observed": len(scenario_summaries),
        "n_scenarios_failed": len(failures),
        "n_events_total": len(all_events_rows),
        "n_events_expected_max": EXPECTED_FAMILY_A_TOTAL * fac.FULL_TRAJECTORY_BRANCHES_PER_SCENARIO,
        "n_scenarios_with_events": n_scenarios_with_events,
        "duplicate_join_keys": duplicate_join_keys,
        "zero_events_detected": zero_events_detected,
        "zero_scenarios_with_events_detected": zero_scenarios_with_events_detected,
        "ok": (
            len(failures) == 0
            and len(scenario_summaries) == EXPECTED_FAMILY_A_TOTAL
            and duplicate_join_keys == 0
            and not zero_events_detected
            and not zero_scenarios_with_events_detected
        ),
    }
    with open(OUTPUT_DIR / "family_a_observability_continuation_integrity_report.json", "w") as f:
        json.dump(integrity, f, indent=2, sort_keys=True)
        f.write("\n")

    end_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report["run_timestamp_utc_end"] = end_utc
    report["wall_clock_seconds"] = time.time() - t_start

    out_path = OUTPUT_DIR / "family_a_observability_continuation_v1_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    print(f"end_utc={end_utc}", flush=True)
    print(f"wall_clock_seconds={report['wall_clock_seconds']:.1f}", flush=True)
    print(f"n_scenarios_with_events={n_scenarios_with_events}", flush=True)
    print(f"integrity_ok={integrity['ok']}", flush=True)
    print(f"Results written to {out_path}", flush=True)
    print("=== DONE ===", flush=True)
    if failures:
        return 2
    if not integrity["ok"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
