#!/usr/bin/env python3
"""Decision-Criticality & Regime-Timescale Diagnostic v1 -- TRAIN/VAL-ONLY
long-running execution.

Executes the diagnostic frozen by
`docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md` over every
TRAIN/VAL MF-PSD scenario (144 total: A=64, B=32, C=48). NEVER reads a
TEST-split scenario or telemetry row, and NEVER reads or launches the
preregistered Family-B held-out replication
(`experiments/family_b_balanced_replication_v1/`,
`scripts/run_family_b_balanced_replication_v1.py`).

DIAGNOSTIC / METHODOLOGY ONLY. Computes no new project-level scientific
verdict. Does not retrain, re-threshold, or tune anything frozen by
`hierarchical_regime_router_v1.py` / `hierarchical_stage2_selectors_v1.py`.
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

import numpy as np
import pandas as pd

from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (
    ACTIVE_REGIMES, REGIME_A, REGIME_B, REGIME_C,
)

DESIGN_DOC = REPO_ROOT / "docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md"
GATES_JSON = REPO_ROOT / "configs/hierarchical_regime_router_v1_gates.json"
OUTPUT_DIR = REPO_ROOT / "experiments/decision_criticality_timescale_trainval_v1"
LOG_DIR = REPO_ROOT / "logs"

EXPECTED_COUNTS = {
    "FAMILY_A_FAIRNESS_STARVATION_V2": 64,
    "FAMILY_B_PREFILL_DECODE_V2": 32,
    "FAMILY_C_KV_PRESSURE_V2": 48,
}
EXPECTED_TOTAL = 144


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

    print("=== Decision-Criticality & Regime-Timescale Diagnostic v1 (TRAIN/VAL-ONLY) ===", flush=True)
    print(f"start_utc={start_utc}", flush=True)
    print(f"git_head_sha={_git_head_sha()}", flush=True)
    print(f"git_tree_dirty={_git_dirty()}", flush=True)
    print(f"python={sys.executable}", flush=True)

    report: dict = {
        "schema_version": dcm.SCHEMA_VERSION,
        "run_timestamp_utc_start": start_utc,
    }
    report["provenance"] = {
        "git_head_sha": _git_head_sha(),
        "git_tree_dirty": _git_dirty(),
        "design_doc_sha256": _sha256_of_file(DESIGN_DOC),
        "gates_json_sha256": _sha256_of_file(GATES_JSON),
        "mf_psd_scenarios_sha256": _sha256_of_file(dcm.MF_PSD_SCENARIOS_CSV),
        "telemetry_csv_sha256": _sha256_of_file(dcm.TELEMETRY_CSV),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": _pkg_version("numpy"),
        "pandas_version": _pkg_version("pandas"),
        "sklearn_version": _pkg_version("sklearn"),
        "horizon_h": dcm.HORIZON_H,
        "dwell_reference": dcm.DWELL_REFERENCE,
        "full_trajectory_max_extra_steps": dcm.FULL_TRAJECTORY_MAX_EXTRA_STEPS,
        "full_trajectory_max_branches_per_scenario": dcm.FULL_TRAJECTORY_MAX_BRANCHES_PER_SCENARIO,
        "exact_command": "python3 scripts/run_decision_criticality_timescale_trainval_v1.py",
    }

    dcm.assert_no_replication_module_imported()

    print("Loading TRAIN/VAL scenario table...", flush=True)
    table = dcm.load_trainval_scenario_table()
    counts = table["mechanism_family"].value_counts().to_dict()
    print(f"TRAIN/VAL scenario counts: {counts}", flush=True)
    for fam, expected in EXPECTED_COUNTS.items():
        actual = int(counts.get(fam, 0))
        if actual != expected:
            print(f"FATAL: expected {expected} {fam} TRAIN/VAL scenarios, got {actual}", flush=True)
            return 1
    if len(table) != EXPECTED_TOTAL:
        print(f"FATAL: expected {EXPECTED_TOTAL} TRAIN/VAL scenarios total, got {len(table)}", flush=True)
        return 1
    assert not (table["split"] == "test").any(), "internal error: TEST row present"
    report["trainval_scenario_counts"] = {k: int(v) for k, v in counts.items()}
    report["trainval_scenario_total"] = int(len(table))

    print("Fitting frozen Stage-1/Stage-2 models (TRAIN only, exact frozen recipe)...", flush=True)
    stage1, stage2_selectors = dcm.fit_frozen_models()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trajectories: dict = {}
    disagreement_frames: dict = {}
    full_trajectory_by_regime: dict = {r: [] for r in ACTIVE_REGIMES}
    scenario_summaries = []
    failures = []

    n = len(table)
    for i, (_, row) in enumerate(table.iterrows()):
        sid = row["canonical_scenario_id"]
        fam = row["mechanism_family"]
        t0 = time.time()
        try:
            res = dcm.run_scenario_diagnostic(row, stage1=stage1, stage2_selectors=stage2_selectors)
        except Exception as exc:  # noqa: BLE001 -- long unattended run; record and continue
            elapsed = time.time() - t0
            print(f"[{i + 1}/{n}] FAILED {sid} ({fam}) after {elapsed:.1f}s: {exc!r}", flush=True)
            failures.append({"canonical_scenario_id": sid, "mechanism_family": fam, "error": repr(exc)})
            continue
        elapsed = time.time() - t0

        trajectories[sid] = res.trajectory
        disagreement_frames[sid] = res.disagreement_rows
        for branch in res.full_trajectory_results:
            full_trajectory_by_regime[branch["regime"]].append(branch)

        n_disagree = (
            int(res.disagreement_rows["disagree"].sum())
            if len(res.disagreement_rows) and "disagree" in res.disagreement_rows.columns
            else 0
        )
        regime_counts = (
            res.trajectory["effective_regime"].value_counts().to_dict() if len(res.trajectory) else {}
        )
        scenario_summaries.append({
            "canonical_scenario_id": sid,
            "mechanism_family": fam,
            "split": row["split"],
            "n_steps": res.n_steps,
            "n_disagreement_steps": n_disagree,
            "n_full_trajectory_branches": len(res.full_trajectory_results),
            "elapsed_s": elapsed,
            **{f"effective_regime_count__{k}": int(v) for k, v in regime_counts.items()},
        })
        print(
            f"[{i + 1}/{n}] {sid} ({fam}, {row['split']}): n_steps={res.n_steps} "
            f"disagreements={n_disagree} full_traj_branches={len(res.full_trajectory_results)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    report["failures"] = failures
    report["n_scenarios_succeeded"] = len(scenario_summaries)
    report["n_scenarios_failed"] = len(failures)

    print("Computing aggregations (episode timescales, dwell latency, disagreement rates, "
          "causal importance, minority-critical episodes, Family-B diagnostic, ceiling diagnostic)...",
          flush=True)

    report["episode_timescales"] = dcm.aggregate_episode_timescales(trajectories)
    report["dwell_latency"] = dcm.aggregate_dwell_latency(trajectories)
    disagreement_rate_summary = dcm.aggregate_disagreement_rates(disagreement_frames, trajectories)
    report["disagreement_rates"] = disagreement_rate_summary
    report["causal_importance"] = dcm.aggregate_causal_importance(disagreement_frames)
    minority_critical = dcm.find_minority_critical_episodes(trajectories, disagreement_frames)
    report["minority_critical_episodes"] = {
        regime: {
            "count": len(eps),
            "representative": eps[:10],
        }
        for regime, eps in minority_critical.items()
    }
    report["policy_library_ceiling"] = dcm.aggregate_ceiling_diagnostic(
        disagreement_rate_summary, full_trajectory_by_regime,
    )

    # -- Family-B primary diagnostic (design doc SS10 / task SS10) --------
    family_b_ids = set(table[table["mechanism_family"] == "FAMILY_B_PREFILL_DECODE_V2"]["canonical_scenario_id"])
    family_b_trajectories = {sid: traj for sid, traj in trajectories.items() if sid in family_b_ids}
    family_b_disagreements = {sid: df for sid, df in disagreement_frames.items() if sid in family_b_ids}
    b_episode_timescales = dcm.aggregate_episode_timescales(family_b_trajectories)
    b_dwell_latency = dcm.aggregate_dwell_latency(family_b_trajectories)
    b_disagreement = dcm.aggregate_disagreement_rates(family_b_disagreements, family_b_trajectories)
    b_causal = dcm.aggregate_causal_importance(family_b_disagreements)
    report["family_b_primary_diagnostic"] = {
        "n_family_b_scenarios": len(family_b_trajectories),
        "episode_duration_distribution": b_episode_timescales.get(dcm.REGIME_B_ACTIVE_LABEL),
        "dwell_latency": b_dwell_latency.get(REGIME_B),
        "disagreement_rate": b_disagreement.get(REGIME_B),
        "causal_importance": b_causal,
        "note": (
            "Computed entirely from real TRAIN/VAL Family-B scenarios "
            "(BurstGPT-backed, datasets_root-resolved). The preregistered held-out "
            "Family-B-balanced replication set and its results were never read, "
            "referenced, or used by this computation."
        ),
    }

    # -- persist compact artifacts (never the raw multi-GB trajectory dump) -
    summary_df = pd.DataFrame(scenario_summaries)
    summary_df.to_csv(OUTPUT_DIR / "scenario_summaries.csv", index=False)

    disagreement_events = []
    for sid, df in disagreement_frames.items():
        if len(df) == 0:
            continue
        sub = df.copy()
        sub["canonical_scenario_id"] = sid
        disagreement_events.append(sub)
    if disagreement_events:
        pd.concat(disagreement_events, ignore_index=True).to_csv(
            OUTPUT_DIR / "disagreement_and_divergence_events.csv", index=False,
        )

    end_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report["run_timestamp_utc_end"] = end_utc
    report["wall_clock_seconds"] = time.time() - t_start

    out_path = OUTPUT_DIR / "decision_criticality_timescale_trainval_v1_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    print(f"end_utc={end_utc}", flush=True)
    print(f"wall_clock_seconds={report['wall_clock_seconds']:.1f}", flush=True)
    print(f"Results written to {out_path}", flush=True)
    print("=== DONE ===", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
