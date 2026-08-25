#!/usr/bin/env python3
"""Family-B-Balanced Replication v1 -- launch-ready live closed-loop
evaluation runner.

Preregistered in docs/design/FAMILY_B_BALANCED_REPLICATION_V1.md
(completing SS 10 of docs/design/HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1.md).
Frozen scenario set:
experiments/family_b_balanced_replication_v1/frozen_scenario_selection_v1.json
(12 Family-A + 12 Family-B + 12 Family-C, all verified never used in
TRAIN fitting).

THIS SCRIPT MUST NOT BE RUN AGAINST THE FROZEN REPLICATION SET WITHOUT
SEPARATE, EXPLICIT SCIENTIFIC AUTHORIZATION. `--source replication` is the
scientific-evaluation path and is refused unless `--i-am-authorized` is
also passed (a deliberate two-flag guard, not a default). `--source
smoke_train` (real Family-B scenarios already used in Stage-1/Stage-2
TRAIN fitting -- safe to inspect, not held-out) and `--source
smoke_synthetic` (freshly-built microcases with a seed outside the frozen
176-scenario dataset -- never used anywhere) are always permitted and are
the only sources exercised by this task's pre-launch smoke checks.

Unlike scripts/run_hierarchical_regime_router_live_reeval_v1.py (the
primary re-evaluation script, whose persisted result carries only
TEST-aggregate scalars -- a gap documented in
docs/audits/hierarchical_regime_router_live_reeval_v1_20260818.md SS G),
this script persists PER-SCENARIO results and full per-step trajectory
logs, so that a future formal G4/G7/G9(a) rescoring does not hit the same
missing-per-regime-breakdown gap.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (
    ACTIVE_REGIMES,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    STAGE2_CANDIDATES,
    Stage1Router,
    add_regime_labels,
    build_splits,
)
from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import run_live_scenario
from llmserveopt.policy_separation.hierarchical_router_evaluation_v1 import (
    catastrophic_misroute_rate,
    group_resampled_bootstrap_ci,
    load_scenario_level_dataset,
    regime_fixed_best_from_train,
)
from llmserveopt.policy_separation.hierarchical_router_gates_v1 import (
    compute_verdict,
    evaluate_all_gates,
    load_gates_config,
)
from llmserveopt.policy_separation.family_b_balanced_replication_v1 import (
    FAMILY_A,
    FAMILY_B,
    FAMILY_C,
    select_balanced_replication_set,
    verify_no_train_leakage,
)
from llmserveopt.selector.hierarchical_stage2_selectors_v1 import fit_all_stage2_selectors
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import case_fairness_vs_size_v2
from llmserveopt.policy_separation.templates_prefill_decode_v2 import case_prefill_decode_ttft_contention
from llmserveopt.policy_separation.templates_kv_pressure_v2 import case_kv_pressure_reserve_contention_v2

TELEMETRY_PATH = REPO_ROOT / "experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv"
MF_PSD_SCENARIOS = REPO_ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"
GATES_JSON = REPO_ROOT / "configs/hierarchical_regime_router_v1_gates.json"
DESIGN_DOC = REPO_ROOT / "docs/design/FAMILY_B_BALANCED_REPLICATION_V1.md"
FROZEN_SELECTION = REPO_ROOT / "experiments/family_b_balanced_replication_v1/frozen_scenario_selection_v1.json"
OUTPUT_DIR = REPO_ROOT / "experiments/family_b_balanced_replication_v1"
DATASETS_ROOT = REPO_ROOT / ".local_data"

REGIME_TO_FAMILY = {
    REGIME_A: FAMILY_A,
    REGIME_B: FAMILY_B,
    REGIME_C: FAMILY_C,
}


def _git_head_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True)
    return bool(out.strip())


def _sha256_of_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_fitted_model(model) -> str:
    """Deterministic-within-this-environment fingerprint of a fitted
    sklearn estimator, for reused-model provenance (not a portable model
    format -- purely a consistency/identity check)."""
    import hashlib
    import pickle
    return hashlib.sha256(pickle.dumps(model)).hexdigest()


def rebuild_scenario(row: pd.Series):
    family = row["mechanism_family"]
    if family == FAMILY_A:
        return case_fairness_vs_size_v2(
            target_utilization=row["feat_A__target_utilization"],
            tenant_weight_skew=row["feat_A__tenant_weight_skew"],
            favored_tenant_size=row["feat_A__favored_tenant_size"],
            prediction_noise_sigma=row["feat_A__prediction_noise_sigma"],
            seed=int(row["seed"]),
            datasets_root=DATASETS_ROOT,
        )
    elif family == FAMILY_B:
        return case_prefill_decode_ttft_contention(
            hog_count=row["feat_B__hog_count"],
            late_pressure=row["feat_B__late_pressure"],
            slo_emphasis=row["feat_B__slo_emphasis"],
            seed=int(row["seed"]),
            datasets_root=DATASETS_ROOT,
        )
    elif family == FAMILY_C:
        return case_kv_pressure_reserve_contention_v2(
            bulk_pressure=row["feat_C__bulk_pressure"],
            urgent_arrival_phase=row["feat_C__urgent_arrival_phase"],
            urgent_tightness=row["feat_C__urgent_tightness"],
            seed=int(row["seed"]),
            datasets_root=DATASETS_ROOT,
        )
    raise ValueError(f"Unknown family {family}")


def load_frozen_replication_frame(scen: pd.DataFrame) -> pd.DataFrame:
    """--source replication: the frozen, preregistered 36-scenario set.
    Verifies the materialized selection still matches the frozen JSON
    exactly (byte-for-byte scenario-id sets) before returning it."""
    frozen = json.loads(FROZEN_SELECTION.read_text())
    replication_set = select_balanced_replication_set(scen)
    for fam, ids in frozen["scenario_ids_by_family"].items():
        actual = sorted(replication_set[replication_set["mechanism_family"] == fam]["canonical_scenario_id"])
        assert actual == ids, f"replication selection drifted from frozen JSON for {fam}"
    verify_no_train_leakage(scen, replication_set)
    return replication_set


def load_smoke_train_frame(scen: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """--source smoke_train: real Family-B scenarios already used in
    Stage-1/Stage-2 TRAIN fitting. Safe to inspect (no held-out data is
    touched); results are scientifically meaningless (train-contaminated),
    used only to exercise the live-B routing path end-to-end."""
    pool = scen[(scen["mechanism_family"] == FAMILY_B) & (scen["split"] == "train")]
    return pool.sort_values("canonical_scenario_id").head(n)


def build_smoke_synthetic_frame() -> pd.DataFrame:
    """--source smoke_synthetic: freshly-built Family-B microcases with a
    seed far outside the frozen 176-scenario dataset (seed 90000001+),
    guaranteed disjoint from TRAIN, VAL, TEST, and the frozen replication
    set. Zero risk of ever touching held-out data."""
    rows = []
    for i, seed in enumerate((90000001, 90000002)):
        rows.append({
            "canonical_scenario_id": f"SMOKE_SYNTHETIC_FAMILY_B::{seed}",
            "mechanism_family": FAMILY_B,
            "group_key": f"SMOKE_SYNTHETIC_FAMILY_B::group{i}",
            "seed": seed,
            "split": "smoke_synthetic",
            "feat_B__hog_count": "high",
            "feat_B__late_pressure": "high",
            "feat_B__slo_emphasis": "hog_ttft",
        })
    return pd.DataFrame(rows)


def evaluate_scenarios(
    frame: pd.DataFrame,
    stage1: Stage1Router,
    stage2_selectors: dict,
) -> tuple[list[dict], list[pd.DataFrame]]:
    per_scenario = []
    trajectories = []
    for _, row in frame.iterrows():
        sid = row["canonical_scenario_id"]
        print(f"[family_b_balanced_replication_v1] running {sid} ...", flush=True)
        scenario = rebuild_scenario(row)
        res = run_live_scenario(
            scenario,
            canonical_scenario_id=sid,
            stage1=stage1,
            stage2_selectors=stage2_selectors,
            record_trajectory=True,
        )
        anwg = res.metrics.arrival_normalized_weighted_goodput
        traj = res.trajectory.copy()
        traj["canonical_scenario_id"] = sid
        trajectories.append(traj)
        counts = traj["effective_regime"].value_counts().to_dict()
        per_scenario.append({
            "canonical_scenario_id": sid,
            "mechanism_family": row["mechanism_family"],
            "group_key": row["group_key"],
            "seed": int(row["seed"]),
            "anwg_live": float(anwg),
            "n_steps": len(traj),
            "dwell_violations": (res.dwell_diagnostics or {}).get("dwell_violation_count", 0),
            "fallback_rate": (res.dwell_diagnostics or {}).get("fallback_rate", 0),
            "total_transitions": (res.dwell_diagnostics or {}).get("total_transitions", 0),
            "A_count": counts.get(REGIME_A, 0),
            "B_count": counts.get(REGIME_B, 0),
            "C_count": counts.get(REGIME_C, 0),
        })
    return per_scenario, trajectories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["replication", "smoke_train", "smoke_synthetic"], required=True,
        help="replication = the frozen scientific 36-scenario set (requires --i-am-authorized); "
             "smoke_train/smoke_synthetic = safe pre-launch smoke checks only.",
    )
    parser.add_argument(
        "--i-am-authorized", action="store_true",
        help="Required in addition to --source replication. Confirms explicit, separate scientific "
             "authorization to launch the held-out Family-B-Balanced Replication evaluation.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write run_<source>_results.json/_trajectories.csv into. Defaults to "
             "the canonical experiments/family_b_balanced_replication_v1/ directory. Tests should "
             "pass a tmp_path here instead of mutating the tracked canonical outputs.",
    )
    args = parser.parse_args()

    if args.source == "replication" and not args.i_am_authorized:
        print(
            "REFUSED: --source replication requires --i-am-authorized. "
            "This is a deliberate guard -- launching the scientific evaluation of the frozen "
            "held-out replication set requires separate, explicit authorization, not a default run.",
            file=sys.stderr,
        )
        return 3

    scen = pd.read_csv(MF_PSD_SCENARIOS)
    split_map = build_splits(scen)
    scen["split"] = scen["canonical_scenario_id"].map(split_map)

    telemetry = add_regime_labels(pd.read_csv(TELEMETRY_PATH))
    telemetry["split"] = telemetry["canonical_scenario_id"].map(split_map)
    train_tel = telemetry[telemetry["split"] == "train"]

    stage1 = Stage1Router().fit(train_tel)

    scenario_df = load_scenario_level_dataset()
    train_df = scenario_df[scenario_df["split"] == "train"]
    train_by_regime = {r: train_df[train_df["regime_ground_truth"] == r] for r in ACTIVE_REGIMES}
    stage2_selectors = fit_all_stage2_selectors(train_by_regime)
    regime_fixed_best = regime_fixed_best_from_train(train_df)

    if args.source == "replication":
        frame = load_frozen_replication_frame(scen)
        run_tag = "replication"
    elif args.source == "smoke_train":
        frame = load_smoke_train_frame(scen)
        run_tag = "smoke_train"
    else:
        frame = build_smoke_synthetic_frame()
        run_tag = "smoke_synthetic"

    per_scenario, trajectories = evaluate_scenarios(frame, stage1, stage2_selectors)
    per_scenario_df = pd.DataFrame(per_scenario)

    for col in ("anwg_live",):
        assert per_scenario_df[col].notna().all(), f"{col} has NaN"
        assert np.isfinite(per_scenario_df[col].to_numpy()).all(), f"{col} has non-finite values"

    report = {
        "schema_version": "family_b_balanced_replication_v1.1.0.0",
        "run_tag": run_tag,
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "git_head_sha": _git_head_sha(),
            "git_tree_dirty": _git_dirty(),
            "design_doc_sha256": _sha256_of_file(DESIGN_DOC),
            "gates_json_sha256": _sha256_of_file(GATES_JSON),
            "mf_psd_scenarios_sha256": _sha256_of_file(MF_PSD_SCENARIOS),
            "telemetry_csv_sha256": _sha256_of_file(TELEMETRY_PATH),
            "frozen_selection_sha256": _sha256_of_file(FROZEN_SELECTION) if FROZEN_SELECTION.exists() else None,
            "reused_model_hashes": {
                "stage1_model_hash": _sha256_of_fitted_model(stage1.model),
                "stage2_model_hashes": {
                    regime: _sha256_of_fitted_model(sel.pipe) for regime, sel in stage2_selectors.items()
                },
            },
        },
        "n_scenarios": len(per_scenario_df),
        "family_counts": {k: int(v) for k, v in per_scenario_df["mechanism_family"].value_counts().items()},
        "per_scenario_results": per_scenario,
    }

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{run_tag}_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    traj_path = out_dir / f"run_{run_tag}_trajectories.csv"
    pd.concat(trajectories, ignore_index=True).to_csv(traj_path, index=False)

    print(json.dumps({k: v for k, v in report.items() if k != "per_scenario_results"}, indent=2, sort_keys=True, default=str))
    print(f"\nWrote {out_path}")
    print(f"Wrote {traj_path}")

    if run_tag != "replication":
        print(
            "\nNOTE: this was a non-scientific smoke/plumbing run "
            f"(--source {run_tag}). No held-out Family-B-Balanced Replication "
            "scientific result was produced or inspected."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
