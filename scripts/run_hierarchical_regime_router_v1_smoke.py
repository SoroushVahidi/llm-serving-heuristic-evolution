#!/usr/bin/env python3
"""Hierarchical Regime Router v1 -- TRAIN/VAL-ONLY implementation smoke.

IMPLEMENTATION + VALIDATION ONLY. Fits Stage-1 on TRAIN, evaluates it on
VAL (never TEST). Fits Stage-2 selectors per regime on TRAIN, evaluates on
VAL. Exercises the dwell/fallback FSM, all 9 metric formulas, and the gate
evaluator/verdict logic against VAL numbers -- this is NOT a scientific
verdict (VAL is not the pre-registered TEST split; no TEST row is ever
read by this script). Writes a JSON summary to
experiments/hierarchical_regime_router_v1_smoke/ for inspection.

Never trains on VAL, never reads `split == "test"` rows, never tunes
anything based on what it observes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (  # noqa: E402
    ACTIVE_REGIMES,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    Stage1Router,
    add_regime_labels,
    apply_dwell_and_fallback,
    assert_group_disjoint,
    build_splits,
    count_dwell_violations,
)
from llmserveopt.policy_separation.hierarchical_router_evaluation_v1 import (  # noqa: E402
    baseline_a_anwg,
    baseline_c_anwg,
    baseline_d_anwg,
    baseline_e_anwg,
    baseline_g_anwg,
    catastrophic_misroute_rate,
    delta_anwg,
    load_scenario_level_dataset,
    multi_regime_benefit_count,
    regime_fixed_best_from_train,
)
from llmserveopt.policy_separation.hierarchical_router_gates_v1 import (  # noqa: E402
    evaluate_all_gates,
    compute_verdict,
)
from llmserveopt.selector.hierarchical_stage2_selectors_v1 import fit_all_stage2_selectors  # noqa: E402

TELEMETRY_PATH = REPO_ROOT / "experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv"
MF_PSD_SCENARIOS = REPO_ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"
OUTPUT_DIR = REPO_ROOT / "experiments/hierarchical_regime_router_v1_smoke"


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256_of_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    summary: dict = {
        "schema_version": "hierarchical_regime_router_v1_smoke.1.0.0",
        "mode": "TRAIN_VAL_SMOKE_ONLY",
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head_sha": _git_head_sha(),
        "source_checksums": {
            "online_regime_telemetry_v1.csv": _sha256_of_file(TELEMETRY_PATH),
            "mf_psd_scenarios_v1.csv": _sha256_of_file(MF_PSD_SCENARIOS),
        },
    }

    scen = pd.read_csv(MF_PSD_SCENARIOS)
    split_map = build_splits(scen)
    assert_group_disjoint(scen, split_map)
    scen["split"] = scen["canonical_scenario_id"].map(split_map)
    summary["split_counts"] = scen["split"].value_counts().to_dict()

    # -----------------------------------------------------------------
    # Stage-1: fit on TRAIN telemetry rows, evaluate on VAL telemetry rows
    # -----------------------------------------------------------------
    telemetry = pd.read_csv(TELEMETRY_PATH)
    telemetry = add_regime_labels(telemetry)
    telemetry["split"] = telemetry["canonical_scenario_id"].map(split_map)
    train_tel = telemetry[telemetry["split"] == "train"]
    val_tel = telemetry[telemetry["split"] == "val"]
    summary["stage1_train_rows"] = int(len(train_tel))
    summary["stage1_val_rows"] = int(len(val_tel))

    stage1 = Stage1Router().fit(train_tel)
    val_pred = stage1.predict(val_tel)
    val_true = val_tel["regime_label"].to_numpy()

    assert not pd.isna(val_pred).any(), "Stage-1 predictions contain NaN"
    macro_f1_val = float(f1_score(val_true, val_pred, average="macro", labels=sorted(set(val_true) | set(val_pred))))
    summary["stage1_val_macro_f1"] = macro_f1_val
    summary["stage1_val_catastrophic_misroute_rate"] = catastrophic_misroute_rate(
        pd.Series(val_pred), pd.Series(val_true)
    )
    summary["stage1_input_validity_fraction"] = 1.0  # structural: extract_inputs() enforces the exact 4-column allowlist

    # -----------------------------------------------------------------
    # Dwell/fallback FSM smoke: run per-scenario, check 0 violations, no NaN
    # -----------------------------------------------------------------
    dwell_ok = True
    total_dwell_violations = 0
    fallback_observed = False
    active_regime_observed = {REGIME_A: False, REGIME_B: False, REGIME_C: False}
    for scenario_id, group in telemetry[telemetry["split"].isin(["train", "val"])].groupby("canonical_scenario_id"):
        raw = group.sort_values("step")["regime_label"].tolist()
        effective, diag = apply_dwell_and_fallback(raw)
        total_dwell_violations += diag.dwell_violation_count
        v = count_dwell_violations(effective)
        if v != 0:
            dwell_ok = False
        if diag.fallback_rate > 0:
            fallback_observed = True
        for r in ACTIVE_REGIMES:
            if r in effective:
                active_regime_observed[r] = True
    summary["dwell_fsm_smoke"] = {
        "total_dwell_violations": total_dwell_violations,
        "independent_check_all_zero": dwell_ok,
        "fallback_observed_on_train_val_scenarios": fallback_observed,
        "active_regime_observed_on_train_val_scenarios": active_regime_observed,
    }

    # -----------------------------------------------------------------
    # Stage-2: fit per regime on TRAIN scenario rows, exercise on VAL
    # -----------------------------------------------------------------
    scenario_df = load_scenario_level_dataset()
    train_df = scenario_df[scenario_df["split"] == "train"]
    val_df = scenario_df[scenario_df["split"] == "val"]
    summary["scenario_level_train_rows"] = int(len(train_df))
    summary["scenario_level_val_rows"] = int(len(val_df))

    train_by_regime = {r: train_df[train_df["regime_ground_truth"] == r] for r in ACTIVE_REGIMES}
    stage2_selectors = fit_all_stage2_selectors(train_by_regime)
    summary["stage2_regimes_fit"] = sorted(stage2_selectors.keys())

    stage2_reachability = {}
    for regime, sel in stage2_selectors.items():
        sub = val_df[val_df["regime_ground_truth"] == regime]
        if len(sub) == 0:
            stage2_reachability[regime] = "no_val_rows"
            continue
        preds = sel.predict(sub)
        stage2_reachability[regime] = {
            "n_val_rows": int(len(sub)),
            "predicted_policies_observed": sorted(set(preds.tolist())),
            "both_candidates_reachable": len(set(preds.tolist())) == 2,
        }
    summary["stage2_reachability_on_val"] = stage2_reachability

    # -----------------------------------------------------------------
    # Baselines A/C/D/E/G + metric formulas, evaluated on VAL only
    # -----------------------------------------------------------------
    # Scenario-level "predicted regime" for VAL: majority effective regime
    # from that scenario's own TRAIN-fit Stage-1 router applied to its
    # per-step telemetry (offline approximation -- see evaluation module
    # docstring).
    val_scenario_ids = set(val_df["canonical_scenario_id"])
    val_tel_by_scenario = {
        sid: g.sort_values("step")["regime_label"].tolist()
        for sid, g in telemetry[telemetry["canonical_scenario_id"].isin(val_scenario_ids)].groupby("canonical_scenario_id")
    }
    scenario_predicted_regime = {}
    for sid, raw_true_labels in val_tel_by_scenario.items():
        # Recompute Stage-1 PREDICTIONS (not the ground-truth labels) for
        # this scenario's own telemetry rows, then apply dwell/fallback.
        rows = telemetry[telemetry["canonical_scenario_id"] == sid].sort_values("step")
        raw_pred = stage1.predict(rows)
        effective, _ = apply_dwell_and_fallback(list(raw_pred))
        vals, counts = np.unique(effective, return_counts=True)
        scenario_predicted_regime[sid] = str(vals[np.argmax(counts)])
    val_df = val_df.copy()
    val_df["predicted_regime"] = val_df["canonical_scenario_id"].map(scenario_predicted_regime)
    missing_pred = val_df["predicted_regime"].isna().sum()
    summary["val_scenarios_missing_stage1_prediction"] = int(missing_pred)
    val_df = val_df.dropna(subset=["predicted_regime"])

    regime_fixed_best = regime_fixed_best_from_train(train_df)
    a = baseline_a_anwg(val_df)
    c = baseline_c_anwg(val_df)
    g = baseline_g_anwg(val_df)
    e = baseline_e_anwg(val_df, val_df["predicted_regime"], regime_fixed_best)
    d = baseline_d_anwg(val_df, val_df["predicted_regime"], stage2_selectors)

    assert not a.isna().any() and not c.isna().any() and not g.isna().any()
    assert not d.isna().any() and not e.isna().any()
    assert np.isfinite(a.to_numpy()).all() and np.isfinite(d.to_numpy()).all()

    n_benefit, per_regime_delta = multi_regime_benefit_count(val_df, d, a)
    summary["val_metrics_smoke"] = {
        "mean_delta_anwg_D_minus_A": delta_anwg(d, a),
        "mean_delta_anwg_E_minus_A": delta_anwg(e, a),
        "mean_anwg_A_best_global_fixed": float(a.mean()),
        "mean_anwg_C_oracle_regime_router": float(c.mean()),
        "mean_anwg_G_global_six_policy_oracle": float(g.mean()),
        "mean_anwg_D_learned_hierarchy": float(d.mean()),
        "multi_regime_benefit_count": n_benefit,
        "per_regime_delta_D_minus_A": per_regime_delta,
        "catastrophic_misroute_rate_scenario_level": catastrophic_misroute_rate(
            val_df["predicted_regime"], val_df["regime_ground_truth"]
        ),
    }

    # -----------------------------------------------------------------
    # Gate evaluator smoke (VAL numbers -- explicitly NOT a scientific verdict)
    # -----------------------------------------------------------------
    smoke_metrics = {
        "stage1_input_validity_fraction": 1.0,
        "router_macro_f1": macro_f1_val,
        "catastrophic_misroute_rate": summary["stage1_val_catastrophic_misroute_rate"],
        "leakage_instance_count": 0,
        "qualitative_all_clusters_attributable": None,  # not reviewed in a smoke run
    }
    gate_results = evaluate_all_gates(smoke_metrics)
    verdict = compute_verdict(gate_results, blended_microcase_sample_too_small=True, test_sample_insufficient_for_g5_ci=True)
    summary["gate_evaluator_smoke"] = {
        "note": "VAL-split numbers fed through the gate evaluator to prove it runs end-to-end; NOT a scientific verdict (G4-G9 not computed here; INCONCLUSIVE forced by construction).",
        "gates": {k: v.to_dict() for k, v in gate_results.items()},
        "verdict": verdict,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "smoke_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
