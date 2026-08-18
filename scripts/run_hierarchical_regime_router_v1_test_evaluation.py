#!/usr/bin/env python3
"""Hierarchical Regime Router v1 -- FIRST authorized held-out TEST
scientific evaluation.

Fits Stage-1 and Stage-2 on TRAIN ONLY, then reads the TEST split exactly
once to compute every SS Q metric and all 9 gates. This script does not
modify docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md,
configs/hierarchical_regime_router_v1_gates.json, or any of the
hierarchical_regime_router_v1 implementation modules -- it is pure
evaluation orchestration, kept separate from the frozen implementation
commit (2923087) so the scientific run is auditable independently of the
code that produced it.

G4 (Stage-2 preservation) and the blended-microcase G9(b) computation are
implemented here (not in the frozen implementation modules) because they
were not yet needed by the TRAIN/VAL-only smoke and are evaluation-only
logic operating on already-committed building blocks
(mean_regret/compute_native_pair_winner/regime_fixed_best_from_train/
catastrophic_misroute_rate), not a change to Stage-1/Stage-2/dwell/split
semantics.

Run exactly once. No result observed here is fed back into a second run.
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
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (  # noqa: E402
    ACTIVE_REGIMES,
    BLENDED_MICROCASE_BUILDERS,
    FALLBACK_REGIMES,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_CLASSES,
    STAGE2_CANDIDATES,
    Stage1Router,
    add_regime_labels,
    apply_dwell_and_fallback,
    assert_group_disjoint,
    build_splits,
    count_dwell_violations,
    regime_label_from_activity,
    route_action,
)
from llmserveopt.policy_separation.hierarchical_router_evaluation_v1 import (  # noqa: E402
    baseline_a_anwg,
    baseline_b_anwg,
    baseline_c_anwg,
    baseline_d_anwg,
    baseline_e_anwg,
    baseline_f_anwg,
    baseline_g_anwg,
    catastrophic_misroute_rate,
    delta_anwg,
    fit_baseline_b,
    fit_baseline_f,
    group_resampled_bootstrap_ci,
    load_scenario_level_dataset,
    mean_regret,
    multi_regime_benefit_count,
    oracle_gap_closure,
    regime_fixed_best_from_train,
)
from llmserveopt.policy_separation.hierarchical_router_gates_v1 import (  # noqa: E402
    compute_verdict,
    evaluate_all_gates,
    load_gates_config,
)
from llmserveopt.policy_separation.online_regime_signals_v1 import TelemetryRecordingPolicy  # noqa: E402
from llmserveopt.selector.hierarchical_stage2_selectors_v1 import fit_all_stage2_selectors  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

TELEMETRY_PATH = REPO_ROOT / "experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv"
MF_PSD_SCENARIOS = REPO_ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"
GATES_DOC = REPO_ROOT / "docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md"
GATES_JSON = REPO_ROOT / "configs/hierarchical_regime_router_v1_gates.json"
OUTPUT_DIR = REPO_ROOT / "experiments/hierarchical_regime_router_v1_test_evaluation"

# Judgment calls made explicit here (not silently baked into thresholds):
BLENDED_MIN_ACTIVE_PAIR_STEPS = 30  # below this, G9(b) is treated as "sample too small"
MIN_TEST_GROUPS_FOR_G5_CI = 5       # below this, G5's CI criterion is treated as "insufficient sample"


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


def stage1_test_metrics(stage1: Stage1Router, test_tel: pd.DataFrame) -> dict:
    y_true = test_tel["regime_label"].to_numpy()
    y_pred = stage1.predict(test_tel)
    accuracy = float((y_true == y_pred).mean())
    present_labels = sorted(set(y_true) | set(y_pred))
    macro_f1_present_only = float(f1_score(y_true, y_pred, average="macro", labels=present_labels))
    macro_f1_all_5_classes = float(f1_score(y_true, y_pred, average="macro", labels=list(REGIME_CLASSES), zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(REGIME_CLASSES))
    cm_dict = {t: {p: int(cm[i, j]) for j, p in enumerate(REGIME_CLASSES)} for i, t in enumerate(REGIME_CLASSES)}
    return {
        "n_test_rows": int(len(test_tel)),
        "true_label_distribution": {k: int(v) for k, v in pd.Series(y_true).value_counts().items()},
        "predicted_label_distribution": {k: int(v) for k, v in pd.Series(y_pred).value_counts().items()},
        "accuracy": accuracy,
        "macro_f1_present_classes_only": macro_f1_present_only,
        "macro_f1_present_classes": present_labels,
        "macro_f1_all_5_classes_zero_division_0": macro_f1_all_5_classes,
        "confusion_matrix": cm_dict,
        "catastrophic_misroute_rate": catastrophic_misroute_rate(pd.Series(y_pred), pd.Series(y_true)),
        "none_rate": float((y_pred == "NONE").mean()),
        "overlap_rate": float((y_pred == "OVERLAP").mean()),
        "true_none_rate": float((y_true == "NONE").mean()),
        "true_overlap_rate": float((y_true == "OVERLAP").mean()),
    }


def stage2_test_metrics(stage2_selectors: dict, test_df: pd.DataFrame, regime_fixed_best: dict) -> dict:
    out = {}
    for regime in (REGIME_A, REGIME_B, REGIME_C):
        sub = test_df[test_df["regime_ground_truth"] == regime]
        if len(sub) == 0:
            out[regime] = {"status": "NOT_EVALUABLE", "reason": "0 TEST scenarios for this regime"}
            continue
        p0, p1 = STAGE2_CANDIDATES[regime]
        oracle = sub[[p0, p1]].max(axis=1)
        sel = stage2_selectors.get(regime)
        if sel is None:
            out[regime] = {"status": "NOT_EVALUABLE", "reason": "no TRAIN rows to fit a selector"}
            continue
        preds = sel.predict(sub)
        achieved = pd.Series([sub[p].iloc[i] for i, p in enumerate(preds)], index=sub.index)
        regret = oracle - achieved
        fixed_col = regime_fixed_best.get(regime)
        fixed_regret = (oracle - sub[fixed_col]) if fixed_col else None
        entry = {
            "status": "EVALUATED",
            "n_test_rows": int(len(sub)),
            "mean_regret": float(regret.mean()),
            "epsilon_optimal_accuracy": float((regret <= 0.01).mean()),
            "predicted_policy_distribution": {k: int(v) for k, v in pd.Series(preds).value_counts().items()},
        }
        if fixed_regret is not None:
            entry["best_fixed_policy"] = fixed_col
            entry["best_fixed_mean_regret"] = float(fixed_regret.mean())
            standalone_gain = float(fixed_regret.mean() - regret.mean())
            entry["standalone_mean_regret_improvement_vs_fixed"] = standalone_gain
        out[regime] = entry
    return out


def stage2_preservation_g4(
    stage2_test: dict, test_df: pd.DataFrame, d_anwg: pd.Series, regime_fixed_best: dict
) -> dict:
    """G4: fraction of each regime's STANDALONE selector regret-improvement
    (vs that regime's best-fixed) retained once the FULL hierarchy (Stage-1
    routing included, so possible misrouting/fallback is charged against
    it) is used instead."""
    out = {}
    for regime, entry in stage2_test.items():
        if entry.get("status") != "EVALUATED" or "standalone_mean_regret_improvement_vs_fixed" not in entry:
            out[regime] = {"status": "NOT_EVALUABLE", "reason": entry.get("reason", "standalone gain not computable")}
            continue
        standalone_gain = entry["standalone_mean_regret_improvement_vs_fixed"]
        sub = test_df[test_df["regime_ground_truth"] == regime]
        p0, p1 = STAGE2_CANDIDATES[regime]
        oracle = sub[[p0, p1]].max(axis=1)
        fixed_col = regime_fixed_best[regime]
        fixed_regret_mean = float((oracle - sub[fixed_col]).mean())
        integrated_achieved = d_anwg.loc[sub.index]
        integrated_regret_mean = float((oracle - integrated_achieved).mean())
        integrated_gain = fixed_regret_mean - integrated_regret_mean
        if abs(standalone_gain) < 1e-12:
            out[regime] = {
                "status": "NOT_EVALUABLE", "reason": "standalone gain ~0 (division undefined)",
                "standalone_gain": standalone_gain, "integrated_gain": integrated_gain,
            }
            continue
        fraction = integrated_gain / standalone_gain
        out[regime] = {
            "status": "EVALUATED",
            "standalone_gain": standalone_gain,
            "integrated_gain": integrated_gain,
            "fraction_retained": fraction,
        }
    return out


def run_blended_microcase(name: str, builder, stage1: Stage1Router) -> dict:
    scenario = builder()
    policy = TelemetryRecordingPolicy(
        FIFOPolicy(), canonical_scenario_id=scenario.scenario_id,
        mechanism_family="blended_microcase_v1", sample_stride_steps=5,
    )
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    sim.run(policy, workload_tag=scenario.scenario_id, seed=scenario.seed)

    n_steps = len(policy.rows)
    if n_steps == 0:
        return {"case": name, "n_steps": 0, "status": "NO_TELEMETRY_ROWS"}

    true_regimes = [
        regime_label_from_activity(r.labels.a_active, r.labels.b_active_v2, r.labels.c_active)
        for r in policy.rows
    ]
    tel_df = pd.DataFrame({
        "contention_score_v2": [r.signals.contention_score_v2 for r in policy.rows],
        "priority_skew": [r.signals.priority_skew for r in policy.rows],
        "kv_pressure": [r.signals.kv_pressure for r in policy.rows],
        "queue_length": [r.signals.queue_length for r in policy.rows],
    })
    raw_pred = list(stage1.predict(tel_df))
    effective, diag = apply_dwell_and_fallback(raw_pred)

    n_overlap_ground_truth = sum(1 for r in true_regimes if r == "OVERLAP")
    n_multi_active_ground_truth = sum(
        1 for r in policy.rows if (int(r.labels.a_active) + int(r.labels.b_active_v2) + int(r.labels.c_active)) > 1
    )
    both_active_mask = [
        (t in ACTIVE_REGIMES and p in ACTIVE_REGIMES) for t, p in zip(true_regimes, effective)
    ]
    n_active_pair_steps = sum(both_active_mask)
    cata_rate = catastrophic_misroute_rate(pd.Series(effective), pd.Series(true_regimes))

    return {
        "case": name,
        "n_steps": n_steps,
        "true_regime_distribution": {k: int(v) for k, v in pd.Series(true_regimes).value_counts().items()},
        "router_raw_prediction_distribution": {k: int(v) for k, v in pd.Series(raw_pred).value_counts().items()},
        "effective_regime_distribution_after_dwell": {k: int(v) for k, v in pd.Series(effective).value_counts().items()},
        "n_overlap_ground_truth_steps": n_overlap_ground_truth,
        "n_multi_active_ground_truth_steps": n_multi_active_ground_truth,
        "dwell_violation_count": diag.dwell_violation_count,
        "independent_dwell_check": count_dwell_violations(effective),
        "fallback_rate": diag.fallback_rate,
        "n_active_pair_comparable_steps": n_active_pair_steps,
        "catastrophic_misroute_rate": cata_rate,
    }


def main() -> int:
    report: dict = {
        "schema_version": "hierarchical_regime_router_v1_test_evaluation.1.0.0",
        "mode": "FIRST_AUTHORIZED_TEST_EVALUATION",
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # -----------------------------------------------------------------
    # 1. Preregistration integrity
    # -----------------------------------------------------------------
    report["preregistration_integrity"] = {
        "git_head_sha": _git_head_sha(),
        "git_tree_dirty": _git_dirty(),
        "design_doc_sha256": _sha256_of_file(GATES_DOC),
        "gates_json_sha256": _sha256_of_file(GATES_JSON),
        "telemetry_csv_sha256": _sha256_of_file(TELEMETRY_PATH),
        "mf_psd_scenarios_sha256": _sha256_of_file(MF_PSD_SCENARIOS),
    }

    scen = pd.read_csv(MF_PSD_SCENARIOS)
    split_map = build_splits(scen)
    assert_group_disjoint(scen, split_map)
    scen["split"] = scen["canonical_scenario_id"].map(split_map)
    report["split_counts"] = scen["split"].value_counts().to_dict()

    telemetry = add_regime_labels(pd.read_csv(TELEMETRY_PATH))
    telemetry["split"] = telemetry["canonical_scenario_id"].map(split_map)
    train_tel = telemetry[telemetry["split"] == "train"]
    test_tel = telemetry[telemetry["split"] == "test"]

    # -----------------------------------------------------------------
    # 2. Fit on TRAIN only
    # -----------------------------------------------------------------
    stage1 = Stage1Router().fit(train_tel)

    scenario_df = load_scenario_level_dataset()
    train_df = scenario_df[scenario_df["split"] == "train"]
    test_df = scenario_df[scenario_df["split"] == "test"]
    report["fit_data_confirmation"] = {
        "stage1_train_telemetry_rows": int(len(train_tel)),
        "stage1_test_telemetry_rows": int(len(test_tel)),
        "stage2_train_scenario_rows": int(len(train_df)),
        "stage2_test_scenario_rows": int(len(test_df)),
        "test_regime_ground_truth_distribution": {k: int(v) for k, v in test_df["regime_ground_truth"].value_counts().items()},
    }

    train_by_regime = {r: train_df[train_df["regime_ground_truth"] == r] for r in ACTIVE_REGIMES}
    stage2_selectors = fit_all_stage2_selectors(train_by_regime)
    regime_fixed_best = regime_fixed_best_from_train(train_df)
    pipe_b, num_cols_b, cat_cols_b = fit_baseline_b(train_df)
    pipe_f, num_cols_f, cat_cols_f = fit_baseline_f(train_df)

    # -----------------------------------------------------------------
    # 3. Stage-1 TEST metrics
    # -----------------------------------------------------------------
    report["stage1_test_metrics"] = stage1_test_metrics(stage1, test_tel)

    # -----------------------------------------------------------------
    # 4. Scenario-level predicted regime for TEST (Stage-1 predictions,
    #    dwell/fallback applied, majority vote per scenario)
    # -----------------------------------------------------------------
    predicted_regime = {}
    for sid, group in telemetry[telemetry["canonical_scenario_id"].isin(test_df["canonical_scenario_id"])].groupby("canonical_scenario_id"):
        rows = group.sort_values("step")
        raw_pred = list(stage1.predict(rows))
        effective, _ = apply_dwell_and_fallback(raw_pred)
        vals, counts = np.unique(effective, return_counts=True)
        predicted_regime[sid] = str(vals[np.argmax(counts)])
    test_df = test_df.copy()
    test_df["predicted_regime"] = test_df["canonical_scenario_id"].map(predicted_regime)
    assert test_df["predicted_regime"].notna().all(), "every TEST scenario must have a predicted regime"

    # -----------------------------------------------------------------
    # 5. Stage-2 TEST metrics (per-regime, standalone)
    # -----------------------------------------------------------------
    stage2_test = stage2_test_metrics(stage2_selectors, test_df, regime_fixed_best)
    report["stage2_test_metrics"] = stage2_test

    # -----------------------------------------------------------------
    # 6. All 7 baselines on TEST
    # -----------------------------------------------------------------
    a = baseline_a_anwg(test_df)
    b = baseline_b_anwg(test_df, pipe_b, num_cols_b, cat_cols_b)
    c = baseline_c_anwg(test_df)
    d = baseline_d_anwg(test_df, test_df["predicted_regime"], stage2_selectors)
    e = baseline_e_anwg(test_df, test_df["predicted_regime"], regime_fixed_best)
    f = baseline_f_anwg(test_df, pipe_f, num_cols_f, cat_cols_f)
    g = baseline_g_anwg(test_df)
    for name, series in (("A", a), ("B", b), ("C", c), ("D", d), ("E", e), ("F", f), ("G", g)):
        assert not series.isna().any(), f"baseline {name} produced NaN"
        assert np.isfinite(series.to_numpy()).all(), f"baseline {name} produced non-finite values"
    report["baseline_table_mean_anwg"] = {
        "A_best_global_fixed": float(a.mean()),
        "B_prior_flat_selector": float(b.mean()),
        "C_oracle_regime_router_native_pair": float(c.mean()),
        "D_learned_stage1_plus_stage2": float(d.mean()),
        "E_learned_stage1_plus_regime_fixed_best": float(e.mean()),
        "F_hidden_family_aware_selector": float(f.mean()),
        "G_global_six_policy_oracle": float(g.mean()),
    }

    # -----------------------------------------------------------------
    # 7. End-to-end metrics
    # -----------------------------------------------------------------
    n_groups_test = test_df["group_key"].nunique()
    ci_lo, ci_hi = group_resampled_bootstrap_ci(test_df, d, a, n_boot=5000, ci=0.90, seed=20260818)
    n_benefit, per_regime_delta = multi_regime_benefit_count(test_df, d, a)
    dwell_total_violations = 0
    total_transitions = 0
    total_steps = 0
    fallback_steps = 0
    for sid, group in telemetry[telemetry["canonical_scenario_id"].isin(test_df["canonical_scenario_id"])].groupby("canonical_scenario_id"):
        rows = group.sort_values("step")
        raw_pred = list(stage1.predict(rows))
        effective, diag = apply_dwell_and_fallback(raw_pred)
        dwell_total_violations += diag.dwell_violation_count
        total_transitions += diag.total_transitions
        total_steps += len(effective)
        fallback_steps += sum(1 for x in effective if x in FALLBACK_REGIMES)

    report["end_to_end_test_metrics"] = {
        "canonical_anwg_D": float(d.mean()),
        "delta_anwg_D_minus_A": delta_anwg(d, a),
        "bootstrap_ci_90_D_minus_A": {"lower": ci_lo, "upper": ci_hi, "n_test_groups": int(n_groups_test)},
        "regret_to_global_six_policy_oracle_D_vs_G": mean_regret(d, g),
        "oracle_gap_closure_D_vs_C": oracle_gap_closure(float(d.mean()), float(a.mean()), float(c.mean())),
        "per_regime_delta_D_minus_A": per_regime_delta,
        "multi_regime_benefit_count": n_benefit,
        "switching_total_transitions_on_test": total_transitions,
        "switching_rate_per_1000_steps_on_test": (1000.0 * total_transitions / total_steps) if total_steps else None,
        "fallback_rate_on_test": (fallback_steps / total_steps) if total_steps else None,
        "dwell_violation_count_on_test": dwell_total_violations,
    }
    # macro-regime ANWG = mean over regimes ACTUALLY PRESENT in TEST of that regime's mean D-ANWG (unweighted)
    per_regime_mean_d = {
        r: float(d[test_df["regime_ground_truth"] == r].mean())
        for r in ACTIVE_REGIMES if (test_df["regime_ground_truth"] == r).any()
    }
    report["end_to_end_test_metrics"]["per_regime_mean_anwg_D"] = per_regime_mean_d
    report["end_to_end_test_metrics"]["macro_regime_anwg_D_present_regimes_only"] = float(np.mean(list(per_regime_mean_d.values())))

    # -----------------------------------------------------------------
    # 8. G4 (Stage-2 preservation)
    # -----------------------------------------------------------------
    g4_by_regime = stage2_preservation_g4(stage2_test, test_df, d, regime_fixed_best)
    report["g4_stage2_preservation_by_regime"] = g4_by_regime

    # -----------------------------------------------------------------
    # 9. Blended-regime microcases
    # -----------------------------------------------------------------
    blended_results = {name: run_blended_microcase(name, builder, stage1) for name, builder in BLENDED_MICROCASE_BUILDERS.items()}
    report["blended_microcase_results"] = blended_results
    n_active_pair_total = sum(r.get("n_active_pair_comparable_steps", 0) for r in blended_results.values())
    weighted_cata = 0.0
    if n_active_pair_total > 0:
        weighted_cata = sum(
            r.get("catastrophic_misroute_rate", 0.0) * r.get("n_active_pair_comparable_steps", 0)
            for r in blended_results.values()
        ) / n_active_pair_total
    blended_sample_too_small = n_active_pair_total < BLENDED_MIN_ACTIVE_PAIR_STEPS
    report["blended_microcase_summary"] = {
        "n_active_pair_comparable_steps_total": n_active_pair_total,
        "sample_too_small_threshold": BLENDED_MIN_ACTIVE_PAIR_STEPS,
        "sample_too_small": blended_sample_too_small,
        "weighted_catastrophic_misroute_rate": weighted_cata if n_active_pair_total > 0 else None,
    }

    # -----------------------------------------------------------------
    # 10. G9(a): Family-C held-out subset delta ANWG (== TEST's KV_MEMORY_PRESSURE rows,
    #     since Family C's *entire* TEST allocation IS its held-out-eval-seed rows)
    # -----------------------------------------------------------------
    fam_c_mask = test_df["regime_ground_truth"] == REGIME_C
    family_c_held_out_delta = float((d[fam_c_mask] - a[fam_c_mask]).mean()) if fam_c_mask.any() else None
    report["g9a_family_c_held_out_delta_anwg"] = family_c_held_out_delta

    # -----------------------------------------------------------------
    # 11. G1-G9 mechanical scoring
    # -----------------------------------------------------------------
    g4_fraction_dict = {
        r: v["fraction_retained"] for r, v in g4_by_regime.items() if v.get("status") == "EVALUATED"
    }
    test_sample_insufficient_for_g5_ci = n_groups_test < MIN_TEST_GROUPS_FOR_G5_CI

    metrics = {
        "stage1_input_validity_fraction": 1.0,  # structural: extract_inputs() enforces the 4-column allowlist (G1)
        "router_macro_f1": report["stage1_test_metrics"]["macro_f1_present_classes_only"],
        "catastrophic_misroute_rate": report["stage1_test_metrics"]["catastrophic_misroute_rate"],
        "stage2_preservation_fraction_by_regime": g4_fraction_dict if g4_fraction_dict else None,
        "mean_delta_anwg": report["end_to_end_test_metrics"]["delta_anwg_D_minus_A"],
        "bootstrap_ci_lower": ci_lo if not test_sample_insufficient_for_g5_ci else None,
        "oracle_gap_closure": report["end_to_end_test_metrics"]["oracle_gap_closure_D_vs_C"],
        "multi_regime_benefit_count": n_benefit,
        "leakage_instance_count": 0,  # structural, verified by the frozen allowlist tests (G8a)
        "qualitative_all_clusters_attributable": None,  # (b) requires human review, not auto-computable
        "family_c_held_out_delta_anwg": family_c_held_out_delta,
        "blended_microcase_catastrophic_rate": weighted_cata if not blended_sample_too_small else None,
    }
    config = load_gates_config()
    gates = evaluate_all_gates(metrics, config)
    report["gate_metrics_input"] = metrics
    report["gate_results"] = {k: v.to_dict() for k, v in gates.items()}
    report["gate_evaluation_notes"] = {
        "G2_uses_present_classes_only": (
            "TEST has 0 true PREFILL_DECODE_CONTENTION and 0 true OVERLAP telemetry rows "
            "(Family B's 8 groups happened to hash entirely into TRAIN/VAL) -- macro-F1 is "
            "computed over the classes actually present in TEST ground truth "
            f"({report['stage1_test_metrics']['macro_f1_present_classes']}), not all 5, to avoid "
            "an artifact of scoring an absent class. The all-5-class variant is also reported "
            "(macro_f1_all_5_classes_zero_division_0) for transparency."
        ),
        "G4_regime_B_not_evaluable": (
            "Stage-2 Regime B (PREFILL_DECODE_CONTENTION) has 0 TEST scenarios for the same "
            "reason -- G4's per-regime fraction can only be computed for Regimes A and C on "
            "this TEST split; Regime B is reported as NOT_EVALUABLE, not silently dropped or "
            "assumed passing."
        ),
        "G5_ci_sample_size": (
            f"TEST has {n_groups_test} unique group_key values; "
            f"{'below' if test_sample_insufficient_for_g5_ci else 'at/above'} the "
            f"{MIN_TEST_GROUPS_FOR_G5_CI}-group threshold used here to judge whether the "
            "group-resampled bootstrap CI is meaningful."
        ),
    }

    # -----------------------------------------------------------------
    # 12. Mechanical verdict
    # -----------------------------------------------------------------
    verdict = compute_verdict(
        gates,
        blended_microcase_sample_too_small=blended_sample_too_small,
        test_sample_insufficient_for_g5_ci=test_sample_insufficient_for_g5_ci,
    )
    report["final_verdict"] = verdict

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "test_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\nWrote {out_path}")
    print(f"\nFINAL VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
