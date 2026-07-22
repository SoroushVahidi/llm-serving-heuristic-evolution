#!/usr/bin/env python3
"""Evaluate the joint state-policy suitability model and produce the
scientific report (docs/current/STATE_POLICY_SUITABILITY_REPORT.md).

Loads the small discriminative fixture built by
scripts/build_state_policy_suitability_fixture.py, fits the three joint
reward-model encodings (identity/structural/hybrid), the pre-existing
independent per-policy regressor and classifier baselines
(selector/advanced.py), the joint discrete selector with conservative
suitability, the Delta_SCORPIO_WSP pairwise-advantage diagnostic, and the
held-out-policy/held-out-family generalization pilots. CPU-only, no GPU/
network/paid service.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES  # noqa: E402
from llmserveopt.selector.advanced import PolicyClassifierSelector, PolicyRewardRegressorSelector, anwg_column  # noqa: E402
from llmserveopt.selector.suitability.dataset import group_by_state, rows_with_reward  # noqa: E402
from llmserveopt.selector.suitability.models import IndependentPerPolicyRewardModel, JointRewardModel  # noqa: E402
from llmserveopt.selector.suitability.selector import (  # noqa: E402
    DeltaModel,
    build_delta_rows,
    delta_consistency_with_joint_model,
    evaluate_delta_model,
    evaluate_selection,
    held_out_family_pilot,
    held_out_policy_pilot,
    joint_select,
    load_policy_families,
    margin_weighted_regret,
    oracle_best,
    structural_distance_vs_performance_disagreement,
)


def _pivot_wide(rows_by_state: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Rebuild the legacy wide per-state row shape (anwg_<policy> columns)
    that the pre-existing selector/advanced.py selectors expect, purely as
    an adapter for this comparison -- not a new canonical schema."""
    wide_rows = []
    for state_id, rows in rows_by_state.items():
        wide = dict(rows[0]["state_features"])
        wide["state_id"] = state_id
        wide["split"] = rows[0]["split"]
        for r in rows:
            wide[anwg_column(r["policy_name"])] = r["reward_anwg"]
        best_policy, _ = oracle_best({r["policy_name"]: r["reward_anwg"] for r in rows})
        wide["label_best_policy"] = best_policy
        wide_rows.append(wide)
    return wide_rows


def _best_fixed_policy(train_rows: List[Dict[str, Any]], all_policies: List[str]) -> str:
    means = {}
    for policy in all_policies:
        vals = [r["reward_anwg"] for r in train_rows if r["policy_name"] == policy]
        if vals:
            means[policy] = float(np.mean(vals))
    return max(means, key=means.get)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", default="results/state_policy_suitability_fixture/report_run")
    parser.add_argument("--out-dir", default="results/state_policy_suitability_report/latest")
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()
    fixture_dir = ROOT / args.fixture_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads((fixture_dir / "long_format_rows.json").read_text())
    rows = rows_with_reward(rows)
    all_policies = list(POLICY_LIBRARY_V2_NAMES)

    train_rows = [r for r in rows if r["split"] in ("TRAIN", "VALIDATION")]
    test_rows = [r for r in rows if r["split"] == "TEST"]
    train_rbs = group_by_state(train_rows)
    test_rbs = group_by_state(test_rows)
    best_fixed = _best_fixed_policy(train_rows, all_policies)

    report: Dict[str, Any] = {
        "n_train_states": len(train_rbs), "n_test_states": len(test_rbs),
        "n_train_rows": len(train_rows), "n_test_rows": len(test_rows),
        "best_fixed_policy": best_fixed, "lambda": args.lam,
    }

    # ------------------------------------------------------------------
    # Models 1/2/3
    # ------------------------------------------------------------------
    models: Dict[str, JointRewardModel] = {}
    for encoding, label in [("identity", "model1_identity"), ("structural", "model2_structural"), ("hybrid", "model3_hybrid")]:
        m = JointRewardModel(name=label, encoding=encoding, all_policies=all_policies, n_estimators=200, max_depth=8, random_state=args.seed).fit(train_rows)
        models[label] = m

    independent = IndependentPerPolicyRewardModel(name="independent_per_policy", all_policies=all_policies, random_state=args.seed).fit(train_rows)

    prediction_quality = {}
    actual_test = np.asarray([r["reward_anwg"] for r in test_rows])
    for label, m in models.items():
        preds = m.predict_mean(test_rows)
        prediction_quality[label] = {
            "mae": float(np.mean(np.abs(preds - actual_test))),
            "rmse": float(np.sqrt(np.mean((preds - actual_test) ** 2))),
        }
    ind_preds = independent.predict_mean(test_rows)
    prediction_quality["independent_per_policy"] = {
        "mae": float(np.mean(np.abs(ind_preds - actual_test))),
        "rmse": float(np.sqrt(np.mean((ind_preds - actual_test) ** 2))),
    }
    report["reward_prediction_quality_on_test"] = prediction_quality

    # ------------------------------------------------------------------
    # Joint discrete selector vs baselines
    # ------------------------------------------------------------------
    selector_results = {}
    for label, m in models.items():
        selections = joint_select(m, test_rbs, lam=args.lam)
        ev = evaluate_selection(test_rbs, selections, best_fixed_policy=best_fixed)
        ev["margin_weighted_regret"] = margin_weighted_regret(test_rbs, selections)
        selector_results[label] = ev

    # Independent-per-policy suitability-based selection (same suitability
    # formula, independent-per-policy mean/uncertainty).
    ind_selections = joint_select(independent, test_rbs, lam=args.lam)
    ev = evaluate_selection(test_rbs, ind_selections, best_fixed_policy=best_fixed)
    ev["margin_weighted_regret"] = margin_weighted_regret(test_rbs, ind_selections)
    selector_results["independent_per_policy"] = ev

    # Fixed-policy and oracle envelope references.
    fixed_selections = {sid: best_fixed for sid in test_rbs}
    selector_results["fixed_best_train_policy"] = evaluate_selection(test_rbs, fixed_selections, best_fixed_policy=best_fixed)
    oracle_selections = {sid: oracle_best({r["policy_name"]: r["reward_anwg"] for r in rs})[0] for sid, rs in test_rbs.items()}
    selector_results["oracle_envelope_27_policy"] = evaluate_selection(test_rbs, oracle_selections, best_fixed_policy=best_fixed)

    # Existing pre-integration selectors (selector/advanced.py) as the
    # "existing strongest selector" / "direct classifier" comparison points,
    # via the legacy wide-row adapter.
    wide_train = _pivot_wide(train_rbs)
    wide_test = _pivot_wide(test_rbs)
    feat_cols = sorted(k for k in wide_train[0].keys() if k.startswith("feat_"))
    existing_regressor = PolicyRewardRegressorSelector(
        name="existing_rf_reward_regression", allowed_policies=all_policies, feature_cols=feat_cols,
        n_estimators=200, max_depth=8, random_state=args.seed,
    ).fit(wide_train)
    existing_classifier = PolicyClassifierSelector(
        name="existing_rf_classifier", allowed_policies=all_policies, feature_cols=feat_cols,
        label_col="label_best_policy", n_estimators=200, max_depth=8, random_state=args.seed,
    ).fit(wide_train)
    for name, model in [("existing_advanced_regressor", existing_regressor), ("existing_advanced_classifier", existing_classifier)]:
        preds = model.predict(wide_test)
        selections = {row["state_id"]: pred for row, pred in zip(wide_test, preds)}
        selector_results[name] = evaluate_selection(test_rbs, selections, best_fixed_policy=best_fixed)

    report["selector_results"] = selector_results

    # ------------------------------------------------------------------
    # Delta_SCORPIO_WSP(x)
    # ------------------------------------------------------------------
    policy_a, policy_b = "scorpio_style_slo_guard", "weighted_shortest_processing"
    delta_train = build_delta_rows(train_rbs, policy_a=policy_a, policy_b=policy_b)
    delta_test = build_delta_rows(test_rbs, policy_a=policy_a, policy_b=policy_b)
    delta_report: Dict[str, Any] = {"n_train": len(delta_train), "n_test": len(delta_test)}
    if delta_train and delta_test:
        delta_model = DeltaModel(random_state=args.seed).fit(delta_train)
        delta_report["evaluation"] = evaluate_delta_model(delta_model, delta_test)
        delta_report["consistency_with_hybrid_joint_model"] = delta_consistency_with_joint_model(
            delta_model, models["model3_hybrid"], delta_test, test_rbs, policy_a=policy_a, policy_b=policy_b,
        )
    else:
        delta_report["status"] = "insufficient_states_with_both_policies"
    report["delta_scorpio_wsp"] = delta_report

    # ------------------------------------------------------------------
    # Held-out-policy pilots -- one per representative family, only
    # policies with a faithful (EXACT/APPROXIMATE, not UNSUPPORTED) genome
    # mapping per docs/current/POLICY_GENOME_COVERAGE_AUDIT.md.
    # ------------------------------------------------------------------
    held_out_candidates = [
        "edf",                       # SLO/deadline-aware (EXACT)
        "weighted_shortest_processing",  # shortest-work/service-time (EXACT)
        "adaptive_chunked_prefill",  # prefill-aware (APPROXIMATE)
        "kv_constrained_online",     # KV-aware (APPROXIMATE)
        "aging_priority",            # fairness/aging (APPROXIMATE)
        "fifo",                      # general baseline (EXACT)
    ]
    report["held_out_policy_pilots"] = {
        policy: held_out_policy_pilot(rows, policy, all_policies=all_policies, encoding="hybrid")
        for policy in held_out_candidates
    }

    # ------------------------------------------------------------------
    # Held-out-family pilots (documented component taxonomy, not invented).
    # slo_deadline_handling now has 9/10 members with a faithful (EXACT or
    # APPROXIMATE) genome mapping, vs. kv_memory_pressure's more mixed
    # coverage -- report both for comparison.
    # ------------------------------------------------------------------
    report["held_out_family_pilots"] = {}
    for family_name in ("slo_deadline_handling", "kv_memory_pressure"):
        family_policies = load_policy_families(family_name)
        report["held_out_family_pilots"][family_name] = held_out_family_pilot(
            rows, family_policies, all_policies=all_policies, family_name=family_name,
        )

    # ------------------------------------------------------------------
    # Structural-distance diagnostics, over every faithfully-mapped
    # (EXACT/APPROXIMATE) policy present in this dataset.
    # ------------------------------------------------------------------
    mapped_present = sorted({r["policy_name"] for r in rows} & set(
        p for p in all_policies
        if p not in ("greedy_token_fill", "least_loaded", "random_feasible", "best_fit",
                     "vllm_style_token_budget", "sarathi_style", "splitfuse_style", "slai_style_phase_aware")
    ))
    all_rbs = group_by_state(rows)
    report["structural_distance_diagnostics"] = structural_distance_vs_performance_disagreement(all_rbs, mapped_present)

    report["runtime_s"] = round(time.perf_counter() - t0, 3)
    (out_dir / "state_policy_suitability_results.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("selector_results", "held_out_policy_pilots")}, indent=2, default=str))
    print(f"Full results: {out_dir / 'state_policy_suitability_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
