#!/usr/bin/env python3
"""Evaluate structure-aware state-policy suitability models against the
existing RF-based baselines (policy-ID, structural-only, hybrid,
nearest-structural-neighbor), and produce the scientific report.

Reuses the same 32-window discriminative fixture as
run_state_policy_suitability_report.py (built by
scripts/build_state_policy_suitability_fixture.py) -- no new simulation is
launched. CPU-only, no GPU/network/paid service.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES  # noqa: E402
from llmserveopt.policies.structural_synthesis import map_policy_to_genome  # noqa: E402
from llmserveopt.selector.suitability.dataset import group_by_state, rows_with_reward  # noqa: E402
from llmserveopt.selector.suitability.models import IndependentPerPolicyRewardModel, JointRewardModel  # noqa: E402
from llmserveopt.selector.suitability.selector import (  # noqa: E402
    build_delta_rows,
    DeltaModel,
    evaluate_delta_model,
    evaluate_selection,
    held_out_family_pilot,
    held_out_policy_split,
    joint_select,
    load_policy_families,
    margin_weighted_regret,
    oracle_best,
    structural_distance_vs_performance_disagreement,
    top2_margin,
    true_reward_row,
)
from llmserveopt.selector.suitability.structural_models import (  # noqa: E402
    KernelSuitabilityModel,
    ResidualTransferModel,
    StateConditionedNeighborModel,
    StructuralDistanceIndex,
    StructuralKNNModel,
)

FAITHFUL_STATUSES = {"EXACT", "APPROXIMATE"}


def faithfully_mapped_policies() -> List[str]:
    return sorted(
        p for p in POLICY_LIBRARY_V2_NAMES
        if map_policy_to_genome(p).metadata.get("mapping_status") in FAITHFUL_STATUSES
    )


def mae(preds: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - actual)))


def rmse(preds: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - actual) ** 2)))


def ranking_quality(rows_by_state: Dict[str, List[Dict[str, Any]]], model, policies: List[str]) -> Dict[str, Any]:
    """Spearman-style rank correlation between predicted and true policy
    ordering within each state, averaged; plus top-1 accuracy."""
    from scipy.stats import spearmanr  # already a transitive dependency via sklearn/pandas stack

    correlations = []
    top1_correct = 0
    n = 0
    for state_id, rows in rows_by_state.items():
        usable = [r for r in rows if r["policy_name"] in policies and r.get("reward_anwg") is not None]
        if len(usable) < 2:
            continue
        preds = model.predict_mean(usable)
        actual = np.asarray([r["reward_anwg"] for r in usable])
        corr, _ = spearmanr(preds, actual)
        if not np.isnan(corr):
            correlations.append(corr)
        pred_best = usable[int(np.argmax(preds))]["policy_name"]
        true_best = usable[int(np.argmax(actual))]["policy_name"]
        top1_correct += int(pred_best == true_best)
        n += 1
    return {
        "mean_spearman_rank_correlation": float(np.mean(correlations)) if correlations else None,
        "top1_accuracy": (top1_correct / n) if n else None,
        "n_states": n,
    }


def evaluate_model_on_rows(model, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    actual = np.asarray([r["reward_anwg"] for r in rows])
    preds = model.predict_mean(rows)
    return {"mae": mae(preds, actual), "rmse": rmse(preds, actual), "n_rows": len(rows)}


def build_models(
    train_rows: List[Dict[str, Any]],
    all_policies: List[str],
    distance_index: StructuralDistanceIndex,
    seed: int,
    *,
    lookup_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """`lookup_rows`: the reward table the transductive structural-lookup
    models (KNN/kernel/state-conditioned/residual-transfer) use to find
    sibling policies' true rewards at query states. Defaults to
    `train_rows`, which is correct for held-out-*policy* evaluation (query
    states are already in train_rows via the other 26 policies). For a
    standard train/test *state* split, `lookup_rows` must be widened to
    include the test states (via their non-target-policy rows) or every
    structural-lookup model degenerates to a constant-zero prediction --
    see the report's "ID evaluation lookup fix" note."""
    models: Dict[str, Any] = {}
    models["policy_id_rf"] = JointRewardModel(name="policy_id_rf", encoding="identity", all_policies=all_policies, random_state=seed).fit(train_rows)
    models["structural_only_rf"] = JointRewardModel(name="structural_only_rf", encoding="structural", all_policies=all_policies, random_state=seed).fit(train_rows)
    models["hybrid_rf"] = JointRewardModel(name="hybrid_rf", encoding="hybrid", all_policies=all_policies, random_state=seed).fit(train_rows)
    models["independent_per_policy"] = IndependentPerPolicyRewardModel(name="independent_per_policy", all_policies=all_policies, random_state=seed).fit(train_rows)

    for k in (1, 3, 5):
        for weighting in ("uniform", "inverse_distance", "exponential"):
            name = f"structural_knn_k{k}_{weighting}"
            models[name] = StructuralKNNModel(name=name, all_policies=all_policies, k=k, weighting=weighting, tau=2.0, distance_index=distance_index).fit(train_rows, lookup_rows=lookup_rows)

    for tau in (0.5, 2.0, 5.0):
        name = f"kernel_tau{tau}"
        models[name] = KernelSuitabilityModel(name=name, all_policies=all_policies, tau=tau, distance_index=distance_index).fit(train_rows, lookup_rows=lookup_rows)

    models["state_conditioned_neighbor"] = StateConditionedNeighborModel(
        name="state_conditioned_neighbor", all_policies=all_policies, tau=2.0, k=5, distance_index=distance_index,
    ).fit(train_rows, lookup_rows=lookup_rows)

    for scheme in ("uniform", "margin_plus_epsilon"):
        name = f"residual_transfer_{scheme}"
        models[name] = ResidualTransferModel(
            name=name, all_policies=all_policies, k=5, weighting="inverse_distance", tau=2.0,
            distance_index=distance_index, weight_scheme=scheme,
        ).fit(train_rows, lookup_rows=lookup_rows)
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", default="results/state_policy_suitability_fixture/report_run_v2")
    parser.add_argument("--out-dir", default="results/structural_suitability_report/latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=0.5)
    args = parser.parse_args()

    t0 = time.perf_counter()
    fixture_dir = ROOT / args.fixture_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads((fixture_dir / "long_format_rows.json").read_text())
    rows = rows_with_reward(rows)
    all_policies = list(POLICY_LIBRARY_V2_NAMES)
    faithful = faithfully_mapped_policies()
    distance_index = StructuralDistanceIndex(all_policies)

    id_train = [r for r in rows if r["split"] in ("TRAIN", "VALIDATION")]
    id_test = [r for r in rows if r["split"] == "TEST"]
    id_train_rbs = group_by_state(id_train)
    id_test_rbs = group_by_state(id_test)

    report: Dict[str, Any] = {"n_id_train_rows": len(id_train), "n_id_test_rows": len(id_test), "faithfully_mapped_policies": faithful}

    # ------------------------------------------------------------------
    # ID (in-distribution) evaluation: same models, held-out TEST-split states.
    # ------------------------------------------------------------------
    # ID evaluation lookup fix: the structural-lookup models (KNN/kernel/
    # state-conditioned/residual-transfer) need sibling policies' true
    # rewards at each *query* state. In this standard TRAIN/TEST split the
    # TEST states never appear in id_train at all, so without widening the
    # lookup every one of them would degenerate to a constant-zero
    # prediction (verified: this happened before this fix, producing an
    # identical 0.2037 MAE across every k/weighting/tau combination -- an
    # artifact of empty neighbor lookups, not a real result). lookup_rows
    # spans the full dataset; the target policy's own value at its query
    # state is still never used (distance_index.nearest always excludes it).
    id_models = build_models(id_train, all_policies, distance_index, args.seed, lookup_rows=rows)
    id_prediction_quality = {name: evaluate_model_on_rows(m, id_test) for name, m in id_models.items()}
    report["id_reward_prediction_quality"] = id_prediction_quality

    best_fixed = max(all_policies, key=lambda p: float(np.mean([r["reward_anwg"] for r in id_train if r["policy_name"] == p] or [-1])))
    id_selector_results = {}
    for name, m in id_models.items():
        selections = joint_select(m, id_test_rbs, lam=args.lam)
        ev = evaluate_selection(id_test_rbs, selections, best_fixed_policy=best_fixed)
        ev["margin_weighted_regret"] = margin_weighted_regret(id_test_rbs, selections)
        id_selector_results[name] = ev
    oracle_sel = {sid: oracle_best({r["policy_name"]: r["reward_anwg"] for r in rs})[0] for sid, rs in id_test_rbs.items()}
    id_selector_results["oracle"] = evaluate_selection(id_test_rbs, oracle_sel, best_fixed_policy=best_fixed)
    fixed_sel = {sid: best_fixed for sid in id_test_rbs}
    id_selector_results["fixed_best_train"] = evaluate_selection(id_test_rbs, fixed_sel, best_fixed_policy=best_fixed)
    for reference_policy in ("weighted_shortest_processing", "scorpio_style_slo_guard"):
        ref_sel = {sid: reference_policy for sid in id_test_rbs}
        id_selector_results[f"fixed_{reference_policy}"] = evaluate_selection(id_test_rbs, ref_sel, best_fixed_policy=best_fixed)
    report["id_selector_results"] = id_selector_results
    report["id_best_fixed_policy"] = best_fixed

    ranking = {name: ranking_quality(id_test_rbs, m, all_policies) for name, m in id_models.items()}
    report["id_ranking_quality"] = ranking

    # ------------------------------------------------------------------
    # Held-out-policy (OOD) evaluation across model types, families.
    # ------------------------------------------------------------------
    held_out_candidates = [
        "edf", "weighted_shortest_processing", "adaptive_chunked_prefill",
        "kv_constrained_online", "aging_priority", "fifo",
    ]
    held_out_policy_results: Dict[str, Any] = {}
    extrapolation_diagnostics: Dict[str, Any] = {}
    for policy in held_out_candidates:
        train_rows, test_rows = held_out_policy_split(rows, [policy])
        models = build_models(train_rows, all_policies, distance_index, args.seed)
        # policy-ID model is excluded from held-out reporting: it structurally
        # cannot see an unseen policy's identity column at all.
        model_quality = {name: evaluate_model_on_rows(m, test_rows) for name, m in models.items() if name != "policy_id_rf"}
        held_out_policy_results[policy] = model_quality

        # Extrapolation diagnostics from the k=5 inverse-distance KNN model.
        # nearest_training_policy_distance is constant across every query
        # state for a fixed held-out policy (the nearest OTHER policy by
        # structure doesn't depend on the state), so a within-policy
        # distance-vs-error correlation is undefined by construction --
        # the meaningful correlation is *across* held-out policies, computed
        # once below after this loop.
        knn = models["structural_knn_k5_inverse_distance"]
        per_state = [knn.neighbor_diagnostics(policy, r["state_id"]) for r in test_rows]
        actual = np.asarray([r["reward_anwg"] for r in test_rows])
        preds = knn.predict_mean(test_rows)
        errors = np.abs(preds - actual)
        nearest_dists = np.asarray([d["nearest_training_policy_distance"] for d in per_state], dtype=float)
        mean_k_dists = np.asarray([d["mean_k_neighbor_distance"] for d in per_state], dtype=float)
        extrapolation_diagnostics[policy] = {
            "nearest_training_policy_distance": float(nearest_dists[0]) if len(nearest_dists) else None,
            "mean_k_neighbor_distance": float(np.mean(mean_k_dists)) if len(mean_k_dists) else None,
            "mean_abs_error": float(np.mean(errors)),
        }
    report["held_out_policy_results"] = held_out_policy_results
    report["structural_extrapolation_diagnostics"] = extrapolation_diagnostics

    # Cross-held-out-policy correlation: does structural distance from the
    # nearest training policy predict prediction error? n = number of
    # held-out policies tested (small; report accordingly, no strong claim).
    cross_nearest = [v["nearest_training_policy_distance"] for v in extrapolation_diagnostics.values()]
    cross_mean_k = [v["mean_k_neighbor_distance"] for v in extrapolation_diagnostics.values()]
    cross_errors = [v["mean_abs_error"] for v in extrapolation_diagnostics.values()]
    report["structural_extrapolation_cross_policy_correlation"] = {
        "n_held_out_policies": len(extrapolation_diagnostics),
        "corr_nearest_distance_vs_error": (
            float(np.corrcoef(cross_nearest, cross_errors)[0, 1]) if len(set(cross_nearest)) > 1 else None
        ),
        "corr_mean_k_distance_vs_error": (
            float(np.corrcoef(cross_mean_k, cross_errors)[0, 1]) if len(set(cross_mean_k)) > 1 else None
        ),
        "note": f"n={len(extrapolation_diagnostics)} held-out policies -- too small for a strong claim either way",
    }

    # Held-out family (reuse existing infra + add structural KNN/kernel).
    family_results: Dict[str, Any] = {}
    for family_name in ("slo_deadline_handling", "kv_memory_pressure"):
        family_policies = load_policy_families(family_name)
        base = held_out_family_pilot(rows, family_policies, all_policies=all_policies, family_name=family_name)
        train_rows, test_rows = held_out_policy_split(rows, family_policies)
        knn = StructuralKNNModel(name="knn5_idw", all_policies=all_policies, k=5, weighting="inverse_distance", tau=2.0, distance_index=distance_index).fit(train_rows)
        kernel = KernelSuitabilityModel(name="kernel_tau2", all_policies=all_policies, tau=2.0, distance_index=distance_index).fit(train_rows)
        base["structural_knn_mae"] = evaluate_model_on_rows(knn, test_rows)["mae"]
        base["kernel_mae"] = evaluate_model_on_rows(kernel, test_rows)["mae"]
        family_results[family_name] = base
    report["held_out_family_results"] = family_results

    # ------------------------------------------------------------------
    # Pairwise advantage transfer: Delta_ij(x) for several pairs, direct
    # state-only DeltaModel vs. structural-neighbor-implied delta.
    # ------------------------------------------------------------------
    pairs = [
        ("scorpio_style_slo_guard", "weighted_shortest_processing"),
        ("edf", "fifo"),
        ("aging_priority", "weighted_shortest_processing"),
        ("kv_constrained_online", "adaptive_chunked_prefill"),
    ]
    pairwise_results: Dict[str, Any] = {}
    for a, b in pairs:
        delta_train = build_delta_rows(id_train_rbs, policy_a=a, policy_b=b)
        delta_test = build_delta_rows(id_test_rbs, policy_a=a, policy_b=b)
        if not delta_train or not delta_test:
            pairwise_results[f"{a}_vs_{b}"] = {"status": "insufficient_states"}
            continue
        direct_model = DeltaModel(random_state=args.seed).fit(delta_train)
        direct_eval = evaluate_delta_model(direct_model, delta_test)

        # Structural-neighbor-implied delta: Rhat_a(x) - Rhat_b(x) from the
        # k=5 inverse-distance structural KNN model fit on the same ID split.
        knn_full = id_models["structural_knn_k5_inverse_distance"]
        implied_preds = []
        for row in delta_test:
            state_id = row["state_id"]
            row_a = {"state_id": state_id, "policy_name": a, "state_features": row["state_features"]}
            row_b = {"state_id": state_id, "policy_name": b, "state_features": row["state_features"]}
            mu_a = float(knn_full.predict_mean([row_a])[0])
            mu_b = float(knn_full.predict_mean([row_b])[0])
            implied_preds.append(mu_a - mu_b)
        implied_preds = np.asarray(implied_preds)
        actual_delta = np.asarray([r["delta"] for r in delta_test])
        implied_err = implied_preds - actual_delta
        pairwise_results[f"{a}_vs_{b}"] = {
            "n_test_states": len(delta_test),
            "direct_delta_model": direct_eval,
            "structural_knn_implied_delta": {
                "mae": float(np.mean(np.abs(implied_err))),
                "rmse": float(np.sqrt(np.mean(implied_err ** 2))),
                "sign_accuracy": float(np.mean(np.sign(implied_preds) == np.sign(actual_delta))),
            },
        }
    report["pairwise_advantage_transfer"] = pairwise_results

    # ------------------------------------------------------------------
    # Uncertainty-aware selection for the strongest structural model
    # (chosen by ID mean_regret_to_oracle among structural_knn/kernel/
    # state_conditioned/residual_transfer candidates).
    # ------------------------------------------------------------------
    structural_candidate_names = [n for n in id_selector_results if any(
        n.startswith(p) for p in ("structural_knn", "kernel", "state_conditioned", "residual_transfer")
    )]
    best_structural_name = min(structural_candidate_names, key=lambda n: id_selector_results[n]["overall"]["mean_regret_to_oracle"])
    best_structural_model = id_models[best_structural_name]
    uncertainty_lam_sweep = {}
    for lam in (0.0, 0.5, 1.0, 2.0, 5.0):
        sel = joint_select(best_structural_model, id_test_rbs, lam=lam)
        ev = evaluate_selection(id_test_rbs, sel, best_fixed_policy=best_fixed)
        uncertainty_lam_sweep[str(lam)] = {
            "mean_regret_to_oracle": ev["overall"]["mean_regret_to_oracle"],
            "policy_match_accuracy": ev["overall"]["policy_match_accuracy"],
        }
    report["uncertainty_aware_selection"] = {"best_structural_model": best_structural_name, "lambda_sweep": uncertainty_lam_sweep}

    # ------------------------------------------------------------------
    # Structural-distance-vs-disagreement diagnostic (carried forward).
    # ------------------------------------------------------------------
    report["structural_distance_diagnostics"] = structural_distance_vs_performance_disagreement(group_by_state(rows), faithful)

    report["runtime_s"] = round(time.perf_counter() - t0, 3)
    (out_dir / "structural_suitability_results.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Full results: {out_dir / 'structural_suitability_results.json'}")
    print(f"Runtime: {report['runtime_s']}s")
    print(f"Best structural model (by ID mean regret): {best_structural_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
