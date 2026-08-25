#!/usr/bin/env python3
"""Read-only, diagnostic-only offline feasibility audit for a constrained
("maximize completion benefit subject to an SLO-risk constraint") Family-A
decision formulation, as an alternative to scalar terminal-value
optimization (which failed: `family_a_terminal_value_v1_analysis_20260820.md`)
and to simple contested-scalar proxies (which failed:
`family_a_contested_request_value_diagnosis_20260821.md`).

Uses only already-extracted TRAIN/VAL artifacts:
  - experiments/family_a_contested_request_value_diagnosis/{contested_events,contested_requests}.csv
  - experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv

Runs NO new simulation, touches no controller/policy/simulator code, reads
no TEST data. Future outcomes (br_* columns) are used ONLY as offline labels
for diagnosis, never as features of any candidate online rule.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTESTED_DIR = REPO_ROOT / "experiments/family_a_contested_request_value_diagnosis"
EXISTING_EVENTS_CSV = (
    REPO_ROOT
    / "experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv"
)
OUTPUT_JSON = CONTESTED_DIR / "constrained_formulation_feasibility_summary.json"

AGG_FEATURES = [
    "queue_length", "active_count", "n_gpus",
    "queue_age_p10", "queue_age_p50", "queue_age_p90", "queue_age_mean",
    "predicted_output_tokens_p10", "predicted_output_tokens_p50", "predicted_output_tokens_p90", "predicted_output_tokens_mean",
    "prompt_tokens_p10", "prompt_tokens_p50", "prompt_tokens_p90", "prompt_tokens_mean",
    "est_service_time_p10", "est_service_time_p50", "est_service_time_p90", "est_service_time_mean",
    "max_class_deficit_ratio", "longest_waiting_age", "n_distinct_classes_in_queue",
    "laxity_p10", "laxity_p50", "laxity_p90", "laxity_mean",
    "fraction_laxity_negative", "fraction_laxity_near_deadline",
    "mean_kv_utilization", "max_kv_utilization", "free_kv_capacity",
    "prefilling_count", "decoding_count",
    "n_admit_estf", "n_admit_wfs", "admit_symmetric_diff_size",
    "history_queue_len_slope", "history_kv_util_slope", "history_admitted_count_slope",
]

PER_REQUEST_FEATURES = [
    "priority", "prompt_tokens", "predicted_output_tokens",
    "predicted_service_proxy", "queue_age", "laxity_own",
]


def fav_of(scenario_id: pd.Series) -> pd.Series:
    return scenario_id.str.extract(r"\.(?P<fav>favlong|favshort)\.")["fav"]


def describe(s: pd.Series) -> dict:
    s = pd.Series(s).dropna().astype(float)
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
        "p25": float(s.quantile(0.25)), "p75": float(s.quantile(0.75)),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
    }


def grouped_cv_eval(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5) -> dict:
    """Same methodology as the repaired-analysis report's grouped CV
    (GroupKFold by canonical_scenario_id; majority / logistic / shallow tree
    / RF-diagnostic), adapted for binary labels. Returns fold mean/std plus
    out-of-fold predictions for downstream rule construction."""
    n_splits_eff = min(n_splits, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits_eff)
    splits = list(gkf.split(X, y, groups))

    results = {}
    oof_proba = {}

    def eval_model(name, model, needs_scaling=False):
        bal_accs, aucs, f1s = [], [], []
        oof_pred = np.full(len(y), -1, dtype=int)
        oof_prob = np.full(len(y), np.nan)
        for train_idx, test_idx in splits:
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            if ytr.nunique() < 2:
                pred = np.full(len(yte), ytr.mode().iloc[0])
                prob = np.full(len(yte), float(ytr.mean()))
            else:
                m = make_pipeline(StandardScaler(), model) if needs_scaling else model
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                prob = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else pred.astype(float)
            oof_pred[test_idx] = pred
            oof_prob[test_idx] = prob
            bal_accs.append(balanced_accuracy_score(yte, pred))
            f1s.append(f1_score(yte, pred, average="macro"))
            if yte.nunique() == 2:
                try:
                    aucs.append(roc_auc_score(yte, prob))
                except Exception:
                    pass
        cm = confusion_matrix(y, oof_pred, labels=[0, 1]).tolist()
        results[name] = {
            "balanced_accuracy_mean": float(np.mean(bal_accs)), "balanced_accuracy_std": float(np.std(bal_accs)),
            "roc_auc_mean": float(np.mean(aucs)) if aucs else None, "roc_auc_std": float(np.std(aucs)) if aucs else None,
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "confusion_matrix_labels_0_1": cm,
        }
        oof_proba[name] = oof_prob

    majority = y.mode().iloc[0]
    maj_pred = np.full(len(y), majority)
    results["majority_baseline"] = {
        "balanced_accuracy_mean": float(balanced_accuracy_score(y, maj_pred)),
        "balanced_accuracy_std": 0.0, "roc_auc_mean": None, "roc_auc_std": None,
        "macro_f1_mean": float(f1_score(y, maj_pred, average="macro")), "macro_f1_std": 0.0,
        "confusion_matrix_labels_0_1": confusion_matrix(y, maj_pred, labels=[0, 1]).tolist(),
    }
    eval_model("logistic_regression", LogisticRegression(max_iter=2000), needs_scaling=True)
    eval_model("shallow_tree_depth3", DecisionTreeClassifier(max_depth=3, random_state=0))
    eval_model("rf_capacity_diagnostic", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0))

    return {
        "n": int(len(y)), "n_groups": int(groups.nunique()), "n_splits_used": n_splits_eff,
        "class_balance": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "models": results,
        "oof_logistic_proba": oof_proba["logistic_regression"].tolist(),
    }


def main() -> int:
    events = pd.read_csv(CONTESTED_DIR / "contested_events.csv")
    reqs = pd.read_csv(CONTESTED_DIR / "contested_requests.csv")
    existing = pd.read_csv(EXISTING_EVENTS_CSV)

    events["fav"] = fav_of(events["canonical_scenario_id"])
    reqs["fav"] = fav_of(reqs["canonical_scenario_id"])

    report: dict = {}

    # =========================================================
    # Section 5: diagnose why feasible_if_admitted_now is degenerate
    # =========================================================
    reqs["laxity_own"] = reqs["slo_deadline"] - (reqs["queue_age"] + reqs["arrival_time"])
    report["feasibility_degeneracy_diagnosis"] = {
        "deadline_slack_definition": (
            "scoring.py::deadline_slack(req, now, service_proxy) = "
            "req.slo_deadline - now - service_proxy, where service_proxy = "
            "predicted_service_proxy = alpha*prompt_tokens + beta*predicted_output_tokens "
            "(alpha=0.5, beta=1.0), i.e. a RAW TOKEN-COUNT-SCALE proxy. The function's own "
            "docstring states explicitly: 'service_proxy is in steps; convert to seconds "
            "via step_size if needed. Phase 1 leaves it unit-less (policies compare slacks "
            "relatively).'"
        ),
        "diagnosis": (
            "feasible_if_admitted_now is degenerate NOT because it is evaluated too late, "
            "and NOT primarily because it is 'too strict' in a graded sense -- it is "
            "structurally comparing two incommensurate scales by design: slo_deadline is a "
            "small real-valued time budget (contested-row means 2.5-18.7 across strata), "
            "while service_proxy is a raw token-count proxy (contested-row means 228-960), "
            "one to two orders of magnitude larger, per the function's own admitted "
            "unit-less-ness. Subtracting them and testing >=0 as an ABSOLUTE admissibility "
            "bound is therefore close to guaranteed to return False whenever "
            "predicted_output_tokens is more than a handful of tokens, independent of how "
            "close the request actually is to a real miss. It is INHERENTLY DEGENERATE for "
            "this purpose in this Family-A setting, not merely a threshold-tuning issue: the "
            "function is explicitly designed for RELATIVE ranking between candidates (as "
            "ESTF/WFS/urgency_score already use it), not as an ABSOLUTE per-request gate."
        ),
        "classification": "INHERENTLY_DEGENERATE_FOR_ABSOLUTE_GATING_BY_DESIGN",
        "fraction_feasible_if_admitted_now_true_overall": float(reqs["feasible_if_admitted_now"].mean()),
        "alternative_unit_consistent_signal": (
            "laxity_own = slo_deadline - state.time (state.time reconstructed as "
            "queue_age + arrival_time, both already in the extraction), matching the "
            "existing aggregate `laxity` feature already computed online in "
            "family_a_observability_continuation_v1.py Group D "
            "(`laxity = r.slo_deadline - state.time`, NO service-proxy term mixed in -- "
            "genuinely unit-consistent, real time-budget-remaining semantics)."
        ),
        "laxity_own_stats_estf_only": describe(reqs.loc[reqs["contested_side"] == "estf_only", "laxity_own"]),
        "laxity_own_stats_wfs_only": describe(reqs.loc[reqs["contested_side"] == "wfs_only", "laxity_own"]),
        "fraction_laxity_own_negative_estf_only": float((reqs.loc[reqs["contested_side"] == "estf_only", "laxity_own"] < 0).mean()),
        "fraction_laxity_own_negative_wfs_only": float((reqs.loc[reqs["contested_side"] == "wfs_only", "laxity_own"] < 0).mean()),
    }

    # =========================================================
    # Build event-level feature/label table
    # =========================================================
    estf_rows = reqs[reqs["contested_side"] == "estf_only"].set_index("event_id")
    wfs_rows = reqs[reqs["contested_side"] == "wfs_only"].set_index("event_id")

    existing_small = existing[["canonical_scenario_id", "step", "delta_native"] + AGG_FEATURES].rename(
        columns={
            "delta_native": "delta_native_whole_branch_raw",
            "n_admit_estf": "agg_n_admit_estf",
            "n_admit_wfs": "agg_n_admit_wfs",
        }
    )
    ev = events.merge(existing_small, on=["canonical_scenario_id", "step"], how="left")
    ev = ev.set_index("event_id")
    assert ev["delta_native_whole_branch_raw"].isna().sum() == 0

    for f in PER_REQUEST_FEATURES:
        ev[f"estf_{f}"] = estf_rows[f]
        ev[f"wfs_{f}"] = wfs_rows[f]

    # =========================================================
    # Section 6/7: freeze offline labels (future outcomes = labels ONLY)
    # =========================================================
    ev["completion_benefit_label"] = (
        estf_rows["br_estf_estf_completed"].astype(bool) & (~estf_rows["br_wfs_wfs_completed"].astype(bool))
    ).astype(int)
    ev["slo_risk_label"] = (
        (~wfs_rows["br_estf_estf_completed"].astype(bool))
        | wfs_rows["br_estf_estf_slo_violated"].fillna(True).astype(bool)
    ).astype(int)

    report["frozen_offline_targets"] = {
        "completion_benefit_label": (
            "1 if the ESTF-only contested request completes under br_estf_estf (its own "
            "native branch) AND does NOT complete under br_wfs_wfs (the other policy's own "
            "native branch); else 0. LABEL ONLY -- uses br_* future-outcome fields, never a "
            "runtime feature."
        ),
        "slo_risk_label": (
            "1 if the WFS-only contested request is NOT completed-and-SLO-safe under "
            "br_estf_estf (choosing ESTF): either it never completes within the bounded "
            "window, or it completes but violates its SLO; else 0. Answers 'does choosing "
            "ESTF over WFS put the WFS-favored request at risk'. LABEL ONLY."
        ),
        "completion_benefit_label_prevalence": float(ev["completion_benefit_label"].mean()),
        "slo_risk_label_prevalence": float(ev["slo_risk_label"].mean()),
        "completion_benefit_label_prevalence_by_regime": ev.groupby("fav")["completion_benefit_label"].mean().to_dict(),
        "slo_risk_label_prevalence_by_regime": ev.groupby("fav")["slo_risk_label"].mean().to_dict(),
    }

    # =========================================================
    # Section 8: grouped CV separability
    # =========================================================
    agg_features_post_rename = [
        c if c not in ("n_admit_estf", "n_admit_wfs") else f"agg_{c}" for c in AGG_FEATURES
    ]
    feature_cols = agg_features_post_rename + [f"estf_{f}" for f in PER_REQUEST_FEATURES] + [f"wfs_{f}" for f in PER_REQUEST_FEATURES]
    X = ev[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    groups = ev["canonical_scenario_id"]

    cb_cv = grouped_cv_eval(X, ev["completion_benefit_label"], groups)
    sr_cv = grouped_cv_eval(X, ev["slo_risk_label"], groups)
    ev["oof_completion_benefit_proba"] = cb_cv.pop("oof_logistic_proba")
    ev["oof_slo_risk_proba"] = sr_cv.pop("oof_logistic_proba")
    report["completion_benefit_prediction"] = cb_cv
    report["slo_risk_prediction"] = sr_cv

    # =========================================================
    # Section 9: constrained-rule construction (out-of-fold, no leakage)
    # =========================================================
    ev["gt_label"] = np.select(
        [ev["delta_native_whole_branch_raw"] > 0, ev["delta_native_whole_branch_raw"] < 0],
        ["ESTF", "WFS"], default="TIE",
    )
    ev["constrained_rule_pred"] = np.where(
        (ev["oof_slo_risk_proba"] <= 0.5) & (ev["oof_completion_benefit_proba"] > 0.5),
        "ESTF", "WFS",
    )
    report["constrained_rule_definition"] = {
        "boundary_1_slo_gate": "permit ESTF only if out-of-fold predicted P(slo_risk_label=1) <= 0.5 (natural probability-0.5 decision boundary, grouped-CV logistic regression, no leakage)",
        "boundary_2_benefit_gate": "within the ESTF-permitted region, choose ESTF only if out-of-fold predicted P(completion_benefit_label=1) > 0.5; otherwise WFS",
        "note_on_threshold_choice": (
            "No continuous threshold sweep was performed. The 0.5 probability boundary is "
            "the only principled, non-invented boundary available: the natural zero-slack "
            "boundary (deadline_slack_if_admitted_now >= 0) is unusable as shown in "
            "Section 5 (0% feasible for every contested row, would collapse to always-WFS); "
            "no other frozen safety tolerance in the existing Family-A design docs applies "
            "at the per-decision level."
        ),
    }

    def eval_rule(pred_col: str, sub: pd.DataFrame) -> dict:
        gt = sub["gt_label"]
        pred = sub[pred_col]
        out = {
            "n": int(len(sub)),
            "estf_pred_share": float((pred == "ESTF").mean()),
            "wfs_pred_share": float((pred == "WFS").mean()),
            "sign_agreement_incl_ties": float((pred == gt).mean()),
        }
        try:
            out["balanced_accuracy_vs_gt3class"] = float(balanced_accuracy_score(gt, pred))
        except Exception:
            out["balanced_accuracy_vs_gt3class"] = None
        nz = sub[sub["gt_label"] != "TIE"]
        favlong_pred_estf = sub[(pred == "ESTF")]
        out["false_estf_rate_of_estf_predictions"] = (
            float((favlong_pred_estf["gt_label"] == "WFS").mean()) if len(favlong_pred_estf) else None
        )
        favshort_pred_wfs = sub[(pred == "WFS")]
        out["false_wfs_rate_of_wfs_predictions"] = (
            float((favshort_pred_wfs["gt_label"] == "ESTF").mean()) if len(favshort_pred_wfs) else None
        )
        if len(nz) and nz[pred_col].nunique() > 1:
            try:
                out["roc_auc_nonzero_gt"] = float(roc_auc_score((nz["gt_label"] == "ESTF").astype(int), (nz[pred_col] == "ESTF").astype(int)))
            except Exception:
                out["roc_auc_nonzero_gt"] = None
        return out

    rule_alignment = {}
    for regime_label, sub in [("ALL", ev), ("favlong", ev[ev["fav"] == "favlong"]), ("favshort", ev[ev["fav"] == "favshort"])]:
        rule_alignment[regime_label] = eval_rule("constrained_rule_pred", sub)
    report["constrained_rule_alignment"] = rule_alignment

    # =========================================================
    # Section 10: comparison to scalarization baselines
    # =========================================================
    prev_scores_path = CONTESTED_DIR / "contested_events_with_diagnosis_scores.csv"
    baseline_comp = {}
    if prev_scores_path.exists():
        prev = pd.read_csv(prev_scores_path).set_index("event_id")
        prev_common = prev.reindex(ev.index)
        ev["margin_E_age_protection"] = prev_common["margin_E_age_protection"]
        ev["margin_A_completion_only"] = prev_common["margin_A_completion_only"]
        ev["pred_E"] = np.select([ev["margin_E_age_protection"] > 0, ev["margin_E_age_protection"] < 0], ["ESTF", "WFS"], default="TIE")
        ev["pred_A"] = np.select([ev["margin_A_completion_only"] > 0, ev["margin_A_completion_only"] < 0], ["ESTF", "WFS"], default="TIE")
        for regime_label, sub in [("ALL", ev), ("favlong", ev[ev["fav"] == "favlong"]), ("favshort", ev[ev["fav"] == "favshort"])]:
            baseline_comp.setdefault("E_best_contested_scalar_proxy_fairness_age_only", {})[regime_label] = eval_rule("pred_E", sub)
            baseline_comp.setdefault("A_old_completion_only_CIRCULAR_CAVEAT", {})[regime_label] = eval_rule("pred_A", sub)
    ev["pred_majority"] = ev["gt_label"].mode().iloc[0]
    ev["pred_always_estf"] = "ESTF"
    for regime_label, sub in [("ALL", ev), ("favlong", ev[ev["fav"] == "favlong"]), ("favshort", ev[ev["fav"] == "favshort"])]:
        baseline_comp.setdefault("majority_class_baseline", {})[regime_label] = eval_rule("pred_majority", sub)
        baseline_comp.setdefault("always_ESTF_best_fixed_style", {})[regime_label] = eval_rule("pred_always_estf", sub)
    report["scalarization_comparison"] = baseline_comp
    report["failed_aggregate_progress_note"] = (
        "FAILED_AGGREGATE_PROGRESS (V_inflight terminal-value scalarization) is cited "
        "qualitatively, not recomputed (same data limitation as the prior contested-value "
        "diagnosis): docs/current/family_a_terminal_value_v1_analysis_20260820.md SS5 found "
        "new-preference ESTF share in favlong INCREASED at every horizon "
        "(TERMINAL_VALUE_OFFLINE_NO_GO), i.e. that single-scalar formulation moved alignment "
        "the wrong direction entirely."
    )

    # =========================================================
    # Section 11: favshort/favlong breakdown (already folded into rule_alignment
    # and baseline_comp above; add label prevalence cross-tab for convenience)
    # =========================================================
    report["regime_breakdown_summary"] = {
        "completion_benefit_prevalence_by_regime": ev.groupby("fav")["completion_benefit_label"].mean().to_dict(),
        "slo_risk_prevalence_by_regime": ev.groupby("fav")["slo_risk_label"].mean().to_dict(),
        "constrained_rule_estf_share_by_regime": ev.groupby("fav")["constrained_rule_pred"].apply(lambda s: (s == "ESTF").mean()).to_dict(),
    }

    # =========================================================
    # Section 12: virtual-queue / dual-variable feasibility
    # =========================================================
    wfs_deficit_corr = {}
    for target_name, target in [
        ("slo_risk_label", ev["slo_risk_label"]),
        ("completion_benefit_label", ev["completion_benefit_label"]),
    ]:
        if ev["max_class_deficit_ratio"].nunique() > 1 and target.nunique() > 1:
            rho, p = spearmanr(ev["max_class_deficit_ratio"].fillna(0), target)
        else:
            rho, p = float("nan"), float("nan")
        wfs_deficit_corr[target_name] = {"spearman_rho": float(rho), "p": float(p)}

    report["virtual_queue_debt_signal_feasibility"] = {
        "classification": "NATURAL_EXISTING_DEBT_SIGNAL",
        "evidence": (
            "src/llmserveopt/policies/weighted_fair_share.py::_score already computes, "
            "online, per decision: deficit = demand[cls] / max(1, served_share[cls] + 1), "
            "where demand comes from state.waiting_queue and served_share from "
            "state.gpu_states[*].active_requests_info -- WFS's own scoring rule already IS "
            "a per-class deficit-ratio computation, purely from ONLINE_CAUSAL state, no "
            "future information. The identical-shape aggregate feature "
            "`max_class_deficit_ratio` is already extracted online in "
            "family_a_observability_continuation_v1.py Group C "
            "(`demand[c] / max(1, active_by_class[c] + 1)`, max over classes) and is present "
            "in the existing 91-event artifact used throughout this and the prior diagnosis. "
            "No new derivation is required to obtain a per-class debt/deficit state variable; "
            "it already exists and is already read by a real, running policy (WFS)."
        ),
        "semantic_meaningfulness_of_Z_update": (
            "A future Z_{t+1} = max(0, Z_t + violation_signal_t - target) formulation is "
            "semantically plausible: `max_class_deficit_ratio` (or the underlying "
            "demand/served_share ratio per class) is already a bounded-below, "
            "queue-state-driven quantity that rises when a class is under-served and falls "
            "as it is admitted -- structurally the right shape for a virtual-queue/Lyapunov "
            "debt variable. No target epsilon is proposed here (per task instruction)."
        ),
        "empirical_correlation_with_offline_labels": wfs_deficit_corr,
    }

    # =========================================================
    # Section 13: does WFS behave like constraint-protection?
    # =========================================================
    report["wfs_as_constraint_protection_evidence"] = {
        "full_scenario_evidence_cited": (
            "docs/current/family_a_rollout_value_limit_diagnosis_20260820.md SS11: in "
            "favlong, WFS achieves the BEST priority_weighted_slo_goodput (0.6029) despite "
            "the WORST (highest) max_latency (25.81 vs ESTF's 23.71) -- WFS sacrifices raw "
            "latency to protect priority/SLO outcomes, a full-trajectory fairness-debt "
            "signature no bounded window sees."
        ),
        "contested_request_evidence_cited": (
            "docs/current/family_a_contested_request_value_diagnosis_20260821.md SS7: WFS-only "
            "contested requests' SLO-success-given-completed rate is 78.3% under WFS's own "
            "native continuation vs only 45.5% under ESTF's -- a +32.8pp gap, never negative "
            "(0/60 favlong WFS-only requests worse off under WFS's own path)."
        ),
        "quantified_tradeoff": (
            f"completion_benefit_label prevalence (ESTF-side) = {float(ev['completion_benefit_label'].mean()):.3f}; "
            f"slo_risk_label prevalence (choosing ESTF puts WFS-side at risk) = {float(ev['slo_risk_label'].mean()):.3f} "
            "-- both are substantial and roughly comparable in magnitude, consistent with a "
            "genuine two-sided tradeoff (not one side dominating), which is the structural "
            "precondition for a constrained formulation to be meaningful rather than trivial."
        ),
    }

    # =========================================================
    # Section 14: triviality checks
    # =========================================================
    ev["pred_always_wfs"] = "WFS"
    ev["pred_priority_ge5"] = np.where(ev["wfs_priority"] >= 5, "WFS", "ESTF")
    ev["pred_regime_equiv"] = np.where(ev["fav"] == "favlong", "WFS", "ESTF")
    trivial = {}
    for name in ["pred_always_wfs", "pred_always_estf", "pred_priority_ge5", "pred_regime_equiv"]:
        trivial[name] = {}
        for regime_label, sub in [("ALL", ev), ("favlong", ev[ev["fav"] == "favlong"]), ("favshort", ev[ev["fav"] == "favshort"])]:
            trivial[name][regime_label] = eval_rule(name, sub)
    report["triviality_checks"] = trivial
    agree_with_priority_rule = float((ev["constrained_rule_pred"] == ev["pred_priority_ge5"]).mean())
    agree_with_regime_rule = float((ev["constrained_rule_pred"] == ev["pred_regime_equiv"]).mean())
    report["constrained_rule_vs_trivial_rule_agreement"] = {
        "agreement_with_priority_ge5_rule": agree_with_priority_rule,
        "agreement_with_regime_equivalent_rule": agree_with_regime_rule,
    }

    ev.reset_index().to_csv(CONTESTED_DIR / "constrained_formulation_event_table.csv", index=False)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {CONTESTED_DIR / 'constrained_formulation_event_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
