#!/usr/bin/env python3
"""Read-only, diagnostic-only offline feasibility study for a TRAINING-FREE
analytic request-index that trades completion value against SLO/deadline
risk, over the already-frozen 91 Family-A ESTF/WFS contested events.

No new simulation is run. No controller/policy/simulator code is touched.
No TEST data is read. No coefficient sweep is performed (every candidate is
either coefficient-free or uses exactly one constant already used elsewhere
in this codebase, per task instruction). Nothing is staged/committed/pushed.

Input (pre-existing, not modified):
    experiments/family_a_contested_request_value_diagnosis/constrained_formulation_event_table.csv
    (91 events; per-side estf_/wfs_ causal features, prior labels, prior
    predictions, and the frozen constrained-formulation rule's prediction,
    all already computed by the two prior diagnostic reports.)

Output (created/updated by this script only):
    experiments/family_a_analytic_index_feasibility/analytic_index_event_table.csv
    experiments/family_a_analytic_index_feasibility/analytic_index_feasibility_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    REPO_ROOT
    / "experiments/family_a_contested_request_value_diagnosis/constrained_formulation_event_table.csv"
)
OUTPUT_DIR = REPO_ROOT / "experiments/family_a_analytic_index_feasibility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9  # identical convention to scoring.py::urgency_score's own eps-clamp


def urgency(laxity: pd.Series) -> pd.Series:
    """1 / max(laxity, eps) -- same clamping convention as the existing,
    already-shipped scoring.py::urgency_score (reused, not invented)."""
    return 1.0 / laxity.clip(lower=EPS)


def decide(i_estf: pd.Series, i_wfs: pd.Series) -> pd.Series:
    """Deterministic tie handling: ESTF if strictly greater, WFS if strictly
    greater, else TIE resolved to WFS (matching the codebase's existing
    fallback-to-WFS convention used by the receding-horizon oracle and
    stateful-controller candidate-region fallback)."""
    out = np.where(i_estf > i_wfs, "ESTF", np.where(i_wfs > i_estf, "WFS", "WFS"))
    return pd.Series(out, index=i_estf.index)


def balanced_accuracy_present_labels(
    y_true: pd.Series, y_pred: pd.Series, labels: list[str]
) -> float:
    """Balanced accuracy over labels present in y_true.

    This matches sklearn's balanced_accuracy_score convention while avoiding
    warnings when a diagnostic rule predicts a class absent from one stratum's
    ground truth (common here because favlong has no WFS native-raw labels).
    """
    present = [label for label in labels if (y_true == label).any()]
    if len(present) <= 1:
        return float("nan")
    recalls = []
    for label in present:
        mask = y_true == label
        recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls))


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    assert len(df) == 91, f"expected 91 events, got {len(df)}"
    assert set(df["split"].unique()) <= {"train", "val"}, "TEST rows present"

    fav = df["fav"]

    # ---- per-side causal features (already ONLINE_CAUSAL per prior audits) ----
    p_e, p_w = df["estf_priority"], df["wfs_priority"]
    s_e, s_w = df["estf_predicted_service_proxy"], df["wfs_predicted_service_proxy"]
    a_e, a_w = df["estf_queue_age"], df["wfs_queue_age"]
    l_e, l_w = df["estf_laxity_own"], df["wfs_laxity_own"]
    u_e, u_w = urgency(l_e), urgency(l_w)
    deficit_event = df["max_class_deficit_ratio"]  # event-level aggregate, identical both sides

    results = {}

    # ---- Candidate A: WEIGHTED_SHORTEST_SERVICE (cmu rule, no deficit) ----
    iA_e, iA_w = p_e / s_e.clip(lower=EPS), p_w / s_w.clip(lower=EPS)
    df["pred_idx_A"] = decide(iA_e, iA_w)

    # ---- Candidate B: DEADLINE_URGENCY_INDEX (priority-weighted urgency,
    # unit-consistent laxity only, no service term) ----
    iB_e, iB_w = p_e * u_e, p_w * u_w
    df["pred_idx_B"] = decide(iB_e, iB_w)

    # ---- Candidate C: GENERALIZED_CMU_STYLE (priority x urgency / service) ----
    iC_e, iC_w = p_e * u_e / s_e.clip(lower=EPS), p_w * u_w / s_w.clip(lower=EPS)
    df["pred_idx_C"] = decide(iC_e, iC_w)

    # ---- Candidate D: FAIRNESS_DEBT_ADJUSTED_INDEX (completion-efficiency x
    # (1 + per-request queue_age), age used as the documented substitute for a
    # true per-request class-deficit value -- see report SS9/limitations) ----
    iD_e, iD_w = iA_e * (1.0 + a_e), iA_w * (1.0 + a_w)
    df["pred_idx_D"] = decide(iD_e, iD_w)

    # ---- Candidate D (aggregate-deficit) falsification check: multiplying
    # both sides of the SAME event by the SAME event-level aggregate deficit
    # ratio must be a mathematical no-op for pairwise ranking. Verified, not
    # assumed. ----
    iD_agg_e, iD_agg_w = iA_e * (1.0 + deficit_event), iA_w * (1.0 + deficit_event)
    df["pred_idx_D_aggregate_deficit"] = decide(iD_agg_e, iD_agg_w)
    d_agg_flips = int((df["pred_idx_D_aggregate_deficit"] != df["pred_idx_A"]).sum())

    # ---- Candidate E: WHITTLE_INSPIRED_DEADLINE_INDEX (priority-weighted
    # inverse normalized-laxity: priority x service / laxity -- a "least
    # laxity per unit remaining processing" criticality ratio, the zeroth-
    # order form appearing in deadline-restless-bandit index heuristics; NOT
    # a derived Whittle index -- see report SS14) ----
    iE_e, iE_w = p_e * s_e / l_e.clip(lower=EPS), p_w * s_w / l_w.clip(lower=EPS)
    df["pred_idx_E"] = decide(iE_e, iE_w)

    # ---- additional simple/trivial comparator rules (task SS12) ----
    df["pred_priority_only_rel"] = decide(p_e, p_w)
    df["pred_service_only_rel"] = decide(-s_e, -s_w)  # shorter service wins -> negate for decide()
    df["pred_age_only_rel"] = decide(a_e, a_w)
    long_request_to_wfs_matches_always_wfs = int((s_w > s_e).sum())  # of 91

    candidates = ["A", "B", "C", "D", "E"]
    comparator_cols = {
        "always_estf": "pred_always_estf",
        "always_wfs": "pred_always_wfs",
        "priority_ge5": "pred_priority_ge5",
        "regime_equiv": "pred_regime_equiv",
        "priority_only_rel": "pred_priority_only_rel",
        "service_only_rel": "pred_service_only_rel",
        "age_only_rel": "pred_age_only_rel",
        "prior_proxy_E_age_protection": "pred_E",
        "prior_proxy_A_completion_only": "pred_A",
        "constrained_rule (prior feasibility study)": "constrained_rule_pred",
    }

    def stratum_metrics(sub: pd.DataFrame, pred_col: str) -> dict:
        pred = sub[pred_col]
        gt = sub["gt_label"]  # biased raw-count native sign; ESTF/WFS/TIE
        crule = sub["constrained_rule_pred"]  # ESTF/WFS only

        cb = sub["completion_benefit_label"]
        slo = sub["slo_risk_label"]

        n = len(sub)
        estf_share = float((pred == "ESTF").mean()) if n else float("nan")

        cb_recall = (
            float((pred[cb == 1] == "ESTF").mean()) if (cb == 1).any() else float("nan")
        )
        slo_recall = (
            float((pred[slo == 1] == "WFS").mean()) if (slo == 1).any() else float("nan")
        )

        false_estf_rate = (
            float((slo[pred == "ESTF"] == 1).mean()) if (pred == "ESTF").any() else float("nan")
        )
        false_wfs_rate = (
            float((cb[pred == "WFS"] == 1).mean()) if (pred == "WFS").any() else float("nan")
        )

        bal_acc_gt = balanced_accuracy_present_labels(gt, pred, ["ESTF", "WFS", "TIE"])
        try:
            macro_f1_gt = f1_score(gt, pred, labels=["ESTF", "WFS", "TIE"], average="macro", zero_division=0)
        except Exception:
            macro_f1_gt = float("nan")
        agree_crule = float((pred == crule).mean())
        bal_acc_crule = (
            balanced_accuracy_present_labels(crule, pred, ["ESTF", "WFS"])
            if pred.nunique() > 1
            else float("nan")
        )

        # quadrant behavior (task SS10)
        quad = {}
        for name, mask in {
            "BOTH_RISKS": (cb == 1) & (slo == 1),
            "NEITHER_RISK": (cb == 0) & (slo == 0),
            "COMPLETION_ONLY": (cb == 1) & (slo == 0),
            "SLO_RISK_ONLY": (cb == 0) & (slo == 1),
        }.items():
            sub_n = int(mask.sum())
            quad[name] = {
                "n": sub_n,
                "estf_share": float((pred[mask] == "ESTF").mean()) if sub_n else float("nan"),
            }

        return {
            "n": n,
            "estf_share": estf_share,
            "completion_benefit_recall": cb_recall,
            "slo_risk_protection_recall": slo_recall,
            "false_estf_rate": false_estf_rate,
            "false_wfs_rate": false_wfs_rate,
            "balanced_acc_vs_gt_label_3class": bal_acc_gt,
            "macro_f1_vs_gt_label_3class": macro_f1_gt,
            "agreement_vs_constrained_rule": agree_crule,
            "balanced_acc_vs_constrained_rule": bal_acc_crule,
            "quadrants": quad,
        }

    strata = {"ALL": df, "favlong": df[fav == "favlong"], "favshort": df[fav == "favshort"]}

    for cand in candidates:
        col = f"pred_idx_{cand}"
        results[cand] = {stratum: stratum_metrics(sub, col) for stratum, sub in strata.items()}

    comparator_results = {}
    for label, col in comparator_cols.items():
        comparator_results[label] = {
            stratum: stratum_metrics(sub, col) for stratum, sub in strata.items()
        }

    # ---- triviality / regime-reconstruction overlap (task SS13) ----
    overlap = {}
    for cand in candidates:
        col = f"pred_idx_{cand}"
        overlap[cand] = {
            "overlap_vs_priority_ge5": float((df[col] == df["pred_priority_ge5"]).mean()),
            "overlap_vs_regime_equiv": float((df[col] == df["pred_regime_equiv"]).mean()),
            "overlap_vs_always_wfs": float((df[col] == df["pred_always_wfs"]).mean()),
        }

    # ---- virtual-debt diagnostic (task SS15): does the ONLY available
    # existing debt signal (event-level aggregate max_class_deficit_ratio)
    # change any pairwise decision relative to the pure completion-efficiency
    # index (A)? ----
    virtual_debt_diagnostic = {
        "n_events": len(df),
        "flips_vs_candidate_A": d_agg_flips,
        "interpretation": (
            "0 flips expected and found: multiplying both sides of the same "
            "event by the same event-level scalar cannot change an argmax "
            "comparison. The only existing fairness-debt signal "
            "(max_class_deficit_ratio) is an event-level aggregate (max over "
            "all classes in queue), identical for the ESTF-only and WFS-only "
            "side of a given event, so it is provably uninformative for this "
            "per-event PAIRWISE decision -- not merely empirically weak."
        ),
    }

    triviality_notes = {
        "long_request_to_wfs_equals_always_wfs_count": long_request_to_wfs_matches_always_wfs,
        "long_request_to_wfs_equals_always_wfs_total": int(len(df)),
        "priority_ge5_equals_regime_equiv_exact": bool(
            (df["pred_priority_ge5"] == df["pred_regime_equiv"]).all()
        ),
    }

    summary = {
        "n_events": int(len(df)),
        "candidates": results,
        "comparators": comparator_results,
        "overlap_triviality_check": overlap,
        "virtual_debt_diagnostic": virtual_debt_diagnostic,
        "triviality_notes": triviality_notes,
    }

    with open(OUTPUT_DIR / "analytic_index_feasibility_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    df.to_csv(OUTPUT_DIR / "analytic_index_event_table.csv", index=False)

    print(json.dumps(summary["overlap_triviality_check"], indent=2))
    print(json.dumps(summary["virtual_debt_diagnostic"], indent=2))
    print(json.dumps(summary["triviality_notes"], indent=2))
    for cand in candidates:
        print(cand, "ALL:", json.dumps({k: v for k, v in summary["candidates"][cand]["ALL"].items() if k != "quadrants"}))


if __name__ == "__main__":
    main()
