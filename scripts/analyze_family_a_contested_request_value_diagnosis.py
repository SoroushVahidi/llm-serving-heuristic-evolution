#!/usr/bin/env python3
"""Read-only diagnostic analysis of the completed Family-A contested-request
value extraction (`experiments/family_a_contested_request_value_diagnosis/`).

Determines whether long-run ESTF-vs-WFS value is concentrated in the
specific policy-differential ("contested") requests. Uses only the
already-extracted artifacts (`contested_events.csv`, `contested_requests.csv`,
`integrity_check.json`) plus the pre-existing repaired 91-event artifact
(`family_a_observability_continuation_events.csv`) for the whole-branch raw
completed-count `delta_native` join. Runs NO new simulation, touches no
controller/simulator/design code, reads no TEST data, and produces only a
JSON summary consumed by the accompanying markdown report.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTESTED_DIR = REPO_ROOT / "experiments/family_a_contested_request_value_diagnosis"
EXISTING_EVENTS_CSV = (
    REPO_ROOT
    / "experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv"
)
OUTPUT_JSON = CONTESTED_DIR / "contested_request_value_diagnosis_summary.json"

CAUSAL_NUMERIC_COLS = [
    "priority",
    "weight",
    "prompt_tokens",
    "predicted_output_tokens",
    "predicted_service_proxy",
    "queue_age",
    "slo_deadline",
    "deadline_slack_if_admitted_now",
]


def fav_of(scenario_id: pd.Series) -> pd.Series:
    return scenario_id.str.extract(r"\.(?P<fav>favlong|favshort)\.")["fav"]


def describe(s: pd.Series) -> dict:
    s = s.dropna().astype(float)
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
    }


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def main() -> int:
    events = pd.read_csv(CONTESTED_DIR / "contested_events.csv")
    reqs = pd.read_csv(CONTESTED_DIR / "contested_requests.csv")
    integrity = json.loads((CONTESTED_DIR / "integrity_check.json").read_text())
    existing = pd.read_csv(EXISTING_EVENTS_CSV)

    report: dict = {}

    # =========================================================
    # Section 2: independent integrity reconfirmation
    # =========================================================
    events["fav"] = fav_of(events["canonical_scenario_id"])
    reqs["fav"] = fav_of(reqs["canonical_scenario_id"])

    dup_event_ids = int(events["event_id"].duplicated().sum())
    dup_req_identity = int(
        reqs.duplicated(subset=["event_id", "contested_side", "request_id"]).sum()
    )
    null_counts_events = {c: int(events[c].isna().sum()) for c in events.columns}
    identity_cols = ["event_id", "canonical_scenario_id", "split", "step", "contested_side", "request_id"]
    null_counts_req_identity = {c: int(reqs[c].isna().sum()) for c in identity_cols}
    branch_cols_expected = []
    for br in ["br_estf_estf", "br_wfs_wfs", "br_wfs_estf", "br_estf_wfs"]:
        for suffix in ["completed", "completion_time", "slo_violated", "weighted_contribution"]:
            branch_cols_expected.append(f"{br}_{suffix}")
    missing_branch_cols = [c for c in branch_cols_expected if c not in reqs.columns]

    existing_keys = set(zip(existing.canonical_scenario_id, existing.step))
    replayed_keys = set(zip(events.canonical_scenario_id, events.step))

    integrity_recheck = {
        "extraction_integrity_check_json": integrity,
        "n_events": int(len(events)),
        "n_contested_rows": int(len(reqs)),
        "event_key_match_vs_existing_91": existing_keys == replayed_keys,
        "duplicate_event_ids": dup_event_ids,
        "duplicate_request_identity_within_event_side": dup_req_identity,
        "null_counts_event_identity_cols": {
            k: v for k, v in null_counts_events.items() if k in
            ["event_id", "canonical_scenario_id", "split", "step"]
        },
        "null_counts_request_identity_cols": null_counts_req_identity,
        "missing_expected_branch_columns": missing_branch_cols,
        "split_values_events": sorted(events["split"].unique().tolist()),
        "split_values_requests": sorted(reqs["split"].unique().tolist()),
        "test_leakage_events": bool((events["split"].str.lower() == "test").any()),
        "test_leakage_requests": bool((reqs["split"].str.lower() == "test").any()),
    }
    blocked = (
        not integrity_recheck["event_key_match_vs_existing_91"]
        or dup_event_ids > 0
        or missing_branch_cols
        or integrity_recheck["test_leakage_events"]
        or integrity_recheck["test_leakage_requests"]
    )
    integrity_recheck["BLOCKED"] = bool(blocked)
    report["integrity"] = integrity_recheck

    if blocked:
        OUTPUT_JSON.write_text(json.dumps(report, indent=2, default=str))
        print("CONTESTED_VALUE_DIAGNOSIS_BLOCKED_BY_INTEGRITY")
        return 1

    # =========================================================
    # Section 3: contested-set semantics
    # =========================================================
    set_size_dist = {
        "n_estf_only": events["n_estf_only"].value_counts().sort_index().to_dict(),
        "n_wfs_only": events["n_wfs_only"].value_counts().sort_index().to_dict(),
        "n_common": events["n_common"].value_counts().sort_index().to_dict(),
    }
    asymmetric = events[events["n_estf_only"] != events["n_wfs_only"]]
    report["contested_set_semantics"] = {
        "set_size_distribution": {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in set_size_dist.items()},
        "n_estf_only_requests_total": int((reqs["contested_side"] == "estf_only").sum()),
        "n_wfs_only_requests_total": int((reqs["contested_side"] == "wfs_only").sum()),
        "n_common_requests_total": int((reqs["contested_side"] == "common").sum()),
        "n_events_with_asymmetric_set_size": int(len(asymmetric)),
        "one_estf_one_wfs_per_event_fraction": float(
            ((events["n_estf_only"] == 1) & (events["n_wfs_only"] == 1)).mean()
        ),
    }

    # =========================================================
    # Section 4: property characterization by side x regime
    # =========================================================
    prop_report = {}
    for regime_label, sub_events in [
        ("all", events),
        ("favlong", events[events["fav"] == "favlong"]),
        ("favshort", events[events["fav"] == "favshort"]),
    ]:
        sub_reqs = reqs[reqs["event_id"].isin(sub_events["event_id"])]
        estf_rows = sub_reqs[sub_reqs["contested_side"] == "estf_only"]
        wfs_rows = sub_reqs[sub_reqs["contested_side"] == "wfs_only"]
        by_side = {}
        for col in CAUSAL_NUMERIC_COLS:
            by_side[col] = {
                "estf_only": describe(estf_rows[col]),
                "wfs_only": describe(wfs_rows[col]),
                "cohens_d_estf_minus_wfs": cohens_d(estf_rows[col], wfs_rows[col]),
            }
        by_side["feasible_if_admitted_now"] = {
            "estf_only_fraction_true": float(estf_rows["feasible_if_admitted_now"].mean()) if len(estf_rows) else None,
            "wfs_only_fraction_true": float(wfs_rows["feasible_if_admitted_now"].mean()) if len(wfs_rows) else None,
        }
        prop_report[regime_label] = {
            "n_estf_only": int(len(estf_rows)),
            "n_wfs_only": int(len(wfs_rows)),
            "properties": by_side,
        }
    report["property_characterization"] = prop_report

    # =========================================================
    # Section 5: eventual branch outcomes by side
    # =========================================================
    branches = ["br_estf_estf", "br_wfs_wfs", "br_wfs_estf", "br_estf_wfs"]
    outcome_report = {}
    for side in ["estf_only", "wfs_only", "common"]:
        side_rows = reqs[reqs["contested_side"] == side]
        side_out = {}
        for br in branches:
            completed = side_rows[f"{br}_completed"].astype(bool)
            slo_violated = side_rows[f"{br}_slo_violated"]
            wc = side_rows[f"{br}_weighted_contribution"]
            ct = side_rows[f"{br}_completion_time"]
            n = len(side_rows)
            side_out[br] = {
                "n": int(n),
                "completion_probability": float(completed.mean()) if n else None,
                "slo_success_probability_given_completed": (
                    float((~slo_violated[completed].astype(bool)).mean()) if completed.sum() else None
                ),
                "mean_completion_time_given_completed": (
                    float(ct[completed].mean()) if completed.sum() else None
                ),
                "mean_weighted_contribution": float(wc.mean()) if n else None,
                "unfinished_probability": float((~completed).mean()) if n else None,
            }
        outcome_report[side] = side_out

    # "does the competing branch eventually serve the request anyway"
    estf_only_rows = reqs[reqs["contested_side"] == "estf_only"]
    wfs_only_rows = reqs[reqs["contested_side"] == "wfs_only"]
    outcome_report["cross_branch_rescue"] = {
        "estf_only_completed_under_br_wfs_wfs": float(estf_only_rows["br_wfs_wfs_completed"].astype(bool).mean()),
        "wfs_only_completed_under_br_estf_estf": float(wfs_only_rows["br_estf_estf_completed"].astype(bool).mean()),
        "note": (
            "br_wfs_wfs = WFS admits+continues natively (never admitted this ESTF-only "
            "request at t0); br_estf_estf = ESTF admits+continues natively (never admitted "
            "this WFS-only request at t0). Nonzero rate means the request re-enters and is "
            "served later in the branch that did not admit it first."
        ),
    }
    report["eventual_outcomes"] = outcome_report

    # =========================================================
    # Section 6: value concentration
    # =========================================================
    existing_small = existing[["canonical_scenario_id", "step", "delta_native"]].rename(
        columns={"delta_native": "delta_native_whole_branch_raw"}
    )
    events_j = events.merge(existing_small, on=["canonical_scenario_id", "step"], how="left")
    assert events_j["delta_native_whole_branch_raw"].isna().sum() == 0

    def contested_raw_delta(event_id: str) -> float:
        rows = reqs[reqs["event_id"] == event_id]
        e = rows[rows["contested_side"] == "estf_only"]
        w = rows[rows["contested_side"] == "wfs_only"]
        val = (
            (e["br_estf_estf_completed"].astype(bool).astype(int) - e["br_wfs_wfs_completed"].astype(bool).astype(int)).sum()
            + (w["br_wfs_wfs_completed"].astype(bool).astype(int) - w["br_estf_estf_completed"].astype(bool).astype(int)).sum()
        )
        return float(val)

    def contested_weighted_delta(event_id: str) -> float:
        rows = reqs[reqs["event_id"] == event_id]
        e = rows[rows["contested_side"] == "estf_only"]
        w = rows[rows["contested_side"] == "wfs_only"]
        val = (
            (e["br_estf_estf_weighted_contribution"] - e["br_wfs_wfs_weighted_contribution"]).sum()
            + (w["br_wfs_wfs_weighted_contribution"] - w["br_estf_estf_weighted_contribution"]).sum()
        )
        return float(val)

    events_j["contested_raw_delta"] = events_j["event_id"].map(contested_raw_delta)
    events_j["contested_weighted_delta"] = events_j["event_id"].map(contested_weighted_delta)

    nonzero_denom = events_j[events_j["delta_native_whole_branch_raw"] != 0].copy()
    zero_denom = events_j[events_j["delta_native_whole_branch_raw"] == 0].copy()
    nonzero_denom["explained_fraction"] = (
        nonzero_denom["contested_raw_delta"] / nonzero_denom["delta_native_whole_branch_raw"]
    )

    total_abs_weighted = events_j["contested_weighted_delta"].abs().sum()
    top_event_share = (
        float(events_j["contested_weighted_delta"].abs().max() / total_abs_weighted)
        if total_abs_weighted > 0 else None
    )
    req_abs_diffs = []
    for br_pair, side in [(("br_estf_estf", "br_wfs_wfs"), "estf_only"), (("br_wfs_wfs", "br_estf_estf"), "wfs_only")]:
        rows = reqs[reqs["contested_side"] == side]
        d = (rows[f"{br_pair[0]}_weighted_contribution"] - rows[f"{br_pair[1]}_weighted_contribution"]).abs()
        req_abs_diffs.extend(d.tolist())
    req_abs_diffs = np.array(req_abs_diffs)
    total_abs_req = req_abs_diffs.sum()
    top_request_share = float(req_abs_diffs.max() / total_abs_req) if total_abs_req > 0 else None

    report["value_concentration"] = {
        "definition": (
            "contested_raw_delta(event) = sum_{i in estf_only}[1(completed under br_estf_estf) - "
            "1(completed under br_wfs_wfs)] + sum_{i in wfs_only}[1(completed under br_wfs_wfs) - "
            "1(completed under br_estf_estf)], in the SAME raw-completed-count units as "
            "delta_native_whole_branch_raw = br_estf_estf_completed - br_wfs_wfs_completed "
            "(whole-branch total, joined from the pre-existing 91-event artifact). "
            "explained_fraction = contested_raw_delta / delta_native_whole_branch_raw, defined "
            "only where the denominator is nonzero (60/91 events per the joined data); the "
            "32 zero-denominator events are reported separately, not silently dropped. "
            "contested_weighted_delta is the SLO/priority-weighted analogue restricted to the "
            "contested set only (no whole-branch weighted denominator exists in this artifact, "
            "so it is reported descriptively, not as a fraction)."
        ),
        "n_events_nonzero_denominator": int(len(nonzero_denom)),
        "n_events_zero_denominator": int(len(zero_denom)),
        "zero_denominator_events_contested_raw_delta_nonzero_count": int((zero_denom["contested_raw_delta"] != 0).sum()),
        "explained_fraction_stats": describe(nonzero_denom["explained_fraction"]),
        "explained_fraction_p90": float(nonzero_denom["explained_fraction"].quantile(0.90)),
        "fraction_events_explained_fraction_ge_0.5_abs": float((nonzero_denom["explained_fraction"].abs() >= 0.5).mean()),
        "fraction_events_explained_fraction_ge_1.0_abs": float((nonzero_denom["explained_fraction"].abs() >= 1.0).mean()),
        "contested_weighted_delta_stats": describe(events_j["contested_weighted_delta"]),
        "top_event_share_of_total_abs_weighted_delta": top_event_share,
        "top_request_share_of_total_abs_weighted_diff": top_request_share,
    }

    # =========================================================
    # Section 7: WFS-protection hypothesis (favlong)
    #
    # NOTE ON DEFINITION CHOICE: the continuation-only isolation
    # (br_wfs_wfs vs br_wfs_estf -- same first action, different
    # continuation) is degenerate: it is IDENTICALLY ZERO for every one
    # of the 60 favlong wfs_only rows (verified: completion/SLO/weighted
    # fields are bit-identical between br_wfs_wfs and br_wfs_estf for
    # every row). This is a real structural finding, not a bug: Family-A
    # GPUs have max_active_sequences=1 and the simulator has no
    # preemption, so once a specific request is admitted onto its slot,
    # ITS OWN eventual fate is already sealed -- which policy continues
    # scheduling *other* requests afterward cannot change it. Reported
    # separately below as `continuation_only_effect_is_degenerate`. The
    # real, non-degenerate protection signal is therefore in WHETHER /
    # HOW SOON the request is admitted at all -- isolated by comparing
    # its own native branch (br_wfs_wfs: WFS admits it immediately and
    # WFS continues) against the OTHER policy's native branch
    # (br_estf_estf: ESTF never admits it at t0; does it get admitted
    # and completed later anyway, and how well). This is
    # `admission_native_effect` below.
    # =========================================================
    wfs_favlong = wfs_only_rows[wfs_only_rows["fav"] == "favlong"].copy()
    continuation_only_effect_wfs = (
        wfs_favlong["br_wfs_wfs_weighted_contribution"] - wfs_favlong["br_wfs_estf_weighted_contribution"]
    )
    wfs_favlong["admission_native_effect"] = (
        wfs_favlong["br_wfs_wfs_weighted_contribution"] - wfs_favlong["br_estf_estf_weighted_contribution"]
    )
    corr_vars = ["priority", "queue_age", "deadline_slack_if_admitted_now", "predicted_service_proxy"]
    protection_corrs = {}
    for v in corr_vars:
        if wfs_favlong[v].nunique() > 1 and wfs_favlong["admission_native_effect"].nunique() > 1:
            rho, p = spearmanr(wfs_favlong[v], wfs_favlong["admission_native_effect"])
        else:
            rho, p = float("nan"), float("nan")
        protection_corrs[v] = {"spearman_rho": float(rho), "p": float(p)}

    report["wfs_protection_test"] = {
        "n_wfs_only_favlong": int(len(wfs_favlong)),
        "continuation_only_effect_is_degenerate": bool((continuation_only_effect_wfs == 0).all()),
        "continuation_only_effect_stats": describe(continuation_only_effect_wfs),
        "admission_native_effect_stats": describe(wfs_favlong["admission_native_effect"]),
        "fraction_admission_native_effect_positive": float((wfs_favlong["admission_native_effect"] > 0).mean()) if len(wfs_favlong) else None,
        "fraction_admission_native_effect_negative": float((wfs_favlong["admission_native_effect"] < 0).mean()) if len(wfs_favlong) else None,
        "completion_prob_own_native_br_wfs_wfs": float(wfs_favlong["br_wfs_wfs_completed"].astype(bool).mean()) if len(wfs_favlong) else None,
        "completion_prob_other_native_br_estf_estf": float(wfs_favlong["br_estf_estf_completed"].astype(bool).mean()) if len(wfs_favlong) else None,
        "slo_success_rate_given_completed_own_native_br_wfs_wfs": (
            float((~wfs_favlong.loc[wfs_favlong["br_wfs_wfs_completed"].astype(bool), "br_wfs_wfs_slo_violated"].astype(bool)).mean())
            if wfs_favlong["br_wfs_wfs_completed"].astype(bool).sum() else None
        ),
        "slo_success_rate_given_completed_other_native_br_estf_estf": (
            float((~wfs_favlong.loc[wfs_favlong["br_estf_estf_completed"].astype(bool), "br_estf_estf_slo_violated"].astype(bool)).mean())
            if wfs_favlong["br_estf_estf_completed"].astype(bool).sum() else None
        ),
        "correlation_admission_native_effect_vs_causal_features": protection_corrs,
    }

    # =========================================================
    # Section 8: ESTF-useful hypothesis (favshort)
    # Same definition-choice note as Section 7: continuation-only effect
    # (br_estf_estf vs br_estf_wfs) is verified degenerate (identically
    # zero) for every one of the 31 favshort estf_only rows. Real signal
    # isolated via admission_native_effect = br_estf_estf - br_wfs_wfs
    # (own native branch vs the other policy's native branch, i.e.
    # never admitted at t0 by WFS).
    # =========================================================
    estf_favshort = estf_only_rows[estf_only_rows["fav"] == "favshort"].copy()
    continuation_only_effect_estf = (
        estf_favshort["br_estf_estf_weighted_contribution"] - estf_favshort["br_estf_wfs_weighted_contribution"]
    )
    estf_favshort["admission_native_effect"] = (
        estf_favshort["br_estf_estf_weighted_contribution"] - estf_favshort["br_wfs_wfs_weighted_contribution"]
    )
    use_corrs = {}
    for v in corr_vars + ["prompt_tokens", "predicted_output_tokens"]:
        if estf_favshort[v].nunique() > 1 and estf_favshort["admission_native_effect"].nunique() > 1:
            rho, p = spearmanr(estf_favshort[v], estf_favshort["admission_native_effect"])
        else:
            rho, p = float("nan"), float("nan")
        use_corrs[v] = {"spearman_rho": float(rho), "p": float(p)}

    report["estf_useful_test"] = {
        "n_estf_only_favshort": int(len(estf_favshort)),
        "continuation_only_effect_is_degenerate": bool((continuation_only_effect_estf == 0).all()),
        "continuation_only_effect_stats": describe(continuation_only_effect_estf),
        "admission_native_effect_stats": describe(estf_favshort["admission_native_effect"]),
        "fraction_admission_native_effect_positive": float((estf_favshort["admission_native_effect"] > 0).mean()) if len(estf_favshort) else None,
        "completion_prob_own_native_br_estf_estf": float(estf_favshort["br_estf_estf_completed"].astype(bool).mean()) if len(estf_favshort) else None,
        "completion_prob_other_native_br_wfs_wfs": float(estf_favshort["br_wfs_wfs_completed"].astype(bool).mean()) if len(estf_favshort) else None,
        "feasible_fraction": float(estf_favshort["feasible_if_admitted_now"].mean()) if len(estf_favshort) else None,
        "correlation_admission_native_effect_vs_causal_features": use_corrs,
    }

    # =========================================================
    # Section 9: diagnostic proxies (event-level scores)
    # =========================================================
    eps = 1e-6

    def side_scores(event_id: str):
        rows = reqs[reqs["event_id"] == event_id]
        e = rows[rows["contested_side"] == "estf_only"]
        w = rows[rows["contested_side"] == "wfs_only"]
        out = {}
        out["C_estf"] = float((e["weight"] * e["feasible_if_admitted_now"].astype(int)).sum())
        out["C_wfs"] = float((w["weight"] * w["feasible_if_admitted_now"].astype(int)).sum())
        out["D_estf"] = float((e["weight"] / e["predicted_service_proxy"].clip(lower=eps)).sum())
        out["D_wfs"] = float((w["weight"] / w["predicted_service_proxy"].clip(lower=eps)).sum())
        out["E_estf"] = float((e["weight"] * e["queue_age"]).sum())
        out["E_wfs"] = float((w["weight"] * w["queue_age"]).sum())
        out["priority_only_estf"] = float(e["weight"].sum())
        out["priority_only_wfs"] = float(w["weight"].sum())
        out["age_only_estf"] = float(e["queue_age"].sum())
        out["age_only_wfs"] = float(w["queue_age"].sum())
        return out

    score_rows = events_j["event_id"].map(side_scores).apply(pd.Series)
    events_j = pd.concat([events_j.reset_index(drop=True), score_rows.reset_index(drop=True)], axis=1)

    def pref_label(margin: pd.Series) -> pd.Series:
        return np.select([margin > 0, margin < 0], ["ESTF", "WFS"], default="TIE")

    events_j["gt_delta_native"] = events_j["delta_native_whole_branch_raw"]
    events_j["gt_label"] = pref_label(events_j["gt_delta_native"])

    events_j["margin_A_completion_only"] = events_j["delta_native_whole_branch_raw"]  # by construction, same sign
    events_j["margin_C_priority_feasibility"] = events_j["C_estf"] - events_j["C_wfs"]
    events_j["margin_D_value_per_service"] = events_j["D_estf"] - events_j["D_wfs"]
    events_j["margin_E_age_protection"] = events_j["E_estf"] - events_j["E_wfs"]
    events_j["margin_always_wfs"] = -1.0
    events_j["margin_always_estf"] = 1.0
    events_j["margin_regime_equivalent"] = np.where(events_j["fav"] == "favlong", -1.0, 1.0)
    events_j["margin_priority_only"] = events_j["priority_only_estf"] - events_j["priority_only_wfs"]
    events_j["margin_age_only"] = events_j["age_only_estf"] - events_j["age_only_wfs"]

    proxy_margin_cols = {
        "A_OLD_COMPLETION_ONLY": "margin_A_completion_only",
        "C_CONTESTED_PRIORITY_FEASIBILITY": "margin_C_priority_feasibility",
        "D_CONTESTED_VALUE_PER_REMAINING_SERVICE": "margin_D_value_per_service",
        "E_CONTESTED_AGE_PROTECTION_PROXY_FOR_FAIRNESS": "margin_E_age_protection",
    }
    baseline_margin_cols = {
        "always_WFS": "margin_always_wfs",
        "always_ESTF": "margin_always_estf",
        "regime_label_equivalent": "margin_regime_equivalent",
        "priority_only": "margin_priority_only",
        "fairness_age_only": "margin_age_only",
    }

    def eval_margin(sub: pd.DataFrame, margin_col: str) -> dict:
        gt = sub["gt_label"]
        pred = pref_label(sub[margin_col])
        out = {
            "n": int(len(sub)),
            "sign_agreement_incl_ties": float((pred == gt).mean()),
            "estf_preference_share": float((pred == "ESTF").mean()),
            "wfs_preference_share": float((pred == "WFS").mean()),
        }
        try:
            out["balanced_accuracy_3class"] = float(balanced_accuracy_score(gt, pred))
        except Exception:
            out["balanced_accuracy_3class"] = None
        try:
            out["macro_f1_3class"] = float(f1_score(gt, pred, average="macro"))
        except Exception:
            out["macro_f1_3class"] = None
        nz = sub[sub["gt_label"] != "TIE"]
        if nz["gt_label"].nunique() == 2 and nz[margin_col].nunique() > 1:
            y = (nz["gt_label"] == "ESTF").astype(int)
            try:
                out["roc_auc_nonzero_gt"] = float(roc_auc_score(y, nz[margin_col]))
            except Exception:
                out["roc_auc_nonzero_gt"] = None
        else:
            out["roc_auc_nonzero_gt"] = None
        if sub[margin_col].nunique() > 1 and sub["gt_delta_native"].nunique() > 1:
            rho, p = spearmanr(sub[margin_col], sub["gt_delta_native"])
            out["spearman_margin_vs_delta_native"] = {"rho": float(rho), "p": float(p)}
        else:
            out["spearman_margin_vs_delta_native"] = None
        return out

    alignment = {}
    collapse = {}
    for name, col in {**proxy_margin_cols, **baseline_margin_cols}.items():
        target = alignment if name in proxy_margin_cols else collapse
        target[name] = {}
        for regime_label, sub in [
            ("ALL", events_j),
            ("favlong", events_j[events_j["fav"] == "favlong"]),
            ("favshort", events_j[events_j["fav"] == "favshort"]),
        ]:
            target[name][regime_label] = eval_margin(sub, col)

    report["proxy_alignment"] = alignment
    report["collapse_triviality_check"] = collapse
    report["proxy_B_note"] = (
        "FAILED_AGGREGATE_PROGRESS (V_inflight over ALL in-flight requests at branch "
        "terminal state) is NOT recomputable from this contested-request-only artifact: "
        "the extraction records outcomes only for the specific contested request IDs, not "
        "the full terminal ObservableState (waiting_queue + all active_requests_info) needed "
        "to sum progress_fraction*feasible over every in-flight request. Its already-measured "
        "closed-form result is cited from "
        "docs/current/family_a_terminal_value_v1_analysis_20260820.md SS5: new-preference "
        "ESTF share in favlong INCREASED at every horizon (+4.8pp/+6.1pp/+5.8pp at H=1/5/20 "
        "ALL; +6.1pp/+8.1pp/+7.8pp favlong), the opposite of the intended direction -- "
        "TERMINAL_VALUE_OFFLINE_NO_GO."
    )

    # =========================================================
    # Section 14: regime-level vs request-level signal
    # (majority-baseline within each stratum for reference)
    # =========================================================
    def majority_baseline_bal_acc(sub: pd.DataFrame) -> float:
        gt = sub["gt_label"]
        majority = gt.value_counts().idxmax()
        pred = pd.Series([majority] * len(sub), index=sub.index)
        return float(balanced_accuracy_score(gt, pred))

    regime_signal = {}
    for regime_label, sub in [("favlong", events_j[events_j["fav"] == "favlong"]), ("favshort", events_j[events_j["fav"] == "favshort"])]:
        regime_signal[regime_label] = {
            "majority_baseline_balanced_accuracy": majority_baseline_bal_acc(sub),
            "n": int(len(sub)),
            "gt_label_counts": sub["gt_label"].value_counts().to_dict(),
        }
    report["regime_vs_request_level"] = {
        "majority_baseline_within_stratum": regime_signal,
        "argument": (
            "Since favlong/favshort strata each hold regime constant, any balanced accuracy "
            "of a contested-causal-feature proxy WITHIN a stratum that exceeds that stratum's "
            "own majority-class baseline cannot be explained by regime identity alone -- it "
            "requires request-level information. Compare proxy_alignment[*][favlong|favshort]"
            "[balanced_accuracy_3class] against regime_vs_request_level.majority_baseline_"
            "within_stratum[favlong|favshort].majority_baseline_balanced_accuracy."
        ),
    }

    # =========================================================
    # Causal availability audit (Section 12)
    # =========================================================
    report["causal_availability_audit"] = {
        "priority / weight": "ONLINE_CAUSAL",
        "prompt_tokens": "ONLINE_CAUSAL",
        "predicted_output_tokens": "ONLINE_CAUSAL",
        "predicted_service_proxy": "ONLINE_CAUSAL (derived from ONLINE_CAUSAL fields via scoring.py::predicted_service_proxy)",
        "queue_age": "ONLINE_CAUSAL",
        "deadline_slack_if_admitted_now / feasible_if_admitted_now": "ONLINE_CAUSAL (scoring.py::deadline_slack at decision time)",
        "class_id": "ONLINE_CAUSAL",
        "contested_side (estf_only/wfs_only label)": (
            "ONLINE_CAUSAL in principle -- requires evaluating both ESTF and WFS on the "
            "identical pre-decision snapshot, which is exactly what a real online shadow-policy "
            "comparison could do (no future information), but is NOT free online (2x scoring "
            "cost per eligible decision); it is a real online signal, not a metadata leak"
        ),
        "br_*_completed / br_*_completion_time / br_*_slo_violated / br_*_weighted_contribution": "FUTURE_OUTCOME (ground-truth/label only, not usable in a deployable proxy)",
        "canonical_scenario_id / split / step / event_id": "EXPERIMENT_METADATA (never usable as a deployable feature)",
        "favlong / favshort (parsed from canonical_scenario_id)": "EXPERIMENT_METADATA (analysis stratum only, per task instruction; forbidden as a runtime input)",
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    events_j.to_csv(CONTESTED_DIR / "contested_events_with_diagnosis_scores.csv", index=False)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {CONTESTED_DIR / 'contested_events_with_diagnosis_scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
