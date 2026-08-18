#!/usr/bin/env python3
"""Diagnostics for the mechanism-choice target-redesign feasibility audit.

FEASIBILITY/DIAGNOSTIC ONLY -- this script does not train or evaluate a
selector. It computes, from the frozen dense unified utility matrix
(experiments/unified_utility_matrix_v2/, no policy re-run) and the frozen
SHARED_CORE_V1 feature table (experiments/shared_cross_family_features_v1/,
no replay re-run), the mechanism-contrast target proposed in
docs/audits/mechanism_choice_target_feasibility_v1_20260817.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.policy_separation.mechanism_choice_target_v1 import (  # noqa: E402
    EPS_DEFAULT,
    MECHANISMS,
    compute_mechanism_gains,
)

UNIFIED = REPO_ROOT / "experiments" / "unified_utility_matrix_v2" / "unified_utility_matrix_wide_v2.csv"
SHARED = REPO_ROOT / "experiments" / "shared_cross_family_features_v1" / "shared_core_v1_scenarios.csv"
OUT_DIR = REPO_ROOT / "experiments" / "mechanism_choice_target_feasibility_v1"

EPS = EPS_DEFAULT


def load() -> pd.DataFrame:
    df = pd.read_csv(UNIFIED)
    gain_rows = [compute_mechanism_gains(row) for _, row in df.iterrows()]
    for mech in MECHANISMS:
        df[f"gain_{mech}"] = [g[mech] for g in gain_rows]
    gains = df[[f"gain_{m}" for m in MECHANISMS]]
    df["top_gain"] = gains.max(axis=1)
    df["top_mechanism"] = gains.idxmax(axis=1).str.replace("gain_", "", regex=False)
    second = gains.apply(lambda row: sorted(row.values)[-2], axis=1)
    df["second_gain"] = second
    df["margin"] = df["top_gain"] - df["second_gain"]
    df["target_mechanism_4way"] = np.where(df["top_gain"] <= EPS, "no_clear_mechanism", df["top_mechanism"])
    return df


def confound_check(df: pd.DataFrame) -> dict:
    """Is gain_kv actually explained by KV-capacity pressure, or by
    least_laxity_first's baseline weakness outside its native family?"""
    shared = pd.read_csv(SHARED).set_index("canonical_scenario_id")
    merged = df.set_index("canonical_scenario_id").join(shared[["token_footprint_per_kv"]])
    by_fam = {}
    for fam, g in merged.groupby("mechanism_family"):
        by_fam[fam] = {
            "mean_token_footprint_per_kv": float(g["token_footprint_per_kv"].mean()),
            "mean_gain_kv": float(g["gain_kv"].mean()),
            "mean_anwg_kv_constrained_online": float(g["anwg__kv_constrained_online"].mean()),
            "mean_anwg_least_laxity_first": float(g["anwg__least_laxity_first"].mean()),
        }
    # WITHIN-family Spearman correlation of gain_kv with actual KV-capacity
    # pressure (token_footprint_per_kv). A pooled across-family correlation
    # is itself confounded by coarse between-family clustering (families
    # differ in both mean footprint and mean gain_kv for unrelated reasons)
    # -- the within-family correlation is the clean dose-response test: does
    # gain_kv track ACTUAL pressure where pressure genuinely varies?
    within_family = {}
    for fam, g in merged.groupby("mechanism_family"):
        if g["token_footprint_per_kv"].std() < 1e-9:
            within_family[fam] = {"rho": None, "p_value": None, "note": "no within-family variance in footprint"}
            continue
        rho, p = spearmanr(g["gain_kv"], g["token_footprint_per_kv"])
        within_family[fam] = {"rho": float(rho), "p_value": float(p)}
    return {
        "per_family": by_fam,
        "within_family_gain_kv_vs_kv_pressure_correlation": within_family,
        "interpretation": (
            "If gain_kv on Family A were genuine KV-mechanism relevance, it "
            "should track actual KV-capacity pressure (token_footprint_per_kv) "
            "the way it does on Family C (native: rho=+0.54, p<1e-6, a real "
            "dose-response relationship). On Family A, where footprint is "
            "uniformly far below 1.0 (no real memory pressure) yet gain_kv is "
            "the largest of any family, the within-family correlation is "
            "small and NOT significant (rho=-0.13, p=0.28) -- no dose-response "
            "relationship at all, consistent with gain_kv there reflecting "
            "least_laxity_first's general weakness outside its native family, "
            "not genuine mechanism activation."
        ),
    }


def target_vs_family(df: pd.DataFrame) -> dict:
    ct3 = pd.crosstab(df["mechanism_family"], df["top_mechanism"])
    ct4 = pd.crosstab(df["mechanism_family"], df["target_mechanism_4way"])
    agree3 = float((df["top_mechanism"] == {
        "FAMILY_A_FAIRNESS_STARVATION_V2": "ranking",
        "FAMILY_B_PREFILL_DECODE_V2": "chunk",
        "FAMILY_C_KV_PRESSURE_V2": "kv",
    }.get(df["mechanism_family"].values[0], None)).mean()) if False else None
    native_map = {
        "FAMILY_A_FAIRNESS_STARVATION_V2": "ranking",
        "FAMILY_B_PREFILL_DECODE_V2": "chunk",
        "FAMILY_C_KV_PRESSURE_V2": "kv",
    }
    df["native_mechanism"] = df["mechanism_family"].map(native_map)
    agree_3way = float((df["top_mechanism"] == df["native_mechanism"]).mean())
    agree_4way = float((df["target_mechanism_4way"] == df["native_mechanism"]).mean())
    probs3 = df["top_mechanism"].value_counts(normalize=True)
    entropy3 = float(-(probs3 * np.log2(probs3)).sum())
    probs4 = df["target_mechanism_4way"].value_counts(normalize=True)
    entropy4 = float(-(probs4 * np.log2(probs4)).sum())
    return {
        "confusion_matrix_3way": ct3.to_dict(),
        "confusion_matrix_4way": ct4.to_dict(),
        "agreement_rate_3way_target_eq_native_family_mechanism": agree_3way,
        "agreement_rate_4way_target_eq_native_family_mechanism": agree_4way,
        "class_distribution_3way": probs3.to_dict(),
        "class_distribution_4way": probs4.to_dict(),
        "entropy_3way_bits": entropy3,
        "entropy_4way_bits": entropy4,
        "max_entropy_3way_bits": float(np.log2(3)),
        "max_entropy_4way_bits": float(np.log2(4)),
    }


def cross_family_activation(df: pd.DataFrame) -> dict:
    out = {}
    for mech in MECHANISMS:
        col = f"gain_{mech}"
        row = {}
        for fam, g in df.groupby("mechanism_family"):
            row[fam] = {
                "n": int(len(g)),
                "n_nonzero": int((g[col].abs() > 1e-9).sum()),
                "n_above_eps": int((g[col] > EPS).sum()),
                "mean": float(g[col].mean()),
            }
        out[mech] = row
    return out


def stability(df: pd.DataFrame) -> dict:
    exact_ties = int((df["margin"].abs() < 1e-9).sum())
    near_ties = int((df["margin"] <= EPS).sum())
    robust = int((df["margin"] > EPS).sum())
    return {
        "n_scenarios": len(df),
        "exact_ties_top_vs_second": exact_ties,
        "near_ties_margin_leq_eps": near_ties,
        "robust_margin_gt_eps": robust,
        "fraction_robust": float(robust / len(df)),
        "fraction_no_clear_mechanism_top_gain_leq_eps": float((df["top_gain"] <= EPS).mean()),
        "margin_summary": {
            "mean": float(df["margin"].mean()),
            "median": float(df["margin"].median()),
            "min": float(df["margin"].min()),
            "max": float(df["margin"].max()),
        },
    }


def oracle_information(df: pd.DataFrame) -> dict:
    anwg_cols = [c for c in df.columns if c.startswith("anwg__")]
    oracle = df[anwg_cols].max(axis=1)
    mech_policies = {
        "ranking": ["anwg__weighted_fair_share", "anwg__estimated_service_time_first"],
        "chunk": ["anwg__chunked_prefill_small", "anwg__full_prefill"],
        "kv": ["anwg__kv_constrained_online", "anwg__least_laxity_first"],
    }
    best_within_chosen_mechanism = df.apply(
        lambda row: row[mech_policies[row["top_mechanism"]]].max(), axis=1
    )
    regret_two_stage = oracle - best_within_chosen_mechanism
    best_fixed_global = df[anwg_cols].mean().idxmax()
    regret_best_fixed = oracle - df[best_fixed_global]
    out = {}
    for mech, winner_dist_col in mech_policies.items():
        sub = df[df["top_mechanism"] == mech]
        winner = sub[winner_dist_col].idxmax(axis=1) if len(sub) else None
        out[mech] = {
            "n_scenarios_assigned": int(len(sub)),
            "oracle_mean_anwg": float(oracle[df["top_mechanism"] == mech].mean()) if len(sub) else None,
            "mean_two_stage_regret": float(regret_two_stage[df["top_mechanism"] == mech].mean()) if len(sub) else None,
        }
    return {
        "per_mechanism": out,
        "overall_mean_two_stage_regret": float(regret_two_stage.mean()),
        "overall_mean_regret_vs_single_best_fixed_policy": float(regret_best_fixed.mean()),
        "best_fixed_global_policy": best_fixed_global,
        "note": (
            "two_stage_regret = oracle_anwg - best ANWG among the 2 policies "
            "native to the argmax-chosen mechanism, i.e. the ceiling of a "
            "perfect Stage-1 (mechanism choice) + perfect Stage-2 (within-"
            "mechanism policy choice) pipeline."
        ),
    }


def shared_feature_same_mechanism_overlap(df: pd.DataFrame) -> dict:
    shared = pd.read_csv(SHARED)
    feats = [c for c in shared.columns if c not in ("canonical_scenario_id", "mechanism_family", "source_scenario_id", "replay_verified")]
    merged = shared.merge(df[["canonical_scenario_id", "top_mechanism"]], on="canonical_scenario_id")
    X = merged[feats].to_numpy()
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xz = (X - mu) / sigma
    fam = merged["mechanism_family"].to_numpy()
    mech = merged["top_mechanism"].to_numpy()
    ids = merged["canonical_scenario_id"].to_numpy()

    results = []
    for i in range(len(merged)):
        same_mech_diff_fam = np.where((mech == mech[i]) & (fam != fam[i]))[0]
        if len(same_mech_diff_fam) == 0:
            continue
        dists = np.linalg.norm(Xz[same_mech_diff_fam] - Xz[i], axis=1)
        j = same_mech_diff_fam[int(np.argmin(dists))]
        results.append(
            {
                "scenario": ids[i],
                "mechanism": mech[i],
                "nn_scenario_same_mechanism_diff_family": ids[j],
                "nn_dist": float(dists.min()),
            }
        )
    df_r = pd.DataFrame(results)
    return {
        "n_scenarios_with_cross_family_same_mechanism_partner": len(df_r),
        "mean_nn_dist_same_mechanism_diff_family": float(df_r["nn_dist"].mean()) if len(df_r) else None,
        "by_mechanism_mean_dist": (
            df_r.groupby("mechanism")["nn_dist"].mean().to_dict() if len(df_r) else {}
        ),
        "comparison_note": (
            "compare against the raw (mechanism-agnostic) mean cross-family "
            "NN distance already reported in the SHARED_CORE_V1 audit "
            "(shared_core_v1_diagnostics.json, cross_family_nn_consistency)."
        ),
    }


def main() -> None:
    df = load()
    out = {
        "n_scenarios": len(df),
        "confound_check_gain_kv": confound_check(df),
        "target_vs_family": target_vs_family(df),
        "cross_family_activation": cross_family_activation(df),
        "stability": stability(df),
        "oracle_information": oracle_information(df),
        "shared_feature_same_mechanism_overlap": shared_feature_same_mechanism_overlap(df),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "mechanism_choice_target_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(f"wrote {out_path}")
    print(json.dumps(out["target_vs_family"], indent=2, default=str))
    print(json.dumps(out["confound_check_gain_kv"]["per_family"], indent=2, default=str))
    print(json.dumps(out["stability"], indent=2, default=str))


if __name__ == "__main__":
    main()
