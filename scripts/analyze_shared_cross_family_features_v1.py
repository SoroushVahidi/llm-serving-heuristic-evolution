#!/usr/bin/env python3
"""Lightweight diagnostics for SHARED_CORE_V1 (family-identity leakage,
distribution overlap, cross-family nearest-neighbor utility consistency).

FEATURE-SCHEMA INVESTIGATION ONLY. Per the task scope: no selector suite is
trained/evaluated here (that is explicitly out of scope -- see the audit
doc). This script only runs (a) a simple, unoptimized diagnostic classifier
for family-identifiability, (b) per-feature distribution-overlap summaries,
and (c) a nearest-neighbor cross-family utility-consistency check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "experiments" / "shared_cross_family_features_v1"
MF_PSD_DIR = REPO_ROOT / "experiments" / "mf_psd_v1"
UNIFIED_DIR = REPO_ROOT / "experiments" / "unified_utility_matrix_v2"

with open(SHARED_DIR / "shared_core_v1_schema.json") as f:
    SCHEMA = json.load(f)
FEATURES = SCHEMA["learnable_feature_allowlist"]


def load_shared() -> pd.DataFrame:
    df = pd.read_csv(SHARED_DIR / "shared_core_v1_scenarios.csv")
    assert df[FEATURES].isnull().sum().sum() == 0
    return df


def load_group_keys() -> pd.Series:
    long_df = pd.read_csv(MF_PSD_DIR / "mf_psd_long_v1.csv")
    g = long_df.drop_duplicates("canonical_scenario_id").set_index("canonical_scenario_id")["group_key"]
    return g


def family_leakage_diagnostic(df: pd.DataFrame) -> dict:
    groups = load_group_keys().reindex(df["canonical_scenario_id"]).values
    X = df[list(FEATURES)].to_numpy()
    y = df["mechanism_family"].to_numpy()

    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    all_true, all_pred = [], []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=6)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append((pred == y[test_idx]).mean())
        all_true.extend(y[test_idx])
        all_pred.extend(pred)

    labels = sorted(set(y))
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    majority_label = pd.Series(y).value_counts().idxmax()
    majority_acc = (y == majority_label).mean()

    return {
        "n_folds": n_splits,
        "fold_accuracies": accs,
        "mean_accuracy": float(np.mean(accs)),
        "majority_baseline_accuracy": float(majority_acc),
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def distribution_overlap(df: pd.DataFrame) -> dict:
    out = {}
    for feat in FEATURES:
        row = {}
        by_fam = {fam: g[feat].to_numpy() for fam, g in df.groupby("mechanism_family")}
        for fam, vals in by_fam.items():
            row[fam] = {
                "min": float(np.min(vals)),
                "q25": float(np.percentile(vals, 25)),
                "median": float(np.median(vals)),
                "q75": float(np.percentile(vals, 75)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        # pairwise standardized mean difference (Cohen's d, pooled std) +
        # range-overlap fraction
        fams = sorted(by_fam.keys())
        pairwise = {}
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                a, b = by_fam[fams[i]], by_fam[fams[j]]
                pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2.0)
                d = float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else float("nan")
                lo = max(np.min(a), np.min(b))
                hi = min(np.max(a), np.max(b))
                union_lo = min(np.min(a), np.min(b))
                union_hi = max(np.max(a), np.max(b))
                overlap_frac = max(0.0, (hi - lo)) / (union_hi - union_lo) if union_hi > union_lo else 1.0
                pairwise[f"{fams[i]}_vs_{fams[j]}"] = {"cohens_d": d, "range_overlap_fraction": float(overlap_frac)}
        row["pairwise"] = pairwise
        out[feat] = row
    return out


def cross_family_nn_consistency(df: pd.DataFrame) -> dict:
    """For each scenario, find its nearest neighbor (standardized Euclidean
    distance in SHARED_CORE_V1 space) among scenarios of a DIFFERENT family,
    then compare policy-preference structure (Spearman corr of the 6-policy
    ANWG vector, and top-1 agreement) using the frozen dense unified utility
    matrix (all 6 policies x 176 scenarios, already built -- no policy is
    re-run here)."""
    wide = pd.read_csv(UNIFIED_DIR / "unified_utility_matrix_wide_v2.csv")
    anwg_cols = [c for c in wide.columns if c.startswith("anwg__")]
    wide = wide.set_index("canonical_scenario_id")

    X = df[list(FEATURES)].to_numpy()
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xz = (X - mu) / sigma
    fam = df["mechanism_family"].to_numpy()
    ids = df["canonical_scenario_id"].to_numpy()

    from scipy.stats import spearmanr

    results = []
    for i in range(len(df)):
        dists = np.linalg.norm(Xz - Xz[i], axis=1)
        dists[fam == fam[i]] = np.inf
        j = int(np.argmin(dists))
        u_i = wide.loc[ids[i], anwg_cols].to_numpy(dtype=float)
        u_j = wide.loc[ids[j], anwg_cols].to_numpy(dtype=float)
        rho, _ = spearmanr(u_i, u_j)
        top1_i = anwg_cols[int(np.argmax(u_i))]
        top1_j = anwg_cols[int(np.argmax(u_j))]
        results.append(
            {
                "scenario": ids[i],
                "family": fam[i],
                "nn_scenario": ids[j],
                "nn_family": fam[j],
                "nn_feature_dist": float(dists[j]),
                "utility_vector_spearman": float(rho) if rho == rho else None,
                "top1_agree": bool(top1_i == top1_j),
            }
        )

    df_r = pd.DataFrame(results)
    valid_rho = df_r["utility_vector_spearman"].dropna()

    # Random-pair baseline: pair each scenario with a random scenario from a
    # different family (fixed seed for reproducibility), same comparison.
    rng = np.random.default_rng(0)
    rand_agree = []
    rand_rho = []
    for i in range(len(df)):
        cross_idx = np.where(fam != fam[i])[0]
        j = rng.choice(cross_idx)
        u_i = wide.loc[ids[i], anwg_cols].to_numpy(dtype=float)
        u_j = wide.loc[ids[j], anwg_cols].to_numpy(dtype=float)
        rho, _ = spearmanr(u_i, u_j)
        if rho == rho:
            rand_rho.append(rho)
        rand_agree.append(int(np.argmax(u_i)) == int(np.argmax(u_j)))

    return {
        "n_scenarios": len(df_r),
        "nearest_neighbor": {
            "mean_utility_vector_spearman": float(valid_rho.mean()),
            "median_utility_vector_spearman": float(valid_rho.median()),
            "top1_agreement_rate": float(df_r["top1_agree"].mean()),
        },
        "random_cross_family_pair_baseline": {
            "mean_utility_vector_spearman": float(np.mean(rand_rho)),
            "top1_agreement_rate": float(np.mean(rand_agree)),
        },
        "per_scenario_sample": df_r.head(10).to_dict(orient="records"),
    }


def feature_regret_correlation(df: pd.DataFrame) -> dict:
    """Per-feature Spearman correlation with each scenario's realized regret
    of the best-fixed-per-family-training policy (a simple, non-model
    diagnostic -- not a trained selector)."""
    wide = pd.read_csv(UNIFIED_DIR / "unified_utility_matrix_wide_v2.csv")
    anwg_cols = [c for c in wide.columns if c.startswith("anwg__")]
    wide = wide.set_index("canonical_scenario_id")
    oracle = wide[anwg_cols].max(axis=1)

    from scipy.stats import spearmanr

    out = {}
    merged = df.set_index("canonical_scenario_id").join(oracle.rename("oracle_anwg"))
    for feat in FEATURES:
        rho, p = spearmanr(merged[feat], merged["oracle_anwg"])
        out[feat] = {"spearman_vs_oracle_anwg": float(rho), "p_value": float(p)}
    return out


def main() -> None:
    df = load_shared()
    out = {
        "family_leakage_diagnostic": family_leakage_diagnostic(df),
        "distribution_overlap": distribution_overlap(df),
        "cross_family_nn_consistency": cross_family_nn_consistency(df),
        "feature_vs_oracle_correlation": feature_regret_correlation(df),
    }
    out_path = SHARED_DIR / "shared_core_v1_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {out_path}")
    print(json.dumps(out["family_leakage_diagnostic"], indent=2))
    print(json.dumps(out["cross_family_nn_consistency"]["nearest_neighbor"], indent=2))
    print(json.dumps(out["cross_family_nn_consistency"]["random_cross_family_pair_baseline"], indent=2))


if __name__ == "__main__":
    main()
