#!/usr/bin/env python3
"""Rigorous scientific analysis of the Policy Separation Sobol Pilot v1 results."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    run_dir = Path("experiments/policy_separation_sobol_pilot_20260816T183600Z_1182183")
    if not run_dir.exists():
        print(f"Error: run directory {run_dir} does not exist.")
        return

    print("======================================================================")
    # 1. LOAD DATA
    print("Loading data...")
    features_df = pd.read_csv(run_dir / "scenario_features.csv")
    results_df = pd.read_csv(run_dir / "per_policy_results.csv")
    print(f"Loaded {len(features_df)} scenario features and {len(results_df)} per-policy evaluations.")

    # 2. STRUCTURAL VALIDATION
    print("----------------------------------------------------------------------")
    print("Performing Structural Validation...")
    
    # 2A. Scenario and evaluation counts
    n_scenarios = len(features_df)
    n_tasks = len(results_df)
    print(f"Total scenarios: {n_scenarios} (Expected: 1616)")
    print(f"Total tasks: {n_tasks} (Expected: 6976)")
    
    # Check scenario IDs match exactly
    feat_ids = set(features_df["scenario_id"])
    res_ids = set(results_df["scenario_id"])
    print(f"Scenario IDs in features and results match exactly: {feat_ids == res_ids}")
    
    # Family breakdown
    family_counts_feat = features_df["generator_family"].value_counts().to_dict()
    family_counts_res = results_df["generator_family"].value_counts().to_dict()
    print("Scenario counts per family (features):", family_counts_feat)
    print("Evaluation counts per family (results):", family_counts_res)
    
    # Expected:
    # Family B: 1024 scenarios, 1024 * 4 = 4096 evals
    # Family C: 512 scenarios, 512 * 5 = 2560 evals
    # FCFS add-on: 80 scenarios, 80 * 4 = 320 evals
    # Let's verify these counts:
    print(f"Family B scenario count valid: {family_counts_feat.get('sobol_family_b_prediction_sensitive') == 1024}")
    print(f"Family B evaluation count valid: {family_counts_res.get('sobol_family_b_prediction_sensitive') == 4096}")
    print(f"Family C scenario count valid: {family_counts_feat.get('sobol_family_c_deadline_admission') == 512}")
    print(f"Family C evaluation count valid: {family_counts_res.get('sobol_family_c_deadline_admission') == 2560}")
    print(f"FCFS add-on scenario count valid: {family_counts_feat.get('fcfs_categorical_add_on') == 80}")
    print(f"FCFS add-on evaluation count valid: {family_counts_res.get('fcfs_categorical_add_on') == 320}")

    # 2B. Missing or duplicate keys
    duplicates = results_df.duplicated(subset=["scenario_id", "policy_name"]).sum()
    print(f"Duplicate (scenario_id, policy_name) keys: {duplicates} (Expected: 0)")
    
    # Check for NaN / Inf
    nan_anwg = results_df["arrival_normalized_weighted_goodput"].isna().sum()
    inf_anwg = np.isinf(results_df["arrival_normalized_weighted_goodput"]).sum()
    print(f"NaN in ANWG: {nan_anwg} (Expected: 0), Inf in ANWG: {inf_anwg} (Expected: 0)")
    
    # ANWG range check
    min_anwg = results_df["arrival_normalized_weighted_goodput"].min()
    max_anwg = results_df["arrival_normalized_weighted_goodput"].max()
    print(f"ANWG range: [{min_anwg}, {max_anwg}] (Expected: [0.0, 1.0])")
    
    # Policy roster per family
    for fam, group in results_df.groupby("generator_family"):
        roster = sorted(list(group["policy_name"].unique()))
        print(f"Roster for {fam}: {roster}")

    # Sobol index uniqueness check
    for fam in ["sobol_family_b_prediction_sensitive", "sobol_family_c_deadline_admission"]:
        fam_feat = features_df[features_df["generator_family"] == fam]
        n_unique_sobol_idx = fam_feat["sobol_index"].nunique()
        expected_points = 128
        print(f"Unique Sobol indices for {fam}: {n_unique_sobol_idx} (Expected: {expected_points})")

    # Joint table for merged analysis
    df = pd.merge(results_df, features_df, on="scenario_id", suffixes=("", "_feat"))

    # 3. ANALYSIS OF FAMILY B (Prediction-sensitive)
    print("----------------------------------------------------------------------")
    print("Analyzing Family B (Prediction-sensitive scheduling)...")
    famB = df[df["generator_family"] == "sobol_family_b_prediction_sensitive"]
    
    # Find best fixed policy by mean ANWG
    policy_meansB = famB.groupby("policy_name")["arrival_normalized_weighted_goodput"].mean()
    print("Mean ANWG per policy in Family B:")
    print(policy_meansB.sort_values(ascending=False))
    best_fixed_policyB = policy_meansB.idxmax()
    print(f"Best fixed policy in Family B: {best_fixed_policyB} with mean {policy_meansB[best_fixed_policyB]:.4f}")

    # Compute oracle headroom and winner frequencies
    # We first pivot to get a policy-by-scenario table of ANWG
    pivotB = famB.pivot(index="scenario_id", columns="policy_name", values="arrival_normalized_weighted_goodput")
    oracleB = pivotB.max(axis=1)
    headroomB = oracleB - pivotB[best_fixed_policyB]
    
    mean_headroomB = headroomB.mean()
    print(f"Mean oracle headroom in Family B: {mean_headroomB:.4f}")
    
    epsilons = [0, 0.005, 0.01, 0.05]
    for eps in epsilons:
        # Scenario has headroom if headroom > eps
        frac_hr = (headroomB > eps).mean()
        print(f"  Fraction of scenarios with headroom > {eps}: {frac_hr:.4f}")

    # Win counts and frequencies
    # Since there can be ties, a policy is a winner if its ANWG >= max_ANWG - epsilon
    print("\nWinner frequencies (with joint wins counted if within epsilon):")
    for eps in epsilons:
        winners = []
        for idx, row in pivotB.iterrows():
            m = row.max()
            win_policies = row[row >= m - eps].index.tolist()
            winners.extend(win_policies)
        counts = pd.Series(winners).value_counts()
        print(f"Epsilon = {eps}:")
        for policy, count in counts.items():
            print(f"  {policy}: {count} / 1024 ({count/1024:.4f})")
            
        # Tie rate: fraction of scenarios with > 1 policy winning
        n_winners_per_scenario = (pivotB.apply(lambda r: (r >= r.max() - eps).sum(), axis=1))
        tie_rate = (n_winners_per_scenario > 1).mean()
        print(f"  Tie rate (scenarios with multiple winners): {tie_rate:.4f}")

    # Best-vs-second margin
    # To get second best, we sort each row and take the second largest element
    def get_margin(row):
        sorted_vals = sorted(row.values, reverse=True)
        return sorted_vals[0] - sorted_vals[1]
    marginsB = pivotB.apply(get_margin, axis=1)
    print(f"Mean best-vs-second margin in Family B: {marginsB.mean():.4f}")

    # Inter-policy variance
    print(f"Mean inter-policy variance in Family B: {pivotB.var(axis=1).mean():.6f}")
    print(f"Mean inter-policy MAD in Family B: {pivotB.mad().mean():.6f}" if hasattr(pivotB, 'mad') else f"Mean inter-policy MAD in Family B: {pivotB.apply(lambda r: np.abs(r - r.median()).median(), axis=1).mean():.6f}")

    # Pairwise separation matrix for Family B (fraction of scenarios where |P1 - P2| > eps)
    print("\nPairwise separation coverage (> 0.01) in Family B:")
    policiesB = sorted(pivotB.columns.tolist())
    sep_matrixB = pd.DataFrame(index=policiesB, columns=policiesB, dtype=float)
    for p1 in policiesB:
        for p2 in policiesB:
            sep_matrixB.loc[p1, p2] = (np.abs(pivotB[p1] - pivotB[p2]) > 0.01).mean()
    print(sep_matrixB)

    # Let's inspect margin surfaces (e.g. ESTF-vs-FIFO, SOF-vs-FIFO, Aging-vs-ESTF) as a function of heterogeneity, target_utilization, inversion_fraction
    # Pivot features to easily merge with pivot results
    featB = features_df[features_df["generator_family"] == "sobol_family_b_prediction_sensitive"].set_index("scenario_id")
    mergedB = pivotB.join(featB[["heterogeneity", "target_utilization", "inversion_fraction"]])

    # ESTF - FIFO, SOF - FIFO, Aging - ESTF
    mergedB["estf_vs_fifo"] = mergedB["estimated_service_time_first"] - mergedB["fifo"]
    mergedB["sof_vs_fifo"] = mergedB["shortest_output_first"] - mergedB["fifo"]
    mergedB["aging_vs_estf"] = mergedB["aging_priority"] - mergedB["estimated_service_time_first"]

    print("\nFamily B margin analysis split by Heterogeneity:")
    for het, g in mergedB.groupby("heterogeneity"):
        print(f"Heterogeneity = {het}:")
        print(f"  ESTF - FIFO mean margin: {g['estf_vs_fifo'].mean():.4f}")
        print(f"  SOF - FIFO mean margin:  {g['sof_vs_fifo'].mean():.4f}")
        print(f"  Aging - ESTF mean margin: {g['aging_vs_estf'].mean():.4f}")
        # Let's also check best policy and headroom
        g_pivot = g[policiesB]
        g_oracle = g_pivot.max(axis=1)
        g_best_fixed = g_pivot.mean().idxmax()
        g_headroom = g_oracle - g_pivot[g_best_fixed]
        print(f"  Best fixed policy: {g_best_fixed} (mean={g_pivot[g_best_fixed].mean():.4f})")
        print(f"  Mean oracle headroom: {g_headroom.mean():.4f}")
        print(f"  Frac headroom > 0.01: {(g_headroom > 0.01).mean():.4f}")

    # Let's do binning of load (target_utilization) and error (inversion_fraction) for strong heterogeneity
    strong_df = mergedB[mergedB["heterogeneity"] == "strong"]
    print("\nStrong Heterogeneity Binned Margins:")
    # We bin target_utilization into low (<0.7), mid [0.7, 0.9), high (>=0.9)
    # We bin inversion_fraction into low (<0.3), mid [0.3, 0.7), high (>=0.7)
    strong_df = strong_df.copy()
    strong_df["util_bin"] = pd.cut(strong_df["target_utilization"], bins=[0.5, 0.7, 0.9, 1.1001], right=False, labels=["low", "mid", "high"])
    strong_df["inv_bin"] = pd.cut(strong_df["inversion_fraction"], bins=[0.0, 0.3, 0.7, 1.0001], right=False, labels=["low", "mid", "high"])

    pivot_util_inv = strong_df.groupby(["util_bin", "inv_bin"], observed=False).apply(
        lambda g: pd.Series({
            "fifo": g["fifo"].mean(),
            "estf": g["estimated_service_time_first"].mean(),
            "sof": g["shortest_output_first"].mean(),
            "aging": g["aging_priority"].mean(),
            "headroom": (g[policiesB].max(axis=1) - g["estimated_service_time_first"]).mean(), # headroom over global best ESTF
            "count": len(g)
        })
    )
    print(pivot_util_inv)

    print("\nWhich policies win in what regions of target_utilization and inversion_fraction under strong heterogeneity?")
    for (ub, ib), g in strong_df.groupby(["util_bin", "inv_bin"], observed=False):
        g_pivot = g[policiesB]
        winners = g_pivot.idxmax(axis=1)
        winner_counts = winners.value_counts().to_dict()
        print(f"  Util: {ub}, Inv: {ib} (n={len(g)}) -> Winners: {winner_counts}")

    # 4. ANALYSIS OF FAMILY C (Deadline/Admission)
    print("----------------------------------------------------------------------")
    print("Analyzing Family C (Deadline/Admission scheduling)...")
    famC = df[df["generator_family"] == "sobol_family_c_deadline_admission"]
    
    # Find best fixed policy by mean ANWG
    policy_meansC = famC.groupby("policy_name")["arrival_normalized_weighted_goodput"].mean()
    print("Mean ANWG per policy in Family C:")
    print(policy_meansC.sort_values(ascending=False))
    best_fixed_policyC = policy_meansC.idxmax()
    print(f"Best fixed policy in Family C: {best_fixed_policyC} with mean {policy_meansC[best_fixed_policyC]:.4f}")

    pivotC = famC.pivot(index="scenario_id", columns="policy_name", values="arrival_normalized_weighted_goodput")
    oracleC = pivotC.max(axis=1)
    headroomC = oracleC - pivotC[best_fixed_policyC]
    print(f"Mean oracle headroom in Family C: {headroomC.mean():.4f}")
    
    # Win counts and frequencies in Family C
    for eps in [0, 0.005, 0.01]:
        winnersC = []
        for idx, row in pivotC.iterrows():
            m = row.max()
            win_policies = row[row >= m - eps].index.tolist()
            winnersC.extend(win_policies)
        countsC = pd.Series(winnersC).value_counts()
        print(f"Epsilon = {eps}:")
        for policy, count in countsC.items():
            print(f"  {policy}: {count} / 512 ({count/512:.4f})")
        n_winners_per_scenario = (pivotC.apply(lambda r: (r >= r.max() - eps).sum(), axis=1))
        tie_rateC = (n_winners_per_scenario > 1).mean()
        print(f"  Tie rate: {tie_rateC:.4f}")

    # Examine secondary metrics in Family C (e.g. completion_fraction, slo_violation_rate, num_dropped)
    # Let's average these by policy
    secondaryC = famC.groupby("policy_name")[["completion_fraction", "slo_violation_rate", "num_dropped", "num_total"]].mean()
    print("\nSecondary metrics per policy in Family C:")
    print(secondaryC)

    # Margin surfaces: SCORPIO - EDF, LLF - EDF, admission_control - EDF
    featC = features_df[features_df["generator_family"] == "sobol_family_c_deadline_admission"].set_index("scenario_id")
    mergedC = pivotC.join(featC[["overload_factor", "fraction_impossible"]])
    mergedC["scorpio_vs_edf"] = mergedC["scorpio_style_slo_guard"] - mergedC["edf"]
    mergedC["llf_vs_edf"] = mergedC["least_laxity_first"] - mergedC["edf"]
    mergedC["admission_vs_edf"] = mergedC["admission_control"] - mergedC["edf"]

    print("\nFamily C Margin Analysis (means):")
    print(f"  SCORPIO - EDF: {mergedC['scorpio_vs_edf'].mean():.4f}")
    print(f"  LLF - EDF:     {mergedC['llf_vs_edf'].mean():.4f}")
    print(f"  Admission - EDF: {mergedC['admission_vs_edf'].mean():.4f}")

    # Let's check how they separate across overload_factor and fraction_impossible
    mergedC["overload_bin"] = pd.cut(mergedC["overload_factor"], bins=[0.85, 1.0, 1.15, 1.4001], right=False, labels=["low", "mid", "high"])
    mergedC["impossible_bin"] = pd.cut(mergedC["fraction_impossible"], bins=[0.0, 0.2, 0.5, 0.8001], right=False, labels=["low", "mid", "high"])

    print("\nFamily C Binned Margins:")
    pivot_overload_imp = mergedC.groupby(["overload_bin", "impossible_bin"], observed=False).apply(
        lambda g: pd.Series({
            "edf": g["edf"].mean(),
            "fifo": g["fifo"].mean(),
            "llf": g["least_laxity_first"].mean(),
            "scorpio": g["scorpio_style_slo_guard"].mean(),
            "admission": g["admission_control"].mean(),
            "scorpio_vs_edf": g["scorpio_vs_edf"].mean(),
            "admission_vs_edf": g["admission_vs_edf"].mean(),
            "count": len(g)
        })
    )
    print(pivot_overload_imp)

    # 5. ANALYSIS OF FCFS CATEGORICAL ADD-ON
    print("----------------------------------------------------------------------")
    print("Analyzing FCFS categorical add-on...")
    famFCFS = df[df["generator_family"] == "fcfs_categorical_add_on"]
    
    pivotFCFS = famFCFS.pivot(index="scenario_id", columns="policy_name", values="arrival_normalized_weighted_goodput")
    featFCFS = features_df[features_df["family"] == "fcfs_convoy"].set_index("scenario_id")
    mergedFCFS = pivotFCFS.join(featFCFS[["ratio", "n_short", "offset", "max_active_sequences"]])

    # Split by offset=0.0 (Template A1) vs offset>0.0 (Template A2)
    a1_df = mergedFCFS[mergedFCFS["offset"] == 0.0]
    a2_df = mergedFCFS[mergedFCFS["offset"] > 0.0]
    
    print(f"Template A1 (offset=0.0, n={len(a1_df)}) means:")
    print(a1_df[["fifo", "estimated_service_time_first", "shortest_output_first", "aging_priority"]].mean())
    
    print(f"Template A2 (offset > 0.0, n={len(a2_df)}) means:")
    print(a2_df[["fifo", "estimated_service_time_first", "shortest_output_first", "aging_priority"]].mean())

    # 6. DATASET POLICY-SEPARATION FOOTPRINT (Section 4)
    print("----------------------------------------------------------------------")
    print("Computing Dataset-wide Policy-Separation Footprint...")
    
    # Let's compute all-policy near-tie rate per family
    # Near-tie means max ANWG - min ANWG <= 0.01 for that scenario
    # Or let's see how they define it: "scenarios where all policies are near-tied (e.g. within 0.005, 0.01, 0.05)"
    # Let's compute both:
    # 1. Near-tie of all policies (max - min <= eps)
    # 2. Pairwise separation coverage: fraction of scenarios where any two policies differ by > 0.01.
    
    for fam_name, pivot_fam in [("Family B", pivotB), ("Family C", pivotC), ("FCFS", pivotFCFS)]:
        print(f"\n--- {fam_name} ---")
        max_min_diff = pivot_fam.max(axis=1) - pivot_fam.min(axis=1)
        for eps in [0.005, 0.01, 0.05]:
            near_tie_all = (max_min_diff <= eps).mean()
            print(f"  All-policy near-tie rate (max - min <= {eps}): {near_tie_all:.4f}")
            
        # Mean inter-policy variance
        var_fam = pivot_fam.var(axis=1).mean()
        print(f"  Mean inter-policy variance: {var_fam:.6f}")
        
        # Unique winners count (epsilon=0)
        winners_list = []
        for idx, row in pivot_fam.iterrows():
            m = row.max()
            win_p = row[row == m].index.tolist()
            winners_list.extend(win_p)
        print(f"  Unique winner policies (eps=0): {set(winners_list)}")
        
        # Policy pairs that appear functionally equivalent (mean absolute difference < 0.001)
        equivalent_pairs = []
        p_list = sorted(pivot_fam.columns.tolist())
        for i in range(len(p_list)):
            for j in range(i+1, len(p_list)):
                p1, p2 = p_list[i], p_list[j]
                mad = np.abs(pivot_fam[p1] - pivot_fam[p2]).mean()
                if mad < 0.001:
                    equivalent_pairs.append((p1, p2, mad))
        if equivalent_pairs:
            print(f"  Functionally equivalent pairs (MAD < 0.001):")
            for p1, p2, mad in equivalent_pairs:
                print(f"    {p1} and {p2} (MAD={mad:.6f})")
        else:
            print("  No functionally equivalent pairs (MAD < 0.001)")

if __name__ == "__main__":
    main()
