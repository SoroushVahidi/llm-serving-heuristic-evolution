#!/usr/bin/env python3
"""Feasibility diagnostics for the online regime-signal telemetry.

FEASIBILITY STUDY ONLY -- no router or family-specific selector trained.
Only simple fixed-rule/single-score diagnostics (precision/recall/AUROC of
already-preregistered activity labels and their underlying continuous
scores) are computed, per
docs/audits/online_regime_signal_feasibility_v1_20260817.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_DIR = REPO_ROOT / "experiments" / "online_regime_signal_feasibility_v1"
TELEMETRY_CSV = TELEMETRY_DIR / "online_regime_telemetry_v1.csv"

FAMILY_A = "FAMILY_A_FAIRNESS_STARVATION_V2"
FAMILY_B = "FAMILY_B_PREFILL_DECODE_V2"
FAMILY_C = "FAMILY_C_KV_PRESSURE_V2"


def temporal_dynamics(df: pd.DataFrame) -> dict:
    out = {}
    for fam, g in df.groupby("mechanism_family"):
        n_scen = g["canonical_scenario_id"].nunique()
        transitions = {"a_active": 0, "b_active_v2": 0, "c_active": 0}
        n_never_active = {"a_active": 0, "b_active_v2": 0, "c_active": 0}
        n_always_active = {"a_active": 0, "b_active_v2": 0, "c_active": 0}
        for sid, gg in g.groupby("canonical_scenario_id"):
            gg = gg.sort_values("step")
            for label in transitions:
                vals = gg[label].to_numpy()
                flips = int((vals[1:] != vals[:-1]).sum())
                transitions[label] += flips
                if not vals.any():
                    n_never_active[label] += 1
                if vals.all():
                    n_always_active[label] += 1
        out[fam] = {
            "n_scenarios": int(n_scen),
            "total_label_transitions": transitions,
            "n_scenarios_never_active": n_never_active,
            "n_scenarios_always_active": n_always_active,
            "mean_transitions_per_scenario": {k: v / n_scen for k, v in transitions.items()},
        }
    return out


def overlap_distribution(df: pd.DataFrame) -> dict:
    combo = df[["a_active", "b_active_v2", "c_active"]].astype(int)
    labels = combo.apply(lambda r: (r["a_active"], r["b_active_v2"], r["c_active"]), axis=1)
    counts = labels.value_counts()
    total = len(df)
    named = {}
    name_map = {
        (0, 0, 0): "none",
        (1, 0, 0): "A_only",
        (0, 1, 0): "B_only",
        (0, 0, 1): "C_only",
        (1, 1, 0): "A+B",
        (1, 0, 1): "A+C",
        (0, 1, 1): "B+C",
        (1, 1, 1): "A+B+C",
    }
    for combo_key, name in name_map.items():
        n = int(counts.get(combo_key, 0))
        named[name] = {"n_rows": n, "fraction": n / total}
    # also per-family
    by_family = {}
    for fam, g in df.groupby("mechanism_family"):
        combo_f = g[["a_active", "b_active_v2", "c_active"]].astype(int)
        labels_f = combo_f.apply(lambda r: (r["a_active"], r["b_active_v2"], r["c_active"]), axis=1)
        counts_f = labels_f.value_counts()
        by_family[fam] = {
            name_map[k]: int(v) for k, v in counts_f.items() if k in name_map
        }
    return {"overall": named, "by_family": by_family}


def family_a_sanity(df: pd.DataFrame) -> dict:
    """A signals should rise under priority/backlog heterogeneity. External
    validation only (never fed back into the a_active formula itself):
    correlate the FRACTION of a_scenario's steps with a_active=True against
    that scenario's own frozen tenant_weight_skew sweep parameter (read
    from MF-PSD, audit metadata, not a learnable feature)."""
    a = df[df["mechanism_family"] == FAMILY_A]
    frac_active = a.groupby("canonical_scenario_id")["a_active"].mean()

    mf_psd = pd.read_csv(REPO_ROOT / "experiments" / "mf_psd_v1" / "mf_psd_scenarios_v1.csv")
    mf_psd_a = mf_psd[mf_psd["mechanism_family"] == FAMILY_A].set_index("canonical_scenario_id")
    skew = mf_psd_a["feat_A__tenant_weight_skew"].astype(float)

    merged = pd.concat([frac_active.rename("frac_a_active"), skew], axis=1, join="inner")
    from scipy.stats import spearmanr

    rho, p = spearmanr(merged["frac_a_active"], merged["feat_A__tenant_weight_skew"])
    control = merged[merged["feat_A__tenant_weight_skew"] == 1.0]
    stress = merged[merged["feat_A__tenant_weight_skew"] > 1.0]
    return {
        "n_scenarios": len(merged),
        "spearman_frac_a_active_vs_tenant_weight_skew": float(rho),
        "p_value": float(p),
        "mean_frac_a_active_control_skew_eq_1": float(control["frac_a_active"].mean()) if len(control) else None,
        "mean_frac_a_active_stress_skew_gt_1": float(stress["frac_a_active"].mean()) if len(stress) else None,
    }


def family_c_sanity(df: pd.DataFrame) -> dict:
    c = df[df["mechanism_family"] == FAMILY_C]
    frac_active = c.groupby("canonical_scenario_id")["c_active"].mean()
    peak_kv = c.groupby("canonical_scenario_id")["kv_pressure"].max()

    mf_psd = pd.read_csv(REPO_ROOT / "experiments" / "mf_psd_v1" / "mf_psd_scenarios_v1.csv")
    mf_psd_c = mf_psd[mf_psd["mechanism_family"] == FAMILY_C].set_index("canonical_scenario_id")
    bulk = mf_psd_c["feat_C__bulk_pressure"]

    merged = pd.concat([frac_active.rename("frac_c_active"), peak_kv.rename("peak_kv_pressure"), bulk], axis=1, join="inner")
    by_bulk = merged.groupby("feat_C__bulk_pressure")[["frac_c_active", "peak_kv_pressure"]].mean()
    return {
        "n_scenarios": len(merged),
        "mean_frac_c_active_by_bulk_pressure": by_bulk["frac_c_active"].to_dict(),
        "mean_peak_kv_pressure_by_bulk_pressure": by_bulk["peak_kv_pressure"].to_dict(),
    }


def falsification_cases(df: pd.DataFrame) -> dict:
    """Within real Family-B replay trajectories, identify windows matching
    each of the 4 required cases directly from telemetry (no synthetic
    microcases needed -- every trajectory naturally passes through a
    prefill-heavy start, a decode-heavy tail, and (when it occurs) a mixed
    middle)."""
    b = df[df["mechanism_family"] == FAMILY_B].sort_values(["canonical_scenario_id", "step"])
    out = {}
    for case_name, cond in [
        ("prefill_only_no_decode", (b["prefill_fraction_of_active"] > 0) & (b["decode_fraction_of_active"] == 0)),
        ("decode_only_no_prefill_backlog", (b["decode_fraction_of_active"] > 0) & (b["prefill_fraction_of_active"] == 0)),
        ("simultaneous_heavy_contention", b["b_active_v2"]),
        ("low_load_mixed", (b["queue_length"] <= 1) & (~b["b_active_v2"]) & ((b["prefill_fraction_of_active"] > 0) | (b["decode_fraction_of_active"] > 0))),
    ]:
        rows = b[cond]
        out[case_name] = {
            "n_rows": int(len(rows)),
            "n_scenarios_with_case": int(rows["canonical_scenario_id"].nunique()),
            "mean_contention_score_v2": float(rows["contention_score_v2"].mean()) if len(rows) else None,
        }
    # Explicit distinguishing check: does b_active_v2 correctly separate
    # case 3 from cases 1/2/4?
    case3_scores = b.loc[b["b_active_v2"], "contention_score_v2"]
    non_case3_scores = b.loc[~b["b_active_v2"], "contention_score_v2"]
    out["separation_check"] = {
        "mean_contention_score_v2_when_b_active": float(case3_scores.mean()) if len(case3_scores) else None,
        "mean_contention_score_v2_when_not_b_active": float(non_case3_scores.mean()) if len(non_case3_scores) else None,
    }
    return out


def diagnostic_performance(df: pd.DataFrame) -> dict:
    """AUROC of each preregistered continuous score against 'is this row
    from the mechanism's own native family' -- a fixed-rule, no-search
    diagnostic (the score/threshold were already frozen before this)."""
    out = {}
    y_a = (df["mechanism_family"] == FAMILY_A).astype(int)
    y_b = (df["mechanism_family"] == FAMILY_B).astype(int)
    y_c = (df["mechanism_family"] == FAMILY_C).astype(int)

    out["priority_skew_auroc_vs_family_A"] = float(roc_auc_score(y_a, df["priority_skew"]))
    out["contention_score_v2_auroc_vs_family_B"] = float(roc_auc_score(y_b, df["contention_score_v2"]))
    out["contention_score_product_auroc_vs_family_B"] = float(roc_auc_score(y_b, df["contention_score_product"]))
    out["kv_pressure_auroc_vs_family_C"] = float(roc_auc_score(y_c, df["kv_pressure"]))

    def prf(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=bool)
        y_pred = np.asarray(y_pred, dtype=bool)
        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

    out["b_active_v2_precision_recall_vs_family_B"] = prf(y_b, df["b_active_v2"])
    out["a_active_precision_recall_vs_family_A"] = prf(y_a, df["a_active"])
    out["c_active_precision_recall_vs_family_C"] = prf(y_c, df["c_active"])
    return out


def temporal_leakage_spotcheck(df: pd.DataFrame) -> dict:
    """Sanity spot-check (not a substitute for the code-level tests):
    queue_length at step 0 of every scenario should be small (only
    requests with arrival_time <= 0 have arrived), never the scenario's
    full request count -- a coarse signature that no future-arrival
    information leaked into step-0 telemetry."""
    first_rows = df.sort_values("step").groupby("canonical_scenario_id").first()
    return {
        "n_scenarios_checked": len(first_rows),
        "max_queue_length_at_first_recorded_step": float(first_rows["queue_length"].max()),
        "mean_queue_length_at_first_recorded_step": float(first_rows["queue_length"].mean()),
    }


def main() -> None:
    df = pd.read_csv(TELEMETRY_CSV)
    out = {
        "n_rows": len(df),
        "temporal_dynamics": temporal_dynamics(df),
        "overlap_distribution": overlap_distribution(df),
        "family_a_sanity": family_a_sanity(df),
        "family_c_sanity": family_c_sanity(df),
        "falsification_cases": falsification_cases(df),
        "diagnostic_performance": diagnostic_performance(df),
        "temporal_leakage_spotcheck": temporal_leakage_spotcheck(df),
    }
    out_path = TELEMETRY_DIR / "online_regime_telemetry_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
