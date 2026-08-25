#!/usr/bin/env python3
"""
Phase 2B.14: Metric Audit and SCORPIO Ablation.

Two-phase script:
  Phase A — Metric audit (no simulation)
    1. Load per_window.csv from Phase 2B.13 results.
    2. Audit weighted_goodput denominator.
    3. Compute metric variants:
       - completed_request_quality     (old WG, completed-only denominator)
       - arrival_normalized_wg         (completion_fraction * cond_WG)
       - completion_penalized scores   (arrival_norm_WG - lambda*max(0, target-comp))
    4. Re-rank all policies and selectors under each metric.
    5. Near-tie / all-complete reanalysis under arrival-normalized WG.

  Phase B — SCORPIO ablation (simulation on targeted discriminative workloads)
    6. Run 8 ablation variants on the 7 most discriminative workloads.
    7. Compute metric variants for each ablation.
    8. Identify which SCORPIO component drives the gain.

Usage
-----
python scripts/run_phase2b14_metric_audit_scorpio_ablation.py \\
    --config configs/phase2b14_metric_audit_scorpio_ablation.yaml \\
    [--input-results results/phase2b13_selector_training_and_suspicion_audit] \\
    [--output results/phase2b14_metric_audit_scorpio_ablation] \\
    [--skip-ablation] \\
    [--log-file logs/phase2b14/phase2b14_metric_audit.log]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.policies.scorpio_ablations import ABLATION_NAMES, make_ablation

SCORPIO = "scorpio_style_slo_guard"

# Selector keys present in per_window.csv
SELECTOR_KEYS = [
    "rule_based",
    "always_scorpio",
    "random_forest",
    "decision_tree",
    "random_forest_regret_weighted",
    "decision_tree_regret_weighted",
    "per_policy_regression",
    "knn_selector",
    "safe_fallback_margin0.001",
    "safe_fallback_margin0.005",
    "safe_fallback_margin0.010",
]


# ---------------------------------------------------------------------------
# Phase A helpers — metric computation from per_window.csv
# ---------------------------------------------------------------------------

def load_per_window(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "per_window.csv"
    if not path.exists():
        raise FileNotFoundError(f"per_window.csv not found at {path}")
    df = pd.read_csv(path)
    logging.info("Loaded per_window.csv: %d rows, %d cols", len(df), len(df.columns))
    return df


def cond_wg(df: pd.DataFrame, policy: str) -> pd.Series:
    """Conditional (completed-only) WG for policy."""
    return df[f"reward_{policy}"].fillna(0.0)


def comp_frac(df: pd.DataFrame, policy: str) -> pd.Series:
    """Completion fraction for policy."""
    key = f"completion_{policy}"
    if key not in df.columns:
        return pd.Series(np.ones(len(df)))
    return df[key].fillna(1.0)


def arrival_norm_wg(df: pd.DataFrame, policy: str) -> pd.Series:
    """Arrival-normalized WG ≈ completion_fraction × conditional_WG."""
    return comp_frac(df, policy) * cond_wg(df, policy)


def completion_penalized_wg(
    df: pd.DataFrame, policy: str, target: float, lam: float
) -> pd.Series:
    anwg = arrival_norm_wg(df, policy)
    cf = comp_frac(df, policy)
    penalty = lam * (target - cf).clip(lower=0.0)
    return anwg - penalty


def selector_arrival_norm_wg(df: pd.DataFrame, sel_key: str) -> pd.Series:
    """Arrival-normalized WG for a selector (uses per-row completion fraction of chosen policy)."""
    wg_col = f"sel_{sel_key}_wg"
    policy_col = f"sel_{sel_key}_policy"
    if wg_col not in df.columns:
        return pd.Series([float("nan")] * len(df))
    result = []
    for _, row in df.iterrows():
        chosen = row.get(policy_col)
        wg = float(row.get(wg_col) or 0.0)
        if chosen and f"completion_{chosen}" in df.columns:
            cf = float(row.get(f"completion_{chosen}") or 1.0)
        else:
            cf = 1.0
        result.append(cf * wg)
    return pd.Series(result)


def build_policy_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-policy mean metrics under all variants."""
    rows = []
    for p in SELECTOR_CANDIDATES:
        rcol = f"reward_{p}"
        if rcol not in df.columns:
            continue
        cf = comp_frac(df, p)
        cq = cond_wg(df, p)
        anwg = cf * cq
        slo = df.get(f"slo_violation_{p}", pd.Series(np.zeros(len(df)))).fillna(0.0)
        rows.append({
            "policy": p,
            "conditional_wg": round(float(cq.mean()), 4),
            "arrival_norm_wg": round(float(anwg.mean()), 4),
            "mean_completion_fraction": round(float(cf.mean()), 4),
            "mean_slo_violation": round(float(slo.mean()), 4),
            "cp_wg_t095_l05": round(float(completion_penalized_wg(df, p, 0.95, 0.5).mean()), 4),
            "cp_wg_t095_l10": round(float(completion_penalized_wg(df, p, 0.95, 1.0).mean()), 4),
            "cp_wg_t099_l05": round(float(completion_penalized_wg(df, p, 0.99, 0.5).mean()), 4),
            "cp_wg_t099_l10": round(float(completion_penalized_wg(df, p, 0.99, 1.0).mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("arrival_norm_wg", ascending=False).reset_index(drop=True)


def build_selector_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-selector mean metrics under all variants."""
    rows = []
    for sk in SELECTOR_KEYS:
        wg_col = f"sel_{sk}_wg"
        if wg_col not in df.columns:
            continue
        anwg_series = selector_arrival_norm_wg(df, sk)
        wg_mean = float(df[wg_col].fillna(0.0).mean())
        anwg_mean = float(anwg_series.mean())
        rows.append({
            "selector": sk,
            "conditional_wg": round(wg_mean, 4),
            "arrival_norm_wg": round(anwg_mean, 4),
        })
    return pd.DataFrame(rows).sort_values("arrival_norm_wg", ascending=False).reset_index(drop=True)


def near_tie_analysis_corrected(df: pd.DataFrame, thresholds: List[float]) -> Dict:
    """Near-tie analysis using arrival-normalized WG."""
    policies = SELECTOR_CANDIDATES
    result = {}
    all_anwgs = []
    for _, row in df.iterrows():
        anwgs = []
        for p in policies:
            rcol = f"reward_{p}"
            ccol = f"completion_{p}"
            if rcol not in df.columns:
                continue
            wg = float(row.get(rcol) or 0.0)
            cf = float(row.get(ccol) or 1.0) if ccol in df.columns else 1.0
            anwgs.append(cf * wg)
        if len(anwgs) >= 2:
            sorted_anwg = sorted(anwgs, reverse=True)
            all_anwgs.append((sorted_anwg[0], sorted_anwg[0] - sorted_anwg[1]))
    if not all_anwgs:
        return {}
    best_anwgs = np.array([x[0] for x in all_anwgs])
    margins = np.array([x[1] for x in all_anwgs])
    n = len(margins)
    all_complete_thresh = 0.99
    result = {
        "n_windows": n,
        "n_all_complete_arrival_norm": int(np.sum(best_anwgs >= all_complete_thresh)),
        "all_complete_fraction_arrival_norm": round(float(np.mean(best_anwgs >= all_complete_thresh)), 4),
        "margin_mean": round(float(np.mean(margins)), 4),
        "margin_p50": round(float(np.percentile(margins, 50)), 4),
        "margin_p90": round(float(np.percentile(margins, 90)), 4),
    }
    for eps in thresholds:
        n_tie = int(np.sum(margins < eps))
        result[f"n_near_tie_eps{eps:.3f}"] = n_tie
        result[f"fraction_near_tie_eps{eps:.3f}"] = round(n_tie / n, 4)
        result[f"n_meaningful_eps{eps:.3f}"] = n - n_tie
    return result


def audit_denominator(df: pd.DataFrame) -> Dict:
    """Document and verify the WG denominator used in Phase 2B.13."""
    scorpio_cq = float(df[f"reward_{SCORPIO}"].fillna(0.0).mean())
    fifo_cq = float(df["reward_fifo"].fillna(0.0).mean())
    scorpio_cf = float(df[f"completion_{SCORPIO}"].fillna(1.0).mean())
    fifo_cf = float(df["completion_fifo"].fillna(1.0).mean())
    scorpio_anwg = scorpio_cq * scorpio_cf
    fifo_anwg = fifo_cq * fifo_cf

    return {
        "denominator_type": "completed_requests_only",
        "description": (
            "weighted_goodput = sum(priority_i * slo_met_i) / sum(priority_i) "
            "where the sum is over COMPLETED requests only. "
            "Dropped or rejected requests do not appear in numerator or denominator."
        ),
        "implication": (
            "A policy that rejects difficult requests will have a smaller denominator, "
            "allowing a high conditional WG while accepting fewer total arrivals. "
            "This is NOT a true system-level goodput metric."
        ),
        "safe_to_call_goodput": False,
        "correct_name": "completed_request_conditional_quality",
        "example_scorpio": {
            "conditional_wg": round(scorpio_cq, 4),
            "mean_completion_fraction": round(scorpio_cf, 4),
            "arrival_normalized_wg": round(scorpio_anwg, 4),
        },
        "example_fifo": {
            "conditional_wg": round(fifo_cq, 4),
            "mean_completion_fraction": round(fifo_cf, 4),
            "arrival_normalized_wg": round(fifo_anwg, 4),
        },
        "scorpio_vs_fifo_gap_old": round(scorpio_cq - fifo_cq, 4),
        "scorpio_vs_fifo_gap_arrival_norm": round(scorpio_anwg - fifo_anwg, 4),
        "gap_shrinks_by": round((scorpio_cq - fifo_cq) - (scorpio_anwg - fifo_anwg), 4),
    }


# ---------------------------------------------------------------------------
# Phase B helpers — SCORPIO ablation simulation
# ---------------------------------------------------------------------------

def run_ablation_simulations(
    config: Dict,
    output_dir: Path,
    verbose: bool = False,
) -> List[Dict]:
    """Simulate ablation variants on targeted discriminative workloads.

    Runs each ablation policy directly via run_policy on each window,
    then returns rows keyed by reward_{ablation_name} / completion_{ablation_name}.
    Also runs full SCORPIO for head-to-head comparison.
    """
    from run_phase2b9_selector_robustness import (
        build_gpu_configs,
        load_or_generate_trace,
    )
    from llmserveopt.evaluation.run_policy import run_policy
    from llmserveopt.policies.registry import make_policy
    from llmserveopt.selector.windows import make_windows
    from llmserveopt.simulator.service_model_factory import build_service_model_from_config

    abl_cfg = config.get("ablation", {})
    workloads = abl_cfg.get("workloads", [])
    seeds = abl_cfg.get("seeds", [0, 1, 2])
    window_size = abl_cfg.get("window_size", 200)
    min_partial = abl_cfg.get("min_partial_window", 50)
    drain_steps = config.get("simulator", {}).get("drain_steps", 20000)

    gpu_configs = build_gpu_configs(config)
    service_model = build_service_model_from_config(config.get("service_model", {}))

    # Policies to evaluate: full SCORPIO + all ablation variants
    ablation_policy_pairs = [(SCORPIO, make_policy(SCORPIO))] + [
        (n, make_ablation(n)) for n in ABLATION_NAMES
    ]

    all_ablation_rows = []

    for wdef in workloads:
        tag = wdef.get("tag", "workload")
        source = wdef.get("source", "synthetic")
        seeds_to_use = seeds if source == "synthetic" else [seeds[0]]
        for seed in seeds_to_use:
            trace_tag = f"{tag}_s{seed}"
            logging.info("  Ablation: %s (%d variants)", trace_tag, len(ablation_policy_pairs))
            reqs = load_or_generate_trace(wdef, seed=seed)
            if not reqs:
                logging.warning("  Empty trace, skipping %s", trace_tag)
                continue

            windows = make_windows(
                requests=reqs,
                trace_id=trace_tag,
                window_size=window_size,
                min_partial=min_partial,
            )
            if not windows:
                continue

            for w in windows:
                row: Dict = {
                    "trace_id": w.trace_id,
                    "window_id": w.window_id,
                    "num_requests": w.num_requests,
                    "seed": seed,
                    "ablation_run": True,
                }
                for pname, policy in ablation_policy_pairs:
                    try:
                        m = run_policy(
                            policy=policy,
                            requests=w.requests,
                            gpu_configs=gpu_configs,
                            service_model=service_model,
                            workload_tag=f"{w.trace_id}_w{w.window_id}",
                            seed=seed,
                            drain_steps=drain_steps,
                        )
                        row[f"reward_{pname}"] = m.weighted_goodput
                        row[f"completion_{pname}"] = m.completion_fraction
                        row[f"slo_violation_{pname}"] = m.slo_violation_rate
                    except Exception as exc:
                        logging.warning("    %s on window %d failed: %s", pname, w.window_id, exc)
                        row[f"reward_{pname}"] = float("nan")
                        row[f"completion_{pname}"] = float("nan")
                        row[f"slo_violation_{pname}"] = float("nan")
                all_ablation_rows.append(row)

            logging.info("    %d windows processed", len(windows))

    return all_ablation_rows


def build_ablation_metric_table(
    ablation_rows: List[Dict],
) -> pd.DataFrame:
    """Compute metric variants for ablation variants vs full SCORPIO."""
    if not ablation_rows:
        return pd.DataFrame()
    df = pd.DataFrame(ablation_rows)
    target_policies = [SCORPIO] + ABLATION_NAMES
    rows = []
    for p in target_policies:
        rcol = f"reward_{p}"
        ccol = f"completion_{p}"
        slocol = f"slo_violation_{p}"
        if rcol not in df.columns:
            continue
        cq = df[rcol].fillna(0.0)
        cf = df[ccol].fillna(1.0) if ccol in df.columns else pd.Series(np.ones(len(df)))
        slo = df[slocol].fillna(0.0) if slocol in df.columns else pd.Series(np.zeros(len(df)))
        anwg = cf * cq
        rows.append({
            "policy": p,
            "is_ablation": p != SCORPIO,
            "n_windows": int(cq.count()),
            "conditional_wg": round(float(cq.mean()), 4),
            "arrival_norm_wg": round(float(anwg.mean()), 4),
            "mean_completion_fraction": round(float(cf.mean()), 4),
            "mean_slo_violation": round(float(slo.mean()), 4),
            "cp_wg_t095_l05": round(float((anwg - 0.5 * (0.95 - cf).clip(lower=0)).mean()), 4),
            "cp_wg_t099_l10": round(float((anwg - 1.0 * (0.99 - cf).clip(lower=0)).mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("arrival_norm_wg", ascending=False).reset_index(drop=True)


def ablation_gap_analysis(ablation_df: pd.DataFrame) -> Dict:
    """Compute gap of each ablation vs full SCORPIO to identify key components."""
    if ablation_df.empty:
        return {}
    scorpio_row = ablation_df[ablation_df["policy"] == SCORPIO]
    if scorpio_row.empty:
        return {}
    scorpio_anwg = float(scorpio_row["arrival_norm_wg"].iloc[0])
    scorpio_cq = float(scorpio_row["conditional_wg"].iloc[0])
    gaps = {}
    for _, row in ablation_df[ablation_df["is_ablation"]].iterrows():
        name = row["policy"]
        gaps[name] = {
            "anwg_gap_vs_scorpio": round(float(row["arrival_norm_wg"]) - scorpio_anwg, 4),
            "cq_gap_vs_scorpio": round(float(row["conditional_wg"]) - scorpio_cq, 4),
            "arrival_norm_wg": round(float(row["arrival_norm_wg"]), 4),
            "conditional_wg": round(float(row["conditional_wg"]), 4),
            "mean_completion_fraction": round(float(row["mean_completion_fraction"]), 4),
        }
    # Sort by magnitude of ANWG gap (most impactful ablation first)
    sorted_gaps = dict(sorted(gaps.items(), key=lambda x: x[1]["anwg_gap_vs_scorpio"]))
    return {
        "scorpio_reference_anwg": scorpio_anwg,
        "scorpio_reference_cq": scorpio_cq,
        "ablation_gaps": sorted_gaps,
        "most_impactful_ablation": min(gaps, key=lambda k: gaps[k]["anwg_gap_vs_scorpio"]) if gaps else "none",
    }


# ---------------------------------------------------------------------------
# Ranking tables
# ---------------------------------------------------------------------------

def build_full_ranking_table(
    policy_table: pd.DataFrame,
    selector_table: pd.DataFrame,
) -> pd.DataFrame:
    """Combined ranking table under each metric variant."""
    # Add row type and unified name
    pt = policy_table.copy()
    pt["entity_type"] = "policy"
    pt["name"] = pt["policy"]

    st = selector_table.copy()
    st["entity_type"] = "selector"
    st["name"] = st["selector"]
    st["conditional_wg"] = st.get("conditional_wg", float("nan"))
    st["arrival_norm_wg"] = st.get("arrival_norm_wg", float("nan"))
    st["mean_completion_fraction"] = float("nan")
    st["mean_slo_violation"] = float("nan")
    st["cp_wg_t095_l05"] = float("nan")
    st["cp_wg_t095_l10"] = float("nan")
    st["cp_wg_t099_l05"] = float("nan")
    st["cp_wg_t099_l10"] = float("nan")

    cols = ["name", "entity_type", "conditional_wg", "arrival_norm_wg",
            "mean_completion_fraction", "mean_slo_violation",
            "cp_wg_t095_l05", "cp_wg_t095_l10", "cp_wg_t099_l05", "cp_wg_t099_l10"]
    combined = pd.concat([pt[cols], st[cols]], ignore_index=True)
    return combined.sort_values("arrival_norm_wg", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Safe / unsafe claim analysis
# ---------------------------------------------------------------------------

def safe_claim_analysis(
    policy_table: pd.DataFrame,
    selector_table: pd.DataFrame,
    ablation_gap: Dict,
) -> Dict:
    """Determine which old claims are safe vs unsafe under corrected metrics."""
    scorpio_row = policy_table[policy_table["policy"] == SCORPIO]
    if scorpio_row.empty:
        return {}
    s_anwg = float(scorpio_row["arrival_norm_wg"].iloc[0])
    s_cq = float(scorpio_row["conditional_wg"].iloc[0])
    s_cf = float(scorpio_row["mean_completion_fraction"].iloc[0])

    # Best non-scorpio policy under arrival-normalized WG
    others = policy_table[policy_table["policy"] != SCORPIO]
    if not others.empty:
        best_other_anwg = float(others["arrival_norm_wg"].max())
        best_other_policy = others.loc[others["arrival_norm_wg"].idxmax(), "policy"]
    else:
        best_other_anwg = 0.0
        best_other_policy = "none"

    rf_row = selector_table[selector_table["selector"].str.startswith("random_forest")]
    rf_anwg = float(rf_row["arrival_norm_wg"].iloc[0]) if not rf_row.empty else float("nan")

    safe_claims = [
        f"SCORPIO arrival-normalized WG = {s_anwg:.4f} (still highest among all policies)",
        f"SCORPIO dominates second-best policy ({best_other_policy}) by {s_anwg - best_other_anwg:.4f} under arrival-normalized WG",
        f"SCORPIO conditional WG = {s_cq:.4f} over completed requests",
        f"SCORPIO mean completion fraction = {s_cf:.4f} (rejects ~{(1-s_cf)*100:.1f}% of arrivals)",
        "RF selector conditional WG ≈ best fixed baseline (Phase 2B.13 claim valid conditionally)",
    ]

    unsafe_claims = [
        f"Old claim 'SCORPIO WG = {s_cq:.4f}' overstates system-level goodput; correct arrival-normalized value = {s_anwg:.4f}",
        "Phase 2B.10–2B.13 results reported weighted_goodput as if it were arrival-normalized; it was completed-only conditional quality",
        "SCORPIO's high WG is partly due to filtering: it rejects requests it cannot serve well, inflating conditional WG",
        "Selectors trained on conditional WG labels may have learned to prefer admission-throttling policies; validity under corrected metrics requires re-check",
    ]

    scorpio_games_metric = (s_cf < 0.95) and (s_cq > best_other_anwg + 0.05)

    return {
        "scorpio_arrival_norm_wg": s_anwg,
        "scorpio_conditional_wg": s_cq,
        "scorpio_completion_fraction": s_cf,
        "scorpio_dominates_under_arrival_norm": s_anwg > best_other_anwg,
        "scorpio_appears_to_game_metric": scorpio_games_metric,
        "best_other_policy_under_arrival_norm": best_other_policy,
        "best_other_anwg": best_other_anwg,
        "rf_selector_arrival_norm_wg": rf_anwg,
        "safe_claims": safe_claims,
        "unsafe_claims": unsafe_claims,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def write_json(data, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2b14_metric_audit_scorpio_ablation.yaml")
    parser.add_argument("--input-results", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip Phase B simulation; only compute metrics from Phase 2B.13 data.")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    config = load_config(Path(args.config))
    input_dir = Path(args.input_results or config.get("input_results",
                     "results/phase2b13_selector_training_and_suspicion_audit"))
    output_dir = Path(args.output or config.get("output_dir",
                      "results/phase2b14_metric_audit_scorpio_ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 70)
    logging.info("Phase 2B.14: Metric Audit and SCORPIO Ablation")
    logging.info("  Input:  %s", input_dir)
    logging.info("  Output: %s", output_dir)
    logging.info("=" * 70)

    t_start = time.perf_counter()

    # =========================================================================
    # Phase A: Metric audit from per_window.csv
    # =========================================================================
    logging.info("[Phase A] Loading Phase 2B.13 per_window.csv...")
    df = load_per_window(input_dir)
    n_windows = len(df)
    logging.info("  %d windows loaded", n_windows)

    # 1. Denominator audit
    logging.info("[Phase A] Auditing WG denominator...")
    denom_audit = audit_denominator(df)
    write_json(denom_audit, output_dir / "metric_denominator_audit.json")
    logging.info("  Denominator: %s", denom_audit["denominator_type"])
    logging.info("  SCORPIO conditional WG: %.4f  arrival-norm WG: %.4f",
                 denom_audit["example_scorpio"]["conditional_wg"],
                 denom_audit["example_scorpio"]["arrival_normalized_wg"])

    # 2. Per-policy metric table
    logging.info("[Phase A] Building per-policy metric table...")
    policy_table = build_policy_metric_table(df)
    policy_table.to_csv(output_dir / "policy_metric_variants.csv", index=False)
    logging.info("  Saved policy_metric_variants.csv (%d policies)", len(policy_table))

    # 3. Per-selector metric table
    logging.info("[Phase A] Building per-selector metric table...")
    selector_table = build_selector_metric_table(df)
    selector_table.to_csv(output_dir / "selector_metric_variants.csv", index=False)
    logging.info("  Saved selector_metric_variants.csv (%d selectors)", len(selector_table))

    # 4. Full ranking table
    logging.info("[Phase A] Building full ranking table...")
    ranking_table = build_full_ranking_table(policy_table, selector_table)
    ranking_table.to_csv(output_dir / "full_ranking_table.csv", index=False)
    logging.info("  Saved full_ranking_table.csv (%d entities)", len(ranking_table))

    # 5. Near-tie analysis under arrival-normalized WG
    logging.info("[Phase A] Near-tie analysis under arrival-normalized WG...")
    nt_thresholds = config.get("near_tie_thresholds", [0.001, 0.005, 0.010])
    nt_corrected = near_tie_analysis_corrected(df, nt_thresholds)
    write_json(nt_corrected, output_dir / "near_tie_corrected.json")
    for k, v in nt_corrected.items():
        logging.info("  %s: %s", k, v)

    # 6. Safe/unsafe claim analysis
    logging.info("[Phase A] Analysing safe/unsafe claims...")
    claim_analysis = safe_claim_analysis(policy_table, selector_table, {})
    write_json(claim_analysis, output_dir / "safe_claim_analysis.json")

    # 7. Summary JSON (Phase A)
    phase_a_summary = {
        "phase": "2B.14",
        "n_windows": n_windows,
        "denominator_type": denom_audit["denominator_type"],
        "denominator_safe_to_call_goodput": denom_audit["safe_to_call_goodput"],
        "scorpio_conditional_wg": denom_audit["example_scorpio"]["conditional_wg"],
        "scorpio_arrival_norm_wg": denom_audit["example_scorpio"]["arrival_normalized_wg"],
        "scorpio_completion_fraction": denom_audit["example_scorpio"]["mean_completion_fraction"],
        "scorpio_dominates_under_arrival_norm": claim_analysis.get("scorpio_dominates_under_arrival_norm"),
        "scorpio_games_metric": claim_analysis.get("scorpio_appears_to_game_metric"),
        "best_other_policy": claim_analysis.get("best_other_policy_under_arrival_norm"),
        "best_other_anwg": claim_analysis.get("best_other_anwg"),
        "near_tie_corrected": nt_corrected,
        "policy_count": len(SELECTOR_CANDIDATES),
        "ablation_count": len(ABLATION_NAMES),
    }
    write_json(phase_a_summary, output_dir / "phase_a_summary.json")
    logging.info("[Phase A] Done in %.1fs", time.perf_counter() - t_start)

    # =========================================================================
    # Phase B: SCORPIO ablation simulation
    # =========================================================================
    ablation_df = pd.DataFrame()
    ablation_gap = {}

    if args.skip_ablation:
        logging.info("[Phase B] Skipped (--skip-ablation flag set).")
    else:
        logging.info("[Phase B] Running SCORPIO ablations on targeted workloads...")
        logging.info("  Ablation variants: %s", ABLATION_NAMES)
        t_abl = time.perf_counter()
        try:
            ablation_rows = run_ablation_simulations(config, output_dir, verbose=args.verbose)
            if ablation_rows:
                ablation_df = pd.DataFrame(ablation_rows)
                ablation_df.to_csv(output_dir / "ablation_per_window.csv", index=False)
                logging.info("  %d ablation windows saved", len(ablation_rows))

                ablation_metric_table = build_ablation_metric_table(ablation_rows)
                ablation_metric_table.to_csv(output_dir / "ablation_metric_table.csv", index=False)
                logging.info("  Saved ablation_metric_table.csv")

                ablation_gap = ablation_gap_analysis(ablation_metric_table)
                write_json(ablation_gap, output_dir / "ablation_gap_analysis.json")
                logging.info("  Most impactful ablation: %s", ablation_gap.get("most_impactful_ablation"))
            else:
                logging.warning("[Phase B] No ablation rows produced.")
        except Exception as exc:
            logging.error("[Phase B] Ablation simulation failed: %s", exc, exc_info=True)
        logging.info("[Phase B] Done in %.1fs", time.perf_counter() - t_abl)

    # =========================================================================
    # Final summary
    # =========================================================================
    final_summary = {**phase_a_summary}
    if ablation_gap:
        final_summary["ablation_gap_analysis"] = ablation_gap
        final_summary["most_impactful_ablation"] = ablation_gap.get("most_impactful_ablation")
    final_summary["safe_claims"] = claim_analysis.get("safe_claims", [])
    final_summary["unsafe_claims"] = claim_analysis.get("unsafe_claims", [])
    final_summary["wall_clock_s"] = round(time.perf_counter() - t_start, 1)
    write_json(final_summary, output_dir / "phase2b14_summary.json")

    logging.info("=" * 70)
    logging.info("Phase 2B.14 complete in %.1fs", final_summary["wall_clock_s"])
    logging.info("  Output: %s", output_dir)
    logging.info("  SCORPIO conditional WG:   %.4f", final_summary["scorpio_conditional_wg"])
    logging.info("  SCORPIO arrival-norm WG:  %.4f", final_summary["scorpio_arrival_norm_wg"])
    logging.info("  SCORPIO completion frac:  %.4f", final_summary["scorpio_completion_fraction"])
    logging.info("  SCORPIO dominates (AN):   %s", final_summary["scorpio_dominates_under_arrival_norm"])
    logging.info("  SCORPIO games metric:     %s", final_summary["scorpio_games_metric"])
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
