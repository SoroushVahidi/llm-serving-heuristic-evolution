#!/usr/bin/env python3
"""
Generate Phase 1.7C consolidated experiment summary.

Reads summary.csv from each experiment's result directory and produces:
  results/phase17c/phase17c_experiment_summary.md
  results/phase17c/prediction_noise_sensitivity.md
  results/phase17c/prediction_noise_sensitivity.csv
  results/phase17c/calibrated_vs_synthetic_comparison.md
  results/phase17c/calibrated_vs_synthetic_rank_correlations.csv
  results/phase17c/plots/ (cross-experiment plots)

Usage:
  python scripts/generate_phase17c_summary.py 2>&1 | tee results/phase17c/summary_gen.log
  python scripts/generate_phase17c_summary.py --output-dir /tmp/phase17c_preview
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "phase17c"
PLOTS = OUT / "plots"

# Map: experiment_name -> (result_dir_prefix, label)
EXPERIMENTS = [
    ("burstgpt_natural_calibrated",            "1. Natural BurstGPT — calibrated service"),
    ("burstgpt_scaled_moderate_calibrated",    "2. Moderate-scaled BurstGPT — calibrated service"),
    ("burstgpt_scaled_high_calibrated",        "3. High-scaled BurstGPT — calibrated service"),
    ("burstgpt_scaled_moderate_synthetic_service", "4. Moderate-scaled BurstGPT — synthetic service"),
    ("burstgpt_moderate_exact_prediction",     "5. Moderate — exact prediction"),
    ("burstgpt_moderate_noise035",             "6. Moderate — noise035 (natural trace)"),
    ("burstgpt_moderate_noise070",             "7. Moderate — noise070 (pre-noised trace)"),
]

DISPLAY_COLS = [
    "policy", "mean_latency", "p95_latency", "p99_latency",
    "mean_queuing_delay", "slo_violation_rate",
    "mean_ttft", "p95_ttft", "mean_tpot", "p95_tpot",
    "request_throughput", "mean_gpu_utilization", "mean_active_batch_size",
    "num_completed", "mean_prefill_delay",
]


def find_latest_summary(exp_name: str) -> Path | None:
    """Find the most recent summary.csv for an experiment."""
    exp_dir = RESULTS / exp_name
    if not exp_dir.exists():
        return None
    # Find all summary.csv files, pick most recent by parent dir name (timestamp)
    candidates = sorted(exp_dir.glob("*/summary.csv"), reverse=True)
    if not candidates:
        # Try workload-level summary
        candidates = sorted(exp_dir.glob("*/*/summary.csv"), reverse=True)
    return candidates[0] if candidates else None


def load_summary(exp_name: str) -> tuple[pd.DataFrame | None, str]:
    path = find_latest_summary(exp_name)
    if path is None:
        return None, "NOT FOUND"
    try:
        df = pd.read_csv(path)
        return df, str(path.parent)
    except Exception as e:
        return None, f"ERROR: {e}"


def fmt_float(v, decimals=4):
    if pd.isna(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def best_policy(df: pd.DataFrame, col: str, lower_is_better=True) -> str:
    if df is None or col not in df.columns:
        return "N/A"
    valid = df[df[col].notna()]
    if valid.empty:
        return "N/A"
    idx = valid[col].idxmin() if lower_is_better else valid[col].idxmax()
    return str(valid.loc[idx, "policy"])


def df_to_md_table(df: pd.DataFrame, cols: list[str]) -> str:
    available = [c for c in cols if c in df.columns]
    sub = df[available].copy()
    lines = ["| " + " | ".join(available) + " |"]
    lines.append("| " + " | ".join(["---"] * len(available)) + " |")
    for _, row in sub.iterrows():
        cells = []
        for c in available:
            v = row[c]
            if isinstance(v, float):
                cells.append(fmt_float(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_experiment_section(exp_name: str, label: str) -> tuple[str, pd.DataFrame | None]:
    df, loc = load_summary(exp_name)
    lines = [f"### {label}", ""]
    if df is None:
        lines.append(f"**Status: MISSING** ({loc})")
        lines.append("")
        return "\n".join(lines), None

    lines.append(f"**Result directory:** `{loc}`  ")
    lines.append(f"**Requests:** {int(df['num_completed'].iloc[0]) if 'num_completed' in df.columns else 'N/A'}  ")
    lines.append(f"**Policies run:** {len(df)}  ")
    lines.append("")

    lines.append(df_to_md_table(df, DISPLAY_COLS))
    lines.append("")

    mean_best = best_policy(df, "mean_latency")
    p95_best = best_policy(df, "p95_latency")
    slo_best = best_policy(df, "slo_violation_rate")
    throughput_best = best_policy(df, "request_throughput", lower_is_better=False)
    mean_worst = best_policy(df, "mean_latency", lower_is_better=False)

    lines.append(f"**Best mean latency:** `{mean_best}`  ")
    lines.append(f"**Best p95 latency:** `{p95_best}`  ")
    lines.append(f"**Best SLO violation rate:** `{slo_best}`  ")
    lines.append(f"**Best throughput:** `{throughput_best}`  ")
    lines.append(f"**Worst mean latency:** `{mean_worst}`  ")
    lines.append("")

    return "\n".join(lines), df


def generate_noise_sensitivity(dataframes: dict[str, pd.DataFrame | None]) -> tuple[str, pd.DataFrame]:
    exact_df = dataframes.get("burstgpt_moderate_exact_prediction")
    n035_df = dataframes.get("burstgpt_moderate_noise035")
    n070_df = dataframes.get("burstgpt_moderate_noise070")

    lines = [
        "# Prediction-Noise Sensitivity Analysis",
        "",
        "Compares three moderate-load variants:",
        "- **exact**: `burstgpt_moderate_exact_prediction` (predicted == actual output tokens)",
        "- **noise035**: `burstgpt_moderate_noise035` (uses natural/moderate trace — note: same trace as moderate calibrated)",
        "- **noise070**: `burstgpt_moderate_noise070` (pre-generated trace with ~70% noise in predicted output tokens)",
        "",
    ]

    # Build combined table
    rows = []
    for policy in sorted(set(
        (list(exact_df["policy"]) if exact_df is not None else []) +
        (list(n035_df["policy"]) if n035_df is not None else []) +
        (list(n070_df["policy"]) if n070_df is not None else [])
    )):
        row = {"policy": policy}
        for tag, df in [("exact", exact_df), ("noise035", n035_df), ("noise070", n070_df)]:
            if df is not None and policy in df["policy"].values:
                prow = df[df["policy"] == policy].iloc[0]
                row[f"mean_lat_{tag}"] = prow.get("mean_latency", float("nan"))
                row[f"p95_lat_{tag}"] = prow.get("p95_latency", float("nan"))
                row[f"slo_viol_{tag}"] = prow.get("slo_violation_rate", float("nan"))
            else:
                row[f"mean_lat_{tag}"] = float("nan")
                row[f"p95_lat_{tag}"] = float("nan")
                row[f"slo_viol_{tag}"] = float("nan")

        # Compute degradations relative to exact (if available)
        base_mean = row.get("mean_lat_exact", float("nan"))
        base_p95 = row.get("p95_lat_exact", float("nan"))
        for tag in ["noise035", "noise070"]:
            ml = row.get(f"mean_lat_{tag}", float("nan"))
            p95 = row.get(f"p95_lat_{tag}", float("nan"))
            if not np.isnan(base_mean) and not np.isnan(ml) and base_mean > 0:
                row[f"delta_mean_{tag}"] = (ml - base_mean) / base_mean
            else:
                row[f"delta_mean_{tag}"] = float("nan")
            if not np.isnan(base_p95) and not np.isnan(p95) and base_p95 > 0:
                row[f"delta_p95_{tag}"] = (p95 - base_p95) / base_p95
            else:
                row[f"delta_p95_{tag}"] = float("nan")
        rows.append(row)

    sensitivity_df = pd.DataFrame(rows)

    # Markdown table
    lines.append("## Per-Policy Comparison")
    lines.append("")
    cols = ["policy", "mean_lat_exact", "mean_lat_noise035", "mean_lat_noise070",
            "delta_mean_noise035", "delta_mean_noise070",
            "p95_lat_exact", "p95_lat_noise035", "p95_lat_noise070",
            "delta_p95_noise035", "delta_p95_noise070"]
    avail_cols = [c for c in cols if c in sensitivity_df.columns]
    lines.append(df_to_md_table(sensitivity_df, avail_cols))
    lines.append("")

    # Observations
    lines.append("## Observations")
    lines.append("")
    if not sensitivity_df.empty and "delta_mean_noise070" in sensitivity_df.columns:
        robust = sensitivity_df[sensitivity_df["delta_mean_noise070"].abs() < 0.05]["policy"].tolist()
        fragile = sensitivity_df[sensitivity_df["delta_mean_noise070"].abs() > 0.20]["policy"].tolist()
        lines.append(f"**Robust policies (< 5% mean latency change at noise070):** {', '.join(robust) or 'none'}")
        lines.append(f"**Fragile policies (> 20% mean latency change at noise070):** {', '.join(fragile) or 'none'}")
        lines.append("")
        lines.append("Policies that depend on predicted output length (shortest_output_first, weighted_shortest_processing,")
        lines.append("multi_bin_batching, vllm_style_token_budget, sarathi_style, splitfuse_style, slo_slack_score)")
        lines.append("are expected to degrade under prediction noise.")
    else:
        lines.append("*Insufficient data for degradation analysis. Check if exact_prediction and noise070 experiments completed.*")
    lines.append("")

    return "\n".join(lines), sensitivity_df


def generate_calibrated_vs_synthetic(
    cal_df: pd.DataFrame | None,
    syn_df: pd.DataFrame | None,
) -> tuple[str, pd.DataFrame]:
    lines = [
        "# Calibrated vs Synthetic Service Model Comparison",
        "",
        "Compares two moderate-scaled BurstGPT experiments that differ only in service model:",
        "- **calibrated**: `burstgpt_scaled_moderate_calibrated` (RTX 5060 Ti / Qwen2.5-0.5B service curves)",
        "- **synthetic**: `burstgpt_scaled_moderate_synthetic_service` (simple synthetic service model)",
        "",
    ]

    rows = []
    policies = sorted(set(
        (list(cal_df["policy"]) if cal_df is not None else []) +
        (list(syn_df["policy"]) if syn_df is not None else [])
    ))

    for policy in policies:
        row = {"policy": policy}
        for tag, df in [("calibrated", cal_df), ("synthetic", syn_df)]:
            if df is not None and policy in df["policy"].values:
                prow = df[df["policy"] == policy].iloc[0]
                row[f"mean_lat_{tag}"] = prow.get("mean_latency", float("nan"))
                row[f"p95_lat_{tag}"] = prow.get("p95_latency", float("nan"))
                row[f"slo_viol_{tag}"] = prow.get("slo_violation_rate", float("nan"))
                row[f"throughput_{tag}"] = prow.get("request_throughput", float("nan"))
                row[f"gpu_util_{tag}"] = prow.get("mean_gpu_utilization", float("nan"))
            else:
                for m in ["mean_lat", "p95_lat", "slo_viol", "throughput", "gpu_util"]:
                    row[f"{m}_{tag}"] = float("nan")
        rows.append(row)

    rank_df = pd.DataFrame(rows)

    # Spearman rank correlations
    lines.append("## Rank Correlations")
    lines.append("")

    try:
        from scipy.stats import spearmanr, kendalltau

        for metric, col_c, col_s in [
            ("mean_latency", "mean_lat_calibrated", "mean_lat_synthetic"),
            ("p95_latency", "p95_lat_calibrated", "p95_lat_synthetic"),
            ("slo_violation_rate", "slo_viol_calibrated", "slo_viol_synthetic"),
        ]:
            if col_c in rank_df.columns and col_s in rank_df.columns:
                valid = rank_df[[col_c, col_s]].dropna()
                if len(valid) >= 3:
                    sp_r, sp_p = spearmanr(valid[col_c], valid[col_s])
                    kt_r, kt_p = kendalltau(valid[col_c], valid[col_s])
                    lines.append(f"**{metric}** — Spearman ρ = {sp_r:.3f} (p={sp_p:.3f}), Kendall τ = {kt_r:.3f} (p={kt_p:.3f})  ")
                else:
                    lines.append(f"**{metric}** — insufficient data (n={len(valid)})")
    except ImportError:
        lines.append("*scipy not available — rank correlations not computed.*")

    lines.append("")
    lines.append("## Per-Policy Table")
    lines.append("")
    cols = ["policy",
            "mean_lat_calibrated", "mean_lat_synthetic",
            "p95_lat_calibrated", "p95_lat_synthetic",
            "slo_viol_calibrated", "slo_viol_synthetic",
            "gpu_util_calibrated", "gpu_util_synthetic"]
    avail = [c for c in cols if c in rank_df.columns]
    lines.append(df_to_md_table(rank_df, avail))
    lines.append("")

    # Policy ranking changes
    lines.append("## Policy Ranking Changes")
    lines.append("")
    if cal_df is not None and "mean_lat_calibrated" in rank_df.columns and "mean_lat_synthetic" in rank_df.columns:
        rank_df["rank_cal"] = rank_df["mean_lat_calibrated"].rank(na_option="bottom")
        rank_df["rank_syn"] = rank_df["mean_lat_synthetic"].rank(na_option="bottom")
        rank_df["rank_delta"] = rank_df["rank_cal"] - rank_df["rank_syn"]
        changed = rank_df[rank_df["rank_delta"].abs() >= 2].sort_values("rank_delta")
        if len(changed) > 0:
            lines.append("Policies with rank shift ≥ 2 positions:")
            for _, r in changed.iterrows():
                lines.append(f"  - `{r['policy']}`: calibrated rank {int(r['rank_cal'])} → synthetic rank {int(r['rank_syn'])} (delta={int(r['rank_delta'])})")
        else:
            lines.append("No policies shifted rank by ≥ 2 positions — calibrated and synthetic models agree on policy ordering.")
    else:
        lines.append("*Insufficient data for ranking comparison.*")
    lines.append("")

    return "\n".join(lines), rank_df


def plot_policy_ranking_heatmap(all_data: dict[str, pd.DataFrame | None]) -> None:
    """Heatmap of mean_latency rank for each policy across experiments."""
    exp_labels = []
    policy_ranks = {}

    for exp_name, (exp_dir_hint, label) in zip(
        [e[0] for e in EXPERIMENTS],
        EXPERIMENTS
    ):
        df = all_data.get(exp_name)
        if df is None or "mean_latency" not in df.columns:
            continue
        short_label = label.split("—")[-1].strip()[:30]
        exp_labels.append(short_label)
        ranked = df.set_index("policy")["mean_latency"].rank()
        for policy, rank in ranked.items():
            if policy not in policy_ranks:
                policy_ranks[policy] = []
            policy_ranks[policy].append(rank)

    if not exp_labels or not policy_ranks:
        return

    policies = sorted(policy_ranks.keys())
    n_exp = len(exp_labels)
    matrix = np.full((len(policies), n_exp), np.nan)
    for i, policy in enumerate(policies):
        ranks = policy_ranks[policy]
        for j, r in enumerate(ranks):
            if j < n_exp:
                matrix[i, j] = r

    fig, ax = plt.subplots(figsize=(max(8, n_exp * 1.5), max(6, len(policies) * 0.4 + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=len(policies))
    ax.set_xticks(range(n_exp))
    ax.set_xticklabels(exp_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies, fontsize=9)
    ax.set_title("Policy Ranking by Mean Latency Across Phase 1.7C Experiments\n(1=best, darker=worse)")
    plt.colorbar(im, ax=ax, label="Rank (1=best)")
    plt.tight_layout()
    out = PLOTS / "policy_ranking_heatmap.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


def plot_latency_comparison(all_data: dict[str, pd.DataFrame | None]) -> None:
    """Bar chart of best-policy mean latency across experiments."""
    labels = []
    best_vals = []

    for exp_name, (_, label) in zip([e[0] for e in EXPERIMENTS], EXPERIMENTS):
        df = all_data.get(exp_name)
        if df is None or "mean_latency" not in df.columns:
            continue
        best = df["mean_latency"].min()
        labels.append(label.split(".")[-1].strip()[:35])
        best_vals.append(best)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), best_vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Best mean latency (s)")
    ax.set_title("Best-Policy Mean Latency Across Phase 1.7C Experiments")
    plt.tight_layout()
    out = PLOTS / "best_mean_latency_by_experiment.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Phase 1.7C consolidated experiment summary (markdown reports, "
            "CSVs, and plots) from per-experiment results/ directories."
        )
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUT,
        help=f"Directory to write summary docs/CSVs/plots into (default: {OUT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    global OUT, PLOTS
    OUT = args.output_dir
    PLOTS = OUT / "plots"
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Phase 1.7C Summary Generator")
    print(f"  Generated: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    all_data = {}
    sections = []
    missing = []

    for exp_name, label in EXPERIMENTS:
        print(f"Loading: {exp_name}")
        df, loc = load_summary(exp_name)
        all_data[exp_name] = df
        section, _ = make_experiment_section(exp_name, label)
        sections.append(section)
        if df is None:
            missing.append(exp_name)
            print(f"  MISSING: {loc}")
        else:
            print(f"  OK: {len(df)} policies from {loc}")

    # Main summary doc
    header = [
        "# Phase 1.7C Experiment Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Experiments",
        "",
        "| # | Experiment | Status |",
        "| --- | --- | --- |",
    ]
    for exp_name, label in EXPERIMENTS:
        status = "MISSING" if exp_name in missing else "COMPLETE"
        header.append(f"| {label.split('.')[0]} | {exp_name} | {status} |")
    header.append("")

    summary_md = "\n".join(header) + "\n\n---\n\n" + "\n\n---\n\n".join(sections)
    summary_path = OUT / "phase17c_experiment_summary.md"
    summary_path.write_text(summary_md)
    print(f"\nWrote: {summary_path}")

    # Noise sensitivity
    print("\nGenerating noise sensitivity analysis...")
    noise_md, noise_df = generate_noise_sensitivity(all_data)
    (OUT / "prediction_noise_sensitivity.md").write_text(noise_md)
    noise_df.to_csv(OUT / "prediction_noise_sensitivity.csv", index=False)
    print(f"  Wrote: {OUT / 'prediction_noise_sensitivity.md'}")
    print(f"  Wrote: {OUT / 'prediction_noise_sensitivity.csv'}")

    # Calibrated vs synthetic
    print("Generating calibrated vs synthetic comparison...")
    cal_df = all_data.get("burstgpt_scaled_moderate_calibrated")
    syn_df = all_data.get("burstgpt_scaled_moderate_synthetic_service")
    cal_syn_md, rank_df = generate_calibrated_vs_synthetic(cal_df, syn_df)
    (OUT / "calibrated_vs_synthetic_comparison.md").write_text(cal_syn_md)
    rank_df.to_csv(OUT / "calibrated_vs_synthetic_rank_correlations.csv", index=False)
    print(f"  Wrote: {OUT / 'calibrated_vs_synthetic_comparison.md'}")
    print(f"  Wrote: {OUT / 'calibrated_vs_synthetic_rank_correlations.csv'}")

    # Plots
    print("Generating cross-experiment plots...")
    plot_policy_ranking_heatmap(all_data)
    plot_latency_comparison(all_data)

    # Also copy per-experiment plots into phase17c/plots/
    for exp_name, _ in EXPERIMENTS:
        exp_dir = RESULTS / exp_name
        if not exp_dir.exists():
            continue
        for png in exp_dir.glob("*/*/figures/*.png"):
            dest = PLOTS / f"{exp_name}__{png.name}"
            shutil.copy2(png, dest)
        for png in exp_dir.glob("*/figures/*.png"):
            dest = PLOTS / f"{exp_name}__{png.name}"
            shutil.copy2(png, dest)

    print(f"\nPlots in: {PLOTS}")

    # Final status
    print(f"\n{'='*60}")
    print(f"Summary complete.")
    print(f"  Experiments found: {len(EXPERIMENTS) - len(missing)}/{len(EXPERIMENTS)}")
    if missing:
        print(f"  Missing: {missing}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
