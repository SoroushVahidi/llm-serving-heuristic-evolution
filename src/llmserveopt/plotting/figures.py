"""
Matplotlib figures for baseline comparison results.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


def _policy_colors(policies: list) -> dict:
    cmap = plt.get_cmap("tab10")
    return {p: cmap(i % 10) for i, p in enumerate(policies)}


def plot_latency_cdf(
    df: pd.DataFrame,
    out_path: Union[str, Path],
    title: str = "Latency CDF by Policy",
) -> None:
    """Plot CDF of mean_latency per policy across seeds."""
    if "mean_latency" not in df.columns:
        return

    policies = df["policy"].unique().tolist()
    colors = _policy_colors(policies)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pol in policies:
        vals = df.loc[df["policy"] == pol, "mean_latency"].dropna().sort_values()
        if len(vals) == 0:
            continue
        y = [(i + 1) / len(vals) for i in range(len(vals))]
        ax.step(vals, y, label=pol, color=colors[pol])

    ax.set_xlabel("Mean Latency (s)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_bar_comparison(
    summary_df: pd.DataFrame,
    metric: str,
    out_path: Union[str, Path],
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Bar chart comparing a single metric across policies."""
    if metric not in summary_df.columns:
        return

    data = summary_df[["policy", metric]].dropna()
    policies = data["policy"].tolist()
    values = data[metric].tolist()
    colors = [_policy_colors(policies)[p] for p in policies]

    fig, ax = plt.subplots(figsize=(max(6, len(policies) * 0.9), 5))
    x = range(len(policies))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} by Policy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_all(
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    summary_df: Optional[pd.DataFrame] = None,
) -> None:
    """Generate all standard comparison figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_latency_cdf(df, out / "latency_cdf.png")

    if summary_df is None:
        summary_df = df.groupby("policy").mean(numeric_only=True).reset_index()

    for metric, label in [
        ("mean_latency", "Mean Latency (s)"),
        ("p95_latency", "P95 Latency (s)"),
        ("slo_violation_rate", "SLO Violation Rate"),
        ("request_throughput", "Throughput (req/s)"),
        ("mean_gpu_utilization", "GPU Utilization"),
        ("mean_active_batch_size", "Mean Active Batch Size"),
    ]:
        if metric in summary_df.columns:
            plot_bar_comparison(
                summary_df,
                metric=metric,
                out_path=out / f"{metric}.png",
                ylabel=label,
            )
