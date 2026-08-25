#!/usr/bin/env python3
"""Regenerate Figure 1 from frozen joint-workload artifacts (presentation only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
JOINT = ROOT / "experiments" / "joint_multimechanism_generalization_v1"
OUT = Path(__file__).resolve().parents[1] / "figures"

POLICY_ORDER = [
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
]
# Short LNCS-safe labels (match Table 1 short names).
POLICY_LABELS = [
    "Full",
    "Chunked",
    "ESTF",
    "WFS",
    "LLF",
    "KV",
]


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.15",
            "axes.labelcolor": "0.1",
            "xtick.color": "0.1",
            "ytick.color": "0.1",
            "text.color": "0.1",
        }
    )


def load_data() -> tuple[dict, pd.DataFrame, dict, dict]:
    with (JOINT / "winner_summary.json").open() as f:
        winner_summary = json.load(f)
    with (JOINT / "coverage_summary.json").open() as f:
        coverage_summary = json.load(f)
    with (JOINT / "oracle_summary.json").open() as f:
        oracle_summary = json.load(f)
    wide = pd.read_csv(JOINT / "utility_matrix_wide.csv")
    return winner_summary, wide, coverage_summary, oracle_summary


def _panel_title(ax: plt.Axes, text: str) -> None:
    # Title sits ABOVE the axes box (outside the data rectangle) with
    # generous padding, never over bars/lines/grid.
    ax.set_title(text, loc="left", pad=7.0)


def _panel_winner(ax: plt.Axes, winner_summary: dict) -> None:
    """Horizontal bars avoid x-label collisions at LNCS width."""
    counts = [winner_summary["winner_counts"][p] for p in POLICY_ORDER]
    y = np.arange(len(POLICY_LABELS))
    ax.barh(
        y,
        counts,
        color="0.75",
        edgecolor="0.2",
        linewidth=0.8,
        height=0.62,
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(POLICY_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Scenarios won")
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 20, 40, 60])
    ax.grid(axis="x", color="0.85", linewidth=0.6, zorder=0)
    for yi, c in zip(y, counts):
        ax.text(c + 1.5, yi, str(c), va="center", ha="left", fontsize=8, color="0.25")
    _panel_title(ax, "(a)")


def _panel_gain(ax: plt.Axes, wide: pd.DataFrame, oracle_summary: dict) -> None:
    gains = wide["oracle_gain_over_best_fixed"].to_numpy(dtype=float)
    mean_gain = float(oracle_summary["oracle_gain_summary"]["mean"])

    ax.hist(
        gains,
        bins=24,
        range=(0.0, max(0.12, float(gains.max()) * 1.05)),
        color="0.82",
        edgecolor="0.35",
        linewidth=0.6,
        zorder=2,
    )
    line_mean = ax.axvline(mean_gain, color="0.1", linewidth=1.3, linestyle="-", zorder=3)
    line_thresh = ax.axvline(
        0.01, color="0.45", linewidth=1.1, linestyle=(0, (4, 3)), zorder=3
    )
    ax.set_xlabel("VBS gain over SBS (ANWG)")
    ax.set_ylabel("Scenarios")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.grid(axis="y", color="0.85", linewidth=0.6, zorder=0)
    # Compact legend outside the data (upper right, where bar heights are
    # low) instead of printing "mean"/"0.01" text on top of the histogram.
    ax.legend(
        [line_mean, line_thresh],
        [f"mean = {mean_gain:.4f}", "0.01 threshold"],
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.7",
        borderpad=0.4,
        handlelength=1.6,
        handletextpad=0.5,
        labelspacing=0.35,
    )
    _panel_title(ax, "(b)")


def _panel_pressure(ax: plt.Axes, coverage_summary: dict) -> None:
    counts = coverage_summary["mechanism_pressure_counts"]
    xs = np.array(sorted(int(k) for k in counts))
    ys = np.array([counts[str(x)] for x in xs])
    ymax = max(ys) * 1.30
    ax.add_patch(
        Rectangle(
            (1.5, 0),
            width=5.0,
            height=ymax,
            facecolor="0.92",
            edgecolor="none",
            alpha=0.55,
            zorder=1,
        )
    )
    ax.bar(
        xs,
        ys,
        color="0.78",
        edgecolor="0.25",
        linewidth=0.8,
        width=0.72,
        zorder=2,
    )
    ge2 = int(coverage_summary["multi_mechanism_scenarios_ge2"])
    frac = float(coverage_summary["multi_mechanism_fraction_ge2"])
    n_scen = coverage_summary["n_scenarios"]
    ax.set_xlabel("Elevated mechanism pressures per scenario")
    ax.set_ylabel("Scenarios")
    ax.set_xticks(xs)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="0.85", linewidth=0.6, zorder=0)
    # Panel title on the left, shading-explainer annotation on the right,
    # both placed ABOVE the axes (outside the data rectangle) so neither
    # competes with the bars.
    _panel_title(ax, "(c)")
    ax.text(
        1.0,
        1.04,
        rf"shaded $\geq$2 pressures: {ge2}/{n_scen} ({100 * frac:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.3",
        clip_on=False,
    )


def plot_figure(
    winner_summary: dict,
    wide: pd.DataFrame,
    coverage_summary: dict,
    oracle_summary: dict,
) -> plt.Figure:
    # Two-row LNCS layout: (a)|(b) on top, (c) full width below. Extra
    # height and hspace vs. the original draft reserve room for
    # above-axes titles/annotations so nothing sits inside the data area.
    fig = plt.figure(figsize=(6.75, 3.55))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        hspace=0.62,
        wspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    _panel_winner(ax_a, winner_summary)
    _panel_gain(ax_b, wide, oracle_summary)
    _panel_pressure(ax_c, coverage_summary)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.92, bottom=0.105)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview-all",
        action="store_true",
        help="Write PNG preview only (same as final layout).",
    )
    args = parser.parse_args()

    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    winner_summary, wide, coverage_summary, oracle_summary = load_data()
    fig = plot_figure(winner_summary, wide, coverage_summary, oracle_summary)

    pdf_path = OUT / "joint_complementarity.pdf"
    png_path = OUT / "joint_complementarity.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    if args.preview_all:
        print("preview layout written as final files")


if __name__ == "__main__":
    main()
