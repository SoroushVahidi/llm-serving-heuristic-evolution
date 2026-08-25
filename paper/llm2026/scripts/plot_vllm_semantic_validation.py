#!/usr/bin/env python3
"""Regenerate Figure 2 from frozen native-vLLM semantic-validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
STAT_PATH = (
    ROOT
    / "experiments"
    / "real_vllm_mechanism_validation_v1"
    / "native_vllm_chunk_budget_semantics_probe_v1"
    / "statistical_summary.json"
)
OUT = Path(__file__).resolve().parents[1] / "figures"


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
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": "0.1",
        }
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    *,
    highlight: bool = False,
    width: float = 0.42,
    height: float = 0.55,
    fontsize: float = 9.0,
) -> None:
    face = "0.95" if not highlight else "0.88"
    edge = "0.25" if not highlight else "0.1"
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
        zorder=3,
    )


def _panel_simulator(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.0,
        0.98,
        "(a) Simulator treatment",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    _box(ax, (0.04, 0.32), "FULL\nmax chunk 65536", height=0.48, width=0.40)
    _box(ax, (0.56, 0.32), "SMALL\nchunk 64", height=0.48, width=0.40)
    ax.annotate(
        "",
        xy=(0.56, 0.22),
        xytext=(0.44, 0.22),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9),
    )
    ax.text(
        0.5,
        0.08,
        "Shared budget 512; chunk size only",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="0.3",
    )


def _panel_native_analogue(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.0,
        0.98,
        "(b) Initial native analogue (bundled)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    _box(
        ax,
        (0.04, 0.32),
        "FULL\nchunking off\nbudget 4096",
        height=0.48,
        width=0.40,
        highlight=True,
        fontsize=8.5,
    )
    _box(
        ax,
        (0.56, 0.32),
        "CHUNKED\nchunking on\nbudget 512",
        height=0.48,
        width=0.40,
        highlight=True,
        fontsize=8.5,
    )
    ax.text(
        0.5,
        0.08,
        "Bundled: chunking + budget change",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="0.3",
    )


def _load_effects_ms() -> list[tuple[str, float, float, float, str]]:
    with STAT_PATH.open() as f:
        stat = json.load(f)
    ll = stat["comparisons"]["late_tight_low_late"]
    hl = stat["comparisons"]["late_tight_high_late"]

    # Top-to-bottom: TTFT then E2E within each regime pair for readability.
    rows = [
        (
            "Low-late late TTFT",
            ll["late_ttft_T4096_minus_T512_s_mean"] * 1000.0,
            ll["late_ttft_T4096_minus_T512_s_ci95"][0] * 1000.0,
            ll["late_ttft_T4096_minus_T512_s_ci95"][1] * 1000.0,
            "o",
        ),
        (
            "High-late late TTFT",
            hl["late_ttft_T4096_minus_T512_s_mean"] * 1000.0,
            hl["late_ttft_T4096_minus_T512_s_ci95"][0] * 1000.0,
            hl["late_ttft_T4096_minus_T512_s_ci95"][1] * 1000.0,
            "o",
        ),
        (
            "Low-late prompt-heavy E2E",
            ll["hog_e2e_T4096_minus_T512_s_mean"] * 1000.0,
            ll["hog_e2e_T4096_minus_T512_s_ci95"][0] * 1000.0,
            ll["hog_e2e_T4096_minus_T512_s_ci95"][1] * 1000.0,
            "s",
        ),
        (
            "High-late prompt-heavy E2E",
            hl["hog_e2e_T4096_minus_T512_s_mean"] * 1000.0,
            hl["hog_e2e_T4096_minus_T512_s_ci95"][0] * 1000.0,
            hl["hog_e2e_T4096_minus_T512_s_ci95"][1] * 1000.0,
            "s",
        ),
    ]
    return rows


def _panel_effects(ax: plt.Axes) -> None:
    rows = _load_effects_ms()
    y = np.arange(len(rows))[::-1]  # first row at top
    ax.axvline(0.0, color="0.45", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    for i, (_label, point, lo, hi, marker) in enumerate(rows):
        yi = y[i]
        ax.errorbar(
            point,
            yi,
            xerr=[[point - lo], [hi - point]],
            fmt=marker,
            color="0.15",
            markerfacecolor="0.95" if marker == "o" else "0.65",
            markeredgecolor="0.15",
            markeredgewidth=0.9,
            markersize=7 if marker == "o" else 6.5,
            elinewidth=1.1,
            capsize=3,
            capthick=0.9,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"Latency difference, T4096 $-$ T512 (ms)")
    ax.grid(axis="x", color="0.88", linewidth=0.6, zorder=0)
    ax.text(
        0.0,
        1.08,
        "(c) Controlled native budget effect (chunked prefill fixed)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )
    xmin = min(r[2] for r in rows) - 5
    xmax = max(r[3] for r in rows) + 5
    ax.set_xlim(xmin, xmax)
    # Sign-convention note inside axes, away from points.
    ax.text(
        0.98,
        0.08,
        "neg. TTFT $\\rightarrow$ T4096 better\npos. E2E $\\rightarrow$ T512 better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.35",
        linespacing=1.2,
    )


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)

    # Vertical stack: schematics on top row, forest plot full-width below
    # so y-labels never collide with panel (b).
    fig = plt.figure(figsize=(6.75, 4.2))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.35],
        hspace=0.58,
        wspace=0.28,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    _panel_simulator(ax_a)
    _panel_native_analogue(ax_b)
    _panel_effects(ax_c)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.95, bottom=0.10)

    pdf_path = OUT / "vllm_semantic_validation.pdf"
    png_path = OUT / "vllm_semantic_validation.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"figsize={fig.get_size_inches()}")


if __name__ == "__main__":
    main()
