#!/usr/bin/env python3
"""Figures 4 and 5: regime-level causal characterization and regime opportunity map.

Reads only frozen confirmatory artifacts (via robustness_numbers.compute(), which asserts agreement with the
regime table and the state-level file).  Typography and grayscale-safe encoding match Figure 6.

Encoding (documented in the captions and legends):
* regime numbers 1-5 are shared by Figure 4, Figure 5 and Table 3;
* fill encodes the pressure axis: dark gray = KV capacity, white = active-sequence cap;
* Figure 5 marker AREA is proportional to the regime's mean oracle headroom (explicit size legend).

No regime-level confidence intervals are drawn: two regimes rest on one or two windows, so window-clustered
regime intervals are not defined for them; state and window counts are shown instead.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("robustness_numbers", HERE / "robustness_numbers.py")
rn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rn)

OUT = HERE.parent / "figures"
WL = {"azure_2023_code": "Azure code", "azure_2023_conv": "Azure conv."}
DARK, LIGHT = "0.30", "white"


def order_key(r):  # same order as Table 3
    return (r["workload"] != "azure_2023_code", r["axis"] != "active_sequence_capacity", -int(r["setting"].split("_")[1]) if r["axis"] == "active_sequence_capacity" else 0)


def short(r):
    setting = r["setting"].split("_")[1]
    tail = f"active cap {setting}" if r["axis"] == "active_sequence_capacity" else f"KV {int(setting):,}"
    return f"{WL[r['workload']]}, {tail}"


def face(r):
    return DARK if r["axis"] == "kv_capacity" else LIGHT


def style():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.5})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def figure4(R):
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.7), sharey=True, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    y = np.arange(len(R))[::-1]
    labels = [f"({i + 1}) {short(r)}\n{r['states']} states, {r['windows']} window{'s' if r['windows'] != 1 else ''}" for i, r in enumerate(R)]
    h = 0.62
    for a in ax:
        a.set_axisbelow(True)
        a.grid(axis="y", visible=False)
    ax[0].barh(y, [r["mean_ms"] for r in R], height=h, color=[face(r) for r in R], edgecolor="black", linewidth=0.8)
    for yi, r in zip(y, R):
        ax[0].text(r["mean_ms"] + 0.06, yi, f"{r['mean_ms']:.2f}", va="center", fontsize=7.5)
    ax[0].set_xlim(0, 4.1)
    ax[0].set_xlabel("Mean oracle headroom $\\bar{H}_{\\mathrm{LAT}}$ (ms)")
    ax[0].set_title("(a) Mean headroom")
    ax[1].barh(y, [100 * r["P_B"] for r in R], height=h, color=[face(r) for r in R], edgecolor="black", linewidth=0.8)
    for yi, r in zip(y, R):
        ax[1].text(100 * r["P_B"] + 2, yi, f"{100 * r['P_B']:.1f}", va="center", fontsize=7.5)
    ax[1].set_xlim(0, 118)
    ax[1].set_xticks([0, 25, 50, 75, 100])
    ax[1].set_xlabel("Beneficial share $P(B_{\\mathrm{LAT}}\\mid D)$ (%)")
    ax[1].set_title("(b) Beneficial share")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels)
    handles = [Line2D([], [], marker="s", ls="none", markerfacecolor=DARK, markeredgecolor="black", markersize=6, label="KV capacity"),
               Line2D([], [], marker="s", ls="none", markerfacecolor=LIGHT, markeredgecolor="black", markersize=6, label="Active cap")]
    ax[0].legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, edgecolor="0.7", borderpad=0.6, labelspacing=0.5)
    fig.tight_layout()
    save(fig, "pe_fresh_headroom")


def figure5(R):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    area = 650.0 / max(r["mean_ms"] for r in R)  # marker area (pt^2) per ms of mean headroom
    offsets = {1: (9, -1), 2: (9, 0), 4: (9, 1), 5: (9, 0)}
    for i, r in enumerate(R, start=1):
        x, yv, s = 100 * r["P_D"], 100 * r["P_B"], area * r["mean_ms"]
        ax.scatter(x, yv, s=s, facecolor=face(r), edgecolor="black", linewidth=0.9, zorder=3)
        if i in offsets:
            ax.annotate(str(i), (x, yv), xytext=offsets[i], textcoords="offset points", ha="left", va="center", fontsize=8.5, fontweight="bold")
        else:
            ax.annotate(str(i), (x, yv), ha="center", va="center", fontsize=8.5, fontweight="bold", color="white", zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.006, 1.2)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Disagreement prevalence $P(D)$ (% of SBS decision states)")
    ax.set_ylabel("Beneficial share $P(B_{\\mathrm{LAT}}\\mid D)$ (%)")
    key = [Line2D([], [], marker="o", ls="none", markerfacecolor=face(r), markeredgecolor="black", markersize=6, label=f"{i}  {short(r)}") for i, r in enumerate(R, start=1)]
    leg1 = ax.legend(handles=key, loc="lower right", title="Regime (fill: KV = dark, active cap = white)", frameon=True, framealpha=0.95, edgecolor="0.7",
                     title_fontsize=7, borderpad=0.7, labelspacing=0.5)
    ax.add_artist(leg1)
    # explicit marker-area key drawn in axes coordinates (circle area = area * headroom)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.27, 0.69), 0.38, 0.30, transform=ax.transAxes, facecolor="white", edgecolor="0.7", linewidth=0.8, zorder=2))
    ax.text(0.46, 0.955, "Marker area = mean headroom", transform=ax.transAxes, ha="center", va="center", fontsize=7, zorder=3)
    for xc, v in zip((0.34, 0.46, 0.58), (0.1, 1.0, 3.0)):
        ax.scatter([xc], [0.85], s=area * v, transform=ax.transAxes, facecolor="0.75", edgecolor="black", linewidth=0.9, zorder=3, clip_on=False)
        ax.text(xc, 0.735, f"{v:g} ms", transform=ax.transAxes, ha="center", va="center", fontsize=7, zorder=3)
    fig.tight_layout()
    save(fig, "pe_regime_map")


def main():
    N = rn.compute()
    R = sorted(N["regimes"], key=order_key)
    assert [r["setting"] for r in R] == ["active_8", "active_4", "kv_16000", "active_4", "kv_16000"], [r["setting"] for r in R]
    style()
    figure4(R)
    figure5(R)


if __name__ == "__main__":
    main()
