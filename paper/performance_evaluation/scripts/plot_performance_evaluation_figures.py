#!/usr/bin/env python3
"""Figures 1-3 of the Performance Evaluation manuscript.

* Figure 1 (pe_pipeline): the staged evidence chain (conceptual; no data).
* Figure 2 (pe_disagreement_rates): native replay versus selected pressure conditions.
* Figure 3 (pe_pressure_transition): Azure-code disagreement prevalence as the active-sequence cap tightens.

Figures 2 and 3 read the frozen Phase A / Phase B artifacts directly (no value is typed in by hand).  All figures
share the typography and grayscale-safe encoding of Figures 4-6 (see figstyle.py).
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("figstyle", HERE / "figstyle.py")
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)

ROOT = HERE.parents[2]
PHASE_A = ROOT / "experiments/industry_realism_action_opportunity_phase_a_v1/PHASE_A_WORKLOAD_SUMMARY_V1.csv"
PHASE_B = ROOT / "experiments/industry_realism_action_opportunity_phase_b_v2/PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv"


def _rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _pct(x):
    return "0" if x == 0 else f"{x:.3g}"


# ---------------------------------------------------------------------------------------------- Figure 1
STAGES = [  # (label, box width in inches); the fifth stage is outside the measurement
    ("Resource\npressure", 0.86),
    ("Action\ndisagreement\n$P(D)$", 0.98),
    ("Beneficial\ndisagreement\n$P(B_{\\mathrm{LAT}}\\mid D)$", 1.10),
    ("Headroom\nmagnitude\n$H_{\\mathrm{LAT}}$", 0.92),
    ("Predictability,\ndeployability", 1.08),
]


def figure1():
    W, H = fs.TEXT_WIDTH_IN, 1.45
    fig, ax = plt.subplots(figsize=(W, H))
    pad = 0.03  # keeps the outer box edges inside the canvas
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.grid(False)
    gap = (W - 2 * pad - sum(w for _, w in STAGES)) / (len(STAGES) - 1)
    bh, yc = 0.80, 0.90
    x, edges = pad, []
    for i, (label, w) in enumerate(STAGES):
        measured = i < 4
        ax.add_patch(FancyBboxPatch((x, yc - bh / 2), w, bh, boxstyle="round,pad=0,rounding_size=0.07", facecolor="0.93" if measured else "white",
                                    edgecolor="black", linewidth=0.9 if measured else 1.0, linestyle="-" if measured else (0, (3, 2))))
        ax.text(x + w / 2, yc, label, ha="center", va="center", fontsize=7.5, linespacing=1.25)
        edges.append((x, x + w))
        x += w + gap
    for (_, r), (l, _) in zip(edges[:-1], edges[1:]):
        ax.add_patch(FancyArrowPatch((r + 0.02, yc), (l - 0.02, yc), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color="black", shrinkA=0, shrinkB=0))
    # brackets: what the study measures
    yb = 0.32
    ax.plot([edges[0][0], edges[3][1]], [yb, yb], color="black", lw=0.9, solid_capstyle="butt")
    ax.plot([edges[4][0], edges[4][1]], [yb, yb], color="black", lw=0.9, ls=(0, (3, 2)), solid_capstyle="butt")
    for xx in (edges[0][0], edges[3][1]):
        ax.plot([xx, xx], [yb, yb + 0.07], color="black", lw=0.9)
    for xx in edges[4]:
        ax.plot([xx, xx], [yb, yb + 0.07], color="black", lw=0.9)
    ax.text((edges[0][0] + edges[3][1]) / 2, yb - 0.11, "Measured in this study", ha="center", va="top", fontsize=7.5)
    ax.text(sum(edges[4]) / 2, yb - 0.11, "Not measured here", ha="center", va="top", fontsize=7.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fs.save(fig, "pe_pipeline")


# ---------------------------------------------------------------------------------------------- Figure 2
def figure2():
    native = {r["source_dataset"]: r for r in _rows(PHASE_A)}
    pressure = {(r["source_dataset"], r["axis"], int(float(r["axis_value"]))): r for r in _rows(PHASE_B)}
    spec = [  # (row, y-tick label, fill)
        (native["azure_2023_code"], "Azure code", fs.LIGHT),
        (native["azure_2023_conv"], "Azure conversation", fs.LIGHT),
        (native["burstgpt"], "BurstGPT", fs.LIGHT),
        (pressure[("azure_2023_code", "active_sequence_capacity", 4)], "Azure code, active cap 4", fs.LIGHT),
        (pressure[("azure_2023_code", "kv_capacity", 16000)], "Azure code, KV 16,000", fs.DARK),
        (pressure[("azure_2023_conv", "active_sequence_capacity", 4)], "Azure conv., active cap 4", fs.LIGHT),
        (pressure[("burstgpt", "active_sequence_capacity", 8)], "BurstGPT, active cap 8", fs.LIGHT),
    ]
    # the artifacts' own counts must reproduce the rate that is plotted
    for r, _, _ in spec:
        assert abs(int(r["true_canonical_disagreement_states"]) / int(r["sbs_decision_states"]) - float(r["disagreement_rate"])) < 1e-12
    assert [int(r["sbs_decision_states"]) for r, _, _ in spec[:3]] == [99992, 461985, 440461]
    assert all(int(r["true_canonical_disagreement_states"]) == 0 for r, _, _ in spec[:3])

    fig, ax = fs.new_figure(3.05)
    ys = [7.0, 6.0, 5.0, 3.4, 2.4, 1.4, 0.4]  # room for the two group headers at y = 8.0 and 4.4
    vals = [100 * float(r["disagreement_rate"]) for r, _, _ in spec]
    ax.barh(ys, vals, height=0.62, color=[f for _, _, f in spec], edgecolor="black", linewidth=0.8, zorder=3)
    for y, v in zip(ys, vals):
        ax.text(v + 0.012, y, _pct(v), va="center", ha="left", fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lab}\n{int(r['true_canonical_disagreement_states']):,} of {int(r['sbs_decision_states']):,} states" for r, lab, _ in spec])
    ax.set_ylim(-0.55, 8.15)
    ax.set_xlim(0, 0.72)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("Disagreement prevalence $P(D)$ (% of SBS decision states)")
    for y, text in ((7.95, "Native replay, abundant resources"), (4.35, "Binding pressure conditions")):
        ax.text(0.995, y, text, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, fontstyle="italic")
    handles = [Line2D([], [], marker="s", ls="none", markerfacecolor=fs.DARK, markeredgecolor="black", markersize=6, label="KV capacity"),
               Line2D([], [], marker="s", ls="none", markerfacecolor=fs.LIGHT, markeredgecolor="black", markersize=6, label="Active-sequence cap")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    fs.fit(fig)
    fs.save(fig, "pe_disagreement_rates")


# ---------------------------------------------------------------------------------------------- Figure 3
def figure3():
    rows = [r for r in _rows(PHASE_B) if r["source_dataset"] == "azure_2023_code" and r["axis"] == "active_sequence_capacity"]
    rows.sort(key=lambda r: int(r["pressure_order"]))
    caps = [int(float(r["axis_value"])) for r in rows]
    assert caps == sorted(caps, reverse=True) == [512, 64, 32, 16, 8, 4], caps  # artifact order = increasing pressure
    x = np.array(caps, dtype=float)
    y = np.array([100 * float(r["disagreement_rate"]) for r in rows])
    states = [(int(r["true_canonical_disagreement_states"]), int(r["sbs_decision_states"])) for r in rows]

    fig, ax = fs.new_figure(2.55, width_in=4.7)
    ax.plot(x, y, color="black", lw=1.6, zorder=2)
    ax.scatter(x, y, s=34, facecolor=fs.LIGHT, edgecolor="black", linewidth=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(700, 3.1)  # reversed on purpose: pressure (a tighter cap) increases from left to right, as in the artifact's pressure_order
    ax.set_xticks(caps)
    ax.set_xticklabels([str(c) for c in caps])
    ax.minorticks_off()
    ax.set_ylim(-0.004, 0.118)
    ax.set_yticks([0, 0.025, 0.05, 0.075, 0.10])
    ax.set_yticklabels(["0", "0.025", "0.05", "0.075", "0.10"])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Active-sequence cap (concurrent sequences); tighter caps to the right")
    ax.set_ylabel("Disagreement prevalence $P(D)$ (%)")
    ax.annotate(f"{_pct(y[4])}%\n({states[4][0]} of {states[4][1]:,} states)", xy=(x[4], y[4]), xytext=(-14, 24), textcoords="offset points",
                ha="right", va="bottom", fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.7, color="0.3", shrinkA=0, shrinkB=4))
    ax.annotate(f"{_pct(y[5])}%\n({states[5][0]} of {states[5][1]:,} states)", xy=(x[5], y[5]), xytext=(-7, -3), textcoords="offset points",
                ha="right", va="center", fontsize=7.5)
    fs.fit(fig)
    fs.save(fig, "pe_pressure_transition")


def main():
    fs.apply()
    figure1()
    figure2()
    figure3()


if __name__ == "__main__":
    main()
