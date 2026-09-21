#!/usr/bin/env python3
"""Figure for the post hoc robustness section: skew of oracle headroom and concentration by window.

Reads only frozen confirmatory artifacts and the post hoc robustness outputs (no experiments are run).
Encoding is grayscale-safe: line style / hatching carry the distinctions, not colour.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("robustness_numbers", HERE / "robustness_numbers.py")
rn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rn)
_fs = importlib.util.spec_from_file_location("figstyle", HERE / "figstyle.py")
fs = importlib.util.module_from_spec(_fs)
_fs.loader.exec_module(fs)

LABELS = fs.WORKLOAD


def window_label(wid: str) -> str:
    src, w = wid.split(":")
    return f"{LABELS[src]}\n{w}"


def main() -> None:
    N = rn.compute()  # asserts that every number agrees with the frozen and robustness artifacts
    S, cw = rn.figure_data()
    n = len(S)
    H = np.array([s["H"] for s in S]) * 1e3
    H_wo = np.array([s["H"] for s in S if s["window"] != rn.W11]) * 1e3

    fs.apply()
    fig, ax = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH_IN, 2.75), gridspec_kw={"width_ratios": [1.05, 1]})

    # ---------------------------------------------------------------- (a) ECDF
    def ecdf(a, x, **kw):
        xs = np.sort(x)
        a.step(np.concatenate([[0], xs]), np.concatenate([[0], np.arange(1, len(xs) + 1) / len(xs)]), where="post", **kw)

    ecdf(ax[0], H, color="black", lw=1.8, label=f"All {n} states")
    ecdf(ax[0], H_wo, color="0.45", lw=1.6, ls="--", label=f"Without w11 ({len(H_wo)} states)")
    med, mean = N["dist"]["median_ms"], N["dist"]["mean_ms"]
    ax[0].axvline(med, color="black", lw=1.0, ls=":")
    ax[0].axvline(mean, color="black", lw=1.0, ls="-.")
    ax[0].text(med * 0.92, 0.9, f"median\n{med:.2f} ms", ha="right", va="center", fontsize=7.5)
    ax[0].text(mean * 1.12, 0.5, f"mean\n{mean:.2f} ms", ha="left", va="center", fontsize=7.5)
    ax[0].set_xscale("symlog", linthresh=0.01, linscale=0.4)
    ax[0].set_xlim(0, 15)
    ax[0].set_xticks([0, 0.01, 0.1, 1, 10])
    ax[0].set_xticklabels(["0", "0.01", "0.1", "1", "10"])
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("Oracle headroom $H_{\\mathrm{LAT}}$ (sim. ms)")
    ax[0].set_ylabel("Share of states with $H_{\\mathrm{LAT}} \\leq x$")
    ax[0].set_title("(a) Distribution across states")
    ax[0].legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.7")

    # ---------------------------------------------------------------- (b) concentration by window
    top = [r for r in cw[:3]]
    st_share = [float(r["share_of_states"]) * 100 for r in top]
    hd_share = [float(r["share_of_total_headroom"]) * 100 for r in top]
    labels = [window_label(r["id"]) for r in top]
    n_other = len(cw) - len(top)
    labels.append(f"Other\n{n_other} windows")
    st_share.append(100 - sum(st_share))
    hd_share.append(100 - sum(hd_share))
    assert abs(hd_share[0] - 100 * N["conc"]["w11_headroom_share"]) < 1e-9 and abs(st_share[0] - 100 * N["conc"]["w11_state_share"]) < 1e-9
    y = np.arange(len(labels))[::-1]
    h = 0.36
    ax[1].barh(y + h / 2, st_share, height=h, color="white", edgecolor="black", hatch="///", linewidth=0.8, label="states")
    ax[1].barh(y - h / 2, hd_share, height=h, color="0.25", edgecolor="black", linewidth=0.8, label="headroom")
    for yi, a_, b_ in zip(y, st_share, hd_share):
        ax[1].text(a_ + 1.5, yi + h / 2, f"{a_:.1f}%", va="center", fontsize=7)
        ax[1].text(b_ + 1.5, yi - h / 2, f"{b_:.1f}%", va="center", fontsize=7)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(labels)
    ax[1].set_xlim(0, 118)
    ax[1].set_xlabel("Share (%)")
    ax[1].set_title("(b) Concentration by trace window")
    ax[1].legend(title="Share of", loc="center right", bbox_to_anchor=(1.0, 0.42), title_fontsize=7.5, handlelength=1.4)

    fs.fit(fig)
    fs.save(fig, "pe_robustness")

if __name__ == "__main__":
    main()
