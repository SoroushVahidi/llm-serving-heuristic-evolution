"""Generate the Performance Evaluation manuscript figures from frozen result artifacts."""
from pathlib import Path
import csv
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "paper" / "performance_evaluation" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close()

# Figure 1: the paper's measurement chain.
fig, ax = plt.subplots(figsize=(8.2, 1.55))
ax.axis("off")
labels = ["Resource\npressure", "Canonical\naction opportunity\nP(D)", "Beneficial\nopportunity\nP(B|D)", "Headroom\nmagnitude", "Predictability /\ndeployability"]
colors = ["#dceaf7", "#e4f2df", "#fff0c9", "#f8dfdf", "#e8e1f4"]
xs = np.linspace(0.1, 0.9, len(labels))
for x, label, color in zip(xs, labels, colors):
    ax.text(x, 0.5, label, ha="center", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="#444"))
for a, b in zip(xs[:-1], xs[1:]):
    ax.annotate("", xy=(b - 0.075, 0.5), xytext=(a + 0.075, 0.5),
                xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.4))
ax.set_title("A staged test of whether adaptive scheduling is actionable")
save("pe_pipeline")

# Figure 2: native null support versus pressure-induced support.
phase_a = {"Azure code": 0.0, "Azure conversation": 0.0, "BurstGPT": 0.0}
phase_b = {"Azure code\nactive_4": 0.0009987, "Azure code\nKV 16k": 0.0059532,
           "Azure conv.\nactive_4": 0.0000670, "BurstGPT\nactive_8": 0.0003}
fig, ax = plt.subplots(figsize=(8.0, 3.8))
names = list(phase_a) + list(phase_b)
vals = list(phase_a.values()) + list(phase_b.values())
colors = ["#9aa7b2"] * 3 + ["#2d6a9f"] * 4
ax.bar(np.arange(len(names)), vals, color=colors)
ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
ax.set_ylabel("Canonical disagreement rate")
ax.set_title("Native replay is action-null; binding pressure creates opportunity")
ax.set_ylim(0, max(vals) * 1.25)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
save("pe_disagreement_rates")

# Figure 3: pressure axis transition for the clearest Azure-code windows.
rows = list(csv.DictReader((ROOT / "experiments/industry_realism_action_opportunity_phase_b_v2/PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv").open()))
active = [r for r in rows if r["source_dataset"] == "azure_2023_code" and r["axis"] == "active_sequence_capacity"]
active.sort(key=lambda r: float(r["axis_value"]), reverse=True)
x = [float(r["axis_value"]) for r in active]
y = [float(r["disagreement_rate"]) for r in active]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(x, y, marker="o", color="#b33a3a", lw=2)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("Active-sequence cap (larger to smaller)")
ax.set_ylabel("Canonical disagreement rate")
ax.set_title("Azure-code active-cap transition")
ax.grid(axis="y", alpha=0.3)
save("pe_pressure_transition")

# Figure 4: fresh causal headroom by selected regime.
fresh = list(csv.DictReader((ROOT / "experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_LATENCY_WORKLOAD_REGIME_V1.csv").open()))
labels = [f"{r['source_dataset'].replace('azure_2023_', 'Azure ')}\n{r['condition_id']}" for r in fresh]
means = [float(r["mean_oracle_headroom"]) * 1000 for r in fresh]
benefit = [float(r["P_B_given_D"]) * 100 for r in fresh]
fig, ax1 = plt.subplots(figsize=(7.5, 3.2))
idx = np.arange(len(labels))
ax1.bar(idx, means, color="#2f7f5f", alpha=0.85)
ax1.set_ylabel("Mean oracle headroom (ms)")
ax1.set_xticks(idx, labels, rotation=25, ha="right")
ax1.axhline(0, color="#333", lw=0.7)
ax2 = ax1.twinx()
ax2.plot(idx, benefit, color="#b36b00", marker="o", lw=1.8)
ax2.set_ylabel("P(B_LAT | D) (%)")
ax2.set_ylim(0, 105)
ax1.set_title("Fresh untouched-window causal headroom")
save("pe_fresh_headroom")

# Figure 5: practitioner map, opportunity prevalence versus conditional benefit.
fig, ax = plt.subplots(figsize=(6.8, 4.5))
for r in fresh:
    pd = float(r["P_D"]) * 100
    pb = float(r["P_B_given_D"]) * 100
    size = max(35, float(r["mean_oracle_headroom"]) * 100000)
    label = f"{r['source_dataset'].replace('azure_2023_', 'Azure ')}\n{r['condition_id']}"
    ax.scatter(pd, pb, s=size, alpha=0.8, label=label)
    x_offset = -8 if pd > 0.3 else 8
    y_offset = -12 if pb > 90 else 8
    ha = "right" if pd > 0.3 else "left"
    ax.annotate(label, (pd, pb), xytext=(x_offset, y_offset), textcoords="offset points", fontsize=9, ha=ha)
ax.set_xscale("log")
ax.set_xlabel("P(D) among all SBS decision states (%)")
ax.set_ylabel("P(B_LAT | D) (%)")
ax.set_ylim(0, 115)
ax.set_xlim(0.005, 1.5)
ax.set_title("Opportunity is a product of prevalence and conditional value")
ax.grid(alpha=0.25)
save("pe_regime_map")
