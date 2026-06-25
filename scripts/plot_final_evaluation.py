#!/usr/bin/env python3
"""
Generate final evaluation plots from summarized results.

Usage:
    python scripts/plot_final_evaluation.py \\
        --summary-dir results/phase2a4_2b4_final_eval/final_summary \\
        --eval-dir results/phase2a4_2b4_final_eval/final_heldout_eval \\
        --selector-eval-dir results/phase2a4_2b4_final_eval/selector_evaluation \\
        --output-dir results/phase2a4_2b4_final_eval/plots
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate final evaluation plots")
    p.add_argument("--summary-dir", required=True,
                   help="Directory with final_summary_table.csv, paired_improvements.csv, etc.")
    p.add_argument("--eval-dir", default=None,
                   help="Directory with per-regime flat CSVs")
    p.add_argument("--selector-eval-dir", default=None,
                   help="Selector evaluation directory")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for plots")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be generated without writing files")
    return p.parse_args(argv)


def _safe_float(v) -> Optional[float]:
    try:
        x = float(v)
        return None if math.isnan(x) or math.isinf(x) else x
    except (TypeError, ValueError):
        return None


def _load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _source_color(source: str) -> str:
    return {"baseline": "steelblue", "heuristic": "darkorange",
            "oracle": "red", "selector": "green"}.get(source, "gray")


def plot_summary_table(rows: List[Dict], output_dir: Path, dry_run: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["method"] for r in rows[:15]]
    wgs = [_safe_float(r.get("mean_wg")) or 0.0 for r in rows[:15]]
    sources = [r.get("source", "baseline") for r in rows[:15]]
    colors = [_source_color(s) for s in sources]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), wgs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Priority-Weighted SLO Goodput")
    ax.set_title("Final Test: Method Ranking by Mean WG")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Baseline"),
        Patch(facecolor="darkorange", label="LLM Heuristic"),
        Patch(facecolor="red", label="Oracle (non-deployable)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    out = output_dir / "final_test_goodput_by_method.png"
    if not dry_run:
        fig.savefig(out, dpi=120)
        print(f"  Saved: {out.name}")
    else:
        print(f"  [dry-run] Would save: {out.name}")
    plt.close(fig)


def plot_paired_improvements(rows: List[Dict], output_dir: Path, dry_run: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_plot = [r for r in rows if r.get("source") in ("heuristic", "selector")]
    if not rows_plot:
        print("  No heuristic/selector paired rows to plot")
        return

    names = [r["method"] for r in rows_plot]
    deltas = [_safe_float(r.get("delta_vs_best_fixed")) or 0.0 for r in rows_plot]
    ci_lo = [_safe_float(r.get("delta_ci_lo")) for r in rows_plot]
    ci_hi = [_safe_float(r.get("delta_ci_hi")) for r in rows_plot]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = range(len(names))
    colors = ["darkorange" if d >= 0 else "firebrick" for d in deltas]
    ax.barh(y, deltas, color=colors, alpha=0.7)

    for i, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [i, i], "k-", linewidth=1.5, zorder=5)
            ax.plot([lo, lo], [i - 0.1, i + 0.1], "k-", linewidth=1)
            ax.plot([hi, hi], [i - 0.1, i + 0.1], "k-", linewidth=1)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Delta WG vs Best Fixed Baseline")
    ax.set_title("Paired Improvement vs Best Fixed Baseline (with 95% CI)")
    plt.tight_layout()
    out = output_dir / "final_test_delta_vs_best_fixed.png"
    if not dry_run:
        fig.savefig(out, dpi=120)
        print(f"  Saved: {out.name}")
    else:
        print(f"  [dry-run] Would save: {out.name}")
    plt.close(fig)


def plot_regret_to_oracle(rows: List[Dict], output_dir: Path, dry_run: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_plot = rows[:12]
    if not rows_plot:
        print("  No regret rows to plot")
        return

    names = [r["method"] for r in rows_plot]
    regrets = [_safe_float(r.get("regret")) or 0.0 for r in rows_plot]
    sources = [r.get("source", "baseline") for r in rows_plot]
    colors = [_source_color(s) for s in sources]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(names)), regrets, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Regret (oracle_srtf WG - method WG)")
    ax.set_title("Regret to Oracle (lower = better)")
    plt.tight_layout()
    out = output_dir / "regret_to_oracle.png"
    if not dry_run:
        fig.savefig(out, dpi=120)
        print(f"  Saved: {out.name}")
    else:
        print(f"  [dry-run] Would save: {out.name}")
    plt.close(fig)


def plot_per_regime(eval_dir: Optional[Path], output_dir: Path, dry_run: bool) -> None:
    if eval_dir is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat_rows = []
    for fname in ["candidate_metrics_by_regime_flat.csv", "baseline_metrics_by_regime_flat.csv"]:
        flat_rows.extend(_load_csv(eval_dir / fname))
    if not flat_rows:
        print("  No per-regime flat rows found for per-regime plot")
        return

    regimes = sorted(set(r.get("regime", "") for r in flat_rows))
    methods = sorted(set(r.get("name", "") for r in flat_rows))

    # Build matrix
    data: Dict[str, Dict[str, float]] = {m: {} for m in methods}
    for row in flat_rows:
        name = row.get("name", "")
        regime = row.get("regime", "")
        wg = _safe_float(row.get("priority_weighted_slo_goodput"))
        if wg is not None:
            data[name][regime] = wg

    # Only plot top methods + baselines of interest
    interesting = []
    for m in methods:
        vals = [data[m].get(r) for r in regimes if data[m].get(r) is not None]
        if vals:
            interesting.append((sum(vals) / len(vals), m))
    interesting.sort(reverse=True)
    plot_methods = [m for _, m in interesting[:12]]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(regimes))
    for i, method in enumerate(plot_methods):
        vals = [data[method].get(r) for r in regimes]
        vals_clean = [v if v is not None else 0.0 for v in vals]
        ax.plot(x, vals_clean, marker="o", linewidth=1, markersize=4, label=method[:25])

    ax.set_xticks(list(x))
    ax.set_xticklabels(regimes, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Priority-Weighted SLO Goodput")
    ax.set_title("Method Performance Across Test Regimes")
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    plt.tight_layout()
    out = output_dir / "final_test_goodput_by_method_per_regime.png"
    if not dry_run:
        fig.savefig(out, dpi=120)
        print(f"  Saved: {out.name}")
    else:
        print(f"  [dry-run] Would save: {out.name}")
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary_dir = Path(args.summary_dir)
    output_dir = Path(args.output_dir)
    eval_dir = Path(args.eval_dir) if args.eval_dir else None

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
    except ImportError:
        print("matplotlib not available — skipping plots")
        return 0

    print(f"Generating plots into: {output_dir}")

    summary_rows = _load_csv(summary_dir / "final_summary_table.csv")
    paired_rows = _load_csv(summary_dir / "paired_improvements.csv")
    regret_rows = _load_csv(summary_dir / "regret_to_oracle.csv")

    if summary_rows:
        plot_summary_table(summary_rows, output_dir, args.dry_run)
    else:
        print("  No summary rows — skipping final_test_goodput_by_method.png")

    if paired_rows:
        plot_paired_improvements(paired_rows, output_dir, args.dry_run)
    else:
        print("  No paired rows — skipping delta plot")

    if regret_rows:
        plot_regret_to_oracle(regret_rows, output_dir, args.dry_run)
    else:
        print("  No regret rows — skipping regret plot")

    plot_per_regime(eval_dir, output_dir, args.dry_run)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
