#!/usr/bin/env python3
"""
Generate plots for Phase 2B.3 LLM heuristic search results.

Usage:
    python scripts/plot_search_results.py \\
        --evaluation-dir results/phase2b3_llm_search/evaluation_train_validation \\
        --output-dir     results/phase2b3_llm_search/plots
"""
import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _read_csv(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _safe_float(v):
    try:
        if v is None or v == "" or v == "None":
            return float("nan")
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def plot_validation_leaderboard(rows, output_path: Path) -> None:
    if not HAS_MPL:
        return
    h_rows = [r for r in rows if r.get("source") == "heuristic"][:10]
    b_rows = [r for r in rows if r.get("source") == "baseline"][:5]
    if not h_rows and not b_rows:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    all_rows = h_rows + b_rows
    names = [r["name"][:30] for r in all_rows]
    wgs = [_safe_float(r.get("val_mean_wg")) for r in all_rows]
    colors = ["steelblue" if r.get("source") == "heuristic" else "coral" for r in all_rows]

    y_pos = range(len(names))
    ax.barh(y_pos, wgs, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Val priority_weighted_slo_goodput")
    ax.set_title("Validation Leaderboard (Phase 2B.3)")
    ax.axvline(x=min(w for w in wgs if not math.isnan(w)) - 0.01, color="gray", linestyle="--", alpha=0.3)

    patches = [
        mpatches.Patch(color="steelblue", label="LLM heuristic"),
        mpatches.Patch(color="coral", label="Baseline"),
    ]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_train_vs_validation(rows, output_path: Path) -> None:
    if not HAS_MPL:
        return
    h_rows = [r for r in rows if r.get("source") == "heuristic"]
    if not h_rows:
        return

    train_wgs = [_safe_float(r.get("train_mean_wg")) for r in h_rows]
    val_wgs = [_safe_float(r.get("val_mean_wg")) for r in h_rows]
    names = [r["name"][:20] for r in h_rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    valid = [(t, v, n) for t, v, n in zip(train_wgs, val_wgs, names)
             if not math.isnan(t) and not math.isnan(v)]
    if not valid:
        plt.close()
        return
    ts, vs, ns = zip(*valid)
    ax.scatter(ts, vs, alpha=0.7, s=60)
    for x, y, name in zip(ts, vs, ns):
        ax.annotate(name, (x, y), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    lo = min(min(ts), min(vs)) - 0.01
    hi = max(max(ts), max(vs)) + 0.01
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5, label="train=val line")
    ax.set_xlabel("Train mean WG")
    ax.set_ylabel("Val mean WG")
    ax.set_title("Train vs. Validation Goodput (LLM heuristics)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_per_regime_heatmap(by_regime_rows, regime_names, output_path: Path) -> None:
    if not HAS_MPL:
        return
    try:
        import numpy as np
    except ImportError:
        return

    h_names_seen = {}
    for row in by_regime_rows:
        if row.get("source") == "heuristic":
            name = row.get("name", "")
            rank = int(row.get("rank", 999))
            if name not in h_names_seen or rank < h_names_seen[name]:
                h_names_seen[name] = rank
    top_names = sorted(h_names_seen, key=lambda n: h_names_seen[n])[:10]

    data = {}
    for row in by_regime_rows:
        name = row.get("name", "")
        if name in top_names:
            for rn in regime_names:
                v = _safe_float(row.get(rn))
                data.setdefault(name, {})[rn] = v

    if not data:
        return
    mat = np.array([[data.get(n, {}).get(rn, float("nan")) for rn in regime_names]
                    for n in top_names], dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(regime_names) * 1.5), max(4, len(top_names) * 0.6)))
    vmin = float(np.nanmin(mat)) if not np.all(np.isnan(mat)) else 0
    vmax = float(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 1
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(regime_names)))
    ax.set_xticklabels([r.replace("train_", "T:").replace("val_", "V:") for r in regime_names],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels([n[:25] for n in top_names], fontsize=7)
    plt.colorbar(im, ax=ax, label="WG")
    ax.set_title("Per-Regime WG (top LLM heuristics)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_verifier_repair_counts(index_path: Path, output_path: Path) -> None:
    if not HAS_MPL or not index_path.exists():
        return
    rows = _read_csv(index_path)
    ok_first = sum(1 for r in rows if r.get("verification_ok") == "True" and int(r.get("repair_attempts", 0)) == 0)
    repaired = sum(1 for r in rows if r.get("verification_ok") == "True" and int(r.get("repair_attempts", 0)) > 0)
    failed = sum(1 for r in rows if r.get("verification_ok") != "True")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["OK (first pass)", "Repaired OK", "Failed"],
                  [ok_first, repaired, failed],
                  color=["steelblue", "gold", "salmon"])
    for bar, val in zip(bars, [ok_first, repaired, failed]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Verification & Repair Outcomes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  Saved: {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--candidates-dir", help="For index.csv; defaults to evaluation-dir's parent candidates_main")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    eval_dir = Path(args.evaluation_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_MPL:
        print("[WARN] matplotlib not available — no plots generated")
        return

    candidates_dir = Path(args.candidates_dir) if args.candidates_dir else eval_dir.parent / "candidates_main"

    # Load data
    overall_rows = _read_csv(eval_dir / "ranking_overall.csv")
    by_regime_rows = _read_csv(eval_dir / "candidate_metrics_by_regime.csv")

    # Detect regime names from by_regime columns
    regime_names = [c for c in (by_regime_rows[0].keys() if by_regime_rows else [])
                    if c not in ("rank", "name", "source")]

    print(f"Generating plots → {out_dir}")
    plot_validation_leaderboard(overall_rows, out_dir / "validation_leaderboard.png")
    plot_train_vs_validation(overall_rows, out_dir / "train_vs_validation_goodput.png")
    if regime_names:
        plot_per_regime_heatmap(by_regime_rows, regime_names, out_dir / "candidate_vs_best_fixed_by_regime.png")
    plot_verifier_repair_counts(candidates_dir / "index.csv", out_dir / "verifier_failure_counts.png")


if __name__ == "__main__":
    main()
