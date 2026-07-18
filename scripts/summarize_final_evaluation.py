#!/usr/bin/env python3
"""
Compute final evaluation summary with bootstrap confidence intervals and
regret-to-oracle reporting.

Reads per-regime metrics from the final held-out evaluation and the
selector evaluation, then produces summary tables and a markdown report.

Usage:
    python scripts/summarize_final_evaluation.py \\
        --eval-dir results/phase2a4_2b4_final_eval/final_heldout_eval \\
        --selector-eval-dir results/phase2a4_2b4_final_eval/selector_evaluation \\
        --output-dir results/phase2a4_2b4_final_eval/final_summary
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute final evaluation summary with CI and regret"
    )
    p.add_argument("--eval-dir", required=True,
                   help="Directory with candidate_metrics_by_regime_flat.csv and "
                        "baseline_metrics_by_regime_flat.csv")
    p.add_argument("--selector-eval-dir", default=None,
                   help="Directory with selector evaluation summary.csv")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for summary tables and report")
    p.add_argument("--n-bootstrap", type=int, default=2000,
                   help="Bootstrap replicates for CI (default: 2000)")
    p.add_argument("--ci-level", type=float, default=0.95,
                   help="Confidence level (default: 0.95)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_flat_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _safe_float(v) -> Optional[float]:
    try:
        x = float(v)
        return None if math.isnan(x) or math.isinf(x) else x
    except (TypeError, ValueError):
        return None


def load_per_regime_metrics(eval_dir: Path) -> List[Dict]:
    """Load per-regime metrics. candidate_metrics_by_regime_flat.csv contains
    heuristics + baselines + oracle, so load only that file to avoid double-counting."""
    all_csv = eval_dir / "candidate_metrics_by_regime_flat.csv"
    if all_csv.exists():
        return _load_flat_csv(all_csv)
    # Fallback: merge without duplicating
    rows = []
    seen = set()
    for fname in ["baseline_metrics_by_regime_flat.csv"]:
        for row in _load_flat_csv(eval_dir / fname):
            key = (row.get("regime"), row.get("name"), row.get("source"))
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def load_selector_summary(selector_eval_dir: Path) -> Optional[Dict]:
    """Load selector evaluation summary.csv (test split row for each model)."""
    path = selector_eval_dir / "summary.csv"
    if not path.exists():
        # Try nested path
        path = selector_eval_dir / "evaluation" / "summary.csv"
    if not path.exists():
        return None
    summary = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "")
            model = row.get("model", "")
            if split == "test":
                summary[model] = {
                    "selected_mean_wg": _safe_float(row.get("selected_mean_wg")),
                    "accuracy": _safe_float(row.get("accuracy")),
                    "macro_f1": _safe_float(row.get("macro_f1")),
                    "regret_to_window_best": _safe_float(row.get("regret_to_window_best")),
                    "diff_vs_best_fixed": _safe_float(row.get("diff_vs_best_fixed")),
                    "n": row.get("n"),
                }
    return summary if summary else None


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    n: int = 2000,
    level: float = 0.95,
    rng: random.Random = None,
) -> Tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    if not values:
        nan = float("nan")
        return nan, nan, nan
    rng = rng or random.Random(42)
    mean = sum(values) / len(values)
    boot_means = []
    k = len(values)
    for _ in range(n):
        sample = [rng.choice(values) for _ in range(k)]
        boot_means.append(sum(sample) / k)
    boot_means.sort()
    alpha = 1.0 - level
    lo_idx = int(math.floor(alpha / 2 * n))
    hi_idx = int(math.ceil((1 - alpha / 2) * n)) - 1
    lo_idx = max(0, min(lo_idx, len(boot_means) - 1))
    hi_idx = max(0, min(hi_idx, len(boot_means) - 1))
    return mean, boot_means[lo_idx], boot_means[hi_idx]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_method(rows: List[Dict]) -> Dict[str, Dict]:
    """Group per-regime rows by (name, source) and collect WG values."""
    by_method: Dict[str, Dict] = {}
    for row in rows:
        name = row.get("name", "")
        source = row.get("source", "")
        if not name:
            continue
        wg = _safe_float(row.get("priority_weighted_slo_goodput"))
        vr = _safe_float(row.get("slo_violation_rate"))
        ttft = _safe_float(row.get("p95_ttft"))
        lat = _safe_float(row.get("p95_latency"))
        regime = row.get("regime", "")
        if name not in by_method:
            by_method[name] = {
                "source": source,
                "wg_values": [],
                "vr_values": [],
                "ttft_values": [],
                "lat_values": [],
                "regimes": [],
            }
        if wg is not None:
            by_method[name]["wg_values"].append(wg)
        if vr is not None:
            by_method[name]["vr_values"].append(vr)
        if ttft is not None:
            by_method[name]["ttft_values"].append(ttft)
        if lat is not None:
            by_method[name]["lat_values"].append(lat)
        by_method[name]["regimes"].append(regime)
    return by_method


def compute_best_fixed_per_regime(rows: List[Dict]) -> Dict[str, float]:
    """Find the best baseline WG per regime."""
    best: Dict[str, float] = {}
    for row in rows:
        if row.get("source") not in ("baseline",):
            continue
        regime = row.get("regime", "")
        wg = _safe_float(row.get("priority_weighted_slo_goodput"))
        if wg is not None:
            best[regime] = max(best.get(regime, float("-inf")), wg)
    return best


def compute_oracle_per_regime(rows: List[Dict]) -> Dict[str, float]:
    """Find oracle_srtf WG per regime (may be NaN if not included)."""
    oracle: Dict[str, float] = {}
    for row in rows:
        if row.get("name") == "oracle_srtf":
            regime = row.get("regime", "")
            wg = _safe_float(row.get("priority_weighted_slo_goodput"))
            if wg is not None:
                oracle[regime] = wg
    return oracle


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v, digits=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.{digits}f}"


def _ci_str(lo, hi):
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def _interpretation(delta: Optional[float], ci_lo: Optional[float], ci_hi: Optional[float]) -> str:
    if delta is None or math.isnan(delta):
        return "insufficient data"
    if ci_lo is None or ci_hi is None:
        return f"delta={_fmt(delta)} (CI unavailable)"
    if ci_lo < 0 < ci_hi:
        return f"delta={_fmt(delta)}, CI crosses zero — not statistically clear"
    if abs(delta) < 0.01:
        return f"delta={_fmt(delta)}, marginal (< 1 pp)"
    if abs(delta) < 0.02:
        return f"delta={_fmt(delta)}, marginal (< 2 pp)"
    direction = "improvement" if delta > 0 else "regression"
    return f"delta={_fmt(delta)}, {direction} (CI does not cross zero)"


def build_final_summary_md(
    by_method: Dict[str, Dict],
    best_fixed_per_regime: Dict[str, float],
    oracle_per_regime: Dict[str, float],
    selector_summary: Optional[Dict],
    n_bootstrap: int,
    ci_level: float,
    rng: random.Random,
) -> Tuple[str, List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Build the final summary markdown and return
    (md, summary_rows, paired_rows, ci_rows, regret_rows).
    """
    # Compute per-method stats
    summary_rows = []
    ci_rows = []
    for name, d in sorted(by_method.items()):
        wg_vals = d["wg_values"]
        mean_wg, ci_lo, ci_hi = bootstrap_ci(wg_vals, n=n_bootstrap, level=ci_level, rng=rng)
        vr_mean = (sum(d["vr_values"]) / len(d["vr_values"])) if d["vr_values"] else float("nan")
        summary_rows.append({
            "method": name,
            "source": d["source"],
            "n_regimes": len(wg_vals),
            "mean_wg": mean_wg,
            "vr_mean": vr_mean,
        })
        ci_rows.append({
            "method": name,
            "source": d["source"],
            "mean_wg": mean_wg,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_level": ci_level,
            "n_regimes": len(wg_vals),
        })
    summary_rows.sort(key=lambda r: -(r["mean_wg"] if not math.isnan(r["mean_wg"]) else -999))

    # Best fixed baseline across regimes
    all_baseline_wg: List[float] = []
    for name, d in by_method.items():
        if d["source"] == "baseline" and name != "oracle_srtf":
            all_baseline_wg.extend(d["wg_values"])
    # Best fixed = row with highest mean WG among baselines (excluding oracle)
    best_fixed_name = None
    best_fixed_mean = float("-inf")
    for name, d in by_method.items():
        if d["source"] == "baseline" and name != "oracle_srtf":
            if d["wg_values"]:
                m = sum(d["wg_values"]) / len(d["wg_values"])
                if m > best_fixed_mean:
                    best_fixed_mean = m
                    best_fixed_name = name

    # Oracle mean (if present)
    oracle_wg = []
    if "oracle_srtf" in by_method:
        oracle_wg = by_method["oracle_srtf"]["wg_values"]
    oracle_mean = sum(oracle_wg) / len(oracle_wg) if oracle_wg else None

    # Paired improvements vs best fixed
    paired_rows = []
    if best_fixed_name and by_method.get(best_fixed_name):
        bf_vals = by_method[best_fixed_name]["wg_values"]
        bf_mean = sum(bf_vals) / len(bf_vals) if bf_vals else None
        for name, d in sorted(by_method.items()):
            if d["source"] in ("baseline",) and name != "oracle_srtf":
                continue
            if name == "oracle_srtf":
                continue
            wg_vals = d["wg_values"]
            if not wg_vals or bf_mean is None:
                continue
            m = sum(wg_vals) / len(wg_vals)
            delta = m - bf_mean
            # Paired bootstrap on delta (only possible when regime counts match)
            if bf_vals and len(wg_vals) == len(bf_vals) and len(wg_vals) >= 2:
                deltas = [a - b for a, b in zip(wg_vals, bf_vals)]
                d_mean, d_lo, d_hi = bootstrap_ci(deltas, n=n_bootstrap, level=ci_level, rng=rng)
            elif len(wg_vals) >= 2:
                # Unpaired: bootstrap on raw values, CI less meaningful
                d_lo, d_hi = float("nan"), float("nan")
            else:
                d_lo, d_hi = float("nan"), float("nan")
            paired_rows.append({
                "method": name,
                "source": d["source"],
                "mean_wg": m,
                "delta_vs_best_fixed": delta,
                "delta_ci_lo": d_lo,
                "delta_ci_hi": d_hi,
                "interpretation": _interpretation(delta, d_lo, d_hi),
            })

    # Regret to oracle
    regret_rows = []
    if oracle_wg:
        oracle_m = oracle_mean
        for name, d in sorted(by_method.items()):
            if name == "oracle_srtf":
                continue
            wg_vals = d["wg_values"]
            if not wg_vals or oracle_m is None:
                continue
            m = sum(wg_vals) / len(wg_vals)
            regret = oracle_m - m   # positive = worse than oracle
            regret_rows.append({
                "method": name,
                "source": d["source"],
                "mean_wg": m,
                "oracle_mean_wg": oracle_m,
                "regret": regret,
                "regret_pct": regret / oracle_m * 100 if oracle_m else float("nan"),
            })
        regret_rows.sort(key=lambda r: r["regret"])

    # Build markdown
    lines = [
        "# Final Evaluation Summary",
        "",
        f"Bootstrap CI: {int(ci_level * 100)}% ({n_bootstrap} replicates)",
        "",
    ]

    lines += [
        "## Method Rankings by Mean WG (Test Regimes)",
        "",
        "| Method | Source | Mean WG | SLO Viol. | n_regimes |",
        "|--------|--------|---------|-----------|-----------|",
    ]
    for r in summary_rows[:20]:
        lines.append(
            f"| {r['method']} | {r['source']} | {_fmt(r['mean_wg'])} "
            f"| {_fmt(r['vr_mean'])} | {r['n_regimes']} |"
        )

    if best_fixed_name:
        lines += [
            "",
            f"**Best fixed baseline**: {best_fixed_name} (mean WG={_fmt(best_fixed_mean)})",
        ]
    if oracle_mean is not None:
        lines += [f"**Oracle (non-deployable)**: oracle_srtf (mean WG={_fmt(oracle_mean)})"]

    if paired_rows:
        lines += [
            "",
            "## Paired Improvements vs Best Fixed Baseline",
            "",
            f"| Method | Delta WG | 95% CI | Interpretation |",
            "|--------|----------|--------|----------------|",
        ]
        for r in sorted(paired_rows, key=lambda x: -x["delta_vs_best_fixed"]):
            lines.append(
                f"| {r['method']} | {_fmt(r['delta_vs_best_fixed'])} "
                f"| {_ci_str(r['delta_ci_lo'], r['delta_ci_hi'])} "
                f"| {r['interpretation']} |"
            )

    if regret_rows:
        lines += [
            "",
            "## Regret to Oracle (oracle_srtf)",
            "",
            "| Method | Mean WG | Regret | Regret % |",
            "|--------|---------|--------|----------|",
        ]
        for r in regret_rows[:15]:
            lines.append(
                f"| {r['method']} | {_fmt(r['mean_wg'])} "
                f"| {_fmt(r['regret'])} | {_fmt(r['regret_pct'], 1)}% |"
            )

    if selector_summary:
        lines += [
            "",
            "## Selector Results (Test Split)",
            "",
            "| Model | Selected WG | Accuracy | Macro F1 | Regret | Delta vs Best Fixed |",
            "|-------|------------|----------|----------|--------|---------------------|",
        ]
        for model, d in sorted(selector_summary.items()):
            lines.append(
                f"| {model} | {_fmt(d.get('selected_mean_wg'))} "
                f"| {_fmt(d.get('accuracy'))} "
                f"| {_fmt(d.get('macro_f1'))} "
                f"| {_fmt(d.get('regret_to_window_best'))} "
                f"| {_fmt(d.get('diff_vs_best_fixed'))} |"
            )

    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- If CI for improvement vs best fixed crosses zero: result is **not statistically clear**.",
        "- If delta < 0.01 (1 pp): result is **marginal**.",
        "- If delta < 0.02 (2 pp): result is **marginal**.",
        "- oracle_srtf is a non-deployable hindsight upper bound. Do not report as achievable.",
        "- LLM heuristics evaluated offline in calibrated simulator; no production claims.",
        "",
        "---",
        "_Generated by scripts/summarize_final_evaluation.py_",
    ]

    return (
        "\n".join(lines),
        summary_rows,
        paired_rows,
        ci_rows,
        regret_rows,
    )


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("# no data\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-regime metrics from: {eval_dir}")
    rows = load_per_regime_metrics(eval_dir)
    if not rows:
        print(f"WARNING: no per-regime metric rows found in {eval_dir}")
        print("  Expected: candidate_metrics_by_regime_flat.csv and/or "
              "baseline_metrics_by_regime_flat.csv")

    selector_summary = None
    if args.selector_eval_dir:
        sel_dir = Path(args.selector_eval_dir)
        selector_summary = load_selector_summary(sel_dir)
        if selector_summary:
            print(f"Loaded selector summary: {list(selector_summary.keys())}")
        else:
            print(f"WARNING: no selector summary found in {sel_dir}")

    by_method = aggregate_by_method(rows)
    best_fixed_per_regime = compute_best_fixed_per_regime(rows)
    oracle_per_regime = compute_oracle_per_regime(rows)

    print(f"Methods: {len(by_method)}")
    print(f"Best fixed baselines by regime: {best_fixed_per_regime}")
    if oracle_per_regime:
        print(f"Oracle WG by regime: {oracle_per_regime}")

    md, summary_rows, paired_rows, ci_rows, regret_rows = build_final_summary_md(
        by_method=by_method,
        best_fixed_per_regime=best_fixed_per_regime,
        oracle_per_regime=oracle_per_regime,
        selector_summary=selector_summary,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
        rng=rng,
    )

    (output_dir / "final_evaluation_summary.md").write_text(md, encoding="utf-8")
    _write_csv(output_dir / "final_summary_table.csv", summary_rows)
    _write_csv(output_dir / "paired_improvements.csv", paired_rows)
    _write_csv(output_dir / "confidence_intervals.csv", ci_rows)
    _write_csv(output_dir / "regret_to_oracle.csv", regret_rows)

    print(f"\nOutputs written to: {output_dir}")
    print(f"  final_evaluation_summary.md")
    print(f"  final_summary_table.csv  ({len(summary_rows)} methods)")
    print(f"  paired_improvements.csv  ({len(paired_rows)} methods)")
    print(f"  confidence_intervals.csv ({len(ci_rows)} methods)")
    print(f"  regret_to_oracle.csv     ({len(regret_rows)} methods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
