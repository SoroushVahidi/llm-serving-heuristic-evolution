#!/usr/bin/env python3
"""
Phase 2B.12 Workload Diversity for Selector Label Analysis.

Builds a ~200-window evaluation suite spanning diverse load, SLO pressure,
token structure, KV pressure, noise, and priority regimes.  Measures
per-window oracle label diversity and determines whether RF/DT training
is feasible (≥200 windows, ≥3 policies each winning ≥10 windows, no single
policy >80-85%).

Groups
------
  regression (dev + heldout)  — Phase 2B.9/2B.11 baseline suite (seeds 0-5)
  diversity                   — 15 new workloads (seeds 6-9)
  overall                     — all rows combined

Usage
-----
python scripts/run_phase2b12_workload_diversity_selector_labels.py \\
    --config configs/phase2b12_workload_diversity_selector_labels.yaml \\
    --log-file logs/phase2b12/phase2b12_workload_diversity.log
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.dataset import DatasetConfig, build_selector_dataset
from llmserveopt.selector.features import FeatureMode
from llmserveopt.selector.models import RuleBasedSelector
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.service_model_factory import build_service_model_from_config

from run_phase2b9_selector_robustness import (  # noqa: E402
    apply_selectors_to_rows,
    build_gpu_configs,
    compute_fixed_baseline_wgs,
    load_config,
    load_or_generate_trace,
    summarize_group,
    write_per_window_csv,
    write_summary_csv,
)

SCORPIO_POLICY = "scorpio_style_slo_guard"
RULE_SELECTOR = "rule_based"


# ---------------------------------------------------------------------------
# Row building helpers (reused from Phase 2B.11)
# ---------------------------------------------------------------------------

def build_rows_for_group(
    workloads: List[Dict],
    seeds: List[int],
    gpu_configs,
    service_model,
    drain_steps: int,
    window_size: int,
    min_partial: int,
    feature_mode: FeatureMode,
    verbose: bool,
) -> List[Dict]:
    """Build selector dataset rows for all workloads/seeds in one group."""
    all_rows: List[Dict] = []
    for wdef in workloads:
        tag = wdef.get("tag", "workload")
        source = wdef.get("source", "synthetic")
        seeds_to_use = seeds if source == "synthetic" else [seeds[0]]
        for seed in seeds_to_use:
            trace_tag = f"{tag}_s{seed}"
            logging.info("  %s", trace_tag)
            reqs = load_or_generate_trace(wdef, seed=seed)
            if not reqs:
                logging.warning("  Empty trace, skipping %s", trace_tag)
                continue
            dataset_cfg = DatasetConfig(
                trace_id=trace_tag,
                window_size=window_size,
                min_partial_window=min_partial,
                feature_mode=feature_mode,
                gpu_configs=gpu_configs,
                service_model=service_model,
                drain_steps=drain_steps,
                seed=seed,
                verbose=verbose,
            )
            rows = build_selector_dataset(reqs, dataset_cfg)
            all_rows.extend(rows)
    return all_rows


def policy_mean_wg(rows: List[Dict], policy: str) -> float:
    key = f"reward_{policy}"
    vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", None)]
    return round(float(np.mean(vals)), 4) if vals else float("nan")


def collect_policy_distribution(rows: List[Dict], selector_key: str) -> Dict[str, int]:
    counts: Counter = Counter()
    for r in rows:
        policy = r.get(f"sel_{selector_key}_policy")
        if policy:
            counts[policy] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Label diversity analysis
# ---------------------------------------------------------------------------

def compute_label_distribution(rows: List[Dict]) -> Dict[str, int]:
    """Count per-window oracle labels (best policy per window)."""
    counts: Counter = Counter()
    for r in rows:
        label = r.get("best_policy")
        if label:
            counts[label] += 1
    return dict(counts.most_common())


def check_rf_feasibility(
    label_dist: Dict[str, int],
    min_windows: int,
    min_policies_winning: int,
    min_windows_per_policy: int,
    max_single_policy_fraction: float,
) -> Tuple[bool, Dict]:
    total = sum(label_dist.values())
    policies_with_enough = [
        p for p, n in label_dist.items() if n >= min_windows_per_policy
    ]
    top_policy_count = max(label_dist.values()) if label_dist else 0
    top_fraction = top_policy_count / total if total > 0 else 1.0
    top_policy = max(label_dist, key=label_dist.get) if label_dist else "none"

    passes_window_count = total >= min_windows
    passes_policy_spread = len(policies_with_enough) >= min_policies_winning
    passes_concentration = top_fraction < max_single_policy_fraction

    feasible = passes_window_count and passes_policy_spread and passes_concentration

    return feasible, {
        "total_windows": total,
        "n_policies_in_labels": len(label_dist),
        "policies_with_enough_windows": policies_with_enough,
        "n_policies_with_enough_windows": len(policies_with_enough),
        "top_policy": top_policy,
        "top_policy_count": top_policy_count,
        "top_policy_fraction": round(top_fraction, 4),
        "passes_window_count": passes_window_count,
        "passes_policy_spread": passes_policy_spread,
        "passes_concentration": passes_concentration,
        "feasible": feasible,
    }


def label_diversity_summary(rows: List[Dict], group_name: str) -> Dict:
    """Per-group label distribution and diversity stats."""
    label_dist = compute_label_distribution(rows)
    total = len(rows)
    return {
        "group": group_name,
        "n_windows": total,
        "label_distribution": label_dist,
        "n_distinct_labels": len(label_dist),
        "top_label": max(label_dist, key=label_dist.get) if label_dist else "none",
        "top_label_fraction": (
            round(max(label_dist.values()) / total, 4) if total > 0 and label_dist else 1.0
        ),
    }


# ---------------------------------------------------------------------------
# Aux metrics (SLO violation rate + completion fraction)
# ---------------------------------------------------------------------------

def collect_aux_metrics(
    workloads: List[Dict],
    seeds: List[int],
    gpu_configs,
    service_model,
    drain_steps: int,
    window_size: int,
    min_partial: int,
    feature_mode: FeatureMode,
    policies: List[str],
    verbose: bool,
) -> Dict[str, Dict[str, float]]:
    from llmserveopt.selector.dataset import run_policy_on_window
    from llmserveopt.selector.windows import make_windows

    slo_totals: Dict[str, List[float]] = defaultdict(list)
    comp_totals: Dict[str, List[float]] = defaultdict(list)

    for wdef in workloads:
        tag = wdef.get("tag", "workload")
        source = wdef.get("source", "synthetic")
        seeds_to_use = seeds if source == "synthetic" else [seeds[0]]
        for seed in seeds_to_use:
            trace_tag = f"{tag}_s{seed}"
            reqs = load_or_generate_trace(wdef, seed=seed)
            if not reqs:
                continue
            windows = make_windows(
                requests=reqs,
                trace_id=trace_tag,
                window_size=window_size,
                min_partial=min_partial,
            )
            for w in windows:
                for pname in policies:
                    m = run_policy_on_window(
                        policy_name=pname,
                        window=w,
                        gpu_configs=gpu_configs,
                        service_model=service_model,
                        drain_steps=drain_steps,
                        seed=seed,
                    )
                    if not np.isnan(m.slo_violation_rate):
                        slo_totals[pname].append(m.slo_violation_rate)
                    if not np.isnan(m.completion_fraction):
                        comp_totals[pname].append(m.completion_fraction)

    return {
        "slo_violation_rate": {
            p: round(float(np.mean(v)), 4) if v else float("nan")
            for p, v in slo_totals.items()
        },
        "completion_fraction": {
            p: round(float(np.mean(v)), 4) if v else float("nan")
            for p, v in comp_totals.items()
        },
    }


# ---------------------------------------------------------------------------
# Per-workload label table
# ---------------------------------------------------------------------------

def per_workload_label_table(all_rows: List[Dict]) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_rows:
        trace_id = r.get("trace_id", "")
        # Strip seed suffix to get workload tag
        parts = trace_id.rsplit("_s", 1)
        tag = parts[0] if len(parts) == 2 and parts[1].isdigit() else trace_id
        groups[tag].append(r)

    table = {}
    for tag, rows in sorted(groups.items()):
        fixed_wgs = compute_fixed_baseline_wgs(rows)
        best_label = max(fixed_wgs, key=fixed_wgs.get) if fixed_wgs else "none"
        best_wg = fixed_wgs.get(best_label, float("nan"))
        oracle = float(np.mean([float(r.get("best_weighted_goodput", 0) or 0) for r in rows]))
        label_dist = compute_label_distribution(rows)
        table[tag] = {
            "n_windows": len(rows),
            "best_fixed_policy": best_label,
            "best_fixed_wg": round(best_wg, 4),
            "oracle_mean_wg": round(oracle, 4),
            "label_distribution": label_dist,
        }
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2B.12 Workload Diversity for Selector Label Analysis"
    )
    p.add_argument(
        "--config",
        default="configs/phase2b12_workload_diversity_selector_labels.yaml",
    )
    p.add_argument("--log-file", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--skip-diversity", action="store_true",
                   help="Run only regression group (faster debug)")
    p.add_argument("--skip-aux-metrics", action="store_true",
                   help="Skip SLO/completion aux pass")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    logging.info("Phase 2B.12 Workload Diversity for Selector Label Analysis")

    cfg = load_config(args.config)
    out_dir = Path(
        args.out_dir
        or cfg.get("output_dir", "results/phase2b12_workload_diversity_selector_labels")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = cfg.get("phase2b11_reference", {})
    rf_thresholds = cfg.get("rf_feasibility", {})
    gpu_configs = build_gpu_configs(cfg)
    drain_steps = cfg.get("simulator", {}).get("drain_steps", 20000)
    window_size = cfg.get("window_size", 200)
    min_partial = cfg.get("min_partial_window", 50)
    feature_mode = FeatureMode(cfg.get("feature_mode", "online_prefix"))

    service_model_cfg = cfg.get("service_model", {"type": "synthetic"})
    try:
        service_model = build_service_model_from_config({"service_model": service_model_cfg})
    except Exception:
        service_model = ServiceModel()

    dev_seeds = cfg.get("dev_seeds", [0, 1, 2])
    heldout_seeds = cfg.get("heldout_seeds", [3, 4, 5])
    diversity_seeds = cfg.get("diversity_seeds", [6, 7, 8, 9])

    all_workloads = cfg.get("workloads", [])
    dev_workloads = [w for w in all_workloads if w.get("group") == "dev"]
    heldout_workloads = [w for w in all_workloads if w.get("group") == "heldout"]
    diversity_workloads = [w for w in all_workloads if w.get("group") == "diversity"]

    logging.info(
        "Workloads: %d dev, %d heldout, %d diversity",
        len(dev_workloads), len(heldout_workloads), len(diversity_workloads),
    )
    logging.info("Deployable policies: %d", len(SELECTOR_CANDIDATES))

    assert SCORPIO_POLICY in SELECTOR_CANDIDATES, f"{SCORPIO_POLICY} not in SELECTOR_CANDIDATES"

    selector = RuleBasedSelector()
    assert SCORPIO_POLICY in selector._POLICY_CHOICES, (
        f"{SCORPIO_POLICY} not in RuleBasedSelector._POLICY_CHOICES"
    )
    models = {RULE_SELECTOR: selector}
    t0 = time.perf_counter()

    # --- Build rows for each group ---
    logging.info("--- Dev group (%d workloads, seeds %s) ---", len(dev_workloads), dev_seeds)
    dev_rows = build_rows_for_group(
        dev_workloads, dev_seeds, gpu_configs, service_model,
        drain_steps, window_size, min_partial, feature_mode, args.verbose,
    )

    logging.info("--- Held-out group (%d workloads, seeds %s) ---",
                 len(heldout_workloads), heldout_seeds)
    heldout_rows = build_rows_for_group(
        heldout_workloads, heldout_seeds, gpu_configs, service_model,
        drain_steps, window_size, min_partial, feature_mode, args.verbose,
    )

    diversity_rows: List[Dict] = []
    if not args.skip_diversity:
        logging.info("--- Diversity group (%d workloads, seeds %s) ---",
                     len(diversity_workloads), diversity_seeds)
        diversity_rows = build_rows_for_group(
            diversity_workloads, diversity_seeds, gpu_configs, service_model,
            drain_steps, window_size, min_partial, feature_mode, args.verbose,
        )

    regression_rows = dev_rows + heldout_rows
    all_rows = regression_rows + diversity_rows

    # Apply selectors to all rows
    dev_rows = apply_selectors_to_rows(dev_rows, models)
    heldout_rows = apply_selectors_to_rows(heldout_rows, models)
    diversity_rows = apply_selectors_to_rows(diversity_rows, models)
    regression_rows = apply_selectors_to_rows(regression_rows, models)
    all_rows = dev_rows + heldout_rows + diversity_rows

    # Summaries
    dev_summary = summarize_group(dev_rows, "dev", models)
    heldout_summary = summarize_group(heldout_rows, "heldout", models)
    regression_summary = summarize_group(regression_rows, "regression", models)
    diversity_summary = summarize_group(diversity_rows, "diversity", models) if diversity_rows else {}
    overall_summary = summarize_group(all_rows, "overall", models)

    elapsed = time.perf_counter() - t0
    logging.info("Main evaluation pass: %.1fs  (%d windows)", elapsed, len(all_rows))

    # --- Label diversity analysis ---
    logging.info("--- Label diversity analysis ---")
    all_label_div = label_diversity_summary(all_rows, "overall")
    regression_label_div = label_diversity_summary(regression_rows, "regression")
    diversity_label_div = label_diversity_summary(diversity_rows, "diversity") if diversity_rows else {}

    min_w = rf_thresholds.get("min_windows", 200)
    min_pol = rf_thresholds.get("min_policies_winning", 3)
    min_win_per_pol = rf_thresholds.get("min_windows_per_policy", 10)
    max_frac = rf_thresholds.get("max_single_policy_fraction", 0.85)

    rf_decision, rf_details = check_rf_feasibility(
        all_label_div["label_distribution"],
        min_windows=min_w,
        min_policies_winning=min_pol,
        min_windows_per_policy=min_win_per_pol,
        max_single_policy_fraction=max_frac,
    )

    logging.info("Label distribution (overall): %s", all_label_div["label_distribution"])
    logging.info("RF/DT feasibility: %s", "FEASIBLE" if rf_decision else "NOT FEASIBLE")
    logging.info("  total_windows=%d  policies_with≥%d_wins=%d  top=%s(%.1f%%)",
                 rf_details["total_windows"], min_win_per_pol,
                 rf_details["n_policies_with_enough_windows"],
                 rf_details["top_policy"],
                 rf_details["top_policy_fraction"] * 100)

    # --- Aux metrics (SCORPIO + 4 baselines) ---
    aux_metrics: Dict = {}
    aux_policies = [SCORPIO_POLICY, "admission_control", "edf",
                    "weighted_shortest_processing", "slo_slack_score"]
    if not args.skip_aux_metrics and all_rows:
        logging.info("--- Aux metrics (regression group only — SLO violation, completion) ---")
        aux_metrics["regression"] = collect_aux_metrics(
            dev_workloads + heldout_workloads,
            dev_seeds + heldout_seeds,
            gpu_configs, service_model,
            drain_steps, window_size, min_partial, feature_mode,
            aux_policies, args.verbose,
        )

    total_elapsed = time.perf_counter() - t0
    logging.info("Total elapsed: %.1fs", total_elapsed)

    # --- Outputs ---
    write_per_window_csv(all_rows, out_dir / "per_window.csv")

    # Selector comparison table (per group)
    comparison_rows = []
    for group_name, summary, rows in [
        ("dev", dev_summary, dev_rows),
        ("heldout", heldout_summary, heldout_rows),
        ("regression", regression_summary, regression_rows),
        ("diversity", diversity_summary, diversity_rows),
        ("overall", overall_summary, all_rows),
    ]:
        if not summary.get("n_windows"):
            continue
        rb_wg = summary.get("sel_rule_based_mean_wg")
        best_fixed_wg = summary.get("best_fixed_mean_wg")
        scorpio_wg = policy_mean_wg(rows, SCORPIO_POLICY)
        comparison_rows.append({
            "group": group_name,
            "n_windows": summary["n_windows"],
            "scorpio_fixed_wg": scorpio_wg,
            "best_fixed_policy": summary.get("best_fixed_policy"),
            "best_fixed_wg": best_fixed_wg,
            "rule_based_wg": rb_wg,
            "rule_based_gap_vs_best_fixed": (
                round(rb_wg - best_fixed_wg, 4)
                if rb_wg is not None and best_fixed_wg is not None else None
            ),
            "rule_based_gap_vs_oracle": summary.get("sel_rule_based_gap_vs_oracle"),
            "oracle_per_window_best_wg": summary.get("oracle_per_window_best_mean_wg"),
            "phase2b11_rule_based_wg": ref.get("rule_based_wg", {}).get(group_name),
            "phase2b11_best_fixed_wg": ref.get("best_fixed_wg", {}).get(group_name),
        })
    write_summary_csv(comparison_rows, out_dir / "selector_comparison.csv")

    # Policy ranking (overall)
    rank_rows = []
    for group_name, rows in [
        ("dev", dev_rows), ("heldout", heldout_rows),
        ("regression", regression_rows), ("diversity", diversity_rows),
        ("overall", all_rows),
    ]:
        if not rows:
            continue
        wg_by_policy = {p: policy_mean_wg(rows, p) for p in SELECTOR_CANDIDATES}
        ranked = sorted(wg_by_policy.items(), key=lambda x: -x[1])
        for rank, (pname, wg) in enumerate(ranked, 1):
            rank_rows.append({
                "group": group_name, "rank": rank, "policy": pname, "mean_wg": wg
            })
    write_summary_csv(rank_rows, out_dir / "policy_ranking.csv")

    # Label distribution CSV
    label_rows = []
    all_policies_in_labels = sorted(
        set().union(*[ld.get("label_distribution", {}).keys()
                      for ld in [all_label_div, regression_label_div, diversity_label_div]
                      if ld])
    )
    for policy in all_policies_in_labels:
        label_rows.append({
            "policy": policy,
            "overall_wins": all_label_div["label_distribution"].get(policy, 0),
            "regression_wins": regression_label_div.get("label_distribution", {}).get(policy, 0),
            "diversity_wins": diversity_label_div.get("label_distribution", {}).get(policy, 0) if diversity_label_div else 0,
        })
    label_rows.sort(key=lambda r: -r["overall_wins"])
    write_summary_csv(label_rows, out_dir / "label_distribution.csv")

    # Per-workload breakdown
    workload_table = per_workload_label_table(all_rows)
    with open(out_dir / "per_workload_labels.json", "w") as f:
        json.dump(workload_table, f, indent=2)

    # Rule selector policy distribution
    all_sel_dist = collect_policy_distribution(all_rows, RULE_SELECTOR)
    regression_sel_dist = collect_policy_distribution(regression_rows, RULE_SELECTOR)
    diversity_sel_dist = collect_policy_distribution(diversity_rows, RULE_SELECTOR)

    # Metadata
    metadata = {
        "experiment": "phase2b12_workload_diversity_selector_labels",
        "n_deployable_policies": len(SELECTOR_CANDIDATES),
        "n_workloads": {
            "dev": len(dev_workloads),
            "heldout": len(heldout_workloads),
            "diversity": len(diversity_workloads),
            "total": len(all_workloads),
        },
        "n_windows": {
            "dev": len(dev_rows),
            "heldout": len(heldout_rows),
            "regression": len(regression_rows),
            "diversity": len(diversity_rows),
            "total": len(all_rows),
        },
        "phase2b11_reference": ref,
        "scorpio_fixed_wg": {
            "dev": policy_mean_wg(dev_rows, SCORPIO_POLICY),
            "heldout": policy_mean_wg(heldout_rows, SCORPIO_POLICY),
            "regression": policy_mean_wg(regression_rows, SCORPIO_POLICY),
            "diversity": policy_mean_wg(diversity_rows, SCORPIO_POLICY) if diversity_rows else None,
            "overall": policy_mean_wg(all_rows, SCORPIO_POLICY),
        },
        "rule_based_wg": {
            "dev": dev_summary.get("sel_rule_based_mean_wg"),
            "heldout": heldout_summary.get("sel_rule_based_mean_wg"),
            "regression": regression_summary.get("sel_rule_based_mean_wg"),
            "diversity": diversity_summary.get("sel_rule_based_mean_wg") if diversity_summary else None,
            "overall": overall_summary.get("sel_rule_based_mean_wg"),
        },
        "best_fixed_wg": {
            "regression": regression_summary.get("best_fixed_mean_wg"),
            "diversity": diversity_summary.get("best_fixed_mean_wg") if diversity_summary else None,
            "overall": overall_summary.get("best_fixed_mean_wg"),
        },
        "best_fixed_policy": {
            "regression": regression_summary.get("best_fixed_policy"),
            "diversity": diversity_summary.get("best_fixed_policy") if diversity_summary else None,
            "overall": overall_summary.get("best_fixed_policy"),
        },
        "label_diversity": {
            "overall": all_label_div,
            "regression": regression_label_div,
            "diversity": diversity_label_div if diversity_label_div else {},
        },
        "rf_feasibility": rf_details,
        "rf_training_recommended": rf_decision,
        "rule_selector_policy_distribution": {
            "overall": all_sel_dist,
            "regression": regression_sel_dist,
            "diversity": diversity_sel_dist,
        },
        "aux_metrics": aux_metrics,
        "elapsed_seconds": round(total_elapsed, 1),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(out_dir / "dev_summary.json", "w") as f:
        json.dump(dev_summary, f, indent=2, default=str)
    with open(out_dir / "heldout_summary.json", "w") as f:
        json.dump(heldout_summary, f, indent=2, default=str)
    with open(out_dir / "diversity_summary.json", "w") as f:
        json.dump(diversity_summary, f, indent=2, default=str)
    with open(out_dir / "regression_summary.json", "w") as f:
        json.dump(regression_summary, f, indent=2, default=str)

    # --- Console summary ---
    logging.info("=" * 60)
    logging.info("Phase 2B.12 Workload Diversity — Results")
    logging.info("=" * 60)
    for gname, gsummary in [
        ("dev", dev_summary), ("heldout", heldout_summary),
        ("regression", regression_summary), ("diversity", diversity_summary),
        ("overall", overall_summary),
    ]:
        if not gsummary.get("n_windows"):
            continue
        rb = gsummary.get("sel_rule_based_mean_wg", float("nan"))
        bf = gsummary.get("best_fixed_mean_wg", float("nan"))
        gap = gsummary.get("sel_rule_based_gap_vs_best_fixed")
        logging.info(
            "[%s] n=%d rule_based=%.4f best_fixed=%.4f gap=%s",
            gname, gsummary["n_windows"], rb, bf,
            f"{gap:+.4f}" if gap is not None else "n/a",
        )
    logging.info(
        "Label distribution (overall, top-5): %s",
        dict(Counter(all_label_div["label_distribution"]).most_common(5)),
    )
    logging.info("RF/DT feasible: %s", rf_decision)
    logging.info("  Criteria: %d windows / %d policies≥%d wins / top≤%.0f%%",
                 min_w, min_pol, min_win_per_pol, max_frac * 100)
    logging.info(
        "  Actual: %d windows / %d policies / top=%s(%.1f%%)",
        rf_details["total_windows"],
        rf_details["n_policies_with_enough_windows"],
        rf_details["top_policy"],
        rf_details["top_policy_fraction"] * 100,
    )
    logging.info("Rule selector dispatch (overall): %s", all_sel_dist)
    logging.info("Outputs → %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
