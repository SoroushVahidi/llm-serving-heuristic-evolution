#!/usr/bin/env python3
"""
Phase 2B.11 SCORPIO Selector Integration.

Integrates scorpio_style_slo_guard into the rule-based selector (3 new routing rules)
and re-evaluates the Phase 2B.9/2B.10 workload suite (60 windows) with the updated
selector.  Reports whether the selector gap vs SCORPIO-style best fixed closes.

New rules vs Phase 2B.10:
  Rule 0: overloaded tight-SLO + recent violations → scorpio_style_slo_guard
  Rule 2a: very high prediction noise (pred_output_cv > 2.0) → scorpio_style_slo_guard
  Rule 3: standalone recent violations → scorpio_style_slo_guard (was AC)

Usage
-----
python scripts/run_phase2b11_scorpio_selector_integration.py \\
    --config configs/phase2b11_scorpio_selector_integration.yaml \\
    --log-file logs/phase2b11/phase2b11_scorpio_selector_integration.log
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

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
    load_config,
    load_or_generate_trace,
    summarize_group,
    write_per_window_csv,
    write_summary_csv,
)

SCORPIO_POLICY = "scorpio_style_slo_guard"
RULE_SELECTOR = "rule_based"


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


def collect_policy_distribution(rows: List[Dict], selector_key: str) -> Dict[str, int]:
    """Count how many times each policy was chosen by a selector."""
    counts: Counter = Counter()
    for r in rows:
        policy = r.get(f"sel_{selector_key}_policy")
        if policy:
            counts[policy] += 1
    return dict(counts)


def policy_mean_wg(rows: List[Dict], policy: str) -> float:
    key = f"reward_{policy}"
    vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", None)]
    return round(float(np.mean(vals)), 4) if vals else float("nan")


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
    """Mean SLO violation rate and completion fraction for selected policies."""
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
        "slo_violation_rate": {p: round(float(np.mean(v)), 4) if v else float("nan")
                               for p, v in slo_totals.items()},
        "completion_fraction": {p: round(float(np.mean(v)), 4) if v else float("nan")
                                for p, v in comp_totals.items()},
    }


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2B.11 SCORPIO Selector Integration")
    p.add_argument("--config", default="configs/phase2b11_scorpio_selector_integration.yaml")
    p.add_argument("--log-file", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--skip-heldout", action="store_true")
    p.add_argument("--skip-aux-metrics", action="store_true",
                   help="Skip SLO/completion aux pass (faster debug)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


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

    logging.info("Phase 2B.11 SCORPIO Selector Integration")
    logging.info("Rule selector: 3 new SCORPIO routing rules added")

    cfg = load_config(args.config)
    out_dir = Path(
        args.out_dir or cfg.get("output_dir", "results/phase2b11_scorpio_selector_integration")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = cfg.get("phase2b10_reference", {})
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
    all_workloads = cfg.get("workloads", [])
    dev_workloads = [w for w in all_workloads if w.get("group") == "dev"]
    heldout_workloads = [w for w in all_workloads if w.get("group") == "heldout"]

    logging.info("Deployable policies: %d", len(SELECTOR_CANDIDATES))
    assert SCORPIO_POLICY in SELECTOR_CANDIDATES, f"{SCORPIO_POLICY} not in SELECTOR_CANDIDATES"

    selector = RuleBasedSelector()
    assert SCORPIO_POLICY in selector._POLICY_CHOICES, (
        f"{SCORPIO_POLICY} not in RuleBasedSelector._POLICY_CHOICES"
    )
    models = {RULE_SELECTOR: selector}
    t0 = time.perf_counter()

    logging.info("--- Dev group (%d workloads, seeds %s) ---", len(dev_workloads), dev_seeds)
    dev_rows = build_rows_for_group(
        dev_workloads, dev_seeds, gpu_configs, service_model,
        drain_steps, window_size, min_partial, feature_mode, args.verbose,
    )

    heldout_rows: List[Dict] = []
    if not args.skip_heldout:
        logging.info("--- Held-out group (%d workloads, seeds %s) ---",
                     len(heldout_workloads), heldout_seeds)
        heldout_rows = build_rows_for_group(
            heldout_workloads, heldout_seeds, gpu_configs, service_model,
            drain_steps, window_size, min_partial, feature_mode, args.verbose,
        )

    dev_rows = apply_selectors_to_rows(dev_rows, models)
    heldout_rows = apply_selectors_to_rows(heldout_rows, models)
    all_rows = dev_rows + heldout_rows

    dev_summary = summarize_group(dev_rows, "dev", models)
    heldout_summary = summarize_group(heldout_rows, "heldout", models) if heldout_rows else {}
    overall_summary = summarize_group(all_rows, "overall", models)

    # Policy distribution
    dev_dist = collect_policy_distribution(dev_rows, RULE_SELECTOR)
    held_dist = collect_policy_distribution(heldout_rows, RULE_SELECTOR)
    all_dist = collect_policy_distribution(all_rows, RULE_SELECTOR)
    logging.info("Rule selector policy distribution (dev): %s", dev_dist)
    logging.info("Rule selector policy distribution (held): %s", held_dist)

    # SCORPIO fixed WG per group
    scorpio_dev = policy_mean_wg(dev_rows, SCORPIO_POLICY)
    scorpio_held = policy_mean_wg(heldout_rows, SCORPIO_POLICY)
    scorpio_all = policy_mean_wg(all_rows, SCORPIO_POLICY)

    # Aux metrics
    aux_metrics: Dict = {}
    aux_policies = [SCORPIO_POLICY, "admission_control", "edf",
                    "weighted_shortest_processing", "slo_slack_score"]
    if not args.skip_aux_metrics:
        logging.info("--- Aux metrics (SLO violation, completion fraction) ---")
        aux_metrics["dev"] = collect_aux_metrics(
            dev_workloads, dev_seeds, gpu_configs, service_model,
            drain_steps, window_size, min_partial, feature_mode, aux_policies, args.verbose,
        )
        if heldout_rows:
            aux_metrics["heldout"] = collect_aux_metrics(
                heldout_workloads, heldout_seeds, gpu_configs, service_model,
                drain_steps, window_size, min_partial, feature_mode, aux_policies, args.verbose,
            )

    elapsed = time.perf_counter() - t0
    logging.info("Completed in %.1fs (%d windows)", elapsed, len(all_rows))

    # ---- Outputs ----
    write_per_window_csv(all_rows, out_dir / "per_window.csv")

    # Selector comparison table
    comparison_rows = []
    for group_name, summary, rows in [
        ("dev", dev_summary, dev_rows),
        ("heldout", heldout_summary, heldout_rows),
        ("overall", overall_summary, all_rows),
    ]:
        if not summary.get("n_windows"):
            continue
        rb_wg = summary.get("sel_rule_based_mean_wg")
        best_fixed_wg = summary.get("best_fixed_mean_wg")
        scorpio_wg = policy_mean_wg(rows, SCORPIO_POLICY)
        row = {
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
            "phase2b10_rule_based_wg": ref.get("rule_based_wg", {}).get(group_name),
            "phase2b10_best_fixed_wg": ref.get("best_fixed_wg", {}).get(group_name),
        }
        comparison_rows.append(row)
    write_summary_csv(comparison_rows, out_dir / "selector_comparison.csv")

    # Policy ranking
    rank_rows = []
    for group_name, rows in [("dev", dev_rows), ("heldout", heldout_rows), ("overall", all_rows)]:
        if not rows:
            continue
        wg_by_policy = {p: policy_mean_wg(rows, p) for p in SELECTOR_CANDIDATES}
        ranked = sorted(wg_by_policy.items(), key=lambda x: -x[1])
        for rank, (pname, wg) in enumerate(ranked, 1):
            rank_rows.append({"group": group_name, "rank": rank, "policy": pname, "mean_wg": wg})
    write_summary_csv(rank_rows, out_dir / "policy_ranking.csv")

    # High-noise failure case
    noise_rows = [r for r in heldout_rows if "heldout_very_high_noise_s4" in r.get("trace_id", "")]
    noise_case: Dict = {}
    if noise_rows:
        rb_wg_val = float(np.mean([float(r.get("sel_rule_based_wg", 0)) for r in noise_rows]))
        scorpio_wg_val = float(np.mean([float(r.get(f"reward_{SCORPIO_POLICY}", 0))
                                        for r in noise_rows]))
        best_wg_val = float(np.mean([float(r.get("best_weighted_goodput", 0)) for r in noise_rows]))
        noise_case = {
            "workload": "heldout_very_high_noise_s4",
            "n_windows": len(noise_rows),
            "rule_based_wg": round(rb_wg_val, 4),
            "rule_based_policy": noise_rows[0].get("sel_rule_based_policy"),
            "scorpio_fixed_wg": round(scorpio_wg_val, 4),
            "per_window_best_wg": round(best_wg_val, 4),
            "per_window_best_policy": noise_rows[0].get("best_policy"),
        }
        logging.info(
            "High-noise s4: rule=%s WG=%.4f, scorpio=%.4f, per-window-best=%.4f",
            noise_case["rule_based_policy"], rb_wg_val, scorpio_wg_val, best_wg_val,
        )

    # Metadata
    metadata = {
        "experiment": "phase2b11_scorpio_selector_integration",
        "n_deployable_policies": len(SELECTOR_CANDIDATES),
        "scorpio_policy_name": SCORPIO_POLICY,
        "scorpio_in_rule_selector": SCORPIO_POLICY in selector._POLICY_CHOICES,
        "rule_selector_policy_choices": selector._POLICY_CHOICES,
        "n_dev_windows": len(dev_rows),
        "n_heldout_windows": len(heldout_rows),
        "phase2b10_reference": ref,
        "scorpio_fixed_wg": {
            "dev": scorpio_dev, "heldout": scorpio_held, "overall": scorpio_all,
        },
        "rule_based_wg": {
            "dev": dev_summary.get("sel_rule_based_mean_wg"),
            "heldout": heldout_summary.get("sel_rule_based_mean_wg"),
            "overall": overall_summary.get("sel_rule_based_mean_wg"),
        },
        "best_fixed_wg": {
            "dev": dev_summary.get("best_fixed_mean_wg"),
            "heldout": heldout_summary.get("best_fixed_mean_wg"),
            "overall": overall_summary.get("best_fixed_mean_wg"),
        },
        "best_fixed_policy": {
            "dev": dev_summary.get("best_fixed_policy"),
            "heldout": heldout_summary.get("best_fixed_policy"),
            "overall": overall_summary.get("best_fixed_policy"),
        },
        "oracle_per_window_wg": {
            "dev": dev_summary.get("oracle_per_window_best_mean_wg"),
            "heldout": heldout_summary.get("oracle_per_window_best_mean_wg"),
            "overall": overall_summary.get("oracle_per_window_best_mean_wg"),
        },
        "rule_selector_gap_vs_best_fixed": {
            "dev": dev_summary.get("sel_rule_based_gap_vs_best_fixed"),
            "heldout": heldout_summary.get("sel_rule_based_gap_vs_best_fixed"),
            "overall": overall_summary.get("sel_rule_based_gap_vs_best_fixed"),
        },
        "rule_selector_policy_distribution": {
            "dev": dev_dist, "heldout": held_dist, "overall": all_dist,
        },
        "heldout_very_high_noise_s4": noise_case,
        "aux_metrics": aux_metrics,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(out_dir / "dev_summary.json", "w") as f:
        json.dump(dev_summary, f, indent=2, default=str)
    with open(out_dir / "heldout_summary.json", "w") as f:
        json.dump(heldout_summary, f, indent=2, default=str)

    # --- Console summary ---
    logging.info("=" * 60)
    logging.info("Phase 2B.11 SCORPIO Selector Integration — Results")
    logging.info("=" * 60)
    for group_name, summary in [("dev", dev_summary), ("heldout", heldout_summary),
                                  ("overall", overall_summary)]:
        if not summary.get("n_windows"):
            continue
        rb = summary.get("sel_rule_based_mean_wg", float("nan"))
        bf = summary.get("best_fixed_mean_wg", float("nan"))
        gap = summary.get("sel_rule_based_gap_vs_best_fixed")
        logging.info(
            "[%s] n=%d rule_based=%.4f best_fixed=%.4f gap=%s",
            group_name, summary["n_windows"], rb, bf,
            f"{gap:+.4f}" if gap is not None else "n/a",
        )
    logging.info("Selector policy distribution (overall): %s", all_dist)
    logging.info("Outputs → %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
