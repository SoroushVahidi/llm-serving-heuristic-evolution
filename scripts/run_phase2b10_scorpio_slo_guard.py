#!/usr/bin/env python3
"""
Phase 2B.10 SCORPIO-Style SLO Guard Comparison.

Re-evaluates the Phase 2B.9 workload suite with 20 deployable policies
(including scorpio_style_slo_guard) and compares against Phase 2B.9 baselines.

Usage
-----
python scripts/run_phase2b10_scorpio_slo_guard.py \\
    --config configs/phase2b10_scorpio_slo_guard.yaml \\
    --log-file logs/phase2b10/phase2b10_scorpio_slo_guard.log
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
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


def _mean_metric(rows: List[Dict], policy: str, field_prefix: str) -> float:
    key = f"{field_prefix}_{policy}"
    vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", None)]
    return float(np.mean(vals)) if vals else float("nan")


def enrich_rows_with_aux_metrics(rows: List[Dict]) -> List[Dict]:
    """Attach per-policy SLO violation and completion fraction from reward lookup."""
    # build_selector_dataset rows only store WG; re-derive aux from RunMetrics is
    # not available post-hoc.  Aux metrics are collected in summarize via
    # per-window side channel during build — handled in build_rows_with_metrics.
    return rows


def build_rows_with_metrics(
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
    """Build selector dataset rows (WG) for all workloads/seeds."""
    all_rows: List[Dict] = []
    for wdef in workloads:
        tag = wdef.get("tag", "workload")
        source = wdef.get("source", "synthetic")
        seeds_to_use = seeds if source == "synthetic" else [seeds[0]]
        for seed in seeds_to_use:
            trace_tag = f"{tag}_s{seed}"
            logging.info("Building dataset for %s", trace_tag)
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


def collect_policy_aux_metrics(
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
        "slo_violation_rate": {p: float(np.mean(v)) if v else float("nan")
                               for p, v in slo_totals.items()},
        "completion_fraction": {p: float(np.mean(v)) if v else float("nan")
                                for p, v in comp_totals.items()},
    }


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2B.10 SCORPIO SLO Guard Comparison")
    p.add_argument("--config", default="configs/phase2b10_scorpio_slo_guard.yaml")
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

    logging.info("Phase 2B.10 SCORPIO-Style SLO Guard Comparison")
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg.get("output_dir", "results/phase2b10_scorpio_slo_guard"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = cfg.get("phase2b9_reference", {})
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
    assert SCORPIO_POLICY in SELECTOR_CANDIDATES

    models = {RULE_SELECTOR: RuleBasedSelector()}
    t0 = time.perf_counter()

    logging.info("--- Dev group ---")
    dev_rows = build_rows_with_metrics(
        dev_workloads, dev_seeds, gpu_configs, service_model,
        drain_steps, window_size, min_partial, feature_mode, args.verbose,
    )
    heldout_rows: List[Dict] = []
    if not args.skip_heldout:
        logging.info("--- Held-out group ---")
        heldout_rows = build_rows_with_metrics(
            heldout_workloads, heldout_seeds, gpu_configs, service_model,
            drain_steps, window_size, min_partial, feature_mode, args.verbose,
        )

    dev_rows = apply_selectors_to_rows(dev_rows, models)
    heldout_rows = apply_selectors_to_rows(heldout_rows, models)
    all_rows = dev_rows + heldout_rows

    dev_summary = summarize_group(dev_rows, "dev", models)
    heldout_summary = summarize_group(heldout_rows, "heldout", models) if heldout_rows else {}
    overall_summary = summarize_group(all_rows, "overall", models)

    def policy_mean_wg(rows: List[Dict], policy: str) -> float:
        key = f"reward_{policy}"
        vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", None)]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    scorpio_dev_wg = policy_mean_wg(dev_rows, SCORPIO_POLICY)
    scorpio_held_wg = policy_mean_wg(heldout_rows, SCORPIO_POLICY)
    scorpio_overall_wg = policy_mean_wg(all_rows, SCORPIO_POLICY)

    aux_policies = [SCORPIO_POLICY, "admission_control", "edf",
                    "weighted_shortest_processing", "slo_slack_score"]
    aux_metrics: Dict = {}
    if not args.skip_aux_metrics:
        logging.info("--- Aux metrics (SLO violation, completion) ---")
        aux_metrics["dev"] = collect_policy_aux_metrics(
            dev_workloads, dev_seeds, gpu_configs, service_model,
            drain_steps, window_size, min_partial, feature_mode,
            aux_policies, args.verbose,
        )
        if heldout_rows:
            aux_metrics["heldout"] = collect_policy_aux_metrics(
                heldout_workloads, heldout_seeds, gpu_configs, service_model,
                drain_steps, window_size, min_partial, feature_mode,
                aux_policies, args.verbose,
            )

    elapsed = time.perf_counter() - t0
    logging.info("Completed in %.1fs (%d windows)", elapsed, len(all_rows))

    # Write outputs
    write_per_window_csv(all_rows, out_dir / "per_window.csv")

    comparison_rows = []
    for group_name, summary in [("dev", dev_summary), ("heldout", heldout_summary), ("overall", overall_summary)]:
        if not summary.get("n_windows"):
            continue
        row = {
            "group": group_name,
            "n_windows": summary["n_windows"],
            "oracle_per_window_best_wg": summary.get("oracle_per_window_best_mean_wg"),
            "best_fixed_policy": summary.get("best_fixed_policy"),
            "best_fixed_wg": summary.get("best_fixed_mean_wg"),
            "scorpio_style_slo_guard_wg": policy_mean_wg(
                dev_rows if group_name == "dev" else
                heldout_rows if group_name == "heldout" else all_rows,
                SCORPIO_POLICY,
            ),
            "rule_based_wg": summary.get("sel_rule_based_mean_wg"),
            "rule_based_gap_vs_best_fixed": summary.get("sel_rule_based_gap_vs_best_fixed"),
            "rule_based_gap_vs_oracle": summary.get("sel_rule_based_gap_vs_oracle"),
        }
        ref_wg = ref.get("rule_based_wg", {})
        ref_bf = ref.get("best_fixed_wg", {})
        row["phase2b9_rule_based_wg"] = ref_wg.get(group_name)
        row["phase2b9_best_fixed_wg"] = ref_bf.get(group_name)
        comparison_rows.append(row)
    write_summary_csv(comparison_rows, out_dir / "selector_comparison.csv")

    scorpio_rank_rows = []
    for group_name, rows in [("dev", dev_rows), ("heldout", heldout_rows), ("overall", all_rows)]:
        if not rows:
            continue
        wg_by_policy = {p: policy_mean_wg(rows, p) for p in SELECTOR_CANDIDATES}
        ranked = sorted(wg_by_policy.items(), key=lambda x: -x[1])
        for rank, (pname, wg) in enumerate(ranked, 1):
            scorpio_rank_rows.append({
                "group": group_name,
                "rank": rank,
                "policy": pname,
                "mean_wg": wg,
            })
    write_summary_csv(scorpio_rank_rows, out_dir / "policy_ranking.csv")

    # High-noise failure case
    noise_rows = [r for r in heldout_rows if "heldout_very_high_noise_s4" in r.get("trace_id", "")]
    noise_case = {}
    if noise_rows:
        rb_wg = float(np.mean([float(r.get("sel_rule_based_wg", 0)) for r in noise_rows]))
        scorpio_wg_val = float(np.mean([float(r.get(f"reward_{SCORPIO_POLICY}", 0)) for r in noise_rows]))
        best_wg = float(np.mean([float(r.get("best_weighted_goodput", 0)) for r in noise_rows]))
        noise_case = {
            "workload": "heldout_very_high_noise_s4",
            "n_windows": len(noise_rows),
            "rule_based_wg": round(rb_wg, 4),
            "scorpio_style_slo_guard_wg": round(scorpio_wg_val, 4),
            "per_window_best_wg": round(best_wg, 4),
            "best_policy": noise_rows[0].get("best_policy"),
            "rule_based_policy": noise_rows[0].get("sel_rule_based_policy"),
        }

    metadata = {
        "experiment": "phase2b10_scorpio_slo_guard",
        "n_deployable_policies": len(SELECTOR_CANDIDATES),
        "scorpio_policy_name": SCORPIO_POLICY,
        "n_dev_windows": len(dev_rows),
        "n_heldout_windows": len(heldout_rows),
        "phase2b9_reference": ref,
        "scorpio_wg": {
            "dev": scorpio_dev_wg,
            "heldout": scorpio_held_wg,
            "overall": scorpio_overall_wg,
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

    logging.info("SCORPIO-style WG: dev=%.4f heldout=%.4f overall=%.4f",
                 scorpio_dev_wg, scorpio_held_wg, scorpio_overall_wg)
    logging.info("Rule selector WG: dev=%.4f heldout=%.4f overall=%.4f",
                 dev_summary.get("sel_rule_based_mean_wg", 0),
                 heldout_summary.get("sel_rule_based_mean_wg", 0),
                 overall_summary.get("sel_rule_based_mean_wg", 0))
    logging.info("Best fixed WG: dev=%.4f (%s) heldout=%.4f (%s)",
                 dev_summary.get("best_fixed_mean_wg", 0), dev_summary.get("best_fixed_policy"),
                 heldout_summary.get("best_fixed_mean_wg", 0), heldout_summary.get("best_fixed_policy"))
    if noise_case:
        logging.info("High-noise s4: rule=%.4f scorpio=%.4f best=%.4f",
                     noise_case["rule_based_wg"], noise_case["scorpio_style_slo_guard_wg"],
                     noise_case["per_window_best_wg"])
    logging.info("Outputs → %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
