#!/usr/bin/env python3
"""
Phase 2B.13 Selector Training After Diversity.

Extends Phase 2B.12 to ≥200 windows, trains RF/DT selectors, evaluates them
alongside the rule selector (pre- and post-repair), and documents the label
diversity and WG performance on the diversified suite.

Phase 2B.12 finding: 166/172 windows are "all-complete" (WG≈1.0 for all
policies). Labels are diverse (9 policies, SCORPIO=45.9%) but primarily reflect
tie-breaking on secondary metrics, not genuine WG differences.

This runner:
  1. Runs all 20 deployable policies on the extended suite (dev + heldout +
     diversity with seeds 6–11 + 2 new high-differentiation workloads).
  2. Computes per-window oracle labels and label distribution.
  3. Trains RF and DT selectors on (dev + diversity seeds 6–10) windows.
  4. Evaluates selectors on held-out regression windows (seeds 3–5).
  5. Evaluates a repaired rule selector (Rule 5: prefill → AC instead of sarathi).
  6. Reports all metrics including admission/completion trade-offs.

Usage
-----
python scripts/run_phase2b13_selector_training_after_diversity.py \\
    --config configs/phase2b13_selector_training_after_diversity.yaml \\
    --log-file logs/phase2b13/phase2b13_selector_training.log
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
from llmserveopt.selector.features import FeatureMode, FEATURE_NAMES
from llmserveopt.selector.models import (
    RuleBasedSelector,
    DecisionTreeSelector,
    RandomForestSelector,
    evaluate_selector,
)
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
from run_phase2b12_workload_diversity_selector_labels import (  # noqa: E402
    build_rows_for_group,
    check_rf_feasibility,
    collect_policy_distribution,
    compute_label_distribution,
    label_diversity_summary,
    per_workload_label_table,
    policy_mean_wg,
)

SCORPIO_POLICY = "scorpio_style_slo_guard"
RULE_SELECTOR = "rule_based"
REPAIRED_RULE_SELECTOR = "rule_based_repaired"
DT_SELECTOR = "decision_tree"
RF_SELECTOR = "random_forest"


# ---------------------------------------------------------------------------
# Repaired rule selector (Phase 2B.13 fix: Rule 5 prefill → AC)
# ---------------------------------------------------------------------------

class RepairedRuleBasedSelector(RuleBasedSelector):
    """Phase 2B.13 minimal repair: Rule 5 changes prefill-heavy target.

    Change: mean_prompt > 512 OR p95_prompt > 1024 → admission_control
            (was: sarathi_style)

    Evidence (Phase 2B.12):
      - div_prefill_heavy_sarathi:   admission_control wins all 8/8 windows
      - div_prefill_moderate_tight: admission_control wins all 8/8 windows
      - Expected: sarathi_style wins; actual: AC wins (WG ≈ 1.0 in both cases)
      - Mechanism: AC's urgency sort + rejection of borderline-SLO requests achieves
        higher priority-weighted WG than sarathi's chunked-prefill throughput approach
        under this WG objective, even though WG≈1.0 for both (secondary metric improvement)

    All other rules unchanged from Phase 2B.11.
    """

    name = "rule_based_repaired"

    _POLICY_CHOICES = [
        "scorpio_style_slo_guard",
        "weighted_shortest_processing",
        "admission_control",
        "slo_slack_score",
        # "sarathi_style" removed from choices (Rule 5 now targets AC)
        "estimated_service_time_first",
        "edf",
    ]

    def predict_one(self, features: Dict[str, float]) -> str:
        g = self._get

        fraction_tight_slo    = g(features, "fraction_tight_slo", 0.0)
        min_slack             = g(features, "min_slack", float("inf"))
        recent_violation_rate = g(features, "recent_slo_violation_rate", 0.0)
        kv_utilization        = g(features, "kv_utilization", 0.0)
        mean_prompt           = g(features, "mean_prompt_tokens", 0.0)
        p95_prompt            = g(features, "p95_prompt_tokens", 0.0)
        mean_pred_output      = g(features, "mean_pred_output_tokens", 0.0)
        pred_output_cv        = g(features, "pred_output_cv", 1.0)
        burstiness_cv         = g(features, "burstiness_cv", 0.0)

        # Rule 0 (unchanged)
        if (fraction_tight_slo > 0.4 or min_slack < 1.0) and recent_violation_rate > 0.2:
            return "scorpio_style_slo_guard"

        # Rule 1 (unchanged)
        if mean_pred_output > 200 or kv_utilization > 0.7:
            return "weighted_shortest_processing"

        # Rule 2a (unchanged)
        if pred_output_cv > 2.0:
            return "scorpio_style_slo_guard"

        # Rule 2b (unchanged)
        if pred_output_cv > 1.0:
            return "admission_control"

        # Rule 3 (unchanged)
        if recent_violation_rate > 0.3:
            return "scorpio_style_slo_guard"

        # Rule 4 (unchanged)
        if fraction_tight_slo > 0.4 or min_slack < 1.0:
            return "slo_slack_score"

        # Rule 5 (REPAIRED): prefill-heavy → admission_control (was sarathi_style)
        if mean_prompt > 512 or p95_prompt > 1024:
            return "admission_control"

        # Rule 6 (unchanged)
        if mean_pred_output < 64 and pred_output_cv < 0.5:
            return "estimated_service_time_first"

        # Rule 7 (unchanged)
        if burstiness_cv > 1.5:
            return "slo_slack_score"

        # Rule 8: default (unchanged)
        return "edf"


# ---------------------------------------------------------------------------
# RF/DT training helpers
# ---------------------------------------------------------------------------

def split_rows_for_training(
    dev_rows: List[Dict],
    diversity_rows: List[Dict],
    heldout_rows: List[Dict],
    train_diversity_seeds: List[int],
    val_diversity_seeds: List[int],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split rows into train / validation / test sets.

    Train: dev rows + diversity rows from train_diversity_seeds
    Val:   diversity rows from val_diversity_seeds
    Test:  heldout_rows (never used for training or rule selection)
    """
    def seed_from_trace_id(trace_id: str) -> Optional[int]:
        parts = trace_id.rsplit("_s", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return None

    train_diversity = [
        r for r in diversity_rows
        if seed_from_trace_id(r.get("trace_id", "")) in set(train_diversity_seeds)
    ]
    val_diversity = [
        r for r in diversity_rows
        if seed_from_trace_id(r.get("trace_id", "")) in set(val_diversity_seeds)
    ]

    train_rows = dev_rows + train_diversity
    val_rows = val_diversity
    test_rows = heldout_rows
    return train_rows, val_rows, test_rows


def train_selectors(
    train_rows: List[Dict],
) -> Tuple[Optional[RandomForestSelector], Optional[DecisionTreeSelector], str]:
    """Train RF and DT selectors. Returns (rf, dt, status_message)."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return None, None, "sklearn not installed"

    if not train_rows:
        return None, None, "no training rows"

    labels = [r.get("best_policy") for r in train_rows]
    if not any(labels):
        return None, None, "no labels in training rows"

    rf = RandomForestSelector(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(train_rows)

    dt = DecisionTreeSelector(max_depth=8, min_samples_leaf=5, random_state=42)
    dt.fit(train_rows)

    return rf, dt, "ok"


def evaluate_ml_selector(
    selector,
    rows: List[Dict],
    selector_name: str,
) -> Dict:
    """Evaluate an ML selector: accuracy, WG, chosen-policy distribution."""
    if not rows:
        return {"n": 0}

    preds = selector.predict(rows)
    labels = [r.get("best_policy", "") for r in rows]
    n = len(rows)
    correct = sum(p == l for p, l in zip(preds, labels))
    acc = correct / n if n > 0 else 0.0

    wgs = []
    for pred, row in zip(preds, rows):
        wg = float(row.get(f"reward_{pred}", 0.0) or 0.0)
        wgs.append(wg)

    mean_wg = float(np.mean(wgs)) if wgs else 0.0
    dist = dict(Counter(preds))

    fixed_wgs = compute_fixed_baseline_wgs(rows)
    best_name = max(fixed_wgs, key=fixed_wgs.get) if fixed_wgs else "none"
    best_wg = fixed_wgs.get(best_name, 0.0)
    oracle_wgs = [float(r.get("best_weighted_goodput", 0.0) or 0.0) for r in rows]
    oracle_mean = float(np.mean(oracle_wgs)) if oracle_wgs else 0.0

    return {
        "selector": selector_name,
        "n_windows": n,
        "accuracy": round(acc, 4),
        "n_correct": correct,
        "mean_wg": round(mean_wg, 4),
        "best_fixed_wg": round(best_wg, 4),
        "best_fixed_policy": best_name,
        "oracle_mean_wg": round(oracle_mean, 4),
        "gap_vs_best_fixed": round(mean_wg - best_wg, 4),
        "gap_vs_oracle": round(mean_wg - oracle_mean, 4),
        "chosen_policy_dist": dist,
        "label_dist": dict(Counter(labels)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2B.13 Selector Training After Diversity"
    )
    p.add_argument(
        "--config",
        default="configs/phase2b13_selector_training_after_diversity.yaml",
    )
    p.add_argument("--log-file", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--skip-diversity", action="store_true",
                   help="Run only regression group (faster debug)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip RF/DT training")
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

    logging.info("Phase 2B.13 Selector Training After Diversity")

    cfg = load_config(args.config)
    out_dir = Path(
        args.out_dir or cfg.get("output_dir",
            "results/phase2b13_selector_training_after_diversity")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = cfg.get("phase2b12_reference", {})
    rf_thresholds = cfg.get("rf_feasibility", {})
    training_cfg = cfg.get("selector_training", {})
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
    diversity_seeds = cfg.get("diversity_seeds", [6, 7, 8, 9, 10, 11])
    train_div_seeds = training_cfg.get("train_diversity_seeds", [6, 7, 8, 9, 10])
    val_div_seeds = training_cfg.get("val_diversity_seeds", [11])

    all_workloads = cfg.get("workloads", [])
    dev_workloads = [w for w in all_workloads if w.get("group") == "dev"]
    heldout_workloads = [w for w in all_workloads if w.get("group") == "heldout"]
    diversity_workloads = [w for w in all_workloads if w.get("group") == "diversity"]

    logging.info(
        "Workloads: %d dev, %d heldout, %d diversity",
        len(dev_workloads), len(heldout_workloads), len(diversity_workloads),
    )
    logging.info("Diversity seeds: %s", diversity_seeds)
    logging.info("Train div seeds: %s  Val div seeds: %s", train_div_seeds, val_div_seeds)
    logging.info("Deployable policies: %d", len(SELECTOR_CANDIDATES))

    assert SCORPIO_POLICY in SELECTOR_CANDIDATES

    # Instantiate selectors
    rule_sel = RuleBasedSelector()
    repaired_rule_sel = RepairedRuleBasedSelector()
    base_models = {RULE_SELECTOR: rule_sel, REPAIRED_RULE_SELECTOR: repaired_rule_sel}

    t0 = time.perf_counter()

    # --- Build rows ---
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

    elapsed_build = time.perf_counter() - t0
    logging.info("Row build complete: %.1fs  dev=%d heldout=%d diversity=%d total=%d",
                 elapsed_build, len(dev_rows), len(heldout_rows),
                 len(diversity_rows), len(all_rows))

    # Apply base selectors (rule_based + repaired)
    dev_rows = apply_selectors_to_rows(dev_rows, base_models)
    heldout_rows = apply_selectors_to_rows(heldout_rows, base_models)
    diversity_rows = apply_selectors_to_rows(diversity_rows, base_models)
    regression_rows = dev_rows + heldout_rows
    all_rows = dev_rows + heldout_rows + diversity_rows

    # --- RF/DT training ---
    train_rows, val_rows, test_rows = split_rows_for_training(
        dev_rows, diversity_rows, heldout_rows,
        train_div_seeds, val_div_seeds,
    )
    logging.info("Training split: train=%d  val=%d  test=%d",
                 len(train_rows), len(val_rows), len(test_rows))

    min_w = rf_thresholds.get("min_windows", 200)
    min_pol = rf_thresholds.get("min_policies_winning", 3)
    min_win_pp = rf_thresholds.get("min_windows_per_policy", 10)
    max_frac = rf_thresholds.get("max_single_policy_fraction", 0.85)

    all_label_div = label_diversity_summary(all_rows, "overall")
    rf_decision, rf_details = check_rf_feasibility(
        all_label_div["label_distribution"],
        min_windows=min_w,
        min_policies_winning=min_pol,
        min_windows_per_policy=min_win_pp,
        max_single_policy_fraction=max_frac,
    )
    logging.info("RF/DT feasibility: %s", "FEASIBLE" if rf_decision else "NOT FEASIBLE")
    logging.info(
        "  total=%d  policies≥%d_wins=%d  top=%s(%.1f%%)",
        rf_details["total_windows"], min_win_pp,
        rf_details["n_policies_with_enough_windows"],
        rf_details["top_policy"], rf_details["top_policy_fraction"] * 100,
    )

    rf_sel = None
    dt_sel = None
    rf_train_metrics: Dict = {}
    rf_val_metrics: Dict = {}
    rf_test_metrics: Dict = {}
    dt_train_metrics: Dict = {}
    dt_val_metrics: Dict = {}
    dt_test_metrics: Dict = {}
    training_status = "skipped"

    ml_models: Dict = {}

    if rf_decision and not args.skip_training:
        logging.info("--- RF/DT training (%d rows) ---", len(train_rows))
        rf_sel, dt_sel, training_status = train_selectors(train_rows)

        if rf_sel is not None:
            logging.info("RF trained. Evaluating…")
            ml_models[RF_SELECTOR] = rf_sel
            rf_train_metrics = evaluate_ml_selector(rf_sel, train_rows, RF_SELECTOR)
            rf_val_metrics = evaluate_ml_selector(rf_sel, val_rows, RF_SELECTOR)
            rf_test_metrics = evaluate_ml_selector(rf_sel, test_rows, RF_SELECTOR)
            logging.info("  RF train acc=%.3f WG=%.4f  val acc=%.3f WG=%.4f  test acc=%.3f WG=%.4f",
                         rf_train_metrics["accuracy"], rf_train_metrics["mean_wg"],
                         rf_val_metrics.get("accuracy", 0), rf_val_metrics.get("mean_wg", 0),
                         rf_test_metrics.get("accuracy", 0), rf_test_metrics.get("mean_wg", 0))

        if dt_sel is not None:
            logging.info("DT trained. Evaluating…")
            ml_models[DT_SELECTOR] = dt_sel
            dt_train_metrics = evaluate_ml_selector(dt_sel, train_rows, DT_SELECTOR)
            dt_val_metrics = evaluate_ml_selector(dt_sel, val_rows, DT_SELECTOR)
            dt_test_metrics = evaluate_ml_selector(dt_sel, test_rows, DT_SELECTOR)
            logging.info("  DT train acc=%.3f WG=%.4f  val acc=%.3f WG=%.4f  test acc=%.3f WG=%.4f",
                         dt_train_metrics["accuracy"], dt_train_metrics["mean_wg"],
                         dt_val_metrics.get("accuracy", 0), dt_val_metrics.get("mean_wg", 0),
                         dt_test_metrics.get("accuracy", 0), dt_test_metrics.get("mean_wg", 0))
    else:
        if not rf_decision:
            logging.info("RF/DT training skipped: feasibility criteria not met")
        elif args.skip_training:
            logging.info("RF/DT training skipped: --skip-training flag")
        training_status = "not_feasible" if not rf_decision else "user_skipped"

    # Apply ML selectors to all rows (if trained)
    all_models = {**base_models, **ml_models}
    if ml_models:
        dev_rows = apply_selectors_to_rows(dev_rows, ml_models)
        heldout_rows = apply_selectors_to_rows(heldout_rows, ml_models)
        diversity_rows = apply_selectors_to_rows(diversity_rows, ml_models)
        regression_rows = dev_rows + heldout_rows
        all_rows = dev_rows + heldout_rows + diversity_rows

    elapsed = time.perf_counter() - t0
    logging.info("Total elapsed: %.1fs", elapsed)

    # --- Group summaries ---
    dev_summary = summarize_group(dev_rows, "dev", all_models)
    heldout_summary = summarize_group(heldout_rows, "heldout", all_models)
    regression_summary = summarize_group(regression_rows, "regression", all_models)
    diversity_summary = summarize_group(diversity_rows, "diversity", all_models) if diversity_rows else {}
    overall_summary = summarize_group(all_rows, "overall", all_models)

    # --- Label diversity ---
    regression_label_div = label_diversity_summary(regression_rows, "regression")
    diversity_label_div = label_diversity_summary(diversity_rows, "diversity") if diversity_rows else {}
    train_label_div = label_diversity_summary(train_rows, "train")
    val_label_div = label_diversity_summary(val_rows, "val")
    test_label_div = label_diversity_summary(test_rows, "test")

    # --- Per-workload table ---
    workload_table = per_workload_label_table(all_rows)

    # Count meaningful vs trivial windows (best_fixed_wg < 0.99)
    meaningful = [r for r in all_rows
                  if float(r.get("best_weighted_goodput", 1.0) or 1.0) < 0.99]
    trivial = [r for r in all_rows
               if float(r.get("best_weighted_goodput", 1.0) or 1.0) >= 0.99]

    # --- Outputs ---
    write_per_window_csv(all_rows, out_dir / "per_window.csv")

    # Selector comparison
    comparison_rows = []
    for gname, gsummary, grows in [
        ("dev", dev_summary, dev_rows),
        ("heldout", heldout_summary, heldout_rows),
        ("regression", regression_summary, regression_rows),
        ("diversity", diversity_summary, diversity_rows),
        ("overall", overall_summary, all_rows),
    ]:
        if not gsummary.get("n_windows"):
            continue
        row = {
            "group": gname,
            "n_windows": gsummary["n_windows"],
            "best_fixed_policy": gsummary.get("best_fixed_policy"),
            "best_fixed_wg": gsummary.get("best_fixed_mean_wg"),
            "oracle_per_window_best_wg": gsummary.get("oracle_per_window_best_mean_wg"),
        }
        for sel in all_models:
            row[f"{sel}_wg"] = gsummary.get(f"sel_{sel}_mean_wg")
            row[f"{sel}_gap_vs_best_fixed"] = gsummary.get(f"sel_{sel}_gap_vs_best_fixed")
            row[f"{sel}_gap_vs_oracle"] = gsummary.get(f"sel_{sel}_gap_vs_oracle")
        comparison_rows.append(row)
    write_summary_csv(comparison_rows, out_dir / "selector_comparison.csv")

    # Label distribution CSV
    all_policies_in_labels: set = set()
    for ld in [all_label_div, regression_label_div, diversity_label_div, train_label_div]:
        all_policies_in_labels.update(ld.get("label_distribution", {}).keys())
    label_rows_out = []
    for policy in sorted(all_policies_in_labels):
        label_rows_out.append({
            "policy": policy,
            "overall_wins": all_label_div["label_distribution"].get(policy, 0),
            "regression_wins": regression_label_div.get("label_distribution", {}).get(policy, 0),
            "diversity_wins": diversity_label_div.get("label_distribution", {}).get(policy, 0) if diversity_label_div else 0,
            "train_wins": train_label_div.get("label_distribution", {}).get(policy, 0),
        })
    label_rows_out.sort(key=lambda r: -r["overall_wins"])
    write_summary_csv(label_rows_out, out_dir / "label_distribution.csv")

    # Per-workload labels
    with open(out_dir / "per_workload_labels.json", "w") as f:
        json.dump(workload_table, f, indent=2)

    # Rule selector policy distributions
    all_sel_dist = collect_policy_distribution(all_rows, RULE_SELECTOR)
    repaired_sel_dist = collect_policy_distribution(all_rows, REPAIRED_RULE_SELECTOR)

    # Policy ranking
    rank_rows = []
    for gname, grows in [
        ("dev", dev_rows), ("heldout", heldout_rows),
        ("regression", regression_rows), ("diversity", diversity_rows),
        ("overall", all_rows),
    ]:
        if not grows:
            continue
        wg_by_policy = {p: policy_mean_wg(grows, p) for p in SELECTOR_CANDIDATES}
        ranked = sorted(wg_by_policy.items(), key=lambda x: -x[1])
        for rank, (pname, wg) in enumerate(ranked, 1):
            rank_rows.append({"group": gname, "rank": rank, "policy": pname, "mean_wg": wg})
    write_summary_csv(rank_rows, out_dir / "policy_ranking.csv")

    # RF/DT training summary JSON
    rf_dt_summary = {
        "training_status": training_status,
        "rf_feasibility_decision": rf_decision,
        "rf_feasibility_details": rf_details,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "train_label_distribution": train_label_div.get("label_distribution", {}),
        "val_label_distribution": val_label_div.get("label_distribution", {}),
        "test_label_distribution": test_label_div.get("label_distribution", {}),
        "rf_metrics": {
            "train": rf_train_metrics,
            "val": rf_val_metrics,
            "test": rf_test_metrics,
        },
        "dt_metrics": {
            "train": dt_train_metrics,
            "val": dt_val_metrics,
            "test": dt_test_metrics,
        },
    }
    with open(out_dir / "rf_dt_training_summary.json", "w") as f:
        json.dump(rf_dt_summary, f, indent=2, default=str)

    # Metadata
    metadata = {
        "experiment": "phase2b13_selector_training_after_diversity",
        "n_deployable_policies": len(SELECTOR_CANDIDATES),
        "n_workloads": {
            "dev": len(dev_workloads),
            "heldout": len(heldout_workloads),
            "diversity": len(diversity_workloads),
        },
        "n_windows": {
            "dev": len(dev_rows),
            "heldout": len(heldout_rows),
            "regression": len(regression_rows),
            "diversity": len(diversity_rows),
            "total": len(all_rows),
            "meaningful_wg_gap": len(meaningful),
            "trivial_all_complete": len(trivial),
        },
        "phase2b12_reference": ref,
        "label_diversity": {
            "overall": all_label_div,
            "regression": regression_label_div,
            "diversity": diversity_label_div if diversity_label_div else {},
            "train": train_label_div,
            "val": val_label_div,
            "test": test_label_div,
        },
        "rf_feasibility": rf_details,
        "rf_training_done": rf_sel is not None,
        "dt_training_done": dt_sel is not None,
        "rule_selector_dispatch": {
            "original": all_sel_dist,
            "repaired": repaired_sel_dist,
        },
        "rule_repair_applied": True,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for fname, summary in [
        ("dev_summary.json", dev_summary),
        ("heldout_summary.json", heldout_summary),
        ("diversity_summary.json", diversity_summary),
        ("overall_summary.json", overall_summary),
    ]:
        with open(out_dir / fname, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    # --- Console output ---
    logging.info("=" * 60)
    logging.info("Phase 2B.13 — Results")
    logging.info("=" * 60)
    logging.info("Windows: total=%d  meaningful=%d  trivial=%d",
                 len(all_rows), len(meaningful), len(trivial))
    for gname, gsummary in [
        ("dev", dev_summary), ("heldout", heldout_summary),
        ("regression", regression_summary), ("diversity", diversity_summary),
        ("overall", overall_summary),
    ]:
        if not gsummary.get("n_windows"):
            continue
        bf = gsummary.get("best_fixed_mean_wg", float("nan"))
        rb = gsummary.get(f"sel_{RULE_SELECTOR}_mean_wg", float("nan"))
        rr = gsummary.get(f"sel_{REPAIRED_RULE_SELECTOR}_mean_wg", float("nan"))
        rf_wg = gsummary.get(f"sel_{RF_SELECTOR}_mean_wg", None)
        dt_wg = gsummary.get(f"sel_{DT_SELECTOR}_mean_wg", None)
        msg = (f"[{gname}] n={gsummary['n_windows']} best_fixed={bf:.4f} "
               f"rule={rb:.4f} rule_repaired={rr:.4f}")
        if rf_wg is not None:
            msg += f" RF={rf_wg:.4f}"
        if dt_wg is not None:
            msg += f" DT={dt_wg:.4f}"
        logging.info(msg)

    logging.info("Label distribution (overall, top-5): %s",
                 dict(Counter(all_label_div["label_distribution"]).most_common(5)))
    logging.info("RF/DT feasible: %s  trained: %s",
                 rf_decision, rf_sel is not None)
    if rf_sel is not None:
        logging.info("RF test: acc=%.3f  WG=%.4f  gap_vs_fixed=%+.4f",
                     rf_test_metrics.get("accuracy", 0),
                     rf_test_metrics.get("mean_wg", 0),
                     rf_test_metrics.get("gap_vs_best_fixed", 0))
        logging.info("DT test: acc=%.3f  WG=%.4f  gap_vs_fixed=%+.4f",
                     dt_test_metrics.get("accuracy", 0),
                     dt_test_metrics.get("mean_wg", 0),
                     dt_test_metrics.get("gap_vs_best_fixed", 0))
    logging.info("Rule selector dispatch (original vs repaired):")
    logging.info("  original: %s", all_sel_dist)
    logging.info("  repaired: %s", repaired_sel_dist)
    logging.info("Outputs → %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
