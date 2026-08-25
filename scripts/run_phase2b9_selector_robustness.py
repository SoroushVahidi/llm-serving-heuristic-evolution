#!/usr/bin/env python3
"""
Phase 2B.9 Selector Robustness Evaluation.

Evaluates the repaired rule selector, Phase 2A.4 RF selector, and Phase 2A.4 DT selector
on broader development and held-out robustness workloads.

Usage
-----
python scripts/run_phase2b9_selector_robustness.py \\
    --config configs/phase2b9_selector_robustness.yaml \\
    --log-file logs/phase2b9/phase2b9_selector_robustness.log

The script
----------
1. Reads the Phase 2B.9 robustness config (workloads tagged dev_* and heldout_*).
2. For dev group: iterates over dev_seeds (default [0,1,2]).
3. For heldout group: iterates over heldout_seeds (default [3,4,5]).
4. For each (workload, seed):
   a. Generates or loads trace.
   b. Splits into windows of 200 requests.
   c. Extracts online-observable features per window.
   d. Runs all 19 deployable policies on each window (isolated simulation).
   e. Computes per-policy WG per window.
   f. Labels: best deployable policy per window (oracle_srtf excluded).
5. Applies all selectors to each window's features:
   - rule_based: Phase 2B.8 repaired rule selector (deterministic)
   - random_forest: Phase 2A.4 trained RF model
   - decision_tree: Phase 2A.4 trained DT model
6. For each selector, looks up chosen-policy's WG in that window.
7. Reports metrics per group (dev vs heldout) and overall.

Outputs
-------
results/phase2b9_selector_robustness/
  per_window.csv        — per-window results for all selectors
  dev_summary.csv       — dev group summary
  heldout_summary.csv   — heldout group summary
  selector_comparison.csv — final comparison table
  policy_distribution.csv — chosen-policy distribution per selector
  metadata.json         — run metadata
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.core.types import GPUConfig
from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.dataset import DatasetConfig, build_selector_dataset
from llmserveopt.selector.features import FeatureMode
from llmserveopt.selector.models import (
    RuleBasedSelector,
    RandomForestSelector,
    DecisionTreeSelector,
)
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.service_model_factory import build_service_model_from_config
from llmserveopt.workloads.synthetic import WorkloadConfig, generate_workload


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_gpu_configs(cfg: dict) -> List[GPUConfig]:
    return [
        GPUConfig(
            gpu_id=g["gpu_id"],
            max_active_sequences=g["max_active_sequences"],
            max_batch_tokens=g["max_batch_tokens"],
            max_kv_tokens=g["max_kv_tokens"],
        )
        for g in cfg["gpus"]
    ]


def build_workload_config(w: dict) -> WorkloadConfig:
    kwargs = dict(w)
    tag = kwargs.pop("tag", "workload")
    kwargs.pop("group", None)
    kwargs.pop("source", None)
    kwargs.pop("trace_path", None)
    kwargs.pop("max_requests", None)
    raw_classes = kwargs.pop("slo_classes", None)
    if raw_classes is not None:
        from llmserveopt.workloads.synthetic import SLOClass
        slo_classes = [
            SLOClass(
                class_id=c["class_id"],
                slo_slack=c["slo_slack"],
                priority=c["priority"],
                weight=c["weight"],
            )
            for c in raw_classes
        ]
        kwargs["slo_classes"] = slo_classes
    return WorkloadConfig(tag=tag, **kwargs)


def _resolve_trace_path(raw_path: str) -> Path:
    """Resolve a (possibly relative) trace path.

    Search order:
    1. As-is (absolute, or relative to cwd).
    2. Relative to the script's project root (scripts/../).
    3. Walk up from project root to find a directory that contains
       data/processed/ (handles git worktree layouts).
    """
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    # Relative to cwd
    if p.exists():
        return p.resolve()

    # Relative to project root (scripts' parent)
    script_root = Path(__file__).parent.parent
    candidate = script_root / p
    if candidate.exists():
        return candidate

    # Walk up looking for the canonical data/ directory (worktree fallback)
    walk = script_root
    for _ in range(6):
        candidate = walk / p
        if candidate.exists():
            return candidate
        if walk.parent == walk:
            break
        walk = walk.parent

    raise FileNotFoundError(
        f"Trace file not found: {raw_path}\n"
        f"  Tried: cwd, {script_root}, and parent directories."
    )


def load_or_generate_trace(wdef: dict, seed: int):
    """Load trace from file or generate synthetic. Returns [] on missing trace (skip)."""
    source = wdef.get("source", "synthetic")
    max_req = wdef.get("max_requests", None)
    tag = wdef.get("tag", "trace")

    if source == "extended_jsonl":
        from llmserveopt.workloads.trace_io_extended import load_extended_jsonl
        try:
            trace_path = _resolve_trace_path(wdef["trace_path"])
        except FileNotFoundError as e:
            logging.warning("  %s — skipping workload %s", e, tag)
            return []
        logging.info("  Loading extended JSONL from %s", trace_path)
        reqs, _ = load_extended_jsonl(str(trace_path))
    elif source == "trace_file":
        from llmserveopt.workloads.trace_io import load_jsonl
        try:
            trace_path = _resolve_trace_path(wdef["trace_path"])
        except FileNotFoundError as e:
            logging.warning("  %s — skipping workload %s", e, tag)
            return []
        logging.info("  Loading trace from %s", trace_path)
        reqs = load_jsonl(str(trace_path))
    else:
        cfg = build_workload_config(wdef)
        logging.info("  Generating synthetic: tag=%s rate=%.1f dur=%.1fs seed=%d",
                     cfg.tag, cfg.arrival_rate, cfg.duration, seed)
        reqs = generate_workload(cfg, seed=seed)

    if max_req is not None and len(reqs) > max_req:
        reqs = reqs[:max_req]
        logging.info("  Trimmed to %d requests", len(reqs))

    return reqs


# ---------------------------------------------------------------------------
# Selector model loading
# ---------------------------------------------------------------------------

def load_selector_models(models_dir: str):
    """Load Phase 2A.4 RF and DT models. Returns dict of name→model."""
    models = {}
    models_path = Path(models_dir)

    # Rule-based selector (no training needed)
    models["rule_based"] = RuleBasedSelector()
    logging.info("Loaded rule_based selector (Phase 2B.8 repair)")

    # RF
    rf_path = models_path / "random_forest" / "model.joblib"
    if rf_path.exists():
        try:
            models["random_forest"] = RandomForestSelector.load(str(rf_path))
            logging.info("Loaded random_forest from %s", rf_path)
        except Exception as e:
            logging.warning("Could not load random_forest: %s", e)
    else:
        logging.warning("RF model not found at %s — skipping", rf_path)

    # DT
    dt_path = models_path / "decision_tree" / "model.joblib"
    if dt_path.exists():
        try:
            models["decision_tree"] = DecisionTreeSelector.load(str(dt_path))
            logging.info("Loaded decision_tree from %s", dt_path)
        except Exception as e:
            logging.warning("Could not load decision_tree: %s", e)
    else:
        logging.warning("DT model not found at %s — skipping", dt_path)

    return models


# ---------------------------------------------------------------------------
# Per-window selector application
# ---------------------------------------------------------------------------

def apply_selectors_to_rows(rows: List[Dict], models: Dict) -> List[Dict]:
    """
    For each window row, apply all selectors and record chosen policy + WG.

    Returns enriched rows with extra columns:
      sel_<name>_policy  — policy chosen by selector <name>
      sel_<name>_wg      — WG of chosen policy
      sel_<name>_correct — 1 if chosen == best_policy, else 0
    """
    if not rows:
        return rows

    enriched = []
    for row in rows:
        r = dict(row)
        features = {k: float(v) for k, v in row.items()
                    if k.startswith("feat_") and v not in ("", None)}

        best_policy = row.get("best_policy", "")

        for sel_name, model in models.items():
            if hasattr(model, "predict_one"):
                chosen = model.predict_one(features)
            else:
                # ML models need list input
                chosen_list = model.predict([row])
                chosen = chosen_list[0] if chosen_list else best_policy

            reward_key = f"reward_{chosen}"
            chosen_wg = float(row.get(reward_key, 0.0) or 0.0)
            r[f"sel_{sel_name}_policy"] = chosen
            r[f"sel_{sel_name}_wg"] = chosen_wg
            r[f"sel_{sel_name}_correct"] = int(chosen == best_policy)

        enriched.append(r)
    return enriched


# ---------------------------------------------------------------------------
# Per-policy summary (fixed baselines)
# ---------------------------------------------------------------------------

def compute_fixed_baseline_wgs(rows: List[Dict]) -> Dict[str, float]:
    """Return mean WG over all windows for each fixed policy."""
    totals: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if k.startswith("reward_") and v not in ("", None):
                policy = k[len("reward_"):]
                try:
                    totals[policy].append(float(v))
                except (ValueError, TypeError):
                    pass
    return {p: float(np.mean(wgs)) for p, wgs in totals.items() if wgs}


def best_fixed_policy(rows: List[Dict]) -> tuple:
    """Return (best_policy_name, best_mean_wg) over all windows."""
    fixed = compute_fixed_baseline_wgs(rows)
    if not fixed:
        return ("none", 0.0)
    best = max(fixed, key=lambda p: fixed[p])
    return (best, fixed[best])


# ---------------------------------------------------------------------------
# Group summarization
# ---------------------------------------------------------------------------

def summarize_group(rows: List[Dict], group_name: str, models: Dict) -> Dict:
    """Compute summary metrics for a group of windows."""
    if not rows:
        return {"group": group_name, "n_windows": 0}

    n = len(rows)
    per_window_best_wgs = [float(r.get("best_weighted_goodput", 0.0) or 0.0) for r in rows]
    oracle_mean_wg = float(np.mean(per_window_best_wgs))

    best_fixed_name, best_fixed_wg = best_fixed_policy(rows)

    summary = {
        "group": group_name,
        "n_windows": n,
        "oracle_per_window_best_mean_wg": round(oracle_mean_wg, 4),
        "best_fixed_policy": best_fixed_name,
        "best_fixed_mean_wg": round(best_fixed_wg, 4),
    }

    # Per-selector metrics
    for sel_name in models:
        wgs = [float(r.get(f"sel_{sel_name}_wg", 0.0) or 0.0) for r in rows]
        corrects = [int(r.get(f"sel_{sel_name}_correct", 0) or 0) for r in rows]
        mean_wg = float(np.mean(wgs))
        accuracy = float(np.mean(corrects))
        gap_to_fixed = mean_wg - best_fixed_wg
        gap_to_oracle = mean_wg - oracle_mean_wg
        policy_dist = Counter(r.get(f"sel_{sel_name}_policy", "") for r in rows)

        summary[f"sel_{sel_name}_mean_wg"] = round(mean_wg, 4)
        summary[f"sel_{sel_name}_accuracy"] = round(accuracy, 4)
        summary[f"sel_{sel_name}_gap_vs_best_fixed"] = round(gap_to_fixed, 4)
        summary[f"sel_{sel_name}_gap_vs_oracle"] = round(gap_to_oracle, 4)
        summary[f"sel_{sel_name}_policy_dist"] = dict(policy_dist)

    # Per-workload breakdown
    workload_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        tag = r.get("trace_id", "unknown").replace("_s0", "").replace("_s1", "").replace("_s2", "")
        workload_groups[tag].append(r)

    summary["per_workload"] = {}
    for wl_tag, wl_rows in sorted(workload_groups.items()):
        wl_fixed = compute_fixed_baseline_wgs(wl_rows)
        wl_best_name = max(wl_fixed, key=lambda p: wl_fixed[p]) if wl_fixed else "none"
        wl_best_wg = wl_fixed.get(wl_best_name, 0.0)
        wl_oracle = float(np.mean([float(r.get("best_weighted_goodput", 0.0) or 0.0) for r in wl_rows]))
        wl_entry = {
            "n_windows": len(wl_rows),
            "best_fixed_policy": wl_best_name,
            "best_fixed_wg": round(wl_best_wg, 4),
            "oracle_wg": round(wl_oracle, 4),
        }
        for sel_name in models:
            wl_sel_wgs = [float(r.get(f"sel_{sel_name}_wg", 0.0) or 0.0) for r in wl_rows]
            wl_sel_policy_choices = Counter(r.get(f"sel_{sel_name}_policy", "") for r in wl_rows)
            wl_entry[f"sel_{sel_name}_wg"] = round(float(np.mean(wl_sel_wgs)), 4)
            wl_entry[f"sel_{sel_name}_policy"] = wl_sel_policy_choices.most_common(1)[0][0] if wl_sel_policy_choices else ""
        summary["per_workload"][wl_tag] = wl_entry

    return summary


# ---------------------------------------------------------------------------
# CSV writing helpers
# ---------------------------------------------------------------------------

def write_per_window_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Wrote %d rows to %s", len(rows), path)


def write_summary_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Wrote %d summary rows to %s", len(rows), path)


def print_group_summary(group_name: str, summary: Dict, models: Dict) -> None:
    logging.info("\n=== %s ===", group_name)
    logging.info("  Windows: %d", summary.get("n_windows", 0))
    logging.info("  Per-window oracle WG: %.4f", summary.get("oracle_per_window_best_mean_wg", 0))
    logging.info("  Best fixed policy: %s (WG=%.4f)",
                 summary.get("best_fixed_policy", "?"), summary.get("best_fixed_mean_wg", 0))

    for sel_name in models:
        sel_wg = summary.get(f"sel_{sel_name}_mean_wg", 0)
        sel_acc = summary.get(f"sel_{sel_name}_accuracy", 0)
        gap = summary.get(f"sel_{sel_name}_gap_vs_best_fixed", 0)
        logging.info("  [%s] WG=%.4f acc=%.2f%% gap_vs_fixed=%.4f",
                     sel_name, sel_wg, sel_acc * 100, gap)

    if "per_workload" in summary:
        logging.info("  Per-workload:")
        for wl_tag, wl_info in summary["per_workload"].items():
            logging.info("    [%s] n=%d best_fixed=%s(%.3f) oracle=%.3f",
                         wl_tag, wl_info["n_windows"],
                         wl_info["best_fixed_policy"], wl_info["best_fixed_wg"],
                         wl_info["oracle_wg"])
            for sel_name in models:
                logging.info("      %s → %s (WG=%.3f)",
                             sel_name, wl_info.get(f"sel_{sel_name}_policy", "?"),
                             wl_info.get(f"sel_{sel_name}_wg", 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2B.9 Selector Robustness Evaluation")
    p.add_argument("--config", default="configs/phase2b9_selector_robustness.yaml")
    p.add_argument("--log-file", default=None,
                   help="Log file path (default: log to stdout only)")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory from config")
    p.add_argument("--models-dir", default=None,
                   help="Override selector models directory from config")
    p.add_argument("--skip-heldout", action="store_true",
                   help="Only run dev group (faster for debugging)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Logging setup
    log_handlers = [logging.StreamHandler()]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=log_handlers,
    )

    logging.info("Phase 2B.9 Selector Robustness Evaluation")
    logging.info("Config: %s", args.config)

    cfg = load_config(args.config)

    out_dir = Path(args.out_dir or cfg.get("output_dir", "results/phase2b9_selector_robustness"))
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = args.models_dir or cfg.get(
        "selector_models_dir", "results/phase2a4_2b4_final_eval/selector_models"
    )

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

    # Split workloads by group
    all_workloads = cfg.get("workloads", [])
    dev_workloads = [w for w in all_workloads if w.get("group") == "dev"]
    heldout_workloads = [w for w in all_workloads if w.get("group") == "heldout"]

    logging.info("Dev workloads: %d, seeds: %s", len(dev_workloads), dev_seeds)
    logging.info("Heldout workloads: %d, seeds: %s", len(heldout_workloads), heldout_seeds)

    # Load selector models
    models = load_selector_models(models_dir)
    logging.info("Loaded %d selector models: %s", len(models), list(models.keys()))

    # ---------------------------------------------------------------------------
    # Run dataset build for dev group
    # ---------------------------------------------------------------------------
    logging.info("\n--- Building dev group dataset ---")
    dev_rows: List[Dict] = []
    t0_dev = time.perf_counter()

    for wdef in dev_workloads:
        tag = wdef.get("tag", "workload")
        source = wdef.get("source", "synthetic")

        for seed in dev_seeds:
            trace_tag = f"{tag}_s{seed}"
            logging.info("[dev] %s", trace_tag)
            reqs = load_or_generate_trace(wdef, seed=seed)
            if not reqs:
                logging.warning("  Empty trace, skipping")
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
                verbose=args.verbose,
            )
            rows = build_selector_dataset(reqs, dataset_cfg)
            logging.info("  %d windows", len(rows))
            dev_rows.extend(rows)

    dev_elapsed = time.perf_counter() - t0_dev
    logging.info("Dev group: %d windows in %.1fs", len(dev_rows), dev_elapsed)

    # ---------------------------------------------------------------------------
    # Run dataset build for heldout group
    # ---------------------------------------------------------------------------
    heldout_rows: List[Dict] = []
    if not args.skip_heldout:
        logging.info("\n--- Building heldout group dataset ---")
        t0_held = time.perf_counter()

        for wdef in heldout_workloads:
            tag = wdef.get("tag", "workload")
            source = wdef.get("source", "synthetic")

            # BurstGPT trace: use seed range for repeatability but trace is fixed
            seeds_to_use = heldout_seeds if source == "synthetic" else [heldout_seeds[0]]
            for seed in seeds_to_use:
                trace_tag = f"{tag}_s{seed}"
                logging.info("[heldout] %s", trace_tag)
                reqs = load_or_generate_trace(wdef, seed=seed)
                if not reqs:
                    logging.warning("  Empty trace, skipping")
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
                    verbose=args.verbose,
                )
                rows = build_selector_dataset(reqs, dataset_cfg)
                logging.info("  %d windows", len(rows))
                heldout_rows.extend(rows)

        held_elapsed = time.perf_counter() - t0_held
        logging.info("Heldout group: %d windows in %.1fs", len(heldout_rows), held_elapsed)

    # ---------------------------------------------------------------------------
    # Apply selectors
    # ---------------------------------------------------------------------------
    logging.info("\n--- Applying selectors ---")
    dev_rows = apply_selectors_to_rows(dev_rows, models)
    heldout_rows = apply_selectors_to_rows(heldout_rows, models)
    all_rows = dev_rows + heldout_rows

    # ---------------------------------------------------------------------------
    # Summarize
    # ---------------------------------------------------------------------------
    dev_summary = summarize_group(dev_rows, "dev", models)
    heldout_summary = summarize_group(heldout_rows, "heldout", models)
    overall_summary = summarize_group(all_rows, "overall", models)

    print_group_summary("Development/Regression Group", dev_summary, models)
    if heldout_rows:
        print_group_summary("Held-Out Robustness Group", heldout_summary, models)
    print_group_summary("Overall", overall_summary, models)

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------
    # Per-window CSV
    write_per_window_csv(all_rows, out_dir / "per_window.csv")

    # Flat summary rows for CSV
    def flat_summary_row(summary: Dict, group: str) -> Dict:
        row = {"group": group}
        row["n_windows"] = summary.get("n_windows", 0)
        row["oracle_per_window_best_wg"] = summary.get("oracle_per_window_best_mean_wg", 0)
        row["best_fixed_policy"] = summary.get("best_fixed_policy", "")
        row["best_fixed_wg"] = summary.get("best_fixed_mean_wg", 0)
        for sel_name in models:
            row[f"{sel_name}_wg"] = summary.get(f"sel_{sel_name}_mean_wg", 0)
            row[f"{sel_name}_accuracy"] = summary.get(f"sel_{sel_name}_accuracy", 0)
            row[f"{sel_name}_gap_vs_fixed"] = summary.get(f"sel_{sel_name}_gap_vs_best_fixed", 0)
            row[f"{sel_name}_gap_vs_oracle"] = summary.get(f"sel_{sel_name}_gap_vs_oracle", 0)
        return row

    summary_rows = [flat_summary_row(dev_summary, "dev")]
    if heldout_rows:
        summary_rows.append(flat_summary_row(heldout_summary, "heldout"))
    summary_rows.append(flat_summary_row(overall_summary, "overall"))

    if summary_rows:
        write_summary_csv(summary_rows, out_dir / "selector_comparison.csv")

    # Per-workload summary rows
    per_wl_rows = []
    for group_name, grp_summary in [("dev", dev_summary), ("heldout", heldout_summary), ("overall", overall_summary)]:
        for wl_tag, wl_info in grp_summary.get("per_workload", {}).items():
            row = {"group": group_name, "workload": wl_tag}
            row.update({k: v for k, v in wl_info.items() if not isinstance(v, dict)})
            per_wl_rows.append(row)
    if per_wl_rows:
        write_summary_csv(per_wl_rows, out_dir / "per_workload_summary.csv")

    # Policy distribution per selector
    pol_dist_rows = []
    for sel_name in models:
        for group_name, grp_summary in [("dev", dev_summary), ("heldout", heldout_summary)]:
            dist = grp_summary.get(f"sel_{sel_name}_policy_dist", {})
            for policy, count in sorted(dist.items(), key=lambda x: -x[1]):
                pol_dist_rows.append({
                    "group": group_name,
                    "selector": sel_name,
                    "chosen_policy": policy,
                    "count": count,
                })
    if pol_dist_rows:
        write_summary_csv(pol_dist_rows, out_dir / "policy_distribution.csv")

    # Metadata JSON
    metadata = {
        "experiment": "phase2b9_selector_robustness",
        "config": args.config,
        "models_dir": models_dir,
        "selector_models_loaded": list(models.keys()),
        "dev_workloads": [w.get("tag") for w in dev_workloads],
        "dev_seeds": dev_seeds,
        "heldout_workloads": [w.get("tag") for w in heldout_workloads],
        "heldout_seeds": heldout_seeds,
        "n_dev_windows": len(dev_rows),
        "n_heldout_windows": len(heldout_rows),
        "window_size": window_size,
        "feature_mode": feature_mode.value,
        "n_selector_candidates": len(SELECTOR_CANDIDATES),
        "oracle_excluded": "oracle_srtf not in SELECTOR_CANDIDATES",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Full summaries JSON
    with open(out_dir / "dev_summary.json", "w") as f:
        json.dump(dev_summary, f, indent=2, default=str)
    with open(out_dir / "heldout_summary.json", "w") as f:
        json.dump(heldout_summary, f, indent=2, default=str)

    logging.info("\nAll outputs written to: %s", out_dir)
    logging.info("Phase 2B.9 robustness evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
