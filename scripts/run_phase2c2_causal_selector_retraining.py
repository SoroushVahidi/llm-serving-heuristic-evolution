#!/usr/bin/env python3
"""
Phase 2C.2: Causal-feature selector retraining and real-trace evaluation.

Workflow
--------
1. Rebuild Phase 2B.13 synthetic training rows with feature_mode: causal.
2. Train deployable selectors on the causal train split only.
3. Evaluate on Phase 2C.1 real-trace workloads (never used for training).
4. Write deployable summaries (ANWG-primary) and external-baseline failure analysis.

Modes
-----
--dry-run   Validate configs, splits, and workload separation; no writes.
--smoke     Tiny causal retrain + tiny real-trace eval.
--allow-full-run  Full causal retrain + six-workload real-trace evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.selector.features import (
    FEATURE_NAMES,
    parse_feature_mode,
    feature_mode_is_deployable,
)
from llmserveopt.selector.roles import (
    DEPLOYABLE_LEARNED_SELECTORS,
    EXTERNAL_STYLE_BASELINES,
    PRIMARY_RANK_METRIC,
    classify_selectors,
    is_deployable_headline_selector,
    is_external_style_baseline,
    is_oracle_assisted_selector,
    selector_role,
)
from llmserveopt.simulator.service_model_factory import build_service_model_from_config

from run_phase2b9_selector_robustness import (
    apply_selectors_to_rows,
    build_gpu_configs,
    load_config,
    write_per_window_csv,
)
from run_phase2b12_workload_diversity_selector_labels import build_rows_for_group
from run_phase2b15_corrected_objective_selector_retraining import (
    _anwg,
    _comp_frac,
    _cond_wg,
    relabel_rows,
    split_rows,
)
from run_phase2b16_fresh_corrected_objective_validation import (
    evaluate_fresh_selector,
    train_selectors_from_rows,
)
from run_phase2c1_real_trace_ingestion_validation import (
    _annotate_groups,
    _build_deployable_headline_rows,
    _evaluate_selector_summary,
    _flatten_dict,
    _write_csv,
    _write_json,
    build_smoke_workloads,
    ensure_azure_2023_inputs,
)

DEFAULT_CONFIG = "configs/phase2c2_causal_selector_retraining.yaml"
DEFAULT_OUTPUT_DIR = "results/phase2c2_causal_selector_retraining"
DEFAULT_LOG_FILE = "logs/phase2c2_causal_selector_retraining/phase2c2_causal.log"
SMOKE_LOG_FILE = "logs/phase2c2_causal_selector_retraining/phase2c2_smoke.log"

REQUIRED_TOP_LEVEL_KEYS = [
    "experiment",
    "output_dir",
    "training_config",
    "evaluation_config",
    "feature_mode",
    "simulator",
    "service_model",
    "gpus",
    "window_size",
    "min_partial_window",
    "selector_training",
    "evaluation_workload_tags",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2C.2 causal selector retraining")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--log-file", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-full-run", action="store_true")
    p.add_argument(
        "--allow-azure-download",
        action="store_true",
        help="Not used by default; Azure traces must exist locally",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logging(log_file: Optional[str], verbose: bool) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = _repo_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _load_nested_config(path: str) -> dict:
    return load_config(_repo_path(path))


def _eval_tags(cfg: dict) -> Set[str]:
    return set(cfg.get("evaluation_workload_tags", []))


def _training_prefixes(cfg: dict) -> List[str]:
    return list(cfg.get("training_trace_prefixes", ["dev_", "heldout_", "div_"]))


def _trace_tag(trace_id: str) -> str:
    return trace_id.rsplit("_s", 1)[0] if "_s" in trace_id else trace_id


def validate_phase2c2_config(cfg: dict) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            issues.append(f"missing top-level key: {key}")

    feature_mode_raw = cfg.get("feature_mode", "causal")
    try:
        feature_mode = parse_feature_mode(feature_mode_raw)
    except ValueError:
        issues.append(f"unsupported feature_mode: {feature_mode_raw}")
        feature_mode = None
    else:
        if not feature_mode_is_deployable(feature_mode):
            issues.append("feature_mode must be 'causal' for Phase 2C.2")

    train_cfg_path = cfg.get("training_config")
    eval_cfg_path = cfg.get("evaluation_config")
    train_cfg = eval_cfg = None
    if train_cfg_path:
        tp = _repo_path(train_cfg_path)
        if not tp.exists():
            issues.append(f"training_config not found: {train_cfg_path}")
        else:
            train_cfg = _load_nested_config(train_cfg_path)
    if eval_cfg_path:
        ep = _repo_path(eval_cfg_path)
        if not ep.exists():
            issues.append(f"evaluation_config not found: {eval_cfg_path}")
        else:
            eval_cfg = _load_nested_config(eval_cfg_path)

    eval_tags = _eval_tags(cfg)
    train_workloads: List[Dict[str, Any]] = []
    eval_workloads: List[Dict[str, Any]] = []
    overlap: List[str] = []

    if train_cfg:
        for w in train_cfg.get("workloads", []):
            tag = w.get("tag", "")
            train_workloads.append({"tag": tag, "group": w.get("group", "")})
            if tag in eval_tags:
                overlap.append(tag)

    if eval_cfg:
        for w in eval_cfg.get("workloads", []):
            tag = w.get("tag", "")
            exists = bool(w.get("trace_path") and _repo_path(w["trace_path"]).exists())
            eval_workloads.append({
                "tag": tag,
                "group": w.get("group", ""),
                "trace_exists": exists,
            })
            if tag not in eval_tags:
                issues.append(f"eval workload {tag!r} not in evaluation_workload_tags allow-list")

    if overlap:
        issues.append(f"training/eval tag overlap (leakage): {overlap}")

    missing_eval = eval_tags - {w["tag"] for w in eval_workloads}
    if missing_eval:
        issues.append(f"evaluation_workload_tags missing from eval config: {sorted(missing_eval)}")

    sel_cfg = cfg.get("selector_training", {})
    plan = {
        "experiment": cfg.get("experiment"),
        "feature_mode": feature_mode.value if feature_mode else feature_mode_raw,
        "feature_mode_deployable": feature_mode_is_deployable(feature_mode) if feature_mode else False,
        "primary_rank_metric": cfg.get("primary_rank_metric", PRIMARY_RANK_METRIC),
        "training_config": train_cfg_path,
        "evaluation_config": eval_cfg_path,
        "train_diversity_seeds": sel_cfg.get("train_diversity_seeds", [6, 7, 8, 9, 10]),
        "val_diversity_seeds": sel_cfg.get("val_diversity_seeds", [11]),
        "n_training_workloads": len(train_workloads),
        "n_evaluation_workloads": len(eval_workloads),
        "evaluation_workloads": eval_workloads,
        "external_style_baselines": list(EXTERNAL_STYLE_BASELINES),
        "deployable_learned_selectors": list(DEPLOYABLE_LEARNED_SELECTORS),
    }
    return issues, plan


def plan_to_stdout(plan: Dict[str, Any], mode: str) -> None:
    print(f"Phase 2C.2 {mode}")
    print(f"  Experiment          : {plan['experiment']}")
    print(f"  feature_mode        : {plan['feature_mode']}")
    print(f"  deployable mode     : {plan['feature_mode_deployable']}")
    print(f"  primary_rank_metric : {plan['primary_rank_metric']}")
    print(f"  training_config     : {plan['training_config']} ({plan['n_training_workloads']} workloads)")
    print(f"  evaluation_config   : {plan['evaluation_config']} ({plan['n_evaluation_workloads']} workloads)")
    print(f"  train div seeds     : {plan['train_diversity_seeds']}")
    print(f"  val div seeds       : {plan['val_diversity_seeds']}")
    print("  Evaluation workloads:")
    for w in plan["evaluation_workloads"]:
        print(f"    {w['tag']} [{w['group']}] exists={w['trace_exists']}")
    print(f"  External baselines  : {len(plan['external_style_baselines'])}")


def build_causal_training_rows(
    cfg: dict,
    train_cfg: dict,
    *,
    smoke: bool = False,
    verbose: bool = False,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Build dev + diversity rows with causal features; return train, val, all-built."""
    feature_mode = parse_feature_mode(cfg.get("feature_mode", "causal"))
    gpu_configs = build_gpu_configs(cfg)
    service_model = build_service_model_from_config(cfg)
    drain_steps = cfg.get("simulator", {}).get("drain_steps", 20000)
    window_size = cfg.get("window_size", 200)
    min_partial = cfg.get("min_partial_window", 50)

    sel_cfg = cfg.get("selector_training", {})
    train_div_seeds = sel_cfg.get("train_diversity_seeds", [6, 7, 8, 9, 10])
    val_div_seeds = sel_cfg.get("val_diversity_seeds", [11])

    if smoke:
        dev_workloads = [w for w in train_cfg.get("workloads", []) if w.get("group") == "dev"][:1]
        div_workloads: List[Dict] = []
        dev_seeds = [0]
        div_seeds: List[int] = []
        window_size = 4
        min_partial = 2
    else:
        dev_workloads = [w for w in train_cfg.get("workloads", []) if w.get("group") == "dev"]
        div_workloads = [w for w in train_cfg.get("workloads", []) if w.get("group") == "diversity"]
        dev_seeds = train_cfg.get("dev_seeds", [0, 1, 2])
        div_seeds = train_cfg.get("diversity_seeds", [6, 7, 8, 9, 10, 11])

    logging.info("Building causal training rows: dev=%d div=%d", len(dev_workloads), len(div_workloads))
    dev_rows = build_rows_for_group(
        dev_workloads, dev_seeds, gpu_configs, service_model,
        drain_steps, window_size, min_partial, feature_mode, verbose,
    )
    div_rows: List[Dict] = []
    if div_workloads and div_seeds:
        div_rows = build_rows_for_group(
            div_workloads, div_seeds, gpu_configs, service_model,
            drain_steps, window_size, min_partial, feature_mode, verbose,
        )

    all_rows = dev_rows + div_rows
    for row in all_rows:
        if row.get("feature_mode") != feature_mode.value:
            row["feature_mode"] = feature_mode.value

    train_rows, val_rows, _heldout_rows = split_rows(all_rows, train_div_seeds, val_div_seeds)
    logging.info(
        "Causal rows built: total=%d train=%d val=%d feature_mode=%s",
        len(all_rows), len(train_rows), len(val_rows), feature_mode.value,
    )
    return train_rows, val_rows, all_rows


def assert_no_eval_leakage(rows: Iterable[Dict], eval_tags: Set[str]) -> None:
    prefixes = ("dev_", "heldout_", "div_")
    for row in rows:
        tag = _trace_tag(row.get("trace_id", ""))
        if tag in eval_tags:
            raise RuntimeError(f"Eval workload {tag!r} found in training rows — leakage")
        if not tag.startswith(prefixes) and tag not in eval_tags:
            logging.warning("Unexpected trace tag in training build: %s", tag)


def run_real_trace_evaluation(
    cfg: dict,
    eval_cfg: dict,
    selectors: Dict[str, Any],
    out_dir: Path,
    *,
    smoke: bool = False,
    smoke_workloads: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict]:
    gpu_configs = build_gpu_configs(cfg)
    service_model = build_service_model_from_config(cfg)
    feature_mode = parse_feature_mode(cfg.get("feature_mode", "causal"))

    if smoke and smoke_workloads is not None:
        workloads = smoke_workloads
        window_size = 4
        min_partial = 2
    else:
        workloads = eval_cfg.get("workloads", [])
        window_size = cfg.get("window_size", 200)
        min_partial = cfg.get("min_partial_window", 50)

    rows = build_rows_for_group(
        workloads=workloads,
        seeds=[0],
        gpu_configs=gpu_configs,
        service_model=service_model,
        drain_steps=cfg.get("simulator", {}).get("drain_steps", 20000),
        window_size=window_size,
        min_partial=min_partial,
        feature_mode=feature_mode,
        verbose=False,
    )
    if not rows:
        raise RuntimeError("No evaluation windows produced")

    _annotate_groups(rows, workloads)
    rows = apply_selectors_to_rows(rows, selectors)
    rows = relabel_rows(rows)
    write_per_window_csv(rows, out_dir / "per_window.csv")

    group_names = sorted({r.get("workload_group", "unknown") for r in rows})
    selector_keys = list(selectors.keys())
    summary_rows: List[Dict[str, Any]] = []
    for group_name in group_names:
        group_rows = [r for r in rows if r.get("workload_group") == group_name]
        for selector_key in selector_keys:
            summary_rows.append(
                _flatten_dict(
                    {"group": group_name, **_evaluate_selector_summary(selector_key, group_rows)}
                )
            )
    for selector_key in selector_keys:
        summary_rows.append(
            _flatten_dict(
                {"group": "overall", **_evaluate_selector_summary(selector_key, rows)}
            )
        )
    _write_csv(out_dir / "selector_summary.csv", summary_rows)
    _write_csv(out_dir / "deployable_selector_summary.csv", _build_deployable_headline_rows(summary_rows))

    return rows


def _policy_list_from_row(row: Dict) -> List[str]:
    return [k.replace("reward_", "") for k in row.keys() if k.startswith("reward_")]


def _best_anwg_among(row: Dict, policies: Iterable[str]) -> Tuple[str, float]:
    best_p, best_v = "", -1.0
    for p in policies:
        v = _anwg(row, p)
        if v > best_v:
            best_p, best_v = p, v
    return best_p, best_v


def analyze_external_baseline_failures(
    rows: List[Dict],
    out_dir: Path,
    *,
    headline_selector: Optional[str] = None,
) -> Dict[str, Any]:
    """Mine windows where deployable selectors lose to external-style baselines."""
    if not rows:
        return {}

    policies = _policy_list_from_row(rows[0])
    external = [p for p in policies if is_external_style_baseline(p)]
    deployable_sels = list(DEPLOYABLE_LEARNED_SELECTORS)

    if headline_selector is None:
        anwg_by_sel = {}
        for s in deployable_sels:
            col = f"sel_{s}_policy"
            if col not in rows[0]:
                continue
            anwg_by_sel[s] = float(np.mean([
                _anwg(r, r.get(col) or "scorpio_style_slo_guard") for r in rows
            ]))
        headline_selector = max(anwg_by_sel, key=anwg_by_sel.get) if anwg_by_sel else "regression_anwg"

    loss_cases: List[Dict[str, Any]] = []
    by_workload: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n_loss": 0, "total_loss": 0.0})
    by_ref: Counter = Counter()
    by_azure_burst: Dict[str, int] = defaultdict(int)

    for row in rows:
        pol_col = f"sel_{headline_selector}_policy"
        chosen = row.get(pol_col) or "scorpio_style_slo_guard"
        sel_an = _anwg(row, chosen)
        fix_p, fix_an = _best_anwg_among(row, policies)
        ext_p, ext_an = _best_anwg_among(row, external)

        wl = _trace_tag(row.get("trace_id", ""))
        if sel_an < fix_an - 1e-9:
            loss_cases.append({
                "workload": wl,
                "window_id": row.get("window_id"),
                "selector": headline_selector,
                "sel_policy": chosen,
                "sel_anwg": round(sel_an, 6),
                "ref_type": "best_fixed_deployable",
                "ref_policy": fix_p,
                "ref_anwg": round(fix_an, 6),
                "loss": round(fix_an - sel_an, 6),
                "feat_arrival_rate_est": row.get("feat_arrival_rate_est"),
                "feat_burstiness_cv": row.get("feat_burstiness_cv"),
                "feat_mean_prompt_tokens": row.get("feat_mean_prompt_tokens"),
                "feat_mean_pred_output_tokens": row.get("feat_mean_pred_output_tokens"),
                "feat_fraction_tight_slo": row.get("feat_fraction_tight_slo"),
                "feat_mean_slack": row.get("feat_mean_slack"),
            })
        if sel_an < ext_an - 1e-9:
            loss_cases.append({
                "workload": wl,
                "window_id": row.get("window_id"),
                "selector": headline_selector,
                "sel_policy": chosen,
                "sel_anwg": round(sel_an, 6),
                "ref_type": "best_external_style",
                "ref_policy": ext_p,
                "ref_anwg": round(ext_an, 6),
                "loss": round(ext_an - sel_an, 6),
                "feat_arrival_rate_est": row.get("feat_arrival_rate_est"),
                "feat_burstiness_cv": row.get("feat_burstiness_cv"),
                "feat_mean_prompt_tokens": row.get("feat_mean_prompt_tokens"),
                "feat_mean_pred_output_tokens": row.get("feat_mean_pred_output_tokens"),
                "feat_fraction_tight_slo": row.get("feat_fraction_tight_slo"),
                "feat_mean_slack": row.get("feat_mean_slack"),
            })
            by_workload[wl]["n_loss"] += 1
            by_workload[wl]["total_loss"] += ext_an - sel_an
            by_ref[ext_p] += 1
            key = "azure" if wl.startswith("azure") else "burstgpt"
            by_azure_burst[key] += 1

    _write_csv(out_dir / "external_baseline_loss_cases.csv", loss_cases)

    summary = {
        "headline_selector": headline_selector,
        "n_windows": len(rows),
        "n_loss_vs_external": sum(1 for c in loss_cases if c["ref_type"] == "best_external_style"),
        "n_loss_vs_fixed": sum(1 for c in loss_cases if c["ref_type"] == "best_fixed_deployable"),
        "winning_external_policies": dict(by_ref.most_common()),
        "loss_by_workload": {
            wl: {"n": v["n_loss"], "mean_loss": round(v["total_loss"] / max(v["n_loss"], 1), 6)}
            for wl, v in by_workload.items()
        },
        "loss_by_group": dict(by_azure_burst),
        "external_style_baselines": list(EXTERNAL_STYLE_BASELINES),
    }
    _write_json(out_dir / "external_baseline_failure_analysis.json", summary)
    return summary


def run_phase2c2(
    cfg: dict,
    out_dir: Path,
    *,
    smoke: bool = False,
) -> Dict[str, Any]:
    train_cfg = _load_nested_config(cfg["training_config"])
    eval_cfg = _load_nested_config(cfg["evaluation_config"])
    eval_tags = _eval_tags(cfg)

    train_rows, val_rows, all_train_built = build_causal_training_rows(
        cfg, train_cfg, smoke=smoke, verbose=False,
    )
    assert_no_eval_leakage(train_rows, eval_tags)

    sel_cfg = cfg.get("selector_training", {})
    selectors = train_selectors_from_rows(
        train_rows,
        rw_eps=float(sel_cfg.get("regret_weight_epsilon", 0.001)),
        sf_margins=cfg.get("safe_fallback_margins", [0.001, 0.005, 0.010]),
        knn_k=int(cfg.get("knn", {}).get("k", 5)),
    )

    train_out = out_dir / "training"
    train_out.mkdir(parents=True, exist_ok=True)
    write_per_window_csv(all_train_built, train_out / "causal_training_rows.csv")
    write_per_window_csv(train_rows, train_out / "causal_train_split.csv")
    write_per_window_csv(val_rows, train_out / "causal_val_split.csv")

    eval_out = out_dir / "evaluation"
    smoke_workloads = build_smoke_workloads(out_dir) if smoke else None
    if not smoke:
        ensure_azure_2023_inputs(eval_cfg, allow_download=False, dry_run=False)

    eval_rows = run_real_trace_evaluation(
        cfg, eval_cfg, selectors, eval_out,
        smoke=smoke, smoke_workloads=smoke_workloads,
    )

    ext_analysis = analyze_external_baseline_failures(eval_rows, eval_out)

    selector_keys = list(selectors.keys())
    metadata = {
        "experiment": cfg.get("experiment"),
        "feature_mode": parse_feature_mode(cfg.get("feature_mode", "causal")).value,
        "feature_mode_deployable": True,
        "n_training_rows_built": len(all_train_built),
        "n_training_rows_fit": len(train_rows),
        "n_eval_windows": len(eval_rows),
        "selectors": selector_keys,
        "selector_roles": classify_selectors(selector_keys),
        "oracle_assisted_selectors": [
            k for k in selector_keys if is_oracle_assisted_selector(k)
        ],
        "deployable_headline_selectors": [
            k for k in selector_keys if is_deployable_headline_selector(k)
        ],
        "primary_rank_metric": cfg.get("primary_rank_metric", PRIMARY_RANK_METRIC),
        "external_style_baselines": list(EXTERNAL_STYLE_BASELINES),
        "training_trace_prefixes": _training_prefixes(cfg),
        "evaluation_workload_tags": sorted(eval_tags),
        "external_baseline_failure_summary": ext_analysis,
    }
    _write_json(out_dir / "metadata.json", metadata)
    return metadata


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(_repo_path(args.config))
    issues, plan = validate_phase2c2_config(cfg)

    if issues:
        for issue in issues:
            print(f"CONFIG ERROR: {issue}", file=sys.stderr)
        return 1

    if args.dry_run:
        plan_to_stdout(plan, "dry-run")
        print("  [dry-run] No files written.")
        return 0

    if args.smoke and args.allow_full_run:
        print("ERROR: choose --smoke or --allow-full-run, not both.", file=sys.stderr)
        return 2

    if not args.smoke and not args.allow_full_run:
        print(
            "ERROR: use --dry-run, --smoke, or --allow-full-run.",
            file=sys.stderr,
        )
        return 2

    if args.allow_azure_download:
        logging.warning("--allow-azure-download ignored; using local Azure traces only")

    base_out = _repo_path(args.out_dir or cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    out_dir = (base_out / "smoke" / _timestamp()) if args.smoke else (base_out / _timestamp())
    log_file = args.log_file or (SMOKE_LOG_FILE if args.smoke else DEFAULT_LOG_FILE)
    _setup_logging(log_file, args.verbose)

    logging.info("Phase 2C.2 %s run starting", "smoke" if args.smoke else "full")
    metadata = run_phase2c2(cfg, out_dir, smoke=args.smoke)
    logging.info("Phase 2C.2 complete: %s", out_dir)
    logging.info("Train rows fit: %d | Eval windows: %d",
                 metadata["n_training_rows_fit"], metadata["n_eval_windows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
