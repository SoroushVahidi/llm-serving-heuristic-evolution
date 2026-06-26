#!/usr/bin/env python3
"""
Phase 2C.1 real-trace ingestion validation runner.

This runner exists to make the Phase 2C.1 config executable without forcing
an expensive real-trace run by default.

Modes
-----
--dry-run
    Validate config structure, inspect planned workloads, and report which
    inputs are already present. Writes no files and performs no downloads.

--smoke
    Build tiny local BurstGPT/Azure traces under an ignored results directory
    and run a minimal end-to-end validation pass.

--allow-full-run
    Permit the real Phase 2C.1 run over configured traces.
    Azure 2023 materialization is still blocked unless --allow-azure-download
    is also supplied when processed traces are missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.selector.features import FeatureMode, FEATURE_NAMES, parse_feature_mode, feature_mode_is_deployable
from llmserveopt.selector.roles import (
    classify_selectors,
    is_deployable_headline_selector,
    is_oracle_assisted_selector,
    selector_role,
)
from llmserveopt.selector.models import RuleBasedSelector
from llmserveopt.simulator.service_model_factory import build_service_model_from_config
from llmserveopt.workloads.augmentation import AugmentationConfig
from llmserveopt.workloads.burstgpt import BurstGPTConversionConfig, load_burstgpt_trace
from llmserveopt.workloads.trace_io_extended import save_extended_jsonl

from run_phase2b9_selector_robustness import (
    apply_selectors_to_rows,
    build_gpu_configs,
    load_config,
    write_per_window_csv,
)
from run_phase2b12_workload_diversity_selector_labels import build_rows_for_group
from run_phase2b15_corrected_objective_selector_retraining import (
    AlwaysScorpioSelector,
    AlwaysWSPSelector,
    relabel_rows,
)
from run_phase2b16_fresh_corrected_objective_validation import (
    evaluate_fresh_selector,
    train_phase2b15_selectors,
)
from scripts.data.convert_azure_llm_trace import (
    convert_azure_to_requests,
    load_azure_csv,
)

DEFAULT_CONFIG = "configs/phase2c1_real_trace_ingestion_validation.yaml"
DEFAULT_OUTPUT_DIR = "results/phase2c1_real_trace_ingestion_validation"
DEFAULT_LOG_FILE = "logs/phase2c1/phase2c1_real_traces.log"
SMOKE_LOG_FILE = "logs/phase2c1/phase2c1_smoke.log"

REQUIRED_TOP_LEVEL_KEYS = [
    "experiment",
    "input_b13_dir",
    "output_dir",
    "simulator",
    "service_model",
    "gpus",
    "window_size",
    "min_partial_window",
    "feature_mode",
    "selector_training",
    "azure",
    "workloads",
]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2C.1 real-trace ingestion validation runner"
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to Phase 2C.1 YAML config")
    p.add_argument("--out-dir", default=None, help="Override output directory")
    p.add_argument("--log-file", default=None, help="Optional log file under logs/")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and print plan only")
    p.add_argument("--smoke", action="store_true", help="Run a tiny local smoke validation")
    p.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Permit the expensive configured Phase 2C.1 evaluation",
    )
    p.add_argument(
        "--allow-azure-download",
        action="store_true",
        help="Allow downloading small Azure 2023 CSVs when processed traces are missing",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_under(root_name: str, raw_path: str | Path) -> Path:
    path = _repo_path(raw_path).resolve()
    repo_root = ROOT.resolve()
    try:
        rel = path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside repository: {path}") from exc
    if not rel.parts or rel.parts[0] != root_name:
        raise ValueError(f"Path must be under {root_name}/: {path}")
    return path


def _setup_logging(log_file: str | None, verbose: bool) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = _ensure_under("logs", log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=full_key + "_"))
        else:
            flat[full_key] = value
    return flat


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _required_azure_sections(cfg: dict) -> Dict[str, dict]:
    azure_root = cfg.get("azure", {})
    azure_2023 = azure_root.get("2023", {})
    missing = [name for name in ("code", "conv") if name not in azure_2023]
    if missing:
        raise ValueError(f"Config missing azure.2023 entries: {missing}")
    return azure_2023


def validate_phase2c1_config(cfg: dict) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            issues.append(f"missing top-level key: {key}")

    if cfg.get("experiment") != "phase2c1_real_trace_ingestion_validation":
        issues.append(
            "experiment must be 'phase2c1_real_trace_ingestion_validation'"
        )

    workloads = cfg.get("workloads", [])
    if not isinstance(workloads, list) or not workloads:
        issues.append("workloads must be a non-empty list")

    workload_plan: List[Dict[str, Any]] = []
    tags_seen = set()
    for workload in workloads:
        tag = workload.get("tag")
        if not tag:
            issues.append("workload missing tag")
            continue
        if tag in tags_seen:
            issues.append(f"duplicate workload tag: {tag}")
        tags_seen.add(tag)
        trace_path = workload.get("trace_path")
        workload_plan.append({
            "tag": tag,
            "group": workload.get("group", ""),
            "source": workload.get("source", ""),
            "trace_path": trace_path,
            "trace_exists": bool(trace_path and _repo_path(trace_path).exists()),
        })

    try:
        azure_2023 = _required_azure_sections(cfg)
    except ValueError as exc:
        issues.append(str(exc))
        azure_plan = []
    else:
        azure_plan = []
        for split_name, entry in azure_2023.items():
            required = ("url", "raw_path", "processed_path", "time_scale", "source_tag")
            missing = [name for name in required if name not in entry]
            if missing:
                issues.append(f"azure.2023.{split_name} missing keys: {missing}")
            azure_plan.append({
                "split": split_name,
                "raw_path": entry.get("raw_path"),
                "raw_exists": bool(entry.get("raw_path") and _repo_path(entry["raw_path"]).exists()),
                "processed_path": entry.get("processed_path"),
                "processed_exists": bool(
                    entry.get("processed_path") and _repo_path(entry["processed_path"]).exists()
                ),
                "time_scale": entry.get("time_scale"),
            })

    feature_mode_raw = cfg.get("feature_mode", "causal")
    try:
        feature_mode = parse_feature_mode(feature_mode_raw)
    except ValueError:
        issues.append(f"unsupported feature_mode: {feature_mode_raw}")
        feature_mode = None
    else:
        if feature_mode is not None and not feature_mode_is_deployable(feature_mode):
            issues.append(
                "feature_mode must be 'causal' for deployable Phase 2C.1 evaluation; "
                f"got {feature_mode_raw!r} (offline/diagnostic modes leak within-window arrivals)"
            )

    plan = {
        "experiment": cfg.get("experiment", ""),
        "input_b13_dir": cfg.get("input_b13_dir"),
        "input_b13_exists": bool(
            cfg.get("input_b13_dir")
            and (_repo_path(cfg["input_b13_dir"]) / "per_window.csv").exists()
        ),
        "feature_mode": feature_mode.value if feature_mode is not None else feature_mode_raw,
        "feature_mode_deployable": (
            feature_mode_is_deployable(feature_mode) if feature_mode is not None else False
        ),
        "workloads": workload_plan,
        "azure_2023": azure_plan,
    }
    return issues, plan


def plan_to_stdout(plan: Dict[str, Any], mode: str) -> None:
    print(f"Phase 2C.1 {mode}")
    print(f"  Experiment        : {plan['experiment']}")
    print(f"  input_b13_dir     : {plan['input_b13_dir']}")
    print(f"  input_b13_exists  : {plan['input_b13_exists']}")
    print(f"  feature_mode      : {plan.get('feature_mode')}")
    print(f"  deployable mode   : {plan.get('feature_mode_deployable')}")
    print(f"  Workloads         : {len(plan['workloads'])}")
    for workload in plan["workloads"]:
        print(
            "    "
            f"{workload['tag']} [{workload['group']}] "
            f"source={workload['source']} exists={workload['trace_exists']}"
        )
    if plan["azure_2023"]:
        print("  Azure 2023 assets :")
        for entry in plan["azure_2023"]:
            print(
                "    "
                f"{entry['split']}: raw={entry['raw_exists']} "
                f"processed={entry['processed_exists']} "
                f"time_scale={entry['time_scale']}"
            )


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 Phase2C1-Azure-Downloader/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as response, open(dest, "wb") as f:
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            f.write(chunk)


def convert_azure_csv_to_jsonl(
    *,
    raw_path: Path,
    processed_path: Path,
    source_tag: str,
    time_scale: float,
    seed: int = 17,
) -> Dict[str, Any]:
    arrivals, context_tokens, generated_tokens, report = load_azure_csv(
        raw_path,
        time_scale=time_scale,
    )
    requests = convert_azure_to_requests(
        arrivals,
        context_tokens,
        generated_tokens,
        AugmentationConfig(),
        seed=seed,
    )
    save_extended_jsonl(requests, processed_path, source=source_tag)
    report = dict(report)
    report["source"] = source_tag
    report["seed"] = seed
    report_path = processed_path.with_suffix(".report.json")
    _write_json(report_path, report)
    return report


def ensure_azure_2023_inputs(
    cfg: dict,
    *,
    allow_download: bool,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for split_name, entry in _required_azure_sections(cfg).items():
        raw_path = _repo_path(entry["raw_path"])
        processed_path = _repo_path(entry["processed_path"])
        item = {
            "split": split_name,
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "status": "",
        }

        if processed_path.exists():
            item["status"] = "processed_ready"
        elif raw_path.exists():
            if dry_run:
                item["status"] = "would_convert_from_raw"
            else:
                convert_azure_csv_to_jsonl(
                    raw_path=raw_path,
                    processed_path=processed_path,
                    source_tag=entry["source_tag"],
                    time_scale=float(entry["time_scale"]),
                )
                item["status"] = "converted_from_raw"
        else:
            if not allow_download:
                raise RuntimeError(
                    "Azure 2023 traces are not materialized. "
                    "Re-run with --allow-azure-download to download and convert them."
                )
            if dry_run:
                item["status"] = "would_download_and_convert"
            else:
                download_file(entry["url"], raw_path)
                convert_azure_csv_to_jsonl(
                    raw_path=raw_path,
                    processed_path=processed_path,
                    source_tag=entry["source_tag"],
                    time_scale=float(entry["time_scale"]),
                )
                item["status"] = "downloaded_and_converted"

        plan.append(item)
    return plan


def load_phase2c1_selectors(
    cfg: dict,
    *,
    allow_fallback: bool,
) -> Dict[str, Any]:
    b13_dir = _repo_path(cfg["input_b13_dir"])
    per_window_path = b13_dir / "per_window.csv"
    if per_window_path.exists():
        selector_cfg = cfg.get("selector_training", {})
        return train_phase2b15_selectors(
            b13_per_window_path=per_window_path,
            train_div_seeds=selector_cfg.get("train_diversity_seeds", [6, 7, 8, 9, 10]),
            near_tie_eps=selector_cfg.get("near_tie_filter_epsilon", 0.005),
            rw_eps=selector_cfg.get("regret_weight_epsilon", 0.001),
            sf_margins=cfg.get("safe_fallback_margins", [0.001, 0.005, 0.010]),
            knn_k=cfg.get("knn", {}).get("k", 5),
        )

    if allow_fallback:
        return {
            "always_scorpio": AlwaysScorpioSelector(),
            "always_wsp": AlwaysWSPSelector(),
            "rule_based": RuleBasedSelector(),
        }

    raise FileNotFoundError(
        f"Required selector training input not found: {per_window_path}. "
        "Run Phase 2B.13 first, or use --smoke for a fallback-selector sanity run."
    )


def build_smoke_workloads(out_dir: Path) -> List[Dict[str, Any]]:
    smoke_dir = out_dir / "_smoke_inputs"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    azure_csv = smoke_dir / "azure_smoke.csv"
    with open(azure_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
        )
        writer.writeheader()
        for idx, row in enumerate([
            ("2024-01-01 00:00:00.0000000", 64, 8),
            ("2024-01-01 00:00:00.5000000", 96, 12),
            ("2024-01-01 00:00:01.0000000", 128, 16),
            ("2024-01-01 00:00:01.5000000", 160, 20),
            ("2024-01-01 00:00:02.0000000", 192, 24),
            ("2024-01-01 00:00:02.5000000", 224, 28),
        ]):
            writer.writerow({
                "TIMESTAMP": row[0],
                "ContextTokens": row[1],
                "GeneratedTokens": row[2],
            })

    azure_jsonl = smoke_dir / "azure_smoke.jsonl"
    convert_azure_csv_to_jsonl(
        raw_path=azure_csv,
        processed_path=azure_jsonl,
        source_tag="azure_smoke",
        time_scale=0.25,
    )

    burst_csv = ROOT / "tests" / "fixtures" / "burstgpt_tiny.csv"
    burst_jsonl = smoke_dir / "burstgpt_smoke.jsonl"
    burst_requests, _ = load_burstgpt_trace(
        burst_csv,
        BurstGPTConversionConfig(max_requests=18),
        seed=17,
    )
    save_extended_jsonl(burst_requests, burst_jsonl, source="burstgpt_smoke")

    return [
        {
            "tag": "burstgpt_smoke",
            "group": "burstgpt",
            "source": "extended_jsonl",
            "trace_path": str(burst_jsonl),
        },
        {
            "tag": "azure_smoke",
            "group": "azure_2023",
            "source": "extended_jsonl",
            "trace_path": str(azure_jsonl),
        },
    ]


def _annotate_groups(rows: List[Dict[str, Any]], workloads: Iterable[Dict[str, Any]]) -> None:
    group_by_tag = {w["tag"]: w.get("group", "unknown") for w in workloads}
    for row in rows:
        trace_id = row.get("trace_id", "")
        tag = trace_id.rsplit("_s", 1)[0] if "_s" in trace_id else trace_id
        row["workload_group"] = group_by_tag.get(tag, "unknown")


def _evaluate_selector_summary(selector_key: str, group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate one selector and attach role / headline metadata."""
    summary = evaluate_fresh_selector(selector_key, group_rows)
    summary["selector_role"] = selector_role(selector_key)
    summary["deployable_headline"] = is_deployable_headline_selector(selector_key)
    summary["oracle_assisted"] = is_oracle_assisted_selector(selector_key)
    return summary


def _build_deployable_headline_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deployable = [
        row for row in summary_rows
        if row.get("deployable_headline") or row.get("selector_role") == "deployable_learned"
    ]
    deployable.sort(
        key=lambda row: (
            -float(row.get("mean_arrival_normalized_wg", 0.0)),
            row.get("group", ""),
            row.get("selector", ""),
        )
    )
    return deployable


def run_phase2c1_validation(
    cfg: dict,
    *,
    workloads: List[Dict[str, Any]],
    out_dir: Path,
    allow_fallback_selectors: bool,
    window_size: int | None = None,
    min_partial_window: int | None = None,
) -> Dict[str, Any]:
    selectors = load_phase2c1_selectors(cfg, allow_fallback=allow_fallback_selectors)
    service_model = build_service_model_from_config(cfg)
    gpu_configs = build_gpu_configs(cfg)
    rows = build_rows_for_group(
        workloads=workloads,
        seeds=[0],
        gpu_configs=gpu_configs,
        service_model=service_model,
        drain_steps=cfg.get("simulator", {}).get("drain_steps", 20000),
        window_size=window_size if window_size is not None else cfg.get("window_size", 200),
        min_partial=(
            min_partial_window
            if min_partial_window is not None
            else cfg.get("min_partial_window", 50)
        ),
        feature_mode=parse_feature_mode(cfg.get("feature_mode", "causal")),
        verbose=False,
    )
    if not rows:
        raise RuntimeError("No validation windows were produced from the requested workloads.")

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
    deployable_rows = _build_deployable_headline_rows(summary_rows)
    _write_csv(out_dir / "deployable_selector_summary.csv", deployable_rows)

    selector_roles = classify_selectors(selector_keys)
    metadata = {
        "n_windows": len(rows),
        "n_workloads": len(workloads),
        "feature_mode": parse_feature_mode(cfg.get("feature_mode", "causal")).value,
        "feature_mode_deployable": feature_mode_is_deployable(
            parse_feature_mode(cfg.get("feature_mode", "causal"))
        ),
        "selectors": selector_keys,
        "selector_roles": selector_roles,
        "oracle_assisted_selectors": selector_roles.get("oracle_assisted", []),
        "deployable_headline_selectors": selector_roles.get("deployable_learned", []),
        "primary_rank_metric": "mean_arrival_normalized_wg",
        "metric_definitions": {
            "mean_completed_request_quality": "completed-only weighted goodput (conditional on completed requests)",
            "mean_completion_fraction": "fraction of arrivals completed",
            "mean_arrival_normalized_wg": "completion_fraction * completed_only_wg",
            "mean_cp_wg_t095_l05": "completion-penalized WG target=0.95 lambda=0.5",
            "mean_cp_wg_t099_l05": "completion-penalized WG target=0.99 lambda=0.5",
            "mean_cp_wg_t099_l10": "completion-penalized WG target=0.99 lambda=1.0",
        },
        "workloads": [
            {
                "tag": w["tag"],
                "group": w.get("group", ""),
                "trace_path": w.get("trace_path", ""),
            }
            for w in workloads
        ],
        "phase2b16_reference": cfg.get("phase2b16_reference", {}),
    }
    _write_json(out_dir / "metadata.json", metadata)
    return metadata


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    issues, plan = validate_phase2c1_config(cfg)

    if issues:
        for issue in issues:
            print(f"CONFIG ERROR: {issue}", file=sys.stderr)
        return 1

    if args.dry_run:
        plan_to_stdout(plan, "dry-run")
        print("  [dry-run] No files written.")
        print("  [dry-run] Full evaluation remains blocked without --allow-full-run.")
        print("  [dry-run] Azure acquisition remains blocked without --allow-azure-download.")
        return 0

    if args.smoke and args.allow_full_run:
        print("ERROR: choose either --smoke or --allow-full-run, not both.", file=sys.stderr)
        return 2

    if not args.smoke and not args.allow_full_run:
        print(
            "ERROR: refusing to run Phase 2C.1 by default. "
            "Use --dry-run for preflight, --smoke for a tiny local check, "
            "or --allow-full-run to permit the expensive configured evaluation.",
            file=sys.stderr,
        )
        return 2

    base_out = _ensure_under("results", args.out_dir or cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    if args.smoke:
        out_dir = base_out / "smoke" / _timestamp()
        log_file = args.log_file or SMOKE_LOG_FILE
        _setup_logging(log_file, args.verbose)
        logging.info("Phase 2C.1 smoke run")
        workloads = build_smoke_workloads(out_dir)
        metadata = run_phase2c1_validation(
            cfg,
            workloads=workloads,
            out_dir=out_dir,
            allow_fallback_selectors=True,
            window_size=4,
            min_partial_window=2,
        )
        logging.info("Smoke run complete: %s", out_dir)
        logging.info("Smoke windows: %d", metadata["n_windows"])
        return 0

    log_file = args.log_file or DEFAULT_LOG_FILE
    _setup_logging(log_file, args.verbose)
    logging.info("Phase 2C.1 full run requested")
    azure_plan = ensure_azure_2023_inputs(
        cfg,
        allow_download=args.allow_azure_download,
        dry_run=False,
    )
    logging.info("Azure input plan: %s", azure_plan)
    out_dir = base_out / _timestamp()
    metadata = run_phase2c1_validation(
        cfg,
        workloads=cfg["workloads"],
        out_dir=out_dir,
        allow_fallback_selectors=False,
    )
    _write_json(out_dir / "azure_materialization.json", azure_plan)
    logging.info("Phase 2C.1 run complete: %s", out_dir)
    logging.info("Validation windows: %d", metadata["n_windows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
