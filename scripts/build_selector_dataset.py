#!/usr/bin/env python3
"""
HISTORICAL ENTRY POINT (Selector Dataset v1, Phase 2A).
Retained for reproducibility of Phase 2A.2/2A.3 results only.
For current Selector Dataset v2 generation (Option B scope, the approved
8-policy historical-monolithic action space) use:
    scripts/build_selector_dataset_v2_calibrated_targeted_pilot.py
See docs/selector_v2_faithful_baseline_scope_audit.md and
docs/selector_dataset_v2.md for the current design/rationale.

Build a selector dataset: per-window features + per-policy rewards + labels.

Usage
-----
python scripts/build_selector_dataset.py \
    --config configs/selector/selector_dataset_smoke.yaml \
    --output results/phase2a2_selector_dataset/smoke_selector_dataset.csv

The script:
1. Loads or generates a trace from the config.
2. Splits into W=200 request windows.
3. Extracts online-observable features per window.
4. Runs each deployable policy on each window (isolated simulation).
5. Assigns labels (best policy by weighted_goodput).
6. Writes CSV, reward matrix CSV, and metadata JSON.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.core.types import GPUConfig
from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.dataset import DatasetConfig, build_selector_dataset, save_dataset
from llmserveopt.selector.features import parse_feature_mode
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.service_model_factory import build_service_model_from_config
from llmserveopt.workloads.synthetic import WorkloadConfig, SLOClass, generate_workload


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_gpu_configs(cfg: dict) -> list:
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
    raw_classes = kwargs.pop("slo_classes", None)
    if raw_classes is not None:
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


def load_or_generate_trace(wcfg: dict, seed: int):
    """Load trace from file or generate synthetic.

    Supports optional max_requests field to cap trace length.
    """
    source = wcfg.get("source", "synthetic")
    max_req = wcfg.get("max_requests", None)

    if source == "trace_file":
        trace_path = wcfg["trace_path"]
        from llmserveopt.workloads.trace_io import load_jsonl
        print(f"  Loading trace from {trace_path}")
        reqs = load_jsonl(trace_path)
    elif source == "extended_jsonl":
        from llmserveopt.workloads.trace_io_extended import load_extended_jsonl
        print(f"  Loading extended JSONL from {wcfg['trace_path']}")
        reqs, _ = load_extended_jsonl(wcfg["trace_path"])
    else:
        cfg = build_workload_config(wcfg)
        print(f"  Generating synthetic trace: tag={cfg.tag} rate={cfg.arrival_rate} dur={cfg.duration}s seed={seed}")
        reqs = generate_workload(cfg, seed=seed)

    if max_req is not None and len(reqs) > max_req:
        reqs = reqs[:max_req]
        print(f"  Trimmed to {len(reqs)} requests (max_requests={max_req})")

    return reqs


def parse_args():
    p = argparse.ArgumentParser(description="Build selector dataset")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--output", required=True, help="Output CSV path or directory")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument("--window-size", type=int, default=None, help="Override window size")
    p.add_argument(
        "--feature-mode",
        choices=["causal", "offline_window_lookahead", "online_prefix", "trace_window_descriptive"],
        default=None,
        help="Override config feature_mode",
    )
    p.add_argument("--policies", nargs="*", default=None,
                   help="Subset of candidate policies to evaluate (default: all)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    window_size = args.window_size or cfg.get("window_size", 200)
    feature_mode_str = args.feature_mode or cfg.get("feature_mode", "causal")
    feature_mode = parse_feature_mode(feature_mode_str)
    verbose = args.verbose or cfg.get("verbose", False)
    drain_steps = cfg.get("simulator", {}).get("drain_steps", 5000)

    gpu_configs = build_gpu_configs(cfg)

    service_model_cfg = cfg.get("service_model", {"type": "synthetic"})
    try:
        service_model = build_service_model_from_config({"service_model": service_model_cfg})
    except Exception:
        service_model = ServiceModel()

    policy_names = args.policies or cfg.get("policies", None) or list(SELECTOR_CANDIDATES)
    # Validate
    for p in policy_names:
        if p not in SELECTOR_CANDIDATES:
            print(f"WARNING: '{p}' not in SELECTOR_CANDIDATES, skipping.")
    policy_names = [p for p in policy_names if p in SELECTOR_CANDIDATES]

    print(f"Selector dataset builder")
    print(f"  config:       {args.config}")
    print(f"  output:       {args.output}")
    print(f"  seed:         {seed}")
    print(f"  window_size:  {window_size}")
    print(f"  feature_mode: {feature_mode.value}")
    print(f"  policies:     {len(policy_names)} ({policy_names[:3]}...)")
    print()

    all_rows = []
    workloads = cfg.get("workloads", [{}])
    t0 = time.perf_counter()

    for w_idx, wdef in enumerate(workloads):
        trace_id = wdef.get("tag", "trace")
        # Per-workload seed: use explicit seed field, or global_seed + workload_index
        w_seed = int(wdef.get("seed", seed + w_idx))
        print(f"[{trace_id}] Loading trace... (seed={w_seed})")
        requests = load_or_generate_trace(wdef, seed=w_seed)
        print(f"  {len(requests)} requests")

        dataset_cfg = DatasetConfig(
            trace_id=trace_id,
            window_size=window_size,
            feature_mode=feature_mode,
            gpu_configs=gpu_configs,
            service_model=service_model,
            drain_steps=drain_steps,
            seed=w_seed,
            verbose=verbose,
        )

        rows = build_selector_dataset(requests, dataset_cfg, policy_names=policy_names)
        all_rows.extend(rows)
        print(f"  Generated {len(rows)} windows")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {len(all_rows)} windows in {elapsed:.1f}s")

    metadata = {
        "config_path": str(args.config),
        "seed": seed,
        "window_size": window_size,
        "feature_mode": feature_mode.value,
        "policy_names": policy_names,
        "build_time_s": round(elapsed, 2),
    }
    save_dataset(all_rows, args.output, metadata=metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
