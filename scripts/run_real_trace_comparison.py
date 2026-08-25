#!/usr/bin/env python3
"""
Run all baseline policies on a real trace JSONL file.

Usage:
    python scripts/run_real_trace_comparison.py \
        --config configs/burstgpt_replay_comparison.yaml \
        2>&1 | tee results/burstgpt_replay.log

Config can use either:

    workloads:
      - tag: burstgpt_replay
        source: trace_file
        trace_path: data/processed/burstgpt/burstgpt_10k.jsonl

or the standard synthetic workload format.
"""
import argparse
import datetime
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.core.types import GPUConfig
from llmserveopt.evaluation.aggregate import (
    metrics_to_dataframe,
    make_summary_table,
    save_results,
    print_summary_table,
)
from llmserveopt.evaluation.compare import compare_policies, generate_traces_for_seeds
from llmserveopt.plotting.figures import plot_all
from llmserveopt.plotting.tables import to_latex, to_markdown
from llmserveopt.policies.registry import make_policy
from llmserveopt.simulator.service_model_factory import build_service_model_from_config
from llmserveopt.workloads.synthetic import WorkloadConfig, SLOClass, DEFAULT_SLO_CLASSES
from llmserveopt.workloads.trace_io_extended import load_extended_jsonl
from llmserveopt.workloads.burstgpt import (
    load_burstgpt_trace,
    BurstGPTConversionConfig,
)


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


def build_service_model(cfg: dict):
    """Build service model; delegates to factory supporting 'synthetic' and 'calibrated'."""
    return build_service_model_from_config(cfg)


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
    else:
        slo_classes = list(DEFAULT_SLO_CLASSES)
    return WorkloadConfig(tag=tag, slo_classes=slo_classes, **kwargs)


def load_trace_file(wl_cfg_raw: dict) -> tuple:
    trace_path = wl_cfg_raw.get("trace_path")
    if not trace_path:
        print("ERROR: workload has source=trace_file but no trace_path specified.", file=sys.stderr)
        sys.exit(1)

    path = Path(trace_path)
    if not path.exists():
        print(f"ERROR: Trace file not found: {path}", file=sys.stderr)
        print("", file=sys.stderr)
        print("To create it:", file=sys.stderr)
        print("  1. python scripts/download_burstgpt.py --output data/raw/burstgpt/", file=sys.stderr)
        print("  2. python scripts/convert_burstgpt.py \\", file=sys.stderr)
        print(f"       --input data/raw/burstgpt/BurstGPT_without_fails.csv \\", file=sys.stderr)
        print(f"       --output {trace_path}", file=sys.stderr)
        sys.exit(1)

    tag = wl_cfg_raw.get("tag", path.stem)
    time_scale = float(wl_cfg_raw.get("time_scale", 1.0))

    suffix = path.suffix.lower()
    if suffix == ".csv":
        config = BurstGPTConversionConfig(time_scale=time_scale)
        requests, report = load_burstgpt_trace(path, config)
        print(f"  Loaded {len(requests)} requests from CSV (time_scale={time_scale})")
    else:
        requests, _ = load_extended_jsonl(path)
        if time_scale != 1.0 and len(requests) > 1:
            import numpy as np
            arrivals = np.array([r.arrival_time for r in requests])
            gaps = np.diff(arrivals) * time_scale
            new_arrivals = np.concatenate([[0.0], np.cumsum(gaps)])
            from llmserveopt.core.types import Request
            requests = [
                Request(
                    request_id=r.request_id,
                    arrival_time=float(new_arrivals[i]),
                    prompt_tokens=r.prompt_tokens,
                    predicted_output_tokens=r.predicted_output_tokens,
                    actual_output_tokens=r.actual_output_tokens,
                    slo_deadline=r.slo_deadline - r.arrival_time + float(new_arrivals[i]),
                    priority=r.priority,
                    class_id=r.class_id,
                )
                for i, r in enumerate(requests)
            ]
        print(f"  Loaded {len(requests)} requests from JSONL (time_scale={time_scale})")

    return tag, requests


def main():
    parser = argparse.ArgumentParser(description="Run policy comparison on a real trace")
    parser.add_argument(
        "--config", default="configs/burstgpt_replay_comparison.yaml",
        help="Path to experiment YAML config",
    )
    parser.add_argument("--out-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = cfg.get("experiment", "experiment")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("results") / experiment_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment : {experiment_name}")
    print(f"Config     : {args.config}")
    print(f"Output     : {out_dir}")

    service_model = build_service_model(cfg)
    print(f"{'='*60}\n")

    gpu_configs = build_gpu_configs(cfg)
    seeds = cfg.get("seeds", [0])
    policy_names = cfg.get("policies", ["fifo", "edf"])
    verbose = cfg.get("output", {}).get("verbose", True)
    save_figures = cfg.get("output", {}).get("save_figures", True)
    save_latex = cfg.get("output", {}).get("save_latex", False)

    all_results = []

    for wl_cfg_raw in cfg.get("workloads", [{}]):
        source = wl_cfg_raw.get("source", "synthetic")

        if source == "trace_file":
            tag, requests = load_trace_file(wl_cfg_raw)
            print(f"\n--- Workload: {tag} (trace replay) ---")
            requests_per_seed = {0: requests}

            policies = [make_policy(n, seed=seeds[0]) for n in policy_names]
            results = compare_policies(
                policies=policies,
                requests_per_seed=requests_per_seed,
                gpu_configs=gpu_configs,
                service_model=service_model,
                workload_tag=tag,
                drain_steps=cfg.get("simulator", {}).get("drain_steps", 50_000),
                verbose=verbose,
            )
        else:
            wl_cfg = build_workload_config(wl_cfg_raw)
            tag = wl_cfg.tag
            print(f"\n--- Workload: {tag} (synthetic) ---")

            traces = generate_traces_for_seeds(wl_cfg, seeds)
            n_total = sum(len(v) for v in traces.values())
            print(f"  Generated {n_total} total requests across {len(seeds)} seeds")

            policies = [make_policy(n, seed=seeds[0]) for n in policy_names]
            results = compare_policies(
                policies=policies,
                requests_per_seed=traces,
                gpu_configs=gpu_configs,
                service_model=service_model,
                workload_tag=tag,
                drain_steps=cfg.get("simulator", {}).get("drain_steps", 50_000),
                verbose=verbose,
            )

        all_results.extend(results)

        wl_dir = out_dir / tag
        save_results(results, wl_dir)

        df = metrics_to_dataframe(results)
        summary = make_summary_table(df, include_phase15=True)

        print(f"\n  Summary for workload '{tag}':")
        print_summary_table(df, include_phase15=True)

        if save_figures:
            try:
                plot_all(df, wl_dir / "figures", summary_df=summary)
                print(f"  Figures saved to {wl_dir / 'figures'}")
            except Exception as e:
                print(f"  Warning: figure generation failed: {e}")

        if save_latex:
            to_latex(summary, wl_dir / "table.tex")
            to_markdown(summary, wl_dir / "table.md")

    if all_results:
        save_results(all_results, out_dir)
        all_df = metrics_to_dataframe(all_results)

        print(f"\n{'='*60}")
        print("OVERALL SUMMARY:")
        print_summary_table(all_df, include_phase15=True)
        print(f"{'='*60}\n")

        if save_figures:
            try:
                plot_all(all_df, out_dir / "figures")
            except Exception as e:
                print(f"  Warning: global figure generation failed: {e}")

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
