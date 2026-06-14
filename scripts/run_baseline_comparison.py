#!/usr/bin/env python3
"""
Run the full baseline comparison experiment.

Usage:
    python scripts/run_baseline_comparison.py --config configs/baseline_comparison.yaml
    python scripts/run_baseline_comparison.py --config configs/prefill_heavy_comparison.yaml
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import yaml

# Allow running from project root without installing
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
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.service_model_factory import build_service_model_from_config
from llmserveopt.workloads.synthetic import WorkloadConfig, SLOClass, DEFAULT_SLO_CLASSES


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

    # Convert inline slo_classes dicts → SLOClass objects
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


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison experiment")
    parser.add_argument(
        "--config", default="configs/baseline_comparison.yaml",
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Override output directory",
    )
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
        wl_cfg = build_workload_config(wl_cfg_raw)
        tag = wl_cfg.tag
        print(f"\n--- Workload: {tag} ---")

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

        # Per-workload output
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

    # Global summary across all workloads
    if all_results:
        save_results(all_results, out_dir)
        all_df = metrics_to_dataframe(all_results)

        print(f"\n{'='*60}")
        print("OVERALL SUMMARY (all workloads, mean across seeds):")
        print_summary_table(all_df, include_phase15=True)
        print(f"{'='*60}\n")

        if save_figures:
            try:
                plot_all(all_df, out_dir / "figures")
            except Exception as e:
                print(f"  Warning: global figure generation failed: {e}")

        if save_latex:
            summary_all = make_summary_table(all_df, include_phase15=True)
            to_markdown(summary_all, out_dir / "summary_table.md")

        _write_run_readme(out_dir, cfg, timestamp, all_df, service_model)

    print(f"\nResults saved to: {out_dir}")


def _write_run_readme(out_dir, cfg, timestamp, df, service_model=None):
    summary = make_summary_table(df, include_phase15=True)
    try:
        table_md = summary.to_markdown(index=False, floatfmt=".4f") or ""
    except Exception:
        table_md = summary.to_string(index=False)

    sm_note = ""
    if service_model is not None and service_model.enable_prefill_modeling:
        sm_note = (
            f"- Prefill modeling **enabled**: "
            f"chunk={service_model.max_prefill_chunk_tokens}, "
            f"budget={service_model.step_token_budget}, "
            f"decode_first={service_model.decode_first}\n"
        )
    else:
        sm_note = "- Prefill modeling: **disabled** (Phase 1 mode)\n"

    readme = f"""# Baseline Comparison Run

**Timestamp**: {timestamp}
**Config**: {cfg.get('experiment', 'N/A')}
{sm_note}
## Summary Table

{table_md}

## Notes

- Results from the deterministic Phase 1.5 simulator.
- Oracle policy not included (use oracle.py separately for small traces).
- TTFT / TPOT only meaningful when prefill modeling is enabled.
- See docs/result_claims.md for safe interpretation of these results.
"""
    (out_dir / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
