#!/usr/bin/env python3
"""
Validate simulator predictions against held-out GPU measurements.

Usage:
    python scripts/validate_simulator_calibration.py \\
        --config configs/gpu_calibration/validation_grid.yaml \\
        2>&1 | tee results/gpu_calibration/validation.log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate simulator calibration")
    parser.add_argument(
        "--config",
        default="configs/gpu_calibration/validation_grid.yaml",
        help="Path to validation_grid.yaml",
    )
    parser.add_argument(
        "--curves",
        default="results/gpu_calibration/service_curves.json",
        help="Path to service_curves.json",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    curves_path = ROOT / args.curves

    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)
    if not curves_path.exists():
        print(f"ERROR: service_curves.json not found: {curves_path}")
        print("Run fit_service_curves.py first.")
        sys.exit(1)

    config = load_yaml(config_path)
    model_config_path = ROOT / config["model_config"]
    model_config = load_yaml(model_config_path)
    grid_config = config["grid"]
    val_csv = ROOT / config["output"]["raw_measurements"]

    model_name = model_config["model"]["name"]
    dtype = model_config["model"]["dtype"]
    device_map = model_config["model"]["device_map"]
    trust_remote = model_config["model"].get("trust_remote_code", False)
    seed = grid_config.get("seed", 43)

    print("=== Validation Grid ===")
    print(f"  model: {model_name}")
    print(f"  validation points:")
    print(f"    prompt_lengths: {grid_config['prompt_lengths']}")
    print(f"    output_lengths: {grid_config['output_lengths']}")
    print(f"    batch_sizes: {grid_config['batch_sizes']}")

    from llmserveopt.calibration.benchmark_backend import BenchmarkBackend
    from llmserveopt.calibration.curve_fitting import load_service_curves
    from llmserveopt.simulator.calibrated_service_model import CalibratedServiceModel

    import pandas as pd

    # Run validation GPU measurements
    backend = BenchmarkBackend(
        model_name=model_name,
        dtype=dtype,
        device_map=device_map,
        seed=seed,
        trust_remote_code=trust_remote,
    )
    val_results = backend.run_validation_grid(grid_config, output_csv=val_csv)

    # Load service curves
    load_service_curves(curves_path)
    csm = CalibratedServiceModel(calibration_file=curves_path, step_size=0.001)

    # Build comparison table
    rows = []
    for r in val_results:
        if r.skipped:
            continue
        actual_prefill_s = r.prefill_time_mean
        actual_decode_per_tok_s = r.decode_time_per_token_mean

        pred_prefill_steps = csm.compute_prefill_steps(r.prompt_tokens)
        pred_prefill_s = pred_prefill_steps * 0.001

        pred_decode_per_tok_s = csm.compute_decode_step_time(
            batch_size=r.batch_size,
            context_tokens=r.prompt_tokens + r.output_tokens,
        )

        prefill_err_pct = abs(actual_prefill_s - pred_prefill_s) / max(actual_prefill_s, 1e-9) * 100
        decode_err_pct = abs(actual_decode_per_tok_s - pred_decode_per_tok_s) / max(actual_decode_per_tok_s, 1e-9) * 100

        rows.append({
            "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.output_tokens,
            "batch_size": r.batch_size,
            "actual_prefill_ms": actual_prefill_s * 1000,
            "predicted_prefill_ms": pred_prefill_s * 1000,
            "prefill_error_pct": prefill_err_pct,
            "actual_decode_per_tok_us": actual_decode_per_tok_s * 1e6,
            "predicted_decode_per_tok_us": pred_decode_per_tok_s * 1e6,
            "decode_error_pct": decode_err_pct,
        })

    val_df = pd.DataFrame(rows)
    val_summary_path = ROOT / "results/gpu_calibration/validation_summary.csv"
    val_df.to_csv(val_summary_path, index=False)
    print(f"\nSaved: {val_summary_path}")

    # Compute summary metrics
    if len(val_df) > 0:
        mape_prefill = val_df["prefill_error_pct"].mean()
        max_err_prefill = val_df["prefill_error_pct"].max()
        mape_decode = val_df["decode_error_pct"].mean()
        max_err_decode = val_df["decode_error_pct"].max()

        calibration_sufficient = bool(mape_prefill < 20.0 and mape_decode < 20.0)
        report = {
            "n_validation_points": int(len(val_df)),
            "prefill": {
                "mape_pct": float(round(mape_prefill, 2)),
                "max_error_pct": float(round(max_err_prefill, 2)),
            },
            "decode": {
                "mape_pct": float(round(mape_decode, 2)),
                "max_error_pct": float(round(max_err_decode, 2)),
            },
            "calibration_sufficient": calibration_sufficient,
            "notes": [
                "Prefill MAPE < 20% considered acceptable for Phase 2 calibration.",
                "Decode MAPE < 20% considered acceptable for Phase 2 calibration.",
            ],
        }
    else:
        report = {"n_validation_points": 0, "error": "No valid validation data"}

    val_report_path = ROOT / "results/gpu_calibration/validation_report.json"
    with open(val_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {val_report_path}")

    # Generate plots
    if len(val_df) > 0:
        try:
            _generate_validation_plots(val_df, ROOT / "results/gpu_calibration/plots")
        except Exception as e:
            print(f"Warning: plot generation failed: {e}")

    print("\n=== Validation Summary ===")
    if len(val_df) > 0:
        print(f"  N points: {len(val_df)}")
        print(f"  Prefill MAPE: {report['prefill']['mape_pct']:.1f}%  max_err: {report['prefill']['max_error_pct']:.1f}%")
        print(f"  Decode MAPE:  {report['decode']['mape_pct']:.1f}%  max_err: {report['decode']['max_error_pct']:.1f}%")
        print(f"  Sufficient for Phase 2: {report.get('calibration_sufficient', False)}")
    else:
        print("  No valid data.")


def _generate_validation_plots(val_df: "pd.DataFrame", plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Actual vs predicted latency (prefill)
    fig, ax = plt.subplots()
    ax.scatter(val_df["actual_prefill_ms"], val_df["predicted_prefill_ms"], alpha=0.7)
    lim = max(val_df["actual_prefill_ms"].max(), val_df["predicted_prefill_ms"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", label="perfect")
    ax.set_xlabel("Actual prefill (ms)")
    ax.set_ylabel("Predicted prefill (ms)")
    ax.set_title("Validation: Actual vs Predicted Prefill Latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted_latency.png", dpi=120)
    plt.close(fig)

    # 2. Actual vs predicted throughput (decode)
    fig, ax = plt.subplots()
    ax.scatter(val_df["actual_decode_per_tok_us"], val_df["predicted_decode_per_tok_us"], alpha=0.7)
    lim2 = max(val_df["actual_decode_per_tok_us"].max(), val_df["predicted_decode_per_tok_us"].max()) * 1.1
    ax.plot([0, lim2], [0, lim2], "r--", label="perfect")
    ax.set_xlabel("Actual decode per token (μs)")
    ax.set_ylabel("Predicted decode per token (μs)")
    ax.set_title("Validation: Actual vs Predicted Decode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted_throughput.png", dpi=120)
    plt.close(fig)

    # 3. Residuals
    fig, ax = plt.subplots()
    residuals = val_df["actual_prefill_ms"] - val_df["predicted_prefill_ms"]
    ax.scatter(val_df["actual_prefill_ms"], residuals, alpha=0.7)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Actual prefill (ms)")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("Validation: Prefill Residuals")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals.png", dpi=120)
    plt.close(fig)

    # 4. Error by prompt length
    fig, ax = plt.subplots()
    agg = val_df.groupby("prompt_tokens")["prefill_error_pct"].mean().reset_index()
    ax.bar(agg["prompt_tokens"].astype(str), agg["prefill_error_pct"])
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Prefill MAPE (%)")
    ax.set_title("Validation: Prefill Error by Prompt Length")
    fig.tight_layout()
    fig.savefig(plots_dir / "error_by_prompt_length.png", dpi=120)
    plt.close(fig)

    # 5. Error by batch size
    fig, ax = plt.subplots()
    agg2 = val_df.groupby("batch_size")["decode_error_pct"].mean().reset_index()
    ax.bar(agg2["batch_size"].astype(str), agg2["decode_error_pct"])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decode MAPE (%)")
    ax.set_title("Validation: Decode Error by Batch Size")
    fig.tight_layout()
    fig.savefig(plots_dir / "error_by_batch_size.png", dpi=120)
    plt.close(fig)

    print(f"  Validation plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
