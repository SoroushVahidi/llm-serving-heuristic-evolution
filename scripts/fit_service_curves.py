#!/usr/bin/env python3
"""
Fit service curves from raw measurements.

Usage:
    python scripts/fit_service_curves.py \\
        --input results/gpu_calibration/raw_measurements.csv \\
        --output results/gpu_calibration/service_curves.json

Outputs:
    results/gpu_calibration/service_curves.json
    results/gpu_calibration/fit_report.json
    results/gpu_calibration/plots/  (various PNG files)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit service curves from calibration data")
    parser.add_argument(
        "--input",
        default="results/gpu_calibration/raw_measurements.csv",
        help="Path to raw_measurements.csv",
    )
    parser.add_argument(
        "--output",
        default="results/gpu_calibration/service_curves.json",
        help="Output path for service_curves.json",
    )
    args = parser.parse_args()

    import pandas as pd
    from llmserveopt.calibration.curve_fitting import (
        ServiceCurves,
        build_lookup_table,
        fit_decode_curve,
        fit_prefill_curve,
        generate_fit_report,
        save_service_curves,
    )
    from llmserveopt.calibration.simulator_adapter import derive_simulator_params

    input_path = ROOT / args.input
    output_path = ROOT / args.output

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        sys.exit(1)

    print(f"Loading measurements from {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows")

    # Convert boolean column
    if "skipped" in df.columns:
        df["skipped"] = df["skipped"].map({"True": True, "False": False, True: True, False: False})
    else:
        df["skipped"] = False

    n_ok = (df["skipped"] == False).sum()  # noqa: E712
    n_skip = (df["skipped"] == True).sum()  # noqa: E712
    print(f"  Valid: {n_ok}, Skipped: {n_skip}")

    print("\nFitting prefill curve ...")
    prefill_fit = fit_prefill_curve(df)
    print(f"  a0={prefill_fit.params['a0']:.6f}s  a1={prefill_fit.params['a1']:.8f}s/tok")
    print(f"  RMSE={prefill_fit.rmse*1000:.3f}ms  MAPE={prefill_fit.mape:.1f}%  R²={prefill_fit.r_squared:.4f}")

    print("\nFitting decode curve ...")
    decode_fit = fit_decode_curve(df)
    p = decode_fit.params
    print(f"  b0={p['b0']:.6f}  b1={p['b1']:.6f}  b2={p['b2']:.8f}")
    print(f"  RMSE={decode_fit.rmse*1000:.4f}ms  MAPE={decode_fit.mape:.1f}%  R²={decode_fit.r_squared:.4f}")

    print("\nBuilding lookup tables ...")
    lookup_tables = build_lookup_table(df)
    print(f"  Prefill table: {len(lookup_tables['prefill_table'])} rows")
    print(f"  Decode table:  {len(lookup_tables['decode_table'])} rows")

    # Infer model_name from data
    model_name = df["model_name"].iloc[0] if "model_name" in df.columns else "unknown"
    step_size = 0.001

    curves = ServiceCurves(
        prefill=prefill_fit,
        decode=decode_fit,
        lookup_tables=lookup_tables,
        step_size=step_size,
        model_name=model_name,
        fit_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_service_curves(curves, output_path)
    print(f"\nSaved: {output_path}")

    # Save fit report
    report = generate_fit_report(curves)
    sim_result = derive_simulator_params(curves)
    report["simulator_params"] = sim_result["simulator_params"]
    report["derivation_notes"] = sim_result["derivation_notes"]

    report_path = output_path.parent / "fit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    # Generate plots
    try:
        _generate_plots(df, curves, output_path.parent / "plots")
    except Exception as e:
        print(f"Warning: plot generation failed: {e}")

    print("\n=== Fit Summary ===")
    print(f"Prefill: a0={prefill_fit.params['a0']:.4f}s, a1={prefill_fit.params['a1']:.6e}s/token")
    print(f"  RMSE={prefill_fit.rmse*1000:.3f}ms, MAPE={prefill_fit.mape:.1f}%, R²={prefill_fit.r_squared:.4f}")
    print(f"Decode:  b0={p['b0']:.4e}, b1={p['b1']:.4e}, b2={p['b2']:.4e}")
    print(f"  RMSE={decode_fit.rmse*1000:.4f}ms, MAPE={decode_fit.mape:.1f}%, R²={decode_fit.r_squared:.4f}")
    print(f"\nSimulator params: {sim_result['simulator_params']}")


def _generate_plots(df: "pd.DataFrame", curves: object, plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir.mkdir(parents=True, exist_ok=True)
    df_ok = df[df["skipped"] == False].copy()  # noqa: E712

    # 1. Prefill: actual vs predicted
    a0 = curves.prefill.params["a0"]
    a1 = curves.prefill.params["a1"]
    y_pred_prefill = a0 + a1 * df_ok["prompt_tokens"]

    fig, ax = plt.subplots()
    ax.scatter(df_ok["prefill_time_s"] * 1000, y_pred_prefill * 1000, alpha=0.5, s=20)
    lims = [0, max(df_ok["prefill_time_s"].max(), y_pred_prefill.max()) * 1000 * 1.1]
    ax.plot(lims, lims, "r--", label="perfect")
    ax.set_xlabel("Actual prefill time (ms)")
    ax.set_ylabel("Predicted prefill time (ms)")
    ax.set_title(f"Prefill Actual vs Predicted (R²={curves.prefill.r_squared:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "fit_prefill_actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    # 2. Decode: actual vs predicted
    b0, b1, b2 = curves.decode.params["b0"], curves.decode.params["b1"], curves.decode.params["b2"]
    ctx = df_ok["prompt_tokens"] + df_ok["output_tokens"]
    y_pred_decode = b0 + b1 * df_ok["batch_size"] + b2 * ctx

    fig, ax = plt.subplots()
    ax.scatter(df_ok["decode_time_per_token_s"] * 1e3, y_pred_decode * 1e3, alpha=0.5, s=20)
    lims2 = [0, max(df_ok["decode_time_per_token_s"].max(), y_pred_decode.max()) * 1e3 * 1.1]
    ax.plot(lims2, lims2, "r--", label="perfect")
    ax.set_xlabel("Actual decode time/token (ms)")
    ax.set_ylabel("Predicted decode time/token (ms)")
    ax.set_title(f"Decode Actual vs Predicted (R²={curves.decode.r_squared:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "fit_decode_actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    # 3. Prefill latency vs tokens
    fig, ax = plt.subplots()
    for bs in sorted(df_ok["batch_size"].unique()):
        sub = df_ok[df_ok["batch_size"] == bs]
        agg = sub.groupby("prompt_tokens")["prefill_time_s"].mean().reset_index()
        ax.plot(agg["prompt_tokens"], agg["prefill_time_s"] * 1000, marker="o", label=f"bs={bs}")
    pt_range = np.linspace(df_ok["prompt_tokens"].min(), df_ok["prompt_tokens"].max(), 100)
    ax.plot(pt_range, (a0 + a1 * pt_range) * 1000, "k--", label="fit")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Prefill time (ms)")
    ax.set_title("Prefill Latency vs Token Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "prefill_latency_vs_tokens.png", dpi=120)
    plt.close(fig)

    # 4. Decode latency vs batch size
    fig, ax = plt.subplots()
    for pl in sorted(df_ok["prompt_tokens"].unique()):
        sub = df_ok[df_ok["prompt_tokens"] == pl]
        agg = sub.groupby("batch_size")["decode_time_per_token_s"].mean().reset_index()
        ax.plot(agg["batch_size"], agg["decode_time_per_token_s"] * 1e3, marker="o", label=f"prompt={pl}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decode time per token (ms)")
    ax.set_title("Decode Latency vs Batch Size")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plots_dir / "decode_latency_vs_batchsize.png", dpi=120)
    plt.close(fig)

    # 5. Residuals prefill
    residuals_prefill = (df_ok["prefill_time_s"] - y_pred_prefill) * 1000
    fig, ax = plt.subplots()
    ax.scatter(df_ok["prompt_tokens"], residuals_prefill, alpha=0.5, s=20)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("Prefill Residuals")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_prefill.png", dpi=120)
    plt.close(fig)

    # 6. Residuals decode
    residuals_decode = (df_ok["decode_time_per_token_s"] - y_pred_decode) * 1e3
    fig, ax = plt.subplots()
    ax.scatter(df_ok["batch_size"], residuals_decode, alpha=0.5, s=20)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("Decode Residuals")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_decode.png", dpi=120)
    plt.close(fig)

    # 7. Memory vs tokens
    if "peak_memory_gb" in df_ok.columns:
        fig, ax = plt.subplots()
        for bs in sorted(df_ok["batch_size"].unique()):
            sub = df_ok[df_ok["batch_size"] == bs]
            agg = sub.groupby("prompt_tokens")["peak_memory_gb"].mean().reset_index()
            ax.plot(agg["prompt_tokens"], agg["peak_memory_gb"], marker="o", label=f"bs={bs}")
        ax.set_xlabel("Prompt tokens")
        ax.set_ylabel("Peak GPU memory (GB)")
        ax.set_title("GPU Memory vs Token Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "memory_vs_tokens.png", dpi=120)
        plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
