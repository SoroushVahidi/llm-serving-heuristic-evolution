#!/usr/bin/env python3
"""
Cheap calibration sweep for AdmissionControlPolicy laxity_threshold.

Runs a small workload under multiple threshold values and reports:
- admission_rate (fraction of requests admitted)
- weighted_goodput
- slo_violation_rate
- completion_fraction

Usage:
    python scripts/calibrate_admission_threshold.py [--help]
    python scripts/calibrate_admission_threshold.py --out-dir results/phase2b6_fair_sweep_failure_audit/admission_calibration

Outputs:
    <out-dir>/calibration_results.csv
    <out-dir>/calibration_summary.md
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.core.types import GPUConfig
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.admission_control import AdmissionControlPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.synthetic import WorkloadConfig, SLOClass


STEP_SIZE = 0.001  # seconds per step
SEEDS = [42, 0, 1]
THRESHOLDS = [
    float("inf"),   # no filtering
    200.0,          # very permissive (~0.2s of slack in raw unit)
    100.0,          # moderate
    50.0,           # tight
    0.0,            # zero: admit only non-negative raw laxity
    -50.0,          # negative: admit even some over-budget requests
    -100.0,         # very permissive (only drops requests with est >> deadline)
]

GPU_CFG = GPUConfig(gpu_id=0, max_active_sequences=4, max_batch_tokens=4096, max_kv_tokens=32768)

WORKLOAD = WorkloadConfig(
    tag="admission_calib",
    arrival_process="poisson",
    arrival_rate=25.0,
    duration=10.0,
    prompt_dist="lognormal",
    prompt_mean=128.0,
    prompt_sigma=0.7,
    prompt_low=16,
    prompt_high=512,
    output_dist="lognormal",
    output_mean=96.0,
    output_sigma=0.7,
    output_low=8,
    output_high=512,
    prediction_noise_rel=0.15,
    slo_classes=[
        SLOClass(class_id="tight",  slo_slack=0.4,  priority=3.0, weight=0.50),
        SLOClass(class_id="medium", slo_slack=5.0,  priority=2.0, weight=0.35),
        SLOClass(class_id="loose",  slo_slack=20.0, priority=1.0, weight=0.15),
    ],
)

SERVICE_MODEL = ServiceModel(
    enable_prefill_modeling=True,
    prefill_cost_per_token=1.0,
    max_prefill_chunk_tokens=512,
    step_token_budget=4096,
    decode_first=False,
)


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def run_calibration(out_dir: Path):
    from llmserveopt.workloads.synthetic import generate_workload

    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate traces once per seed
    traces = {}
    for seed in SEEDS:
        traces[seed] = generate_workload(WORKLOAD, seed=seed)

    rows = []
    for threshold in THRESHOLDS:
        wgs, viols, comps, admits = [], [], [], []
        for seed, requests in traces.items():
            policy = AdmissionControlPolicy(laxity_threshold=threshold)
            m = run_policy(
                policy=policy,
                requests=requests,
                gpu_configs=[GPU_CFG],
                service_model=SERVICE_MODEL,
                workload_tag=WORKLOAD.tag,
                seed=seed,
                drain_steps=20000,
            )
            wgs.append(m.weighted_goodput)
            viols.append(m.slo_violation_rate)
            comps.append(m.completion_fraction)
            n_total = m.num_total if m.num_total > 0 else len(requests)
            admits.append(m.num_completed / n_total if n_total > 0 else float("nan"))

        threshold_label = "inf" if threshold == float("inf") else f"{threshold:.1f}"
        rows.append({
            "laxity_threshold": threshold_label,
            "mean_wg": round(_mean(wgs), 4),
            "mean_slo_violation_rate": round(_mean(viols), 4),
            "mean_completion_fraction": round(_mean(comps), 4),
            "mean_admission_rate": round(_mean(admits), 4),
            "note": _note(threshold),
        })
        print(f"  threshold={threshold_label:>8}  wg={_mean(wgs):.4f}  slo_viol={_mean(viols):.4f}  "
              f"completion={_mean(comps):.4f}")

    # Write CSV
    csv_path = out_dir / "calibration_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to: {csv_path}")

    # Write markdown summary
    _write_summary(out_dir / "calibration_summary.md", rows)
    return rows


def _note(threshold: float) -> str:
    if threshold == float("inf"):
        return "default — no filtering, pure urgency sort"
    elif threshold >= 200:
        return "very permissive — drops only deeply infeasible"
    elif threshold >= 50:
        return "moderate filtering"
    elif threshold == 0.0:
        return "zero — admits requests with non-negative raw laxity"
    elif threshold < 0:
        return "negative — admits requests even if laxity is negative by this margin"
    return ""


def _write_summary(path: Path, rows: list):
    lines = [
        "# AdmissionControlPolicy Threshold Calibration",
        "",
        "**Generated by:** `scripts/calibrate_admission_threshold.py`",
        "**Workload:** mixed-SLO, Poisson 25 req/s, 10s, 3 seeds",
        "**GPU:** 1×RTX-5060Ti proxy (max_active_seq=4)",
        "",
        "## Unit Note",
        "",
        "The laxity formula mixes units:",
        "```",
        "laxity = slo_deadline(s) - now(s) - est(steps)",
        "```",
        "where `est = α×prompt_tokens + β×predicted_output_tokens` is in **decode steps**",
        "(dimensionless), while `(slo_deadline - now)` is in **seconds**.",
        "",
        f"With `step_size={STEP_SIZE}`, one step ≈ {STEP_SIZE*1000:.0f} ms.",
        "A service proxy of 100 steps corresponds to ≈0.1s wall time.",
        "A tight SLO slack of 0.4s → (deadline - now) ≈ 0.4, while est ≈ 100.",
        "So raw laxity ≈ 0.4 − 100 = −99.6 even for a fully feasible request.",
        "",
        "**Implication:** `laxity_threshold=0` drops nearly all requests.",
        "Use `laxity_threshold=float('inf')` (default) to disable filtering.",
        "To enable meaningful filtering, convert est to seconds: `est_s = est * step_size`.",
        "",
        "## Results",
        "",
        "| threshold | mean_wg | slo_violation | completion_fraction | note |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['laxity_threshold']} | {r['mean_wg']} | {r['mean_slo_violation_rate']} | "
            f"{r['mean_completion_fraction']} | {r['note']} |"
        )
    lines += [
        "",
        "## Recommendations",
        "",
        "1. **Default (`inf`):** All requests admitted; policy acts as urgency-sorted FIFO.",
        "   Appropriate when admission filtering is not the research goal.",
        "",
        "2. **Threshold calibration:** To use as a genuine admission-control filter,",
        "   compute laxity in consistent units:",
        "   ```python",
        "   # In AdmissionControlPolicy._laxity(), use step-time units:",
        "   laxity_steps = (req.slo_deadline - now) / step_size - est",
        "   # Then set threshold in steps, e.g., threshold=0 means 'drop if est > remaining steps'",
        "   ```",
        "",
        "3. **Current state:** The default `threshold=inf` is safe and well-tested.",
        "   The unit mismatch is documented in `docs/external_baseline_correctness_audit.md`.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Summary saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate AdmissionControlPolicy threshold")
    parser.add_argument("--out-dir", default="results/phase2b6_fair_sweep_failure_audit/admission_calibration")
    args = parser.parse_args()
    print("=== AdmissionControlPolicy Threshold Calibration ===")
    print(f"Workload: {WORKLOAD.tag}, rate={WORKLOAD.arrival_rate}, seeds={SEEDS}")
    print(f"Thresholds: {THRESHOLDS}")
    print()
    run_calibration(Path(args.out_dir))


if __name__ == "__main__":
    main()
