#!/usr/bin/env python3
"""CPU-only statistical postprocessing for the repeated-trial Sarathi-vs-
vLLM validation (mistralai/Mistral-7B-Instruct-v0.1, 5 primary scenarios,
N independent trials per system via Slurm job arrays).

Reads each trial's scenario_results.json (unmodified, produced by
scripts/run_sarathi_gpu_smoke_and_validation.py and
scripts/run_gpu_external_validity_audit.py --phase mistral_match_stress),
computes per-scenario/per-system descriptive statistics across trials, and
-- because trials are matched by trial_index (both systems run trial 0..N-1
against byte-identical prompts) -- paired differences with a paired
bootstrap 95% CI for the Sarathi-vs-vLLM mean E2E/TTFT/TPOT difference.

Deliberately conservative about small-n: with N=5 (or fewer if some trials
failed), bootstrap CIs are wide and p50/p95 percentiles are only loosely
informative. This script reports sample sizes prominently and does not
compute or claim a p-value; the ROBUST/SUGGESTIVE/NOT_REPRODUCED
classification is a simple, stated, pre-registered-in-code rule, not a
significance test.

Usage:
    python scripts/analyze_repeated_trials.py \
        --sarathi-glob "/mmfs1/scratch/ikoutis/sv96/sarathi_repeated_trial_<ARRAY_JOB_ID>_*" \
        --vllm-glob "/mmfs1/scratch/ikoutis/sv96/vllm_repeated_trial_<ARRAY_JOB_ID>_*" \
        --output-dir experiments/gpu_external_validity/sarathi_vllm_repeated_trials
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import statistics
from pathlib import Path
from typing import Any

SCENARIO_NAME_PAIRS = [
    ("sarathi_long_prompt_moderate_output", "mistral_match_long_prompt_moderate_output"),
    ("sarathi_active_decode_plus_arriving_prefill", "mistral_match_active_decode_plus_arriving_prefill"),
    ("sarathi_prefill_heavy_burst", "mistral_match_prefill_heavy_burst"),
    ("sarathi_mixed_prompt_lengths", "mistral_match_mixed_prompt_lengths"),
    ("sarathi_matched_vllm_kv_pressure", "mistral_match_kv_pressure"),
]

METRICS = [("ttft_s", "mean_ttft_s"), ("tpot_s", "mean_tpot_s"), ("e2e_s", "mean_latency_s")]

RNG_SEED = 20260719
N_BOOTSTRAP = 10000


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def load_trial_dirs(pattern: str) -> list[Path]:
    dirs = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
    return dirs


def load_sarathi_trial(trial_dir: Path) -> tuple[int | None, dict[str, dict[str, float]]]:
    """Returns (trial_index, {scenario_name: {metric: value}})."""
    results_path = trial_dir / "scenario_results.json"
    smoke_path = trial_dir / "smoke_result.json"
    if not results_path.exists():
        return None, {}
    reports = json.loads(results_path.read_text())
    trial_index = None
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text())
        trial_index = smoke.get("env", {}).get("trial_index")
    out = {}
    for r in reports:
        rt = r["runtime_summary"]
        out[r["scenario_name"]] = {
            "ttft_s": rt.get("mean_ttft_s"),
            "tpot_s": rt.get("mean_tpot_s"),
            "e2e_s": rt.get("mean_latency_s"),
            "completion_fraction": rt.get("completion_fraction"),
        }
    return trial_index, out


def load_vllm_trial(trial_dir: Path) -> tuple[int | None, dict[str, dict[str, float]]]:
    results_path = trial_dir / "scenario_results.json"
    env_path = trial_dir / "environment.json"
    if not results_path.exists():
        return None, {}
    reports = json.loads(results_path.read_text())
    trial_index = None
    if env_path.exists():
        env = json.loads(env_path.read_text())
        trial_index = env.get("trial_index")
    out = {}
    for r in reports:
        rt = r["runtime_summary"]
        out[r["scenario"]["name"]] = {
            "ttft_s": rt.get("mean_ttft_s"),
            "tpot_s": rt.get("mean_tpot_s"),
            "e2e_s": rt.get("mean_latency_s"),
            "completion_fraction": rt.get("completion_fraction"),
        }
    return trial_index, out


def describe(values: list[float]) -> dict[str, Any]:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "stdev": None, "p50": None, "p95": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95) if len(values) >= 5 else None,
        "raw_values": values,
    }


def paired_bootstrap_diff_ci(pairs: list[tuple[float, float]], rng: random.Random, n_boot: int = N_BOOTSTRAP) -> dict[str, Any]:
    """pairs: list of (sarathi_value, vllm_value) for matched trial indices.
    Returns 95% CI for mean(vllm - sarathi) via paired bootstrap resampling
    of trial indices (preserves the pairing)."""
    n = len(pairs)
    if n < 2:
        return {"n_pairs": n, "note": "fewer than 2 matched pairs -- no bootstrap CI computed"}
    diffs = [v - s for s, v in pairs]
    point_estimate = statistics.fmean(diffs)
    boot_means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.fmean(sample))
    boot_means.sort()
    lo = percentile(boot_means, 0.025)
    hi = percentile(boot_means, 0.975)
    return {
        "n_pairs": n,
        "mean_diff_vllm_minus_sarathi": point_estimate,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ci_excludes_zero": (lo is not None and hi is not None and (lo > 0 or hi < 0)),
    }


def classify_robustness(sarathi_win_count: int, n_trials: int, ci_excludes_zero: bool, direction_favors_sarathi: bool) -> str:
    if n_trials == 0:
        return "NOT_REPRODUCED"
    win_frac = sarathi_win_count / n_trials
    if win_frac >= 0.8 and ci_excludes_zero and direction_favors_sarathi:
        return "ROBUST"
    if win_frac >= 0.6 and direction_favors_sarathi:
        return "SUGGESTIVE"
    return "NOT_REPRODUCED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sarathi-glob", required=True)
    parser.add_argument("--vllm-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RNG_SEED)

    sarathi_dirs = load_trial_dirs(args.sarathi_glob)
    vllm_dirs = load_trial_dirs(args.vllm_glob)

    # trial_index -> {scenario: {metric: value}}
    sarathi_trials: dict[int, dict[str, dict[str, float]]] = {}
    for d in sarathi_dirs:
        idx, data = load_sarathi_trial(d)
        if idx is not None and data:
            sarathi_trials[idx] = data
    vllm_trials: dict[int, dict[str, dict[str, float]]] = {}
    for d in vllm_dirs:
        idx, data = load_vllm_trial(d)
        if idx is not None and data:
            vllm_trials[idx] = data

    matched_trial_indices = sorted(set(sarathi_trials) & set(vllm_trials))

    summary_rows = []
    classification: dict[str, Any] = {}

    for sarathi_name, vllm_name in SCENARIO_NAME_PAIRS:
        for metric_key, _ in METRICS:
            sarathi_values = [sarathi_trials[i][sarathi_name][metric_key] for i in sarathi_trials if sarathi_name in sarathi_trials[i]]
            vllm_values = [vllm_trials[i][vllm_name][metric_key] for i in vllm_trials if vllm_name in vllm_trials[i]]
            s_stats = describe(sarathi_values)
            v_stats = describe(vllm_values)
            row = {
                "scenario": vllm_name, "metric": metric_key,
                "sarathi_n": s_stats["n"], "sarathi_mean": s_stats["mean"], "sarathi_median": s_stats["median"],
                "sarathi_stdev": s_stats["stdev"], "sarathi_p50": s_stats["p50"], "sarathi_p95": s_stats["p95"],
                "vllm_n": v_stats["n"], "vllm_mean": v_stats["mean"], "vllm_median": v_stats["median"],
                "vllm_stdev": v_stats["stdev"], "vllm_p50": v_stats["p50"], "vllm_p95": v_stats["p95"],
            }
            if s_stats["mean"] and v_stats["mean"]:
                row["sarathi_over_vllm_ratio_of_means"] = s_stats["mean"] / v_stats["mean"]
            summary_rows.append(row)

        # Paired analysis (E2E is the metric of primary scientific interest per task).
        pairs = []
        for i in matched_trial_indices:
            sd = sarathi_trials[i].get(sarathi_name)
            vd = vllm_trials[i].get(vllm_name)
            if sd and vd and sd["e2e_s"] is not None and vd["e2e_s"] is not None:
                pairs.append((sd["e2e_s"], vd["e2e_s"]))  # (sarathi, vllm)
        boot = paired_bootstrap_diff_ci(pairs, rng)
        sarathi_wins_e2e = sum(1 for s, v in pairs if s < v)
        n_trials = len(pairs)
        direction_favors_sarathi = boot.get("mean_diff_vllm_minus_sarathi", 0) is not None and boot.get("mean_diff_vllm_minus_sarathi", 0) > 0
        robustness = classify_robustness(sarathi_wins_e2e, n_trials, boot.get("ci_excludes_zero", False), direction_favors_sarathi)
        classification[vllm_name] = {
            "n_trials": n_trials,
            "sarathi_e2e_wins": sarathi_wins_e2e,
            "vllm_e2e_wins": n_trials - sarathi_wins_e2e,
            "e2e_paired_bootstrap": boot,
            "robustness": robustness,
        }

    with (out_dir / "repeated_trials_summary.csv").open("w", newline="") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    (out_dir / "repeated_trials_summary.json").write_text(json.dumps(summary_rows, indent=2, default=str))
    (out_dir / "bootstrap_comparison.json").write_text(json.dumps(classification, indent=2, default=str))

    lines = ["# Repeated-Trial Sarathi vs vLLM Statistical Summary (Mistral-7B-Instruct-v0.1)", ""]
    lines.append(f"- Sarathi trials found: {sorted(sarathi_trials.keys())} (n={len(sarathi_trials)})")
    lines.append(f"- vLLM trials found: {sorted(vllm_trials.keys())} (n={len(vllm_trials)})")
    lines.append(f"- Matched trial indices used for paired analysis: {matched_trial_indices}")
    lines.append("")
    lines.append("## E2E robustness classification per scenario")
    lines.append("")
    lines.append("| Scenario | N trials | Sarathi E2E wins | vLLM E2E wins | Mean diff (vLLM-Sarathi) | 95% CI | CI excludes 0 | Robustness |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for name, c in classification.items():
        boot = c["e2e_paired_bootstrap"]
        md = boot.get("mean_diff_vllm_minus_sarathi")
        lo, hi = boot.get("ci95_lo"), boot.get("ci95_hi")
        md_s = f"{md:.4f}" if isinstance(md, (int, float)) else "n/a"
        ci_s = f"[{lo:.4f}, {hi:.4f}]" if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else "n/a"
        lines.append(
            f"| {name} | {c['n_trials']} | {c['sarathi_e2e_wins']} | {c['vllm_e2e_wins']} | {md_s} | {ci_s} | "
            f"{boot.get('ci_excludes_zero')} | **{c['robustness']}** |"
        )
    lines.append("")
    lines.append(
        "Classification rule (stated explicitly, not a significance test): ROBUST requires Sarathi "
        "winning E2E in >=80% of trials AND the bootstrap 95% CI for the mean vLLM-minus-Sarathi E2E "
        "difference excluding zero in Sarathi's favor. SUGGESTIVE requires >=60% win rate in Sarathi's "
        "favor without a CI that excludes zero. Otherwise NOT_REPRODUCED. With N<=5 trials, bootstrap "
        "CIs are wide; do not read a ROBUST label here as a formal significance claim."
    )
    lines.append("")
    lines.append("## Full per-metric descriptive statistics")
    lines.append("")
    lines.append("| Scenario | Metric | Sarathi n/mean/median/stdev | vLLM n/mean/median/stdev |")
    lines.append("|---|---|---|---|")
    for row in summary_rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        lines.append(
            f"| {row['scenario']} | {row['metric']} | "
            f"{row['sarathi_n']}/{fmt(row['sarathi_mean'])}/{fmt(row['sarathi_median'])}/{fmt(row['sarathi_stdev'])} | "
            f"{row['vllm_n']}/{fmt(row['vllm_mean'])}/{fmt(row['vllm_median'])}/{fmt(row['vllm_stdev'])} |"
        )
    (out_dir / "repeated_trials_report.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "output_dir": str(out_dir),
        "sarathi_trials_found": sorted(sarathi_trials.keys()),
        "vllm_trials_found": sorted(vllm_trials.keys()),
        "matched_trial_indices": matched_trial_indices,
        "classification": {k: v["robustness"] for k, v in classification.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
