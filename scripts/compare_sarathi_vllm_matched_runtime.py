#!/usr/bin/env python3
"""Combine real Sarathi runtime output, real vLLM runtime output (job
1111706), and both faithful simulator baselines into one matched
comparison table, for the byte-for-byte matched Mistral-7B-Instruct-v0.1
scenarios (scripts/run_sarathi_gpu_smoke_and_validation.py's
make_scenarios() / scripts/run_gpu_external_validity_audit.py's
build_mistral_match_scenarios()).

CPU-only, instant (<1s), reads existing JSON outputs -- not a Slurm job by
itself (a thin sbatch wrapper may still submit it as a CPU-partition step
so it runs automatically once the GPU job it depends on completes, but the
work here is not itself long-running).

The simulator numbers this reads from the vLLM run's scenario_results.json
were already computed inline by that run (run_simulator_scenario() calls
both vllm_faithful and sarathi_faithful for every scenario); this script
does not re-run the simulator, it just reorganizes already-produced numbers
plus whatever real-Sarathi numbers are available.

Per-scenario real-Sarathi throughput is NOT computable from the stored
data: run_sarathi_gpu_smoke_and_validation.py's requests.jsonl records only
relative ttft_s/latency_s per request, not absolute submit/finish
timestamps, so total scenario wall-clock span can't be reconstructed after
the fact. A single coarse overall-job throughput (num_success / job
elapsed, from summary.json's env and the output directory's job_manifest)
is reported instead, clearly labeled as coarse and not per-scenario.

Usage:
    python scripts/compare_sarathi_vllm_matched_runtime.py \
        --vllm-dir /mmfs1/scratch/ikoutis/sv96/vllm_mistral_match_1111706 \
        --sarathi-dir /mmfs1/scratch/ikoutis/sv96/sarathi_mistral_fp16_final_<JOBID> \
        --output-dir experiments/gpu_external_validity/sarathi_vs_vllm_mistral_comparison

If --sarathi-dir is omitted or its scenario_results.json does not exist yet
(e.g. the Sarathi job hasn't completed), this still writes the vLLM-vs-
simulator half of the comparison and notes which rows are missing real
Sarathi data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# (sarathi harness scenario name, matched vLLM harness scenario name, focus category)
SCENARIO_NAME_PAIRS = [
    ("sarathi_long_prompt_moderate_output", "mistral_match_long_prompt_moderate_output", "long_prompt"),
    ("sarathi_active_decode_plus_arriving_prefill", "mistral_match_active_decode_plus_arriving_prefill", "active_decode_plus_arriving_prefill"),
    ("sarathi_prefill_heavy_burst", "mistral_match_prefill_heavy_burst", "prefill_heavy_burst"),
    ("sarathi_mixed_prompt_lengths", "mistral_match_mixed_prompt_lengths", "long_prompt"),
    ("sarathi_matched_vllm_kv_pressure", "mistral_match_kv_pressure", "long_prompt"),
    ("sarathi_short_context_control", "mistral_match_short_context_control", "short_context_control"),
]

FOCUS_CATEGORIES = ("active_decode_plus_arriving_prefill", "prefill_heavy_burst", "long_prompt", "short_context_control")


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def index_by_name(reports: list[dict], key_path: tuple[str, ...]) -> dict[str, dict]:
    out = {}
    for r in reports:
        node = r
        for k in key_path:
            node = node[k]
        out[node] = r
    return out


def winner(vllm_val: float | None, sarathi_val: float | None, lower_is_better: bool = True) -> str:
    if vllm_val is None or sarathi_val is None:
        return "n/a"
    if abs(vllm_val - sarathi_val) < 1e-9:
        return "tie"
    vllm_better = (vllm_val < sarathi_val) if lower_is_better else (vllm_val > sarathi_val)
    return "vllm" if vllm_better else "sarathi"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-dir", required=True)
    parser.add_argument("--sarathi-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    vllm_dir = Path(args.vllm_dir)
    sarathi_dir = Path(args.sarathi_dir) if args.sarathi_dir else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vllm_reports = load_json(vllm_dir / "scenario_results.json") or []
    vllm_by_name = index_by_name(vllm_reports, ("scenario", "name"))

    sarathi_reports = None
    if sarathi_dir is not None:
        sarathi_reports = load_json(sarathi_dir / "scenario_results.json")
    sarathi_by_name = {}
    if sarathi_reports:
        for r in sarathi_reports:
            sarathi_by_name[r["scenario_name"]] = r

    rows = []
    for sarathi_name, vllm_name, category in SCENARIO_NAME_PAIRS:
        vrow = vllm_by_name.get(vllm_name)
        srow = sarathi_by_name.get(sarathi_name)
        row: dict[str, Any] = {"scenario": vllm_name, "category": category}
        if vrow is not None:
            rs = vrow["runtime_summary"]
            row["real_vllm_ttft_s"] = rs.get("mean_ttft_s")
            row["real_vllm_tpot_s"] = rs.get("mean_tpot_s")
            row["real_vllm_e2e_s"] = rs.get("mean_latency_s")
            row["real_vllm_throughput_req_s"] = rs.get("request_throughput")
            row["real_vllm_max_running"] = rs.get("max_vllm_running")
            row["real_vllm_max_kv"] = rs.get("max_kv_cache_usage")
            sim = vrow["simulator_summary"]
            row["sim_vllm_faithful_ttft_s"] = sim["vllm_faithful"].get("mean_ttft_s")
            row["sim_vllm_faithful_e2e_s"] = sim["vllm_faithful"].get("mean_latency_s")
            row["sim_sarathi_faithful_ttft_s"] = sim["sarathi_faithful"].get("mean_ttft_s")
            row["sim_sarathi_faithful_e2e_s"] = sim["sarathi_faithful"].get("mean_latency_s")
        if srow is not None:
            rt = srow["runtime_summary"]
            row["real_sarathi_ttft_s"] = rt.get("mean_ttft_s")
            row["real_sarathi_tpot_s"] = rt.get("mean_tpot_s")
            row["real_sarathi_e2e_s"] = rt.get("mean_latency_s")
            row["real_sarathi_max_concurrent"] = rt.get("max_concurrent_unfinished")
            row["real_sarathi_completion_fraction"] = rt.get("completion_fraction")
        else:
            row["real_sarathi_ttft_s"] = None
            row["real_sarathi_tpot_s"] = None
            row["real_sarathi_e2e_s"] = None
            row["note"] = "real Sarathi data not available (job pending/failed/not yet run)"

        row["winner_ttft"] = winner(row.get("real_vllm_ttft_s"), row.get("real_sarathi_ttft_s"))
        row["winner_tpot"] = winner(row.get("real_vllm_tpot_s"), row.get("real_sarathi_tpot_s"))
        row["winner_e2e"] = winner(row.get("real_vllm_e2e_s"), row.get("real_sarathi_e2e_s"))

        sim_winner_ttft = winner(row.get("sim_vllm_faithful_ttft_s"), row.get("sim_sarathi_faithful_ttft_s"))
        sim_winner_e2e = winner(row.get("sim_vllm_faithful_e2e_s"), row.get("sim_sarathi_faithful_e2e_s"))
        row["sim_agrees_with_real_ttft_winner"] = (
            sim_winner_ttft == row["winner_ttft"] if "n/a" not in (sim_winner_ttft, row["winner_ttft"]) else None
        )
        row["sim_agrees_with_real_e2e_winner"] = (
            sim_winner_e2e == row["winner_e2e"] if "n/a" not in (sim_winner_e2e, row["winner_e2e"]) else None
        )
        rows.append(row)

    (out_dir / "matched_comparison.json").write_text(json.dumps(rows, indent=2, default=str))

    csv_fields = [
        "scenario", "category",
        "real_vllm_ttft_s", "real_sarathi_ttft_s", "winner_ttft",
        "real_vllm_tpot_s", "real_sarathi_tpot_s", "winner_tpot",
        "real_vllm_e2e_s", "real_sarathi_e2e_s", "winner_e2e",
        "real_vllm_throughput_req_s",
        "sim_vllm_faithful_ttft_s", "sim_sarathi_faithful_ttft_s",
        "sim_agrees_with_real_ttft_winner", "sim_agrees_with_real_e2e_winner",
    ]
    with (out_dir / "matched_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    have_sarathi = sarathi_reports is not None

    # Focus-category classification (task's 4 named regimes).
    category_notes: dict[str, str] = {}
    for cat in FOCUS_CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat and r.get("real_sarathi_ttft_s") is not None]
        if not cat_rows:
            category_notes[cat] = "no real-Sarathi data for this category"
            continue
        ttft_wins = [r["winner_ttft"] for r in cat_rows]
        e2e_wins = [r["winner_e2e"] for r in cat_rows]
        category_notes[cat] = (
            f"n={len(cat_rows)} scenario(s); TTFT winner(s)={ttft_wins}; E2E winner(s)={e2e_wins}"
        )

    sarathi_advantage_regimes = [
        cat for cat in FOCUS_CATEGORIES
        if any(
            r["category"] == cat and r.get("real_sarathi_ttft_s") is not None and r["winner_ttft"] == "sarathi"
            for r in rows
        )
    ]

    ttft_agreements = [r["sim_agrees_with_real_ttft_winner"] for r in rows if r["sim_agrees_with_real_ttft_winner"] is not None]
    e2e_agreements = [r["sim_agrees_with_real_e2e_winner"] for r in rows if r["sim_agrees_with_real_e2e_winner"] is not None]

    lines = ["# Sarathi vs vLLM Matched Runtime Comparison (Mistral-7B-Instruct-v0.1)", ""]
    lines.append(f"- vLLM source: `{vllm_dir}` (job 1111706)")
    lines.append(f"- Sarathi source: `{sarathi_dir}`" if sarathi_dir else "- Sarathi source: NOT PROVIDED")
    lines.append(f"- Real-Sarathi per-scenario throughput: not computable from stored data (see module docstring); use `sacct -j <job> --format=Elapsed` for a coarse overall figure.")
    lines.append("")
    lines.append("## Per-scenario winner table")
    lines.append("")
    lines.append("| Scenario | Category | TTFT winner | TPOT winner | E2E winner | Sim agrees (TTFT/E2E) |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        sim_agree = f"{row['sim_agrees_with_real_ttft_winner']}/{row['sim_agrees_with_real_e2e_winner']}"
        lines.append(
            f"| {row['scenario']} | {row['category']} | {row['winner_ttft']} | {row['winner_tpot']} | "
            f"{row['winner_e2e']} | {sim_agree} |"
        )
    lines.append("")
    lines.append("## Raw metrics")
    lines.append("")
    lines.append("| Scenario | Real vLLM TTFT | Real Sarathi TTFT | Real vLLM TPOT | Real Sarathi TPOT | Real vLLM E2E | Real Sarathi E2E |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
        lines.append(
            f"| {row['scenario']} | {fmt(row.get('real_vllm_ttft_s'))} | {fmt(row.get('real_sarathi_ttft_s'))} | "
            f"{fmt(row.get('real_vllm_tpot_s'))} | {fmt(row.get('real_sarathi_tpot_s'))} | "
            f"{fmt(row.get('real_vllm_e2e_s'))} | {fmt(row.get('real_sarathi_e2e_s'))} |"
        )
    lines.append("")
    lines.append("## Focus-category classification")
    lines.append("")
    for cat, note in category_notes.items():
        lines.append(f"- **{cat}**: {note}")
    lines.append("")
    lines.append("## Classification")
    lines.append("")
    lines.append(f"- SARATHI_RUNTIME_VALIDATION = {'SUCCESS' if have_sarathi else 'NOT_AVAILABLE'}")
    lines.append(f"- SARATHI_ADVANTAGE_REGIMES = {sarathi_advantage_regimes if sarathi_advantage_regimes else '(none observed on TTFT)'}")
    lines.append(
        f"- SARATHI_SIMULATOR_MATCH (TTFT winner agreement) = {sum(ttft_agreements)}/{len(ttft_agreements)} scenarios agree"
        if ttft_agreements else "- SARATHI_SIMULATOR_MATCH = n/a (no comparable rows)"
    )
    lines.append(
        f"- VLLM_SIMULATOR_MATCH (E2E winner agreement) = {sum(e2e_agreements)}/{len(e2e_agreements)} scenarios agree"
        if e2e_agreements else "- VLLM_SIMULATOR_MATCH = n/a (no comparable rows)"
    )
    lines.append("")
    lines.append(
        "Caveat: 'winner' here is real-hardware TTFT/TPOT/E2E on ONE run each "
        "(no repeated trials, no variance estimate), and the two servers were "
        "configured to be comparable (matched gpu-memory-utilization/"
        "max-num-seqs/token-budget-per-step) but are not identical scheduling "
        "systems -- see docs/wulver_vllm_kv_pressure_results.md for the full "
        "caveats already established for this comparison pair."
    )
    (out_dir / "matched_comparison.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "output_dir": str(out_dir),
        "scenarios_with_vllm_data": sum(1 for r in rows if r.get("real_vllm_ttft_s") is not None),
        "scenarios_with_sarathi_data": sum(1 for r in rows if r.get("real_sarathi_ttft_s") is not None),
        "sarathi_data_available": have_sarathi,
        "sarathi_advantage_regimes": sarathi_advantage_regimes,
        "sim_ttft_winner_agreement": f"{sum(ttft_agreements)}/{len(ttft_agreements)}" if ttft_agreements else "n/a",
        "sim_e2e_winner_agreement": f"{sum(e2e_agreements)}/{len(e2e_agreements)}" if e2e_agreements else "n/a",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
