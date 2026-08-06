#!/usr/bin/env python3
"""Sarathi-specific stress-test headroom check.

Companion to scripts/stress_tests/run_stress_test_smoke.py (the generic
catalog runner), scoped to the 7 Sarathi-Serve entries added 2026-08-05
(configs/stress_tests/algorithm_stress_test_catalog.yaml section 12).
Does two things the generic runner does not:

1. Dumps each entry's generated workload to
   configs/stress_tests/generated/sarathi/<entry_id>_seed<seed>.json for
   at least 3 deterministic seeds (per entry, where the generator is
   randomized -- see note below), so the exact requests behind any
   reported number are inspectable and reproducible without re-running
   Python.
2. Runs the 4-way headroom comparison (sarathi_faithful vs. a non-chunked
   FCFS baseline (fifo), decode-first-without-chunking (vllm_faithful),
   and a throughput baseline (shortest_output_first), plus
   vllm_chunked_prefill_faithful as the closest real-vLLM analog) and
   reports whether each entry's declared acceptance_gates direction is
   actually reproduced -- i.e. whether the workload genuinely
   distinguishes the intended mechanism, per this project's own
   "reject or revise any workload that does not" standard.

Most of the 7 generators (all except
sarathi_counter_short_prompt_decode_dominated_regime) construct fully
deterministic request sets with no randomness at all -- matching the real
Wulver validation's own methodology (byte-identical prompts,
temperature=0.0, so any run-to-run variance is attributable to
system/execution noise, not workload content). Dumping 3 seeds for those
entries therefore produces 3 byte-identical files by design; this is
disclosed in the report, not silently redundant.

See docs/research/algorithm_stress_tests/SARATHI_MECHANISM_CALIBRATION_20260805.md
for the diagnostic investigation that led to each entry's current
acceptance_gates (a completion_fraction-based chunked-vs-non-chunked
check, not the originally-attempted sarathi_faithful-vs-
vllm_chunked_prefill_faithful latency comparison, which was found to be
structurally undistinguishable under FCFS-strict admission).

Usage: python scripts/stress_tests/run_sarathi_headroom_check.py [--full]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generators  # noqa: E402
import yaml  # noqa: E402
from run_stress_test_smoke import EXECUTABLE_ALGORITHM_IDS, run_policy  # noqa: E402

SEEDS = [0, 1, 2]
GENERATED_DIR = _ROOT / "configs" / "stress_tests" / "generated" / "sarathi"
RESULTS_DIR = _ROOT / "results" / "stress_test_catalog" / "sarathi_smoke"

_HEADROOM_COMPARISON_SET = ["fifo", "vllm_faithful", "vllm_chunked_prefill_faithful", "shortest_output_first"]


def _request_to_dict(r) -> dict:
    return {
        "request_id": r.request_id, "arrival_time": r.arrival_time, "prompt_tokens": r.prompt_tokens,
        "predicted_output_tokens": r.predicted_output_tokens, "actual_output_tokens": r.actual_output_tokens,
        "slo_deadline": r.slo_deadline, "priority": r.priority, "class_id": r.class_id,
    }


def dump_workload(entry_id: str, smoke: bool) -> List[dict]:
    gen_fn = generators.GENERATORS.get(entry_id)
    dumps = []
    for seed in SEEDS:
        try:
            reqs = gen_fn(smoke=smoke, seed=seed)
        except NotImplementedError as e:
            dumps.append({"seed": seed, "status": "NOT_EXECUTABLE", "reason": str(e)})
            continue
        payload = {
            "entry_id": entry_id, "seed": seed, "smoke": smoke, "n_requests": len(reqs),
            "requests": [_request_to_dict(r) for r in reqs],
        }
        out_path = GENERATED_DIR / f"{entry_id}_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        dumps.append({"seed": seed, "status": "GENERATED", "n_requests": len(reqs), "path": str(out_path.relative_to(_ROOT))})
    return dumps


def run_headroom(entry: dict, smoke: bool) -> dict:
    eid = entry["stress_test_id"]
    algo = entry["algorithm_id"]
    gen_fn = generators.GENERATORS.get(eid)
    sim_req = entry.get("simulator_requirements", {}) or {}
    gpu_overrides = {k: sim_req[k] for k in ("max_active_sequences", "max_batch_tokens", "max_kv_tokens") if k in sim_req}

    try:
        requests = gen_fn(smoke=smoke)
    except NotImplementedError as e:
        return {"id": eid, "status": "NOT_EXECUTABLE", "reason": str(e)}

    algos = [algo] + [a for a in _HEADROOM_COMPARISON_SET if a in EXECUTABLE_ALGORITHM_IDS]
    algos = list(dict.fromkeys(algos))
    results = {a: run_policy(requests, a, gpu_overrides, sim_req) for a in algos}

    gate_expr = entry.get("acceptance_gates", "")
    from run_stress_test_smoke import evaluate_gate
    evaluable, passed, detail = evaluate_gate(gate_expr, results, eid, {})

    return {
        "id": eid, "test_role": entry["test_role"], "evidence_class": entry["evidence_class"],
        "n_requests": len(requests), "gate": gate_expr, "gate_passed": passed, "gate_detail": detail,
        "results_by_policy": {
            a: {
                "mean_latency": r["mean_latency"], "completion_fraction": r["completion_fraction"],
                "mean_ttft": r["mean_ttft"], "mean_tpot": r["mean_tpot"],
                "throughput_completions_per_sec": r["throughput_completions_per_sec"],
                "stall_frequency_proxy": r["stall_frequency_proxy"],
                "scheduling_disagreement_proxy": r["scheduling_disagreement_proxy"],
            }
            for a, r in results.items()
        },
        "distinguishes_mechanism": (
            results.get(algo, {}).get("completion_fraction") != results.get("vllm_faithful", {}).get("completion_fraction")
            if "vllm_faithful" in results else None
        ),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", default=False)
    args = parser.parse_args()
    smoke = not args.full

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    catalog_path = _ROOT / "configs" / "stress_tests" / "algorithm_stress_test_catalog.yaml"
    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)
    sarathi_entries = [e for e in catalog["stress_tests"] if e["algorithm_id"] == "sarathi_faithful"]

    print(f"{len(sarathi_entries)} Sarathi catalog entries found\n")

    generation_report: Dict[str, list] = {}
    headroom_report: List[dict] = []
    accepted, rejected = [], []

    for entry in sarathi_entries:
        eid = entry["stress_test_id"]
        generation_report[eid] = dump_workload(eid, smoke)
        row = run_headroom(entry, smoke)
        headroom_report.append(row)

        if row.get("status") == "NOT_EXECUTABLE":
            print(f"{eid:60s} NOT_EXECUTABLE (spec-only, disclosed)")
            continue

        status = "ACCEPT" if row["gate_passed"] else "REJECT"
        (accepted if row["gate_passed"] else rejected).append(eid)
        print(f"{eid:60s} {status:8s} gate={row['gate_detail'][:70]}")

    report = {"generation": generation_report, "headroom": headroom_report,
              "accepted": accepted, "rejected": rejected}
    out_json = RESULTS_DIR / "report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    md_lines = [
        "# Sarathi Stress-Test Headroom Check",
        "",
        f"Scale: {'smoke' if smoke else 'full'}. {len(accepted)} accepted, {len(rejected)} rejected, "
        f"{sum(1 for r in headroom_report if r.get('status') == 'NOT_EXECUTABLE')} not-executable (spec-only).",
        "",
        "| entry | test_role | gate | mechanism distinguished |",
        "|---|---|---|---|",
    ]
    for row in headroom_report:
        if row.get("status") == "NOT_EXECUTABLE":
            md_lines.append(f"| {row['id']} | - | NOT_EXECUTABLE | - |")
            continue
        md_lines.append(
            f"| {row['id']} | {row['test_role']} | "
            f"{'PASS' if row['gate_passed'] else 'FAIL'} | {row['distinguishes_mechanism']} |"
        )
    out_md = RESULTS_DIR / "report.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nAccepted: {len(accepted)}  Rejected: {len(rejected)}")
    print(f"Wrote {out_json.relative_to(_ROOT)}, {out_md.relative_to(_ROOT)}")
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
