#!/usr/bin/env python3
"""Evaluate one external baseline on frozen public_trace_stress_v1 (M,C).

Does not modify P6 public-trace Layer-3 outputs. Policy-blind stress already
frozen in stress_protocol.json — this script only evaluates named baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from dataclasses import replace  # noqa: E402

from llmserveopt.core.metrics import metrics_to_dict  # noqa: E402
from llmserveopt.core.types import GPUConfig  # noqa: E402
from llmserveopt.policy_separation import public_trace_replay_v1 as ptr  # noqa: E402
from llmserveopt.policy_separation.schema import PolicySeparationScenario  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402


def load_protocol() -> dict:
    return json.loads((REPO / "experiments/external_baseline_comparison_v1/stress_protocol.json").read_text())


def transform(scenario, M: float, C: int) -> PolicySeparationScenario:
    reqs = []
    for r in scenario.requests:
        arr = float(r.arrival_time) / M
        slack = max(0.0, float(r.slo_deadline) - float(r.arrival_time))
        reqs.append(replace(r, arrival_time=arr, slo_deadline=arr + slack))
    g0 = scenario.gpu_configs[0]
    gpu = GPUConfig(
        gpu_id=g0.gpu_id,
        max_active_sequences=int(C),
        max_batch_tokens=int(C),
        max_kv_tokens=int(g0.max_kv_tokens),
    )
    params = dict(scenario.params)
    params["time_compression_M"] = float(M)
    params["max_active_sequences_override"] = int(C)
    return PolicySeparationScenario(
        scenario_id=f"{scenario.scenario_id}__stress_M{int(M)}_C{C}",
        family=scenario.family,
        template_name=scenario.template_name,
        generator_version=scenario.generator_version,
        seed=scenario.seed,
        params=params,
        requests=tuple(sorted(reqs, key=lambda x: (x.arrival_time, x.request_id))),
        gpu_configs=(gpu,),
        service_model_kwargs=dict(scenario.service_model_kwargs),
        target_policy_family=scenario.target_policy_family,
        target_mechanism="public_trace_stress_v1",
        expected_qualitative_hypothesis=scenario.expected_qualitative_hypothesis,
        stress_control_relationship="stress",
    )


def make_policy(name: str, scenario):
    if name == "vllm_style_continuous_batching":
        from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy

        return VLLMFaithfulPolicy(), {"allow_chunked_prefill": False, "decode_first": True}
    if name == "official_vtc_joint_token_budget_remap":
        from baselines.vtc.adapter.simulator_policy import VTCFairnessPolicy

        tenants = sorted({r.class_id for r in scenario.requests})
        max_prompt = max(int(r.prompt_tokens) for r in scenario.requests)
        step_budget = int(scenario.service_model_kwargs.get("step_token_budget", 512))
        budget = max(step_budget, max_prompt)
        return VTCFairnessPolicy(known_tenants=tenants, batch_token_budget_override=budget), {}
    raise SystemExit(f"unsupported baseline {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        required=True,
        choices=["vllm_style_continuous_batching", "official_vtc_joint_token_budget_remap"],
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    proto = load_protocol()
    if proto.get("status") != "PUBLIC_TRACE_STRESS_V1_FROZEN":
        print(f"FATAL: stress not frozen; status={proto.get('status')}", flush=True)
        return 2
    M = float(proto["selected_primary_M"])
    C = int(proto["selected_primary_C"])

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "cells.jsonl"
    log = out_dir / "run.log"
    if jsonl.exists():
        jsonl.unlink()

    records = [
        r
        for r in ptr.build_all_scenarios()
        if r["scenario_evidence_class"] == ptr.AUGMENTED
    ]
    if args.limit > 0:
        records = records[: args.limit]

    started = datetime.now(timezone.utc).isoformat()
    ok = fail = 0
    t_all = time.time()
    with log.open("a") as lf:
        msg = f"start {started} baseline={args.baseline} M={M} C={C} n={len(records)}"
        print(f"[stress_pub] {msg}", flush=True)
        lf.write(msg + "\n")
        lf.flush()

        for idx, rec in enumerate(records):
            base_sc = rec["scenario"]
            s = transform(base_sc, M, C)
            t0 = time.time()
            row = {
                "workload_id": "public_trace_stress_v1",
                "source_dataset": rec["source_dataset"],
                "window_index": rec["window_index"],
                "canonical_scenario_id": rec["canonical_scenario_id"],
                "scenario_id": s.scenario_id,
                "baseline": args.baseline,
                "seed": int(s.seed),
                "M": M,
                "C": C,
                "implementation_version": args.baseline,
                "config_hash": f"stress_v1_M{int(M)}_C{C}",
                "raw_result_pointer": str(jsonl),
            }
            try:
                policy, sm_over = make_policy(args.baseline, s)
                sm = dict(s.service_model_kwargs)
                sm.update(sm_over)
                sim = Simulator(
                    SimulatorConfig(
                        gpu_configs=list(s.gpu_configs),
                        service_model=ServiceModel(**sm),
                        max_steps=200_000,
                        drain_steps=50_000,
                    )
                )
                sim.load_trace(list(s.requests))
                metrics = sim.run(policy, workload_tag=s.scenario_id, seed=s.seed)
                md = metrics_to_dict(metrics)
                slo_viol = md.get("slo_violation_rate")
                if slo_viol is None:
                    completed = list(getattr(sim, "_completed", []) or [])
                    slo_viol = (
                        sum(1 for c in completed if getattr(c, "slo_violated", False))
                        / max(len(s.requests), 1)
                        if completed
                        else 0.0
                    )
                row.update(
                    {
                        "failure_status": "success",
                        "anwg": float(metrics.arrival_normalized_weighted_goodput),
                        "slo_attainment_proxy": float(1.0 - float(slo_viol)),
                        "completion_fraction": float(metrics.completion_fraction),
                        "num_completed": int(md.get("num_completed") or 0),
                        "num_dropped": int(md.get("num_dropped") or 0),
                        "mean_ttft": md.get("mean_ttft"),
                        "p95_ttft": md.get("p95_ttft"),
                        "request_throughput": md.get("request_throughput"),
                        "elapsed_s": time.time() - t0,
                    }
                )
                if args.baseline.startswith("official_vtc"):
                    max_prompt = max(int(r.prompt_tokens) for r in s.requests)
                    step_budget = int(s.service_model_kwargs.get("step_token_budget", 512))
                    row["vtc_batch_token_budget_override"] = max(step_budget, max_prompt)
                ok += 1
            except Exception as e:  # noqa: BLE001
                import traceback

                row.update(
                    {
                        "failure_status": "failed",
                        "anwg": None,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                        "elapsed_s": time.time() - t0,
                    }
                )
                fail += 1

            with jsonl.open("a") as jf:
                jf.write(json.dumps(row, sort_keys=True) + "\n")
            if (idx + 1) % 10 == 0 or idx == 0:
                prog = f"progress {idx+1}/{len(records)} ok={ok} fail={fail}"
                print(f"[stress_pub] {prog}", flush=True)
                lf.write(prog + "\n")
                lf.flush()

        summary = {
            "baseline": args.baseline,
            "workload": "public_trace_stress_v1",
            "M": M,
            "C": C,
            "n_scenarios": len(records),
            "n_success": ok,
            "n_failed": fail,
            "elapsed_s": time.time() - t_all,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "cells_path": str(jsonl),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[stress_pub] DONE {summary}", flush=True)
        lf.write(f"DONE {json.dumps(summary)}\n")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
