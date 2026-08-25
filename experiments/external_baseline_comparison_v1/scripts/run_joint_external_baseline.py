#!/usr/bin/env python3
"""Run one external baseline over frozen joint-240 scenarios.

Does NOT modify canonical P6 utility matrices. Writes a separate result JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import importlib.util  # noqa: E402

from llmserveopt.core.metrics import metrics_to_dict  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402


def load_joint_module():
    path = REPO / "experiments/joint_multimechanism_generalization_v1/run_joint_multimechanism_generalization_v1.py"
    spec = importlib.util.spec_from_file_location("joint_mm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_all_scenarios(joint):
    rng = np.random.default_rng(joint.SEED)
    scenarios = []
    for i in range(joint.N_SCENARIOS):
        scenarios.append(joint.build_scenario(joint.sample_params(rng, i)))
    return scenarios


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
        return (
            VTCFairnessPolicy(known_tenants=tenants, batch_token_budget_override=budget),
            {},
        )
    raise SystemExit(f"unsupported baseline {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        required=True,
        choices=["vllm_style_continuous_batching", "official_vtc_joint_token_budget_remap"],
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="optional smoke limit")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    jsonl = out_dir / "cells.jsonl"
    manifest = REPO / "experiments/joint_multimechanism_generalization_v1/scenario_manifest.csv"
    expected_hash = "9c1319eec8a107ced39277504caa17ffe47e3b3dc3ab866771470dd223363782"
    actual_hash = sha256_file(manifest)
    if actual_hash != expected_hash:
        print(f"FATAL: scenario_manifest hash mismatch {actual_hash} != {expected_hash}", flush=True)
        return 2

    joint = load_joint_module()
    scenarios = build_all_scenarios(joint)
    if args.limit > 0:
        scenarios = scenarios[: args.limit]

    started = datetime.now(timezone.utc).isoformat()
    with log.open("a") as lf:
        lf.write(f"start {started} baseline={args.baseline} n={len(scenarios)}\n")
        lf.flush()
        print(f"[joint_ext] start baseline={args.baseline} n={len(scenarios)}", flush=True)

        # Fresh jsonl
        if jsonl.exists():
            jsonl.unlink()

        ok = 0
        fail = 0
        t_all = time.time()
        for idx, s in enumerate(scenarios):
            t0 = time.time()
            row = {
                "workload_id": "joint_240",
                "scenario_id": s.scenario_id,
                "baseline": args.baseline,
                "seed": int(s.seed),
                "implementation_version": args.baseline,
                "config_hash": expected_hash,
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
                        max_steps=80_000,
                        drain_steps=20_000,
                    )
                )
                sim.load_trace(list(s.requests))
                metrics = sim.run(policy, workload_tag=s.scenario_id, seed=s.seed)
                md = metrics_to_dict(metrics)
                slo_viol = md.get("slo_violation_rate")
                if slo_viol is None:
                    # Some ServiceModel/policy combinations omit this field;
                    # derive a causal proxy from completed-request flags when present.
                    completed = list(getattr(sim, "_completed", []) or [])
                    if completed:
                        slo_viol = sum(1 for c in completed if getattr(c, "slo_violated", False)) / max(len(s.requests), 1)
                    else:
                        slo_viol = 0.0
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
                if args.baseline == "official_vtc_joint_token_budget_remap":
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
                        "slo_attainment_proxy": None,
                        "completion_fraction": None,
                        "num_completed": None,
                        "num_dropped": None,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                        "elapsed_s": time.time() - t0,
                    }
                )
                fail += 1

            with jsonl.open("a") as jf:
                jf.write(json.dumps(row, sort_keys=True) + "\n")

            if (idx + 1) % 20 == 0 or idx == 0:
                msg = f"progress {idx+1}/{len(scenarios)} ok={ok} fail={fail} last_status={row['failure_status']}"
                print(f"[joint_ext] {msg}", flush=True)
                lf.write(msg + "\n")
                lf.flush()

        summary = {
            "baseline": args.baseline,
            "n_scenarios": len(scenarios),
            "n_success": ok,
            "n_failed": fail,
            "elapsed_s": time.time() - t_all,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": actual_hash,
            "cells_path": str(jsonl),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[joint_ext] DONE {summary}", flush=True)
        lf.write(f"DONE {json.dumps(summary)}\n")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    # Allow importing baselines.* from repo root
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
