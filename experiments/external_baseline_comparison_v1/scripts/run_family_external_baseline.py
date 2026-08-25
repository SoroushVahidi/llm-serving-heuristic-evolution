#!/usr/bin/env python3
"""Run one external baseline on a frozen MF-PSD Family A/B/C scenario set.

Uses mf_psd_scenarios_v1.csv + the same case_* rebuilders as the unified
matrix. Does NOT modify unified_utility_matrix_v2 or P6 artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from llmserveopt.core.metrics import metrics_to_dict  # noqa: E402
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (  # noqa: E402
    case_fairness_vs_size_v2,
)
from llmserveopt.policy_separation.templates_kv_pressure_v2 import (  # noqa: E402
    case_kv_pressure_reserve_contention_v2,
)
from llmserveopt.policy_separation.templates_prefill_decode_v2 import (  # noqa: E402
    case_prefill_decode_ttft_contention,
)
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

FAMILY_A = "FAMILY_A_FAIRNESS_STARVATION_V2"
FAMILY_B = "FAMILY_B_PREFILL_DECODE_V2"
FAMILY_C = "FAMILY_C_KV_PRESSURE_V2"
SCENARIO_CSV = REPO / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"
DATASETS = REPO / "datasets"


def rebuild(row: pd.Series):
    fam = row["mechanism_family"]
    if fam == FAMILY_A:
        return case_fairness_vs_size_v2(
            target_utilization=float(row["feat_A__target_utilization"]),
            tenant_weight_skew=float(row["feat_A__tenant_weight_skew"]),
            favored_tenant_size=str(row["feat_A__favored_tenant_size"]),
            prediction_noise_sigma=float(row["feat_A__prediction_noise_sigma"]),
            seed=int(row["seed"]),
            datasets_root=DATASETS,
        )
    if fam == FAMILY_B:
        return case_prefill_decode_ttft_contention(
            hog_count=str(row["feat_B__hog_count"]),
            late_pressure=str(row["feat_B__late_pressure"]),
            slo_emphasis=str(row["feat_B__slo_emphasis"]),
            seed=int(row["seed"]),
            datasets_root=DATASETS,
        )
    if fam == FAMILY_C:
        return case_kv_pressure_reserve_contention_v2(
            bulk_pressure=str(row["feat_C__bulk_pressure"]),
            urgent_arrival_phase=str(row["feat_C__urgent_arrival_phase"]),
            urgent_tightness=str(row["feat_C__urgent_tightness"]),
            seed=int(row["seed"]),
            datasets_root=DATASETS,
        )
    raise ValueError(fam)


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
        "--family",
        required=True,
        choices=[FAMILY_A, FAMILY_B, FAMILY_C, "A", "B", "C"],
    )
    ap.add_argument(
        "--baseline",
        required=True,
        choices=["official_vtc_joint_token_budget_remap", "vllm_style_continuous_batching"],
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    fam_map = {"A": FAMILY_A, "B": FAMILY_B, "C": FAMILY_C}
    family = fam_map.get(args.family, args.family)

    scen = pd.read_csv(SCENARIO_CSV)
    frame = scen[scen["mechanism_family"] == family].sort_values("canonical_scenario_id")
    if args.limit > 0:
        frame = frame.head(args.limit)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "cells.jsonl"
    log = out_dir / "run.log"
    if jsonl.exists():
        jsonl.unlink()

    started = datetime.now(timezone.utc).isoformat()
    ok = fail = 0
    t_all = time.time()
    with log.open("a") as lf:
        msg = f"start {started} family={family} baseline={args.baseline} n={len(frame)}"
        print(f"[family_ext] {msg}", flush=True)
        lf.write(msg + "\n")
        lf.flush()

        for idx, (_, row) in enumerate(frame.iterrows()):
            t0 = time.time()
            cid = row["canonical_scenario_id"]
            cell = {
                "workload_id": f"family_{family}",
                "mechanism_family": family,
                "canonical_scenario_id": cid,
                "baseline": args.baseline,
                "seed": int(row["seed"]),
                "implementation_version": args.baseline,
                "config_hash": "mf_psd_scenarios_v1",
                "raw_result_pointer": str(jsonl),
            }
            try:
                s = rebuild(row)
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
                    )
                cell.update(
                    {
                        "scenario_id": s.scenario_id,
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
                    cell["vtc_batch_token_budget_override"] = max(step_budget, max_prompt)
                    # Official VTC service counters if present on policy
                    for attr in ("service_counters", "tenant_service", "fairness_metric"):
                        if hasattr(policy, attr):
                            try:
                                cell[f"vtc_{attr}"] = getattr(policy, attr)
                            except Exception:  # noqa: BLE001
                                pass
                ok += 1
            except Exception as e:  # noqa: BLE001
                import traceback

                cell.update(
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
                jf.write(json.dumps(cell, sort_keys=True, default=str) + "\n")
            if (idx + 1) % 10 == 0 or idx == 0:
                prog = f"progress {idx+1}/{len(frame)} ok={ok} fail={fail}"
                print(f"[family_ext] {prog}", flush=True)
                lf.write(prog + "\n")
                lf.flush()

        summary = {
            "family": family,
            "baseline": args.baseline,
            "n_scenarios": int(len(frame)),
            "n_success": ok,
            "n_failed": fail,
            "elapsed_s": time.time() - t_all,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "cells_path": str(jsonl),
            "scenario_csv": str(SCENARIO_CSV),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[family_ext] DONE {summary}", flush=True)
        lf.write(f"DONE {json.dumps(summary)}\n")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
