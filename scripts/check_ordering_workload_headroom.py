#!/usr/bin/env python3
"""Design-only workload-headroom checker (2026-08-04 parallel audit -- see
docs/audits/ordering_workload_headroom_audit_20260804.md).

Purpose: given a candidate workload (a synthetic WorkloadConfig-shaped YAML,
or the existing real WildChat evaluation workload as a control), determine
CHEAPLY -- before committing to an expensive multi-policy comparison sweep
-- whether that workload gives request-ORDERING policies any opportunity
to differ at all. Runs only FIFO / estimated_service_time_first /
shortest_output_first / oracle_srtf (no learned selector, no vLLM-LTR
checkpoint inference, no GPU). New, isolated script: does not import, call,
or modify anything in scripts/run_vllm_ltr_first_comparative_evaluation.py,
src/llmserveopt/selector/, or baselines/vllm_ltr/.

Metrics computed (see the audit doc for full justification):
  - fifo_srtf_anwg_gap: oracle_srtf ANWG - fifo ANWG
  - fifo_srtf_completion_gap / fifo_srtf_slo_violation_gap
  - queue_contention_fraction: fraction of (nonempty-queue) decision steps
    with >=2 queued requests, from an instrumented FIFO run
  - fifo_srtf_decision_disagreement_fraction: fraction of (nonempty-queue)
    decision steps where a same-snapshot SRTF-ordered admission pass
    (computed locally, diagnostic-only, using each request's real
    actual_output_tokens -- never exposed to any deployable policy) would
    admit a different SET of request_ids than FIFO actually did
  - service_time_cv: coefficient of variation of actual_output_tokens
    (diagnostic-only)
  - deadline_slack_cv: coefficient of variation of (slo_deadline - arrival_time)
  - prompt_predicted_output_correlation: Pearson r between prompt_tokens
    and predicted_output_tokens

Smoke gate (PASS requires ALL of):
  - fifo_srtf_decision_disagreement_fraction > 0  (SRTF and FIFO must
    actually choose differently at least sometimes)
  - fifo_srtf_anwg_gap >= 0.01  (oracle must beat fifo by a non-trivial
    margin -- 1 percentage point of ANWG, an order of magnitude above the
    ~0 gap the WildChat control workload showed)
  - queue_contention_fraction >= 0.05  (at least 5% of decisions must see
    a real multi-request choice, not near-zero)
  - NOT all four policies (fifo/est/sof/oracle_srtf) produce bit-identical
    ANWG (the exact degenerate pattern found in the WildChat control run)

Usage:
  python scripts/check_ordering_workload_headroom.py --preset staggered_heterogeneous --seed 0
  python scripts/check_ordering_workload_headroom.py --config configs/workload_headroom_candidates/burst_independent_lengths.yaml --seed 0
  python scripts/check_ordering_workload_headroom.py --wildchat-control --seed 0
  python scripts/check_ordering_workload_headroom.py --config ... --dry-run
  python scripts/check_ordering_workload_headroom.py --config ... --json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from llmserveopt.core.metrics import RunMetrics, metrics_to_dict
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.oracle import build_oracle
from llmserveopt.policies.registry import make_policy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from llmserveopt.workloads.synthetic import SLOClass, WorkloadConfig, generate_workload

CHEAP_POLICIES = ["fifo", "estimated_service_time_first", "shortest_output_first"]

DEFAULT_GPU = dict(max_active_sequences=8, max_batch_tokens=8192, max_kv_tokens=131072)
DEFAULT_SERVICE_MODEL = dict(
    step_size=0.001,
    enable_prefill_modeling=True,
    prefill_cost_per_token=1.0,
    max_prefill_chunk_tokens=512,
    step_token_budget=8192,
    decode_first=False,
)

GATE_MIN_DISAGREEMENT_FRACTION = 0.0  # strictly > 0, checked separately
GATE_MIN_ANWG_GAP = 0.01
GATE_MIN_QUEUE_CONTENTION_FRACTION = 0.05


class _InstrumentedFIFO(BasePolicy):
    """Wraps the real, unmodified FIFOPolicy. At each decision step with a
    nonempty queue, records (a) the queue size and (b) whether a
    same-snapshot SRTF-ordered admission pass (diagnostic-only, using real
    actual_output_tokens via a request_id->length map built from the
    Request list this script itself generated -- never exposed through
    ObservableState/ObservableRequest to any policy) would have admitted a
    different SET of request_ids than FIFO's real decision.

    Mirrors FIFOPolicy's own round-robin admission loop and
    BasePolicy._feasible_on_gpu exactly, applied to a SRTF-sorted request
    order over a deep-copied (pre-mutation) GPU-state snapshot, so the
    comparison reflects real feasibility constraints (KV/active-sequence
    capacity), not just naive reordering."""

    name = "fifo_instrumented"

    def __init__(self, actual_output_by_id: Dict[int, int]):
        self._inner = make_policy("fifo")
        self._actual_output_by_id = actual_output_by_id
        self.decisions: List[dict] = []

    def reset(self) -> None:
        self._inner.reset()
        self.decisions = []

    def select_action(self, state):
        queue = list(state.waiting_queue)
        queue_ids = [r.request_id for r in queue]

        srtf_admit: List[int] = []
        if queue_ids:
            srtf_order = sorted(
                queue, key=lambda r: (self._actual_output_by_id[r.request_id], r.request_id)
            )
            gpu_copies = copy.deepcopy(state.gpu_states)
            gpu_idx = 0
            n_gpus = len(gpu_copies)
            for req in srtf_order:
                for offset in range(n_gpus):
                    gpu = gpu_copies[(gpu_idx + offset) % n_gpus]
                    if BasePolicy._feasible_on_gpu(gpu, req):
                        srtf_admit.append(req.request_id)
                        gpu.active_request_ids.append(req.request_id)
                        gpu.current_kv_tokens += req.prompt_tokens
                        gpu_idx = (gpu_idx + offset + 1) % n_gpus
                        break

        action = self._inner.select_action(state)  # real FIFO decision; mutates state in place

        if queue_ids:
            fifo_admit = sorted(action.all_admitted_ids())
            self.decisions.append({
                "queue_size": len(queue_ids),
                "fifo_admit": fifo_admit,
                "srtf_admit": sorted(srtf_admit),
                "disagree": set(fifo_admit) != set(srtf_admit),
            })
        return action


def _build_sim_config(gpu_overrides: dict, service_model_overrides: dict, drain_steps: int) -> SimulatorConfig:
    gpu_cfg = dict(DEFAULT_GPU)
    gpu_cfg.update(gpu_overrides or {})
    sm_cfg = dict(DEFAULT_SERVICE_MODEL)
    sm_cfg.update(service_model_overrides or {})
    return SimulatorConfig(
        gpu_configs=[GPUConfig(gpu_id=0, **gpu_cfg)],
        service_model=ServiceModel(**sm_cfg),
        drain_steps=drain_steps,
    )


def _run_policy(policy: BasePolicy, requests: List[Request], sim_cfg: SimulatorConfig, seed: int) -> RunMetrics:
    sim = Simulator(sim_cfg)
    sim.load_trace(list(requests))
    policy.reset()
    return sim.run(policy, workload_tag="headroom_check", seed=seed)


def _cv(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or np.mean(arr) == 0:
        return float("nan")
    return float(np.std(arr) / np.mean(arr))


def compute_headroom_metrics(
    requests: List[Request],
    gpu_overrides: dict,
    service_model_overrides: dict,
    drain_steps: int,
    seed: int,
) -> dict:
    sim_cfg = _build_sim_config(gpu_overrides, service_model_overrides, drain_steps)

    # Static distributional metrics (no simulation required).
    prompt_tokens = [r.prompt_tokens for r in requests]
    predicted_output = [r.predicted_output_tokens for r in requests]
    actual_output = [r.actual_output_tokens for r in requests]
    slack = [r.slo_deadline - r.arrival_time for r in requests]
    priorities = [r.priority for r in requests]

    corr = float("nan")
    if len(requests) >= 2 and np.std(prompt_tokens) > 0 and np.std(predicted_output) > 0:
        corr = float(np.corrcoef(prompt_tokens, predicted_output)[0, 1])

    # Dynamic metrics: instrumented FIFO (also gives real fifo ANWG), plus
    # EST / SOF / oracle_srtf uninstrumented for their own ANWG.
    actual_output_by_id = {r.request_id: r.actual_output_tokens for r in requests}
    fifo_instrumented = _InstrumentedFIFO(actual_output_by_id)
    fifo_metrics = _run_policy(fifo_instrumented, requests, sim_cfg, seed)

    est_metrics = _run_policy(make_policy("estimated_service_time_first"), requests, sim_cfg, seed)
    sof_metrics = _run_policy(make_policy("shortest_output_first"), requests, sim_cfg, seed)
    oracle_metrics = _run_policy(build_oracle(requests), requests, sim_cfg, seed)

    decisions = fifo_instrumented.decisions
    n_decisions = len(decisions)
    n_contended = sum(1 for d in decisions if d["queue_size"] >= 2)
    n_disagree = sum(1 for d in decisions if d["disagree"])

    anwg = {
        "fifo": fifo_metrics.arrival_normalized_weighted_goodput,
        "estimated_service_time_first": est_metrics.arrival_normalized_weighted_goodput,
        "shortest_output_first": sof_metrics.arrival_normalized_weighted_goodput,
        "oracle_srtf": oracle_metrics.arrival_normalized_weighted_goodput,
    }
    all_tied = len({round(v, 6) for v in anwg.values()}) == 1

    return {
        "n_requests": len(requests),
        "n_decision_steps_with_nonempty_queue": n_decisions,
        "queue_contention_fraction": (n_contended / n_decisions) if n_decisions else float("nan"),
        "fifo_srtf_decision_disagreement_fraction": (n_disagree / n_decisions) if n_decisions else float("nan"),
        "service_time_cv": _cv(actual_output),
        "deadline_slack_cv": _cv(slack),
        "prompt_predicted_output_correlation": corr,
        "priority_distribution": {
            str(p): priorities.count(p) for p in sorted(set(priorities))
        },
        "anwg": anwg,
        "fifo_srtf_anwg_gap": anwg["oracle_srtf"] - anwg["fifo"],
        "completion_fraction": {
            "fifo": fifo_metrics.completion_fraction,
            "oracle_srtf": oracle_metrics.completion_fraction,
        },
        "fifo_srtf_completion_gap": oracle_metrics.completion_fraction - fifo_metrics.completion_fraction,
        "slo_violation_rate": {
            "fifo": fifo_metrics.slo_violation_rate,
            "oracle_srtf": oracle_metrics.slo_violation_rate,
        },
        "fifo_srtf_slo_violation_gap": fifo_metrics.slo_violation_rate - oracle_metrics.slo_violation_rate,
        "all_four_policies_bit_identical_anwg": all_tied,
    }


def evaluate_gate(metrics: dict) -> dict:
    disagreement = metrics["fifo_srtf_decision_disagreement_fraction"]
    anwg_gap = metrics["fifo_srtf_anwg_gap"]
    contention = metrics["queue_contention_fraction"]
    all_tied = metrics["all_four_policies_bit_identical_anwg"]

    checks = {
        "disagreement_fraction_nonzero": (not np.isnan(disagreement)) and disagreement > GATE_MIN_DISAGREEMENT_FRACTION,
        "anwg_gap_meaningful": (not np.isnan(anwg_gap)) and anwg_gap >= GATE_MIN_ANWG_GAP,
        "queue_contention_sufficient": (not np.isnan(contention)) and contention >= GATE_MIN_QUEUE_CONTENTION_FRACTION,
        "not_degenerate_tie": not all_tied,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _workload_config_from_dict(d: dict) -> WorkloadConfig:
    d = dict(d)
    slo_classes = d.pop("slo_classes", None)
    if slo_classes is not None:
        d["slo_classes"] = [SLOClass(**c) for c in slo_classes]
    return WorkloadConfig(**d)


def _load_wildchat_control_requests(seed: int, max_requests: Optional[int]) -> List[Request]:
    """Read-only reuse of the real WildChat evaluation workload, for use as
    the negative/control case -- does NOT import or touch
    scripts/run_vllm_ltr_first_comparative_evaluation.py (that file is
    being actively edited by a separate, currently-running task) or its
    live results directory. Reimplements the same
    convert_sharegpt_to_requests call directly against the same
    already-ingested, committed data file."""
    from llmserveopt.workloads.augmentation import AugmentationConfig
    from llmserveopt.workloads.sharegpt import (
        ShareGPTConversionConfig,
        convert_sharegpt_to_requests,
        load_sharegpt_raw,
    )

    pairs_path = "data/processed/wildchat/wildchat_eval_sharegpt_shaped.json"
    records = load_sharegpt_raw(pairs_path)
    config = ShareGPTConversionConfig(
        arrival_mode="poisson", arrival_rate=10.0, tokenizer_name="facebook/opt-125m",
        fallback_whitespace=False,
    )
    requests, _report = convert_sharegpt_to_requests(
        records, config=config, seed=seed, aug_config=AugmentationConfig()
    )
    if max_requests is not None:
        requests = sorted(requests, key=lambda r: r.request_id)[:max_requests]
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to a workload_headroom_candidates/*.yaml file")
    parser.add_argument("--preset", help="Name of a preset in llmserveopt.workloads.synthetic (e.g. make_bursty_trace)")
    parser.add_argument("--wildchat-control", action="store_true",
                         help="Use the real (committed) WildChat evaluation workload as the negative/control case.")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drain-steps", type=int, default=200_000)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config and exit without simulating.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text summary.")
    parser.add_argument("--output", help="Optional path to also write the JSON report to.")
    args = parser.parse_args()

    gpu_overrides: dict = {}
    service_model_overrides: dict = {}
    drain_steps = args.drain_steps
    name = "unnamed"

    if args.wildchat_control:
        name = "wildchat_control"
        requests = None if args.dry_run else _load_wildchat_control_requests(args.seed, args.max_requests)
        resolved_config = {"source": "wildchat (real data/processed/wildchat/wildchat_eval_sharegpt_shaped.json)"}
    elif args.config:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        name = raw.get("name", os.path.splitext(os.path.basename(args.config))[0])
        gpu_overrides = raw.get("gpu", {})
        service_model_overrides = raw.get("service_model", {})
        wl_cfg = _workload_config_from_dict(raw["workload"])
        resolved_config = raw
        requests = None if args.dry_run else generate_workload(wl_cfg, seed=args.seed)
    elif args.preset:
        import llmserveopt.workloads.synthetic as synth
        fn = getattr(synth, args.preset)
        name = args.preset
        resolved_config = {"source": f"llmserveopt.workloads.synthetic.{args.preset}"}
        requests = None if args.dry_run else fn(seed=args.seed)
    else:
        parser.error("one of --config, --preset, --wildchat-control is required")
        return

    if args.dry_run:
        print(json.dumps({"name": name, "resolved_config": resolved_config, "dry_run": True}, indent=2, default=str))
        return

    if args.max_requests is not None and not args.wildchat_control:
        requests = sorted(requests, key=lambda r: r.request_id)[: args.max_requests]

    t0 = time.perf_counter()
    metrics = compute_headroom_metrics(requests, gpu_overrides, service_model_overrides, drain_steps, args.seed)
    runtime_s = time.perf_counter() - t0
    gate = evaluate_gate(metrics)

    report = {
        "name": name,
        "seed": args.seed,
        "runtime_s": runtime_s,
        "metrics": metrics,
        "gate": gate,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"=== ordering headroom check: {name} (seed={args.seed}) ===")
        print(f"n_requests={metrics['n_requests']}  runtime={runtime_s:.2f}s")
        print(f"ANWG: fifo={metrics['anwg']['fifo']:.4f}  est={metrics['anwg']['estimated_service_time_first']:.4f}  "
              f"sof={metrics['anwg']['shortest_output_first']:.4f}  oracle_srtf={metrics['anwg']['oracle_srtf']:.4f}")
        print(f"fifo_srtf_anwg_gap={metrics['fifo_srtf_anwg_gap']:.4f}  "
              f"fifo_srtf_completion_gap={metrics['fifo_srtf_completion_gap']:.4f}  "
              f"fifo_srtf_slo_violation_gap={metrics['fifo_srtf_slo_violation_gap']:.4f}")
        print(f"queue_contention_fraction={metrics['queue_contention_fraction']:.4f}  "
              f"fifo_srtf_decision_disagreement_fraction={metrics['fifo_srtf_decision_disagreement_fraction']:.4f}")
        print(f"service_time_cv={metrics['service_time_cv']:.3f}  deadline_slack_cv={metrics['deadline_slack_cv']:.3f}  "
              f"prompt_predicted_output_correlation={metrics['prompt_predicted_output_correlation']:.3f}")
        print(f"all_four_policies_bit_identical_anwg={metrics['all_four_policies_bit_identical_anwg']}")
        print(f"gate checks: {gate['checks']}")
        print(f"GATE RESULT: {'PASS' if gate['passed'] else 'FAIL'}")

    sys.exit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
