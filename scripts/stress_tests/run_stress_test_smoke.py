#!/usr/bin/env python3
"""Smoke runner for the Algorithm Stress-Test Library catalog.

Loads configs/stress_tests/algorithm_stress_test_catalog.yaml, generates
each entry's workload (via scripts/stress_tests/generators.py), runs the
algorithm under test plus its `comparison_algorithms` through the
simulator, computes the entry's `validation_metrics`, and evaluates its
`acceptance_gates` expression.

Executable coverage this pass (18 of 22 entries have a generator; of
those, 16 are executed here):
  - fifo, shortest_output_first, estimated_service_time_first,
    weighted_shortest_processing, edf, least_laxity_first, aging_priority,
    scorpio_style_slo_guard: fully executed (8 algorithms, 16 entries).
  - regression_anwg (2 entries): generator exists, but `PerPolicyRegressionAnwgSelector`
    requires a persisted, trained model artifact this task does not load
    or retrain -- workload generation is validated structurally
    (test_stress_test_generators.py), execution is explicitly OUT OF
    SCOPE this pass, not silently skipped.
  - vllm_ltr / pars (4 entries): generators raise NotImplementedError by
    design (offline-scoring infrastructure constraint, disclosed in the
    catalog's own `requires_new_scoring_pass`/`real_system_followup_required`
    fields) -- also out of scope this pass.

Usage: python scripts/stress_tests/run_stress_test_smoke.py [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generators  # noqa: E402
from llmserveopt.core.types import GPUConfig  # noqa: E402
from llmserveopt.policies.aging_priority import AgingPriorityPolicy  # noqa: E402
from llmserveopt.policies.edf import EDFPolicy  # noqa: E402
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy  # noqa: E402
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy  # noqa: E402
from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy  # noqa: E402
from llmserveopt.policies.scorpio_style_slo_guard import ScorpioStyleSloGuardPolicy  # noqa: E402
from llmserveopt.policies.shortest_output_first import ShortestOutputFirstPolicy  # noqa: E402
from llmserveopt.policies.vllm_chunked_prefill_faithful import VLLMChunkedPrefillFaithfulPolicy  # noqa: E402
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy  # noqa: E402
from llmserveopt.policies.weighted_shortest_processing import WeightedShortestProcessingPolicy  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

POLICY_FACTORIES = {
    "fifo": lambda: FIFOPolicy(),
    "shortest_output_first": lambda: ShortestOutputFirstPolicy(),
    "estimated_service_time_first": lambda: EstimatedServiceTimeFirstPolicy(),
    "weighted_shortest_processing": lambda: WeightedShortestProcessingPolicy(),
    "edf": lambda: EDFPolicy(),
    "least_laxity_first": lambda: LeastLaxityFirstPolicy(),
    "aging_priority": lambda: AgingPriorityPolicy(),
    "scorpio_style_slo_guard": lambda: ScorpioStyleSloGuardPolicy(),
    # Sarathi-targeted stress-test entries (see catalog section 12) need
    # Phase-1.5 prefill/decode-contention modeling to make chunking
    # behavior observable at all -- see run_policy()'s `simulator_requirements`
    # handling below and docs/decode_prefill_contention_execution_model.md.
    # Chunk/block/seq-count defaults match the real Wulver validation
    # config exactly (gpu-memory-utilization=0.35 equivalent, chunk-size=512,
    # block-size=16, max-num-seqs=16 -- docs/wulver_sarathi_vllm_repeated_validation.md).
    "sarathi_faithful": lambda: SarathiFaithfulPolicy(chunk_size=512, block_size=16, max_num_seqs=16, watermark=0.0),
    "vllm_chunked_prefill_faithful": lambda: VLLMChunkedPrefillFaithfulPolicy(
        block_size=16, max_num_batched_tokens=512, max_num_seqs=16, watermark=0.0
    ),
    "vllm_faithful": lambda: VLLMFaithfulPolicy(block_size=16, max_num_batched_tokens=512, max_num_seqs=16, watermark=0.0),
}

# Policies that require Phase-1.5 prefill/decode-contention modeling to
# exhibit their intended scheduling behavior at all (see
# docs/decode_prefill_contention_execution_model.md). `decode_first` is
# set per-policy: sarathi_faithful exercises the decode-protected path,
# vllm_chunked_prefill_faithful exercises the shared-FCFS-contention path
# -- matching microsoft/sarathi-serve's own stall-free guarantee vs.
# vLLM's chunked-prefill scheduler having no decode priority, which is the
# exact mechanism the real Wulver A/B comparison measured.
_CONTENTION_DECODE_FIRST = {
    "sarathi_faithful": True,
    "vllm_chunked_prefill_faithful": False,
    # vllm_faithful has no chunked-admission model of its own to exercise
    # the contention path meaningfully -- given decode_first=True for a
    # fair side-by-side, per docs/decode_prefill_contention_execution_model.md's
    # own stated convention ("Which policies opt in").
    "vllm_faithful": True,
}

EXECUTABLE_ALGORITHM_IDS = set(POLICY_FACTORIES)

DEFAULT_GPU = GPUConfig(gpu_id=0, max_active_sequences=4, max_batch_tokens=4096, max_kv_tokens=16384)


def _build_service_model(policy_name: str, sim_req: dict) -> ServiceModel:
    """Default: ServiceModel() -- bit-identical to every pre-existing
    catalog entry's behavior (none of them set `enable_prefill_modeling`
    in `simulator_requirements`, so this is fully backward-compatible).
    Sarathi-targeted entries opt into Phase-1.5 prefill/decode-contention
    modeling via `simulator_requirements: {enable_prefill_modeling: true, ...}`
    -- see docs/decode_prefill_contention_execution_model.md."""
    if not sim_req.get("enable_prefill_modeling"):
        return ServiceModel()
    kwargs = dict(
        enable_prefill_modeling=True,
        max_prefill_chunk_tokens=int(sim_req.get("max_prefill_chunk_tokens", 512)),
        step_token_budget=int(sim_req.get("step_token_budget", 4096)),
    )
    if sim_req.get("enable_decode_prefill_contention"):
        kwargs["enable_decode_prefill_contention"] = True
        kwargs["decode_first"] = _CONTENTION_DECODE_FIRST.get(policy_name, sim_req.get("decode_first", False))
    else:
        kwargs["decode_first"] = bool(sim_req.get("decode_first", True))
    return ServiceModel(**kwargs)


def run_policy(requests: List, policy_name: str, gpu_overrides: dict, sim_req: dict | None = None) -> dict:
    sim_req = sim_req or {}
    gpu_kwargs = dict(gpu_id=0, max_active_sequences=DEFAULT_GPU.max_active_sequences,
                       max_batch_tokens=DEFAULT_GPU.max_batch_tokens, max_kv_tokens=DEFAULT_GPU.max_kv_tokens)
    if "max_active_sequences" in gpu_overrides:
        gpu_kwargs["max_active_sequences"] = gpu_overrides["max_active_sequences"]
    if "max_batch_tokens" in gpu_overrides:
        gpu_kwargs["max_batch_tokens"] = gpu_overrides["max_batch_tokens"]
    if "max_kv_tokens" in gpu_overrides:
        gpu_kwargs["max_kv_tokens"] = gpu_overrides["max_kv_tokens"]
    gpu = GPUConfig(**gpu_kwargs)
    service_model = _build_service_model(policy_name, sim_req)
    cfg = SimulatorConfig(gpu_configs=[gpu], service_model=service_model)
    sim = Simulator(cfg)
    sim.load_trace(requests)
    policy = POLICY_FACTORIES[policy_name]()
    metrics = sim.run(policy, workload_tag=policy_name)

    ttfts = [cr.ttft for cr in sim._completed if cr.ttft == cr.ttft]  # noqa: SLF001 -- filter NaN
    tpots = [cr.tpot for cr in sim._completed if cr.tpot == cr.tpot]  # noqa: SLF001
    duration = max((cr.completion_time for cr in sim._completed), default=0.0)  # noqa: SLF001

    by_class = defaultdict(list)
    for cr in sim._completed:  # noqa: SLF001
        by_class[cr.request.class_id].append(cr)

    max_qd_by_class = {c: max((cr.queuing_delay for cr in crs), default=float("nan")) for c, crs in by_class.items()}
    mean_qd_by_class = {
        c: (sum(cr.queuing_delay for cr in crs) / len(crs) if crs else float("nan"))
        for c, crs in by_class.items()
    }
    p95_qd_by_class = {}
    for c, crs in by_class.items():
        qds = sorted(cr.queuing_delay for cr in crs)
        p95_qd_by_class[c] = qds[int(0.95 * (len(qds) - 1))] if qds else float("nan")

    starvation_by_class = {}
    for c, crs in by_class.items():
        times = sorted(cr.completion_time for cr in crs)
        if not times:
            starvation_by_class[c] = float("nan")
            continue
        gaps = [times[0]] + [b - a for a, b in zip(times, times[1:])]
        starvation_by_class[c] = max(gaps)

    weighted_completion_time_proxy = sum(
        cr.request.priority * cr.completion_time for cr in sim._completed
    )

    def _p95(vals: list) -> float:
        s = sorted(vals)
        return s[int(0.95 * (len(s) - 1))] if s else float("nan")

    mean_ttft = statistics.mean(ttfts) if ttfts else float("nan")
    mean_tpot = statistics.mean(tpots) if tpots else float("nan")
    p95_tpot = _p95(tpots)
    # Stall-frequency proxy: NOT a direct instrumentation of scheduler-
    # level stalls (this simulator does not expose a per-step stall
    # counter) -- ratio of tail-to-mean TPOT, where episodic decode
    # interruptions (e.g. an arriving prefill delaying an active decode's
    # next token under enable_decode_prefill_contention=True) inflate the
    # tail relative to the mean. Disclosed as a proxy, not a ground-truth
    # stall count.
    stall_frequency_proxy = (p95_tpot / mean_tpot) if (mean_tpot == mean_tpot and mean_tpot > 0) else float("nan")

    # Scheduling-disagreement proxy: fraction of completed-request pairs
    # whose COMPLETION order disagrees with their ARRIVAL order (a simple
    # inversion-count / total-pairs ratio, not a formal rank-correlation
    # statistic) -- 0.0 means this policy completed requests in exactly
    # arrival order (FIFO-like); higher values mean it reordered more.
    completed_sorted_by_arrival = sorted(sim._completed, key=lambda cr: cr.request.arrival_time)  # noqa: SLF001
    completion_order = [cr.request.request_id for cr in
                         sorted(completed_sorted_by_arrival, key=lambda cr: cr.completion_time)]
    arrival_order = [cr.request.request_id for cr in completed_sorted_by_arrival]
    rank = {rid: i for i, rid in enumerate(arrival_order)}
    n_pairs = 0
    n_inversions = 0
    for i in range(len(completion_order)):
        for j in range(i + 1, len(completion_order)):
            n_pairs += 1
            if rank[completion_order[i]] > rank[completion_order[j]]:
                n_inversions += 1
    scheduling_disagreement_proxy = (n_inversions / n_pairs) if n_pairs else float("nan")

    return {
        "mean_latency": metrics.mean_latency,
        "p95_latency": metrics.p95_latency,
        "mean_queuing_delay": metrics.mean_queuing_delay,
        "p95_queuing_delay": metrics.p95_queuing_delay,
        "max_queuing_delay": max((cr.queuing_delay for cr in sim._completed), default=float("nan")),
        "max_queuing_delay_by_priority_class": max_qd_by_class,
        "mean_queuing_delay_by_priority_class": mean_qd_by_class,
        "p95_queuing_delay_by_priority_class": p95_qd_by_class,
        "starvation_interval": starvation_by_class,
        "completion_fraction": metrics.completion_fraction,
        "num_dropped": metrics.num_dropped,
        "slo_violation_rate": metrics.slo_violation_rate,
        "anwg": metrics.arrival_normalized_weighted_goodput,
        "weighted_completion_time_proxy": weighted_completion_time_proxy,
        "mean_ttft": mean_ttft,
        "p95_ttft": _p95(ttfts),
        "mean_tpot": mean_tpot,
        "p95_tpot": p95_tpot,
        "throughput_completions_per_sec": (len(sim._completed) / duration) if duration > 0 else float("nan"),  # noqa: SLF001
        "stall_frequency_proxy": stall_frequency_proxy,
        "scheduling_disagreement_proxy": scheduling_disagreement_proxy,
        "chunk_utilization": None,  # NOT_INSTRUMENTED -- see completion audit doc §7
    }


def evaluate_gate(gate_expr: str, results_by_policy: Dict[str, dict], entry_id: str, all_results: Dict[str, dict]) -> tuple:
    """Evaluate a catalog `acceptance_gates` expression like
    'fifo.p95_queuing_delay(short) > 5 * shortest_output_first.p95_queuing_delay(short)'
    against the collected results. Returns (evaluable: bool, passed: bool|None, detail: str).
    Only a constrained subset of expressions (direct scalar metric
    comparisons, optionally with a class-name in parens) is auto-evaluated;
    anything referencing another catalog entry by id or a metric this
    runner does not compute is reported as not-auto-evaluable rather than
    guessed at."""
    import re

    # Pattern: policy.metric or policy.metric(class)
    pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)(?:\(([a-zA-Z0-9_]+)\))?")

    def substitute(match):
        policy, metric, cls = match.group(1), match.group(2), match.group(3)
        if policy not in results_by_policy:
            # Might be a reference to a DIFFERENT catalog entry's result
            # (e.g. "selector_target_in_distribution_regime.selector_regret...")
            other = all_results.get(policy)
            if other is None:
                raise KeyError(f"unknown policy/entry reference: {policy}")
            value = other.get(metric)
        else:
            value = results_by_policy[policy].get(metric)
        if isinstance(value, dict):
            if cls is None:
                raise ValueError(f"{policy}.{metric} is per-class but no class given")
            value = value.get(cls)
        elif cls is not None:
            # A class qualifier was given (e.g. "...(low)") but the named
            # metric resolved to a scalar, not a per-class dict -- almost
            # always a typo'd metric name (e.g. "mean_queuing_delay" instead
            # of "mean_queuing_delay_by_priority_class"). Silently ignoring
            # the qualifier would make the gate compare the SAME scalar to
            # itself under two different-looking expressions -- exactly the
            # bug this check exists to catch.
            raise ValueError(
                f"{policy}.{metric}({cls}) was given a class qualifier but "
                f"{metric!r} is a scalar metric, not per-class -- check for a typo'd metric name"
            )
        if value is None or (isinstance(value, float) and value != value):
            raise ValueError(f"{policy}.{metric}" + (f"({cls})" if cls else "") + " is unavailable/NaN")
        return repr(float(value))

    try:
        substituted = pattern.sub(substitute, gate_expr)
        # Constrained namespace: no builtins except `abs`, explicitly
        # allowed since several catalog gates use it (e.g. relative-
        # difference checks) -- everything else stays blocked.
        safe_namespace = {"__builtins__": {}, "abs": abs}
        passed = bool(eval(substituted, safe_namespace, {}))  # noqa: S307 -- constrained namespace, own-authored expressions
        return True, passed, substituted
    except Exception as e:  # noqa: BLE001
        return False, None, f"not auto-evaluable: {e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--full", action="store_true", default=False,
                         help="Use default_full_setting scale instead of default_smoke_setting.")
    args = parser.parse_args()

    catalog_path = _ROOT / "configs" / "stress_tests" / "algorithm_stress_test_catalog.yaml"
    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)

    all_entry_results: Dict[str, dict] = {}
    report_rows = []

    for entry in catalog["stress_tests"]:
        eid = entry["stress_test_id"]
        algo = entry["algorithm_id"]
        gen_fn = generators.GENERATORS.get(eid)

        if algo == "regression_anwg":
            report_rows.append({"id": eid, "status": "OUT_OF_SCOPE",
                                 "reason": "requires persisted trained PerPolicyRegressionAnwgSelector artifact"})
            continue
        if gen_fn is None:
            report_rows.append({"id": eid, "status": "NO_GENERATOR"})
            continue

        try:
            requests = gen_fn(smoke=not args.full)
        except NotImplementedError as e:
            report_rows.append({"id": eid, "status": "OUT_OF_SCOPE", "reason": str(e)})
            continue

        algorithms_to_run = [algo] + [a for a in entry.get("comparison_algorithms", []) if a in EXECUTABLE_ALGORITHM_IDS]
        algorithms_to_run = [a for a in dict.fromkeys(algorithms_to_run) if a in EXECUTABLE_ALGORITHM_IDS]

        gpu_overrides = {}
        sim_req = entry.get("simulator_requirements", {}) or {}
        if "max_active_sequences" in sim_req:
            gpu_overrides["max_active_sequences"] = sim_req["max_active_sequences"]
        if "max_batch_tokens" in sim_req:
            gpu_overrides["max_batch_tokens"] = sim_req["max_batch_tokens"]
        if "max_kv_tokens" in sim_req:
            gpu_overrides["max_kv_tokens"] = sim_req["max_kv_tokens"]

        results_by_policy = {}
        for pname in algorithms_to_run:
            results_by_policy[pname] = run_policy(requests, pname, gpu_overrides, sim_req)

        entry_summary = results_by_policy.get(algo, {})
        all_entry_results[eid] = entry_summary

        gate_expr = entry.get("acceptance_gates", "")
        evaluable, passed, detail = evaluate_gate(gate_expr, results_by_policy, eid, all_entry_results)

        report_rows.append({
            "id": eid,
            "algorithm": algo,
            "test_role": entry["test_role"],
            "status": "EVALUATED" if evaluable else "NOT_AUTO_EVALUABLE",
            "gate_passed": passed,
            "detail": detail,
            "results_by_policy": results_by_policy,
        })

    n_evaluated = sum(1 for r in report_rows if r["status"] == "EVALUATED")
    n_passed = sum(1 for r in report_rows if r.get("gate_passed") is True)
    n_failed = sum(1 for r in report_rows if r.get("gate_passed") is False)
    n_not_auto = sum(1 for r in report_rows if r["status"] == "NOT_AUTO_EVALUABLE")
    n_out_of_scope = sum(1 for r in report_rows if r["status"] == "OUT_OF_SCOPE")
    n_no_gen = sum(1 for r in report_rows if r["status"] == "NO_GENERATOR")

    print(f"{'id':55s} {'status':18s} {'gate'}")
    for r in report_rows:
        gate_str = "PASS" if r.get("gate_passed") is True else ("FAIL" if r.get("gate_passed") is False else "-")
        print(f"{r['id']:55s} {r['status']:18s} {gate_str}")

    print(f"\nEvaluated: {n_evaluated} (pass={n_passed}, fail={n_failed})")
    print(f"Not auto-evaluable: {n_not_auto}")
    print(f"Out of scope this pass: {n_out_of_scope}")
    print(f"No generator: {n_no_gen}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report_rows, f, indent=2, default=str)
        print(f"\nWrote {args.json}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
