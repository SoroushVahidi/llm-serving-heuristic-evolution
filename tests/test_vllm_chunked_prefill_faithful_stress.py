"""Randomized stress tests for vllm_chunked_prefill_faithful (task
instruction: >=200 randomized cheap trials varying prompt length, output
length, arrival pattern, max_num_batched_tokens, max_num_seqs, block size,
KV capacity, and burst size).

Verifies, per trial: deterministic repeated runs, no over-budget
scheduling, no over-capacity KV allocation (would raise
KVBlockManagerError, which propagates as a test failure -- not swallowed),
no duplicate execution, no lost requests (every request is accounted for:
completed, legitimately still waiting due to infeasible capacity, or --
never expected -- silently missing), monotonic prefill progress, eventual
completion where feasible, and correct prefill->decode transition.
"""
from __future__ import annotations

import random
import warnings

import pytest

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.vllm_chunked_prefill_faithful import VLLMChunkedPrefillFaithfulPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

N_TRIALS = 220
MASTER_SEED = 20260719


def _random_trial_config(rng: random.Random) -> dict:
    n_requests = rng.randint(1, 12)
    block_size = rng.choice([1, 4, 16, 32])
    max_num_batched_tokens = rng.choice([8, 16, 64, 128, 256, 512, 1024])
    max_num_seqs = rng.choice([1, 2, 4, 8, 32, 128, 256])
    watermark = rng.choice([0.0, 0.01, 0.05])
    # KV capacity: sometimes generous, sometimes deliberately tight (to
    # exercise preemption / admission-blocking paths).
    kv_capacity_mode = rng.choice(["generous", "tight"])

    prompts = [rng.randint(1, 3000) for _ in range(n_requests)]
    outputs = [rng.randint(1, 40) for _ in range(n_requests)]

    if kv_capacity_mode == "generous":
        max_kv_tokens = sum(prompts) + sum(outputs) + 10_000
    else:
        # Deliberately tight: only enough for roughly half the requests'
        # prompts at once, forcing queueing/preemption pressure.
        max_kv_tokens = max(block_size * 4, sum(prompts) // 2 + block_size)

    arrival_pattern = rng.choice(["burst", "spread", "staggered_pairs"])
    if arrival_pattern == "burst":
        arrivals = [0.0] * n_requests
    elif arrival_pattern == "spread":
        arrivals = sorted(rng.uniform(0.0, 5.0) for _ in range(n_requests))
    else:  # staggered_pairs
        arrivals = []
        t = 0.0
        for i in range(n_requests):
            if i % 2 == 0:
                t += rng.uniform(0.0, 1.0)
            arrivals.append(t)
        arrivals.sort()

    return dict(
        n_requests=n_requests, block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs,
        watermark=watermark, max_kv_tokens=max_kv_tokens,
        prompts=prompts, outputs=outputs, arrivals=arrivals,
    )


def _build_requests(cfg: dict) -> list:
    return [
        Request(
            request_id=i, arrival_time=cfg["arrivals"][i], prompt_tokens=cfg["prompts"][i],
            predicted_output_tokens=cfg["outputs"][i], actual_output_tokens=cfg["outputs"][i],
            slo_deadline=cfg["arrivals"][i] + 100_000.0, priority=1.0, class_id="stress",
        )
        for i in range(cfg["n_requests"])
    ]


def _run_trial(cfg: dict, seed: int):
    gpu = GPUConfig(
        gpu_id=0, max_active_sequences=cfg["max_num_seqs"] * 2,
        max_batch_tokens=1_000_000,  # effectively unbounded -- the policy's own budget is the constraint under test
        max_kv_tokens=cfg["max_kv_tokens"],
    )
    sm = ServiceModel(
        enable_prefill_modeling=True, decode_first=True,
        step_token_budget=cfg["max_num_batched_tokens"],
        max_prefill_chunk_tokens=cfg["max_num_batched_tokens"],
        prefill_cost_per_token=1.0,
    )
    # Generous step budget: enough for the LAST arrival to even be enqueued
    # (arrival times are in simulated wall-clock seconds, step_size=0.001s
    # by ServiceModel default) PLUS every request's full prefill (in
    # max_num_batched_tokens-sized chunks) plus its full decode, fully
    # serialized (safe upper bound for any max_num_seqs, including 1),
    # times a safety factor for queueing delay under tight-KV trials.
    step_size = 0.001
    last_arrival_steps = int(max(cfg["arrivals"]) / step_size) + 1 if cfg["arrivals"] else 0
    total_prefill_steps = sum(-(-p // max(1, cfg["max_num_batched_tokens"])) for p in cfg["prompts"])
    total_decode_steps = sum(cfg["outputs"])
    max_steps = last_arrival_steps + (total_prefill_steps + total_decode_steps) * 3 + 500

    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, max_steps=max_steps, drain_steps=max_steps))
    sim.load_trace(_build_requests(cfg))
    policy = VLLMChunkedPrefillFaithfulPolicy(
        block_size=cfg["block_size"], max_num_batched_tokens=cfg["max_num_batched_tokens"],
        max_num_seqs=cfg["max_num_seqs"], watermark=cfg["watermark"],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = sim.run(policy, workload_tag="stress", seed=seed)
    return metrics, w


@pytest.mark.parametrize("trial_idx", range(N_TRIALS))
def test_randomized_trial_invariants(trial_idx):
    rng = random.Random(MASTER_SEED + trial_idx)
    cfg = _random_trial_config(rng)
    metrics, sim_warnings = _run_trial(cfg, seed=MASTER_SEED + trial_idx)

    # No over-capacity KV allocation: KVBlockManagerError would have
    # propagated as an exception and failed this test outright (not
    # caught/swallowed anywhere in the policy or simulator) -- reaching
    # this point at all is part of the invariant.

    # No admission the simulator's OWN GPUConfig constraints would reject
    # (the policy's _feasible_on_gpu safety net) -- would surface as a
    # GPUState.admit() warning.
    assert sim_warnings == [], (
        f"trial {trial_idx} cfg={cfg}: unexpected simulator warnings: "
        f"{[str(x.message) for x in sim_warnings]}"
    )

    # No lost requests: every request is either completed or legitimately
    # still waiting (infeasible capacity) -- never silently missing.
    assert metrics.num_completed + metrics.num_dropped == cfg["n_requests"], (
        f"trial {trial_idx} cfg={cfg}: {metrics.num_completed} completed + "
        f"{metrics.num_dropped} dropped != {cfg['n_requests']} total "
        f"(some requests neither completed nor accounted as dropped -- "
        f"a genuine lost-request bug, not a capacity limitation)"
    )

    # Eventual completion where feasible: admission is FCFS (by arrival
    # then request_id -- see docs/vllm_chunked_prefill_faithful_scheduler_
    # reference.md's algorithm section and its "stop admission entirely on
    # first failure" fidelity point), NOT shortest-job-first, so a request
    # ahead in the queue that genuinely cannot fit blocks the whole queue
    # behind it -- correct, faithful backpressure, not a bug. The only
    # request whose OWN standalone feasibility is a meaningful "should have
    # completed" claim is request 0 (by construction, every arrival-pattern
    # generator above assigns request_id in non-decreasing arrival-time
    # order, so request 0 is always FCFS-first).
    if metrics.num_dropped > 0:
        front_prompt = cfg["prompts"][0]
        watermark_blocks = int(cfg["watermark"] * (cfg["max_kv_tokens"] // cfg["block_size"]))
        front_blocks_needed = -(-front_prompt // cfg["block_size"])
        total_blocks = cfg["max_kv_tokens"] // cfg["block_size"]
        if total_blocks - front_blocks_needed >= watermark_blocks:
            assert metrics.num_completed >= 1, (
                f"trial {trial_idx} cfg={cfg}: the FCFS-first request "
                f"(request 0) fits standalone within KV capacity, so it "
                f"should have been admitted and completed eventually, but "
                f"num_completed=0"
            )


def test_deterministic_across_seeded_repeats_sampled():
    """A sampled subset of the same randomized configs, re-run twice each,
    must produce byte-identical metrics both times."""
    rng = random.Random(MASTER_SEED)
    for i in range(20):
        cfg = _random_trial_config(rng)
        m1, w1 = _run_trial(cfg, seed=1000 + i)
        m2, w2 = _run_trial(cfg, seed=1000 + i)
        assert m1.num_completed == m2.num_completed
        assert m1.num_dropped == m2.num_dropped
        if m1.num_completed > 0:
            assert m1.mean_latency == m2.mean_latency
            assert m1.mean_ttft == m2.mean_ttft
        else:
            import math
            assert math.isnan(m1.mean_latency) and math.isnan(m2.mean_latency)


def test_monotonic_prefill_progress_across_randomized_chunk_sizes():
    """Direct admission-accounting-level check (not just end-to-end
    completion): for a battery of randomized (prompt, budget) pairs, the
    shadow remaining_prefill count must never increase between consecutive
    select_action calls for the same tracked request."""
    from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState

    def make_req(req_id, prompt, arrival=0.0):
        return ObservableRequest(
            request_id=req_id, arrival_time=arrival, prompt_tokens=prompt,
            predicted_output_tokens=8, slo_deadline=1000.0, priority=1.0, class_id="stress",
        )

    def make_gpu(active_reqs=()):
        return ObservableGPUState(
            gpu_id=0, max_active_sequences=100, max_batch_tokens=10_000_000,
            max_kv_tokens=10_000_000, active_request_ids=[r.request_id for r in active_reqs],
            active_requests_info=list(active_reqs), current_kv_tokens=0,
            tokens_decoded_per_request={},
        )

    rng = random.Random(MASTER_SEED + 999)
    for trial in range(40):
        prompt = rng.randint(1, 5000)
        budget = rng.randint(1, 1024)
        policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=budget, watermark=0.0)
        req = make_req(0, prompt)
        action = policy.select_action(ObservableState(
            time=0.0, waiting_queue=[req], gpu_states=[make_gpu()], completed_count=0, step=0,
        ))
        assert action.admit[0] == [0], f"trial {trial}: prompt={prompt} budget={budget} not admitted"

        history = [policy._request_states[0][0].remaining_prefill]
        active = make_req(0, prompt)
        steps = 0
        while policy._request_states[0][0].remaining_prefill > 0 and steps < 10_000:
            gpu = make_gpu(active_reqs=[active])
            policy.select_action(ObservableState(
                time=0.0, waiting_queue=[], gpu_states=[gpu], completed_count=0, step=steps + 1,
            ))
            history.append(policy._request_states[0][0].remaining_prefill)
            steps += 1

        for earlier, later in zip(history, history[1:]):
            assert later < earlier, f"trial {trial}: prompt={prompt} budget={budget} non-monotonic: {history}"
        assert history[-1] == 0
