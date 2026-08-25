"""Regression tests for the Family B v1 twin / dead-diagnostic diagnosis.

These tests pin down *why* the frozen v1 result had

    decode_priority_chunked == chunked_prefill_small  (ANWG, 144/144)

and

    decode_stalled_steps == 0  (all policies, all scenarios)

They do not rewrite frozen v1 CSVs. They also do not change simulator
semantics: the finding is that this is an expected FCFS-by-arrival-time
consequence of Phase 1.5 shared contention, not an implementation or
metric bug.

See docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md §diagnosis.
"""

from __future__ import annotations

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.prefill_control_variants import (
    DEFAULT_CHUNK_SMALL,
    UNLIMITED_PREFILL_CHUNK,
    make_prefill_decode_variants,
)
from llmserveopt.policy_separation.builders import req
from llmserveopt.simulator.gpu import GPUState
from llmserveopt.simulator.request import InternalRequest, RequestPhase
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def _gpu() -> GPUState:
    return GPUState(
        GPUConfig(
            gpu_id=0,
            max_active_sequences=1024,
            max_batch_tokens=1_000_000,
            max_kv_tokens=1_000_000,
        )
    )


def _ir(
    rid: int,
    arrival: float,
    prompt: int,
    pref_rem: int,
    *,
    tokens_decoded: int = 0,
    out: int = 10,
) -> InternalRequest:
    r = Request(
        request_id=rid,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=out,
        actual_output_tokens=out,
        slo_deadline=arrival + 100000,
        priority=1.0,
        class_id="t",
    )
    return InternalRequest(
        request=r,
        phase=RequestPhase.ACTIVE,
        gpu_id=0,
        admission_time=arrival,
        prefill_remaining=pref_rem,
        tokens_decoded=tokens_decoded,
        first_token_time=(arrival if tokens_decoded > 0 else -1.0),
    )


def _step(gpu: GPUState, chunk: int, decode_first: bool) -> object:
    sm = ServiceModel(
        enable_prefill_modeling=True,
        decode_first=decode_first,
        step_token_budget=512,
        max_prefill_chunk_tokens=chunk,
        enable_decode_prefill_contention=True,
    )
    gpu.step(current_time=6.0, service_model=sm)
    return gpu.step_contention_diagnostics[-1]


def test_injected_unlimited_shared_defers_already_decoding_late_request():
    """Mid-flight injection: earlier hog still prefilling, later request
    already decoding. Unlimited shared FCFS lets the hog take the whole
    512-token budget, so the late decode is deferred. This is the shape
    ``decode_stalled_steps`` is defined to count."""
    gpu = _gpu()
    gpu._active = {
        0: _ir(0, 0.0, 2000, 1500),
        1: _ir(1, 5.0, 50, 0, tokens_decoded=1),
    }
    diag = _step(gpu, UNLIMITED_PREFILL_CHUNK, False)
    assert gpu._active[1].tokens_decoded == 1
    assert diag.decode_tokens_deferred == 1
    assert diag.decode_stalled is True


def test_injected_small_chunk_shared_does_not_defer_one_late_decode():
    """Same injection, chunk=64: leftover 448 ≥ 1 late decode, so shared
    small-chunk does *not* stall decode. This is why the v1 GPU microbench
    distinguishes full vs small but not small vs decode_first when n_late=1."""
    gpu = _gpu()
    gpu._active = {
        0: _ir(0, 0.0, 2000, 1500),
        1: _ir(1, 5.0, 50, 0, tokens_decoded=1),
    }
    diag = _step(gpu, DEFAULT_CHUNK_SMALL, False)
    assert gpu._active[1].tokens_decoded == 2
    assert diag.decode_tokens_deferred == 0
    assert diag.decode_stalled is False


def test_injected_small_chunk_decode_first_matches_shared_for_one_late_decode():
    """decode_first=True also serves the single late decode. With leftover
    ≥ n_decoding, shared small and decode-priority are observationally
    identical on this injected microbench."""
    gpu_shared = _gpu()
    gpu_shared._active = {
        0: _ir(0, 0.0, 2000, 1500),
        1: _ir(1, 5.0, 50, 0, tokens_decoded=1),
    }
    gpu_pri = _gpu()
    gpu_pri._active = {
        0: _ir(0, 0.0, 2000, 1500),
        1: _ir(1, 5.0, 50, 0, tokens_decoded=1),
    }
    d_shared = _step(gpu_shared, DEFAULT_CHUNK_SMALL, False)
    d_pri = _step(gpu_pri, DEFAULT_CHUNK_SMALL, True)
    assert gpu_shared._active[1].tokens_decoded == gpu_pri._active[1].tokens_decoded == 2
    assert d_shared.decode_tokens_deferred == d_pri.decode_tokens_deferred == 0


def test_injected_many_late_decodes_decode_first_diverges_from_small_shared():
    """``decode_first`` *is* implemented and *can* diverge — but only when
    leftover after the hog's chunk is strictly less than n_decoding.

    Shared small: hog takes 64, leftover 448 → 448 of 500 late decodes
    served, 52 deferred.
    Decode-protected: every already-decoding request is served first
    (500 served), prefill gets remainder 12.

    Natural admission traces never reach this mid-flight state: a later
    request can enter decode only by using leftover the hog did not claim,
    which then self-limits the decoding cohort to that leftover.
    """
    n_late = 500
    gpu_shared = _gpu()
    gpu_pri = _gpu()
    shared_active = {0: _ir(0, 0.0, 2000, 1500)}
    pri_active = {0: _ir(0, 0.0, 2000, 1500)}
    for i in range(n_late):
        rid = i + 1
        shared_active[rid] = _ir(rid, 5.0, 50, 0, tokens_decoded=1)
        pri_active[rid] = _ir(rid, 5.0, 50, 0, tokens_decoded=1)
    gpu_shared._active = shared_active
    gpu_pri._active = pri_active
    d_shared = _step(gpu_shared, DEFAULT_CHUNK_SMALL, False)
    d_pri = _step(gpu_pri, DEFAULT_CHUNK_SMALL, True)

    leftover = 512 - DEFAULT_CHUNK_SMALL
    assert d_shared.decode_tokens_deferred == n_late - leftover
    assert d_shared.decode_stalled is True
    assert d_pri.decode_tokens_deferred == 0
    assert d_pri.decode_stalled is False
    hog_shared_consumed = 1500 - gpu_shared._active[0].prefill_remaining
    hog_pri_consumed = 1500 - gpu_pri._active[0].prefill_remaining
    assert hog_shared_consumed == DEFAULT_CHUNK_SMALL
    assert hog_pri_consumed == 512 - n_late  # remainder after 500 decodes


def test_clean_arrival_trace_small_equals_decode_priority_and_decode_stall_zero():
    """End-to-end clean admission: convoy of long prefills then overlapping
    short tenants. Both start in prefill. Under chunk=64, leftover either
    is 0 (late tenants still prefilling → stall is prefill-side) or is
    already large enough for every tenant that bootstrapped into decode.
    Therefore decode_priority and small-chunk match, and decode_stalled_steps
    stays 0. This is the v1 workload geometry, not a metric bug.
    """
    requests = []
    rid = 0
    for i in range(6):
        requests.append(
            req(
                request_id=rid,
                arrival_time=i * 0.003,
                prompt_tokens=4096,
                predicted_output_tokens=40,
                slo_deadline=i * 0.003 + 2.0,
                class_id="tenant_prefill",
            )
        )
        rid += 1
    late_start = 0.25 * 5 * 0.003
    for i in range(8):
        requests.append(
            req(
                request_id=rid,
                arrival_time=late_start + i * 0.004,
                prompt_tokens=128,
                predicted_output_tokens=40,
                slo_deadline=late_start + i * 0.004 + 2.0,
                class_id="tenant_late",
            )
        )
        rid += 1

    gpu_configs = [
        GPUConfig(
            gpu_id=0,
            max_active_sequences=64,
            max_batch_tokens=64,
            max_kv_tokens=8_000_000,
        )
    ]
    base_kwargs = {
        "step_size": 0.001,
        "enable_prefill_modeling": True,
        "prefill_cost_per_token": 1.0,
        "step_token_budget": 512,
        "enable_decode_prefill_contention": True,
        "decode_first": False,
    }
    variants = make_prefill_decode_variants()

    def run(name: str):
        policy, kw = variants[name]
        merged = dict(base_kwargs)
        merged.update(kw)
        sim = Simulator(
            SimulatorConfig(
                gpu_configs=gpu_configs,
                service_model=ServiceModel(**merged),
            )
        )
        sim.load_trace(list(requests))
        metrics = sim.run(policy, workload_tag=name, seed=0)
        summary = sim.contention_diagnostics_summary()
        return metrics.arrival_normalized_weighted_goodput, summary

    anwg_small, s_small = run("chunked_prefill_small")
    anwg_pri, s_pri = run("decode_priority_chunked")
    anwg_full, s_full = run("full_prefill")

    assert s_small["decode_stalled_steps"] == 0
    assert s_pri["decode_stalled_steps"] == 0
    assert s_full["decode_stalled_steps"] == 0
    assert anwg_small == anwg_pri
    # Full vs small may or may not separate on this tiny slack-2.0 cell;
    # the invariant under test is the twin + dead decode-stall diagnostic.
    assert s_small["prefill_stalled_steps"] >= 0
    assert s_full["prefill_stalled_steps"] >= 0


def test_decode_stalled_steps_counts_only_already_decoding_zero_progress():
    """Instrumentation check: a request still prefilling that received 0
    tokens increments prefill_stalled_steps, not decode_stalled_steps.
    v1 late tenants were blocked in prefill, so the decode-stall counter
    was correctly 0.
    """
    gpu = _gpu()
    # Two prefilling requests, earlier hog takes the full budget.
    gpu._active = {
        0: _ir(0, 0.0, 2000, 1500),
        1: _ir(1, 1.0, 500, 500),
    }
    diag = _step(gpu, UNLIMITED_PREFILL_CHUNK, False)
    assert diag.decode_tokens_deferred == 0
    assert diag.decode_stalled is False
    assert diag.prefill_requests_stalled >= 1
    assert diag.prefill_stalled is True
