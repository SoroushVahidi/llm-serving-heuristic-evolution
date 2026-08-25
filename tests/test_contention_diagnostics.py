"""Tests for the diagnostic-only per-step contention/saturation signals
added in `contention_diagnostics.py` (see
docs/selector_v2_contention_frontier_search.md).

These signals exist to distinguish "the decode/prefill contention
mechanism never fired this step" from "it fired but didn't change the
chosen objective" -- the ambiguity that made the Selector v2 overnight
contention-validation pilot's 300/300-tied result hard to root-cause from
outcome metrics alone. Diagnostic only: no execution or objective code
reads these fields.
"""
from __future__ import annotations

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.weighted_shortest_processing import WeightedShortestProcessingPolicy
from llmserveopt.simulator.contention_diagnostics import summarize
from llmserveopt.simulator.gpu import GPUState
from llmserveopt.simulator.request import InternalRequest, RequestPhase
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

GPU_CFG = GPUConfig(gpu_id=0, max_active_sequences=64, max_batch_tokens=1_000_000, max_kv_tokens=1_000_000)


def _make_active_ir(rid: int, arrival: float, prompt_tokens: int, prefill_remaining: int,
                     tokens_decoded: int = 0, output_tokens: int = 10) -> InternalRequest:
    req = Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt_tokens,
        predicted_output_tokens=output_tokens, actual_output_tokens=output_tokens,
        slo_deadline=arrival + 100_000.0, priority=1.0, class_id="test",
    )
    return InternalRequest(
        request=req, phase=RequestPhase.ACTIVE, gpu_id=0, admission_time=arrival,
        prefill_remaining=prefill_remaining, tokens_decoded=tokens_decoded,
        first_token_time=(arrival if tokens_decoded > 0 else -1.0),
    )


def _early_prefill_late_decode_gpu() -> GPUState:
    """Same hand-constructed micro-counterexample used by
    test_decode_prefill_contention_execution.py: one still-prefilling
    request that arrived first (demands the full 512-token chunk every
    step), one already-decoding request that arrived later. Under
    decode_first=False (shared contention), the earlier-arrived prefiller
    sorts first and exhausts the budget, so the later-arrived decoder gets
    zero progress -- the one mechanism this simulator can actually produce
    from a directly-injected state (not from a clean arrival trace; see
    contention_fixtures.py's module docstring for why)."""
    prefill_req = _make_active_ir(0, arrival=0.0, prompt_tokens=2000, prefill_remaining=1500)
    decode_req = _make_active_ir(1, arrival=5.0, prompt_tokens=50, prefill_remaining=0, tokens_decoded=1)
    gpu = GPUState(GPU_CFG)
    gpu._active = {0: prefill_req, 1: decode_req}
    return gpu


class TestDecodeProtectedNeverDefers:

    def test_decode_protected_never_defers_decode(self):
        """Invariant: `_advance_decode_protected` calls `advance_decode`
        unconditionally for every decoding request, regardless of budget
        -- so `decode_tokens_deferred` must be exactly 0 every step,
        regardless of how contended the scenario is."""
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                           enable_decode_prefill_contention=True,
                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu = _early_prefill_late_decode_gpu()
        gpu.step(current_time=6.0, service_model=sm)

        assert len(gpu.step_contention_diagnostics) == 1
        diag = gpu.step_contention_diagnostics[0]
        assert diag.decode_tokens_deferred == 0
        assert diag.decode_stalled is False
        assert diag.decode_tokens_served == 1
        assert diag.num_decoding == 1


class TestSharedContentionDefersDecode:

    def test_shared_contention_defers_the_later_arrived_decoder(self):
        """The hand-constructed counterexample: under decode_first=False,
        the diagnostics must show the decoder was deferred and that
        prefill work was scheduled in the very same step it was deferred
        in."""
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=False,
                           enable_decode_prefill_contention=True,
                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu = _early_prefill_late_decode_gpu()
        gpu.step(current_time=6.0, service_model=sm)

        diag = gpu.step_contention_diagnostics[0]
        assert diag.decode_tokens_deferred == 1
        assert diag.decode_stalled is True
        assert diag.decode_tokens_served == 0
        assert diag.prefill_tokens_served == 512
        assert diag.prefill_scheduled_while_decode_deferred is True
        assert diag.budget_used == 512
        assert diag.budget_saturated is True

    def test_legacy_default_mode_never_populates_deferred(self):
        """`enable_decode_prefill_contention=False` (the default) still
        goes through `_advance_decode_protected` regardless of
        `decode_first` -- diagnostics must reflect that (zero deferred),
        matching the documented dead-branch behavior."""
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=False,
                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu = _early_prefill_late_decode_gpu()
        gpu.step(current_time=6.0, service_model=sm)
        assert gpu.step_contention_diagnostics[0].decode_tokens_deferred == 0


class TestSummarize:

    def test_summarize_empty_history(self):
        s = summarize([])
        assert s["decode_stalled_steps"] == 0
        assert s["n_steps"] == 0
        assert s["budget_saturation_fraction"] == 0.0

    def test_summarize_aggregates_across_steps(self):
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=False,
                           enable_decode_prefill_contention=True,
                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu = _early_prefill_late_decode_gpu()
        gpu.step(current_time=6.0, service_model=sm)
        gpu.step(current_time=6.001, service_model=sm)
        s = summarize(gpu.step_contention_diagnostics)
        assert s["n_steps"] == 2
        assert s["decode_stalled_steps"] == 2
        assert s["cumulative_decode_tokens_deferred"] == 2
        assert s["steps_with_prefill_while_decode_deferred"] == 2
        assert s["budget_saturation_fraction"] == 1.0


class TestRealisticAdmissionOrderDivergence:
    """The genuinely new finding from the frontier-search audit: divergence
    between decode_first=True/False is reachable from a normal (non-
    injected) arrival trace when the ADMITTING POLICY's insertion order
    disagrees with strict (arrival_time, request_id) order -- e.g. a
    shortest-job-first-style policy that admits a short request ahead of a
    longer one that arrived no later. `_advance_decode_protected`'s prefill
    loop follows insertion order; `_advance_shared_contention`'s follows
    strict arrival order -- so when a policy's admission order disagrees
    with arrival order, the two execution models schedule the SAME two
    still-prefilling requests in a DIFFERENT relative order. See
    docs/selector_v2_contention_frontier_search.md for the full derivation
    (this is distinct from -- and does not contradict -- the decode-vs-
    prefill "later decode stalled behind an earlier persistent prefill"
    mechanism contention_fixtures.py already documented as self-limiting
    under normal traces)."""

    def _run(self, decode_first: bool):
        reqs = [
            Request(request_id=0, arrival_time=0.0, prompt_tokens=10_000, predicted_output_tokens=1,
                    actual_output_tokens=1, slo_deadline=1000.0, priority=1.0, class_id="long"),
            Request(request_id=1, arrival_time=0.0, prompt_tokens=500, predicted_output_tokens=1,
                    actual_output_tokens=1, slo_deadline=1000.0, priority=1.0, class_id="short"),
        ]
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=decode_first,
                           enable_decode_prefill_contention=True,
                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu_configs = [GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000, max_kv_tokens=200_000)]
        sim = Simulator(SimulatorConfig(gpu_configs=gpu_configs, service_model=sm, drain_steps=200))
        sim.load_trace(list(reqs))
        policy = WeightedShortestProcessingPolicy()
        policy.reset()
        metrics = sim.run(policy=policy, workload_tag="sjf_admission_reorder", seed=1)
        per_request = {c.request.request_id: c.completion_time - c.admission_time for c in sim._completed}
        return metrics, per_request

    def test_sjf_admission_order_produces_real_latency_divergence(self):
        m_protected, lat_protected = self._run(decode_first=True)
        m_shared, lat_shared = self._run(decode_first=False)

        # Decode-protected: insertion order follows the policy's SJF
        # admission order, so the SHORT request (id=1) is served first.
        assert lat_protected[1] < lat_protected[0]
        # Shared/FCFS: strict arrival order (tie-broken by id ascending)
        # gives the LONG request (id=0) priority regardless of the
        # policy's own admission preference.
        assert lat_shared[0] <= lat_shared[1]
        # The aggregate objective genuinely differs -- this is the
        # divergence the 300-window random search's hog-plus-tiny-runner
        # window shape structurally could not reach (see the frontier-
        # search doc's root-cause section).
        assert abs(m_protected.mean_latency - m_shared.mean_latency) > 1e-9

    def test_mechanism_is_prefill_vs_prefill_not_decode_vs_prefill(self):
        """Neither request in this scenario ever decodes before the other
        finishes prefill (both have predicted_output_tokens=1) -- so this
        divergence is a PURE prefill-vs-prefill admission-order effect,
        distinct from `decode_tokens_deferred`. `prefill_requests_stalled`
        must be the diagnostic that actually captures it."""
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=False,
                           enable_decode_prefill_contention=True,
                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu_configs = [GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000, max_kv_tokens=200_000)]
        reqs = [
            Request(request_id=0, arrival_time=0.0, prompt_tokens=10_000, predicted_output_tokens=1,
                    actual_output_tokens=1, slo_deadline=1000.0, priority=1.0, class_id="long"),
            Request(request_id=1, arrival_time=0.0, prompt_tokens=500, predicted_output_tokens=1,
                    actual_output_tokens=1, slo_deadline=1000.0, priority=1.0, class_id="short"),
        ]
        sim = Simulator(SimulatorConfig(gpu_configs=gpu_configs, service_model=sm, drain_steps=200))
        sim.load_trace(list(reqs))
        policy = WeightedShortestProcessingPolicy()
        policy.reset()
        sim.run(policy=policy, workload_tag="sjf_prefill_stall", seed=1)
        history = sim._gpus[0].step_contention_diagnostics
        assert any(d.prefill_requests_stalled > 0 for d in history)
        assert all(d.decode_tokens_deferred == 0 for d in history)
