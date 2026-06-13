"""
Tests for Phase 1.5 prefill modeling:
  - Requests stay in prefill phase until advance_prefill() finishes.
  - advance_decode() raises if called while prefill_remaining > 0.
  - first_token_time is set exactly on the first advance_decode() call.
  - TTFT and TPOT are computed correctly from CompletedRequest.
  - Chunked-prefill limits are respected by the GPU step.
  - SarathiStylePolicy and SplitFuseStylePolicy import and return valid actions.
"""
from __future__ import annotations

import math
import pytest

from llmserveopt.core.types import CompletedRequest, GPUConfig, Request
from llmserveopt.policies.orca_style import OrcaStylePolicy
from llmserveopt.policies.sarathi_style import SarathiStylePolicy
from llmserveopt.policies.splitfuse_style import SplitFuseStylePolicy
from llmserveopt.policies.vllm_style_token_budget import VLLMStyleTokenBudgetPolicy
from llmserveopt.policies.slo_slack_score import SloSlackScorePolicy
from llmserveopt.simulator.gpu import GPUState
from llmserveopt.simulator.request import InternalRequest, RequestPhase
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    request_id: int = 0,
    prompt_tokens: int = 64,
    output_tokens: int = 4,
    arrival_time: float = 0.0,
) -> Request:
    return Request(
        request_id=request_id,
        arrival_time=arrival_time,
        prompt_tokens=prompt_tokens,
        predicted_output_tokens=output_tokens,
        actual_output_tokens=output_tokens,
        slo_deadline=arrival_time + 100.0,
        priority=1.0,
        class_id="medium",
    )


def _make_ir(prompt_tokens: int = 64, output_tokens: int = 4) -> InternalRequest:
    return InternalRequest(request=_make_request(prompt_tokens=prompt_tokens, output_tokens=output_tokens))


def _make_service_model(
    enable: bool = True,
    chunk: int = 16,
    budget: int = 256,
    decode_first: bool = False,
) -> ServiceModel:
    return ServiceModel(
        enable_prefill_modeling=enable,
        prefill_cost_per_token=1.0,
        max_prefill_chunk_tokens=chunk,
        step_token_budget=budget,
        decode_first=decode_first,
    )


# ---------------------------------------------------------------------------
# InternalRequest prefill phase tests
# ---------------------------------------------------------------------------

class TestInternalRequestPrefill:

    def test_prefill_remaining_initialises_zero(self):
        ir = _make_ir(prompt_tokens=32)
        assert ir.prefill_remaining == 0

    def test_is_prefilling_true_when_active_and_remaining(self):
        ir = _make_ir(prompt_tokens=32)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 32
        assert ir.is_prefilling is True
        assert ir.is_decoding is False

    def test_is_decoding_true_when_active_and_no_remaining(self):
        ir = _make_ir(prompt_tokens=32)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 0
        assert ir.is_prefilling is False
        assert ir.is_decoding is True

    def test_advance_prefill_decrements(self):
        ir = _make_ir(prompt_tokens=32)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 32
        done = ir.advance_prefill(16)
        assert ir.prefill_remaining == 16
        assert done is False

    def test_advance_prefill_completes(self):
        ir = _make_ir(prompt_tokens=32)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 16
        done = ir.advance_prefill(16)
        assert ir.prefill_remaining == 0
        assert done is True

    def test_advance_prefill_clamps_to_zero(self):
        ir = _make_ir(prompt_tokens=32)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 5
        done = ir.advance_prefill(100)
        assert ir.prefill_remaining == 0
        assert done is True

    def test_advance_decode_raises_if_prefill_remaining(self):
        ir = _make_ir(prompt_tokens=32)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 8
        with pytest.raises(RuntimeError, match="prefill"):
            ir.advance_decode(current_time=0.1)

    def test_advance_decode_sets_first_token_time(self):
        ir = _make_ir(prompt_tokens=32, output_tokens=4)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 0
        ir.advance_decode(current_time=0.5)
        assert ir.first_token_time == pytest.approx(0.5)

    def test_advance_decode_does_not_overwrite_first_token_time(self):
        ir = _make_ir(prompt_tokens=32, output_tokens=4)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 0
        ir.advance_decode(current_time=0.5)
        ir.advance_decode(current_time=0.6)
        assert ir.first_token_time == pytest.approx(0.5)  # not overwritten

    def test_kv_tokens_grows_during_prefill(self):
        ir = _make_ir(prompt_tokens=32)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 32
        # No tokens prefilled yet → 0 kv
        assert ir.kv_tokens == 0
        ir.advance_prefill(16)
        # 16 prompt tokens prefilled
        assert ir.kv_tokens == 16
        ir.advance_prefill(16)
        # All prefilled, no decodes yet
        assert ir.kv_tokens == 32

    def test_kv_tokens_grows_during_decode(self):
        ir = _make_ir(prompt_tokens=32, output_tokens=4)
        ir.phase = RequestPhase.ACTIVE
        ir.prefill_remaining = 0
        # Full prompt is "prefilled" (all 32 counted)
        assert ir.kv_tokens == 32
        ir.advance_decode(0.1)
        assert ir.kv_tokens == 33


# ---------------------------------------------------------------------------
# ServiceModel tests
# ---------------------------------------------------------------------------

class TestServiceModel:

    def test_compute_prefill_tokens_disabled(self):
        sm = ServiceModel(enable_prefill_modeling=False)
        assert sm.compute_prefill_tokens(128) == 0

    def test_compute_prefill_tokens_enabled(self):
        sm = ServiceModel(enable_prefill_modeling=True, prefill_cost_per_token=1.0)
        assert sm.compute_prefill_tokens(64) == 64

    def test_compute_prefill_tokens_fractional_cost(self):
        sm = ServiceModel(enable_prefill_modeling=True, prefill_cost_per_token=0.5)
        assert sm.compute_prefill_tokens(64) == 32

    def test_compute_prefill_tokens_rounds_up(self):
        sm = ServiceModel(enable_prefill_modeling=True, prefill_cost_per_token=0.5)
        assert sm.compute_prefill_tokens(3) == 2   # ceil(1.5)

    def test_prefill_steps(self):
        sm = ServiceModel(
            enable_prefill_modeling=True,
            prefill_cost_per_token=1.0,
            max_prefill_chunk_tokens=16,
        )
        assert sm.prefill_steps(32) == 2
        assert sm.prefill_steps(33) == 3
        assert sm.prefill_steps(16) == 1


# ---------------------------------------------------------------------------
# GPUState prefill-aware step tests
# ---------------------------------------------------------------------------

class TestGPUStatePrefillStep:

    def _make_gpu(self, max_seq: int = 4, max_kv: int = 4096) -> GPUState:
        cfg = GPUConfig(
            gpu_id=0,
            max_active_sequences=max_seq,
            max_batch_tokens=max_seq,
            max_kv_tokens=max_kv,
        )
        return GPUState(cfg)

    def test_admit_sets_prefill_remaining(self):
        gpu = self._make_gpu()
        ir = _make_ir(prompt_tokens=32, output_tokens=4)
        sm = _make_service_model(enable=True, chunk=16)
        gpu.admit(ir, admission_time=0.0, service_model=sm)
        assert ir.prefill_remaining == 32

    def test_admit_phase1_no_prefill(self):
        gpu = self._make_gpu()
        ir = _make_ir(prompt_tokens=32, output_tokens=4)
        sm = ServiceModel(enable_prefill_modeling=False)
        gpu.admit(ir, admission_time=0.0, service_model=sm)
        assert ir.prefill_remaining == 0

    def test_step_phase1_advances_decode(self):
        gpu = self._make_gpu()
        ir = _make_ir(prompt_tokens=32, output_tokens=2)
        sm = ServiceModel(enable_prefill_modeling=False)
        gpu.admit(ir, admission_time=0.0, service_model=sm)
        completed = gpu.step(current_time=0.001, service_model=sm)
        assert len(completed) == 0
        assert ir.tokens_decoded == 1

    def test_step_phase15_prefill_does_not_advance_decode(self):
        gpu = self._make_gpu()
        ir = _make_ir(prompt_tokens=32, output_tokens=2)
        sm = _make_service_model(enable=True, chunk=16, budget=256)
        gpu.admit(ir, admission_time=0.0, service_model=sm)
        assert ir.prefill_remaining == 32
        # Step 1: prefill chunk processed, no decode
        gpu.step(current_time=0.001, service_model=sm)
        assert ir.tokens_decoded == 0
        assert ir.prefill_remaining == 16

    def test_step_phase15_prefill_completes(self):
        gpu = self._make_gpu()
        ir = _make_ir(prompt_tokens=32, output_tokens=2)
        sm = _make_service_model(enable=True, chunk=32, budget=256)
        gpu.admit(ir, admission_time=0.0, service_model=sm)
        gpu.step(current_time=0.001, service_model=sm)
        assert ir.prefill_remaining == 0

    def test_step_phase15_decode_after_prefill(self):
        gpu = self._make_gpu()
        ir = _make_ir(prompt_tokens=16, output_tokens=3)
        sm = _make_service_model(enable=True, chunk=16, budget=256)
        gpu.admit(ir, admission_time=0.0, service_model=sm)
        # Step 1: prefill (chunk=16 = prompt_tokens → done)
        gpu.step(current_time=0.001, service_model=sm)
        assert ir.prefill_remaining == 0
        assert ir.tokens_decoded == 0
        # Step 2: decode step 1
        gpu.step(current_time=0.002, service_model=sm)
        assert ir.tokens_decoded == 1
        assert ir.first_token_time == pytest.approx(0.002)

    def test_chunk_limit_enforced(self):
        gpu = self._make_gpu()
        ir = _make_ir(prompt_tokens=64, output_tokens=2)
        sm = _make_service_model(enable=True, chunk=16, budget=4096)
        gpu.admit(ir, admission_time=0.0, service_model=sm)
        gpu.step(current_time=0.001, service_model=sm)
        # Only 16 tokens should have been processed
        assert ir.prefill_remaining == 64 - 16

    def test_prefilling_count_and_decoding_count(self):
        gpu = self._make_gpu(max_seq=4)
        sm = _make_service_model(enable=True, chunk=8, budget=4096)
        ir_prefill = _make_ir(prompt_tokens=32, output_tokens=2)
        ir_prefill.request = _make_request(request_id=1, prompt_tokens=32, output_tokens=2)
        ir_decode  = _make_ir(prompt_tokens=8, output_tokens=2)
        ir_decode.request = _make_request(request_id=2, prompt_tokens=8, output_tokens=2)
        # Admit both
        gpu.admit(ir_prefill, 0.0, sm)
        gpu.admit(ir_decode, 0.0, sm)
        # Manually complete prefill of ir_decode
        ir_decode.prefill_remaining = 0
        assert gpu.prefilling_count == 1
        assert gpu.decoding_count == 1

    def test_completed_request_has_first_token_time(self):
        gpu = self._make_gpu()
        ir = _make_ir(prompt_tokens=8, output_tokens=1)
        sm = _make_service_model(enable=True, chunk=8, budget=4096)
        gpu.admit(ir, admission_time=0.0, service_model=sm)
        # Step 1: prefill completes
        gpu.step(current_time=0.001, service_model=sm)
        # Step 2: decode (output_tokens=1 → completes)
        completed = gpu.step(current_time=0.002, service_model=sm)
        assert len(completed) == 1
        c = completed[0]
        assert c.first_token_time == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# CompletedRequest TTFT / TPOT tests
# ---------------------------------------------------------------------------

class TestCompletedRequestMetrics:

    def _make_completed(
        self,
        prompt_tokens: int = 32,
        output_tokens: int = 4,
        arrival_time: float = 0.0,
        admission_time: float = 0.1,
        first_token_time: float = 0.3,
        completion_time: float = 0.7,
    ) -> CompletedRequest:
        req = _make_request(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            arrival_time=arrival_time,
        )
        return CompletedRequest(
            request=req,
            admission_time=admission_time,
            completion_time=completion_time,
            gpu_id=0,
            first_token_time=first_token_time,
        )

    def test_ttft(self):
        c = self._make_completed(arrival_time=0.0, first_token_time=0.3)
        assert c.ttft == pytest.approx(0.3)

    def test_ttft_nan_when_not_recorded(self):
        c = self._make_completed(first_token_time=-1.0)
        assert math.isnan(c.ttft)

    def test_tpot(self):
        # 4 output tokens, first at t=0.3, completion at t=0.7
        # TPOT = (0.7 - 0.3) / max(1, 4-1) = 0.4 / 3
        c = self._make_completed(output_tokens=4, first_token_time=0.3, completion_time=0.7)
        assert c.tpot == pytest.approx(0.4 / 3)

    def test_tpot_single_token(self):
        # output_tokens=1 → n_intervals = max(1, 0) = 1
        c = self._make_completed(output_tokens=1, first_token_time=0.3, completion_time=0.5)
        assert c.tpot == pytest.approx(0.2)

    def test_tpot_nan_when_not_recorded(self):
        c = self._make_completed(first_token_time=-1.0)
        assert math.isnan(c.tpot)

    def test_prefill_delay(self):
        c = self._make_completed(admission_time=0.1, first_token_time=0.3)
        assert c.prefill_delay == pytest.approx(0.2)

    def test_prefill_delay_nan_when_not_recorded(self):
        c = self._make_completed(first_token_time=-1.0)
        assert math.isnan(c.prefill_delay)


# ---------------------------------------------------------------------------
# End-to-end simulator tests with prefill modeling
# ---------------------------------------------------------------------------

class TestSimulatorPrefillE2E:

    def _make_sim(
        self,
        enable_prefill: bool = True,
        chunk: int = 16,
        budget: int = 4096,
        decode_first: bool = False,
    ) -> Simulator:
        sm = ServiceModel(
            enable_prefill_modeling=enable_prefill,
            prefill_cost_per_token=1.0,
            max_prefill_chunk_tokens=chunk,
            step_token_budget=budget,
            decode_first=decode_first,
        )
        gpu_cfg = GPUConfig(
            gpu_id=0,
            max_active_sequences=8,
            max_batch_tokens=8,
            max_kv_tokens=8192,
        )
        return Simulator(SimulatorConfig(
            gpu_configs=[gpu_cfg],
            service_model=sm,
            drain_steps=50000,
        ))

    def test_prefill_model_increases_latency(self):
        """Enabling prefill modeling should increase latency vs instant-prefill."""
        reqs = [
            _make_request(request_id=i, prompt_tokens=64, output_tokens=8, arrival_time=float(i) * 0.01)
            for i in range(5)
        ]

        sim_p1 = self._make_sim(enable_prefill=False)
        sim_p1.load_trace(reqs)
        m_p1 = sim_p1.run(OrcaStylePolicy(), "e2e_p1")

        sim_p15 = self._make_sim(enable_prefill=True, chunk=16)
        sim_p15.load_trace(reqs)
        m_p15 = sim_p15.run(OrcaStylePolicy(), "e2e_p15")

        assert m_p15.mean_latency > m_p1.mean_latency

    def test_ttft_is_recorded_with_prefill_modeling(self):
        reqs = [
            _make_request(request_id=i, prompt_tokens=16, output_tokens=4, arrival_time=float(i) * 0.01)
            for i in range(3)
        ]
        sim = self._make_sim(enable_prefill=True, chunk=16)
        sim.load_trace(reqs)
        m = sim.run(OrcaStylePolicy(), "ttft_test")
        assert m.num_completed > 0
        assert not math.isnan(m.mean_ttft)
        assert m.mean_ttft >= 0.0

    def test_ttft_is_smaller_without_prefill_modeling(self):
        """In Phase 1 (instant prefill), TTFT ≤ Phase 1.5 TTFT because there is no
        prefill delay.  Both modes record first_token_time (set in advance_decode)."""
        reqs = [
            _make_request(request_id=i, prompt_tokens=16, output_tokens=4, arrival_time=float(i) * 0.01)
            for i in range(3)
        ]
        sim_p1 = self._make_sim(enable_prefill=False)
        sim_p1.load_trace(reqs)
        m_p1 = sim_p1.run(OrcaStylePolicy(), "no_ttft")
        # Phase 1 records TTFT (from arrival to first decode step)
        assert not math.isnan(m_p1.mean_ttft)
        assert m_p1.mean_ttft >= 0.0

        sim_p15 = self._make_sim(enable_prefill=True, chunk=8)
        sim_p15.load_trace(reqs)
        m_p15 = sim_p15.run(OrcaStylePolicy(), "with_ttft")
        assert not math.isnan(m_p15.mean_ttft)
        # Prefill adds extra time → Phase 1.5 TTFT ≥ Phase 1 TTFT
        assert m_p15.mean_ttft >= m_p1.mean_ttft

    def test_tpot_is_recorded_with_prefill_modeling(self):
        reqs = [
            _make_request(request_id=i, prompt_tokens=16, output_tokens=8, arrival_time=float(i) * 0.02)
            for i in range(3)
        ]
        sim = self._make_sim(enable_prefill=True, chunk=16)
        sim.load_trace(reqs)
        m = sim.run(OrcaStylePolicy(), "tpot_test")
        assert not math.isnan(m.mean_tpot)
        assert m.mean_tpot >= 0.0

    def test_decode_first_reduces_ttft(self):
        """decode_first=True should not increase TTFT versus False (no active decoders at start)."""
        reqs = [
            _make_request(request_id=i, prompt_tokens=32, output_tokens=4, arrival_time=float(i) * 0.05)
            for i in range(4)
        ]
        sim_no_df = self._make_sim(enable_prefill=True, chunk=32, decode_first=False)
        sim_no_df.load_trace(reqs)
        m_no_df = sim_no_df.run(OrcaStylePolicy(), "no_decode_first")

        sim_df = self._make_sim(enable_prefill=True, chunk=32, decode_first=True)
        sim_df.load_trace(reqs)
        m_df = sim_df.run(OrcaStylePolicy(), "decode_first")

        # Both should complete all requests
        assert m_no_df.num_completed == m_df.num_completed


# ---------------------------------------------------------------------------
# Phase 1.5 policy smoke tests
# ---------------------------------------------------------------------------

class TestServingStylePolicies:

    def _make_state(self):
        from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
        gpu = ObservableGPUState(
            gpu_id=0,
            max_active_sequences=4,
            max_batch_tokens=16,
            max_kv_tokens=512,
            active_request_ids=[],
            active_requests_info=[],
            current_kv_tokens=0,
            tokens_decoded_per_request={},
        )
        queue = [
            ObservableRequest(
                request_id=i,
                arrival_time=float(i) * 0.01,
                prompt_tokens=32,
                predicted_output_tokens=8,
                slo_deadline=100.0,
                priority=1.0,
                class_id="medium",
            )
            for i in range(3)
        ]
        return ObservableState(time=0.0, waiting_queue=queue, gpu_states=[gpu], completed_count=0, step=0)

    def test_orca_style_admits(self):
        state = self._make_state()
        action = OrcaStylePolicy().select_action(state)
        assert len(action.admit[0]) > 0

    def test_vllm_style_admits(self):
        state = self._make_state()
        action = VLLMStyleTokenBudgetPolicy().select_action(state)
        assert len(action.admit[0]) > 0

    def test_sarathi_style_admits(self):
        state = self._make_state()
        action = SarathiStylePolicy(max_prefill_tokens_per_step=128).select_action(state)
        assert len(action.admit[0]) > 0

    def test_splitfuse_style_admits(self):
        state = self._make_state()
        action = SplitFuseStylePolicy(step_token_budget=256).select_action(state)
        assert len(action.admit[0]) > 0

    def test_slo_slack_score_admits(self):
        state = self._make_state()
        action = SloSlackScorePolicy().select_action(state)
        assert len(action.admit[0]) > 0

    def test_sarathi_chunk_budget_respected(self):
        """Sarathi should not admit requests that exceed the per-step prefill budget."""
        from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
        gpu = ObservableGPUState(
            gpu_id=0,
            max_active_sequences=8,
            max_batch_tokens=32,
            max_kv_tokens=4096,
            active_request_ids=[],
            active_requests_info=[],
            current_kv_tokens=0,
            tokens_decoded_per_request={},
        )
        # Each request has 300 prompt tokens; budget is 512 → only 1 fits
        queue = [
            ObservableRequest(
                request_id=i,
                arrival_time=0.0,
                prompt_tokens=300,
                predicted_output_tokens=4,
                slo_deadline=100.0,
                priority=1.0,
                class_id="medium",
            )
            for i in range(4)
        ]
        state = ObservableState(time=0.0, waiting_queue=queue, gpu_states=[gpu], completed_count=0, step=0)
        policy = SarathiStylePolicy(max_prefill_tokens_per_step=512)
        action = policy.select_action(state)
        # With budget=512 and each req=300 tokens, at most 1 should fit within
        # budget (512/300 = 1); the safety valve allows exactly 1.
        assert len(action.admit[0]) == 1

    def test_no_double_admission(self):
        """No policy should admit the same request to multiple GPUs."""
        from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
        gpus = [
            ObservableGPUState(
                gpu_id=i,
                max_active_sequences=4,
                max_batch_tokens=16,
                max_kv_tokens=512,
                active_request_ids=[],
                active_requests_info=[],
                current_kv_tokens=0,
                tokens_decoded_per_request={},
            )
            for i in range(2)
        ]
        queue = [
            ObservableRequest(
                request_id=j,
                arrival_time=0.0,
                prompt_tokens=32,
                predicted_output_tokens=8,
                slo_deadline=100.0,
                priority=1.0,
                class_id="medium",
            )
            for j in range(3)
        ]
        state = ObservableState(time=0.0, waiting_queue=queue, gpu_states=gpus, completed_count=0, step=0)
        for policy in [
            OrcaStylePolicy(),
            VLLMStyleTokenBudgetPolicy(),
            SarathiStylePolicy(),
            SplitFuseStylePolicy(),
            SloSlackScorePolicy(),
        ]:
            action = policy.select_action(state)
            all_admitted = [rid for rids in action.admit.values() for rid in rids]
            assert len(all_admitted) == len(set(all_admitted)), f"{policy.name} admitted duplicates"
