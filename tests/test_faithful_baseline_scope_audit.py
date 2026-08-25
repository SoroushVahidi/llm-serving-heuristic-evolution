"""Tests locking in the key findings of the faithful-baseline scope audit
(see docs/selector_v2_faithful_baseline_scope_audit.md).
"""
from __future__ import annotations

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.admission_control import AdmissionControlPolicy
from llmserveopt.policies.edf import EDFPolicy
from llmserveopt.policies.sarathi_faithful import DEFAULT_CHUNK_SIZE as SARATHI_DEFAULT_CHUNK
from llmserveopt.policies.vllm_chunked_prefill_faithful import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS as VLLM_CHUNKED_DEFAULT_BUDGET,
)
from llmserveopt.policies.vllm_faithful import DEFAULT_MAX_NUM_BATCHED_TOKENS as VLLM_DEFAULT_BUDGET
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


class TestAdmitChunkMaskingFinding:
    """The audit's key methodological finding: the ADMIT_CHUNK=100_000
    override used by every prior task's harness erases the real
    admission-budget distinction between vllm_faithful (all-or-nothing)
    and vllm_chunked_prefill_faithful/sarathi_faithful (chunked)."""

    def test_real_defaults_differ_between_all_or_nothing_and_chunked_policies(self):
        assert VLLM_DEFAULT_BUDGET == 2560
        assert VLLM_CHUNKED_DEFAULT_BUDGET == 512
        assert SARATHI_DEFAULT_CHUNK == 512
        assert VLLM_DEFAULT_BUDGET != VLLM_CHUNKED_DEFAULT_BUDGET

    def test_admit_chunk_override_would_mask_the_long_context_admission_difference(self):
        """A prompt longer than vllm_faithful's real budget (2560) can
        NEVER be admitted under its real default, but always can under
        vllm_chunked_prefill_faithful's chunked admission -- unless both
        are overridden to a huge shared budget (ADMIT_CHUNK), which is
        exactly what every prior task's harness did."""
        from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
        long_prompt = 6000  # > 2560
        real_default_policy = VLLMFaithfulPolicy()
        overridden_policy = VLLMFaithfulPolicy(max_num_batched_tokens=100_000)
        assert long_prompt > real_default_policy.max_num_batched_tokens
        assert long_prompt < overridden_policy.max_num_batched_tokens


class TestRejectingVsNonRejectingPolicyClassification:
    """The audit's admission-control fairness finding: only
    admission_control/scorpio_style_slo_guard voluntarily reject a
    feasible-to-admit request via a laxity filter; edf and the faithful
    policies never do -- they only ever reorder."""

    def test_edf_admits_greedily_with_no_laxity_filter(self):
        gpu = GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000, max_kv_tokens=200_000)
        reqs = [
            Request(request_id=0, arrival_time=0.0, prompt_tokens=10, predicted_output_tokens=5,
                     actual_output_tokens=5, slo_deadline=0.0001, priority=1.0, class_id="hopeless"),
        ]
        sm = ServiceModel(enable_prefill_modeling=False)
        sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=200))
        sim.load_trace(reqs)
        policy = EDFPolicy()
        policy.reset()
        m = sim.run(policy=policy, workload_tag="test", seed=0)
        # EDF admits it anyway (no laxity filter) -- it completes, even though
        # its deadline (0.0001s) was already unmeetable at arrival.
        assert m.num_completed == 1

    def test_admission_control_can_skip_a_hopeless_request(self):
        gpu = GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000, max_kv_tokens=200_000)
        reqs = [
            Request(request_id=0, arrival_time=0.0, prompt_tokens=10, predicted_output_tokens=5,
                     actual_output_tokens=5, slo_deadline=0.0, priority=1.0, class_id="hopeless"),
        ]
        sm = ServiceModel(enable_prefill_modeling=False)
        sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=200))
        sim.load_trace(reqs)
        policy = AdmissionControlPolicy(laxity_threshold=0.0)
        policy.reset()
        m = sim.run(policy=policy, workload_tag="test", seed=0)
        # Laxity is already negative at arrival (deadline == arrival_time) --
        # admission_control with laxity_threshold=0.0 must never admit it.
        assert m.num_completed == 0
        assert m.num_dropped == 1


class TestFaithfulPoliciesNeverRaise:
    """Regression lock on the audit's execution-health finding: all
    three faithful policies run cleanly (no exception) across the same
    window shapes used in the 910-window search, using both the
    ADMIT_CHUNK override and each policy's real default."""

    def _window(self):
        return [
            Request(request_id=i, arrival_time=0.0 if i < 2 else 0.002, prompt_tokens=500 + i * 300,
                     predicted_output_tokens=10, actual_output_tokens=10, slo_deadline=1.0,
                     priority=1.0, class_id="test")
            for i in range(5)
        ]

    def test_all_three_faithful_policies_run_without_exception(self):
        from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy
        from llmserveopt.policies.vllm_chunked_prefill_faithful import VLLMChunkedPrefillFaithfulPolicy
        from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy

        gpu = GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000, max_kv_tokens=200_000)
        for policy_cls, kwargs_variants in [
            (VLLMFaithfulPolicy, [{}, dict(max_num_batched_tokens=100_000)]),
            (SarathiFaithfulPolicy, [{}, dict(chunk_size=100_000)]),
            (VLLMChunkedPrefillFaithfulPolicy, [{}, dict(max_num_batched_tokens=100_000)]),
        ]:
            for kwargs in kwargs_variants:
                decode_first = policy_cls is not VLLMChunkedPrefillFaithfulPolicy
                sm = ServiceModel(enable_prefill_modeling=True, decode_first=decode_first,
                                   enable_decode_prefill_contention=True,
                                   step_token_budget=512, max_prefill_chunk_tokens=512)
                sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=2000))
                sim.load_trace(self._window())
                policy = policy_cls(**kwargs)
                policy.reset()
                m = sim.run(policy=policy, workload_tag="test", seed=0)
                assert m.num_total == 5

    def test_slai_faithful_runs_without_exception(self):
        """slai_faithful added after this audit (see
        docs/slai_faithful_scheduler_reference.md); same execution-health
        regression lock, both real default and an overridden token budget."""
        from llmserveopt.policies.slai_faithful import SlaiFaithfulPolicy

        gpu = GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000, max_kv_tokens=200_000)
        for kwargs in [{}, dict(token_budget=100_000)]:
            sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                               enable_decode_prefill_contention=True,
                               step_token_budget=512, max_prefill_chunk_tokens=512)
            sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=2000))
            sim.load_trace(self._window())
            policy = SlaiFaithfulPolicy(**kwargs)
            policy.reset()
            m = sim.run(policy=policy, workload_tag="test", seed=0)
            assert m.num_total == 5


class TestSlaiFaithfulScopeInvariants:
    """slai_faithful must follow the same faithful-external-baseline
    conventions as the other six: never selector-eligible, never in the
    historical registry, registered with a pinned commit."""

    def test_slai_faithful_not_selector_eligible(self):
        from llmserveopt.policies.external_baselines_registry import get_external_baseline_spec
        spec = get_external_baseline_spec("slai_faithful")
        assert spec.selector_eligible is False
        assert spec.historical is False

    def test_slai_faithful_not_in_historical_registry(self):
        from llmserveopt.policies.registry import BASELINE_NAMES, SELECTOR_CANDIDATE_NAMES
        assert "slai_faithful" not in BASELINE_NAMES
        assert "slai_faithful" not in SELECTOR_CANDIDATE_NAMES

    def test_slai_faithful_not_in_policy_library_v2(self):
        from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES
        assert "slai_faithful" not in POLICY_LIBRARY_V2_NAMES

    def test_slai_faithful_pinned_source_recorded(self):
        from llmserveopt.policies.external_baselines_registry import get_external_baseline_spec
        spec = get_external_baseline_spec("slai_faithful")
        assert "5098a7aba05e3edbcfa3a509d6cc9cd248fc4380" in spec.pinned_source
        assert spec.reference_doc == "docs/slai_faithful_scheduler_reference.md"
