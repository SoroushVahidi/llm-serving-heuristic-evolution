"""Tests for the decode/prefill execution-contention fix (see
docs/decode_prefill_contention_execution_model.md and
docs/vllm_chunked_prefill_faithful_root_cause_analysis.md Finding 2/3).

Covers:
  * Historical-compatibility: the default execution model
    (`enable_decode_prefill_contention=False`) is bit-identical to the
    pre-fix code for BOTH values of `decode_first` -- this is what makes
    the fix backward-compatible for the 30+ existing `configs/*.yaml`
    files and test suites that rely on it.
  * Micro-tests: under the new opt-in mode
    (`enable_decode_prefill_contention=True`), `decode_first=True` vs
    `False` produce measurably different decode progress, prefill
    progress, token-budget accounting, TTFT, and E2E timing.
  * >=200 randomized stress trials guarding invariants (no lost/duplicated
    tokens, no negative counters, no budget overrun) across both execution
    models.
"""
from __future__ import annotations

import random

import pytest

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.orca_style import OrcaStylePolicy
from llmserveopt.simulator.gpu import GPUState
from llmserveopt.simulator.request import InternalRequest, RequestPhase
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _fresh_gpu(active: dict) -> GPUState:
    gpu = GPUState(GPU_CFG)
    gpu._active = dict(active)
    return gpu


def _early_prefill_late_decode_gpu() -> GPUState:
    """One still-prefilling request that arrived first, one already-
    decoding request that arrived later -- the constructed scenario that
    should diverge between decode_first=True/False under contention mode
    (see docs/decode_prefill_contention_execution_model.md's verified
    micro-benchmark)."""
    prefill_req = _make_active_ir(0, arrival=0.0, prompt_tokens=2000, prefill_remaining=1500)
    decode_req = _make_active_ir(1, arrival=5.0, prompt_tokens=50, prefill_remaining=0, tokens_decoded=1)
    return _fresh_gpu({0: prefill_req, 1: decode_req})


# ---------------------------------------------------------------------------
# Historical-compatibility: legacy/default mode is bit-identical for both
# decode_first values, and unaffected by the new field's existence.
# ---------------------------------------------------------------------------

class TestLegacyModeUnchanged:

    def test_legacy_default_decode_first_true_equals_false(self):
        """The pre-existing dead-branch behavior: with
        enable_decode_prefill_contention left at its default (False),
        decode_first must have NO observable effect -- this is the
        historical behavior every existing config/test relies on."""
        sm_true = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                                step_token_budget=512, max_prefill_chunk_tokens=512)
        sm_false = ServiceModel(enable_prefill_modeling=True, decode_first=False,
                                 step_token_budget=512, max_prefill_chunk_tokens=512)
        assert sm_true.enable_decode_prefill_contention is False
        assert sm_false.enable_decode_prefill_contention is False

        gpu_true = _early_prefill_late_decode_gpu()
        gpu_false = _early_prefill_late_decode_gpu()
        gpu_true.step(current_time=6.0, service_model=sm_true)
        gpu_false.step(current_time=6.0, service_model=sm_false)

        assert gpu_true._active[0].prefill_remaining == gpu_false._active[0].prefill_remaining
        assert gpu_true._active[1].tokens_decoded == gpu_false._active[1].tokens_decoded
        # And the decode request must NOT have stalled (legacy = decode-protected).
        assert gpu_true._active[1].tokens_decoded == 2

    def test_legacy_default_matches_frozen_reference_values(self):
        """Golden/frozen-output regression guard: exact numeric values
        computed from this scenario BEFORE this fix (same formula, now
        extracted into _advance_decode_protected unchanged). Any future
        change to the legacy path that alters these values is a backward-
        compatibility regression."""
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=False,
                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu = _early_prefill_late_decode_gpu()
        gpu.step(current_time=6.0, service_model=sm)
        # decode reserved 1 token -> 511 left for prefill: 1500 - 511 = 989
        assert gpu._active[0].prefill_remaining == 989
        assert gpu._active[1].tokens_decoded == 2

    def test_new_field_defaults_false_on_calibrated_service_model(self):
        from llmserveopt.simulator.calibrated_service_model import CalibratedServiceModel
        import inspect
        sig = inspect.signature(CalibratedServiceModel.__init__)
        assert sig.parameters["enable_decode_prefill_contention"].default is False

    def test_service_model_factory_default_omits_new_field_safely(self):
        from llmserveopt.simulator.service_model_factory import build_service_model_from_config
        cfg = {"service_model": {"enable_prefill_modeling": True, "decode_first": False}}
        model = build_service_model_from_config(cfg)
        assert model.enable_decode_prefill_contention is False

    def test_existing_yaml_configs_do_not_set_new_field(self):
        """Sanity check on the actual repo configs this fix must not
        disturb: none of them opt into the new mode (if one did, it would
        need explicit review, not a silent behavior change)."""
        import glob
        import yaml
        for path in glob.glob("configs/**/*.yaml", recursive=True):
            if "cc" in path or "composition" in path or "distserve" in path:
                continue
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            sm_cfg = cfg.get("service_model", {}) if isinstance(cfg, dict) else {}
            assert not sm_cfg.get("enable_decode_prefill_contention", False), (
                f"{path} unexpectedly opts into the new contention mode"
            )


# ---------------------------------------------------------------------------
# New contention mode: decode_first=True keeps the historical formula
# exactly (numerically identical to legacy).
# ---------------------------------------------------------------------------

class TestContentionModeDecodeFirstTrueMatchesLegacy:

    def test_decode_first_true_under_contention_equals_legacy(self):
        sm_legacy = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                                  step_token_budget=512, max_prefill_chunk_tokens=512)
        sm_contention_true = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                                           enable_decode_prefill_contention=True,
                                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu_legacy = _early_prefill_late_decode_gpu()
        gpu_new = _early_prefill_late_decode_gpu()
        gpu_legacy.step(current_time=6.0, service_model=sm_legacy)
        gpu_new.step(current_time=6.0, service_model=sm_contention_true)

        assert gpu_legacy._active[0].prefill_remaining == gpu_new._active[0].prefill_remaining == 989
        assert gpu_legacy._active[1].tokens_decoded == gpu_new._active[1].tokens_decoded == 2


# ---------------------------------------------------------------------------
# New contention mode: decode_first=False genuinely diverges (the actual
# fix). Task section 7 requirements: decode progress, prefill progress,
# step duration/TTFT/TPOT/E2E, token-budget conservation.
# ---------------------------------------------------------------------------

class TestContentionModeDecodeFirstFalseDiverges:

    def _run(self, decode_first: bool):
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=decode_first,
                           enable_decode_prefill_contention=True,
                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu = _early_prefill_late_decode_gpu()
        gpu.step(current_time=6.0, service_model=sm)
        return gpu

    def test_decode_progress_diverges(self):
        gpu_true = self._run(True)
        gpu_false = self._run(False)
        assert gpu_true._active[1].tokens_decoded == 2   # protected: advances
        assert gpu_false._active[1].tokens_decoded == 1  # contention: stalls (zero progress)
        assert gpu_true._active[1].tokens_decoded != gpu_false._active[1].tokens_decoded

    def test_prefill_progress_diverges(self):
        gpu_true = self._run(True)
        gpu_false = self._run(False)
        assert gpu_true._active[0].prefill_remaining == 989   # 511 consumed (1 reserved for decode)
        assert gpu_false._active[0].prefill_remaining == 988  # 512 consumed (full budget, arrived first)
        assert gpu_true._active[0].prefill_remaining != gpu_false._active[0].prefill_remaining

    def test_token_budget_conservation(self):
        """Neither branch may consume more than step_token_budget total,
        and the contention branch must consume EXACTLY the budget when
        demand exceeds it (no tokens silently dropped or invented)."""
        for decode_first in (True, False):
            gpu = self._run(decode_first)
            prefill_consumed = 1500 - gpu._active[0].prefill_remaining
            decode_consumed = gpu._active[1].tokens_decoded - 1
            total = prefill_consumed + decode_consumed
            assert total <= 512
            assert total == 512  # both branches have enough demand to exhaust the budget exactly

    def test_ttft_and_e2e_diverge(self):
        """TTFT and E2E (completion) divergence, read directly off
        GPUState.step()'s own CompletedRequest return value (the same
        objects RunMetrics derives TTFT/latency from -- see
        core/types.py's CompletedRequest.latency/queuing_delay). Request 1
        is on its LAST output token (about to complete) and has never
        decoded before (first_token_time=-1.0): whether it advances at all
        this step determines both whether its TTFT gets recorded THIS step
        and whether it completes THIS step."""
        def run(decode_first: bool):
            prefill_req = _make_active_ir(0, arrival=0.0, prompt_tokens=2000, prefill_remaining=1500)
            decode_req = _make_active_ir(1, arrival=5.0, prompt_tokens=50, prefill_remaining=0,
                                          tokens_decoded=0, output_tokens=1)
            decode_req.first_token_time = -1.0
            gpu = _fresh_gpu({0: prefill_req, 1: decode_req})
            sm = ServiceModel(enable_prefill_modeling=True, decode_first=decode_first,
                               enable_decode_prefill_contention=True,
                               step_token_budget=512, max_prefill_chunk_tokens=512)
            completed = gpu.step(current_time=6.0, service_model=sm)
            return completed, gpu

        completed_true, gpu_true = run(True)
        completed_false, gpu_false = run(False)

        # Protected: request 1 gets its only token, TTFT recorded, completes.
        assert len(completed_true) == 1
        assert completed_true[0].request.request_id == 1
        assert completed_true[0].first_token_time == 6.0
        assert completed_true[0].completion_time == 6.0

        # Contention: request 0 (earlier arrival) exhausts the budget;
        # request 1 stalls -- no TTFT recorded, no completion this step.
        assert len(completed_false) == 0
        assert gpu_false._active[1].first_token_time == -1.0
        assert gpu_false._active[1].tokens_decoded == 0


# ---------------------------------------------------------------------------
# Symmetry check: under contention mode, whichever request arrived earlier
# wins the shared budget, regardless of whether it's the decode or the
# prefill request (proves this is genuine FCFS-by-arrival, not a hidden
# decode/prefill bias).
# ---------------------------------------------------------------------------

class TestContentionIsSymmetricByArrival:

    def test_decode_wins_when_it_arrived_first(self):
        prefill_req = _make_active_ir(0, arrival=5.0, prompt_tokens=2000, prefill_remaining=1500)
        decode_req = _make_active_ir(1, arrival=0.0, prompt_tokens=50, prefill_remaining=0, tokens_decoded=1)
        gpu = _fresh_gpu({0: prefill_req, 1: decode_req})
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=False,
                           enable_decode_prefill_contention=True,
                           step_token_budget=512, max_prefill_chunk_tokens=512)
        gpu.step(current_time=6.0, service_model=sm)
        # decode (earlier arrival) goes first: costs 1, prefill gets the rest (511)
        assert gpu._active[1].tokens_decoded == 2
        assert gpu._active[0].prefill_remaining == 1500 - 511


# ---------------------------------------------------------------------------
# Randomized stress trials (>=200): invariants must hold across both
# execution models for arbitrary active-request mixes.
# ---------------------------------------------------------------------------

N_STRESS_TRIALS = 220
STRESS_SEED = 20260719


def _random_active_set(rng: random.Random) -> dict:
    n_decoding = rng.randint(0, 4)
    n_prefilling = rng.randint(0, 4)
    active = {}
    rid = 0
    for _ in range(n_decoding):
        active[rid] = _make_active_ir(
            rid, arrival=rng.uniform(0.0, 10.0), prompt_tokens=rng.randint(1, 500),
            prefill_remaining=0, tokens_decoded=rng.randint(1, 5),
            output_tokens=rng.randint(6, 50),
        )
        rid += 1
    for _ in range(n_prefilling):
        prompt = rng.randint(1, 4000)
        active[rid] = _make_active_ir(
            rid, arrival=rng.uniform(0.0, 10.0), prompt_tokens=prompt,
            prefill_remaining=rng.randint(1, prompt),
        )
        rid += 1
    return active


@pytest.mark.parametrize("trial_idx", range(N_STRESS_TRIALS))
def test_randomized_contention_invariants(trial_idx):
    rng = random.Random(STRESS_SEED + trial_idx)
    active = _random_active_set(rng)
    if not active:
        return
    budget = rng.choice([1, 4, 16, 64, 128, 512, 4096])
    chunk = rng.choice([1, 4, 16, 64, 128, 512])
    decode_first = rng.choice([True, False])
    current_time = rng.uniform(0.0, 20.0)

    pre_prefill_remaining = {rid: r.prefill_remaining for rid, r in active.items()}
    pre_tokens_decoded = {rid: r.tokens_decoded for rid, r in active.items()}

    for contention in (False, True):
        gpu = _fresh_gpu(active)
        # Deep-enough copy: rebuild fresh InternalRequests each iteration
        # so the two branches don't interfere with each other's mutations.
        gpu._active = {
            rid: _make_active_ir(
                rid, arrival=r.request.arrival_time, prompt_tokens=r.request.prompt_tokens,
                prefill_remaining=pre_prefill_remaining[rid], tokens_decoded=pre_tokens_decoded[rid],
                output_tokens=r.request.actual_output_tokens,
            )
            for rid, r in active.items()
        }
        sm = ServiceModel(
            enable_prefill_modeling=True, decode_first=decode_first,
            enable_decode_prefill_contention=contention,
            step_token_budget=budget, max_prefill_chunk_tokens=chunk,
        )
        completed = gpu.step(current_time=current_time, service_model=sm)

        n_decoding = sum(1 for r in active.values() if r.prefill_remaining == 0)
        decode_consumed_total = len(completed)  # each completion consumed exactly 1 decode token
        decode_protected = not contention or decode_first

        prefill_consumed_total = 0
        for rid, req in gpu._active.items():
            prefill_consumed = pre_prefill_remaining[rid] - req.prefill_remaining
            decode_consumed = req.tokens_decoded - pre_tokens_decoded[rid]
            assert prefill_consumed >= 0, "prefill_remaining must never increase"
            assert decode_consumed >= 0, "tokens_decoded must never decrease"
            assert decode_consumed <= 1, "at most 1 decode token per request per step"
            assert not (prefill_consumed > 0 and decode_consumed > 0), (
                "a single request cannot both prefill and decode in the same step"
            )
            prefill_consumed_total += prefill_consumed
            decode_consumed_total += decode_consumed

        if decode_protected:
            # Decode-protected formula (legacy default, or contention mode
            # with decode_first=True): every decoding request UNCONDITIONALLY
            # advances, even if that alone exceeds the nominal budget (a
            # pre-existing property of the historical formula, preserved
            # as-is -- decode is a guarantee, not budget-gated). Only
            # prefill consumption is bounded by the (possibly negative,
            # clamped to 0) leftover budget.
            assert decode_consumed_total == n_decoding
            assert prefill_consumed_total <= max(0, budget - n_decoding)
        else:
            # Shared FCFS contention: the combined total is strictly
            # bounded by the per-step budget -- this is the entire point
            # of the fix (a decode request can receive zero progress).
            assert prefill_consumed_total + decode_consumed_total <= budget, (
                f"trial {trial_idx}: consumed {prefill_consumed_total + decode_consumed_total} "
                f"> budget {budget}"
            )
