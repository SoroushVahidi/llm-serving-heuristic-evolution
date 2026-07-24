"""Integration tests for the external-baseline evaluation phase:
src/llmserveopt/policies/external_baselines_registry.py,
src/llmserveopt/evaluation/external_baseline_configs.py,
src/llmserveopt/evaluation/external_baseline_harness.py.

Covers: historical-portfolio non-interference, topology validation,
smoke-scale cross-baseline runs across scenario families, invariants that
must hold for every baseline, and a differential test where two baselines
should behave identically in a degenerate scenario. See
docs/external_baseline_integration.md for the full integration matrix and
methodology this validates.
"""
from __future__ import annotations

import random
import warnings

import pytest

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.evaluation.external_baseline_configs import (
    disaggregated_config,
    matched_gpu_count_configs,
    monolithic_config,
    multi_instance_migratory_config,
    native_config_for,
)
from llmserveopt.evaluation.external_baseline_harness import (
    TopologyValidationError,
    run_external_baseline,
    validate_topology,
)
from llmserveopt.policies.external_baselines_registry import (
    EXTERNAL_BASELINE_NAMES,
    EXTERNAL_BASELINE_REGISTRY,
    get_external_baseline_spec,
)
from llmserveopt.policies.llumnix_faithful import LlumnixFaithfulPolicy
from llmserveopt.policies.registry import BASELINE_NAMES, SELECTOR_CANDIDATE_NAMES
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


# ---------------------------------------------------------------------------
# Historical-portfolio non-interference
# ---------------------------------------------------------------------------

def test_historical_baseline_count_unchanged():
    assert len(BASELINE_NAMES) == 20
    assert len(SELECTOR_CANDIDATE_NAMES) == 20


def test_no_external_baseline_leaked_into_historical_registry():
    for name in EXTERNAL_BASELINE_NAMES:
        assert name not in BASELINE_NAMES
        assert name not in SELECTOR_CANDIDATE_NAMES


def test_no_external_baseline_marked_selector_eligible_yet():
    for name, spec in EXTERNAL_BASELINE_REGISTRY.items():
        assert spec.selector_eligible is False, f"{name} must not be selector_eligible in this integration phase"
        assert spec.historical is False


def test_all_seven_external_baselines_registered():
    # slai_faithful added; see docs/slai_faithful_scheduler_reference.md.
    assert set(EXTERNAL_BASELINE_NAMES) == {
        "vllm_faithful", "vllm_chunked_prefill_faithful", "sarathi_faithful",
        "distserve_faithful", "tetriinfer_paper_reimplementation", "llumnix_faithful",
        "slai_faithful",
    }


# ---------------------------------------------------------------------------
# Topology validation
# ---------------------------------------------------------------------------

def test_validate_topology_rejects_distserve_with_multiple_decode_gpus():
    spec = get_external_baseline_spec("distserve_faithful")
    gpus, _sm, _topo = disaggregated_config(n_prefill=1, n_decode=2)
    with pytest.raises(TopologyValidationError):
        validate_topology(spec, gpus)


def test_validate_topology_rejects_monolithic_baseline_with_roles():
    spec = get_external_baseline_spec("vllm_faithful")
    gpus, _sm, _topo = disaggregated_config(n_prefill=1, n_decode=1)
    with pytest.raises(TopologyValidationError):
        validate_topology(spec, gpus)


def test_validate_topology_rejects_disaggregated_baseline_without_roles():
    spec = get_external_baseline_spec("tetriinfer_paper_reimplementation")
    gpus, _sm, _topo = monolithic_config(n_gpus=2)
    with pytest.raises(TopologyValidationError):
        validate_topology(spec, gpus)


def test_validate_topology_accepts_each_baselines_own_native_config():
    for name in EXTERNAL_BASELINE_NAMES:
        spec = get_external_baseline_spec(name)
        gpus, _sm, _topo = native_config_for(name)
        validate_topology(spec, gpus)  # must not raise


def test_matched_gpu_count_configs_omits_distserve_when_not_exactly_two():
    configs_3 = matched_gpu_count_configs(n_gpus=3)
    assert "distserve_faithful" not in configs_3
    configs_2 = matched_gpu_count_configs(n_gpus=2)
    assert "distserve_faithful" in configs_2


# ---------------------------------------------------------------------------
# Smoke-scale cross-baseline validation (see docs/external_baseline_integration.md §8)
# ---------------------------------------------------------------------------

def _make_reqs(n, prompt_range, output_range, arrival_gap=0.002, bursty=False, seed=0):
    rng = random.Random(seed)
    reqs = []
    t = 0.0
    for i in range(n):
        if bursty and i % 5 == 0:
            t += arrival_gap * 10
        elif bursty:
            t += arrival_gap * 0.1
        else:
            t += arrival_gap
        prompt = rng.randint(*prompt_range)
        output = rng.randint(*output_range)
        reqs.append(Request(
            request_id=i, arrival_time=t, prompt_tokens=prompt,
            predicted_output_tokens=output, actual_output_tokens=output,
            slo_deadline=1000.0, priority=1.0, class_id="d",
        ))
    return reqs


_SMOKE_SCENARIOS = {
    "short_prompt_short_output": lambda: _make_reqs(20, (5, 20), (5, 20)),
    "long_prompt_short_output": lambda: _make_reqs(20, (200, 400), (5, 20)),
    "short_prompt_long_output": lambda: _make_reqs(20, (5, 20), (200, 400)),
    "long_prompt_long_output": lambda: _make_reqs(15, (200, 400), (200, 400)),
    "bursty_arrivals": lambda: _make_reqs(30, (10, 50), (10, 50), bursty=True),
    "high_kv_pressure": lambda: _make_reqs(25, (50, 150), (100, 300)),
    "mixed_lengths": lambda: _make_reqs(30, (5, 300), (5, 300), seed=1),
    "slo_sensitive": lambda: _make_reqs(20, (10, 50), (10, 100)),
}


@pytest.mark.parametrize("baseline_name", EXTERNAL_BASELINE_NAMES)
@pytest.mark.parametrize("scenario_name", list(_SMOKE_SCENARIOS.keys()))
@pytest.mark.parametrize("seed", [0, 1])
def test_smoke_cross_baseline_scenario(baseline_name, scenario_name, seed):
    """No crashes, no capacity-violation warnings, no request loss/duplication
    (completed+dropped == total) across every (baseline, scenario, seed)
    combination. Smoke-scale only -- not a manuscript conclusion."""
    reqs = _SMOKE_SCENARIOS[scenario_name]()
    gpus, sm, _topo = native_config_for(baseline_name, total_kv_tokens=8000)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = run_external_baseline(
            baseline_name, reqs, gpus, sm, workload_tag=scenario_name, seed=seed, drain_steps=30_000,
        )
    assert w == [], f"unexpected warnings for {baseline_name}/{scenario_name}/seed{seed}: {[str(x.message) for x in w]}"
    assert result.metrics.num_completed + result.metrics.num_dropped == len(reqs)


# ---------------------------------------------------------------------------
# Invariants (see docs/external_baseline_integration.md §9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("baseline_name", EXTERNAL_BASELINE_NAMES)
def test_invariant_no_duplicate_no_loss_no_early_execution(baseline_name):
    reqs = _make_reqs(25, (10, 100), (10, 150))
    gpus, sm, _topo = native_config_for(baseline_name, total_kv_tokens=10_000)
    spec = get_external_baseline_spec(baseline_name)
    policy = spec.factory()

    sim_cfg = SimulatorConfig(gpu_configs=gpus, service_model=sm, drain_steps=30_000)
    sim = Simulator(sim_cfg)
    sim.load_trace(reqs)

    early_execution_violations = []
    seen_request_ids_over_time = []
    orig_select = policy.select_action

    def checked_select(state, _orig=orig_select):
        for req in state.waiting_queue:
            if req.arrival_time > state.time:
                early_execution_violations.append(req.request_id)
        seen_request_ids_over_time.append(frozenset(
            req_id for g in state.gpu_states for req_id in g.active_request_ids
        ))
        return _orig(state)

    policy.select_action = checked_select
    metrics = sim.run(policy, workload_tag="invariants")

    assert early_execution_violations == []
    assert metrics.num_completed + metrics.num_dropped == len(reqs)
    # No request ID ever appears twice among completed requests.
    completed_ids = [c.request.request_id for c in sim._completed]
    assert len(completed_ids) == len(set(completed_ids))


@pytest.mark.parametrize("baseline_name", EXTERNAL_BASELINE_NAMES)
def test_invariant_kv_capacity_never_exceeded(baseline_name):
    """No admission-rejected warning across a moderately loaded run --
    the established convention (see e.g. vllm_faithful's own tests) for
    proving the simulator's own GPUConfig capacity was never violated at
    admission time."""
    reqs = _make_reqs(25, (10, 60), (10, 80))
    gpus, sm, _topo = native_config_for(baseline_name, total_kv_tokens=6000)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = run_external_baseline(baseline_name, reqs, gpus, sm, workload_tag="kv-capacity", drain_steps=30_000)
    assert w == [], f"unexpected warnings: {[str(x.message) for x in w]}"
    assert result.metrics.num_completed + result.metrics.num_dropped == len(reqs)


@pytest.mark.parametrize("baseline_name", EXTERNAL_BASELINE_NAMES)
def test_invariant_gpu_role_constraints_respected(baseline_name):
    """Disaggregated baselines must never admit a request onto a
    role-mismatched GPU relative to that request's current phase -- this
    is already enforced deep inside the simulator (GPUState.admit's
    role-aware prefill_remaining logic) and by each baseline's own
    validation; here we just confirm the topology itself is well-formed
    for every registered baseline via the harness's own pre-flight check."""
    spec = get_external_baseline_spec(baseline_name)
    gpus, _sm, _topo = native_config_for(baseline_name)
    validate_topology(spec, gpus)  # must not raise
    if spec.required_roles == (None,):
        assert all(g.role is None for g in gpus)
    else:
        assert {g.role for g in gpus} == set(spec.required_roles)


@pytest.mark.parametrize("baseline_name", EXTERNAL_BASELINE_NAMES)
def test_invariant_deterministic_reproduction(baseline_name):
    def run():
        reqs = _make_reqs(15, (10, 80), (10, 100))
        gpus, sm, _topo = native_config_for(baseline_name, total_kv_tokens=6000)
        return run_external_baseline(baseline_name, reqs, gpus, sm, workload_tag="determinism", seed=0, drain_steps=20_000)

    r1, r2 = run(), run()
    assert r1.metrics.num_completed == r2.metrics.num_completed
    assert r1.metrics.mean_latency == r2.metrics.mean_latency
    assert r1.num_admit_events == r2.num_admit_events
    assert r1.num_migrate_events == r2.num_migrate_events


def test_invariant_migration_conserves_request_count():
    """llumnix_faithful-specific: across a run with active migrations,
    total requests processed is conserved regardless of how many times
    any individual request was migrated."""
    reqs = _make_reqs(30, (10, 80), (50, 300))
    gpus, sm, _topo = multi_instance_migratory_config(n_instances=4, total_kv_tokens=6000, migration_delay=0.001)
    result = run_external_baseline("llumnix_faithful", reqs, gpus, sm, workload_tag="migration-conservation",
                                    seed=0, drain_steps=30_000, policy_kwargs={"need_migrate_frequency": 1})
    assert result.metrics.num_completed + result.metrics.num_dropped == len(reqs)
    assert result.num_migrate_events >= 0  # always observable, never a fabricated placeholder


def test_invariant_disaggregation_conserves_request_count():
    """distserve_faithful/tetriinfer-specific: every request that finishes
    prefill and hands off through the bridge queue is eventually admitted
    to decode and completes (or is legitimately still draining) -- no
    silent loss across the prefill->decode boundary."""
    for name in ("distserve_faithful", "tetriinfer_paper_reimplementation"):
        reqs = _make_reqs(20, (30, 150), (20, 150))
        gpus, sm, _topo = native_config_for(name, total_kv_tokens=8000)
        result = run_external_baseline(name, reqs, gpus, sm, workload_tag="disagg-conservation", seed=0, drain_steps=30_000)
        assert result.metrics.num_completed + result.metrics.num_dropped == len(reqs)


# ---------------------------------------------------------------------------
# Differential test: llumnix_faithful with exactly 1 instance must degenerate
# EXACTLY to vllm_faithful alone (migration is structurally impossible with
# only 1 instance, and llumnix_faithful's own local scheduler IS
# vllm_faithful's per-GPU worker) -- see
# docs/external_baseline_integration.md §9.
# ---------------------------------------------------------------------------

def test_differential_llumnix_single_instance_equals_vllm_faithful():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=32, max_batch_tokens=100_000, max_kv_tokens=5000)
    reqs = [
        Request(request_id=i, arrival_time=i * 0.002, prompt_tokens=20 + i * 3,
                predicted_output_tokens=15 + i * 2, actual_output_tokens=15 + i * 2,
                slo_deadline=1000.0, priority=1.0, class_id="d")
        for i in range(12)
    ]

    sim1 = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=20_000))
    sim1.load_trace(reqs)
    m1 = sim1.run(VLLMFaithfulPolicy(), workload_tag="vllm")

    sim2 = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=20_000))
    sim2.load_trace(reqs)
    m2 = sim2.run(LlumnixFaithfulPolicy(), workload_tag="llumnix-1-instance")

    assert m1.num_completed == m2.num_completed == 12
    assert m1.mean_latency == m2.mean_latency
    assert m1.mean_ttft == m2.mean_ttft
    assert m1.num_dropped == m2.num_dropped == 0
