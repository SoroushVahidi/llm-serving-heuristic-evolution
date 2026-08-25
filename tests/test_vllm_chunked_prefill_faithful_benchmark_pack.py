"""Benchmark-pack acceptance tests for vllm_chunked_prefill_faithful.

Loads the canonical, checksummed real-A100-hardware benchmark pack
(experiments/runtime_validation_benchmark_pack/, see
docs/runtime_validation_benchmark_pack.md) and runs vllm_faithful,
sarathi_faithful, and vllm_chunked_prefill_faithful against all six
scenarios (the five request-level scenarios plus the long-context
xlong_context_burst16 fixture).

Per task instruction, this does NOT hard-code an expectation that every
simulated ranking must match the real-hardware winner -- it classifies each
scenario's outcome (MATCHES_REAL_WINNER / MISMATCHES_REAL_WINNER /
TIE_NEAR_TIE / STRUCTURALLY_UNREPRESENTABLE) and asserts the classification
itself, which is honest about where this baseline does and does not close
the real-hardware gap. See docs/vllm_chunked_prefill_faithful_root_cause_
analysis.md for the full investigation this file's fixed-point numbers were
originally derived from, and
docs/decode_prefill_contention_execution_model.md for the follow-up fix to
Finding 2/3 (the `decode_first` dead branch) revalidated here.

All three policies are evaluated under Phase-1.5 (`enable_prefill_modeling=
True`, matching sarathi_faithful's own evaluation config) -- NOT the
pre-existing, asymmetric scripts/run_gpu_external_validity_audit.py
convention of giving vllm_faithful a zero-prefill-cost ServiceModel() while
sarathi_faithful gets a real one. See the root-cause doc's Finding 1 for why
that asymmetry would make any comparison here meaningless.

Per-policy execution semantics (post decode_first fix)
--------------------------------------------------------
Each policy now opts into `enable_decode_prefill_contention=True` with the
`decode_first` value matching its own pinned reference's real execution
semantics (previously this flag was dead code -- see the doc above):
  * `sarathi_faithful`: `decode_first=True` (Sarathi's own genuine
    decode-protected stall-free guarantee).
  * `vllm_chunked_prefill_faithful`: `decode_first=False` (vLLM v0.4.2's
    genuine shared-FCFS-by-arrival contention, no decode-priority phase).
  * `vllm_faithful`: `decode_first=True` -- it has no chunked/continuing-
    prefill scheduling model of its own to exercise the contention path
    meaningfully, so it is left on the decode-protected formula (numerically
    identical to how it was evaluated before this fix).

Result (revalidated live below, not just asserted): all five request-level
scenarios still classify as TIE_NEAR_TIE, but for a materially different,
now load-bearing reason. Before this fix, `decode_first` was dead code, so
the tie was mechanical. After the fix, decode_first genuinely diverges
execution (see tests/test_decode_prefill_contention_execution.py for a
constructed scenario that DOES diverge) -- but in all five of these
specific fixtures, every already-decoding request happens to have arrived
no later than any competing still-prefilling request (verified against each
scenario's own `arrival_time_s` fixture data), so FCFS-by-arrival gives
decode priority "for free" regardless of `decode_first`. This is a genuine
workload-construction fact about these six fixtures, not a simulator
limitation -- see docs/decode_prefill_contention_execution_model.md's
final-report addendum for the per-scenario arrival-order audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.external_baselines_registry import make_external_baseline
from llmserveopt.simulator.service_model import ServiceModel

PACK_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "runtime_validation_benchmark_pack"

REQUEST_LEVEL_SCENARIOS = [
    "long_prompt_moderate_output",
    "active_decode_plus_arriving_prefill",
    "prefill_heavy_burst",
    "mixed_prompt_lengths",
    "kv_pressure",
]

POSITIVE_TARGETS = {"active_decode_plus_arriving_prefill", "kv_pressure"}
NEGATIVE_CONTROLS = {"long_prompt_moderate_output", "prefill_heavy_burst", "mixed_prompt_lengths"}

_COMMON = dict(enable_prefill_modeling=True, enable_decode_prefill_contention=True,
               step_token_budget=512, max_prefill_chunk_tokens=512)
SERVICE_MODELS = {
    "vllm_faithful": ServiceModel(decode_first=True, **_COMMON),
    "sarathi_faithful": ServiceModel(decode_first=True, **_COMMON),
    "vllm_chunked_prefill_faithful": ServiceModel(decode_first=False, **_COMMON),
}
GPU_CONFIGS = [GPUConfig(0, max_active_sequences=256, max_batch_tokens=2560, max_kv_tokens=131_072)]

POLICY_NAMES = ["vllm_faithful", "sarathi_faithful", "vllm_chunked_prefill_faithful"]


# ---------------------------------------------------------------------------
# Pack integrity (manifest sha256 check -- see docs/runtime_validation_
# benchmark_pack.md's own "Regeneration and verification" section)
# ---------------------------------------------------------------------------

def test_pack_manifest_integrity():
    manifest = json.loads((PACK_ROOT / "manifest.json").read_text())
    for f in manifest["files"]:
        actual = hashlib.sha256((PACK_ROOT / f["path"]).read_bytes()).hexdigest()
        assert actual == f["sha256"], f"checksum mismatch: {f['path']}"


def test_pack_declares_expected_positive_targets_and_negative_controls():
    """Labels must not be altered to make this baseline look better --
    check they match what's actually recorded in the pack (task instruction)."""
    manifest = json.loads((PACK_ROOT / "manifest.json").read_text())
    assert set(manifest["positive_targets"]) == POSITIVE_TARGETS
    assert set(manifest["negative_controls"]) == NEGATIVE_CONTROLS


# ---------------------------------------------------------------------------
# Load scenario fixtures into simulator Requests
# ---------------------------------------------------------------------------

def _load_requests(scenario_id: str) -> list:
    data = json.loads((PACK_ROOT / "scenarios" / f"{scenario_id}.json").read_text())
    return [
        Request(
            request_id=r["request_id"],
            arrival_time=r["arrival_time_s"],
            prompt_tokens=max(1, int(r["intended_prompt_tokens_approx"])),
            predicted_output_tokens=r["target_output_tokens"],
            actual_output_tokens=r["target_output_tokens"],
            slo_deadline=r["arrival_time_s"] + 10_000.0,
            priority=1.0,
            class_id="benchmark_pack",
        )
        for r in data["requests"]
    ], data


def _run_all_policies(scenario_id: str) -> dict:
    reqs, _data = _load_requests(scenario_id)
    out = {}
    for name in POLICY_NAMES:
        policy = make_external_baseline(name)
        metrics = run_policy(
            policy=policy, requests=reqs, gpu_configs=GPU_CONFIGS,
            service_model=SERVICE_MODELS[name], workload_tag=scenario_id, seed=20260719,
        )
        out[name] = metrics
    return out


# ---------------------------------------------------------------------------
# Structural correctness (task requirement, checked BEFORE any ranking
# comparison): no request loss, deterministic completion, no token-budget
# violation, no duplicate execution. Capacity/budget-violation invariants
# are checked exhaustively by the 200+-trial stress suite
# (tests/test_vllm_chunked_prefill_faithful_stress.py); here we check the
# outcome-level facts specific to these six real fixtures.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_id", REQUEST_LEVEL_SCENARIOS)
def test_structural_no_request_loss_all_policies(scenario_id):
    reqs, data = _load_requests(scenario_id)
    results = _run_all_policies(scenario_id)
    for name, metrics in results.items():
        assert metrics.num_completed + metrics.num_dropped == len(reqs), (
            f"{name}/{scenario_id}: {metrics.num_completed} completed + "
            f"{metrics.num_dropped} dropped != {len(reqs)} total requests"
        )
        assert metrics.num_dropped == 0, f"{name}/{scenario_id}: unexpected drops"
        assert metrics.completion_fraction == 1.0


@pytest.mark.parametrize("scenario_id", REQUEST_LEVEL_SCENARIOS)
def test_structural_deterministic_repeated_runs(scenario_id):
    reqs, _data = _load_requests(scenario_id)
    policy = make_external_baseline("vllm_chunked_prefill_faithful")
    m1 = run_policy(policy=policy, requests=reqs, gpu_configs=GPU_CONFIGS,
                     service_model=SERVICE_MODELS["vllm_chunked_prefill_faithful"], seed=20260719)
    policy2 = make_external_baseline("vllm_chunked_prefill_faithful")
    m2 = run_policy(policy=policy2, requests=reqs, gpu_configs=GPU_CONFIGS,
                     service_model=SERVICE_MODELS["vllm_chunked_prefill_faithful"], seed=20260719)
    assert m1.num_completed == m2.num_completed
    assert m1.mean_latency == m2.mean_latency
    assert m1.mean_ttft == m2.mean_ttft


# ---------------------------------------------------------------------------
# Winner-direction classification (honest -- see module docstring). Derived
# empirically and recorded in docs/vllm_chunked_prefill_faithful_root_cause_
# analysis.md; re-verified live here rather than only asserted in prose.
# ---------------------------------------------------------------------------

def _classify(vllm_e2e: float, sarathi_e2e: float, positive_target: bool) -> str:
    """positive_target=True means real hardware wants sarathi to win
    (lower E2E); False means real hardware wants vllm to win."""
    if vllm_e2e == pytest.approx(sarathi_e2e, rel=1e-9, abs=1e-9):
        return "TIE_NEAR_TIE"
    sarathi_wins = sarathi_e2e < vllm_e2e
    real_wants_sarathi = positive_target
    return "MATCHES_REAL_WINNER" if sarathi_wins == real_wants_sarathi else "MISMATCHES_REAL_WINNER"


@pytest.mark.parametrize("scenario_id", REQUEST_LEVEL_SCENARIOS)
def test_fair_comparison_classification(scenario_id):
    """Under a FAIR (equal-ServiceModel) comparison, all five canonical
    request-level scenarios tie exactly at this request scale for all
    three policies -- see root-cause doc's Addendum. This is the honest,
    reproducible baseline this test pins down; it is a TIE, not a false
    negative-control pass, unlike the pre-existing asymmetric-config
    numbers recorded in experiments/runtime_validation_benchmark_pack/
    simulator_baseline_results/."""
    results = _run_all_policies(scenario_id)
    vllm_e2e = results["vllm_faithful"].mean_latency
    sarathi_e2e = results["sarathi_faithful"].mean_latency
    chunked_e2e = results["vllm_chunked_prefill_faithful"].mean_latency
    positive_target = scenario_id in POSITIVE_TARGETS

    classification_vllm_faithful = _classify(vllm_e2e, sarathi_e2e, positive_target)
    classification_chunked = _classify(chunked_e2e, sarathi_e2e, positive_target)

    assert classification_vllm_faithful == "TIE_NEAR_TIE", (
        f"{scenario_id}: expected vllm_faithful/sarathi_faithful to tie "
        f"under a fair ServiceModel at this request scale, got "
        f"{classification_vllm_faithful} ({vllm_e2e} vs {sarathi_e2e})"
    )
    assert classification_chunked == "TIE_NEAR_TIE", (
        f"{scenario_id}: expected vllm_chunked_prefill_faithful/"
        f"sarathi_faithful to tie under a fair ServiceModel at this "
        f"request scale (see root-cause doc Finding 3), got "
        f"{classification_chunked} ({chunked_e2e} vs {sarathi_e2e})"
    )


def test_negative_controls_do_not_show_false_sarathi_advantage():
    """Explicit negative-control requirement (task instruction): the new
    baseline must not create a FALSE sarathi win on the three robust
    real-vLLM-wins scenarios. A tie is acceptable (see above); a reversal
    to a sarathi win would not be."""
    for scenario_id in NEGATIVE_CONTROLS:
        results = _run_all_policies(scenario_id)
        chunked_e2e = results["vllm_chunked_prefill_faithful"].mean_latency
        sarathi_e2e = results["sarathi_faithful"].mean_latency
        assert chunked_e2e <= sarathi_e2e + 1e-9, (
            f"{scenario_id}: vllm_chunked_prefill_faithful ({chunked_e2e}) "
            f"lost to sarathi_faithful ({sarathi_e2e}) -- a FALSE sarathi "
            f"advantage on a negative-control scenario"
        )


# ---------------------------------------------------------------------------
# Long-context fixture: this baseline's direct acceptance-test target
# (task instruction: "the direct negative-image of the long-context-drop
# finding"). STRUCTURALLY differentiated, not a tie -- vllm_faithful is
# STRUCTURALLY_UNREPRESENTABLE-equivalent here (it cannot admit these
# requests at all by construction of its v0.1.0 pin), while
# vllm_chunked_prefill_faithful MATCHES_REAL_WINNER (real vLLM completes
# all 16; historical vllm_faithful completes 0; this baseline must
# complete all 16 too).
# ---------------------------------------------------------------------------

def test_long_context_historical_vllm_faithful_completes_zero():
    data = json.loads((PACK_ROOT / "long_context" / "xlong_context_burst16.json").read_text())
    assert data["vllm_faithful_simulator_result"]["completion_fraction"] == 0.0

    reqs = [
        Request(
            request_id=r["request_id"], arrival_time=r["arrival_time_s"],
            prompt_tokens=r["target_input_tokens"], predicted_output_tokens=r["target_output_tokens"],
            actual_output_tokens=r["target_output_tokens"], slo_deadline=r["arrival_time_s"] + 10_000.0,
            priority=1.0, class_id="xlong",
        )
        for r in data["requests"]
    ]
    gpu_configs = [GPUConfig(0, max_active_sequences=256, max_batch_tokens=2560, max_kv_tokens=300_000)]
    policy = make_external_baseline("vllm_faithful")
    metrics = run_policy(policy=policy, requests=reqs, gpu_configs=gpu_configs,
                          service_model=SERVICE_MODELS["vllm_faithful"], seed=20260719, drain_steps=5000)
    assert metrics.completion_fraction == 0.0
    assert metrics.num_dropped == len(reqs)


def test_long_context_new_baseline_completes_all_16_via_chunked_admission():
    """This baseline's direct acceptance-test target (task instruction):
    real vLLM 0.24.0 completes all 16 via chunked admission;
    vllm_chunked_prefill_faithful must too, unlike historical vllm_faithful."""
    data = json.loads((PACK_ROOT / "long_context" / "xlong_context_burst16.json").read_text())
    assert data["real_vllm_result"]["completion_fraction"] == 1.0

    reqs = [
        Request(
            request_id=r["request_id"], arrival_time=r["arrival_time_s"],
            prompt_tokens=r["target_input_tokens"], predicted_output_tokens=r["target_output_tokens"],
            actual_output_tokens=r["target_output_tokens"], slo_deadline=r["arrival_time_s"] + 10_000.0,
            priority=1.0, class_id="xlong",
        )
        for r in data["requests"]
    ]
    gpu_configs = [GPUConfig(0, max_active_sequences=256, max_batch_tokens=2560, max_kv_tokens=300_000)]
    policy = make_external_baseline("vllm_chunked_prefill_faithful")
    metrics = run_policy(policy=policy, requests=reqs, gpu_configs=gpu_configs,
                          service_model=SERVICE_MODELS["vllm_chunked_prefill_faithful"],
                          seed=20260719, drain_steps=5000)
    assert metrics.completion_fraction == 1.0
    assert metrics.num_completed == 16
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# Summary classification table (task instruction: "Record before/after
# changes relative to historical vllm_faithful"). Asserted as data, not
# just prose -- this IS the acceptance record.
# ---------------------------------------------------------------------------

BENCHMARK_PACK_CLASSIFICATION = {
    "long_prompt_moderate_output": "TIE_NEAR_TIE",
    "active_decode_plus_arriving_prefill": "TIE_NEAR_TIE",
    "prefill_heavy_burst": "TIE_NEAR_TIE",
    "mixed_prompt_lengths": "TIE_NEAR_TIE",
    "kv_pressure": "TIE_NEAR_TIE",
    "xlong_context_burst16": "MATCHES_REAL_WINNER",  # completion_fraction, not E2E ranking
}


def test_classification_table_matches_live_reruns():
    for scenario_id in REQUEST_LEVEL_SCENARIOS:
        results = _run_all_policies(scenario_id)
        chunked_e2e = results["vllm_chunked_prefill_faithful"].mean_latency
        sarathi_e2e = results["sarathi_faithful"].mean_latency
        positive_target = scenario_id in POSITIVE_TARGETS
        assert _classify(chunked_e2e, sarathi_e2e, positive_target) == BENCHMARK_PACK_CLASSIFICATION[scenario_id]
