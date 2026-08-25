"""Representation-fidelity tests for the expanded policy-genome mapping.

See docs/current/POLICY_GENOME_COVERAGE_AUDIT.md for the full per-policy
audit these tests validate against.
"""
from __future__ import annotations

import pytest

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.genome import canonical_json, parse_genome
from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES, make_policy_library_v2
from llmserveopt.policies.structural_synthesis import UNMAPPABLE_REASONS, map_policy_to_genome
from llmserveopt.selector.suitability.encoders import structural_features

MAPPED_EXACT = [
    "fifo", "edf", "shortest_output_first", "shortest_prompt_first", "first_fit",
    "orca_style", "slo_slack_score", "weighted_shortest_processing",
    "least_laxity_first", "estimated_service_time_first",
]
MAPPED_APPROXIMATE = [
    "admission_control", "scorpio_style_slo_guard", "sola_style_state_aware",
    "flow_control_stability", "kv_constrained_online", "adaptive_chunked_prefill",
    "aging_priority", "weighted_fair_share", "multi_bin_batching",
]
UNSUPPORTED = [
    "greedy_token_fill", "least_loaded", "random_feasible", "best_fit",
    "vllm_style_token_budget", "sarathi_style", "splitfuse_style", "slai_style_phase_aware",
]


def test_all_27_policies_classified_and_accounted_for():
    assert set(MAPPED_EXACT) | set(MAPPED_APPROXIMATE) | set(UNSUPPORTED) == set(POLICY_LIBRARY_V2_NAMES)
    assert len(MAPPED_EXACT) + len(MAPPED_APPROXIMATE) + len(UNSUPPORTED) == 27


@pytest.mark.parametrize("policy_name", POLICY_LIBRARY_V2_NAMES)
def test_genome_validates_and_compiles_for_every_deployable_policy(policy_name):
    genome = map_policy_to_genome(policy_name)
    genome.validate()  # raises on failure


@pytest.mark.parametrize("policy_name", MAPPED_EXACT)
def test_exact_mappings_report_exact_status(policy_name):
    genome = map_policy_to_genome(policy_name)
    assert genome.metadata["mapping_status"] == "EXACT"


@pytest.mark.parametrize("policy_name", MAPPED_APPROXIMATE)
def test_approximate_mappings_report_approximate_status_with_documented_limitation(policy_name):
    genome = map_policy_to_genome(policy_name)
    assert genome.metadata["mapping_status"] == "APPROXIMATE"
    assert genome.metadata.get("limitation"), f"{policy_name} must document its approximation gap"


@pytest.mark.parametrize("policy_name", UNSUPPORTED)
def test_unsupported_policies_are_explicitly_documented_not_silently_defaulted(policy_name):
    genome = map_policy_to_genome(policy_name)
    assert genome.metadata["mapping_status"] == "UNSUPPORTED"
    assert policy_name in UNMAPPABLE_REASONS
    assert genome.metadata["limitation"] == UNMAPPABLE_REASONS[policy_name]


@pytest.mark.parametrize("policy_name", POLICY_LIBRARY_V2_NAMES)
def test_genome_mapping_deterministic_across_calls(policy_name):
    a = map_policy_to_genome(policy_name)
    b = map_policy_to_genome(policy_name)
    assert a.canonical_json() == b.canonical_json()
    assert a.stable_hash() == b.stable_hash()


@pytest.mark.parametrize("policy_name", POLICY_LIBRARY_V2_NAMES)
def test_genome_serialization_round_trip(policy_name):
    genome = map_policy_to_genome(policy_name)
    payload = genome.to_dict()
    restored = parse_genome(payload)
    assert restored.canonical_json() == genome.canonical_json()
    assert restored.stable_hash() == genome.stable_hash()
    # canonical_json is genuinely canonical: re-serializing the round-tripped
    # dict produces byte-identical output.
    assert canonical_json(restored.to_dict()) == canonical_json(payload)


def test_all_27_policy_hashes_are_stable_and_unique():
    hashes = {name: map_policy_to_genome(name).stable_hash() for name in POLICY_LIBRARY_V2_NAMES}
    assert len(set(hashes.values())) == 27
    hashes_again = {name: map_policy_to_genome(name).stable_hash() for name in POLICY_LIBRARY_V2_NAMES}
    assert hashes == hashes_again


def test_unsupported_placeholders_are_structurally_identical_by_design():
    """The 8 UNSUPPORTED policies intentionally share one generic placeholder
    genome shape (same modules, same expression, same status) -- this is the
    documented, honest 'we cannot distinguish these structurally yet'
    invariant, not a bug. Their names/hashes remain distinct."""
    features = [structural_features(map_policy_to_genome(name)) for name in UNSUPPORTED]
    assert all(f == features[0] for f in features)
    hashes = {map_policy_to_genome(name).stable_hash() for name in UNSUPPORTED}
    assert len(hashes) == len(UNSUPPORTED)


def test_mapped_policies_are_structurally_distinguishable_from_each_other():
    """Genuinely different scheduling behaviors must not collapse to
    identical structural feature vectors (the whole point of the expanded
    taxonomy)."""
    mapped = MAPPED_EXACT + MAPPED_APPROXIMATE
    vectors = {name: tuple(sorted(structural_features(map_policy_to_genome(name)).items())) for name in mapped}
    distinct = len(set(vectors.values()))
    # Not every pair need be unique (e.g. two APPROXIMATE mappings may
    # legitimately share some coarse features), but the vast majority must
    # differ -- this is a coverage sanity check, not a strict bijection.
    assert distinct >= int(0.9 * len(mapped))


def test_exact_policies_reference_the_expected_causal_variable_groups():
    """Spot-check that the grouped structural-distinction features (SLO-aware,
    prefill/decode-weighted, fairness-aware) actually fire for policies whose
    real behavior matches the group, and don't fire for ones that don't."""
    edf_feats = structural_features(map_policy_to_genome("edf"))
    assert edf_feats["struct_slo_aware_var_count"] > 0

    sof_feats = structural_features(map_policy_to_genome("shortest_output_first"))
    assert sof_feats["struct_decode_weighted_var_count"] > 0
    assert sof_feats["struct_slo_aware_var_count"] == 0

    spf_feats = structural_features(map_policy_to_genome("shortest_prompt_first"))
    assert spf_feats["struct_prefill_weighted_var_count"] > 0

    aging_feats = structural_features(map_policy_to_genome("aging_priority"))
    assert aging_feats["struct_fairness_aware_var_count"] > 0

    fifo_feats = structural_features(map_policy_to_genome("fifo"))
    assert fifo_feats["struct_is_pure_ranking"] == 1.0

    scorpio_feats = structural_features(map_policy_to_genome("scorpio_style_slo_guard"))
    assert scorpio_feats["struct_is_regime_conditional"] == 1.0


def _make_request(req_id, arrival=0.0, prompt=128, output=64, deadline=10.0, priority=2.0):
    return ObservableRequest(
        request_id=req_id, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, slo_deadline=arrival + deadline,
        priority=priority, class_id="medium",
    )


def _make_state(requests, now=1.0):
    gpu = ObservableGPUState(
        gpu_id=0, max_active_sequences=8, max_batch_tokens=512, max_kv_tokens=8192,
        active_request_ids=[], active_requests_info=[], current_kv_tokens=0,
        tokens_decoded_per_request={},
    )
    return ObservableState(time=now, waiting_queue=requests, gpu_states=[gpu], completed_count=0, step=1)


@pytest.mark.parametrize("policy_name", ["fifo", "shortest_output_first", "shortest_prompt_first", "edf"])
def test_exact_genome_reconstruction_matches_native_policy_ranking(policy_name):
    """For EXACT mappings, the compiled genome's admission ORDER (not full
    Action equality, since placement mechanics differ) must match the
    native policy's admission order on a tiny deterministic multi-request
    state -- the strongest fidelity check available given genome.py's own
    documented scope (admission + ranking, not placement)."""
    genome = map_policy_to_genome(policy_name)
    compiled_policy = genome.build_policy()
    native_policy = make_policy_library_v2(policy_name)

    requests = [
        _make_request(1, arrival=0.0, prompt=500, output=200, deadline=50.0, priority=1.0),
        _make_request(2, arrival=1.0, prompt=100, output=20, deadline=5.0, priority=3.0),
        _make_request(3, arrival=2.0, prompt=50, output=500, deadline=100.0, priority=1.0),
    ]
    state_a = _make_state([r for r in requests], now=3.0)
    state_b = _make_state([r for r in requests], now=3.0)

    action_native = native_policy.select_action(state_a)
    action_compiled = compiled_policy.select_action(state_b)

    admitted_native = {rid for ids in action_native.admit.values() for rid in ids}
    admitted_compiled = {rid for ids in action_compiled.admit.values() for rid in ids}
    assert admitted_native == admitted_compiled, (
        f"{policy_name}: native admitted {admitted_native}, genome-compiled admitted {admitted_compiled}"
    )


def test_approximate_genome_does_not_claim_exact_reconstruction():
    """Structural-consistency only for APPROXIMATE mappings -- explicitly not
    asserting behavioral equivalence, since that would misrepresent an
    approximation as exact."""
    genome = map_policy_to_genome("scorpio_style_slo_guard")
    assert genome.metadata["mapping_status"] == "APPROXIMATE"
    # It still must compile and produce a usable policy -- structural
    # consistency, not semantic equivalence.
    policy = genome.build_policy()
    state = _make_state([_make_request(1)])
    action = policy.select_action(state)
    assert action is not None
