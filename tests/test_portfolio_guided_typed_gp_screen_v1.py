from __future__ import annotations

import copy

import pytest

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.portfolio_gp import (
    DEFAULT_CHUNK_SMALL,
    PARENT_GENOMES_V1,
    PARENT_POLICY_IDS,
    PortfolioGPError,
    TreatmentBudgetAccountant,
    decision_overlap,
    equal_budget_summary,
    envelope_values,
    make_original_parent_policy,
    make_parent_reproduction_probe_states,
    marginal_gains,
    mutate_genome,
    parse_genome_string,
    policy_behavior_fingerprint,
    summarize_marginal_gain,
    typed_subtree_crossover,
)
from llmserveopt.policies.prefill_control_variants import GreedyArrivalPrefillControlPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def test_parent_genome_round_trip_and_hash_are_stable() -> None:
    hashes = {}
    for policy_id, genome in PARENT_GENOMES_V1.items():
        restored = parse_genome_string(genome.genome_string())
        assert restored.genome_string() == genome.genome_string()
        assert restored.stable_hash() == genome.stable_hash()
        assert restored.root.parameters["canonical_parent_id"] == policy_id
        hashes[policy_id] = restored.stable_hash()
    assert len(set(hashes.values())) == len(PARENT_POLICY_IDS)


@pytest.mark.parametrize("policy_id", PARENT_POLICY_IDS)
def test_parent_reproduction_action_trace_exact(policy_id: str) -> None:
    from llmserveopt.policies.portfolio_gp import compare_parent_on_probe_states

    result = compare_parent_on_probe_states(policy_id, make_parent_reproduction_probe_states())
    assert result.status == "PARENT_REPRODUCTION_PASS", result.first_mismatch
    assert result.exact_action_agreement is True
    assert result.decision_points == len(make_parent_reproduction_probe_states())


def _prefill_trace() -> list[Request]:
    return [
        Request(
            request_id=1,
            arrival_time=0.0,
            prompt_tokens=512,
            predicted_output_tokens=3,
            actual_output_tokens=3,
            slo_deadline=1.0,
            priority=1.0,
            class_id="hog",
        ),
        Request(
            request_id=2,
            arrival_time=0.001,
            prompt_tokens=64,
            predicted_output_tokens=3,
            actual_output_tokens=3,
            slo_deadline=1.0,
            priority=1.0,
            class_id="late",
        ),
    ]


def _run_prefill(policy, *, chunk: int):
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=[GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=128, max_kv_tokens=10000)],
            service_model=ServiceModel(
                enable_prefill_modeling=True,
                enable_decode_prefill_contention=True,
                decode_first=False,
                step_token_budget=128,
                max_prefill_chunk_tokens=chunk,
            ),
            max_steps=2000,
        )
    )
    sim.load_trace(_prefill_trace())
    return sim.run(policy, workload_tag="prefill_reproduction", seed=7)


def test_prefill_execution_control_reproduces_fixed_service_chunk_metrics() -> None:
    # Original fixed chunk execution: the policy emits no override, but the
    # service model carries the chunk cap.  Decoded GP parent execution: the
    # service model starts with the opposite chunk cap and the policy emits the
    # exact per-step override.  The simulator treats these as equivalent.
    original = _run_prefill(GreedyArrivalPrefillControlPolicy(), chunk=DEFAULT_CHUNK_SMALL)
    decoded = _run_prefill(PARENT_GENOMES_V1["chunked_prefill_small"].build_policy(), chunk=512)
    assert decoded.num_completed == original.num_completed
    assert decoded.mean_ttft == original.mean_ttft
    assert decoded.mean_prefill_delay == original.mean_prefill_delay
    assert decoded.arrival_normalized_weighted_goodput == original.arrival_normalized_weighted_goodput


def test_full_and_chunked_prefill_are_structurally_and_behaviorally_distinct() -> None:
    probe = make_parent_reproduction_probe_states()
    full = PARENT_GENOMES_V1["full_prefill"].build_policy()
    small = PARENT_GENOMES_V1["chunked_prefill_small"].build_policy()
    assert PARENT_GENOMES_V1["full_prefill"].stable_hash() != PARENT_GENOMES_V1["chunked_prefill_small"].stable_hash()
    assert policy_behavior_fingerprint(full, probe) != policy_behavior_fingerprint(small, probe)
    assert decision_overlap(full, small, probe) < 1.0


def test_valid_typed_crossover_and_invalid_crossover_rejection() -> None:
    estf = PARENT_GENOMES_V1["estimated_service_time_first"]
    llf = PARENT_GENOMES_V1["least_laxity_first"]
    child = typed_subtree_crossover(estf, llf, "RankingRule", seed=20260824)
    child.validate()
    assert child.root.parameters["canonical_parent_id"] is None
    assert child.root.parameters["exactness_status"] == "COMPOSED_CANDIDATE"
    assert child.build_policy().select_action(copy.deepcopy(make_parent_reproduction_probe_states()[0])) is not None
    with pytest.raises(PortfolioGPError):
        typed_subtree_crossover(estf, llf, "PlacementRule", seed=0)


def test_valid_mutation_stays_in_grammar_and_changes_hash() -> None:
    parent = PARENT_GENOMES_V1["kv_constrained_online"]
    child = mutate_genome(parent, seed=5)
    child.validate()
    assert child.stable_hash() != parent.stable_hash()
    assert child.build_policy().select_action(copy.deepcopy(make_parent_reproduction_probe_states()[2])) is not None


def test_fingerprint_determinism_and_decision_overlap() -> None:
    probe = make_parent_reproduction_probe_states()
    a = PARENT_GENOMES_V1["weighted_fair_share"].build_policy()
    b = PARENT_GENOMES_V1["weighted_fair_share"].build_policy()
    c = PARENT_GENOMES_V1["estimated_service_time_first"].build_policy()
    assert policy_behavior_fingerprint(a, probe) == policy_behavior_fingerprint(b, probe)
    assert decision_overlap(a, b, probe) == 1.0
    assert decision_overlap(a, c, probe) < 1.0


def test_envelope_marginal_gain_and_unique_wins() -> None:
    parents = {
        "p1": [0.50, 0.70, 0.40],
        "p2": [0.60, 0.65, 0.45],
    }
    envelope = envelope_values(parents)
    assert envelope == [0.60, 0.70, 0.45]
    assert marginal_gains([0.55, 0.70, 0.40], envelope) == [0.0, 0.0, 0.0]
    assert marginal_gains([0.61, 0.71, 0.50], envelope) == pytest.approx([0.01, 0.01, 0.05])
    summary = summarize_marginal_gain(
        [0.61, 0.70, 0.50],
        parents,
        ["A", "A", "B"],
        epsilon=0.005,
    )
    assert summary["mean_MG"] == pytest.approx((0.01 + 0.0 + 0.05) / 3.0)
    assert summary["unique_wins_eps"] == 2
    assert summary["positive_regions"] == 2


def test_equal_budget_accounting_requires_equal_evaluated_candidates() -> None:
    a = TreatmentBudgetAccountant("A_RANDOM_GRAMMAR_GP", 2)
    b = TreatmentBudgetAccountant("B_PARENT_SEEDED_MUTATION_ONLY", 2)
    for acct in (a, b):
        acct.record_proposed()
        acct.record_valid_unique(evaluated=True)
        acct.record_proposed()
        acct.record_valid_unique(evaluated=True)
    summary = equal_budget_summary([a, b])
    assert summary["equal_evaluated_candidates"] is True
    assert all(item["complete"] for item in summary["treatments"])
