from __future__ import annotations

import json

import pytest

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.genome import GenomeValidationError, parse_genome
from llmserveopt.policies.structural_synthesis import (
    conditional_composition,
    frontier_value,
    map_policy_to_genome,
    module_swap,
    mutate_constants,
    mutate_feature_or_operator,
    render_llm_synthesis_prompt,
    typed_subtree_crossover,
    verify_child,
)
from llmserveopt.selector.parent_selection import (
    CompositionGateConfig,
    ParentEvidence,
    composition_gate,
    score_parent_pair,
    select_parent_pairs,
)


def _requests() -> list[Request]:
    return [
        Request(0, 0.0, 64, 32, 32, 3.0, 1.0, "medium"),
        Request(1, 0.01, 512, 64, 64, 3.0, 1.0, "medium"),
        Request(2, 0.02, 32, 512, 512, 0.5, 3.0, "tight"),
        Request(3, 0.03, 128, 128, 128, 2.0, 2.0, "medium"),
    ]


def test_genome_serialization_and_hash_are_stable():
    genome = map_policy_to_genome("weighted_shortest_processing")
    payload = genome.to_dict()
    parsed = parse_genome(json.loads(json.dumps(payload)))
    assert parsed.canonical_json() == genome.canonical_json()
    assert parsed.stable_hash() == genome.stable_hash()


def test_genome_builds_verified_heuristic_policy():
    genome = map_policy_to_genome("weighted_shortest_processing")
    policy = genome.build_policy()
    metrics = run_policy(policy, _requests(), [GPUConfig(0, 2, 128, 4096)], drain_steps=1000)
    assert metrics.num_total == 4
    assert metrics.num_completed >= 1


def test_causal_whitelist_rejects_future_feature():
    payload = map_policy_to_genome("weighted_shortest_processing").to_dict()
    payload["priority_rule"]["expression"] = {"var": "future_arrivals"}
    with pytest.raises(GenomeValidationError):
        parse_genome(payload)


def test_policy_mappings_record_exact_approximate_unsupported():
    assert map_policy_to_genome("weighted_shortest_processing").metadata["mapping_status"] == "EXACT"
    assert map_policy_to_genome("edf").metadata["mapping_status"] == "EXACT"
    assert map_policy_to_genome("scorpio_style_slo_guard").metadata["mapping_status"] == "APPROXIMATE"
    assert map_policy_to_genome("adaptive_chunked_prefill").metadata["mapping_status"] == "APPROXIMATE"
    assert map_policy_to_genome("unknown_policy").metadata["mapping_status"] == "UNSUPPORTED"


def test_module_swap_is_deterministic():
    wsp = map_policy_to_genome("weighted_shortest_processing")
    scorpio = map_policy_to_genome("scorpio_style_slo_guard")
    child_a = module_swap(wsp, scorpio, "admission_rule", child_name="child")
    child_b = module_swap(wsp, scorpio, "admission_rule", child_name="child")
    assert child_a.stable_hash() == child_b.stable_hash()
    assert child_a.admission_rule is not None
    assert verify_child(child_a)


def test_conditional_composition_verifies():
    wsp = map_policy_to_genome("weighted_shortest_processing")
    scorpio = map_policy_to_genome("scorpio_style_slo_guard")
    child = conditional_composition(
        {"op": "sub", "args": [{"var": "sys.slo_pressure"}, {"const": 0.2}]},
        then_parent=scorpio,
        else_parent=wsp,
        child_name="if_slo_pressure_then_scorpio_else_wsp",
    )
    assert child.regime_conditions
    assert verify_child(child)


def test_invalid_module_crossover_rejected():
    wsp = map_policy_to_genome("weighted_shortest_processing")
    scorpio = map_policy_to_genome("scorpio_style_slo_guard")
    with pytest.raises(GenomeValidationError):
        typed_subtree_crossover(wsp, scorpio, "admission_rule", child_name="bad")


def test_typed_subtree_crossover_accepts_compatible_modules():
    wsp = map_policy_to_genome("weighted_shortest_processing")
    edf = map_policy_to_genome("edf")
    child = typed_subtree_crossover(wsp, edf, "priority_rule", child_name="wsp_with_edf_priority")
    assert child.priority_rule.description.startswith("earliest deadline")
    assert verify_child(child)


def test_constant_mutation_is_bounded_and_deterministic():
    scorpio = map_policy_to_genome("scorpio_style_slo_guard")
    child_a = mutate_constants(scorpio, scale=0.05, seed=7, child_name="mut")
    child_b = mutate_constants(scorpio, scale=0.05, seed=7, child_name="mut")
    assert child_a.stable_hash() == child_b.stable_hash()
    assert verify_child(child_a)


def test_feature_operator_mutation_uses_whitelist():
    wsp = map_policy_to_genome("weighted_shortest_processing")
    child = mutate_feature_or_operator(wsp, child_name="feature_mut")
    assert "req.estimated_decode_cost" in child.canonical_json()
    assert verify_child(child)


def test_parent_selection_is_deterministic():
    evidence = ParentEvidence(
        expected_advantage={("wsp", "scorpio"): 0.02, ("wsp", "edf"): 0.01},
        complementarity={("wsp", "scorpio"): 0.4, ("wsp", "edf"): 0.1},
        marginal_frontier_value={"wsp": 0.01, "scorpio": 0.03, "edf": 0.0},
        incompatibility={("wsp", "scorpio"): 0.0, ("wsp", "edf"): 0.2},
        uncertainty={("wsp", "scorpio"): 0.05, ("wsp", "edf"): 0.0},
    )
    best = select_parent_pairs(["edf", "scorpio", "wsp"], evidence, top_n=1)[0]
    assert (best.parent_a, best.parent_b) == ("scorpio", "wsp")
    again = select_parent_pairs(["wsp", "edf", "scorpio"], evidence, top_n=1)[0]
    assert best == again


def test_composition_gate_behavior():
    evidence = ParentEvidence(
        expected_advantage={("a", "b"): 0.01},
        complementarity={("a", "b"): 0.3},
        marginal_frontier_value={"a": 0.0, "b": 0.0},
        incompatibility={("a", "b"): 0.0},
        uncertainty={("a", "b"): 0.1},
    )
    score = score_parent_pair("a", "b", evidence)
    decision = composition_gate(top1_top2_margin=0.001, pair_score=score)
    assert decision.action == "ATTEMPT_STRUCTURAL_COMPOSITION"
    blocked = composition_gate(top1_top2_margin=0.001, pair_score=score_parent_pair("a", "b", ParentEvidence(
        expected_advantage={("a", "b"): 0.01},
        complementarity={("a", "b"): 0.3},
        marginal_frontier_value={},
        incompatibility={("a", "b"): 1.0},
        uncertainty={("a", "b"): 0.0},
    )), config=CompositionGateConfig(max_incompatibility=0.5))
    assert blocked.action == "SELECT_SINGLE"


def test_frontier_value_calculation():
    value = frontier_value(
        {"wsp": [0.5, 0.2, 0.1], "scorpio": [0.4, 0.3, 0.1]},
        [0.45, 0.35, 0.2],
        meaningful_margin=0.02,
        complexity_penalty=0.0,
    )
    assert value["unique_win_count"] == 2
    assert value["meaningful_unique_win_count"] == 2
    assert value["marginal_frontier_value"] == pytest.approx((0.0 + 0.05 + 0.1) / 3)


def test_llm_prompt_template_is_structured_and_forbids_features():
    wsp = map_policy_to_genome("weighted_shortest_processing")
    prompt = render_llm_synthesis_prompt(
        target_workload_niche="high SLO pressure WSP/SCORPIO boundary",
        parent_genomes=[wsp],
        parent_strengths={wsp.name: "strong on short-work regimes"},
        pairwise_advantage_evidence={"delta": "development only"},
        frontier_gap="missing robust child",
        allowed_primitives=["admission_rule", "priority_rule"],
        forbidden_features=["actual_output_tokens", "future_arrivals"],
    )
    payload = json.loads(prompt)
    assert payload["task"] == "propose_scheduler_genome_child"
    assert "actual_output_tokens" in payload["forbidden_features"]
