from __future__ import annotations

import math

import pytest

from llmserveopt.core.metrics import RunMetrics
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.selector.dataset_v2.builder import metrics_to_outcome_vector
from llmserveopt.selector.dataset_v2.candidates import (
    candidate_policies_for_topology,
    is_policy_compatible_with_topology,
    monolithic_candidate_policies,
)
from llmserveopt.selector.dataset_v2.discriminativeness import (
    Objective,
    compute_discriminativeness,
    compute_regrets,
)
from llmserveopt.selector.dataset_v2.features import (
    extract_selector_v2_features,
    selector_v2_feature_columns,
)
from llmserveopt.selector.dataset_v2.scenario_families import all_scenario_family_specs
from llmserveopt.selector.dataset_v2.scenario_redesign import (
    DISCRIMINATIVE_POOL,
    REPRESENTATIVE_POOL,
    bottleneck_taxonomy_specs,
    sampled_bottleneck_specs,
    transform_requests,
)
from llmserveopt.selector.dataset_v2.scenario_search import (
    TrialSummary,
    retained_pool_for_trial,
)
from llmserveopt.selector.dataset_v2.schema import (
    PolicyOutcomeVector,
    ScenarioIdentifiers,
    WindowRecordV2,
)
from llmserveopt.selector.dataset_v2.splits import (
    OOD_TEST,
    assign_group_aware_split,
    verify_group_atomicity,
    verify_ood_holdout,
)
from llmserveopt.selector.dataset_v2.workload_sources import (
    WORKLOAD_SOURCE_MANIFEST,
    acquired_sources,
)


def _req(
    request_id: int,
    arrival_time: float,
    prompt: int = 64,
    pred: int = 32,
    actual: int = 32,
    slack: float = 5.0,
    priority: float = 1.0,
    class_id: str = "medium",
) -> Request:
    return Request(
        request_id=request_id,
        arrival_time=arrival_time,
        prompt_tokens=prompt,
        predicted_output_tokens=pred,
        actual_output_tokens=actual,
        slo_deadline=arrival_time + slack,
        priority=priority,
        class_id=class_id,
    )


def test_schema_flatten_preserves_one_row_per_policy_and_full_vector():
    identifiers = ScenarioIdentifiers(
        scenario_id="s0",
        scenario_family_id="fam",
        dataset_family="controlled_stress",
        source_trace="synthetic",
        seed=7,
        topology_class="monolithic",
        resource_configuration_id="gpus1_kv100",
        window_id=0,
    )
    record = WindowRecordV2(
        identifiers=identifiers,
        features={"arrival_rate_recent": 1.0, "source_trace": 99.0},
        outcomes=[
            PolicyOutcomeVector("fifo", "historical", weighted_goodput=0.5, available_metrics=["weighted_goodput"]),
            PolicyOutcomeVector("edf", "historical", weighted_goodput=0.8, available_metrics=["weighted_goodput"]),
        ],
        discriminativeness=[],
        regrets=[],
    )

    rows = record.to_flat_rows()
    assert len(rows) == 2
    assert {r["policy_name"] for r in rows} == {"fifo", "edf"}
    assert {r["metric_weighted_goodput"] for r in rows} == {0.5, 0.8}


def test_actual_output_tokens_cannot_influence_v2_features():
    window_start = 10.0
    prefix_a = [_req(i, float(i), actual=10) for i in range(10)]
    prefix_b = [_req(i, float(i), actual=10_000) for i in range(10)]
    win_a = [_req(100, window_start, actual=10), _req(101, window_start + 1.0, actual=10)]
    win_b = [_req(100, window_start, actual=10_000), _req(101, window_start + 1.0, actual=10_000)]

    fa = extract_selector_v2_features(
        window_requests=win_a,
        window_start_time=window_start,
        prefix_requests=prefix_a,
        gpu_configs=[GPUConfig(0, 8, 128, 4096)],
    )
    fb = extract_selector_v2_features(
        window_requests=win_b,
        window_start_time=window_start,
        prefix_requests=prefix_b,
        gpu_configs=[GPUConfig(0, 8, 128, 4096)],
    )

    assert fa == fb


def test_future_requests_cannot_influence_current_window_v2_features():
    window_start = 10.0
    prefix = [_req(i, float(i), prompt=20) for i in range(5)]
    base_window = [
        _req(100, window_start, prompt=100),
        _req(101, window_start + 1.0, prompt=200),
        _req(102, window_start + 2.0, prompt=300),
    ]
    mutated_window = [
        _req(100, window_start, prompt=100),
        _req(101, window_start + 1.0, prompt=9999),
        _req(102, window_start + 2.0, prompt=8888),
    ]

    base = extract_selector_v2_features(
        window_requests=base_window,
        window_start_time=window_start,
        prefix_requests=prefix,
        gpu_configs=[GPUConfig(0, 8, 128, 4096)],
    )
    mutated = extract_selector_v2_features(
        window_requests=mutated_window,
        window_start_time=window_start,
        prefix_requests=prefix,
        gpu_configs=[GPUConfig(0, 8, 128, 4096)],
    )

    assert base == mutated


def test_held_out_trace_family_identifiers_do_not_leak_into_model_features():
    rows = [{
        "scenario_id": "secret",
        "scenario_family_id": "heldout_family",
        "source_trace": "azure_2023_code",
        "topology_class": "monolithic",
        "policy_name": "fifo",
        "metric_weighted_goodput": 0.1,
        "feat_arrival_rate_recent": 2.0,
        "feat_prompt_mean": 64.0,
    }]

    assert selector_v2_feature_columns(rows) == [
        "feat_arrival_rate_recent",
        "feat_prompt_mean",
    ]


def test_grouped_split_integrity_and_ood_holdout():
    groups = ["burstgpt", "azure", "stress_a", "stress_b"]
    assignment = assign_group_aware_split(groups, ood_group_keys={"azure"})
    rows = [
        {"scenario_family_id": g, "split": assignment[g], "row": i}
        for i, g in enumerate(groups * 2)
    ]
    assert assignment["azure"] == OOD_TEST
    verify_group_atomicity(rows, "scenario_family_id")
    verify_ood_holdout(rows, "scenario_family_id", {"azure"})

    bad = rows + [{"scenario_family_id": "azure", "split": "TRAIN"}]
    with pytest.raises(ValueError):
        verify_group_atomicity(bad, "scenario_family_id")


def test_source_provenance_manifest_distinguishes_real_and_synthetic_fields():
    sources = {s.name: s for s in WORKLOAD_SOURCE_MANIFEST}
    assert "burstgpt" in sources
    assert "azure_llm_inference_traces" in sources
    assert "synthetic_workload_config" in sources
    assert sources["burstgpt"].acquired is True
    assert sources["azure_llm_inference_traces"].acquired is True
    assert "Arrival timestamps" in sources["azure_llm_inference_traces"].real_fields
    assert "Everything" in sources["synthetic_workload_config"].synthetic_fields
    assert acquired_sources()


def test_deterministic_feature_extraction_and_scenario_generation():
    spec = all_scenario_family_specs()[0]
    a = spec.build(123)
    b = spec.build(123)
    assert a == b

    feats_a = extract_selector_v2_features(
        window_requests=a[:10],
        window_start_time=a[0].arrival_time,
        prefix_requests=[],
        gpu_configs=[GPUConfig(0, 8, 128, 4096)],
    )
    feats_b = extract_selector_v2_features(
        window_requests=b[:10],
        window_start_time=b[0].arrival_time,
        prefix_requests=[],
        gpu_configs=[GPUConfig(0, 8, 128, 4096)],
    )
    assert feats_a == feats_b


def test_near_tie_classification_and_regret_computation():
    objective = Objective("weighted_goodput", True, lambda o: o.weighted_goodput)
    outcomes = [
        PolicyOutcomeVector("a", "historical", weighted_goodput=1.0),
        PolicyOutcomeVector("b", "historical", weighted_goodput=0.998),
        PolicyOutcomeVector("c", "historical", weighted_goodput=0.7),
    ]
    disc = compute_discriminativeness(outcomes, objective)
    assert disc is not None
    assert disc.classification == "NEAR_TIE"
    assert disc.best_policy == "a"
    regrets = {r.policy_name: r.regret for r in compute_regrets(outcomes, objective)}
    assert regrets["a"] == pytest.approx(0.0)
    assert regrets["b"] == pytest.approx(0.002)
    assert regrets["c"] == pytest.approx(0.3)


def test_practical_threshold_logic_separates_near_moderate_strong():
    objective = Objective("weighted_goodput", True, lambda o: o.weighted_goodput)
    near = compute_discriminativeness([
        PolicyOutcomeVector("a", "historical", weighted_goodput=1.0),
        PolicyOutcomeVector("b", "historical", weighted_goodput=0.9985),
        PolicyOutcomeVector("c", "historical", weighted_goodput=0.9),
    ], objective)
    moderate = compute_discriminativeness([
        PolicyOutcomeVector("a", "historical", weighted_goodput=1.0),
        PolicyOutcomeVector("b", "historical", weighted_goodput=0.99),
    ], objective)
    strong = compute_discriminativeness([
        PolicyOutcomeVector("a", "historical", weighted_goodput=1.0),
        PolicyOutcomeVector("b", "historical", weighted_goodput=0.96),
    ], objective)
    assert near.classification == "NEAR_TIE"
    assert moderate.classification == "MODERATELY_DISCRIMINATIVE"
    assert strong.classification == "STRONGLY_DISCRIMINATIVE"


def test_lower_is_better_regret_computation():
    objective = Objective("p95_latency", False, lambda o: o.p95_latency)
    outcomes = [
        PolicyOutcomeVector("fast", "historical", p95_latency=10.0),
        PolicyOutcomeVector("slow", "historical", p95_latency=18.0),
    ]
    regrets = {r.policy_name: r.regret for r in compute_regrets(outcomes, objective)}
    assert regrets["fast"] == pytest.approx(0.0)
    assert regrets["slow"] == pytest.approx(8.0)


def test_policy_compatibility_by_topology():
    mono = monolithic_candidate_policies()
    assert {"vllm_faithful", "sarathi_faithful"}.issubset(set(mono))
    assert {"shortest_output_first", "estimated_service_time_first", "best_fit", "multi_bin_batching"}.issubset(set(mono))
    assert is_policy_compatible_with_topology("fifo", "monolithic")
    assert not is_policy_compatible_with_topology("fifo", "disaggregated_prefill_decode")
    assert not is_policy_compatible_with_topology("distserve_faithful", "monolithic")
    assert candidate_policies_for_topology("multi_instance_migratory") == ["llumnix_faithful"]


def test_missing_metric_handling_uses_none_not_zero():
    metrics = RunMetrics(policy_name="fifo", workload_tag="w", seed=1)
    metrics.num_total = 10
    metrics.num_completed = 0
    metrics.num_dropped = 10
    metrics.weighted_goodput = math.nan
    outcome = metrics_to_outcome_vector("fifo", metrics, {"admit": 0, "preempt": 0, "swap": 0, "migrate": 0}, 1)

    assert outcome.weighted_goodput == pytest.approx(0.0)
    assert outcome.p50_ttft is None
    assert outcome.p99_tpot is None
    assert "weighted_goodput" in outcome.available_metrics
    assert outcome.num_dropped == 10
    assert outcome.rejection_rate == pytest.approx(1.0)


def test_redesigned_scenario_generation_is_deterministic_and_preserves_ancestor():
    specs_a = sampled_bottleneck_specs(seed=1234, count=4)
    specs_b = sampled_bottleneck_specs(seed=1234, count=4)
    assert [s.family_id for s in specs_a] == [s.family_id for s in specs_b]
    assert [s.request_plan_ancestor_id for s in specs_a] == [s.request_plan_ancestor_id for s in specs_b]
    assert all(s.scenario_pool == DISCRIMINATIVE_POOL for s in specs_a)
    reqs_a = specs_a[0].build(99)
    reqs_b = specs_b[0].build(99)
    assert reqs_a == reqs_b


def test_transform_requests_preserves_request_order_and_source_grouping_inputs():
    base = [_req(i, float(i), pred=10, actual=20, slack=10.0) for i in range(12)]
    transformed_a = transform_requests(
        base,
        time_scale=0.5,
        slo_scale=0.2,
        prediction_noise_rel=0.1,
        prediction_bias=0.5,
        burst_amplification=3.0,
        seed=7,
    )
    transformed_b = transform_requests(
        base,
        time_scale=0.5,
        slo_scale=0.2,
        prediction_noise_rel=0.1,
        prediction_bias=0.5,
        burst_amplification=3.0,
        seed=7,
    )
    assert transformed_a == transformed_b
    assert [r.request_id for r in transformed_a] == list(range(12))
    assert transformed_a[-1].arrival_time < base[-1].arrival_time
    assert transformed_a[0].slo_deadline - transformed_a[0].arrival_time == pytest.approx(2.0)


def test_adaptive_retention_logic_prefers_discriminative_trials():
    summary = TrialSummary(
        family_id="f",
        seed=1,
        bottleneck_class="kv_pressure",
        source_trace="synthetic",
        request_plan_ancestor_id="ancestor",
        num_windows=4,
        class_counts={"STRONGLY_DISCRIMINATIVE": 2, "NEAR_TIE": 2},
        winner_counts={"weighted_shortest_processing": 2, "scorpio_style_slo_guard": 2},
        max_spread=0.15,
        mean_spread=0.08,
        mean_best_score=0.8,
        mean_best_fixed_score=0.75,
        oracle_headroom=0.05,
    )
    pool, reason = retained_pool_for_trial(summary, {}, 0, 0)
    assert pool == DISCRIMINATIVE_POOL
    assert reason in {
        "discriminative_underrepresented_winner",
        "discriminative_oracle_headroom",
        "discriminative_density",
    }


def test_bottleneck_taxonomy_has_required_classes_and_pool_labels():
    specs = bottleneck_taxonomy_specs()
    required = {
        "admission_pressure",
        "kv_pressure",
        "prefill_heavy",
        "decode_heavy",
        "slo_heterogeneous",
        "prediction_noise",
        "bursty_transient",
        "resource_scarcity",
    }
    assert required.issubset({s.bottleneck_class for s in specs})
    assert all(s.scenario_pool in {REPRESENTATIVE_POOL, DISCRIMINATIVE_POOL} for s in specs)
