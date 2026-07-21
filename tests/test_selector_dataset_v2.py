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
    PRIMARY_SELECTOR_OBJECTIVE,
    compute_discriminativeness,
    compute_regrets,
)
from llmserveopt.selector.dataset_v2.features import (
    extract_selector_v2_features,
    selector_v2_feature_columns,
)
from llmserveopt.selector.dataset_v2.scenario_families import all_scenario_family_specs
from llmserveopt.selector.dataset_v2.scenario_families import ScenarioFamilySpec
from llmserveopt.selector.dataset_v2.scenario_redesign import (
    DISCRIMINATIVE_POOL,
    REPRESENTATIVE_POOL,
    bottleneck_taxonomy_specs,
    local_real_trace_stress_specs,
    sampled_bottleneck_specs,
    targeted_counterexample_specs,
    transform_requests,
)
from llmserveopt.selector.dataset_v2.scenario_search import (
    TrialSummary,
    diversity_aware_retained_pool_for_trial,
    retained_pool_for_trial,
    summarize_trial,
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
from llmserveopt.selector.dataset_v2.calibrated_targeted_pilot import (
    CandidateWindow,
    HISTORICAL_POOL,
    OOD_RESERVED_POOL,
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


def test_real_trace_split_group_is_ancestor_pool_not_transform_specific():
    from scripts.build_selector_dataset_v2_calibrated_targeted_pilot import _split_group_key

    common = dict(
        dataset_family="real_trace",
        source_trace="burstgpt",
        requests=[],
        budget=512,
        chunk=512,
        max_kv_tokens=7000,
        max_active_sequences=8,
        time_slice_pool=HISTORICAL_POOL,
        time_slice_row_range=(100, 244),
        request_plan_ancestor_id="real_trace__burstgpt_scaled_moderate",
    )
    representative = CandidateWindow(
        group_key="real_trace__burstgpt_scaled_moderate__representative__historical",
        shape="real_trace_stress__burstgpt_scaled_moderate__representative",
        **common,
    )
    compressed = CandidateWindow(
        group_key="real_trace__burstgpt_scaled_moderate__compressed_tight__historical",
        shape="real_trace_stress__burstgpt_scaled_moderate__compressed_tight",
        **common,
    )

    assert _split_group_key(representative) == _split_group_key(compressed)


def test_real_trace_ood_split_group_is_distinct_and_forced():
    from llmserveopt.selector.dataset_v2.splits import split_for_group
    from scripts.build_selector_dataset_v2_calibrated_targeted_pilot import _split_group_key

    historical = CandidateWindow(
        group_key="real_trace__azure_2023_code__representative__historical",
        dataset_family="real_trace",
        source_trace="azure_llm_2023",
        shape="real_trace_stress__azure_2023_code__representative",
        requests=[],
        budget=512,
        chunk=512,
        max_kv_tokens=7000,
        max_active_sequences=8,
        time_slice_pool=HISTORICAL_POOL,
        request_plan_ancestor_id="real_trace__azure_2023_code",
    )
    ood = CandidateWindow(
        group_key="real_trace__azure_2023_code__representative__ood_reserved",
        dataset_family="real_trace",
        source_trace="azure_llm_2023",
        shape="real_trace_stress__azure_2023_code__representative",
        requests=[],
        budget=512,
        chunk=512,
        max_kv_tokens=7000,
        max_active_sequences=8,
        time_slice_pool=OOD_RESERVED_POOL,
        request_plan_ancestor_id="real_trace__azure_2023_code",
    )
    groups = [_split_group_key(historical), _split_group_key(ood)]
    assignment = assign_group_aware_split(groups, ood_group_keys={groups[1]})

    assert groups[0] != groups[1]
    assert split_for_group(groups[1], assignment) == OOD_TEST


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


def test_corrected_primary_objective_penalizes_selective_service_winner():
    objective = Objective(
        PRIMARY_SELECTOR_OBJECTIVE,
        True,
        lambda o: o.arrival_normalized_weighted_goodput,
    )
    outcomes = [
        PolicyOutcomeVector(
            "selective",
            "historical",
            weighted_goodput=1.0,
            weighted_completion_fraction=0.2,
            arrival_normalized_weighted_goodput=0.2,
        ),
        PolicyOutcomeVector(
            "serves_workload",
            "historical",
            weighted_goodput=0.8,
            weighted_completion_fraction=1.0,
            arrival_normalized_weighted_goodput=0.8,
        ),
    ]
    disc = compute_discriminativeness(outcomes, objective)
    assert disc is not None
    assert disc.best_policy == "serves_workload"
    assert disc.absolute_winner_margin == pytest.approx(0.6)
    assert disc.classification == "STRONGLY_DISCRIMINATIVE"


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


def test_metrics_to_outcome_vector_preserves_corrected_utility_fields():
    metrics = RunMetrics(policy_name="fifo", workload_tag="w", seed=1)
    metrics.num_total = 10
    metrics.num_completed = 8
    metrics.num_dropped = 2
    metrics.weighted_goodput = 0.75
    metrics.weighted_completion_fraction = 0.8
    metrics.arrival_normalized_weighted_goodput = 0.6
    metrics.request_throughput = 5.0
    outcome = metrics_to_outcome_vector(
        "fifo",
        metrics,
        {"admit": 8, "preempt": 0, "swap": 0, "migrate": 0},
        1,
    )
    assert outcome.weighted_goodput == pytest.approx(0.75)
    assert outcome.weighted_completion_fraction == pytest.approx(0.8)
    assert outcome.arrival_normalized_weighted_goodput == pytest.approx(0.6)
    assert outcome.rejection_fraction == pytest.approx(0.2)
    assert outcome.slo_success_throughput == pytest.approx(3.0)
    assert "slo_success_throughput" in outcome.available_metrics


def test_trial_summary_uses_arrival_normalized_primary_objective():
    spec = ScenarioFamilySpec(
        family_id="unit",
        dataset_family="controlled_stress",
        description="unit",
        build=lambda seed: [],
        source_trace="synthetic",
        request_plan_ancestor_id="ancestor",
        bottleneck_class="unit",
    )
    outcomes = [
        PolicyOutcomeVector(
            "scorpio_style_slo_guard",
            "historical",
            weighted_goodput=1.0,
            weighted_completion_fraction=0.2,
            arrival_normalized_weighted_goodput=0.2,
        ),
        PolicyOutcomeVector(
            "vllm_faithful",
            "faithful",
            weighted_goodput=0.8,
            weighted_completion_fraction=1.0,
            arrival_normalized_weighted_goodput=0.8,
        ),
    ]
    identifiers = ScenarioIdentifiers(
        scenario_id="s",
        scenario_family_id="unit",
        dataset_family="controlled_stress",
        source_trace="synthetic",
        seed=1,
        topology_class="monolithic",
        resource_configuration_id="gpus1",
        window_id=0,
        request_plan_ancestor_id="ancestor",
        bottleneck_class="unit",
    )
    primary = compute_discriminativeness(
        outcomes,
        Objective(PRIMARY_SELECTOR_OBJECTIVE, True, lambda o: o.arrival_normalized_weighted_goodput),
    )
    historical = compute_discriminativeness(
        outcomes,
        Objective("weighted_goodput", True, lambda o: o.weighted_goodput),
    )
    record = WindowRecordV2(
        identifiers=identifiers,
        features={},
        outcomes=outcomes,
        discriminativeness=[historical, primary],
        regrets=[],
    )
    summary = summarize_trial(spec, seed=1, records=[record])
    assert summary.winner_counts == {"vllm_faithful": 1}
    assert summary.strong_winner_counts == {"vllm_faithful": 1}
    assert summary.mean_best_score == pytest.approx(0.8)


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


def test_diversity_aware_retention_rewards_underrepresented_target_winner():
    summary = TrialSummary(
        family_id="counterexample__sarathi_faithful__000",
        seed=1,
        bottleneck_class="prefill_heavy",
        source_trace="synthetic",
        request_plan_ancestor_id="counterexample__sarathi_faithful",
        num_windows=4,
        class_counts={"STRONGLY_DISCRIMINATIVE": 3, "NEAR_TIE": 1},
        winner_counts={"sarathi_faithful": 3, "scorpio_style_slo_guard": 1},
        strong_winner_counts={"sarathi_faithful": 3},
        max_spread=0.10,
        mean_spread=0.06,
        mean_best_score=0.82,
        mean_best_fixed_score=0.75,
        oracle_headroom=0.07,
    )
    pool, reason = diversity_aware_retained_pool_for_trial(
        summary,
        current_winner_counts={"scorpio_style_slo_guard": 50},
        strong_winner_counts={"scorpio_style_slo_guard": 50},
        representative_windows=0,
        discriminative_windows=50,
        target_policies={"sarathi_faithful", "vllm_faithful"},
    )
    assert pool == DISCRIMINATIVE_POOL
    assert reason == "strong_target_policy_winner"


def test_diversity_aware_retention_caps_redundant_scorpio_strong_wins():
    summary = TrialSummary(
        family_id="admission_pressure__more_scorpio",
        seed=2,
        bottleneck_class="admission_pressure",
        source_trace="synthetic",
        request_plan_ancestor_id="admission_pressure",
        num_windows=4,
        class_counts={"STRONGLY_DISCRIMINATIVE": 4},
        winner_counts={"scorpio_style_slo_guard": 4},
        strong_winner_counts={"scorpio_style_slo_guard": 4},
        max_spread=0.20,
        mean_spread=0.12,
        mean_best_score=0.9,
        mean_best_fixed_score=0.8,
        oracle_headroom=0.1,
    )
    pool, reason = diversity_aware_retained_pool_for_trial(
        summary,
        current_winner_counts={"scorpio_style_slo_guard": 80},
        strong_winner_counts={"scorpio_style_slo_guard": 80, "vllm_faithful": 10},
        representative_windows=0,
        discriminative_windows=90,
        target_policies={"sarathi_faithful", "vllm_faithful"},
        max_single_strong_winner_share=0.85,
    )
    assert pool is None
    assert reason == "skipped_dominant_strong_winner_cap"


def test_targeted_counterexample_generation_is_deterministic_without_forced_labels():
    specs_a = targeted_counterexample_specs(seed=2026, count_per_target=3)
    specs_b = targeted_counterexample_specs(seed=2026, count_per_target=3)
    assert [s.family_id for s in specs_a] == [s.family_id for s in specs_b]
    assert [s.request_plan_ancestor_id for s in specs_a] == [s.request_plan_ancestor_id for s in specs_b]
    assert all(s.scenario_pool == DISCRIMINATIVE_POOL for s in specs_a)
    assert {s.bottleneck_class for s in specs_a}.issuperset({"prefill_heavy", "decode_heavy", "slo_heterogeneous"})
    assert all("winner" not in s.description.lower() for s in specs_a)
    assert specs_a[0].build(3) == specs_b[0].build(3)


def test_local_real_source_variants_preserve_ancestor_groups_when_present(tmp_path):
    data_dir = tmp_path / "data" / "processed" / "azure"
    data_dir.mkdir(parents=True)
    trace = data_dir / "azure_llm_2023_code.jsonl"
    trace.write_text(
        "\n".join([
            '{"request_id": 0, "arrival_time": 0.0, "prompt_tokens": 64, "predicted_output_tokens": 32, "actual_output_tokens": 32, "slo_deadline": 5.0, "priority": 1.0, "class_id": "medium"}',
            '{"request_id": 1, "arrival_time": 1.0, "prompt_tokens": 80, "predicted_output_tokens": 40, "actual_output_tokens": 40, "slo_deadline": 6.0, "priority": 1.0, "class_id": "medium"}',
        ])
        + "\n"
    )
    specs = local_real_trace_stress_specs(tmp_path, max_requests=2)
    assert specs
    assert {s.request_plan_ancestor_id for s in specs} == {"real_trace__azure_2023_code"}
    assert {s.source_trace for s in specs} == {"azure_llm_2023"}


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
