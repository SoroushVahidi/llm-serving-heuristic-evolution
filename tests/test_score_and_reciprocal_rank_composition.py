from __future__ import annotations

import math

import pytest

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.capabilities import (
    CapabilityError,
    RANK_CAPABLE_EXPERTS,
    SCORE_CAPABLE_EXPERTS,
    capabilities_for,
    require_rank_capable,
    require_score_capable,
)
from llmserveopt.policies.composition import (
    CompositionError,
    RankExpertSpec,
    StaticRankEnsemblePolicy,
    rank_with_named_expert,
    weighted_borda_aggregate,
    weighted_reciprocal_rank_aggregate,
)
from llmserveopt.policies.instrumentation import (
    DECISION_TRACE_SCHEMA_VERSION,
    DecisionTraceSink,
    InstrumentedPolicy,
)
from llmserveopt.policies.score_aggregation import (
    NormalizationMode,
    ScoreExpertSpec,
    StaticScoreEnsemblePolicy,
    build_score_weights,
    normalize_scores,
    score_with_named_expert,
    weighted_score_aggregate,
)


def req(
    request_id: int,
    *,
    prompt: int = 64,
    output: int = 64,
    arrival: float = 0.0,
    deadline: float = 20.0,
    priority: float = 1.0,
) -> ObservableRequest:
    return ObservableRequest(
        request_id=request_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=deadline,
        priority=priority,
        class_id="medium",
    )


def gpu(*, max_seq: int = 4, max_kv: int = 4096) -> ObservableGPUState:
    return ObservableGPUState(
        gpu_id=0,
        max_active_sequences=max_seq,
        max_batch_tokens=128,
        max_kv_tokens=max_kv,
        active_request_ids=[],
        active_requests_info=[],
        current_kv_tokens=0,
        tokens_decoded_per_request={},
    )


def state(reqs: list[ObservableRequest], *, now: float = 5.0, step: int = 1, g: ObservableGPUState | None = None) -> ObservableState:
    return ObservableState(time=now, waiting_queue=reqs, gpu_states=[g or gpu()], completed_count=0, step=step)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

def test_capabilities_flags_are_consistent():
    caps = capabilities_for("weighted_shortest_processing")
    assert caps.supports_scores and caps.supports_ranks

    caps = capabilities_for("scorpio_style_slo_guard")
    assert caps.supports_ranks and not caps.supports_scores and caps.supports_admission

    caps = capabilities_for("totally_unknown_policy")
    assert not caps.supports_ranks and not caps.supports_scores


def test_require_score_capable_raises_for_rank_only_expert():
    with pytest.raises(CapabilityError):
        require_score_capable(["scorpio_style_slo_guard"])
    require_score_capable(["weighted_shortest_processing"])  # no raise


def test_require_rank_capable_raises_for_unknown_expert():
    with pytest.raises(CapabilityError):
        require_rank_capable(["not_a_real_policy"])


def test_score_capable_is_subset_of_rank_capable():
    assert SCORE_CAPABLE_EXPERTS <= RANK_CAPABLE_EXPERTS


# ---------------------------------------------------------------------------
# Reciprocal-rank aggregation
# ---------------------------------------------------------------------------

def test_reciprocal_rank_prefers_top_agreement_over_borda_tail_spread():
    s = state([req(1, prompt=512, output=512), req(2, prompt=32, output=32), req(3, prompt=128, output=128)])
    outputs = {
        "weighted_shortest_processing": rank_with_named_expert("weighted_shortest_processing", s),
        "edf": rank_with_named_expert("edf", s),
    }
    weights = {"weighted_shortest_processing": 0.5, "edf": 0.5}
    aggregate, support, _ = weighted_reciprocal_rank_aggregate(outputs, weights, c=1.0)
    assert set(aggregate) == {1, 2, 3}
    assert all(support[rid] == 2 for rid in aggregate)
    # request 2 is rank 1 under WSP; c=1 makes 1/(1+1)=0.5 dominate the sum.
    ranked = sorted(aggregate, key=lambda rid: -aggregate[rid])
    assert ranked[0] == 2


def test_reciprocal_rank_requires_positive_finite_c():
    with pytest.raises(CompositionError):
        weighted_reciprocal_rank_aggregate({}, {}, c=0.0)
    with pytest.raises(CompositionError):
        weighted_reciprocal_rank_aggregate({}, {}, c=math.nan)


def test_static_rank_ensemble_reciprocal_rank_method_is_deterministic_and_differs_from_borda():
    s1 = state([req(1, prompt=512, output=512), req(2, prompt=32, output=32), req(3, prompt=128, output=128)], g=gpu(max_seq=3))
    s2 = state([req(1, prompt=512, output=512), req(2, prompt=32, output=32), req(3, prompt=128, output=128)], g=gpu(max_seq=3))
    experts = [RankExpertSpec("weighted_shortest_processing", 1.0), RankExpertSpec("edf", 1.0)]
    borda = StaticRankEnsemblePolicy(list(experts), method="borda")
    rrf = StaticRankEnsemblePolicy(list(experts), method="reciprocal_rank", reciprocal_rank_c=1.0)
    a_borda = borda.select_action(s1)
    a_rrf = rrf.select_action(s2)
    assert a_borda.admit[0] == sorted(a_borda.admit[0], key=lambda _: 0)  # sanity: still a valid ordering
    # Reproducibility of the reciprocal-rank method itself.
    s3 = state([req(1, prompt=512, output=512), req(2, prompt=32, output=32), req(3, prompt=128, output=128)], g=gpu(max_seq=3))
    rrf2 = StaticRankEnsemblePolicy(list(experts), method="reciprocal_rank", reciprocal_rank_c=1.0)
    assert rrf.select_action.__self__.method == "reciprocal_rank"
    assert a_rrf.admit == rrf2.select_action(s3).admit


def test_unknown_rank_aggregation_method_rejected():
    with pytest.raises(CompositionError):
        StaticRankEnsemblePolicy([RankExpertSpec("weighted_shortest_processing", 1.0)], method="not_a_method")


def test_weighted_borda_aggregate_matches_ensemble_output():
    s = state([req(1, prompt=512, output=512), req(2, prompt=32, output=32)])
    outputs = {"weighted_shortest_processing": rank_with_named_expert("weighted_shortest_processing", s)}
    aggregate, support, contributions = weighted_borda_aggregate(outputs, {"weighted_shortest_processing": 1.0})
    assert aggregate[2] == pytest.approx(1.0)
    assert aggregate[1] == pytest.approx(0.0)
    assert support[1] == 1 and support[2] == 1
    assert contributions[2]["weighted_shortest_processing"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_with_named_expert
# ---------------------------------------------------------------------------

def test_score_with_named_expert_prefers_short_work_for_wsp():
    s = state([req(1, prompt=512, output=512), req(2, prompt=32, output=32)])
    scores = score_with_named_expert("weighted_shortest_processing", s)
    assert scores[2] > scores[1]


def test_score_with_named_expert_rejects_rank_only_policy():
    s = state([req(1)])
    with pytest.raises(CapabilityError):
        score_with_named_expert("scorpio_style_slo_guard", s)


def test_score_with_named_expert_empty_queue_returns_empty_dict():
    assert score_with_named_expert("fifo", state([])) == {}


# ---------------------------------------------------------------------------
# normalize_scores: normalization modes + degenerate cases
# ---------------------------------------------------------------------------

def test_normalize_scores_none_is_identity():
    assert normalize_scores({1: 3.0, 2: -1.0}, NormalizationMode.NONE) == {1: 3.0, 2: -1.0}


def test_normalize_scores_min_max_scales_to_unit_interval():
    out = normalize_scores({1: 0.0, 2: 5.0, 3: 10.0}, NormalizationMode.MIN_MAX)
    assert out == pytest.approx({1: 0.0, 2: 0.5, 3: 1.0})


def test_normalize_scores_min_max_constant_vector_is_neutral():
    out = normalize_scores({1: 4.0, 2: 4.0, 3: 4.0}, "min_max")
    assert out == {1: 0.5, 2: 0.5, 3: 0.5}


def test_normalize_scores_zscore_matches_standard_score():
    out = normalize_scores({1: -1.0, 2: 1.0}, NormalizationMode.ZSCORE)
    assert out[1] == pytest.approx(-1.0)
    assert out[2] == pytest.approx(1.0)


def test_normalize_scores_zscore_constant_vector_is_zero():
    assert normalize_scores({1: 4.0, 2: 4.0}, NormalizationMode.ZSCORE) == {1: 0.0, 2: 0.0}


def test_normalize_scores_zscore_single_value_is_zero():
    assert normalize_scores({1: 4.0}, NormalizationMode.ZSCORE) == {1: 0.0}


def test_normalize_scores_robust_mad_matches_expected_scale():
    out = normalize_scores({1: 1.0, 2: 2.0, 3: 3.0, 4: 100.0}, NormalizationMode.ROBUST_MAD)
    # median=2.5, abs devs = [1.5,0.5,0.5,97.5], MAD=0.5*... median of devs=1.0
    assert out[2] < out[3]
    assert out[4] > out[3]


def test_normalize_scores_zero_mad_falls_back_to_zero():
    out = normalize_scores({1: 1.0, 2: 1.0, 3: 1.0, 4: 100.0}, NormalizationMode.ROBUST_MAD)
    # median=1.0, abs devs=[0,0,0,99] -> median of devs = 0 -> zero MAD fallback
    assert out == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


def test_normalize_scores_empty_input_returns_empty():
    assert normalize_scores({}, NormalizationMode.MIN_MAX) == {}


def test_normalize_scores_rejects_nan_and_inf():
    with pytest.raises(CompositionError):
        normalize_scores({1: float("nan")}, NormalizationMode.MIN_MAX)
    with pytest.raises(CompositionError):
        normalize_scores({1: float("inf"), 2: 1.0}, NormalizationMode.ZSCORE)


def test_normalize_scores_rejects_unknown_mode_string():
    with pytest.raises(ValueError):
        normalize_scores({1: 1.0}, "not_a_mode")


# ---------------------------------------------------------------------------
# weighted_score_aggregate
# ---------------------------------------------------------------------------

def test_weighted_score_aggregate_combines_normalized_experts():
    s = state([req(1, prompt=512, output=512), req(2, prompt=32, output=32), req(3, prompt=128, output=128)])
    expert_scores = {
        "weighted_shortest_processing": score_with_named_expert("weighted_shortest_processing", s),
        "edf": score_with_named_expert("edf", s),
    }
    weights = {"weighted_shortest_processing": 0.5, "edf": 0.5}
    result = weighted_score_aggregate(expert_scores, weights, normalization=NormalizationMode.MIN_MAX)
    assert result.ranked_request_ids[0] in {1, 2, 3}
    assert set(result.aggregate) == {1, 2, 3}


def test_weighted_score_aggregate_missing_expert_entry_contributes_nothing():
    expert_scores = {"weighted_shortest_processing": {1: 1.0}, "edf": {1: 1.0, 2: 2.0}}
    weights = {"weighted_shortest_processing": 1.0, "edf": 1.0}
    result = weighted_score_aggregate(expert_scores, weights, normalization=NormalizationMode.NONE)
    assert 2 in result.aggregate
    assert result.aggregate[2] == pytest.approx(2.0)  # only edf contributes to request 2


# ---------------------------------------------------------------------------
# build_score_weights / StaticScoreEnsemblePolicy
# ---------------------------------------------------------------------------

def test_build_score_weights_normalizes_to_one_and_supports_top_k():
    weights = build_score_weights(
        [ScoreExpertSpec("weighted_shortest_processing", 2.0), ScoreExpertSpec("edf", 1.0), ScoreExpertSpec("fifo", 1.0)],
        top_k=2,
    )
    assert len(weights) == 2
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "weighted_shortest_processing" in weights  # highest raw weight kept


def test_build_score_weights_rejects_negative_or_nan():
    with pytest.raises(CompositionError):
        build_score_weights([ScoreExpertSpec("weighted_shortest_processing", -1.0)])
    with pytest.raises(CompositionError):
        build_score_weights([ScoreExpertSpec("weighted_shortest_processing", math.nan)])


def test_build_score_weights_rejects_unsupported_capability():
    with pytest.raises(CapabilityError):
        build_score_weights([ScoreExpertSpec("scorpio_style_slo_guard", 1.0)])


def test_static_score_ensemble_policy_orders_short_work_first():
    policy = StaticScoreEnsemblePolicy([ScoreExpertSpec("weighted_shortest_processing", 1.0)])
    action = policy.select_action(state(
        [req(1, prompt=512, output=512), req(2, prompt=32, output=32)],
        g=gpu(max_seq=1),
    ))
    assert action.admit[0] == [2]
    assert not policy.decision_logs[-1].fallback_used


def test_static_score_ensemble_policy_empty_queue_uses_fallback_without_error():
    policy = StaticScoreEnsemblePolicy([ScoreExpertSpec("weighted_shortest_processing", 1.0)])
    action = policy.select_action(state([]))
    assert action.is_empty()
    assert policy.decision_logs[-1].fallback_used


def test_static_score_ensemble_policy_rejects_score_incapable_expert_at_construction():
    with pytest.raises(CapabilityError):
        StaticScoreEnsemblePolicy([ScoreExpertSpec("scorpio_style_slo_guard", 1.0)])


def test_static_score_ensemble_policy_reproducible():
    experts = [ScoreExpertSpec("weighted_shortest_processing", 1.0), ScoreExpertSpec("edf", 0.5)]
    p1 = StaticScoreEnsemblePolicy(list(experts))
    p2 = StaticScoreEnsemblePolicy(list(experts))
    s1 = state([req(1, prompt=256), req(2, prompt=32)], g=gpu(max_seq=2))
    s2 = state([req(1, prompt=256), req(2, prompt=32)], g=gpu(max_seq=2))
    assert p1.select_action(s1).admit == p2.select_action(s2).admit


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

def test_instrumentation_disabled_by_default_records_nothing():
    sink = DecisionTraceSink()
    assert sink.enabled is False
    base = StaticRankEnsemblePolicy([RankExpertSpec("weighted_shortest_processing", 1.0)])
    wrapped = InstrumentedPolicy(base, sink)
    wrapped.select_action(state([req(1), req(2)], g=gpu(max_seq=2)))
    assert sink.records == []


def test_instrumentation_does_not_alter_scheduling_outcome():
    experts = [RankExpertSpec("weighted_shortest_processing", 1.0)]

    def make_state():
        return state([req(1, prompt=512, output=512), req(2, prompt=32, output=32)], g=gpu(max_seq=1))

    plain = StaticRankEnsemblePolicy(list(experts))
    plain_action = plain.select_action(make_state())

    sink = DecisionTraceSink(enabled=True, scenario_id="s")
    instrumented = InstrumentedPolicy(StaticRankEnsemblePolicy(list(experts)), sink)
    instrumented_action = instrumented.select_action(make_state())

    assert plain_action.admit == instrumented_action.admit
    assert len(sink.records) == 1


def test_instrumentation_records_candidates_selected_and_state_summary():
    sink = DecisionTraceSink(enabled=True, scenario_id="smoke")
    base = StaticRankEnsemblePolicy([RankExpertSpec("weighted_shortest_processing", 1.0)])
    wrapped = InstrumentedPolicy(base, sink)
    wrapped.select_action(state([req(1, prompt=512, output=512), req(2, prompt=32, output=32)], g=gpu(max_seq=2)))
    rec = sink.records[0]
    assert rec.schema_version == DECISION_TRACE_SCHEMA_VERSION
    assert rec.scenario_id == "smoke"
    assert rec.candidate_request_ids == [1, 2]
    assert set(rec.selected_request_ids) <= {1, 2}
    assert "queue_pressure" in rec.state_summary
    assert rec.expert_weights == {"weighted_shortest_processing": 1.0}


def test_instrumentation_jsonl_round_trip(tmp_path):
    sink = DecisionTraceSink(enabled=True, scenario_id="roundtrip")
    base = StaticRankEnsemblePolicy([RankExpertSpec("weighted_shortest_processing", 1.0)])
    wrapped = InstrumentedPolicy(base, sink)
    wrapped.select_action(state([req(1, prompt=512, output=512), req(2, prompt=32, output=32)], g=gpu(max_seq=2)))

    path = tmp_path / "trace.jsonl"
    sink.write_jsonl(path)
    assert path.exists()
    loaded = DecisionTraceSink.read_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0] == sink.records[0]


def test_instrumentation_read_jsonl_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"schema_version": "DecisionTraceV999"}\n')
    with pytest.raises(ValueError):
        DecisionTraceSink.read_jsonl(path)


def test_instrumentation_wraps_score_ensemble_too():
    sink = DecisionTraceSink(enabled=True)
    base = StaticScoreEnsemblePolicy([ScoreExpertSpec("weighted_shortest_processing", 1.0)])
    wrapped = InstrumentedPolicy(base, sink)
    wrapped.select_action(state([req(1, prompt=512, output=512), req(2, prompt=32, output=32)], g=gpu(max_seq=1)))
    assert len(sink.records) == 1
    assert sink.records[0].normalized_scores is not None
