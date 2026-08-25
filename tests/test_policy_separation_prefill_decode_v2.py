"""Tests for Family B v2 prefill/decode TTFT-contention refinement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from llmserveopt.core.types import (
    ObservableGPUState,
    ObservableRequest,
    ObservableState,
)
from llmserveopt.policies.prefill_control_variants import (
    DEFAULT_CHUNK_SMALL,
    UNLIMITED_PREFILL_CHUNK,
    make_prefill_decode_variants,
    make_prefill_decode_variants_v2,
)
from llmserveopt.policy_separation.templates_prefill_decode import BurstGPTUnavailableError
from llmserveopt.policy_separation.templates_prefill_decode_v2 import (
    ALLOWED_CLASS_IDS,
    CLASS_HOG,
    CLASS_LATE,
    assert_policy_visible_fields_clean_v2,
    case_prefill_decode_ttft_contention,
)
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PILOT = ROOT / "configs" / "policy_separation_prefill_decode_pilot_v2.yaml"
CONFIG_SMOKE = ROOT / "configs" / "policy_separation_prefill_decode_smoke_v2.yaml"


def _scen(**kwargs):
    defaults = dict(
        hog_count="low",
        late_pressure="low",
        slo_emphasis="hog_ttft",
        seed=42,
        allow_synthetic_tokens=True,
    )
    defaults.update(kwargs)
    return case_prefill_decode_ttft_contention(**defaults)


def test_v2_policy_set_is_exactly_two_anchors():
    v2 = make_prefill_decode_variants_v2()
    assert set(v2) == {"full_prefill", "chunked_prefill_small"}
    assert v2["full_prefill"][1]["max_prefill_chunk_tokens"] == UNLIMITED_PREFILL_CHUNK
    assert v2["full_prefill"][1]["decode_first"] is False
    assert v2["chunked_prefill_small"][1]["max_prefill_chunk_tokens"] == DEFAULT_CHUNK_SMALL
    assert v2["chunked_prefill_small"][1]["decode_first"] is False
    # v1 factory remains intact for frozen-run replay.
    v1 = make_prefill_decode_variants()
    assert "decode_priority_chunked" in v1
    assert "chunked_prefill_large" in v1
    assert "adaptive_prefill_control" in v1


def test_v2_policies_use_only_online_observables():
    """Greedy arrival ranking uses arrival_time + request_id only."""
    policy, _ = make_prefill_decode_variants_v2()["full_prefill"]
    waiting = [
        ObservableRequest(
            request_id=2,
            arrival_time=1.0,
            prompt_tokens=128,
            predicted_output_tokens=80,
            slo_deadline=3.0,
            priority=1.0,
            class_id=CLASS_LATE,
        ),
        ObservableRequest(
            request_id=1,
            arrival_time=0.0,
            prompt_tokens=8192,
            predicted_output_tokens=80,
            slo_deadline=0.2,
            priority=1.0,
            class_id=CLASS_HOG,
        ),
    ]
    gpu = ObservableGPUState(
        gpu_id=0,
        max_active_sequences=64,
        max_batch_tokens=1_000_000,
        max_kv_tokens=1_000_000,
        active_request_ids=[],
        active_requests_info=[],
        current_kv_tokens=0,
        tokens_decoded_per_request={},
    )
    state = ObservableState(
        time=0.0,
        waiting_queue=waiting,
        gpu_states=[gpu],
        completed_count=0,
        step=0,
    )
    action = policy.select_action(state)
    # Must not prefer the tight hog deadline or class_id over arrival order.
    admitted_ids = list(action.admit.get(0, []))
    assert admitted_ids[0] == 1


def test_determinism_synthetic():
    a = _scen()
    b = _scen()
    assert a.scenario_id == b.scenario_id
    assert len(a.requests) == len(b.requests)
    for r1, r2 in zip(a.requests, b.requests):
        assert r1.arrival_time == r2.arrival_time
        assert r1.prompt_tokens == r2.prompt_tokens
        assert r1.actual_output_tokens == r2.actual_output_tokens
        assert r1.class_id == r2.class_id


def test_unique_scenario_ids_full_config_grid():
    cfg = yaml.safe_load(CONFIG_PILOT.read_text(encoding="utf-8"))
    grid = cfg["sweep_grid"]
    ids = []
    for hog in grid["hog_count"]:
        for late in grid["late_pressure"]:
            for slo in grid["slo_emphasis"]:
                for seed in grid["seeds"]:
                    s = case_prefill_decode_ttft_contention(
                        hog_count=hog,
                        late_pressure=late,
                        slo_emphasis=slo,
                        seed=seed,
                        allow_synthetic_tokens=True,
                    )
                    ids.append(s.scenario_id)
    assert len(ids) == 32
    assert len(set(ids)) == 32
    assert cfg["policies"] == ["full_prefill", "chunked_prefill_small"]
    assert 20260823 in grid["seeds"]


def test_no_duplicate_cells_in_smoke_config():
    cfg = yaml.safe_load(CONFIG_SMOKE.read_text(encoding="utf-8"))
    grid = cfg["sweep_grid"]
    ids = []
    for hog in grid["hog_count"]:
        for late in grid["late_pressure"]:
            for slo in grid["slo_emphasis"]:
                for seed in grid["seeds"]:
                    s = case_prefill_decode_ttft_contention(
                        hog_count=hog,
                        late_pressure=late,
                        slo_emphasis=slo,
                        seed=seed,
                        allow_synthetic_tokens=True,
                    )
                    ids.append(s.scenario_id)
    assert len(ids) == 8
    assert len(set(ids)) == 8


def test_anti_leakage_policy_visible_fields():
    s = _scen(slo_emphasis="late_ttft", late_pressure="high")
    assert_policy_visible_fields_clean_v2(s)
    for r in s.requests:
        assert r.class_id in ALLOWED_CLASS_IDS
        assert "hog_ttft" not in r.class_id
        assert "late_ttft" not in r.class_id
        assert "slo_emphasis" not in r.class_id
        assert str(s.seed) not in r.class_id
        assert s.scenario_id not in r.class_id


def test_arrival_shape_hog_then_midoverlap_late():
    s = _scen(hog_count="high", late_pressure="high")
    hog = [r for r in s.requests if r.class_id == CLASS_HOG]
    late = [r for r in s.requests if r.class_id == CLASS_LATE]
    assert hog and late
    assert min(r.arrival_time for r in hog) <= min(r.arrival_time for r in late)
    convoy_end = max(r.arrival_time for r in hog)
    assert min(r.arrival_time for r in late) < convoy_end
    assert s.params["arrival_shape"] == "hog_convoy_midoverlap_late_tenants"
    assert s.params["output_intervention"] == "synthetic_short_output_for_ttft_isolation"


def test_prompt_windows_and_short_outputs():
    s = _scen(seed=7)
    hog_p = [r.prompt_tokens for r in s.requests if r.class_id == CLASS_HOG]
    late_p = [r.prompt_tokens for r in s.requests if r.class_id == CLASS_LATE]
    outs = [r.actual_output_tokens for r in s.requests]
    assert hog_p and all(4096 <= p <= 16384 for p in hog_p)
    assert late_p and all(64 <= p <= 256 for p in late_p)
    assert all(48 <= o <= 128 for o in outs)


def test_slo_emphasis_changes_observable_deadlines_not_class_labels():
    hog = _scen(slo_emphasis="hog_ttft", seed=1)
    late = _scen(slo_emphasis="late_ttft", seed=1)
    hog_h = [r for r in hog.requests if r.class_id == CLASS_HOG]
    hog_l = [r for r in hog.requests if r.class_id == CLASS_LATE]
    late_h = [r for r in late.requests if r.class_id == CLASS_HOG]
    late_l = [r for r in late.requests if r.class_id == CLASS_LATE]
    hog_slack_h = np.mean([r.slo_deadline - r.arrival_time for r in hog_h])
    hog_slack_l = np.mean([r.slo_deadline - r.arrival_time for r in hog_l])
    late_slack_h = np.mean([r.slo_deadline - r.arrival_time for r in late_h])
    late_slack_l = np.mean([r.slo_deadline - r.arrival_time for r in late_l])
    assert hog_slack_h < hog_slack_l
    assert late_slack_l < late_slack_h
    assert {r.class_id for r in hog.requests} == {CLASS_HOG, CLASS_LATE}


def test_production_fails_without_burstgpt(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_SERVEOPT_BURSTGPT_CSV", raising=False)
    with pytest.raises(BurstGPTUnavailableError):
        case_prefill_decode_ttft_contention(
            hog_count="low",
            late_pressure="low",
            slo_emphasis="hog_ttft",
            seed=1,
            allow_synthetic_tokens=False,
            datasets_root=tmp_path,
        )


def test_burstgpt_path_and_provenance(tmp_path, monkeypatch):
    from llmserveopt.policy_separation import templates_prefill_decode as mod

    raw = tmp_path / "burstgpt_v2" / "raw"
    raw.mkdir(parents=True)
    csv_path = raw / "BurstGPT_without_fails_1.csv"
    lines = ["Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"]
    for i in range(200):
        req_tok = 5000 + (i % 100) if i % 2 == 0 else 80 + (i % 40)
        resp = 60 + (i % 20)
        lines.append(f"{i},ChatGPT,{req_tok},{resp},{req_tok + resp},Conversation log\n")
    csv_path.write_text("".join(lines), encoding="utf-8")
    mod._load_burstgpt_arrays.cache_clear()
    monkeypatch.setenv("LLM_SERVEOPT_BURSTGPT_CSV", str(csv_path))
    s = case_prefill_decode_ttft_contention(
        hog_count="low",
        late_pressure="high",
        slo_emphasis="late_ttft",
        seed=1,
        allow_synthetic_tokens=False,
        datasets_root=tmp_path,
    )
    sources = s.params["token_sources"]
    assert sources["burstgpt_path"].endswith("BurstGPT_without_fails_1.csv")
    assert sources["hog_prompt"] in {"burstgpt_staged", "burstgpt_anchored"}
    assert sources["late_prompt"] in {"burstgpt_staged", "burstgpt_anchored"}
    assert sources["output_intervention"] == "synthetic_short_output_for_ttft_isolation"
    assert sources["hog_output"] in {"burstgpt_anchored", "synthetic_lognormal"}


def test_metric_schema_and_contention_activation_smoke_run():
    s = _scen(n_hog=8, n_late=8, seed=1)
    variants = make_prefill_decode_variants_v2()
    policy, kw = variants["full_prefill"]
    merged = dict(s.service_model_kwargs)
    merged.update(kw)
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(s.gpu_configs),
            service_model=ServiceModel(**merged),
        )
    )
    sim.load_trace(list(s.requests))
    metrics = sim.run(policy, workload_tag=s.scenario_id, seed=s.seed)
    assert np.isfinite(metrics.arrival_normalized_weighted_goodput)
    assert metrics.completion_fraction == 1.0
    summary = sim.contention_diagnostics_summary()
    assert "decode_stalled_steps" in summary
    assert "prefill_stalled_steps" in summary
    # Prefill-side contention should fire on a hog convoy; decode-stall may
    # remain 0 (expected FCFS equilibrium).
    assert summary["prefill_stalled_steps"] > 0 or summary["budget_saturation_fraction"] > 0


def test_no_kv_infeasibility_on_small_grid():
    s = _scen(n_hog=5, n_late=5, hog_count="low")
    for name, (policy, kw) in make_prefill_decode_variants_v2().items():
        merged = dict(s.service_model_kwargs)
        merged.update(kw)
        sim = Simulator(
            SimulatorConfig(
                gpu_configs=list(s.gpu_configs),
                service_model=ServiceModel(**merged),
            )
        )
        sim.load_trace(list(s.requests))
        metrics = sim.run(policy, workload_tag=name, seed=s.seed)
        assert metrics.completion_fraction == 1.0
        for g in sim._gpus:  # noqa: SLF001
            assert g.current_kv_tokens <= g.config.max_kv_tokens


def test_micro_both_anchors_can_win_under_opposite_slo_emphasis():
    """Tiny deterministic cells: full should beat small under hog_ttft,
    small should beat full under late_ttft, by a practical margin.
    If this fails, the v2 geometry is not discriminative.
    """
    variants = make_prefill_decode_variants_v2()

    def anwg(slo: str, name: str) -> float:
        s = case_prefill_decode_ttft_contention(
            hog_count="low",
            late_pressure="high",
            slo_emphasis=slo,
            seed=3,
            n_hog=12,
            n_late=40,
            allow_synthetic_tokens=True,
        )
        policy, kw = variants[name]
        merged = dict(s.service_model_kwargs)
        merged.update(kw)
        sim = Simulator(
            SimulatorConfig(
                gpu_configs=list(s.gpu_configs),
                service_model=ServiceModel(**merged),
            )
        )
        sim.load_trace(list(s.requests))
        metrics = sim.run(policy, workload_tag=s.scenario_id, seed=s.seed)
        return float(metrics.arrival_normalized_weighted_goodput)

    full_hog = anwg("hog_ttft", "full_prefill")
    small_hog = anwg("hog_ttft", "chunked_prefill_small")
    full_late = anwg("late_ttft", "full_prefill")
    small_late = anwg("late_ttft", "chunked_prefill_small")
    assert full_hog - small_hog > 0.01
    assert small_late - full_late > 0.01
