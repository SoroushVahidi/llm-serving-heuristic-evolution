"""Tests for Family B v1 prefill/decode chunk-control separation family."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from llmserveopt.core.types import GPUConfig
from llmserveopt.policies.prefill_control_variants import (
    DEFAULT_CHUNK_LARGE,
    DEFAULT_CHUNK_SMALL,
    UNLIMITED_PREFILL_CHUNK,
    make_prefill_decode_variants,
)
from llmserveopt.policy_separation.builders import req
from llmserveopt.policy_separation.templates_prefill_decode import (
    BurstGPTUnavailableError,
    assert_policy_visible_fields_clean,
    case_prefill_decode_interference,
)
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "policy_separation_prefill_decode_pilot_v1.yaml"


def _scen(**kwargs):
    defaults = dict(
        prefill_size_class="long",
        decode_occupancy="high",
        slo_regime="tbt_tight",
        offered_load="moderate",
        seed=42,
        allow_synthetic_tokens=True,
    )
    defaults.update(kwargs)
    return case_prefill_decode_interference(**defaults)


def test_determinism_synthetic():
    a = _scen()
    b = _scen()
    assert a.scenario_id == b.scenario_id
    assert len(a.requests) == len(b.requests)
    for r1, r2 in zip(a.requests, b.requests):
        assert r1.arrival_time == r2.arrival_time
        assert r1.prompt_tokens == r2.prompt_tokens
        assert r1.actual_output_tokens == r2.actual_output_tokens
        assert r1.predicted_output_tokens == r2.predicted_output_tokens
        assert r1.class_id == r2.class_id


def test_unique_scenario_ids_full_config_grid():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    grid = cfg["sweep_grid"]
    fixed = cfg["fixed"]
    ids = []
    for psize in grid["prefill_size_class"]:
        for occ in grid["decode_occupancy"]:
            for slo in grid["slo_regime"]:
                for load in grid["offered_load"]:
                    for seed in grid["seeds"]:
                        s = case_prefill_decode_interference(
                            prefill_size_class=psize,
                            decode_occupancy=occ,
                            slo_regime=slo,
                            offered_load=load,
                            seed=seed,
                            n_decode=fixed.get("n_decode"),
                            n_prefill=fixed.get("n_prefill"),
                            max_active_sequences=fixed["max_active_sequences"],
                            step_token_budget=fixed["step_token_budget"],
                            allow_synthetic_tokens=True,
                        )
                        ids.append(s.scenario_id)
    assert len(ids) == 144
    assert len(set(ids)) == 144


def test_anti_leakage_policy_visible_fields():
    s = _scen(prefill_size_class="mixed", slo_regime="ttft_tight")
    assert_policy_visible_fields_clean(s)
    for r in s.requests:
        assert r.class_id in {"tenant_prefill", "tenant_decode"}
        assert "ttft" not in r.class_id
        assert "long" not in r.class_id
        assert "high" not in r.class_id


def test_prefill_convoy_arrives_before_decode_overlap():
    s = _scen(decode_occupancy="high")
    pref = [r for r in s.requests if r.class_id == "tenant_prefill"]
    dec = [r for r in s.requests if r.class_id == "tenant_decode"]
    assert pref and dec
    assert min(r.arrival_time for r in pref) <= min(r.arrival_time for r in dec)
    assert s.params["arrival_shape"] == "prefill_convoy_then_overlapping_decode"


def test_prefill_size_class_windows():
    for psize, lo, hi in (
        ("short", 256, 1024),
        ("medium", 1024, 4096),
        ("long", 4096, 16384),
    ):
        s = _scen(prefill_size_class=psize, seed=7)
        prompts = [r.prompt_tokens for r in s.requests if r.class_id == "tenant_prefill"]
        assert prompts
        assert all(lo <= p <= hi for p in prompts)


def test_mixed_prefill_interleaves_short_and_long():
    s = _scen(prefill_size_class="mixed", seed=3)
    prompts = [r.prompt_tokens for r in s.requests if r.class_id == "tenant_prefill"]
    assert any(p < 1024 for p in prompts)
    assert any(p >= 4096 for p in prompts)


def test_production_fails_without_burstgpt(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_SERVEOPT_BURSTGPT_CSV", raising=False)
    with pytest.raises(BurstGPTUnavailableError):
        case_prefill_decode_interference(
            prefill_size_class="long",
            decode_occupancy="low",
            slo_regime="balanced",
            offered_load="moderate",
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
        # Cover long-prefill and short-decode windows.
        req_tok = 5000 + (i % 100) if i % 2 == 0 else 100 + (i % 50)
        resp = 200 + (i % 40)
        lines.append(f"{i},ChatGPT,{req_tok},{resp},{req_tok + resp},Conversation log\n")
    csv_path.write_text("".join(lines), encoding="utf-8")
    mod._load_burstgpt_arrays.cache_clear()
    monkeypatch.setenv("LLM_SERVEOPT_BURSTGPT_CSV", str(csv_path))
    s = case_prefill_decode_interference(
        prefill_size_class="long",
        decode_occupancy="medium",
        slo_regime="tbt_tight",
        offered_load="moderate",
        seed=1,
        allow_synthetic_tokens=False,
        datasets_root=tmp_path,
    )
    sources = s.params["token_sources"]
    assert sources["burstgpt_path"].endswith("BurstGPT_without_fails_1.csv")
    assert sources["prefill_prompt"] in {"burstgpt_staged", "burstgpt_anchored"}
    assert sources["decode_output"] == "burstgpt_anchored"  # prefer_real=False


def test_mechanism_variant_kwargs():
    variants = make_prefill_decode_variants()
    assert set(variants) == {
        "full_prefill",
        "chunked_prefill_small",
        "chunked_prefill_large",
        "decode_priority_chunked",
        "adaptive_prefill_control",
    }
    assert variants["full_prefill"][1]["max_prefill_chunk_tokens"] == UNLIMITED_PREFILL_CHUNK
    assert variants["full_prefill"][1]["decode_first"] is False
    assert variants["chunked_prefill_small"][1]["max_prefill_chunk_tokens"] == DEFAULT_CHUNK_SMALL
    assert variants["chunked_prefill_large"][1]["max_prefill_chunk_tokens"] == DEFAULT_CHUNK_LARGE
    assert variants["decode_priority_chunked"][1]["decode_first"] is True
    assert variants["chunked_prefill_small"][1]["max_prefill_chunk_tokens"] < (
        variants["chunked_prefill_large"][1]["max_prefill_chunk_tokens"]
    )
    assert variants["chunked_prefill_large"][1]["max_prefill_chunk_tokens"] < UNLIMITED_PREFILL_CHUNK


def test_full_vs_chunk_semantics_on_early_prefill_late_decode_microbench():
    """GPU-level: unlimited shared chunk can defer a later decode; small chunk does not."""
    from llmserveopt.simulator.gpu import GPUState
    from llmserveopt.simulator.request import InternalRequest, RequestPhase
    from llmserveopt.core.types import Request

    def make_ir(rid, arrival, prompt, pref_rem, tokens_decoded=0, out=10):
        r = Request(
            request_id=rid,
            arrival_time=arrival,
            prompt_tokens=prompt,
            predicted_output_tokens=out,
            actual_output_tokens=out,
            slo_deadline=arrival + 100000,
            priority=1.0,
            class_id="t",
        )
        return InternalRequest(
            request=r,
            phase=RequestPhase.ACTIVE,
            gpu_id=0,
            admission_time=arrival,
            prefill_remaining=pref_rem,
            tokens_decoded=tokens_decoded,
            first_token_time=(arrival if tokens_decoded > 0 else -1.0),
        )

    gpu_cfg = GPUConfig(
        gpu_id=0,
        max_active_sequences=64,
        max_batch_tokens=1_000_000,
        max_kv_tokens=1_000_000,
    )

    def step(chunk, decode_first):
        gpu = GPUState(gpu_cfg)
        gpu._active = {
            0: make_ir(0, 0.0, 2000, 1500),
            1: make_ir(1, 5.0, 50, 0, tokens_decoded=1),
        }
        sm = ServiceModel(
            enable_prefill_modeling=True,
            decode_first=decode_first,
            step_token_budget=512,
            max_prefill_chunk_tokens=chunk,
            enable_decode_prefill_contention=True,
        )
        gpu.step(current_time=6.0, service_model=sm)
        return gpu._active[1].tokens_decoded, gpu.step_contention_diagnostics[-1]

    dec_full, d_full = step(UNLIMITED_PREFILL_CHUNK, False)
    dec_small, d_small = step(DEFAULT_CHUNK_SMALL, False)
    dec_pri, d_pri = step(DEFAULT_CHUNK_SMALL, True)
    assert dec_full == 1 and d_full.decode_tokens_deferred == 1
    assert dec_small == 2 and d_small.decode_tokens_deferred == 0
    assert dec_pri == 2 and d_pri.decode_tokens_deferred == 0


def test_config_chunk_budgets_match_defaults():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["chunk_budgets"]["chunk_small"] == DEFAULT_CHUNK_SMALL
    assert cfg["chunk_budgets"]["chunk_large"] == DEFAULT_CHUNK_LARGE
    assert cfg["fixed"]["step_token_budget"] == 512


def test_metric_schema_smoke_run():
    s = _scen(
        prefill_size_class="medium",
        decode_occupancy="low",
        slo_regime="balanced",
        offered_load="moderate",
        seed=1,
        n_prefill=8,
        n_decode=8,
    )
    variants = make_prefill_decode_variants()
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
    assert 0.0 <= metrics.completion_fraction <= 1.0
    summary = sim.contention_diagnostics_summary()
    assert "decode_stalled_steps" in summary
    assert "prefill_stalled_steps" in summary


def test_no_illegal_kv_overflow_on_small_grid():
    s = _scen(n_prefill=5, n_decode=5, prefill_size_class="short")
    for name, (policy, kw) in make_prefill_decode_variants().items():
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
        assert metrics.completion_fraction >= 0.0
        for g in sim._gpus:  # noqa: SLF001
            assert g.current_kv_tokens <= g.config.max_kv_tokens
