"""
Tests for service_model_factory: config-driven service model building.

Covers:
  - synthetic model (default and explicit)
  - calibrated model loaded from fixture
  - missing calibration file → FileNotFoundError
  - unknown type → ValueError
  - smoke test: calibrated model runs a tiny simulation
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.calibrated_service_model import CalibratedServiceModel
from llmserveopt.simulator.service_model_factory import build_service_model_from_config

FIXTURE_CURVES = str(ROOT / "tests/fixtures/service_curves_fixture.json")


# ---------------------------------------------------------------------------
# Synthetic model
# ---------------------------------------------------------------------------

def test_default_is_synthetic():
    cfg = {}
    sm = build_service_model_from_config(cfg)
    assert isinstance(sm, ServiceModel)
    assert sm.enable_prefill_modeling is False


def test_explicit_synthetic():
    cfg = {"service_model": {"type": "synthetic", "enable_prefill_modeling": True}}
    sm = build_service_model_from_config(cfg)
    assert isinstance(sm, ServiceModel)
    assert sm.enable_prefill_modeling is True


def test_synthetic_passes_all_params():
    cfg = {
        "simulator": {"step_size": 0.002},
        "service_model": {
            "type": "synthetic",
            "enable_prefill_modeling": True,
            "prefill_cost_per_token": 2.5,
            "max_prefill_chunk_tokens": 128,
            "step_token_budget": 2048,
            "decode_first": True,
        },
    }
    sm = build_service_model_from_config(cfg)
    assert isinstance(sm, ServiceModel)
    assert sm.step_size == pytest.approx(0.002)
    assert sm.prefill_cost_per_token == pytest.approx(2.5)
    assert sm.max_prefill_chunk_tokens == 128
    assert sm.step_token_budget == 2048
    assert sm.decode_first is True


# ---------------------------------------------------------------------------
# Calibrated model
# ---------------------------------------------------------------------------

def test_calibrated_model_loads_from_fixture():
    cfg = {
        "service_model": {
            "type": "calibrated",
            "calibration_file": FIXTURE_CURVES,
        }
    }
    sm = build_service_model_from_config(cfg)
    assert isinstance(sm, CalibratedServiceModel)
    assert sm.enable_prefill_modeling is True


def test_calibrated_model_prefill_steps_positive():
    cfg = {
        "service_model": {
            "type": "calibrated",
            "calibration_file": FIXTURE_CURVES,
        }
    }
    sm = build_service_model_from_config(cfg)
    steps = sm.compute_prefill_steps(512)
    assert steps >= 1


def test_calibrated_model_decode_time_positive():
    cfg = {
        "service_model": {
            "type": "calibrated",
            "calibration_file": FIXTURE_CURVES,
        }
    }
    sm = build_service_model_from_config(cfg)
    t = sm.compute_decode_step_time(batch_size=4, context_tokens=256)
    assert t > 0.0


def test_calibrated_model_disable_prefill():
    cfg = {
        "service_model": {
            "type": "calibrated",
            "calibration_file": FIXTURE_CURVES,
            "enable_prefill_modeling": False,
        }
    }
    sm = build_service_model_from_config(cfg)
    assert sm.compute_prefill_steps(512) == 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_missing_calibration_file_raises():
    cfg = {
        "service_model": {
            "type": "calibrated",
            "calibration_file": "/nonexistent/path/curves.json",
        }
    }
    with pytest.raises(FileNotFoundError):
        build_service_model_from_config(cfg)


def test_unknown_type_raises():
    cfg = {"service_model": {"type": "banana"}}
    with pytest.raises(ValueError, match="Unknown service_model.type"):
        build_service_model_from_config(cfg)


# ---------------------------------------------------------------------------
# Smoke test: calibrated model in simulator
# ---------------------------------------------------------------------------

def test_calibrated_model_smoke_simulation():
    """Calibrated service model runs through a tiny trace end-to-end."""
    from llmserveopt.core.types import GPUConfig, Request
    from llmserveopt.policies.fifo import FIFOPolicy
    from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

    cfg = {
        "service_model": {
            "type": "calibrated",
            "calibration_file": FIXTURE_CURVES,
            "enable_prefill_modeling": True,
        }
    }
    sm = build_service_model_from_config(cfg)

    gpu = GPUConfig(gpu_id=0, max_active_sequences=4, max_batch_tokens=512, max_kv_tokens=2048)
    sim_cfg = SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=5000)
    sim = Simulator(sim_cfg)

    requests = [
        Request(
            request_id=i,
            arrival_time=float(i) * 0.05,
            prompt_tokens=64,
            predicted_output_tokens=32,
            actual_output_tokens=32,
            slo_deadline=float(i) * 0.05 + 10.0,
            priority=1.0,
            class_id="standard",
        )
        for i in range(5)
    ]
    sim.load_trace(requests)
    metrics = sim.run(FIFOPolicy(), workload_tag="smoke", seed=0)
    assert metrics.num_completed == 5
    assert metrics.mean_latency > 0
