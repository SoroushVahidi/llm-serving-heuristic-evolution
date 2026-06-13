"""
Unit tests for calibration utilities (no GPU required).

Tests:
1.  test_prompt_generator_length          — mock tokenizer, verify output
2.  test_prefill_curve_fit                — fake linear data, check RMSE < 0.01
3.  test_decode_curve_fit                 — fake linear data, check params
4.  test_service_curves_serialization     — save/load round-trip
5.  test_calibrated_service_model_load    — load fixture, call compute_prefill_steps
6.  test_calibrated_service_model_missing — FileNotFoundError on missing file
7.  test_calibrated_service_model_oor     — out-of-range returns clamped int >= 1
8.  test_simulator_params_derivation      — fake curves, check required keys
9.  test_lookup_table_build               — fake df, check structure
10. test_fit_report_serialization         — JSON serializable
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.calibration.curve_fitting import (
    FitResult,
    ServiceCurves,
    build_lookup_table,
    fit_decode_curve,
    fit_prefill_curve,
    generate_fit_report,
    load_service_curves,
    save_service_curves,
)
from llmserveopt.calibration.prompt_generator import generate_prompt_of_length
from llmserveopt.calibration.simulator_adapter import derive_simulator_params
from llmserveopt.simulator.calibrated_service_model import CalibratedServiceModel

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "service_curves_fixture.json"


# ---------------------------------------------------------------------------
# 1. Prompt generator
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer mock for testing prompt_generator without transformers."""

    def __init__(self) -> None:
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def __call__(self, text: str, add_special_tokens: bool = True):
        # Simple word-split tokenizer: each space-separated token → 1 id
        words = text.split()
        ids = [hash(w) % 50000 for w in words]
        return {"input_ids": ids}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        # Return a fixed-width text that decodes to len(ids) words
        return " ".join([f"w{x}" for x in ids])


def test_prompt_generator_length():
    """Prompt generator returns a dict with realized_length close to target."""
    tok = MockTokenizer()
    for target in [32, 128, 512]:
        result = generate_prompt_of_length(tok, target, seed=42)
        assert isinstance(result, dict)
        assert "text" in result
        assert "input_ids" in result
        assert "realized_length" in result
        assert len(result["input_ids"]) == target
        # realized_length allowed ±1
        assert abs(result["realized_length"] - target) <= 1, (
            f"target={target}, realized={result['realized_length']}"
        )


def test_prompt_generator_positive_length():
    """Raises ValueError for non-positive target."""
    tok = MockTokenizer()
    with pytest.raises(ValueError):
        generate_prompt_of_length(tok, 0)


# ---------------------------------------------------------------------------
# 2. Prefill curve fit
# ---------------------------------------------------------------------------

def _make_prefill_df(a0: float = 0.002, a1: float = 1.5e-6, noise: float = 1e-6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    prompt_tokens = np.repeat([32, 128, 256, 512, 1024, 2048], 5)
    prefill_time_s = a0 + a1 * prompt_tokens + rng.normal(0, noise, len(prompt_tokens))
    return pd.DataFrame({
        "prompt_tokens": prompt_tokens,
        "output_tokens": 64,
        "batch_size": 1,
        "prefill_time_s": prefill_time_s,
        "decode_time_per_token_s": 0.001,
        "skipped": False,
    })


def test_prefill_curve_fit():
    """Linear prefill fit recovers ground-truth params with RMSE < 0.01."""
    a0_true, a1_true = 0.002, 1.5e-6
    df = _make_prefill_df(a0_true, a1_true, noise=1e-8)
    fit = fit_prefill_curve(df)

    assert fit.fit_method == "linear"
    assert abs(fit.params["a0"] - a0_true) < 1e-4, f"a0={fit.params['a0']}"
    assert abs(fit.params["a1"] - a1_true) < 1e-8, f"a1={fit.params['a1']}"
    assert fit.rmse < 0.01
    assert fit.r_squared > 0.99
    assert fit.n_samples == 30


# ---------------------------------------------------------------------------
# 3. Decode curve fit
# ---------------------------------------------------------------------------

def _make_decode_df(
    b0: float = 0.0008,
    b1: float = 0.0002,
    b2: float = 1e-7,
    noise: float = 1e-8,
) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    prompt_tokens_list = []
    output_tokens_list = []
    batch_sizes_list = []
    decode_times = []
    for p in [32, 128, 512, 1024]:
        for o in [16, 64, 128]:
            for bs in [1, 2, 4, 8]:
                for _ in range(3):
                    ctx = p + o
                    t = b0 + b1 * bs + b2 * ctx + rng.normal(0, noise)
                    prompt_tokens_list.append(p)
                    output_tokens_list.append(o)
                    batch_sizes_list.append(bs)
                    decode_times.append(t)
    return pd.DataFrame({
        "prompt_tokens": prompt_tokens_list,
        "output_tokens": output_tokens_list,
        "batch_size": batch_sizes_list,
        "prefill_time_s": 0.002,
        "decode_time_per_token_s": decode_times,
        "skipped": False,
    })


def test_decode_curve_fit():
    """Linear decode fit recovers ground-truth params."""
    b0, b1, b2 = 0.0008, 0.0002, 1e-7
    df = _make_decode_df(b0, b1, b2, noise=1e-10)
    fit = fit_decode_curve(df)

    assert fit.fit_method == "linear"
    assert abs(fit.params["b0"] - b0) < 1e-5
    assert abs(fit.params["b1"] - b1) < 1e-5
    assert abs(fit.params["b2"] - b2) < 1e-9
    assert fit.rmse < 0.01
    assert fit.r_squared > 0.99


# ---------------------------------------------------------------------------
# 4. ServiceCurves round-trip serialization
# ---------------------------------------------------------------------------

def _make_service_curves() -> ServiceCurves:
    prefill = FitResult(
        params={"a0": 0.002, "a1": 1.5e-6},
        rmse=0.0001, mape=2.5, max_error=0.0003, r_squared=0.99,
        n_samples=100, fit_method="linear",
    )
    decode = FitResult(
        params={"b0": 0.0008, "b1": 0.0002, "b2": 1e-7},
        rmse=5e-5, mape=1.5, max_error=0.0001, r_squared=0.98,
        n_samples=100, fit_method="linear",
    )
    return ServiceCurves(
        prefill=prefill,
        decode=decode,
        lookup_tables={"prefill_table": [], "decode_table": []},
        step_size=0.001,
        model_name="test-model",
        fit_timestamp="2026-06-10T00:00:00+00:00",
    )


def test_service_curves_serialization():
    """save_service_curves / load_service_curves round-trip."""
    curves = _make_service_curves()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "curves.json"
        save_service_curves(curves, path)
        loaded = load_service_curves(path)

    assert loaded.model_name == curves.model_name
    assert loaded.step_size == curves.step_size
    assert abs(loaded.prefill.params["a0"] - curves.prefill.params["a0"]) < 1e-12
    assert abs(loaded.prefill.params["a1"] - curves.prefill.params["a1"]) < 1e-15
    assert abs(loaded.decode.params["b0"] - curves.decode.params["b0"]) < 1e-12
    assert loaded.prefill.r_squared == curves.prefill.r_squared


# ---------------------------------------------------------------------------
# 5. CalibratedServiceModel loads and returns valid steps
# ---------------------------------------------------------------------------

def test_calibrated_service_model_load():
    """Load fixture, call compute_prefill_steps, check returns int >= 1."""
    csm = CalibratedServiceModel(calibration_file=FIXTURE_PATH, step_size=0.001)
    steps_128 = csm.compute_prefill_steps(128)
    steps_512 = csm.compute_prefill_steps(512)

    assert isinstance(steps_128, int)
    assert isinstance(steps_512, int)
    assert steps_128 >= 1
    assert steps_512 >= 1
    # Longer prompts should take more steps
    assert steps_512 >= steps_128


# ---------------------------------------------------------------------------
# 6. CalibratedServiceModel raises FileNotFoundError on missing file
# ---------------------------------------------------------------------------

def test_calibrated_service_model_missing_file():
    """FileNotFoundError raised when calibration file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        CalibratedServiceModel(calibration_file="/nonexistent/path/curves.json")


# ---------------------------------------------------------------------------
# 7. CalibratedServiceModel out-of-range values return clamped result
# ---------------------------------------------------------------------------

def test_calibrated_service_model_out_of_range():
    """Very large prompt tokens extrapolates but returns int >= 1 (no crash)."""
    csm = CalibratedServiceModel(calibration_file=FIXTURE_PATH, step_size=0.001)
    # Way beyond calibration grid
    steps = csm.compute_prefill_steps(100_000)
    assert isinstance(steps, int)
    assert steps >= 1
    assert steps <= csm.max_prefill_steps


def test_calibrated_service_model_zero_tokens():
    """Zero prompt tokens returns 0 steps."""
    csm = CalibratedServiceModel(calibration_file=FIXTURE_PATH, step_size=0.001)
    assert csm.compute_prefill_steps(0) == 0


def test_calibrated_service_model_disable_prefill():
    """enable_prefill_modeling=False always returns 0."""
    csm = CalibratedServiceModel(
        calibration_file=FIXTURE_PATH,
        step_size=0.001,
        enable_prefill_modeling=False,
    )
    assert csm.compute_prefill_steps(512) == 0
    assert csm.compute_prefill_steps(0) == 0


# ---------------------------------------------------------------------------
# 8. Simulator params derivation
# ---------------------------------------------------------------------------

def test_simulator_params_derivation():
    """derive_simulator_params returns dict with required keys."""
    curves = _make_service_curves()
    result = derive_simulator_params(curves)

    assert "simulator_params" in result
    sp = result["simulator_params"]
    assert "step_size" in sp
    assert "prefill_cost_per_token" in sp
    assert "decode_steps_per_token" in sp
    assert "step_size_suggestion" in sp

    assert sp["prefill_cost_per_token"] > 0
    assert sp["decode_steps_per_token"] > 0
    assert sp["step_size"] == 0.001


# ---------------------------------------------------------------------------
# 9. Lookup table build
# ---------------------------------------------------------------------------

def test_lookup_table_build():
    """build_lookup_table returns dict with prefill_table and decode_table."""
    df = _make_prefill_df()
    df["decode_time_per_token_s"] = 0.001
    df["output_tokens"] = 64
    tables = build_lookup_table(df)

    assert "prefill_table" in tables
    assert "decode_table" in tables
    assert isinstance(tables["prefill_table"], list)
    assert isinstance(tables["decode_table"], list)
    # Should have at least one row per unique prompt_tokens
    n_unique = df["prompt_tokens"].nunique()
    assert len(tables["prefill_table"]) >= 1


# ---------------------------------------------------------------------------
# 10. Fit report JSON serializable
# ---------------------------------------------------------------------------

def test_fit_report_serialization():
    """generate_fit_report returns a JSON-serializable dict."""
    curves = _make_service_curves()
    report = generate_fit_report(curves)

    # Must not raise
    serialized = json.dumps(report)
    reloaded = json.loads(serialized)

    assert "prefill" in reloaded
    assert "decode" in reloaded
    assert reloaded["prefill"]["r_squared"] == curves.prefill.r_squared
    assert reloaded["decode"]["r_squared"] == curves.decode.r_squared
