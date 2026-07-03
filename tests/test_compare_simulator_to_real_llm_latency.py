"""Tests for scripts/compare_simulator_to_real_llm_latency.py on small
synthetic fixtures. No network access, no live pilot data, no provider SDK
required — everything here is fabricated in tmp_path.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "compare_simulator_to_real_llm_latency",
        ROOT / "scripts" / "compare_simulator_to_real_llm_latency.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_fitted_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "fitted.yaml"
    path.write_text(yaml.safe_dump({
        "providers": {
            "cohere": {
                "model": "command-r7b-12-2024",
                "ttft_seconds": {"mean": 0.246, "p50": 0.235, "p95": 0.336, "p99": 0.387},
                "decode_rate_overall": {"effective_decode_rate_tokens_per_sec": 88.5, "r2": 0.887, "n": 108},
            },
            "gemini": {
                "model": "gemini-3.1-flash-lite",
                "ttft_seconds": {"mean": 0.674, "p50": 0.601, "p95": 1.268, "p99": 1.792},
                "decode_rate_overall": {"effective_decode_rate_tokens_per_sec": 288.9, "r2": 0.833, "n": 108},
            },
        },
    }))
    return path


def _make_fitted_model_dir(tmp_path: Path) -> Path:
    out_dir = tmp_path / "fit_dir"
    out_dir.mkdir()
    raw_by_target = []
    for provider, ttft in (("cohere", 0.246), ("gemini", 0.674)):
        for target, latency in zip((64, 128, 256), (0.9, 1.8, 3.0)):
            raw_by_target.append({
                "provider": provider, "target_output_tokens": target, "n": 36,
                "mean_latency_s": latency, "mean_ttft_s": ttft,
                "mean_output_tokens": target * 0.95,
            })
    (out_dir / "latency_model_fit_v2.json").write_text(json.dumps({
        "raw_latency_by_target": raw_by_target,
    }))
    return out_dir


# ---------------------------------------------------------------------------
# Reading fitted v2 outputs
# ---------------------------------------------------------------------------

def test_load_fitted_yaml_parses_providers(tmp_path):
    mod = _load_module()
    path = _make_fitted_yaml(tmp_path)
    data = mod.load_fitted_yaml(path)
    assert data is not None
    assert set(data["providers"].keys()) == {"cohere", "gemini"}
    assert data["providers"]["cohere"]["decode_rate_overall"]["effective_decode_rate_tokens_per_sec"] == 88.5


def test_load_fitted_yaml_missing_file_returns_none(tmp_path):
    mod = _load_module()
    assert mod.load_fitted_yaml(tmp_path / "does_not_exist.yaml") is None


def test_load_fitted_model_json_and_raw_by_target(tmp_path):
    mod = _load_module()
    fit_dir = _make_fitted_model_dir(tmp_path)
    payload = mod.load_fitted_model_json(fit_dir)
    assert payload is not None
    raw = mod.raw_latency_by_target(payload)
    assert len(raw) == 6
    assert {r["provider"] for r in raw} == {"cohere", "gemini"}


def test_raw_latency_by_target_handles_missing_field_gracefully(tmp_path):
    mod = _load_module()
    assert mod.raw_latency_by_target(None) == []
    assert mod.raw_latency_by_target({"some_other_key": 1}) == []


def test_load_fitted_model_json_missing_dir_returns_none(tmp_path):
    mod = _load_module()
    assert mod.load_fitted_model_json(tmp_path / "nope") is None


# ---------------------------------------------------------------------------
# Simulator service-model instantiation (no live calls, no network)
# ---------------------------------------------------------------------------

def test_build_synthetic_service_model_decode_rate():
    mod = _load_module()
    model = mod.build_synthetic_service_model(step_size=0.002)
    assert mod.synthetic_decode_rate_tokens_per_sec(model) == pytest.approx(500.0)


def test_synthetic_prefill_is_zero_by_default():
    mod = _load_module()
    model = mod.build_synthetic_service_model(step_size=0.001)
    assert mod.synthetic_prefill_seconds(model, prompt_tokens=512) == 0.0


def test_build_calibrated_service_model_missing_file_returns_none(tmp_path):
    mod = _load_module()
    result = mod.build_calibrated_service_model(tmp_path / "no_such_curves.json")
    assert result is None


def test_build_calibrated_service_model_loads_real_curves():
    mod = _load_module()
    real_curves = ROOT / "results" / "gpu_calibration" / "service_curves.json"
    if not real_curves.exists():
        pytest.skip("results/gpu_calibration/service_curves.json not present in this checkout")
    model = mod.build_calibrated_service_model(real_curves)
    assert model is not None
    rate = mod.calibrated_decode_rate_tokens_per_sec(model, batch_size=1, context_tokens=128)
    assert rate > 0


# ---------------------------------------------------------------------------
# Comparison tables, including graceful degradation when calibrated model
# or hosted data is absent
# ---------------------------------------------------------------------------

def test_build_comparison_by_target_without_calibrated_model(tmp_path):
    mod = _load_module()
    fit_dir = _make_fitted_model_dir(tmp_path)
    raw = mod.raw_latency_by_target(mod.load_fitted_model_json(fit_dir))
    synthetic_model = mod.build_synthetic_service_model(step_size=0.001)
    rows = mod.build_comparison_by_target(raw, synthetic_model, calibrated_model=None)
    assert len(rows) == 3
    for row in rows:
        assert row["simulator_synthetic_decode_rate_tokens_per_sec"] == 1000.0
        assert "simulator_calibrated_b1_decode_rate_tokens_per_sec" in row
        assert row["simulator_calibrated_b1_decode_rate_tokens_per_sec"] is None
        assert "cohere_mean_latency_s" in row
        assert "gemini_mean_latency_s" in row


def test_build_comparison_by_target_with_empty_hosted_data():
    mod = _load_module()
    synthetic_model = mod.build_synthetic_service_model(step_size=0.001)
    rows = mod.build_comparison_by_target([], synthetic_model, calibrated_model=None)
    # Still produces one row per target_output_tokens, just without hosted columns.
    assert len(rows) == 3
    assert all("cohere_mean_latency_s" not in row for row in rows)


def test_build_comparison_by_provider_without_yaml_or_calibrated_model():
    mod = _load_module()
    synthetic_model = mod.build_synthetic_service_model(step_size=0.001)
    rows = mod.build_comparison_by_provider(None, synthetic_model, calibrated_model=None)
    assert len(rows) == 1
    assert rows[0]["entity"] == "simulator_synthetic_default"


def test_build_findings_identifies_closest_provider(tmp_path):
    mod = _load_module()
    by_provider = [
        {"entity": "cohere", "source": "hosted_api", "decode_rate_tokens_per_sec": 88.5},
        {"entity": "gemini", "source": "hosted_api", "decode_rate_tokens_per_sec": 288.9},
        {"entity": "simulator_synthetic_default", "source": "simulator", "decode_rate_tokens_per_sec": 1000.0},
    ]
    findings = mod.build_findings([], by_provider, calibrated_available=False)
    assert findings["simulator_synthetic_closest_hosted_provider"] == "gemini"
    assert findings["simulator_synthetic_faster_than_all_hosted"] is True
    assert findings["calibrated_model_available"] is False


# ---------------------------------------------------------------------------
# End-to-end script: required output files, no secrets, no provider SDK
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_FILES = (
    "summary.json", "summary.md",
    "comparison_by_target_output_tokens.csv", "comparison_by_provider.csv",
)


def test_script_end_to_end_writes_all_required_outputs(tmp_path, monkeypatch):
    for var in ("COHERE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "AZURE_OPENAI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    fitted_yaml = _make_fitted_yaml(tmp_path)
    fitted_model_dir = _make_fitted_model_dir(tmp_path)
    out_dir = tmp_path / "sanity_out"

    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "compare_simulator_to_real_llm_latency.py"),
            "--fitted-config", str(fitted_yaml),
            "--fitted-model-dir", str(fitted_model_dir),
            "--calibration-file", str(tmp_path / "no_such_calibration.json"),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    for fname in REQUIRED_OUTPUT_FILES:
        assert (out_dir / fname).exists(), f"missing {fname}"

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["findings"]["calibrated_model_available"] is False
    assert len(summary["comparison_by_target_output_tokens"]) == 3
    assert len(summary["comparison_by_provider"]) >= 3  # cohere, gemini, simulator default

    import pandas as pd
    by_target = pd.read_csv(out_dir / "comparison_by_target_output_tokens.csv")
    assert set(by_target["target_output_tokens"]) == {64, 128, 256}
    by_provider = pd.read_csv(out_dir / "comparison_by_provider.csv")
    assert "simulator_synthetic_default" in set(by_provider["entity"])


def test_script_handles_missing_fitted_model_dir_gracefully(tmp_path):
    fitted_yaml = _make_fitted_yaml(tmp_path)
    out_dir = tmp_path / "sanity_out_no_model_dir"

    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "compare_simulator_to_real_llm_latency.py"),
            "--fitted-config", str(fitted_yaml),
            "--fitted-model-dir", str(tmp_path / "does_not_exist"),
            "--calibration-file", str(tmp_path / "no_such_calibration.json"),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    # No hosted per-target data available, but the script must still run and
    # report simulator-only rows rather than crashing.
    assert len(summary["comparison_by_provider"]) >= 1


def test_script_refuses_gracefully_when_no_inputs_at_all(tmp_path):
    out_dir = tmp_path / "sanity_out_empty"
    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "compare_simulator_to_real_llm_latency.py"),
            "--fitted-config", str(tmp_path / "nope.yaml"),
            "--fitted-model-dir", str(tmp_path / "nope_dir"),
            "--calibration-file", str(tmp_path / "nope_curves.json"),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2
    assert not (out_dir / "summary.json").exists()


def test_script_never_imports_provider_sdks(tmp_path):
    fitted_yaml = _make_fitted_yaml(tmp_path)
    fitted_model_dir = _make_fitted_model_dir(tmp_path)
    out_dir = tmp_path / "sanity_out_sdk_check"

    proc = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, {str(ROOT / "scripts")!r})
import compare_simulator_to_real_llm_latency as mod
mod.main([
    "--fitted-config", {str(fitted_yaml)!r},
    "--fitted-model-dir", {str(fitted_model_dir)!r},
    "--calibration-file", {str(tmp_path / "no_such_calibration.json")!r},
    "--output-dir", {str(out_dir)!r},
])
for forbidden in ("cohere", "google.genai", "openai", "azure"):
    assert forbidden not in sys.modules, f"unexpectedly imported {{forbidden}}"
print("OK")
""",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_no_secrets_written_to_outputs(tmp_path, monkeypatch):
    secret = "sk-SHOULD-NEVER-APPEAR-12345"
    monkeypatch.setenv("COHERE_API_KEY", secret)
    fitted_yaml = _make_fitted_yaml(tmp_path)
    fitted_model_dir = _make_fitted_model_dir(tmp_path)
    out_dir = tmp_path / "sanity_out_secret_check"

    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "compare_simulator_to_real_llm_latency.py"),
            "--fitted-config", str(fitted_yaml),
            "--fitted-model-dir", str(fitted_model_dir),
            "--calibration-file", str(tmp_path / "no_such_calibration.json"),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, timeout=60,
        env={**__import__("os").environ, "COHERE_API_KEY": secret},
    )
    assert proc.returncode == 0, proc.stderr
    for f in out_dir.rglob("*"):
        if f.is_file():
            assert secret not in f.read_text(errors="ignore"), f"secret leaked into {f}"
