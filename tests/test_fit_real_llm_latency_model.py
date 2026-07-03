"""Tests for scripts/fit_real_llm_latency_model.py on small synthetic
fixtures. No network access, no API credentials, no live pilot data
required — everything here is fabricated in tmp_path.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_fit_module():
    spec = importlib.util.spec_from_file_location(
        "fit_real_llm_latency_model", ROOT / "scripts" / "fit_real_llm_latency_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_pilot_dir(
    tmp_path: Path, name: str, *, provider: str, model: str,
    n: int, base_ttft: float, decode_rate: float, include_provider_field: bool = True,
    ttft_drift: bool = True,
) -> Path:
    """Build a small synthetic pilot directory with a requests.jsonl whose
    output_tokens actually vary, so a decode-rate fit is meaningful (unlike
    the real v1 pilots, which is exactly the gap this fixture is designed to
    exercise test coverage for)."""
    out_dir = tmp_path / name
    out_dir.mkdir()
    cfg = {"model": model}
    if include_provider_field:
        cfg["provider"] = provider
    (out_dir / "run_config.json").write_text(json.dumps(cfg))

    rows = []
    for i in range(n):
        output_tokens = 20.0 + (i % 5) * 20.0  # varies 20..100
        ttft = base_ttft + (0.001 * i if ttft_drift else 0.0)
        decode_seconds = output_tokens / decode_rate
        latency = ttft + decode_seconds
        rows.append({
            "request_id": f"{name}_{i}",
            "experiment_id": name,
            "model": model,
            "prompt_bucket": ["short", "medium", "long"][i % 3],
            "intended_prompt_tokens": 100 + i,
            "actual_prompt_tokens": 100.0 + i,
            "max_tokens": 256,
            "concurrency_level": [1, 2, 4][i % 3],
            "request_index": i,
            "start_time_iso": "2026-07-03T00:00:00+00:00",
            "end_time_iso": "2026-07-03T00:00:01+00:00",
            "rate_limiter_wait_seconds": 0.0,
            "provider_request_latency_seconds": round(latency, 6),
            "ttft_seconds": round(ttft, 6),
            "total_wall_time_seconds": round(latency, 6),
            "output_text_length_chars": 100,
            "output_tokens": output_tokens,
            "billed_units": {"input_tokens": 100.0, "output_tokens": output_tokens},
            "finish_reason": "COMPLETE",
            "status": "success",
            "error_type": None,
            "error_message": None,
            "retry_count": 0,
            "was_resumed": False,
        })
    with open(out_dir / "requests.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out_dir


def test_load_dataset_reads_provider_and_model_from_run_config(tmp_path):
    mod = _load_fit_module()
    pilot_dir = _make_pilot_dir(
        tmp_path, "fake_pilot", provider="fakeprov", model="fake-model",
        n=10, base_ttft=0.2, decode_rate=50.0,
    )
    records = mod.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    assert len(records) == 10
    assert all(r["provider"] == "fakeprov" for r in records)
    assert all(r["model"] == "fake-model" for r in records)


def test_load_dataset_falls_back_to_dir_name_when_provider_field_missing(tmp_path):
    mod = _load_fit_module()
    pilot_dir = _make_pilot_dir(
        tmp_path, "cohere_pilot_fixture", provider="ignored", model="m",
        n=5, base_ttft=0.2, decode_rate=50.0, include_provider_field=False,
    )
    records = mod.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    assert all(r["provider"] == "cohere_pilot_fixture".split("_pilot")[0] for r in records)


def test_fit_ttft_model_recovers_known_intercept(tmp_path):
    mod = _load_fit_module()
    pilot_dir = _make_pilot_dir(
        tmp_path, "provA_pilot_x", provider="provA", model="m",
        n=60, base_ttft=0.3, decode_rate=50.0, ttft_drift=False,
    )
    records = mod.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    fit = mod.fit_ttft_model(records)
    assert fit["fit"] is True
    # Constant ttft across all rows -> intercept should recover it exactly
    # and every feature coefficient should be ~0.
    assert fit["intercept"] == pytest.approx(0.3, abs=1e-6)
    assert fit["coef_prompt_tokens"] == pytest.approx(0.0, abs=1e-6)
    assert fit["coef_output_tokens"] == pytest.approx(0.0, abs=1e-6)
    assert fit["coef_concurrency_level"] == pytest.approx(0.0, abs=1e-6)


def test_fit_latency_model_ttft_plus_decode_recovers_known_decode_rate(tmp_path):
    mod = _load_fit_module()
    pilot_dir = _make_pilot_dir(
        tmp_path, "provB_pilot_x", provider="provB", model="m",
        n=60, base_ttft=0.2, decode_rate=80.0,
    )
    records = mod.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    fit = mod.fit_latency_model_ttft_plus_decode(records)
    assert fit["fit"] is True
    assert fit["effective_decode_rate_tokens_per_sec"] == pytest.approx(80.0, rel=0.05)


def test_robust_stats_reports_percentiles_not_just_mean():
    mod = _load_fit_module()
    stats = mod.robust_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(22.0)
    assert stats["p50"] is not None
    assert stats["p95"] is not None
    assert stats["p99"] is not None
    # Median should be far below the mean given the outlier — proves we
    # aren't just reporting mean under a different key.
    assert stats["p50"] < stats["mean"]


def test_exclude_rpm_wait_outliers_flag_drops_flagged_requests(tmp_path):
    mod = _load_fit_module()
    pilot_dir = _make_pilot_dir(
        tmp_path, "provC_pilot_x", provider="provC", model="m",
        n=10, base_ttft=0.2, decode_rate=50.0,
    )
    # Inject one legacy-shaped polluted row (no rate_limiter_wait_seconds
    # field, huge latency vs. ttft) alongside the clean synthetic rows.
    requests_path = pilot_dir / "requests.jsonl"
    extra = {
        "request_id": "polluted_extra", "experiment_id": "provC_pilot_x", "model": "m",
        "prompt_bucket": "short", "intended_prompt_tokens": 100, "actual_prompt_tokens": 100.0,
        "max_tokens": 256, "concurrency_level": 1, "request_index": 99,
        "start_time_iso": "x", "end_time_iso": "y",
        "elapsed_seconds": 53.0, "ttft_seconds": 0.2, "total_latency_seconds": 53.0,
        "output_text_length_chars": 10, "output_tokens": 20.0,
        "billed_units": None, "finish_reason": "COMPLETE", "status": "success",
        "error_type": None, "error_message": None, "retry_count": 0, "was_resumed": False,
    }
    with open(requests_path, "a") as f:
        f.write(json.dumps(extra) + "\n")

    records_with = mod.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=True)
    records_without = mod.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    assert len(records_without) == 11
    assert len(records_with) == 10
    assert "polluted_extra" not in {r["request_id"] for r in records_with}


def test_fit_script_end_to_end_writes_outputs_no_credentials(tmp_path, monkeypatch):
    for var in ("COHERE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "AZURE_OPENAI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    dir_a = _make_pilot_dir(tmp_path, "provA_pilot_x", provider="provA", model="mA", n=30, base_ttft=0.2, decode_rate=50.0)
    dir_b = _make_pilot_dir(tmp_path, "provB_pilot_x", provider="provB", model="mB", n=30, base_ttft=0.5, decode_rate=30.0)
    out_dir = tmp_path / "fit_out"

    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "fit_real_llm_latency_model.py"),
            "--experiment-dir", str(dir_a),
            "--experiment-dir", str(dir_b),
            "--output-dir", str(out_dir),
            "--no-plots",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "latency_model_fit.json").exists()
    assert (out_dir / "latency_model_fit.md").exists()
    assert (out_dir / "latency_model_fit.csv").exists()
    assert not (out_dir / "latency_model_fit.png").exists()

    payload = json.loads((out_dir / "latency_model_fit.json").read_text())
    assert set(payload["providers"].keys()) == {"provA", "provB"}
    assert "pooled" in payload

    combined = (out_dir / "latency_model_fit.json").read_text() + (out_dir / "latency_model_fit.md").read_text()
    for forbidden in ("API_KEY", "Bearer ", "sk-"):
        assert forbidden not in combined


def test_fit_script_with_plots(tmp_path):
    dir_a = _make_pilot_dir(tmp_path, "provA_pilot_x", provider="provA", model="mA", n=20, base_ttft=0.2, decode_rate=50.0)
    out_dir = tmp_path / "fit_out_plots"

    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "fit_real_llm_latency_model.py"),
            "--experiment-dir", str(dir_a),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "latency_model_fit.png").exists()
