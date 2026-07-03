"""Tests for scripts/fit_real_llm_latency_model_v2.py on small synthetic
fixtures. No network access, no API credentials, no live pilot data
required — everything here is fabricated in tmp_path. Mirrors the fixture
style of tests/test_fit_real_llm_latency_model.py but adds the v2
length-targeted fields (target_output_tokens, workload_version,
reached_target_output_range) the v1 fixtures don't have.
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
sys.path.insert(0, str(ROOT / "scripts"))


def _load_v2_module():
    spec = importlib.util.spec_from_file_location(
        "fit_real_llm_latency_model_v2", ROOT / "scripts" / "fit_real_llm_latency_model_v2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_v2_pilot_dir(
    tmp_path: Path, name: str, *, provider: str, model: str,
    targets, requests_per_target: int, base_ttft: float, decode_rate: float,
    min_output_token_ratio: float = 0.70,
) -> Path:
    """Build a small synthetic v2 pilot directory: requests.jsonl (with
    target_output_tokens/workload_version/reached_target_output_range set),
    run_config.json, and summary.json (model_inputs_manifest reads both)."""
    out_dir = tmp_path / name
    out_dir.mkdir()
    (out_dir / "run_config.json").write_text(json.dumps({
        "provider": provider, "model": model, "workload_version": "v2",
        "min_output_token_ratio": min_output_token_ratio,
    }))

    rows = []
    by_target = {}
    i = 0
    for target in targets:
        for j in range(requests_per_target):
            output_tokens = target * (0.9 + 0.02 * j)  # varies slightly around target
            # Periodic (not linear-in-i) jitter so ttft isn't perfectly
            # collinear with prompt_tokens=100+i, which IS linear in i.
            ttft = base_ttft + 0.001 * (i % 7)
            decode_seconds = output_tokens / decode_rate
            latency = ttft + decode_seconds
            reached = output_tokens >= min_output_token_ratio * target
            rows.append({
                "request_id": f"{name}_{i}",
                "experiment_id": name,
                "model": model,
                "prompt_bucket": ["short", "medium", "long"][i % 3],
                "intended_prompt_tokens": 100 + i,
                "actual_prompt_tokens": 100.0 + i,
                "max_tokens": target * 2,
                "concurrency_level": [1, 2, 4][i % 3],
                "request_index": j,
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
                "target_output_tokens": target,
                "workload_version": "v2",
                "output_text_preview": "ok",
                "reached_target_output_range": reached,
            })
            by_target.setdefault(target, []).append((output_tokens, reached))
            i += 1

    with open(out_dir / "requests.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_target_summary = []
    for target, vals in sorted(by_target.items()):
        outs = [v[0] for v in vals]
        reached_vals = [v[1] for v in vals]
        by_target_summary.append({
            "target_output_tokens": target,
            "n_success": len(vals),
            "mean_output_tokens": sum(outs) / len(outs),
            "mean_output_token_ratio": (sum(outs) / len(outs)) / target,
            "frac_reached_target_range": sum(reached_vals) / len(reached_vals),
        })
    (out_dir / "summary.json").write_text(json.dumps({
        "total_records": len(rows),
        "status_counts": {"success": len(rows)},
        "frac_reached_target_output_range": sum(r["reached_target_output_range"] for r in rows) / len(rows),
        "by_target_output_tokens": by_target_summary,
    }))
    return out_dir


# ---------------------------------------------------------------------------
# v2 fields parsed via the (extended) v1 loader
# ---------------------------------------------------------------------------

def test_load_dataset_captures_v2_fields(tmp_path):
    mod = _load_v2_module()
    pilot_dir = _make_v2_pilot_dir(
        tmp_path, "provA_v2", provider="provA", model="mA",
        targets=[64, 128], requests_per_target=10, base_ttft=0.2, decode_rate=50.0,
    )
    records = mod.v1.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    assert len(records) == 20
    assert {r["target_output_tokens"] for r in records} == {64.0, 128.0}
    assert all(r["workload_version"] == "v2" for r in records)
    assert all(r["reached_target_output_range"] is not None for r in records)


def test_ttft_length_feature_prefers_target_over_output_tokens():
    mod = _load_v2_module()
    row = {"target_output_tokens": 64.0, "output_tokens": 999.0}
    assert mod._ttft_length_feature(row) == 64.0


def test_ttft_length_feature_falls_back_to_output_tokens_when_target_missing():
    mod = _load_v2_module()
    row = {"target_output_tokens": None, "output_tokens": 42.0}
    assert mod._ttft_length_feature(row) == 42.0


# ---------------------------------------------------------------------------
# v2 TTFT / latency model fitting
# ---------------------------------------------------------------------------

def test_fit_ttft_model_v2_uses_target_output_tokens(tmp_path):
    mod = _load_v2_module()
    pilot_dir = _make_v2_pilot_dir(
        tmp_path, "provB_v2", provider="provB", model="mB",
        targets=[64, 128, 256], requests_per_target=15, base_ttft=0.3, decode_rate=60.0,
    )
    records = mod.v1.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    bundle = mod.fit_ttft_model_v2(records)
    assert bundle["fit"]["fit"] is True
    assert bundle["feature_names"][0] == "target_or_output_tokens"
    # All rows have a target_output_tokens, so the feature used must be it, not output_tokens.
    assert bundle["features"][0][0] == records[0]["target_output_tokens"]


def test_fit_latency_model_v2_recovers_ttft_coefficient(tmp_path):
    mod = _load_v2_module()
    pilot_dir = _make_v2_pilot_dir(
        tmp_path, "provC_v2", provider="provC", model="mC",
        targets=[64, 128, 256], requests_per_target=20, base_ttft=0.25, decode_rate=80.0,
    )
    records = mod.v1.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    bundle = mod.fit_latency_model_v2(records)
    fit = bundle["fit"]
    assert fit["fit"] is True
    # latency = ttft + output_tokens/decode_rate exactly by construction, so
    # coef_ttft_seconds should recover ~1.0 and coef_output_tokens ~1/80.
    assert fit["coef_ttft_seconds"] == pytest.approx(1.0, abs=0.05)
    assert fit["coef_output_tokens"] == pytest.approx(1.0 / 80.0, rel=0.1)


def test_fit_ttft_model_v2_falls_back_gracefully_when_no_target_or_output_tokens():
    mod = _load_v2_module()
    rows = [
        {"ttft_seconds": 0.2, "target_output_tokens": None, "output_tokens": None,
         "prompt_tokens": 100.0, "concurrency_level": 1.0, "provider": "p", "request_id": "r1"}
        for _ in range(20)
    ]
    bundle = mod.fit_ttft_model_v2(rows)
    assert bundle["fit"]["fit"] is False
    assert "reason" in bundle["fit"]


def test_fit_latency_model_v2_missing_fields_fails_gracefully():
    mod = _load_v2_module()
    rows = [
        {"latency_seconds": None, "ttft_seconds": 0.2, "output_tokens": None,
         "prompt_tokens": 100.0, "concurrency_level": 1.0, "provider": "p", "request_id": "r1"}
        for _ in range(20)
    ]
    bundle = mod.fit_latency_model_v2(rows)
    assert bundle["fit"]["fit"] is False


# ---------------------------------------------------------------------------
# Rate-limiter wait exclusion (records use provider_request_latency_seconds,
# never total_wall_time_seconds, for the fitted "latency_seconds" field)
# ---------------------------------------------------------------------------

def test_rate_limiter_wait_excluded_from_latency_used_in_fit(tmp_path):
    mod = _load_v2_module()
    out_dir = tmp_path / "provD_v2"
    out_dir.mkdir()
    (out_dir / "run_config.json").write_text(json.dumps({
        "provider": "provD", "model": "mD", "workload_version": "v2",
    }))
    rows = []
    for i in range(15):
        # Clean provider latency is always ~1s; a huge rate_limiter_wait_seconds
        # is injected on some rows and must NOT leak into the fitted feature.
        rows.append({
            "request_id": f"provD_{i}", "experiment_id": "provD_v2", "model": "mD",
            "prompt_bucket": "short", "intended_prompt_tokens": 100, "actual_prompt_tokens": 100.0,
            "max_tokens": 128, "concurrency_level": 1, "request_index": i,
            "start_time_iso": "x", "end_time_iso": "y",
            "rate_limiter_wait_seconds": 50.0 if i % 3 == 0 else 0.0,
            "provider_request_latency_seconds": 1.0,
            "ttft_seconds": 0.2,
            "total_wall_time_seconds": 51.0 if i % 3 == 0 else 1.0,
            "output_text_length_chars": 10, "output_tokens": 64.0,
            "billed_units": None, "finish_reason": "COMPLETE", "status": "success",
            "error_type": None, "error_message": None, "retry_count": 0, "was_resumed": False,
            "target_output_tokens": 64, "workload_version": "v2",
            "output_text_preview": "ok", "reached_target_output_range": True,
        })
    with open(out_dir / "requests.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    records = mod.v1.load_dataset([str(out_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    assert all(r["latency_seconds"] == pytest.approx(1.0) for r in records), (
        "latency_seconds must come from provider_request_latency_seconds, "
        "not total_wall_time_seconds (which includes rate_limiter_wait_seconds)"
    )
    # rate_limiter_wait_seconds is captured separately, not folded into latency.
    assert any(r["rate_limiter_wait_seconds"] == 50.0 for r in records)


# ---------------------------------------------------------------------------
# Decode-rate table by (provider, target_output_tokens)
# ---------------------------------------------------------------------------

def test_decode_rate_table_per_provider_and_target(tmp_path):
    mod = _load_v2_module()
    pilot_dir = _make_v2_pilot_dir(
        tmp_path, "provE_v2", provider="provE", model="mE",
        targets=[64, 128], requests_per_target=30, base_ttft=0.2, decode_rate=100.0,
    )
    records = mod.v1.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    rows = mod.build_decode_rate_table(records)
    by_key = {(r["provider"], r["target_output_tokens"]): r for r in rows}
    assert ("provE", "overall") in by_key
    assert ("provE", 64) in by_key
    assert ("provE", 128) in by_key
    assert by_key[("provE", "overall")]["effective_decode_rate_tokens_per_sec"] == pytest.approx(100.0, rel=0.1)


# ---------------------------------------------------------------------------
# Raw crossover table (model-free)
# ---------------------------------------------------------------------------

def test_raw_latency_by_target_and_crossover_description(tmp_path):
    mod = _load_v2_module()
    dir_fast_short = _make_v2_pilot_dir(
        tmp_path, "fastShort_v2", provider="fastShort", model="m1",
        targets=[64], requests_per_target=20, base_ttft=0.1, decode_rate=50.0,
    )
    dir_fast_long = _make_v2_pilot_dir(
        tmp_path, "fastLong_v2", provider="fastLong", model="m2",
        targets=[64], requests_per_target=20, base_ttft=0.9, decode_rate=500.0,
    )
    records = mod.v1.load_dataset(
        [str(dir_fast_short), str(dir_fast_long)], None, root=tmp_path, exclude_rpm_wait_outliers=False,
    )
    raw = mod.build_raw_latency_by_target(records)
    assert {row["provider"] for row in raw} == {"fastShort", "fastLong"}
    crossover_lines = mod.describe_crossover(raw)
    assert len(crossover_lines) == 1
    assert "fastShort" in crossover_lines[0]  # lower base_ttft + slower-but-irrelevant decode wins at target=64


# ---------------------------------------------------------------------------
# Residual summary
# ---------------------------------------------------------------------------

def test_summarize_residuals_near_zero_for_exact_linear_data(tmp_path):
    mod = _load_v2_module()
    pilot_dir = _make_v2_pilot_dir(
        tmp_path, "provF_v2", provider="provF", model="mF",
        targets=[64, 128, 256], requests_per_target=25, base_ttft=0.2, decode_rate=70.0,
    )
    records = mod.v1.load_dataset([str(pilot_dir)], None, root=tmp_path, exclude_rpm_wait_outliers=False)
    latency_bundle = mod.fit_latency_model_v2(records)
    residuals = mod.residuals_from_bundle(latency_bundle, "latency")
    summary = mod.summarize_residuals(residuals)
    assert len(summary) == 1
    assert summary[0]["rmse"] < 0.01  # data is exactly linear by construction


# ---------------------------------------------------------------------------
# End-to-end script: all required output files, no secrets, no network
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_FILES = (
    "latency_model_fit_v2.json",
    "latency_model_fit_v2.md",
    "latency_model_fit_v2.csv",
    "provider_decode_rates.csv",
    "residuals_by_provider.csv",
    "model_inputs_manifest.json",
)


def test_fit_v2_script_end_to_end_writes_all_required_outputs(tmp_path, monkeypatch):
    for var in ("COHERE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "AZURE_OPENAI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    dir_a = _make_v2_pilot_dir(
        tmp_path, "cohere_v2_fixture", provider="Cohere", model="command-r7b-12-2024",
        targets=[64, 128, 256], requests_per_target=12, base_ttft=0.25, decode_rate=88.0,
    )
    dir_b = _make_v2_pilot_dir(
        tmp_path, "gemini_v2_fixture", provider="Gemini", model="gemini-3.1-flash-lite",
        targets=[64, 128, 256], requests_per_target=12, base_ttft=0.65, decode_rate=200.0,
    )
    out_dir = tmp_path / "fit_v2_out"

    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "fit_real_llm_latency_model_v2.py"),
            "--experiment-dir", str(dir_a),
            "--experiment-dir", str(dir_b),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    for fname in REQUIRED_OUTPUT_FILES:
        assert (out_dir / fname).exists(), f"missing {fname}"

    payload = json.loads((out_dir / "latency_model_fit_v2.json").read_text())
    assert set(payload["providers"].keys()) == {"Cohere", "Gemini"}
    assert "pooled" in payload
    assert "decode_rate_table" in payload
    assert "residual_summary" in payload
    assert "raw_latency_by_target" in payload

    manifest = json.loads((out_dir / "model_inputs_manifest.json").read_text())
    assert len(manifest["experiment_dirs"]) == 2
    assert all(d["workload_version"] == "v2" for d in manifest["experiment_dirs"])
    assert all(d["error_rate"] == 0.0 for d in manifest["experiment_dirs"])

    import pandas as pd
    decode_rates = pd.read_csv(out_dir / "provider_decode_rates.csv")
    assert set(decode_rates["provider"]) == {"Cohere", "Gemini"}
    assert set(decode_rates["target_output_tokens"].astype(str)) >= {"overall", "64", "128", "256"}

    residuals = pd.read_csv(out_dir / "residuals_by_provider.csv")
    assert set(residuals["model_type"]) == {"ttft", "latency"}

    combined = "".join((out_dir / f).read_text() for f in REQUIRED_OUTPUT_FILES if (out_dir / f).suffix != ".csv")
    combined += (out_dir / "provider_decode_rates.csv").read_text()
    combined += (out_dir / "residuals_by_provider.csv").read_text()
    for forbidden in ("API_KEY", "Bearer ", "sk-", "AIza"):
        assert forbidden not in combined


def test_fit_v2_script_refuses_gracefully_with_no_successful_records(tmp_path):
    empty_dir = tmp_path / "empty_v2"
    empty_dir.mkdir()
    (empty_dir / "run_config.json").write_text(json.dumps({"provider": "p", "model": "m", "workload_version": "v2"}))
    (empty_dir / "requests.jsonl").write_text("")
    out_dir = tmp_path / "fit_v2_empty_out"

    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "fit_real_llm_latency_model_v2.py"),
            "--experiment-dir", str(empty_dir),
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2
    assert not (out_dir / "latency_model_fit_v2.json").exists()
