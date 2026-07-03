"""Tests for legacy real-LLM log reprocessing: heuristic RPM-wait-outlier
flagging and corrected-summary regeneration from an existing requests.jsonl,
without any network access or API credentials.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402


def _legacy_row(request_id, *, latency, ttft, status="success", concurrency=1):
    """A row shaped like the pre-fix schema: only elapsed_seconds/
    total_latency_seconds/ttft_seconds, no rate_limiter_wait_seconds or
    provider_request_latency_seconds."""
    return {
        "request_id": request_id,
        "experiment_id": "legacy_pilot",
        "model": "some-model",
        "prompt_bucket": "short",
        "intended_prompt_tokens": 100,
        "actual_prompt_tokens": 105.0,
        "max_tokens": 64,
        "concurrency_level": concurrency,
        "request_index": 0,
        "start_time_iso": "2026-07-03T00:00:00+00:00",
        "end_time_iso": "2026-07-03T00:00:01+00:00",
        "elapsed_seconds": latency,
        "ttft_seconds": ttft,
        "total_latency_seconds": latency,
        "output_text_length_chars": 50,
        "output_tokens": 20.0,
        "billed_units": {"input_tokens": 105.0, "output_tokens": 20.0},
        "finish_reason": "COMPLETE",
        "status": status,
        "error_type": None,
        "error_message": None,
        "retry_count": 0,
        "was_resumed": False,
    }


def _write_legacy_requests_jsonl(path: Path, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_flag_likely_rate_limiter_wait_outliers_uses_ttft_gap():
    rows = [
        _legacy_row("clean_1", latency=0.5, ttft=0.2),
        _legacy_row("clean_2", latency=0.6, ttft=0.25),
        _legacy_row("polluted_1", latency=53.0, ttft=0.22),  # matches observed pilot artifact
    ]
    flagged = cc.flag_likely_rate_limiter_wait_outliers(rows)
    assert flagged == ["polluted_1"]


def test_flag_likely_rate_limiter_wait_outliers_ignores_non_success():
    rows = [
        _legacy_row("failed_but_slow", latency=99.0, ttft=None, status="error"),
    ]
    assert cc.flag_likely_rate_limiter_wait_outliers(rows) == []


def test_flag_likely_rate_limiter_wait_outliers_non_streaming_fallback():
    # No ttft (non-streaming); falls back to an absolute latency threshold.
    rows = [
        _legacy_row("clean_nostream", latency=1.0, ttft=None),
        _legacy_row("polluted_nostream", latency=45.0, ttft=None),
    ]
    flagged = cc.flag_likely_rate_limiter_wait_outliers(rows)
    assert flagged == ["polluted_nostream"]


def test_reprocess_legacy_summary_separates_raw_and_corrected(tmp_path):
    rows = [_legacy_row(f"clean_{i}", latency=0.5 + i * 0.01, ttft=0.2 + i * 0.01) for i in range(20)]
    rows.append(_legacy_row("polluted_1", latency=53.0, ttft=0.22))
    requests_path = tmp_path / "requests.jsonl"
    _write_legacy_requests_jsonl(requests_path, rows)

    result = cc.reprocess_legacy_summary(requests_path)
    assert result["n_success"] == 21
    assert result["n_flagged_likely_rate_limiter_wait"] == 1
    assert result["flagged_request_ids"] == ["polluted_1"]
    assert result["has_rate_limiter_wait_field"] is False

    # Raw p99 is dragged toward the 53s outlier; corrected excludes it.
    assert result["raw_stats"]["p99_latency_s"] > 10.0
    assert result["corrected_stats_excluding_flagged"]["p99_latency_s"] < 1.0
    # p50 should be nearly identical raw vs. corrected — the artifact only
    # affects the tail, matching what was observed in the real pilots.
    raw_p50 = result["raw_stats"]["p50_latency_s"]
    corrected_p50 = result["corrected_stats_excluding_flagged"]["p50_latency_s"]
    assert raw_p50 == pytest.approx(corrected_p50, abs=0.05)

    # TTFT stats must be identical raw vs. corrected — never polluted.
    assert (
        result["raw_stats"]["mean_ttft_s"]
        == result["corrected_stats_excluding_flagged"]["mean_ttft_s"]
    )


def test_reprocess_legacy_summary_detects_new_schema_and_skips_heuristic(tmp_path):
    plan = cc.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=2, seed=1,
    )
    writer = cc.JsonlWriter(tmp_path / "requests.jsonl")
    for p in plan:
        result = cc.execute_one_request(
            p, client=None, stream=False, timeout_s=5, mock=True,
            rpm_limiter=cc.RpmLimiter(1000), was_resumed=False,
            call_streaming_fn=None, call_non_streaming_fn=None,
        )
        writer.write(result)
    writer.close()

    result = cc.reprocess_legacy_summary(tmp_path / "requests.jsonl")
    assert result["has_rate_limiter_wait_field"] is True
    assert "already has rate_limiter_wait_seconds" in result["caveat"]


def test_write_legacy_reprocessed_summary_writes_files_no_secrets(tmp_path):
    rows = [_legacy_row("clean_1", latency=0.5, ttft=0.2)]
    requests_path = tmp_path / "requests.jsonl"
    _write_legacy_requests_jsonl(requests_path, rows)

    result = cc.reprocess_legacy_summary(requests_path)
    cc.write_legacy_reprocessed_summary(tmp_path, result)

    json_path = tmp_path / "summary_corrected.json"
    md_path = tmp_path / "summary_corrected.md"
    assert json_path.exists()
    assert md_path.exists()

    combined = json_path.read_text() + md_path.read_text()
    for forbidden in ("API_KEY", "Bearer ", "sk-", "COHERE_API_KEY=", "GOOGLE_API_KEY="):
        assert forbidden not in combined


def test_reprocess_script_runs_without_api_credentials(tmp_path, monkeypatch):
    for var in ("COHERE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "AZURE_OPENAI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    rows = [_legacy_row(f"clean_{i}", latency=0.5, ttft=0.2) for i in range(5)]
    requests_path = tmp_path / "requests.jsonl"
    _write_legacy_requests_jsonl(requests_path, rows)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reprocess_real_llm_pilot_logs.py"),
         "--input-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "summary_corrected.json").exists()
    assert (tmp_path / "summary_corrected.md").exists()
