from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gpu_external_validity_audit.py"
spec = importlib.util.spec_from_file_location("gpu_external_validity_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def test_metric_parser_reads_vllm_runtime_gauges():
    text = """
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 2.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.125
"""
    parsed = audit._parse_vllm_metrics(text)
    assert parsed["running"] == 3.0
    assert parsed["waiting"] == 2.0
    assert parsed["kv_cache_usage"] == 0.125
    assert parsed["preemptions_total"] == 0.0


def test_controlled_scenario_construction_is_deterministic():
    a = audit.build_scenarios(Path("/no/such/root"), include_real_traces=False)
    b = audit.build_scenarios(Path("/no/such/root"), include_real_traces=False)
    assert [s.name for s in a] == [s.name for s in b]
    assert len(a) == 12
    assert a[0].requests[0] == b[0].requests[0]
    assert {s.scenario_family for s in a}.issuperset({"prefill_heavy", "decode_heavy", "bursty", "kv_pressure"})


def test_stress_scenario_construction_targets_pressure_regimes():
    scenarios = audit.build_stress_scenarios(Path("/no/such/root"), include_real_traces=False)
    assert [s.name for s in scenarios] == [
        "stress_high_concurrency_queue",
        "stress_long_decode_kv",
        "stress_long_prefill",
        "stress_kv_pressure",
        "stress_mixed_prefill_decode_contention",
        "stress_burst_overload_recovery",
    ]
    assert max(s.max_client_concurrency for s in scenarios) >= 24
    assert max(r.target_output_tokens for s in scenarios for r in s.requests) >= 768
    assert any(r.arrival_time_s > 0 for s in scenarios for r in s.requests)


def test_calibration_profile_records_incomplete_pressure_warning():
    reports = [{
        "runtime_summary": {
            "num_requests": 1,
            "num_success": 1,
            "mean_prompt_tokens": 100.0,
            "mean_ttft_s": 0.02,
            "mean_tpot_s": 0.005,
            "max_vllm_waiting": 0.0,
            "max_kv_cache_usage": 0.01,
        },
        "simulator_summary": {
            "vllm_faithful": {"mean_latency_s": 0.01},
            "sarathi_faithful": {"mean_latency_s": 0.011},
        },
    }]
    summary = audit.summarize_audit(reports)
    profile = audit.make_calibration_profile({"model": "m", "server_command": "cmd"}, summary, reports)
    assert profile["historical_defaults_changed"] is False
    assert profile["prefill_latency_fit"]["intercept_s"] == 0.02
    assert profile["prefill_latency_fit"]["slope_s_per_prompt_token"] == 0.0
    assert any("KV usage" in warning for warning in profile["warnings"])


def test_mock_runtime_scenario_completes_without_network():
    scenario = audit.build_scenarios(Path("/no/such/root"), include_real_traces=False)[0]
    rows, summary = audit.run_runtime_scenario(
        scenario,
        server_url="http://127.0.0.1:9",
        model="mock",
        timeout_s=1.0,
        mock=True,
    )
    assert len(rows) == len(scenario.requests)
    assert summary["num_success"] == len(scenario.requests)
    assert summary["completion_fraction"] == 1.0
    assert summary["mean_latency_s"] is not None
