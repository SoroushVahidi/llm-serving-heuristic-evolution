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
    assert parsed == {"running": 3.0, "waiting": 2.0, "kv_cache_usage": 0.125}


def test_controlled_scenario_construction_is_deterministic():
    a = audit.build_scenarios(Path("/no/such/root"), include_real_traces=False)
    b = audit.build_scenarios(Path("/no/such/root"), include_real_traces=False)
    assert [s.name for s in a] == [s.name for s in b]
    assert len(a) == 12
    assert a[0].requests[0] == b[0].requests[0]
    assert {s.scenario_family for s in a}.issuperset({"prefill_heavy", "decode_heavy", "bursty", "kv_pressure"})


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
