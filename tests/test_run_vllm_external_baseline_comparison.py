"""Tests for scripts/run_vllm_external_baseline_comparison.py.

No hosted API calls, no real vLLM server, no GPU needed: --mock exercises
the full external-admission-controller loop against a local deterministic
stub, and a small stdlib http.server fake (mirroring
tests/test_run_vllm_serving_baseline_pilot.py's pattern) validates the real
HTTP/SSE dispatch path end-to-end.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_vllm_external_baseline_comparison",
        ROOT / "scripts" / "run_vllm_external_baseline_comparison.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fake vLLM-OpenAI-compatible server (reused pattern)
# ---------------------------------------------------------------------------

class _FakeVllmHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/health" or self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "fake-model"}]}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"choices": [{"text": "hi ", "finish_reason": None}]},
            {"choices": [{"text": "there.", "finish_reason": "stop"}]},
            {"choices": [{"text": "", "finish_reason": None}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture()
def fake_vllm_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeVllmHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Policy name handling
# ---------------------------------------------------------------------------

def test_policy_aliases_normalize_correctly():
    mod = _load_module()
    assert mod.normalize_policy_name("vllm_default") == "vllm_direct"
    assert mod.normalize_policy_name("llf") == "least_laxity_first"
    assert mod.normalize_policy_name("estf") == "estimated_service_time_first"
    assert mod.normalize_policy_name("fifo") == "fifo"  # passthrough


def test_unsupported_policy_fails_clearly(tmp_path):
    mod = _load_module()
    result = mod.main(["--mock", "--policies", "fifo,not_a_real_policy", "--output-dir", str(tmp_path)])
    assert result == 2
    assert not (tmp_path / "requests.jsonl").exists()


def test_generated_heuristic_and_selector_rejected_with_explanation(tmp_path):
    mod = _load_module()
    result = mod.main(["--mock", "--policies", "fifo,generated_heuristic", "--output-dir", str(tmp_path)])
    assert result == 8
    result2 = mod.main(["--mock", "--policies", "selector", "--output-dir", str(tmp_path)])
    assert result2 == 8
    result3 = mod.main(["--mock", "--policies", "best_generated", "--output-dir", str(tmp_path)])
    assert result3 == 8


# ---------------------------------------------------------------------------
# Fixed request plan reuse across policies
# ---------------------------------------------------------------------------

def test_request_plan_identical_across_policies(tmp_path):
    mod = _load_module()
    plan_a = mod.build_request_plan(["short", "medium"], [64, 128], [1, 2], 2, seed=20260703)
    plan_b = mod.build_request_plan(["short", "medium"], [64, 128], [1, 2], 2, seed=20260703)
    assert [r.prompt_text for r in plan_a] == [r.prompt_text for r in plan_b]
    assert [r.priority for r in plan_a] == [r.priority for r in plan_b]
    assert [r.slo_slack_seconds for r in plan_a] == [r.slo_slack_seconds for r in plan_b]


def test_end_to_end_uses_same_plan_for_every_policy(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock", "--policies", "fifo,edf,vllm_direct",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    rows = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    for policy in ("fifo", "edf", "vllm_direct"):
        policy_rows = [r for r in rows if r["policy"] == policy]
        ids = sorted(r["request_id"] for r in policy_rows)
        assert ids == [0, 1]  # same 2 request_ids for every policy


# ---------------------------------------------------------------------------
# External admission-controller behavior (regression test for the
# gpu_state-snapshot bug: concurrency=1 with 2+ pending requests must admit
# both, not leak a phantom slot from policy-internal bookkeeping mutation)
# ---------------------------------------------------------------------------

def test_concurrency_one_admits_all_pending_requests_eventually(tmp_path):
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 3, seed=1)  # 3 requests, 1 slot
    results = mod.run_cell_for_policy("fifo", plan, concurrency=1, model="m", base_url=None, mock=True, timeout_s=30)
    assert len(results) == 3
    assert all(r.status == "success" for r in results)


def test_concurrency_one_admits_all_requests_for_every_wired_policy(tmp_path):
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 4, seed=1)
    for policy in mod.WIRED_POLICIES:
        results = mod.run_cell_for_policy(policy, plan, concurrency=1, model="m", base_url=None, mock=True, timeout_s=30)
        assert len(results) == 4, f"{policy} dropped requests at concurrency=1"
        assert all(r.status == "success" for r in results), f"{policy} had failures"


def test_no_policy_sees_actual_output_tokens():
    for fname in ("fifo.py", "edf.py", "shortest_output_first.py", "least_laxity_first.py", "estimated_service_time_first.py"):
        src = (ROOT / "src" / "llmserveopt" / "policies" / fname).read_text()
        # Look for attribute access, not the plain-English docstring
        # sentence "actual_output_tokens is never accessed" that some of
        # these files carry as documentation.
        assert ".actual_output_tokens" not in src, f"{fname} must never access .actual_output_tokens"


# ---------------------------------------------------------------------------
# Arrival-normalized weighted goodput: denominator over ALL arrivals
# ---------------------------------------------------------------------------

def test_arrival_normalized_wg_counts_failures_as_zero():
    mod = _load_module()
    rows = [
        {"policy": "p", "status": "success", "priority": 1.0, "slo_violated": False,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
        {"policy": "p", "status": "success", "priority": 1.0, "slo_violated": False,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
        {"policy": "p", "status": "error", "priority": 1.0, "slo_violated": None,
         "server_request_latency_seconds": None, "ttft_seconds": None,
         "total_wall_time_seconds": 1.0, "output_tokens": None},
        {"policy": "p", "status": "timeout", "priority": 1.0, "slo_violated": None,
         "server_request_latency_seconds": None, "ttft_seconds": None,
         "total_wall_time_seconds": 30.0, "output_tokens": None},
    ]
    m = mod.compute_policy_metrics(rows, policy_wall_clock_s=10.0)
    assert m["n_total"] == 4
    assert m["n_completed"] == 2
    assert m["n_failed"] == 2
    # conditional_WG among completed only = 1.0 (both met SLO)
    assert m["conditional_weighted_goodput"] == pytest.approx(1.0)
    # completion_fraction = 2/4 = 0.5, so arrival-normalized = 0.5 * 1.0 = 0.5
    assert m["completion_fraction"] == pytest.approx(0.5)
    assert m["arrival_normalized_weighted_goodput"] == pytest.approx(0.5)


def test_arrival_normalized_wg_all_completed_equals_conditional():
    mod = _load_module()
    rows = [
        {"policy": "p", "status": "success", "priority": 2.0, "slo_violated": False,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
        {"policy": "p", "status": "success", "priority": 1.0, "slo_violated": True,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
    ]
    m = mod.compute_policy_metrics(rows, policy_wall_clock_s=10.0)
    assert m["completion_fraction"] == pytest.approx(1.0)
    assert m["arrival_normalized_weighted_goodput"] == pytest.approx(m["conditional_weighted_goodput"])
    # priority-weighted: (2*1 + 1*0) / 3 = 0.6667
    assert m["conditional_weighted_goodput"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Timeout / error handling
# ---------------------------------------------------------------------------

def test_run_cell_for_policy_records_dispatch_errors_not_crash():
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 2, seed=1)
    # base_url=None with mock=False forces query_vllm_completion to fail
    # (no server reachable), exercising the error path without crashing.
    results = mod.run_cell_for_policy(
        "fifo", plan, concurrency=1, model="m", base_url="http://127.0.0.1:1", mock=False, timeout_s=2,
    )
    assert len(results) == 2
    assert all(r.status in ("error", "timeout") for r in results)
    assert all(r.error_type is not None for r in results)


# ---------------------------------------------------------------------------
# Warm-up excluded from metrics
# ---------------------------------------------------------------------------

def test_warmup_writes_separate_files_and_excluded_from_requests_jsonl(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock", "--warmup",
        "--policies", "fifo",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    assert (tmp_path / "warmup_requests.jsonl").exists()
    assert (tmp_path / "warmup_summary.md").exists()

    warmup_rows = [json.loads(l) for l in (tmp_path / "warmup_requests.jsonl").read_text().strip().splitlines()]
    assert len(warmup_rows) == 2
    assert all(r["request_id"] < 0 for r in warmup_rows)  # negative IDs, never collide with real plan

    requests_rows = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert all(r["request_id"] >= 0 for r in requests_rows)  # no warmup rows leaked in

    summary = json.loads((tmp_path / "summary.json").read_text())
    for policy_metrics in summary["per_policy"].values():
        assert policy_metrics["n_total"] == 1  # only the real 1-request plan, not +2 warmup


def test_no_warmup_flag_skips_warmup_files(tmp_path):
    mod = _load_module()
    mod.main([
        "--mock", "--policies", "fifo",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert not (tmp_path / "warmup_requests.jsonl").exists()


# ---------------------------------------------------------------------------
# Real HTTP/SSE dispatch against a fake server (not real vLLM)
# ---------------------------------------------------------------------------

def test_dispatch_against_fake_server_parses_usage_and_text(fake_vllm_server):
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 1, seed=1)
    out = mod._dispatch(plan[0], model="fake-model", base_url=fake_vllm_server, mock=False, timeout_s=10)
    assert out["text"] == "hi there."
    assert out["finish_reason"] == "stop"
    assert out["output_tokens"] == 2.0
    assert out["prompt_tokens"] == 10.0
    assert out["ttft_seconds"] is not None


def test_end_to_end_against_fake_server(tmp_path, fake_vllm_server):
    mod = _load_module()
    result = mod.main([
        "--allow-live-server", "--server-url", fake_vllm_server,
        "--policies", "fifo,edf",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1,2", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    assert (tmp_path / "server_status.json").exists()
    server_status = json.loads((tmp_path / "server_status.json").read_text())
    assert "data" in server_status
    rows = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert all(r["status"] == "success" for r in rows)


# ---------------------------------------------------------------------------
# No hosted provider SDKs, no secrets
# ---------------------------------------------------------------------------

def test_no_hosted_provider_sdks_imported(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, {str(ROOT / "scripts")!r})
import run_vllm_external_baseline_comparison as mod
mod.main(["--mock", "--policies", "fifo", "--output-dir", {str(tmp_path)!r}])
for forbidden in ("cohere", "google.genai", "openai", "azure"):
    assert forbidden not in sys.modules, f"unexpectedly imported {{forbidden}}"
print("OK")
""",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_no_secrets_written_to_outputs(tmp_path, monkeypatch):
    secret = "sk-SHOULD-NEVER-APPEAR-EXTBASE-12345"
    monkeypatch.setenv("COHERE_API_KEY", secret)
    mod = _load_module()
    mod.main([
        "--mock", "--policies", "fifo,edf",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert secret not in f.read_text(errors="ignore"), f"secret leaked into {f}"


# ---------------------------------------------------------------------------
# Required output files present
# ---------------------------------------------------------------------------

REQUIRED_FILES = (
    "request_plan.jsonl", "requests.jsonl", "summary.json", "summary.md",
    "aggregate_by_policy.csv", "aggregate_by_concurrency.csv",
    "aggregate_by_target_output_tokens.csv", "aggregate_by_prompt_bucket.csv",
    "manifest.json", "run_config.json", "reproducibility.md", "errors.jsonl",
)


def test_all_required_output_files_present(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock", "--policies", "fifo,edf",
        "--prompt-buckets", "short,medium", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    for fname in REQUIRED_FILES:
        assert (tmp_path / fname).exists(), f"missing {fname}"
