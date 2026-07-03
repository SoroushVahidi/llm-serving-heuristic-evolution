"""Tests for scripts/run_vllm_serving_baseline_pilot.py.

No real vLLM server, no GPU, no network beyond localhost is required: the
HTTP/SSE-parsing logic is validated against a small stdlib http.server fake
that reproduces vLLM's documented OpenAI-compatible streaming-completions
response shape (see the script's module docstring for why this couldn't be
validated against a real vLLM instance in this environment). Everything
else uses --dry-run / --mock, which never open a network connection at all.
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


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_vllm_serving_baseline_pilot", ROOT / "scripts" / "run_vllm_serving_baseline_pilot.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fake vLLM-OpenAI-compatible server (SSE streaming completions + /health)
# ---------------------------------------------------------------------------

class _FakeVllmHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence test output
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain request body

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        chunks = [
            {"choices": [{"text": "Hello ", "finish_reason": None}]},
            {"choices": [{"text": "world.", "finish_reason": "stop"}]},
            {"choices": [{"text": "", "finish_reason": None}],
             "usage": {"prompt_tokens": 42, "completion_tokens": 2}},
        ]
        for chunk in chunks:
            time.sleep(0.02)  # make TTFT measurable and nonzero
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
# Argument parsing
# ---------------------------------------------------------------------------

def test_requires_dry_run_mock_or_live(tmp_path):
    mod = _load_module()
    result = mod.main(["--output-dir", str(tmp_path)])
    assert result == 2


def test_unknown_prompt_bucket_rejected(tmp_path):
    mod = _load_module()
    result = mod.main(["--dry-run", "--prompt-buckets", "extra_long", "--output-dir", str(tmp_path)])
    assert result == 2


def test_default_model_is_small_open_model():
    mod = _load_module()
    args = mod.parse_args(["--dry-run", "--output-dir", "/tmp/x"])
    assert args.model == "Qwen/Qwen2.5-0.5B"


def test_default_grid_matches_hosted_pilot_shape():
    mod = _load_module()
    args = mod.parse_args(["--dry-run", "--output-dir", "/tmp/x"])
    assert args.target_output_tokens_list == [64, 128, 256]
    assert args.concurrency_list == [1, 2, 4, 8]
    assert args.requests_per_cell == 3


# ---------------------------------------------------------------------------
# Dry-run: no network, no vllm import, correct plan
# ---------------------------------------------------------------------------

def test_dry_run_plans_108_requests(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--dry-run",
        "--prompt-buckets", "short,medium,long",
        "--target-output-tokens-list", "64,128,256",
        "--concurrency-list", "1,2,4,8",
        "--requests-per-cell", "3",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["planned_requests"] == 3 * 3 * 4 * 3 == 108
    assert not (tmp_path / "requests.jsonl").exists()


def test_dry_run_reports_vllm_not_installed_status(tmp_path):
    mod = _load_module()
    mod.main(["--dry-run", "--output-dir", str(tmp_path)])
    cfg = json.loads((tmp_path / "run_config.json").read_text())
    if mod.vllm_cli_available():
        pytest.skip("vLLM CLI is installed in this environment; status would differ")
    assert cfg["run_status"] == "planned_only_vllm_not_installed"


def test_dry_run_writes_reproducibility_and_manifest(tmp_path):
    mod = _load_module()
    mod.main(["--dry-run", "--output-dir", str(tmp_path)])
    assert (tmp_path / "run_config.json").exists()
    assert (tmp_path / "reproducibility.md").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "summary.md").exists() or (tmp_path / "summary.json").exists()


def test_dry_run_never_imports_vllm(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, {str(ROOT / "scripts")!r})
import run_vllm_serving_baseline_pilot as mod
mod.main(["--dry-run", "--output-dir", {str(tmp_path)!r}])
assert "vllm" not in sys.modules, "dry-run must never import vllm"
print("OK")
""",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# Mock mode: full pipeline, still no network
# ---------------------------------------------------------------------------

def test_mock_mode_produces_expected_schema(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert len(lines) == 2
    for row in lines:
        assert set(row.keys()) == mod.VLLM_RESULT_FIELDS
        assert row["status"] == "success"
        assert row["ttft_seconds"] is not None


def test_mock_mode_writes_summary_and_aggregate(tmp_path):
    mod = _load_module()
    mod.main([
        "--mock",
        "--prompt-buckets", "short,medium", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "aggregate_by_target_output_tokens.csv").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status_counts"]["success"] == 2 * 2 * 2 * 2  # 16


def test_mock_mode_never_opens_network(tmp_path):
    # Point --server-url at a port nothing is listening on; --mock must
    # still succeed because it never uses base_url at all.
    mod = _load_module()
    result = mod.main([
        "--mock", "--server-url", "http://127.0.0.1:1",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0


def test_refuses_to_overwrite_nonempty_output_dir(tmp_path):
    mod = _load_module()
    args = [
        "--mock", "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1", "--output-dir", str(tmp_path),
    ]
    assert mod.main(args) == 0
    assert mod.main(args) == 3


def test_hard_cap_violation_refuses_before_dispatch(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--dry-run",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "10",
        "--max-total-requests", "3",
        "--output-dir", str(tmp_path),
    ])
    assert result == 4
    assert not (tmp_path / "requests.jsonl").exists()


# ---------------------------------------------------------------------------
# Live-server gating: no vLLM installed in this environment -> graceful exit
# ---------------------------------------------------------------------------

def test_allow_live_server_without_server_url_or_launch_flag_errors(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--allow-live-server",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 2


def test_launch_server_without_vllm_cli_fails_gracefully(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "vllm_cli_available", lambda: False)
    result = mod.main([
        "--allow-live-server", "--launch-server",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 6
    assert not (tmp_path / "requests.jsonl").exists()


def test_launch_local_vllm_server_raises_clear_error_when_cli_missing(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "vllm_cli_available", lambda: False)
    with pytest.raises(RuntimeError, match="vLLM CLI not found"):
        mod.launch_local_vllm_server("Qwen/Qwen2.5-0.5B", port=8000)


# ---------------------------------------------------------------------------
# Real HTTP/SSE query logic, validated against a fake local server (not a
# real vLLM instance — see module docstring)
# ---------------------------------------------------------------------------

def test_wait_for_server_ready_detects_health_endpoint(fake_vllm_server):
    mod = _load_module()
    assert mod.wait_for_server_ready(fake_vllm_server, timeout_s=5, poll_interval_s=0.1) is True


def test_wait_for_server_ready_times_out_when_unreachable():
    mod = _load_module()
    assert mod.wait_for_server_ready("http://127.0.0.1:1", timeout_s=0.3, poll_interval_s=0.1) is False


def test_query_vllm_completion_parses_ttft_text_and_usage(fake_vllm_server):
    mod = _load_module()
    out = mod.query_vllm_completion(
        fake_vllm_server, model="Qwen/Qwen2.5-0.5B", prompt="hello", max_tokens=16, timeout_s=10,
    )
    assert out["text"] == "Hello world."
    assert out["finish_reason"] == "stop"
    assert out["ttft_seconds"] is not None and out["ttft_seconds"] > 0
    assert out["server_request_latency_seconds"] >= out["ttft_seconds"]
    assert out["prompt_tokens"] == 42.0
    assert out["output_tokens"] == 2.0


def test_execute_one_request_against_fake_server(fake_vllm_server):
    mod = _load_module()
    planned = mod.VllmPlannedRequest(
        request_id="r1", prompt_bucket="short", target_output_tokens=64,
        concurrency_level=1, request_index=0, intended_prompt_tokens=100,
        prompt_text="hello",
    )
    result = mod.execute_one_request(
        planned, model="Qwen/Qwen2.5-0.5B", base_url=fake_vllm_server, timeout_s=10,
        mock=False, min_output_token_ratio=0.70, output_text_preview_chars=80,
    )
    assert result.status == "success"
    assert result.output_tokens == 2.0
    assert result.ttft_seconds is not None
    assert result.output_text_preview == "Hello world."


def test_execute_one_request_handles_connection_error_gracefully():
    mod = _load_module()
    planned = mod.VllmPlannedRequest(
        request_id="r1", prompt_bucket="short", target_output_tokens=64,
        concurrency_level=1, request_index=0, intended_prompt_tokens=100,
        prompt_text="hello",
    )
    result = mod.execute_one_request(
        planned, model="Qwen/Qwen2.5-0.5B", base_url="http://127.0.0.1:1", timeout_s=2,
        mock=False, min_output_token_ratio=0.70, output_text_preview_chars=80,
    )
    assert result.status == "error"
    assert result.error_type is not None


# ---------------------------------------------------------------------------
# No secrets, no live hosted-API SDKs ever imported
# ---------------------------------------------------------------------------

def test_no_hosted_provider_sdks_imported(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, {str(ROOT / "scripts")!r})
import run_vllm_serving_baseline_pilot as mod
mod.main(["--mock", "--output-dir", {str(tmp_path)!r}])
for forbidden in ("cohere", "google.genai", "openai", "azure"):
    assert forbidden not in sys.modules, f"unexpectedly imported {{forbidden}}"
print("OK")
""",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_no_secrets_written_to_output_files(tmp_path, monkeypatch):
    secret = "sk-SHOULD-NEVER-APPEAR-VLLM-12345"
    monkeypatch.setenv("COHERE_API_KEY", secret)
    monkeypatch.setenv("GOOGLE_API_KEY", secret)
    mod = _load_module()
    mod.main([
        "--mock",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert secret not in f.read_text(errors="ignore"), f"secret leaked into {f}"
