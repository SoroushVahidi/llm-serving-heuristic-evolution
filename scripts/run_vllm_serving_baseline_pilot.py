#!/usr/bin/env python3
"""
vLLM real-serving external-baseline pilot.

Unlike the Cohere/Gemini hosted-API pilots (black-box, no visibility into
provider-internal batching/scheduling — see
docs/real_llm_simulator_integration_plan.md), vLLM is a serving engine this
project can actually inspect: its batching, KV-cache paging, and scheduling
are things a future task could compare directly against the simulator's own
policies, not just against an opaque wall-clock number.

This script does NOT call Cohere, Gemini, OpenAI, Azure, or any other
hosted API. It only ever talks to a vLLM server (local subprocess or an
already-running one at --server-url, e.g. on an HPC node) via its
documented OpenAI-compatible HTTP endpoint.

**Environment note:** vLLM is NOT installed in this repo's environment
(CUDA 13.0 / PyTorch 2.12.0 is too new for vLLM's prebuilt wheels — see
configs/gpu_calibration/online_validation.yaml, which documents the same
constraint for the GPU-calibration pipeline). The live-server code paths
below (query_vllm_completion, launch_local_vllm_server) are written against
vLLM's documented OpenAI-compatible streaming-completions protocol but have
NOT been exercised against a real running vLLM instance here — tests
instead validate the parsing logic against a small stdlib-http.server fake
that reproduces the same response shape (see
tests/test_run_vllm_serving_baseline_pilot.py). Do not treat this as a
verified integration until it has actually been run against real vLLM.

DEFAULT MODE (no flags): refuse to run anything.
--dry-run:            plan the request grid, write manifest and
                       reproducibility metadata, never import vllm or open
                       any network connection.
--allow-live-server:  actually issue HTTP requests to a vLLM server.
                       Requires either --server-url (query an already-
                       running server) or --launch-server (spawn one
                       locally via the `vllm` CLI, which must be on PATH).
--mock:                replace the vLLM HTTP calls with a local
                       deterministic stub (for tests only; no network).

Usage (dry-run):
    python scripts/run_vllm_serving_baseline_pilot.py \\
        --dry-run \\
        --model Qwen/Qwen2.5-0.5B \\
        --output-dir experiments/real_llm/vllm_serving_baseline_pilot

Usage (query an already-running server, e.g. on HPC):
    python scripts/run_vllm_serving_baseline_pilot.py \\
        --allow-live-server --server-url http://localhost:8000 \\
        --model Qwen/Qwen2.5-0.5B \\
        --output-dir experiments/real_llm/vllm_serving_baseline_pilot_TIMESTAMP

Usage (launch a local server, requires `vllm` on PATH):
    python scripts/run_vllm_serving_baseline_pilot.py \\
        --allow-live-server --launch-server --port 8000 \\
        --model Qwen/Qwen2.5-0.5B \\
        --output-dir experiments/real_llm/vllm_serving_baseline_pilot_TIMESTAMP
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"  # same small model as the GPU-calibration pipeline (Phase 1.7B)
KNOWN_PROMPT_BUCKETS = cc.KNOWN_PROMPT_BUCKETS


# ---------------------------------------------------------------------------
# Plan / result schema (field names mirror calibration_common.RequestResult
# where the concept is the same, so future analysis code — e.g. a v3 of
# fit_real_llm_latency_model_v2.py — can treat vLLM logs the same way)
# ---------------------------------------------------------------------------

@dataclass
class VllmPlannedRequest:
    request_id: str
    prompt_bucket: str
    target_output_tokens: int
    concurrency_level: int
    request_index: int
    intended_prompt_tokens: int
    prompt_text: str = field(repr=False)


@dataclass
class VllmRequestResult:
    request_id: str
    model: str
    prompt_bucket: str
    target_output_tokens: int
    intended_prompt_tokens: int
    actual_prompt_tokens: Optional[float]
    concurrency_level: int
    request_index: int
    start_time_iso: str
    end_time_iso: str
    ttft_seconds: Optional[float]
    server_request_latency_seconds: Optional[float]
    total_wall_time_seconds: float
    output_tokens: Optional[float]
    finish_reason: Optional[str]
    status: str  # success | error | timeout | skipped
    error_type: Optional[str]
    error_message: Optional[str]
    reached_target_output_range: Optional[bool]
    output_text_preview: Optional[str]


VLLM_RESULT_FIELDS = frozenset(VllmRequestResult.__dataclass_fields__.keys())


def expand_plan(
    prompt_buckets: List[str], target_output_tokens_list: List[int],
    concurrency_list: List[int], requests_per_cell: int, seed: int,
) -> List[VllmPlannedRequest]:
    plan: List[VllmPlannedRequest] = []
    for bucket, target, concurrency in product(prompt_buckets, target_output_tokens_list, concurrency_list):
        for i in range(requests_per_cell):
            prompt_text = cc.build_length_targeted_prompt(bucket, target, seed, i)
            plan.append(VllmPlannedRequest(
                request_id=f"{bucket}__tgt{target}__c{concurrency}__i{i}",
                prompt_bucket=bucket,
                target_output_tokens=target,
                concurrency_level=concurrency,
                request_index=i,
                intended_prompt_tokens=cc.approx_token_count(prompt_text),
                prompt_text=prompt_text,
            ))
    return plan


# ---------------------------------------------------------------------------
# vLLM availability / server lifecycle (best-effort; untested against real
# vLLM in this environment — see module docstring)
# ---------------------------------------------------------------------------

def vllm_cli_available() -> bool:
    return shutil.which("vllm") is not None


def launch_local_vllm_server(model: str, port: int, extra_args: Optional[List[str]] = None) -> subprocess.Popen:
    if not vllm_cli_available():
        raise RuntimeError(
            "vLLM CLI not found on PATH. Install vLLM (pip install vllm) on "
            "hardware/CUDA it supports, or query an already-running server "
            "with --server-url instead of --launch-server."
        )
    cmd = ["vllm", "serve", model, "--port", str(port)] + (extra_args or [])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for_server_ready(base_url: str, timeout_s: float, poll_interval_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(poll_interval_s)
    return False


# ---------------------------------------------------------------------------
# HTTP query against vLLM's OpenAI-compatible streaming completions endpoint
# ---------------------------------------------------------------------------

def query_vllm_completion(
    base_url: str, model: str, prompt: str, max_tokens: int, timeout_s: float,
) -> Dict[str, Any]:
    """Issue one streaming /v1/completions request and measure TTFT/latency.

    Implements vLLM's documented OpenAI-compatible server-sent-events
    protocol (each line `data: {...}`, terminated by `data: [DONE]`, with
    `stream_options.include_usage` requesting token counts in the final
    chunk). See the module docstring: this has not been run against a real
    vLLM server in this environment.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )

    t0 = time.monotonic()
    first_token_t: Optional[float] = None
    text_chunks: List[str] = []
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            choices = chunk.get("choices") or []
            if choices:
                delta_text = choices[0].get("text") or ""
                if delta_text and first_token_t is None:
                    first_token_t = time.monotonic()
                if delta_text:
                    text_chunks.append(delta_text)
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]
    t1 = time.monotonic()

    return {
        "text": "".join(text_chunks),
        "finish_reason": finish_reason,
        "ttft_seconds": (first_token_t - t0) if first_token_t is not None else None,
        "server_request_latency_seconds": t1 - t0,
        "prompt_tokens": float(usage["prompt_tokens"]) if usage and usage.get("prompt_tokens") is not None else None,
        "output_tokens": float(usage["completion_tokens"]) if usage and usage.get("completion_tokens") is not None else None,
    }


def mock_call(planned: VllmPlannedRequest) -> Dict[str, Any]:
    """Generic deterministic stub — no network, no vllm import. Mirrors
    calibration_common.mock_call's role for the hosted-API scripts."""
    time.sleep(0.001)
    output_tokens = float(min(planned.target_output_tokens, 8))
    return {
        "text": "mock vllm response text",
        "finish_reason": "stop",
        "ttft_seconds": 0.01,
        "server_request_latency_seconds": 0.02,
        "prompt_tokens": float(planned.intended_prompt_tokens),
        "output_tokens": output_tokens,
    }


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------

def execute_one_request(
    planned: VllmPlannedRequest, *, model: str, base_url: Optional[str], timeout_s: float,
    mock: bool, min_output_token_ratio: float, output_text_preview_chars: int,
) -> VllmRequestResult:
    start = datetime.now(timezone.utc)
    t_wall0 = time.monotonic()
    try:
        if mock:
            out = mock_call(planned)
        else:
            out = query_vllm_completion(
                base_url, model, planned.prompt_text,
                max_tokens=planned.target_output_tokens * 2, timeout_s=timeout_s,
            )
        end = datetime.now(timezone.utc)
        wall = time.monotonic() - t_wall0
        output_tokens = out["output_tokens"]
        reached = None
        if output_tokens is not None:
            reached = output_tokens >= min_output_token_ratio * planned.target_output_tokens
        preview = None
        if output_text_preview_chars > 0 and out.get("text"):
            preview = out["text"][:output_text_preview_chars]
        return VllmRequestResult(
            request_id=planned.request_id, model=model, prompt_bucket=planned.prompt_bucket,
            target_output_tokens=planned.target_output_tokens,
            intended_prompt_tokens=planned.intended_prompt_tokens,
            actual_prompt_tokens=out.get("prompt_tokens"),
            concurrency_level=planned.concurrency_level, request_index=planned.request_index,
            start_time_iso=start.isoformat(), end_time_iso=end.isoformat(),
            ttft_seconds=out["ttft_seconds"],
            server_request_latency_seconds=out["server_request_latency_seconds"],
            total_wall_time_seconds=round(wall, 4),
            output_tokens=output_tokens, finish_reason=out.get("finish_reason"),
            status="success", error_type=None, error_message=None,
            reached_target_output_range=reached, output_text_preview=preview,
        )
    except Exception as exc:  # noqa: BLE001 - classify broadly, log safely
        end = datetime.now(timezone.utc)
        wall = time.monotonic() - t_wall0
        status = "timeout" if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower() else "error"
        return VllmRequestResult(
            request_id=planned.request_id, model=model, prompt_bucket=planned.prompt_bucket,
            target_output_tokens=planned.target_output_tokens,
            intended_prompt_tokens=planned.intended_prompt_tokens, actual_prompt_tokens=None,
            concurrency_level=planned.concurrency_level, request_index=planned.request_index,
            start_time_iso=start.isoformat(), end_time_iso=end.isoformat(),
            ttft_seconds=None, server_request_latency_seconds=None,
            total_wall_time_seconds=round(wall, 4), output_tokens=None, finish_reason=None,
            status=status, error_type=type(exc).__name__, error_message=str(exc)[:500],
            reached_target_output_range=None, output_text_preview=None,
        )


def make_skipped_result(planned: VllmPlannedRequest, model: str, reason: str) -> VllmRequestResult:
    now = datetime.now(timezone.utc).isoformat()
    return VllmRequestResult(
        request_id=planned.request_id, model=model, prompt_bucket=planned.prompt_bucket,
        target_output_tokens=planned.target_output_tokens,
        intended_prompt_tokens=planned.intended_prompt_tokens, actual_prompt_tokens=None,
        concurrency_level=planned.concurrency_level, request_index=planned.request_index,
        start_time_iso=now, end_time_iso=now,
        ttft_seconds=None, server_request_latency_seconds=None, total_wall_time_seconds=0.0,
        output_tokens=None, finish_reason=None, status="skipped", error_type=None,
        error_message=reason[:500], reached_target_output_range=None, output_text_preview=None,
    )


def run_requests(
    plan: List[VllmPlannedRequest], out_dir: Path, *, model: str, base_url: Optional[str],
    timeout_s: float, mock: bool, min_output_token_ratio: float, output_text_preview_chars: int,
    max_total_requests: int,
) -> None:
    import concurrent.futures

    requests_path = out_dir / "requests.jsonl"
    fh = open(requests_path, "a")
    write_lock = threading.Lock()
    dispatched = 0
    dispatch_lock = threading.Lock()

    def _write(result: VllmRequestResult) -> None:
        with write_lock:
            fh.write(json.dumps(asdict(result)) + "\n")
            fh.flush()

    cells: Dict[tuple, List[VllmPlannedRequest]] = {}
    for p in plan:
        cells.setdefault((p.prompt_bucket, p.target_output_tokens, p.concurrency_level), []).append(p)

    for (bucket, target, concurrency), cell_requests in cells.items():
        def _run_one(planned: VllmPlannedRequest) -> VllmRequestResult:
            nonlocal dispatched
            with dispatch_lock:
                if dispatched >= max_total_requests:
                    return make_skipped_result(planned, model, "max_total_requests cap reached")
                dispatched += 1
            return execute_one_request(
                planned, model=model, base_url=base_url, timeout_s=timeout_s, mock=mock,
                min_output_token_ratio=min_output_token_ratio,
                output_text_preview_chars=output_text_preview_chars,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_run_one, p) for p in cell_requests]
            for fut in concurrent.futures.as_completed(futures):
                _write(fut.result())

    fh.close()


# ---------------------------------------------------------------------------
# Aggregation (mirrors calibration_common.aggregate_results' shape, no
# dollar-cost accounting since this is a local/self-hosted server)
# ---------------------------------------------------------------------------

def aggregate_results(out_dir: Path) -> Dict[str, Any]:
    import pandas as pd

    requests_path = out_dir / "requests.jsonl"
    rows: List[Dict[str, Any]] = []
    if requests_path.exists():
        with open(requests_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    successes = [r for r in rows if r["status"] == "success"]
    ttfts = [r["ttft_seconds"] for r in successes if r.get("ttft_seconds") is not None]
    latencies = [r["server_request_latency_seconds"] for r in successes if r.get("server_request_latency_seconds") is not None]
    output_tokens = [r["output_tokens"] for r in successes if r.get("output_tokens")]

    def stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "p50": cc._percentile(values, 0.50),
            "p95": cc._percentile(values, 0.95),
            "p99": cc._percentile(values, 0.99),
        }

    overall = {
        "total_records": len(rows),
        "status_counts": status_counts,
        "ttft_stats": stats(ttfts),
        "latency_stats": stats(latencies),
        "mean_output_tokens": (sum(output_tokens) / len(output_tokens)) if output_tokens else None,
    }

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    by_target_records: List[Dict[str, Any]] = []
    if not df.empty:
        for target, g in df.groupby("target_output_tokens"):
            g_success = g[g["status"] == "success"]
            out_toks = g_success["output_tokens"].dropna().tolist()
            reached = g_success["reached_target_output_range"].dropna().tolist()
            lat = g_success["server_request_latency_seconds"].dropna().tolist()
            by_target_records.append({
                "target_output_tokens": int(target),
                "n_total": int(len(g)),
                "n_success": int(len(g_success)),
                "mean_output_tokens": (sum(out_toks) / len(out_toks)) if out_toks else None,
                "mean_server_latency_s": (sum(lat) / len(lat)) if lat else None,
                "frac_reached_target_range": (sum(reached) / len(reached)) if reached else None,
            })
    by_target_records.sort(key=lambda r: r["target_output_tokens"])
    pd.DataFrame(by_target_records).to_csv(out_dir / "aggregate_by_target_output_tokens.csv", index=False)
    overall["by_target_output_tokens"] = by_target_records

    return overall


def write_summary(out_dir: Path, overall: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2))

    lines = [
        "# vLLM Serving Baseline Pilot — Summary",
        "",
        f"**Model:** `{cfg.get('model')}`",
        f"**Run status:** {cfg.get('run_status')}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    if cfg.get("run_status") != "completed":
        lines += [
            "**No live vLLM server was launched or queried for this run.** "
            "See `run_config.json`/`reproducibility.md` for why "
            f"(`{cfg.get('run_status')}`).",
            "",
        ]
    lines += [
        "## Status counts",
        "```json",
        json.dumps(overall.get("status_counts", {}), indent=2),
        "```",
        "",
        "## TTFT / server request latency (successful requests)",
        f"- TTFT (s): {overall['ttft_stats']}",
        f"- server_request_latency_seconds (s): {overall['latency_stats']}",
        f"- mean output tokens: {overall.get('mean_output_tokens')}",
        "",
        "See `aggregate_by_target_output_tokens.csv` for the breakdown by "
        "target output length, and `requests.jsonl` for the full per-request log.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Reproducibility metadata (git state only — no full-repo diff snapshot;
# see docs/real_llm_latency_model_v2.md's note on why that swept in
# unrelated content in an earlier pilot)
# ---------------------------------------------------------------------------

def _git_info(root: Path) -> Dict[str, Any]:
    branch = cc._run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    commit = cc._run_git(root, ["rev-parse", "HEAD"])
    dirty = bool(cc._run_git(root, ["status", "--porcelain"]))
    return {"git_branch": branch or None, "git_commit": commit or None, "git_dirty": dirty}


def write_reproducibility_md(out_dir: Path, cfg: Dict[str, Any], git_info: Dict[str, Any]) -> None:
    lines = [
        "# Reproducibility Metadata",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Git branch: `{git_info['git_branch']}`",
        f"- Git commit: `{git_info['git_commit']}`",
        f"- Git dirty: {git_info['git_dirty']}",
        f"- Run status: `{cfg.get('run_status')}`",
        "",
    ]
    if cfg.get("run_status") == "planned_only_vllm_not_installed":
        lines += [
            "**vLLM is not installed in this environment** (CUDA 13.0 / "
            "PyTorch 2.12.0 is too new for vLLM's prebuilt wheels — see "
            "`configs/gpu_calibration/online_validation.yaml`, which "
            "documents the same constraint for GPU calibration). This "
            "directory contains a dry-run plan only: no vLLM server was "
            "launched or queried, no GPU inference occurred. Rerun with "
            "`--allow-live-server` (plus `--server-url` or `--launch-server`) "
            "once vLLM is installed on compatible hardware.",
            "",
        ]
    lines += [
        "## Config",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
    ]
    (out_dir / "reproducibility.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-live-server", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Local deterministic stub; no network, no vllm import.")
    parser.add_argument("--server-url", default=None, help="Query an already-running vLLM OpenAI-compatible server.")
    parser.add_argument("--launch-server", action="store_true", help="Launch a local vLLM server via the `vllm` CLI.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-buckets", type=cc.csv_str_list, default=list(KNOWN_PROMPT_BUCKETS))
    parser.add_argument("--target-output-tokens-list", type=cc.csv_int_list, default=[64, 128, 256])
    parser.add_argument("--concurrency-list", type=cc.csv_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--requests-per-cell", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-total-requests", type=int, default=200)
    parser.add_argument("--min-output-token-ratio", type=float, default=0.70)
    parser.add_argument("--record-output-text-preview-chars", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def repo_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else root / p


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    unknown_buckets = set(args.prompt_buckets) - set(KNOWN_PROMPT_BUCKETS)
    if unknown_buckets:
        print(f"ERROR: unknown prompt buckets: {sorted(unknown_buckets)}. Known: {KNOWN_PROMPT_BUCKETS}", file=sys.stderr)
        return 2

    if not args.dry_run and not args.allow_live_server and not args.mock:
        print("ERROR: specify --dry-run, --mock, or --allow-live-server.", file=sys.stderr)
        return 2

    out_dir = repo_path(ROOT, args.output_dir)
    requests_path = out_dir / "requests.jsonl"
    if out_dir.exists() and requests_path.exists() and requests_path.stat().st_size > 0 and not args.resume:
        print(
            f"ERROR: output dir {out_dir} already has a non-empty requests.jsonl.\n"
            "Pass --resume to continue, or choose a new --output-dir.", file=sys.stderr,
        )
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = expand_plan(
        args.prompt_buckets, args.target_output_tokens_list, args.concurrency_list,
        args.requests_per_cell, args.seed,
    )
    if len(plan) > args.max_total_requests:
        print(
            f"HARD CAP VIOLATION: planned {len(plan)} requests exceeds "
            f"--max-total-requests={args.max_total_requests}", file=sys.stderr,
        )
        return 4

    git_info = _git_info(ROOT)
    run_status = "planned_only"
    base_url: Optional[str] = None
    launched_process: Optional[subprocess.Popen] = None

    if args.allow_live_server and not args.mock:
        if args.server_url:
            base_url = args.server_url
        elif args.launch_server:
            if not vllm_cli_available():
                print(
                    "ERROR: --launch-server requires the `vllm` CLI on PATH, which is "
                    "not installed in this environment. Install vLLM on compatible "
                    "hardware, or use --server-url to query an already-running server, "
                    "or use --dry-run/--mock instead.", file=sys.stderr,
                )
                return 6
            launched_process = launch_local_vllm_server(args.model, args.port)
            base_url = f"http://127.0.0.1:{args.port}"
            if not wait_for_server_ready(base_url, args.startup_timeout_seconds):
                launched_process.terminate()
                print(f"ERROR: vLLM server did not become ready within {args.startup_timeout_seconds}s.", file=sys.stderr)
                return 7
        else:
            print("ERROR: --allow-live-server requires --server-url or --launch-server.", file=sys.stderr)
            return 2
        run_status = "completed"
    elif args.mock:
        run_status = "completed_mock"
    elif args.dry_run:
        run_status = "planned_only" if vllm_cli_available() else "planned_only_vllm_not_installed"

    cfg = {
        "model": args.model,
        "seed": args.seed,
        "prompt_buckets": args.prompt_buckets,
        "target_output_tokens_list": args.target_output_tokens_list,
        "concurrency_list": args.concurrency_list,
        "requests_per_cell": args.requests_per_cell,
        "timeout_seconds": args.timeout_seconds,
        "max_total_requests": args.max_total_requests,
        "min_output_token_ratio": args.min_output_token_ratio,
        "record_output_text_preview_chars": args.record_output_text_preview_chars,
        "mock": args.mock,
        "server_url": args.server_url,
        "launch_server": args.launch_server,
        "run_status": run_status,
    }
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps({
        **git_info,
        "planned_requests": len(plan),
        "cells": sorted({(p.prompt_bucket, p.target_output_tokens, p.concurrency_level) for p in plan}),
        "requests_preview": [
            {
                "request_id": p.request_id, "prompt_bucket": p.prompt_bucket,
                "target_output_tokens": p.target_output_tokens,
                "concurrency_level": p.concurrency_level, "request_index": p.request_index,
                "intended_prompt_tokens": p.intended_prompt_tokens,
            }
            for p in plan[:5]
        ],
    }, indent=2))
    write_reproducibility_md(out_dir, cfg, git_info)

    print("vLLM serving baseline pilot")
    print(f"  output_dir:       {out_dir}")
    print(f"  planned_requests: {len(plan)}")
    print(f"  run_status:       {run_status}")

    if args.dry_run and not args.allow_live_server and not args.mock:
        write_summary(out_dir, {"total_records": 0, "status_counts": {}, "ttft_stats": {}, "latency_stats": {}}, cfg)
        print("  No vLLM server was launched or queried (dry-run).")
        return 0

    try:
        run_requests(
            plan, out_dir, model=args.model, base_url=base_url, timeout_s=args.timeout_seconds,
            mock=args.mock, min_output_token_ratio=args.min_output_token_ratio,
            output_text_preview_chars=args.record_output_text_preview_chars,
            max_total_requests=args.max_total_requests,
        )
    finally:
        if launched_process is not None:
            launched_process.terminate()

    overall = aggregate_results(out_dir)
    write_summary(out_dir, overall, cfg)
    print(f"  completed: {overall['status_counts'].get('success', 0)}")
    print(f"  failed:    {sum(overall['status_counts'].get(s, 0) for s in ('error', 'timeout'))}")
    print(f"  skipped:   {overall['status_counts'].get('skipped', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
