#!/usr/bin/env python3
"""Tiny fidelity diagnostics for real vLLM prefill/decode validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/soroush/llm-serving-heuristic-evolution")
VENV_BIN = Path("/home/soroush/.venvs/vllm_real_validation_v1/bin")
MODEL_PATH = Path(
    "/home/soroush/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
MODEL_NAME = "qwen05b-local"
OUT_DIR = REPO_ROOT / "experiments/real_vllm_mechanism_validation_v1/prefill_decode_fidelity_diagnosis_v1"
SEED = 20260824


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, separators=(",", ": "))


def load_validation_runner():
    path = REPO_ROOT / "experiments/real_vllm_mechanism_validation_v1/run_prefill_decode_local_v1.py"
    spec = importlib.util.spec_from_file_location("prefill_decode_local_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_sitecustomize(trace_path: Path, patch_dir: Path) -> None:
    patch_dir.mkdir(parents=True, exist_ok=True)
    code = f'''
import json
import os
import time

TRACE_PATH = {str(trace_path)!r}

try:
    from vllm.v1.core.sched.scheduler import Scheduler
    _ORIGINAL_SCHEDULE = Scheduler.schedule

    def _req_snapshot(req):
        return {{
            "request_id": getattr(req, "request_id", None),
            "num_computed_tokens": getattr(req, "num_computed_tokens", None),
            "num_prompt_tokens": getattr(req, "num_prompt_tokens", None),
            "num_tokens": getattr(req, "num_tokens", None),
            "is_prefill_chunk": bool(getattr(req, "is_prefill_chunk", False)),
            "status": str(getattr(req, "status", None)),
        }}

    def _traced_schedule(self, *args, **kwargs):
        before = {{}}
        try:
            for req_id, req in getattr(self, "requests", {{}}).items():
                before[req_id] = _req_snapshot(req)
        except Exception:
            before = {{}}
        out = _ORIGINAL_SCHEDULE(self, *args, **kwargs)
        try:
            scheduled = []
            prefill_tokens = 0
            decode_tokens = 0
            for req_id, n_tokens in getattr(out, "num_scheduled_tokens", {{}}).items():
                snap = before.get(req_id, {{}})
                req = getattr(self, "requests", {{}}).get(req_id)
                prompt = snap.get("num_prompt_tokens")
                computed = snap.get("num_computed_tokens")
                if prompt is None and req is not None:
                    prompt = getattr(req, "num_prompt_tokens", None)
                if computed is None and req is not None:
                    computed = getattr(req, "num_computed_tokens", None)
                if prompt is not None and computed is not None:
                    p = max(0, min(int(n_tokens), int(prompt) - int(computed)))
                    d = max(0, int(n_tokens) - p)
                else:
                    p = None
                    d = None
                if p is not None:
                    prefill_tokens += p
                if d is not None:
                    decode_tokens += d
                scheduled.append({{
                    "request_id": req_id,
                    "num_scheduled_tokens": int(n_tokens),
                    "computed_before": computed,
                    "prompt_tokens": prompt,
                    "prefill_tokens": p,
                    "decode_tokens": d,
                    "was_prefill_chunk_before": snap.get("is_prefill_chunk"),
                }})
            row = {{
                "time": time.time(),
                "pid": os.getpid(),
                "current_step": getattr(self, "current_step", None),
                "max_num_scheduled_tokens": getattr(self, "max_num_scheduled_tokens", None),
                "running_len_after_schedule": len(getattr(self, "running", [])),
                "waiting_len_after_schedule": len(getattr(self, "waiting", [])),
                "scheduled_req_count": len(scheduled),
                "total_num_scheduled_tokens": getattr(out, "total_num_scheduled_tokens", None),
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "has_prefill_and_decode": prefill_tokens > 0 and decode_tokens > 0,
                "scheduled": scheduled,
            }}
            with open(TRACE_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\\n")
        except Exception as exc:
            with open(TRACE_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({{"time": time.time(), "pid": os.getpid(), "trace_error": repr(exc)}}, sort_keys=True) + "\\n")
        return out

    Scheduler.schedule = _traced_schedule
except Exception as exc:
    with open(TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({{"time": time.time(), "pid": os.getpid(), "patch_error": repr(exc)}}, sort_keys=True) + "\\n")
'''
    (patch_dir / "sitecustomize.py").write_text(code, encoding="utf-8")


def get_url(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def wait_health(port: int, timeout_s: float = 240.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            get_url(f"http://127.0.0.1:{port}/health", timeout=2.0)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"server on port {port} did not become healthy: {last_error}")


def launch_server(treatment: str, port: int, trace_path: Path, patch_dir: Path, log_path: Path) -> subprocess.Popen[str]:
    max_batched = 4096 if treatment == "FULL" else 512
    chunk_flag = "--no-enable-chunked-prefill" if treatment == "FULL" else "--enable-chunked-prefill"
    cmd = [
        str(VENV_BIN / "vllm"),
        "serve",
        str(MODEL_PATH),
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.35",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        str(max_batched),
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--enforce-eager",
        chunk_flag,
    ]
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_SCHED_TRACE_PATH": str(trace_path),
            "PYTHONPATH": str(patch_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    proc._llmserveopt_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
    log_fh = getattr(proc, "_llmserveopt_log_fh", None)
    if log_fh:
        log_fh.close()


def run_scheduled_requests(runner: Any, port: int, rows: list[dict[str, Any]], treatment: str, label: str) -> list[dict[str, Any]]:
    t0 = time.monotonic()
    results = []

    def task(row: dict[str, Any]) -> dict[str, Any]:
        delay = t0 + row["arrival_offset_s"] - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        meta = {
            "treatment": treatment,
            "probe": label,
            "request_id": row["request_id"],
            "arrival_offset_s": row["arrival_offset_s"],
            "workload_class": row["workload_class"],
            "actual_input_tokens": row["actual_input_tokens"],
            "max_output_tokens": row["max_output_tokens"],
        }
        return runner.send_streaming_request(port, row["request_id"], row["prompt"], row["max_output_tokens"], meta)

    with ThreadPoolExecutor(max_workers=len(rows)) as pool:
        futures = [pool.submit(task, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: (r["arrival_offset_s"], r["request_id"]))
    return results


def make_prompt(tokenizer: Any, target_tokens: int, seed: int, kind: str) -> tuple[str, int]:
    randomizer = random.Random(seed)
    vocab = ["prefill", "decode", "queue", "batch", "token", "latency", "engine", "iteration"]
    chunks = []
    while True:
        chunks.append(f"{kind} {randomizer.choice(vocab)} block {len(chunks)}. " * 2)
        ids = tokenizer.encode("".join(chunks), add_special_tokens=False)
        if len(ids) >= target_tokens + 16:
            text = tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)
            actual = len(tokenizer.encode(text, add_special_tokens=False))
            while actual > target_tokens:
                text = text.rsplit(" ", 1)[0]
                actual = len(tokenizer.encode(text, add_special_tokens=False))
            return text, actual


def build_overlap_probe(tokenizer: Any) -> list[dict[str, Any]]:
    rows = []
    for i, length in enumerate([3200, 3280, 3360]):
        prompt, actual = make_prompt(tokenizer, length, SEED + i, "hog")
        rows.append(
            {
                "request_id": f"overlap.hog.{i}",
                "arrival_offset_s": 0.0,
                "workload_class": "hog",
                "prompt_token_target": length,
                "actual_input_tokens": actual,
                "max_output_tokens": 96,
                "prompt": prompt,
            }
        )
    for i, length in enumerate([96, 104, 112, 96]):
        prompt, actual = make_prompt(tokenizer, length, SEED + 100 + i, "late")
        rows.append(
            {
                "request_id": f"overlap.late.{i}",
                "arrival_offset_s": 0.05 + 0.01 * i,
                "workload_class": "late",
                "prompt_token_target": length,
                "actual_input_tokens": actual,
                "max_output_tokens": 24,
                "prompt": prompt,
            }
        )
    return rows


def summarize_trace(trace_path: Path) -> dict[str, Any]:
    rows = []
    if trace_path.exists():
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    valid = [r for r in rows if "scheduled" in r]
    both = [r for r in valid if r.get("has_prefill_and_decode")]
    prefill_only = [r for r in valid if r.get("prefill_tokens", 0) > 0 and r.get("decode_tokens", 0) == 0]
    decode_only = [r for r in valid if r.get("decode_tokens", 0) > 0 and r.get("prefill_tokens", 0) == 0]
    partial_prefills = []
    for row in valid:
        for item in row.get("scheduled", []):
            p = item.get("prefill_tokens")
            prompt = item.get("prompt_tokens")
            computed = item.get("computed_before")
            if p and prompt is not None and computed is not None and computed + p < prompt:
                partial_prefills.append({"step": row.get("current_step"), **item})
    return {
        "trace_path": str(trace_path),
        "raw_rows": len(rows),
        "scheduled_steps": len(valid),
        "steps_with_prefill_and_decode": len(both),
        "steps_prefill_only": len(prefill_only),
        "steps_decode_only": len(decode_only),
        "max_prefill_tokens_in_step": max([r.get("prefill_tokens", 0) for r in valid], default=0),
        "max_decode_tokens_in_step": max([r.get("decode_tokens", 0) for r in valid], default=0),
        "max_total_tokens_in_step": max([r.get("total_num_scheduled_tokens", 0) or 0 for r in valid], default=0),
        "partial_prefill_chunks": len(partial_prefills),
        "first_partial_prefill_chunks": partial_prefills[:10],
        "example_mixed_steps": both[:10],
        "trace_errors": [r for r in rows if "trace_error" in r or "patch_error" in r],
    }


def run_microbench(runner: Any, tokenizer: Any, port: int) -> dict[str, Any]:
    lengths = [96, 512, 1024, 2048, 3200]
    prefill = []
    for length in lengths:
        prompt, actual = make_prompt(tokenizer, length, SEED + 500 + length, "micro")
        row = runner.send_streaming_request(
            port,
            f"micro.prefill.{length}",
            prompt,
            1,
            {"probe": "prefill_microbench", "prompt_token_target": length, "actual_input_tokens": actual},
        )
        prefill.append(row)
    batch_rows = []
    prompts = []
    for i in range(4):
        prompt, actual = make_prompt(tokenizer, 96, SEED + 900 + i, "decode")
        prompts.append(
            {
                "request_id": f"micro.decode_batch.{i}",
                "arrival_offset_s": 0.0,
                "workload_class": "decode_batch",
                "actual_input_tokens": actual,
                "max_output_tokens": 64,
                "prompt": prompt,
            }
        )
    batch_rows = run_scheduled_requests(runner, port, prompts, "FULL", "decode_microbench")
    itls = []
    for row in batch_rows:
        output = row.get("output_tokens") or 0
        if row.get("success") and output > 1 and row.get("ttft_s") is not None:
            itls.append((row["e2e_s"] - row["ttft_s"]) / max(1, output - 1))
    prefill_3200 = next(r for r in prefill if r["request_id"] == "micro.prefill.3200")
    representative_decode_itl = sum(itls) / len(itls) if itls else None
    return {
        "prefill_single_request_rows": prefill,
        "decode_batch_rows": batch_rows,
        "representative_decode_itl_s": representative_decode_itl,
        "prefill_3200_ttft_to_decode_itl_ratio": (
            prefill_3200["ttft_s"] / representative_decode_itl if representative_decode_itl else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "server_logs").mkdir(exist_ok=True)
    patch_dir = out_dir / "instrumentation"

    runner = load_validation_runner()
    tokenizer = runner.load_tokenizer()
    overlap_manifest = build_overlap_probe(tokenizer)
    (out_dir / "diagnostic_probe_manifest.json").write_text(stable_json(overlap_manifest) + "\n", encoding="utf-8")

    all_request_rows = []
    overlap_summaries = {}
    microbench = None
    for treatment, port in [("FULL", 8071), ("CHUNKED", 8072)]:
        trace_path = out_dir / f"scheduler_trace_{treatment.lower()}.jsonl"
        write_sitecustomize(trace_path, patch_dir)
        proc = launch_server(
            treatment,
            port,
            trace_path,
            patch_dir,
            out_dir / "server_logs" / f"{treatment.lower()}_diagnostic.log",
        )
        try:
            wait_health(port)
            runner.warmup(port)
            rows = run_scheduled_requests(runner, port, overlap_manifest, treatment, "overlap_probe")
            all_request_rows.extend(rows)
            if treatment == "FULL":
                microbench = run_microbench(runner, tokenizer, port)
            time.sleep(1.0)
        finally:
            stop_server(proc)
            time.sleep(3.0)
        overlap_summaries[treatment] = summarize_trace(trace_path)

    with (out_dir / "diagnostic_requests.jsonl").open("w", encoding="utf-8") as fh:
        for row in all_request_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "contention_overlap_analysis.json").write_text(stable_json(overlap_summaries) + "\n", encoding="utf-8")
    (out_dir / "prefill_decode_microbenchmark.json").write_text(stable_json(microbench) + "\n", encoding="utf-8")

    decision = {
        "overlap_probe_success": all(r.get("success") for r in all_request_rows),
        "full_steps_with_prefill_and_decode": overlap_summaries["FULL"]["steps_with_prefill_and_decode"],
        "chunked_steps_with_prefill_and_decode": overlap_summaries["CHUNKED"]["steps_with_prefill_and_decode"],
        "chunked_partial_prefill_chunks": overlap_summaries["CHUNKED"]["partial_prefill_chunks"],
        "full_partial_prefill_chunks": overlap_summaries["FULL"]["partial_prefill_chunks"],
        "prefill_3200_ttft_to_decode_itl_ratio": microbench["prefill_3200_ttft_to_decode_itl_ratio"] if microbench else None,
        "diagnosis": "pending_analysis",
    }
    (out_dir / "diagnostic_probe_decision_raw.json").write_text(stable_json(decision) + "\n", encoding="utf-8")
    print(stable_json(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
