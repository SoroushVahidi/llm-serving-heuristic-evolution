#!/usr/bin/env python3
"""Native vLLM chunked-prefill budget semantics probe.

Both treatments keep chunked prefill enabled. The only intended intervention is
max_num_batched_tokens: 512 vs 4096.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
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
PREVIOUS_MANIFEST = (
    REPO_ROOT
    / "experiments/real_vllm_mechanism_validation_v1/prefill_decode_local_v1/workload_manifest.json"
)
PREVIOUS_MANIFEST_HASH = "194d8c5f2eb3f3dc8a6840c7a1a293e2afe895dae71e4e08748a6238eefe28c3"
OUT_DIR = (
    REPO_ROOT
    / "experiments/real_vllm_mechanism_validation_v1/native_vllm_chunk_budget_semantics_probe_v1"
)
SEED = 20260824
REGIMES = ["late_tight_low_late", "late_tight_high_late"]


SERVER_CONFIGS = {
    "T512": {"port": 8081, "max_num_batched_tokens": 512},
    "T4096": {"port": 8082, "max_num_batched_tokens": 4096},
}


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, separators=(",", ": "))


def load_validation_runner():
    path = REPO_ROOT / "experiments/real_vllm_mechanism_validation_v1/run_prefill_decode_local_v1.py"
    spec = importlib.util.spec_from_file_location("prefill_decode_local_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_sitecustomize(trace_path: Path, context_path: Path, patch_dir: Path) -> None:
    patch_dir.mkdir(parents=True, exist_ok=True)
    code = f'''
import json
import os
import time

TRACE_PATH = {str(trace_path)!r}
CONTEXT_PATH = {str(context_path)!r}

try:
    from vllm.v1.core.sched.scheduler import Scheduler
    _ORIGINAL_SCHEDULE = Scheduler.schedule

    def _load_context():
        try:
            with open(CONTEXT_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {{}}

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
            partial_prefill_items = 0
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
                    if p > 0 and int(computed) + p < int(prompt):
                        partial_prefill_items += 1
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
                **_load_context(),
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
                "partial_prefill_items": partial_prefill_items,
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


def launch_server(treatment: str, trace_path: Path, context_path: Path, patch_dir: Path, log_path: Path):
    cfg = SERVER_CONFIGS[treatment]
    cmd = [
        str(VENV_BIN / "vllm"),
        "serve",
        str(MODEL_PATH),
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg["port"]),
        "--gpu-memory-utilization",
        "0.35",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        str(cfg["max_num_batched_tokens"]),
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--enforce-eager",
        "--enable-chunked-prefill",
    ]
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "PYTHONPATH": str(patch_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    proc._llmserveopt_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def stop_server(proc) -> None:
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


def parse_metrics(text: str) -> dict[str, float]:
    import re

    patterns = {
        "waiting": re.compile(r"^vllm:num_requests_waiting\\{[^}]*\\}\\s+([0-9.eE+-]+)$"),
        "running": re.compile(r"^vllm:num_requests_running\\{[^}]*\\}\\s+([0-9.eE+-]+)$"),
        "kv_cache_usage": re.compile(r"^vllm:kv_cache_usage_perc\\{[^}]*\\}\\s+([0-9.eE+-]+)$"),
        "preemptions": re.compile(r"^vllm:num_preemptions_total\\{[^}]*\\}\\s+([0-9.eE+-]+)$"),
        "prompt_tokens_total": re.compile(r"^vllm:prompt_tokens_total\\{[^}]*\\}\\s+([0-9.eE+-]+)$"),
        "generation_tokens_total": re.compile(r"^vllm:generation_tokens_total\\{[^}]*\\}\\s+([0-9.eE+-]+)$"),
    }
    vals = {k: [] for k in patterns}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, pat in patterns.items():
            m = pat.match(line)
            if m:
                vals[key].append(float(m.group(1)))
    out = {}
    for key, row in vals.items():
        if row:
            out[key] = max(row) if key in {"waiting", "running", "kv_cache_usage"} else sum(row)
    return out


def poll_metrics(port: int, stop_event: threading.Event, context: dict[str, Any], rows: list[dict[str, Any]]):
    start = time.monotonic()
    while not stop_event.is_set():
        try:
            text = get_url(f"http://127.0.0.1:{port}/metrics", timeout=2.0)
            rows.append({**context, "wall_time": time.time(), "elapsed_s": time.monotonic() - start, "metrics": parse_metrics(text)})
        except Exception as exc:  # noqa: BLE001
            rows.append({**context, "wall_time": time.time(), "elapsed_s": time.monotonic() - start, "error": repr(exc)})
        time.sleep(0.05)


def run_regime(runner: Any, treatment: str, port: int, regime: str, repetition: int, request_rows: list[dict[str, Any]], context_path: Path, metrics_rows: list[dict[str, Any]]):
    context = {"treatment": treatment, "regime": regime, "repetition": repetition}
    context_path.write_text(json.dumps(context, sort_keys=True), encoding="utf-8")
    stop_event = threading.Event()
    poller = threading.Thread(target=poll_metrics, args=(port, stop_event, context, metrics_rows), daemon=True)
    poller.start()
    t0 = time.monotonic()
    results = []

    def task(row: dict[str, Any]):
        delay = t0 + float(row["arrival_offset_s"]) - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        metadata = {
            "treatment": treatment,
            "regime": regime,
            "repetition": repetition,
            "request_id": row["request_id"],
            "workload_class": row["workload_class"],
            "arrival_offset_s": row["arrival_offset_s"],
            "prompt_token_target": row["prompt_token_target"],
            "actual_input_tokens": row["actual_input_tokens"],
            "max_output_tokens": row["max_output_tokens"],
            "slo_metric": row.get("slo_metric"),
            "slo_threshold_s": row.get("slo_threshold_s"),
        }
        return runner.send_streaming_request(port, row["request_id"], row["prompt"], row["max_output_tokens"], metadata)

    with ThreadPoolExecutor(max_workers=len(request_rows)) as pool:
        for future in as_completed([pool.submit(task, row) for row in request_rows]):
            results.append(future.result())
    stop_event.set()
    poller.join(timeout=2.0)
    return sorted(results, key=lambda r: (r["arrival_offset_s"], r["request_id"]))


def percentile(values: list[float], p: float):
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    return clean[lo] if lo == hi else clean[lo] * (hi - rank) + clean[hi] * (rank - lo)


def stats(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "sd": None, "p95": None}
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "p95": percentile(clean, 0.95),
    }


def bootstrap_ci(values: list[float], samples: int = 4000):
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None, None
    rng = random.Random(SEED)
    means = []
    for _ in range(samples):
        means.append(statistics.mean(rng.choice(clean) for _ in clean))
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize_trace(trace_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "scheduled" in r]
    by = {}
    for treatment in SERVER_CONFIGS:
        trs = [r for r in rows if r.get("treatment") == treatment]
        prompt_steps = [r.get("prefill_tokens", 0) for r in trs if r.get("prefill_tokens", 0) > 0]
        total_steps = [r.get("total_num_scheduled_tokens", 0) or 0 for r in trs if (r.get("total_num_scheduled_tokens", 0) or 0) > 0]
        decode = sum(r.get("decode_tokens", 0) for r in trs)
        prefill = sum(r.get("prefill_tokens", 0) for r in trs)
        by[treatment] = {
            "scheduled_steps": len(trs),
            "mixed_prefill_decode_steps": sum(1 for r in trs if r.get("has_prefill_and_decode")),
            "mixed_step_fraction": sum(1 for r in trs if r.get("has_prefill_and_decode")) / len(trs) if trs else None,
            "partial_prefill_items": sum(r.get("partial_prefill_items", 0) for r in trs),
            "prompt_tokens_per_prefill_step": stats(prompt_steps),
            "total_tokens_per_step": stats(total_steps),
            "decode_token_share": decode / (decode + prefill) if decode + prefill > 0 else None,
            "max_waiting_after_schedule": max([r.get("waiting_len_after_schedule", 0) for r in trs], default=0),
            "max_running_after_schedule": max([r.get("running_len_after_schedule", 0) for r in trs], default=0),
        }
    return by


def analyze(requests: list[dict[str, Any]], metrics: list[dict[str, Any]], trace_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    summary_rows = []
    rep_metrics = {}
    for regime in REGIMES:
        for treatment in SERVER_CONFIGS:
            rows = [r for r in requests if r["regime"] == regime and r["treatment"] == treatment]
            for klass in ["hog", "late", "all"]:
                subset = rows if klass == "all" else [r for r in rows if r["workload_class"] == klass]
                good = [r for r in subset if r["success"]]
                slo = [r["slo_met"] for r in good if r.get("slo_met") is not None]
                ttft = stats([r["ttft_s"] for r in good])
                e2e = stats([r["e2e_s"] for r in good])
                summary_rows.append(
                    {
                        "regime": regime,
                        "treatment": treatment,
                        "class": klass,
                        "n_requests": len(subset),
                        "n_success": len(good),
                        "ttft_mean_s": ttft["mean"],
                        "ttft_median_s": ttft["median"],
                        "ttft_sd_s": ttft["sd"],
                        "ttft_p95_s": ttft["p95"],
                        "e2e_mean_s": e2e["mean"],
                        "e2e_median_s": e2e["median"],
                        "e2e_sd_s": e2e["sd"],
                        "e2e_p95_s": e2e["p95"],
                        "slo_attainment": sum(bool(x) for x in slo) / len(slo) if slo else None,
                    }
                )
            for rep in range(5):
                rr = [r for r in rows if r["repetition"] == rep and r["success"]]
                late = [r for r in rr if r["workload_class"] == "late"]
                hog = [r for r in rr if r["workload_class"] == "hog"]
                span = max(r["completion_wall_time"] for r in rr) - min(r["send_wall_time"] for r in rr)
                rep_metrics[(regime, treatment, rep)] = {
                    "late_ttft_mean_s": statistics.mean(r["ttft_s"] for r in late),
                    "hog_e2e_mean_s": statistics.mean(r["e2e_s"] for r in hog),
                    "throughput_rps": len(rr) / span,
                }
    comparisons = {}
    for regime in REGIMES:
        late_diffs = []
        hog_diffs = []
        thr_diffs = []
        for rep in range(5):
            a = rep_metrics[(regime, "T512", rep)]
            b = rep_metrics[(regime, "T4096", rep)]
            late_diffs.append(b["late_ttft_mean_s"] - a["late_ttft_mean_s"])
            hog_diffs.append(b["hog_e2e_mean_s"] - a["hog_e2e_mean_s"])
            thr_diffs.append(b["throughput_rps"] - a["throughput_rps"])
        comparisons[regime] = {
            "late_ttft_T4096_minus_T512_s_by_rep": late_diffs,
            "late_ttft_T4096_minus_T512_s_mean": statistics.mean(late_diffs),
            "late_ttft_T4096_minus_T512_s_ci95": bootstrap_ci(late_diffs),
            "late_ttft_T4096_lower_reps": sum(1 for x in late_diffs if x < 0),
            "hog_e2e_T4096_minus_T512_s_by_rep": hog_diffs,
            "hog_e2e_T4096_minus_T512_s_mean": statistics.mean(hog_diffs),
            "hog_e2e_T4096_minus_T512_s_ci95": bootstrap_ci(hog_diffs),
            "hog_e2e_T4096_lower_reps": sum(1 for x in hog_diffs if x < 0),
            "throughput_T4096_minus_T512_rps_by_rep": thr_diffs,
            "throughput_T4096_minus_T512_rps_mean": statistics.mean(thr_diffs),
            "throughput_T4096_minus_T512_rps_ci95": bootstrap_ci(thr_diffs),
            "throughput_T4096_higher_reps": sum(1 for x in thr_diffs if x > 0),
        }
    max_metrics = {"waiting": 0.0, "running": 0.0, "kv_cache_usage": 0.0, "preemptions": 0.0}
    for row in metrics:
        for k, v in (row.get("metrics") or {}).items():
            if k in max_metrics:
                max_metrics[k] = max(max_metrics[k], v)
    stable_latency = any(
        abs(c["late_ttft_T4096_minus_T512_s_mean"]) > 0.01
        and (c["late_ttft_T4096_lower_reps"] in {0, 1, 4, 5})
        for c in comparisons.values()
    )
    stable_mech = (
        trace_summary["T512"]["partial_prefill_items"] > trace_summary["T4096"]["partial_prefill_items"]
        and trace_summary["T512"]["prompt_tokens_per_prefill_step"]["mean"]
        < trace_summary["T4096"]["prompt_tokens_per_prefill_step"]["mean"]
    )
    tradeoff = any(
        (c["late_ttft_T4096_minus_T512_s_mean"] < -0.01 and c["hog_e2e_T4096_minus_T512_s_mean"] > 0.01)
        or (c["late_ttft_T4096_minus_T512_s_mean"] > 0.01 and c["hog_e2e_T4096_minus_T512_s_mean"] < -0.01)
        for c in comparisons.values()
    )
    if stable_latency and stable_mech and tradeoff:
        verdict = "NATIVE_VLLM_BUDGET_EFFECT_STRONG"
    elif stable_latency or stable_mech:
        verdict = "NATIVE_VLLM_BUDGET_EFFECT_PARTIAL"
    else:
        verdict = "NATIVE_VLLM_BUDGET_EFFECT_NULL"
    statistical = {
        "comparisons": comparisons,
        "max_server_metrics": max_metrics,
        "verdict_criteria": {
            "stable_latency": stable_latency,
            "stable_mechanism_separation": stable_mech,
            "meaningful_tradeoff": tradeoff,
        },
        "verdict": verdict,
    }
    mechanism = {"trace_summary_by_treatment": trace_summary, "interpretation": "pending"}
    return summary_rows, mechanism, statistical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "server_logs").mkdir(exist_ok=True)
    patch_dir = out / "instrumentation"
    context_path = out / "trace_context.json"
    trace_path = out / "scheduler_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    write_sitecustomize(trace_path, context_path, patch_dir)

    runner = load_validation_runner()
    manifest = json.loads(PREVIOUS_MANIFEST.read_text())
    selected = {r: manifest["requests_by_regime"][r] for r in REGIMES}
    (out / "request_manifest_reference.json").write_text(
        stable_json(
            {
                "source_manifest": str(PREVIOUS_MANIFEST),
                "source_manifest_sha256": PREVIOUS_MANIFEST_HASH,
                "selected_regimes": REGIMES,
                "reuse_contract": "exact prompts, tokenized lengths, output caps, arrivals, request classes reused",
                "requests_by_regime": selected,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    server_configs = {
        t: {
            "model_path": str(MODEL_PATH),
            "served_model_name": MODEL_NAME,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": cfg["max_num_batched_tokens"],
            "max_model_len": 4096,
            "max_num_seqs": 4,
            "gpu_memory_utilization": 0.35,
            "prefix_caching": False,
            "enforce_eager": True,
            "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        }
        for t, cfg in SERVER_CONFIGS.items()
    }
    (out / "server_configs.json").write_text(stable_json(server_configs) + "\n", encoding="utf-8")

    blocks = [
        {"block": 0, "treatment": "T512", "repetitions": [0, 1]},
        {"block": 1, "treatment": "T4096", "repetitions": [0, 1]},
        {"block": 2, "treatment": "T4096", "repetitions": [2, 3, 4]},
        {"block": 3, "treatment": "T512", "repetitions": [2, 3, 4]},
    ]
    rng = random.Random(SEED)
    run_order = []
    for block in blocks:
        pairs = [(r, rep) for rep in block["repetitions"] for r in REGIMES]
        rng.shuffle(pairs)
        run_order.append({**block, "regime_repetition_order": [{"regime": r, "repetition": rep} for r, rep in pairs]})
    (out / "run_order.json").write_text(stable_json({"seed": SEED, "blocks": run_order}) + "\n", encoding="utf-8")

    all_requests = []
    all_metrics = []
    warmups = []
    server_events = []
    for block in run_order:
        treatment = block["treatment"]
        port = SERVER_CONFIGS[treatment]["port"]
        proc = launch_server(treatment, trace_path, context_path, patch_dir, out / "server_logs" / f"{block['block']:02d}_{treatment.lower()}.log")
        started = time.time()
        try:
            wait_health(port)
            healthy = time.time()
            context_path.write_text(json.dumps({"treatment": treatment, "phase": "warmup", "block": block["block"]}), encoding="utf-8")
            warm = runner.warmup(port)
            warmups.append({"block": block["block"], "treatment": treatment, "health_wait_s": healthy - started, "warmup_requests": warm})
            for rr in block["regime_repetition_order"]:
                all_requests.extend(run_regime(runner, treatment, port, rr["regime"], rr["repetition"], selected[rr["regime"]], context_path, all_metrics))
            server_events.append({"block": block["block"], "treatment": treatment, "log": str(out / "server_logs" / f"{block['block']:02d}_{treatment.lower()}.log")})
        finally:
            stop_server(proc)
            time.sleep(3.0)

    with (out / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_requests:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    with (out / "server_metrics.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_metrics:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    trace_summary = summarize_trace(trace_path)
    summary_rows, mechanism, statistical = analyze(all_requests, all_metrics, trace_summary)
    mechanism["interpretation"] = (
        "Both treatments use native vLLM chunked prefill; T512 produces smaller prompt-token steps "
        "and more partial prefill chunks than T4096. The latency comparison tests native budget semantics, not simulator Family-B."
    )
    with (out / "summary_by_regime.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(summary_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    (out / "mechanism_summary.json").write_text(stable_json(mechanism) + "\n", encoding="utf-8")
    (out / "statistical_summary.json").write_text(stable_json(statistical) + "\n", encoding="utf-8")

    failures = [r for r in all_requests if not r["success"]]
    run_integrity = {
        "expected_regime_runs": 20,
        "observed_regime_runs": len({(r["treatment"], r["regime"], r["repetition"]) for r in all_requests}),
        "request_failures": len(failures),
        "failure_records": failures,
        "warmups": warmups,
        "server_events": server_events,
        "integrity_pass": len(failures) == 0 and len({(r["treatment"], r["regime"], r["repetition"]) for r in all_requests}) == 20,
    }
    (out / "run_integrity.json").write_text(stable_json(run_integrity) + "\n", encoding="utf-8")
    decision = {
        "experiment": "native_vllm_chunk_budget_semantics_probe_v1",
        "integrity_pass": run_integrity["integrity_pass"],
        "verdict": statistical["verdict"] if run_integrity["integrity_pass"] else "NATIVE_VLLM_BUDGET_EFFECT_NULL",
        "treatments": ["T512", "T4096"],
        "selected_regimes": REGIMES,
        "source_manifest_sha256": PREVIOUS_MANIFEST_HASH,
        "no_family_b_validation_claim": True,
        "next_task": "stop_or_publication_integration_decision; do not automatically launch more real-vLLM experiments",
    }
    (out / "decision.json").write_text(stable_json(decision) + "\n", encoding="utf-8")
    print(stable_json(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
