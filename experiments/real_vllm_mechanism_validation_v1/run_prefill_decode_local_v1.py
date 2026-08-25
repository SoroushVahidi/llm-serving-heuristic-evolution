#!/usr/bin/env python3
"""Run local vLLM prefill/decode contention validation.

This script is intentionally scoped to the full-prefill vs chunked-prefill
mechanism comparison for real_vllm_mechanism_validation_v1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/soroush/llm-serving-heuristic-evolution")
MODEL_PATH = Path(
    "/home/soroush/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
VENV_BIN = Path("/home/soroush/.venvs/vllm_real_validation_v1/bin")
OUT_DIR = REPO_ROOT / "experiments/real_vllm_mechanism_validation_v1/prefill_decode_local_v1"
SEED = 20260824
MODEL_NAME = "qwen05b-local"
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 4


SERVER_CONFIGS: dict[str, dict[str, Any]] = {
    "FULL": {
        "port": 8061,
        "chunked_prefill": False,
        "max_num_batched_tokens": 4096,
        "extra_flags": ["--no-enable-chunked-prefill"],
    },
    "CHUNKED": {
        "port": 8062,
        "chunked_prefill": True,
        "max_num_batched_tokens": 512,
        "extra_flags": ["--enable-chunked-prefill"],
    },
}


REGIMES: dict[str, dict[str, Any]] = {
    "hog_tight_low_late": {
        "slo_focus": "hog",
        "late_pressure": "low",
        "hog_count": 3,
        "late_count": 3,
        "expected": "FULL protects hog completion/E2E; CHUNKED may improve late TTFT.",
    },
    "late_tight_low_late": {
        "slo_focus": "late",
        "late_pressure": "low",
        "hog_count": 3,
        "late_count": 3,
        "expected": "CHUNKED protects late-request TTFT under modest late pressure.",
    },
    "hog_tight_high_late": {
        "slo_focus": "hog",
        "late_pressure": "high",
        "hog_count": 3,
        "late_count": 6,
        "expected": "FULL preserves hog completion/E2E; margin may shrink under high late pressure.",
    },
    "late_tight_high_late": {
        "slo_focus": "late",
        "late_pressure": "high",
        "hog_count": 3,
        "late_count": 6,
        "expected": "CHUNKED protects late-request TTFT under high late pressure.",
    },
}


GATE_DEFINITIONS = {
    "late_effect": (
        "For late_tight regimes, CHUNKED late-class mean TTFT must be at least "
        "5% lower than FULL with sign consistency >=4/5 repetitions."
    ),
    "hog_effect": (
        "For hog_tight regimes, FULL hog-class mean E2E must be at least 5% "
        "lower than CHUNKED with sign consistency >=4/5 repetitions."
    ),
    "PREFILL_REAL_VALIDATION_STRONG": (
        "late_effect and hog_effect both hold, giving a stable interpretable "
        "tradeoff/winner reversal across the four regimes."
    ),
    "PREFILL_REAL_VALIDATION_PARTIAL": (
        "exactly one of late_effect or hog_effect holds, or one stable expected "
        "effect is present but broader reversal is incomplete/noisy."
    ),
    "PREFILL_REAL_VALIDATION_NO_GO": (
        "no stable expected effect, opposite effect without systems explanation, "
        "or instrumentation/workload inadequacy."
    ),
}


METRIC_PATTERNS = {
    "waiting": re.compile(r"^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.eE+-]+)$"),
    "running": re.compile(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+-]+)$"),
    "kv_cache_usage": re.compile(r"^vllm:kv_cache_usage_perc\{[^}]*\}\s+([0-9.eE+-]+)$"),
    "preemptions": re.compile(r"^vllm:num_preemptions_total\{[^}]*\}\s+([0-9.eE+-]+)$"),
    "prompt_tokens_total": re.compile(r"^vllm:prompt_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)$"),
    "generation_tokens_total": re.compile(r"^vllm:generation_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)$"),
}


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, separators=(",", ": "))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def post_json(url: str, payload: dict[str, Any], timeout: float = 120.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


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


def launch_server(treatment: str, log_path: Path) -> subprocess.Popen[str]:
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
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(cfg["max_num_batched_tokens"]),
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--enforce-eager",
        *cfg["extra_flags"],
    ]
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
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


def load_tokenizer():
    sys.path.insert(0, str(VENV_BIN.parent / "lib/python3.12/site-packages"))
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)


def make_prompt(tokenizer: Any, target_tokens: int, seed: int, kind: str) -> tuple[str, int]:
    randomizer = random.Random(seed)
    topics = [
        "scheduler",
        "prefill",
        "decode",
        "latency",
        "queue",
        "service",
        "deadline",
        "request",
        "throughput",
        "cache",
    ]
    parts = []
    while True:
        word = randomizer.choice(topics)
        parts.append(
            f"{kind} workload token block {word} {len(parts)}. "
            "Explain the mechanism briefly and continue. "
        )
        text = "".join(parts)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= target_tokens + 32:
            clipped = tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)
            actual = len(tokenizer.encode(clipped, add_special_tokens=False))
            while actual > target_tokens:
                clipped = clipped.rsplit(" ", 1)[0]
                actual = len(tokenizer.encode(clipped, add_special_tokens=False))
            return clipped, actual


def build_manifest(tokenizer: Any) -> dict[str, Any]:
    requests_by_regime: dict[str, list[dict[str, Any]]] = {}
    for regime_id, regime in REGIMES.items():
        rows = []
        for i in range(regime["hog_count"]):
            prompt, actual = make_prompt(tokenizer, 3200 + 80 * i, SEED + 1000 + 17 * i, "hog")
            rows.append(
                {
                    "request_id": f"{regime_id}.hog.{i}",
                    "arrival_offset_s": 0.0,
                    "prompt_token_target": 3200 + 80 * i,
                    "prompt_construction_seed": SEED + 1000 + 17 * i,
                    "actual_input_tokens": actual,
                    "max_output_tokens": 128,
                    "workload_class": "hog",
                    "slo_metric": "e2e",
                    "slo_threshold_s": 3.0 if regime["slo_focus"] == "hog" else 12.0,
                    "prompt": prompt,
                }
            )
        for i in range(regime["late_count"]):
            prompt, actual = make_prompt(tokenizer, 96 + 8 * (i % 3), SEED + 2000 + 19 * i, "late")
            rows.append(
                {
                    "request_id": f"{regime_id}.late.{i}",
                    "arrival_offset_s": 0.05 + 0.01 * i,
                    "prompt_token_target": 96 + 8 * (i % 3),
                    "prompt_construction_seed": SEED + 2000 + 19 * i,
                    "actual_input_tokens": actual,
                    "max_output_tokens": 32,
                    "workload_class": "late",
                    "slo_metric": "ttft",
                    "slo_threshold_s": 0.75 if regime["slo_focus"] == "late" else 5.0,
                    "prompt": prompt,
                }
            )
        for row in rows:
            if row["actual_input_tokens"] + row["max_output_tokens"] > MAX_MODEL_LEN:
                raise ValueError(f"request exceeds max model len: {row['request_id']}")
        requests_by_regime[regime_id] = rows
    return {
        "experiment": "real_vllm_mechanism_validation_v1",
        "subrun": "prefill_decode_local_v1",
        "seed": SEED,
        "model_path": str(MODEL_PATH),
        "model_name": MODEL_NAME,
        "max_model_len": MAX_MODEL_LEN,
        "regimes": REGIMES,
        "requests_by_regime": requests_by_regime,
        "gate_definitions": GATE_DEFINITIONS,
    }


def warmup(port: int) -> list[dict[str, Any]]:
    rows = []
    warmups = [
        ("warmup.short", "Say hello in one short sentence.", 8),
        ("warmup.medium", " ".join(["This is a scheduler warmup prompt."] * 180), 16),
    ]
    for request_id, prompt, max_tokens in warmups:
        rows.append(send_streaming_request(port, request_id, prompt, max_tokens, {}, phase="warmup"))
    return rows


def send_streaming_request(
    port: int,
    request_id: str,
    prompt: str,
    max_tokens: int,
    metadata: dict[str, Any],
    phase: str = "measured",
) -> dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    wall_send = time.time()
    mono_send = time.monotonic()
    first_token_wall = None
    completion_wall = None
    output_text = []
    usage = None
    error = None
    try:
        with post_json(f"http://127.0.0.1:{port}/v1/completions", payload, timeout=180.0) as resp:
            for raw in resp:
                now = time.time()
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if chunk == "[DONE]":
                    completion_wall = now
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if data.get("usage"):
                    usage = data["usage"]
                choices = data.get("choices") or []
                text = ""
                if choices:
                    text = choices[0].get("text") or ""
                if text:
                    output_text.append(text)
                    if first_token_wall is None:
                        first_token_wall = now
        completion_wall = completion_wall or time.time()
    except Exception as exc:  # noqa: BLE001
        completion_wall = time.time()
        error = repr(exc)
    ttft = None if first_token_wall is None else first_token_wall - wall_send
    e2e = completion_wall - wall_send
    row = {
        **metadata,
        "phase": phase,
        "request_id": request_id,
        "send_wall_time": wall_send,
        "send_monotonic_time": mono_send,
        "first_token_wall_time": first_token_wall,
        "completion_wall_time": completion_wall,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "input_tokens": usage.get("prompt_tokens") if usage else metadata.get("actual_input_tokens"),
        "output_tokens": usage.get("completion_tokens") if usage else None,
        "success": error is None and first_token_wall is not None,
        "error": error,
    }
    slo_metric = metadata.get("slo_metric")
    threshold = metadata.get("slo_threshold_s")
    if slo_metric == "ttft" and threshold is not None and ttft is not None:
        row["slo_met"] = ttft <= threshold
    elif slo_metric == "e2e" and threshold is not None:
        row["slo_met"] = e2e <= threshold
    else:
        row["slo_met"] = None
    return row


def parse_metrics(text: str) -> dict[str, float]:
    values = {key: [] for key in METRIC_PATTERNS}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, pattern in METRIC_PATTERNS.items():
            match = pattern.match(line)
            if match:
                values[key].append(float(match.group(1)))
    parsed = {}
    for key, vals in values.items():
        if vals:
            parsed[key] = max(vals) if key in {"waiting", "running", "kv_cache_usage"} else sum(vals)
    return parsed


def poll_metrics(
    port: int,
    stop_event: threading.Event,
    context: dict[str, Any],
    rows: list[dict[str, Any]],
    interval_s: float = 0.05,
) -> None:
    start = time.monotonic()
    while not stop_event.is_set():
        try:
            text = get_url(f"http://127.0.0.1:{port}/metrics", timeout=2.0)
            rows.append(
                {
                    **context,
                    "wall_time": time.time(),
                    "elapsed_s": time.monotonic() - start,
                    "metrics": parse_metrics(text),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({**context, "wall_time": time.time(), "elapsed_s": time.monotonic() - start, "error": repr(exc)})
        time.sleep(interval_s)


def run_regime(
    treatment: str,
    port: int,
    regime_id: str,
    repetition: int,
    manifest: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    request_rows = manifest["requests_by_regime"][regime_id]
    stop_event = threading.Event()
    poller = threading.Thread(
        target=poll_metrics,
        args=(
            port,
            stop_event,
            {
                "treatment": treatment,
                "regime": regime_id,
                "repetition": repetition,
            },
            metrics_rows,
        ),
        daemon=True,
    )
    poller.start()
    t0 = time.monotonic()
    results = []

    def task(row: dict[str, Any]) -> dict[str, Any]:
        delay = t0 + float(row["arrival_offset_s"]) - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        metadata = {
            "treatment": treatment,
            "regime": regime_id,
            "repetition": repetition,
            "workload_class": row["workload_class"],
            "arrival_offset_s": row["arrival_offset_s"],
            "prompt_token_target": row["prompt_token_target"],
            "actual_input_tokens": row["actual_input_tokens"],
            "max_output_tokens": row["max_output_tokens"],
            "slo_metric": row["slo_metric"],
            "slo_threshold_s": row["slo_threshold_s"],
        }
        return send_streaming_request(port, row["request_id"], row["prompt"], row["max_output_tokens"], metadata)

    with ThreadPoolExecutor(max_workers=len(request_rows)) as pool:
        futures = [pool.submit(task, row) for row in request_rows]
        for future in as_completed(futures):
            results.append(future.result())
    stop_event.set()
    poller.join(timeout=2.0)
    results.sort(key=lambda row: (row["arrival_offset_s"], row["request_id"]))
    return results


def percentile(values: list[float], p: float) -> float | None:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - rank) + clean[hi] * (rank - lo)


def summarize_values(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "std": None, "p50": None, "p95": None}
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
    }


def bootstrap_ci(values: list[float], rng_seed: int = SEED, samples: int = 4000) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None, None
    rng = random.Random(rng_seed)
    means = []
    for _ in range(samples):
        draw = [rng.choice(clean) for _ in clean]
        means.append(statistics.mean(draw))
    return percentile(means, 0.025), percentile(means, 0.975)


def analyze(request_rows: list[dict[str, Any]], metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime_treatment: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in request_rows:
        by_regime_treatment.setdefault((row["regime"], row["treatment"]), []).append(row)

    summary_rows = []
    repetition_metrics: dict[tuple[str, str, int], dict[str, float]] = {}
    for (regime, treatment), rows in sorted(by_regime_treatment.items()):
        for klass in ["hog", "late", "all"]:
            subset = rows if klass == "all" else [r for r in rows if r["workload_class"] == klass]
            ttft = [r["ttft_s"] for r in subset if r["success"]]
            e2e = [r["e2e_s"] for r in subset if r["success"]]
            slo = [r["slo_met"] for r in subset if r.get("slo_met") is not None]
            ttft_s = summarize_values(ttft)
            e2e_s = summarize_values(e2e)
            summary_rows.append(
                {
                    "regime": regime,
                    "treatment": treatment,
                    "class": klass,
                    "n_requests": len(subset),
                    "n_success": sum(1 for r in subset if r["success"]),
                    "ttft_mean_s": ttft_s["mean"],
                    "ttft_median_s": ttft_s["median"],
                    "ttft_std_s": ttft_s["std"],
                    "ttft_p95_s": ttft_s["p95"],
                    "e2e_mean_s": e2e_s["mean"],
                    "e2e_median_s": e2e_s["median"],
                    "e2e_std_s": e2e_s["std"],
                    "e2e_p95_s": e2e_s["p95"],
                    "slo_attainment": (sum(bool(x) for x in slo) / len(slo)) if slo else None,
                }
            )

    for treatment in SERVER_CONFIGS:
        for regime in REGIMES:
            for rep in range(5):
                rows = [r for r in request_rows if r["treatment"] == treatment and r["regime"] == regime and r["repetition"] == rep]
                success = [r for r in rows if r["success"]]
                hog = [r for r in success if r["workload_class"] == "hog"]
                late = [r for r in success if r["workload_class"] == "late"]
                if success:
                    span = max(r["completion_wall_time"] for r in success) - min(r["send_wall_time"] for r in success)
                else:
                    span = float("nan")
                repetition_metrics[(regime, treatment, rep)] = {
                    "late_mean_ttft": statistics.mean([r["ttft_s"] for r in late]) if late else float("nan"),
                    "hog_mean_e2e": statistics.mean([r["e2e_s"] for r in hog]) if hog else float("nan"),
                    "throughput_rps": len(success) / span if span and math.isfinite(span) and span > 0 else float("nan"),
                    "success_count": len(success),
                }

    comparisons: dict[str, Any] = {}
    late_effects = []
    hog_effects = []
    for regime in REGIMES:
        late_diffs = []
        late_rel = []
        hog_diffs = []
        hog_rel = []
        throughput_diffs = []
        for rep in range(5):
            full = repetition_metrics[(regime, "FULL", rep)]
            chunked = repetition_metrics[(regime, "CHUNKED", rep)]
            late_diff = chunked["late_mean_ttft"] - full["late_mean_ttft"]
            hog_diff = chunked["hog_mean_e2e"] - full["hog_mean_e2e"]
            throughput_diff = chunked["throughput_rps"] - full["throughput_rps"]
            late_diffs.append(late_diff)
            hog_diffs.append(hog_diff)
            throughput_diffs.append(throughput_diff)
            if full["late_mean_ttft"] and math.isfinite(full["late_mean_ttft"]):
                late_rel.append((full["late_mean_ttft"] - chunked["late_mean_ttft"]) / full["late_mean_ttft"])
            if chunked["hog_mean_e2e"] and math.isfinite(chunked["hog_mean_e2e"]):
                hog_rel.append((chunked["hog_mean_e2e"] - full["hog_mean_e2e"]) / chunked["hog_mean_e2e"])
        comparisons[regime] = {
            "late_ttft_chunked_minus_full_s_by_rep": late_diffs,
            "late_ttft_mean_chunked_minus_full_s": statistics.mean(late_diffs),
            "late_ttft_improvement_fraction_chunked_vs_full": statistics.mean(late_rel),
            "late_ttft_diff_bootstrap_ci_s": bootstrap_ci(late_diffs),
            "late_ttft_sign_consistency_chunked_lower": sum(1 for x in late_diffs if x < 0),
            "hog_e2e_chunked_minus_full_s_by_rep": hog_diffs,
            "hog_e2e_mean_chunked_minus_full_s": statistics.mean(hog_diffs),
            "hog_e2e_improvement_fraction_full_vs_chunked": statistics.mean(hog_rel),
            "hog_e2e_diff_bootstrap_ci_s": bootstrap_ci(hog_diffs),
            "hog_e2e_sign_consistency_full_lower": sum(1 for x in hog_diffs if x > 0),
            "throughput_chunked_minus_full_rps_by_rep": throughput_diffs,
            "throughput_mean_chunked_minus_full_rps": statistics.mean(throughput_diffs),
        }
        if regime.startswith("late_tight"):
            late_effects.append(
                comparisons[regime]["late_ttft_improvement_fraction_chunked_vs_full"] >= 0.05
                and comparisons[regime]["late_ttft_sign_consistency_chunked_lower"] >= 4
            )
        if regime.startswith("hog_tight"):
            hog_effects.append(
                comparisons[regime]["hog_e2e_improvement_fraction_full_vs_chunked"] >= 0.05
                and comparisons[regime]["hog_e2e_sign_consistency_full_lower"] >= 4
            )

    max_metrics = {"waiting": 0.0, "running": 0.0, "kv_cache_usage": 0.0, "preemptions": 0.0}
    for row in metrics_rows:
        m = row.get("metrics") or {}
        for key in max_metrics:
            if key in m:
                max_metrics[key] = max(max_metrics[key], m[key])

    late_effect = all(late_effects) if late_effects else False
    hog_effect = all(hog_effects) if hog_effects else False
    if late_effect and hog_effect:
        verdict = "PREFILL_REAL_VALIDATION_STRONG"
    elif late_effect or hog_effect:
        verdict = "PREFILL_REAL_VALIDATION_PARTIAL"
    else:
        verdict = "PREFILL_REAL_VALIDATION_NO_GO"

    alignment = []
    for regime in REGIMES:
        c = comparisons[regime]
        if regime.startswith("late_tight"):
            expected = "CHUNKED lowers late TTFT"
            observed = "CHUNKED lower" if c["late_ttft_mean_chunked_minus_full_s"] < 0 else "FULL lower/equal"
            agree = c["late_ttft_mean_chunked_minus_full_s"] < 0
        else:
            expected = "FULL lowers hog E2E/completion"
            observed = "FULL lower" if c["hog_e2e_mean_chunked_minus_full_s"] > 0 else "CHUNKED lower/equal"
            agree = c["hog_e2e_mean_chunked_minus_full_s"] > 0
        alignment.append(
            {
                "simulator_regime_mechanism": regime,
                "real_vllm_analogue": regime,
                "expected_direction": expected,
                "observed_direction": observed,
                "agreement": agree,
            }
        )

    return {
        "summary_rows": summary_rows,
        "repetition_metrics": {f"{k[0]}::{k[1]}::{k[2]}": v for k, v in repetition_metrics.items()},
        "comparisons": comparisons,
        "max_telemetry": max_metrics,
        "gate_definitions": GATE_DEFINITIONS,
        "late_effect": late_effect,
        "hog_effect": hog_effect,
        "verdict": verdict,
        "simulator_alignment": alignment,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "server_logs").mkdir(exist_ok=True)

    tokenizer = load_tokenizer()
    manifest = build_manifest(tokenizer)
    manifest_text = stable_json(manifest)
    (out_dir / "workload_manifest.json").write_text(manifest_text + "\n", encoding="utf-8")
    manifest_hash = sha256_text(manifest_text)
    (out_dir / "workload_manifest.sha256").write_text(manifest_hash + "  workload_manifest.json\n", encoding="utf-8")

    server_configs = {}
    for treatment, cfg in SERVER_CONFIGS.items():
        server_configs[treatment] = {
            "model_path": str(MODEL_PATH),
            "served_model_name": MODEL_NAME,
            "host": "127.0.0.1",
            "port": cfg["port"],
            "gpu_memory_utilization": 0.35,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": cfg["max_num_batched_tokens"],
            "block_size": 16,
            "enable_prefix_caching": False,
            "enforce_eager": True,
            "chunked_prefill": cfg["chunked_prefill"],
            "extra_flags": cfg["extra_flags"],
            "environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
        }
    (out_dir / "server_configs.json").write_text(stable_json(server_configs) + "\n", encoding="utf-8")

    run_blocks = [
        {"block": 0, "treatment": "FULL", "repetitions": [0, 1]},
        {"block": 1, "treatment": "CHUNKED", "repetitions": [0, 1]},
        {"block": 2, "treatment": "CHUNKED", "repetitions": [2, 3, 4]},
        {"block": 3, "treatment": "FULL", "repetitions": [2, 3, 4]},
    ]
    rng = random.Random(SEED)
    run_order = []
    for block in run_blocks:
        pairs = [(regime, rep) for rep in block["repetitions"] for regime in REGIMES]
        rng.shuffle(pairs)
        run_order.append({**block, "regime_repetition_order": [{"regime": r, "repetition": rep} for r, rep in pairs]})
    (out_dir / "run_order.json").write_text(stable_json({"seed": SEED, "blocks": run_order}) + "\n", encoding="utf-8")

    all_requests: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    warmup_records = []
    server_events = []

    for block in run_order:
        treatment = block["treatment"]
        port = SERVER_CONFIGS[treatment]["port"]
        log_path = out_dir / "server_logs" / f"{block['block']:02d}_{treatment.lower()}.log"
        proc = launch_server(treatment, log_path)
        started = time.time()
        try:
            wait_health(port)
            healthy = time.time()
            warm = warmup(port)
            warmup_done = time.time()
            warmup_records.append(
                {
                    "block": block["block"],
                    "treatment": treatment,
                    "server_start_wall_time": started,
                    "server_health_wall_time": healthy,
                    "warmup_done_wall_time": warmup_done,
                    "health_wait_s": healthy - started,
                    "warmup_duration_s": warmup_done - healthy,
                    "warmup_requests": warm,
                }
            )
            for rr in block["regime_repetition_order"]:
                rows = run_regime(treatment, port, rr["regime"], rr["repetition"], manifest, all_metrics)
                all_requests.extend(rows)
            server_events.append({"block": block["block"], "treatment": treatment, "returncode": proc.poll(), "log": str(log_path)})
        finally:
            stop_server(proc)
            time.sleep(3.0)

    write_jsonl(out_dir / "requests.jsonl", all_requests)
    write_jsonl(out_dir / "server_metrics.jsonl", all_metrics)

    result = analyze(all_requests, all_metrics)
    with (out_dir / "summary_by_regime.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "regime",
            "treatment",
            "class",
            "n_requests",
            "n_success",
            "ttft_mean_s",
            "ttft_median_s",
            "ttft_std_s",
            "ttft_p95_s",
            "e2e_mean_s",
            "e2e_median_s",
            "e2e_std_s",
            "e2e_p95_s",
            "slo_attainment",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["summary_rows"])

    request_failures = [r for r in all_requests if not r["success"]]
    prompt_mismatch = [
        r
        for regime_rows in manifest["requests_by_regime"].values()
        for r in regime_rows
        if abs(r["actual_input_tokens"] - r["prompt_token_target"]) > 16
    ]
    run_integrity = {
        "workload_manifest_sha256": manifest_hash,
        "expected_measured_regime_runs": 40,
        "observed_measured_regime_runs": len(
            {(r["treatment"], r["regime"], r["repetition"]) for r in all_requests}
        ),
        "expected_treatments": ["FULL", "CHUNKED"],
        "expected_repetitions_per_treatment_regime": 5,
        "request_failures": len(request_failures),
        "request_failure_records": request_failures,
        "prompt_generation_mismatch_records": prompt_mismatch,
        "warmup_records": warmup_records,
        "server_events": server_events,
        "integrity_pass": len(request_failures) == 0
        and len(prompt_mismatch) == 0
        and len({(r["treatment"], r["regime"], r["repetition"]) for r in all_requests}) == 40,
    }
    (out_dir / "run_integrity.json").write_text(stable_json(run_integrity) + "\n", encoding="utf-8")
    (out_dir / "statistical_summary.json").write_text(
        stable_json({k: v for k, v in result.items() if k not in {"summary_rows", "simulator_alignment"}}) + "\n",
        encoding="utf-8",
    )
    (out_dir / "simulator_alignment.json").write_text(stable_json(result["simulator_alignment"]) + "\n", encoding="utf-8")
    decision = {
        "experiment": "real_vllm_mechanism_validation_v1",
        "subrun": "prefill_decode_local_v1",
        "workload_manifest_sha256": manifest_hash,
        "integrity_pass": run_integrity["integrity_pass"],
        "late_effect": result["late_effect"],
        "hog_effect": result["hog_effect"],
        "verdict": result["verdict"] if run_integrity["integrity_pass"] else "PREFILL_REAL_VALIDATION_NO_GO",
        "note": "TRAIN/TEST/FINAL/DEV-free real-system mechanism validation; no Wulver/GPU cluster/API use.",
    }
    (out_dir / "decision.json").write_text(stable_json(decision) + "\n", encoding="utf-8")

    print(stable_json(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
