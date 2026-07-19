#!/usr/bin/env python3
"""GPU-backed external-validity audit for monolithic faithful baselines.

This is intentionally small-scale. It queries an already-running vLLM server
for controlled request patterns, then runs matched simulator traces for
``vllm_faithful`` and ``sarathi_faithful``. It does not train a selector and
does not launch a GPU sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import GPUConfig, Request  # noqa: E402
from llmserveopt.evaluation.run_policy import run_policy  # noqa: E402
from llmserveopt.policies.external_baselines_registry import make_external_baseline  # noqa: E402
from llmserveopt.real_llm import calibration_common as cc  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.workloads.trace_io_extended import load_extended_jsonl  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SERVER_URL = "http://127.0.0.1:8001"
DEFAULT_VLLM_EXECUTABLE = "/home/soroush/.venvs/vllm_baseline_pilot/bin/vllm"


@dataclass(frozen=True)
class PlannedRequest:
    scenario_name: str
    request_id: int
    arrival_time_s: float
    prompt_bucket: str
    target_output_tokens: int
    intended_prompt_tokens: int
    prompt_text: str = field(repr=False)
    source_trace: str = "synthetic"


@dataclass(frozen=True)
class RuntimeScenario:
    name: str
    description: str
    requests: list[PlannedRequest]
    max_client_concurrency: int
    scenario_family: str


@dataclass
class RuntimeResult:
    scenario_name: str
    request_id: int
    status: str
    scheduled_arrival_s: float
    client_queue_delay_s: float | None
    ttft_s: float | None
    latency_s: float | None
    prompt_tokens: float | None
    output_tokens: float | None
    finish_reason: str | None
    error: str | None = None


class MetricsPoller:
    def __init__(self, base_url: str, interval_s: float = 0.1) -> None:
        self.base_url = base_url.rstrip("/")
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = _urlopen_text(f"{self.base_url}/metrics", timeout_s=2.0)
                sample = _parse_vllm_metrics(text)
                sample["monotonic_s"] = time.monotonic()
                self.samples.append(sample)
            except Exception:
                pass
            time.sleep(self.interval_s)


def query_vllm_completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout_s: float) -> dict:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    first_token_t: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    chunks: list[str] = []

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if choices:
                text = choices[0].get("text") or ""
                if text and first_token_t is None:
                    first_token_t = time.monotonic()
                chunks.append(text)
                finish_reason = choices[0].get("finish_reason") or finish_reason
            usage = chunk.get("usage") or usage
    t1 = time.monotonic()
    return {
        "ttft_s": (first_token_t - t0) if first_token_t is not None else None,
        "latency_s": t1 - t0,
        "prompt_tokens": float(usage["prompt_tokens"]) if usage and usage.get("prompt_tokens") is not None else None,
        "output_tokens": float(usage["completion_tokens"]) if usage and usage.get("completion_tokens") is not None else None,
        "finish_reason": finish_reason,
        "text_preview": "".join(chunks)[:80],
    }


def build_scenarios(root: Path, *, include_real_traces: bool = True) -> list[RuntimeScenario]:
    scenarios = [
        _scenario("short_short", "short prompt / short output", ["short"], [16], 4, "spread", 2, "short_short"),
        _scenario("long_prompt_short_output", "long prompt / short output", ["long"], [16], 4, "spread", 2, "prefill_heavy"),
        _scenario("short_prompt_long_output", "short prompt / long output", ["short"], [96], 4, "burst", 4, "decode_heavy"),
        _scenario("long_prompt_long_output", "long prompt / long output", ["long"], [96], 4, "burst", 4, "long_long"),
        _scenario("mixed_prompt_lengths", "mixed prompt lengths", ["short", "medium", "long"], [32], 6, "spread", 4, "mixed_prompt"),
        _scenario("mixed_output_lengths", "mixed output lengths", ["medium"], [16, 64, 128], 6, "spread", 4, "mixed_output"),
        _scenario("bursty_arrivals", "bursty arrivals", ["medium"], [64], 8, "burst", 8, "bursty"),
        _scenario("high_concurrency", "high concurrency", ["short"], [64], 12, "burst", 12, "high_concurrency"),
        _scenario("prefill_heavy", "prefill-heavy long prompts", ["long"], [16], 6, "burst", 6, "prefill_heavy"),
        _scenario("decode_heavy", "decode-heavy long outputs", ["short"], [128], 6, "burst", 6, "decode_heavy"),
        _scenario("kv_pressure_long_context", "bounded KV-pressure attempt with long contexts", ["long"], [128], 12, "burst", 12, "kv_pressure"),
        _scenario("batch_turnover_mixed", "mixed prompt/output batch turnover", ["short", "medium", "long"], [16, 64, 128], 16, "burst", 16, "batch_turnover"),
    ]
    if include_real_traces:
        scenarios.extend(_real_trace_scenarios(root))
    return scenarios


def build_stress_scenarios(root: Path, *, include_real_traces: bool = True) -> list[RuntimeScenario]:
    scenarios = [
        _scenario("stress_high_concurrency_queue", "burst concurrency above max_num_seqs", ["short"], [128], 24, "burst", 24, "high_concurrency"),
        _scenario("stress_long_decode_kv", "long decode with overlapping sequences", ["short"], [512], 16, "burst", 16, "decode_heavy"),
        _scenario("stress_long_prefill", "long prompt prefill contention", ["long"], [96], 16, "burst", 16, "prefill_heavy"),
        _scenario("stress_kv_pressure", "long context plus long decode", ["long"], [768], 12, "burst", 12, "kv_pressure"),
        _mixed_contention_scenario(),
        _burst_recovery_scenario(),
    ]
    if include_real_traces:
        scenarios.extend(_stress_real_trace_scenarios(root))
    return scenarios


# ---------------------------------------------------------------------------
# Long-context ("xlong") KV-pressure scenarios.
#
# These are deliberately kept out of build_stress_scenarios() so the
# historical baseline/aggressive scenario suite (jobs 1111541, 1111545) is
# unchanged. They are also kept out of calibration_common.KNOWN_PROMPT_BUCKETS
# / PROMPT_BUCKET_TARGET_TOKENS: those are shared defaults consumed by other
# calibration scripts (e.g. run_cohere_api_calibration.py,
# run_vllm_serving_baseline_pilot.py) whose default --prompt-buckets sweep
# would silently pick up a much larger/costlier prompt size if a new bucket
# were added there. Instead this module builds long prompts locally with an
# explicit numeric token target, reusing the same deterministic, non-
# copyrighted synthetic-sentence approach as
# calibration_common.build_length_targeted_prompt.
# ---------------------------------------------------------------------------

_XLONG_SENTENCE_BANK = [
    "The request scheduler assigns incoming jobs to available GPU workers.",
    "Each worker maintains a key-value cache that grows during decoding.",
    "Batching multiple requests together can improve overall throughput.",
    "A scheduling policy decides the order in which requests are served.",
    "Prefill computes the initial hidden state for the full input prompt.",
    "Decoding produces one output token at a time using the cached state.",
    "Admission control can reject requests when the system is overloaded.",
    "Latency service-level objectives constrain how long a request may wait.",
    "Preemption allows a scheduler to pause one request to serve another.",
    "Throughput and tail latency are often in tension with each other.",
    "A simulator can replay traffic traces to compare scheduling policies.",
    "Token generation speed depends on batch size and sequence length.",
    "Streaming responses let a client observe output as it is produced.",
    "Fairness across tenants is one goal of a multi-tenant serving system.",
    "Cache eviction policies determine which sequences are dropped first.",
]


def _build_xlong_prompt(target_input_tokens: int, target_output_tokens: int, seed: int, variant_index: int) -> str:
    target_input_words = max(8, int(target_input_tokens * 0.75))
    words: list[str] = []
    idx = 0
    while len(words) < target_input_words:
        sentence = _XLONG_SENTENCE_BANK[idx % len(_XLONG_SENTENCE_BANK)]
        words.extend(sentence.split())
        idx += 1
    body = " ".join(words[:target_input_words])
    variant_tag = f"(request variant {seed}-xlong-{target_output_tokens}-{variant_index})"
    target_output_words = max(20, int(target_output_tokens * 0.75))
    instruction = (
        f"Using only the concepts mentioned in the text above, write a "
        f"plain-text explanation of approximately {target_output_words} "
        "words (not more than a few words short or over). Use complete "
        "sentences and do not use lists, markdown, or code blocks."
    )
    return f"{body} {variant_tag}\n\n{instruction}"


def _xlong_prompt_request(
    name: str, request_id: int, arrival_time_s: float, target_input_tokens: int,
    target_output_tokens: int, variant_index: int,
) -> PlannedRequest:
    prompt = _build_xlong_prompt(target_input_tokens, target_output_tokens, seed=20260719, variant_index=variant_index)
    return PlannedRequest(
        scenario_name=name,
        request_id=request_id,
        arrival_time_s=arrival_time_s,
        prompt_bucket="xlong",
        target_output_tokens=target_output_tokens,
        intended_prompt_tokens=cc.approx_token_count(prompt),
        prompt_text=prompt,
    )


def _xlong_context_burst_scenario() -> RuntimeScenario:
    """16 concurrent ~12k-token-prompt requests, all arriving at once.

    Sized against job 1111545's measured KV pool (226,960 tokens at
    gpu-memory-utilization=0.35): 16 requests x ~12,500 tokens (12,000
    prompt + up to 512 output) =~ 200,000 tokens, ~88% of that pool if all
    16 are simultaneously near their target length.
    """
    n = 16
    requests = [
        _xlong_prompt_request("stress_xlong_context_burst16", i, 0.0, 12000, 512, 3000 + i)
        for i in range(n)
    ]
    return RuntimeScenario(
        "stress_xlong_context_burst16",
        "16 concurrent ~12k-token-context requests arriving simultaneously",
        requests,
        n,
        "xlong_kv_pressure",
    )


def _xlong_context_saturate_scenario() -> RuntimeScenario:
    """24 requests total: 16 long-decode ~12k-token-prompt requests burst at
    t=0 (sized to exceed the ~18-19-request capacity implied by job
    1111545's pool once they are all admitted and decoding/growing), plus 8
    more ~12k-token requests arriving staggered at t=15..50s while the
    first 16 are still mid-decode. This keeps KV demand growing after the
    initial admission wave, which is the condition under which vLLM's
    recompute-based preemption is expected to trigger if the pool is
    insufficient, rather than only testing admission-time queueing.
    """
    requests: list[PlannedRequest] = []
    for i in range(16):
        requests.append(_xlong_prompt_request(
            "stress_xlong_context_saturate", i, 0.0, 12000, 1024, 4000 + i,
        ))
    late_arrivals = [15.0 + 5.0 * i for i in range(8)]
    for j, arrival in enumerate(late_arrivals):
        i = 16 + j
        requests.append(_xlong_prompt_request(
            "stress_xlong_context_saturate", i, arrival, 12000, 256, 4000 + i,
        ))
    return RuntimeScenario(
        "stress_xlong_context_saturate",
        "16 long-decode ~12k-context requests burst, plus 8 more ~12k-context "
        "requests arriving while the first 16 are still mid-decode",
        requests,
        24,
        "xlong_kv_pressure",
    )


def build_xlong_stress_scenarios(root: Path, *, include_real_traces: bool = False) -> list[RuntimeScenario]:
    """Long-context KV-pressure/preemption scenarios (job 1111541/1111545
    follow-up). Deliberately separate from build_stress_scenarios()."""
    del root, include_real_traces  # no real-trace variants for this phase
    return [
        _xlong_context_burst_scenario(),
        _xlong_context_saturate_scenario(),
    ]


def _scenario(
    name: str,
    description: str,
    prompt_buckets: list[str],
    output_targets: list[int],
    n: int,
    arrival_pattern: str,
    concurrency: int,
    family: str,
) -> RuntimeScenario:
    arrivals = _arrivals(n, arrival_pattern)
    requests = []
    for i in range(n):
        bucket = prompt_buckets[i % len(prompt_buckets)]
        target = output_targets[i % len(output_targets)]
        prompt = cc.build_length_targeted_prompt(bucket, target, seed=20260718, variant_index=i)
        requests.append(PlannedRequest(
            scenario_name=name,
            request_id=i,
            arrival_time_s=arrivals[i],
            prompt_bucket=bucket,
            target_output_tokens=target,
            intended_prompt_tokens=cc.approx_token_count(prompt),
            prompt_text=prompt,
        ))
    return RuntimeScenario(name, description, requests, concurrency, family)


def _arrivals(n: int, pattern: str) -> list[float]:
    if pattern == "burst":
        return [0.0] * n
    if pattern == "spread":
        return [0.15 * i for i in range(n)]
    raise ValueError(pattern)


def _mixed_contention_scenario() -> RuntimeScenario:
    requests: list[PlannedRequest] = []
    arrivals = [0.0] * 8 + [0.75] * 8
    for i in range(16):
        bucket = "short" if i < 8 else "long"
        target = 512 if i < 8 else 128
        prompt = cc.build_length_targeted_prompt(bucket, target, seed=20260718, variant_index=1000 + i)
        requests.append(PlannedRequest(
            scenario_name="stress_mixed_prefill_decode_contention",
            request_id=i,
            arrival_time_s=arrivals[i],
            prompt_bucket=bucket,
            target_output_tokens=target,
            intended_prompt_tokens=cc.approx_token_count(prompt),
            prompt_text=prompt,
        ))
    return RuntimeScenario(
        "stress_mixed_prefill_decode_contention",
        "new long-prefill arrivals while long decodes are active",
        requests,
        16,
        "mixed_prefill_decode",
    )


def _burst_recovery_scenario() -> RuntimeScenario:
    requests: list[PlannedRequest] = []
    arrivals = [0.0] * 16 + [1.5 + 0.25 * i for i in range(8)]
    for i in range(24):
        bucket = "medium" if i % 3 else "long"
        target = 256 if i % 4 else 384
        prompt = cc.build_length_targeted_prompt(bucket, target, seed=20260718, variant_index=2000 + i)
        requests.append(PlannedRequest(
            scenario_name="stress_burst_overload_recovery",
            request_id=i,
            arrival_time_s=arrivals[i],
            prompt_bucket=bucket,
            target_output_tokens=target,
            intended_prompt_tokens=cc.approx_token_count(prompt),
            prompt_text=prompt,
        ))
    return RuntimeScenario(
        "stress_burst_overload_recovery",
        "burst above sustainable capacity followed by recovery arrivals",
        requests,
        24,
        "bursty_transient",
    )


def _stress_real_trace_scenarios(root: Path) -> list[RuntimeScenario]:
    specs = [
        ("stress_burstgpt_replay", "burstgpt", root / "data/processed/burstgpt/burstgpt_scaled_moderate_10k.jsonl"),
        ("stress_azure_2023_replay", "azure_llm_2023", root / "data/processed/azure/azure_llm_2023_conv.jsonl"),
    ]
    out: list[RuntimeScenario] = []
    for name, source, path in specs:
        if not path.exists():
            continue
        reqs, _meta = load_extended_jsonl(path)
        selected = list(reqs[:16])
        if not selected:
            continue
        planned = []
        for i, r in enumerate(selected):
            bucket = "long" if r.prompt_tokens >= 1200 else "medium" if r.prompt_tokens >= 400 else "short"
            target = max(64, min(512, int(r.actual_output_tokens) * 4))
            prompt = cc.build_length_targeted_prompt(bucket, target, seed=20260718, variant_index=3000 + i)
            planned.append(PlannedRequest(
                scenario_name=name,
                request_id=i,
                arrival_time_s=0.0 if i < 10 else 1.0 + 0.1 * (i - 10),
                prompt_bucket=bucket,
                target_output_tokens=target,
                intended_prompt_tokens=cc.approx_token_count(prompt),
                prompt_text=prompt,
                source_trace=source,
            ))
        out.append(RuntimeScenario(name, f"stress {source} replay with synthetic prompt text", planned, 16, "real_trace_stress"))
    return out


def _real_trace_scenarios(root: Path) -> list[RuntimeScenario]:
    specs = [
        ("burstgpt_replay_small", "burstgpt", root / "data/processed/burstgpt/burstgpt_scaled_moderate_10k.jsonl"),
        ("azure_2023_replay_small", "azure_llm_2023", root / "data/processed/azure/azure_llm_2023_conv.jsonl"),
    ]
    out: list[RuntimeScenario] = []
    for name, source, path in specs:
        if not path.exists():
            continue
        reqs, _meta = load_extended_jsonl(path)
        selected = list(reqs[:8])
        if not selected:
            continue
        base = selected[0].arrival_time
        arrivals = [min(0.08 * i, max(0.0, (r.arrival_time - base) * 0.02)) for i, r in enumerate(selected)]
        planned = []
        for i, r in enumerate(selected):
            bucket = "long" if r.prompt_tokens >= 1200 else "medium" if r.prompt_tokens >= 400 else "short"
            target = max(8, min(128, int(r.actual_output_tokens)))
            prompt = cc.build_length_targeted_prompt(bucket, target, seed=20260718, variant_index=i)
            planned.append(PlannedRequest(
                scenario_name=name,
                request_id=i,
                arrival_time_s=arrivals[i],
                prompt_bucket=bucket,
                target_output_tokens=target,
                intended_prompt_tokens=cc.approx_token_count(prompt),
                prompt_text=prompt,
                source_trace=source,
            ))
        out.append(RuntimeScenario(name, f"small {source} replay with synthetic prompt text", planned, 4, "real_trace_replay"))
    return out


def run_runtime_scenario(
    scenario: RuntimeScenario,
    *,
    server_url: str,
    model: str,
    timeout_s: float,
    mock: bool,
) -> tuple[list[RuntimeResult], dict]:
    poller = MetricsPoller(server_url)
    sem = threading.Semaphore(scenario.max_client_concurrency)
    start = time.monotonic()
    poller.start()

    def _run_one(planned: PlannedRequest) -> RuntimeResult:
        sleep_s = start + planned.arrival_time_s - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        acquire_start = time.monotonic()
        with sem:
            queue_delay = time.monotonic() - acquire_start
            try:
                if mock:
                    time.sleep(0.001)
                    out = {
                        "ttft_s": 0.01,
                        "latency_s": 0.02 + planned.target_output_tokens * 0.001,
                        "prompt_tokens": float(planned.intended_prompt_tokens),
                        "output_tokens": float(planned.target_output_tokens),
                        "finish_reason": "length",
                    }
                else:
                    out = query_vllm_completion(
                        server_url,
                        model,
                        planned.prompt_text,
                        max_tokens=planned.target_output_tokens,
                        timeout_s=timeout_s,
                    )
                return RuntimeResult(
                    scenario.name,
                    planned.request_id,
                    "success",
                    planned.arrival_time_s,
                    queue_delay,
                    out.get("ttft_s"),
                    out.get("latency_s"),
                    out.get("prompt_tokens"),
                    out.get("output_tokens"),
                    out.get("finish_reason"),
                )
            except Exception as exc:  # noqa: BLE001
                return RuntimeResult(
                    scenario.name,
                    planned.request_id,
                    "error",
                    planned.arrival_time_s,
                    queue_delay,
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"{type(exc).__name__}: {str(exc)[:300]}",
                )

    with ThreadPoolExecutor(max_workers=max(1, scenario.max_client_concurrency)) as pool:
        futures = [pool.submit(_run_one, req) for req in scenario.requests]
        results = [f.result() for f in as_completed(futures)]
    samples = poller.stop()
    return sorted(results, key=lambda r: r.request_id), _summarize_runtime(scenario, results, samples)


def run_simulator_scenario(scenario: RuntimeScenario) -> dict:
    requests = [
        Request(
            request_id=p.request_id,
            arrival_time=p.arrival_time_s,
            prompt_tokens=max(1, int(p.intended_prompt_tokens)),
            predicted_output_tokens=p.target_output_tokens,
            actual_output_tokens=p.target_output_tokens,
            slo_deadline=p.arrival_time_s + 10_000.0,
            priority=1.0,
            class_id="audit",
        )
        for p in scenario.requests
    ]
    gpu_configs = [GPUConfig(0, max_active_sequences=256, max_batch_tokens=2560, max_kv_tokens=131_072)]
    policies = {
        "vllm_faithful": (make_external_baseline("vllm_faithful"), ServiceModel()),
        "sarathi_faithful": (
            make_external_baseline("sarathi_faithful"),
            ServiceModel(
                enable_prefill_modeling=True,
                decode_first=True,
                step_token_budget=512,
                max_prefill_chunk_tokens=512,
            ),
        ),
    }
    out: dict[str, dict] = {}
    for name, (policy, svc) in policies.items():
        metrics = run_policy(
            policy=policy,
            requests=requests,
            gpu_configs=gpu_configs,
            service_model=svc,
            workload_tag=scenario.name,
            seed=20260718,
            drain_steps=50_000,
        )
        out[name] = {
            "num_completed": metrics.num_completed,
            "num_dropped": metrics.num_dropped,
            "completion_fraction": metrics.completion_fraction,
            "request_throughput": metrics.request_throughput,
            "mean_latency_s": metrics.mean_latency,
            "p95_latency_s": metrics.p95_latency,
            "mean_ttft_s": metrics.mean_ttft,
            "p95_ttft_s": metrics.p95_ttft,
            "mean_tpot_s": metrics.mean_tpot,
            "arrival_normalized_weighted_goodput": metrics.arrival_normalized_weighted_goodput,
        }
    return out


def _summarize_runtime(scenario: RuntimeScenario, results: list[RuntimeResult], samples: list[dict[str, float]]) -> dict:
    successes = [r for r in results if r.status == "success"]
    latencies = [r.latency_s for r in successes if r.latency_s is not None]
    ttfts = [r.ttft_s for r in successes if r.ttft_s is not None]
    outputs = [r.output_tokens for r in successes if r.output_tokens is not None]
    prompt_tokens = [r.prompt_tokens for r in successes if r.prompt_tokens is not None]
    elapsed = (
        max((r.scheduled_arrival_s + (r.latency_s or 0.0) for r in successes), default=0.0)
        - min((r.scheduled_arrival_s for r in successes), default=0.0)
    )
    return {
        "scenario_name": scenario.name,
        "description": scenario.description,
        "scenario_family": scenario.scenario_family,
        "num_requests": len(results),
        "num_success": len(successes),
        "completion_fraction": len(successes) / len(results) if results else 0.0,
        "request_throughput": len(successes) / elapsed if elapsed > 0 else None,
        "mean_prompt_tokens": _mean(prompt_tokens),
        "mean_output_tokens": _mean(outputs),
        "mean_latency_s": _mean(latencies),
        "p50_latency_s": _percentile(latencies, 0.50),
        "p95_latency_s": _percentile(latencies, 0.95),
        "mean_ttft_s": _mean(ttfts),
        "p50_ttft_s": _percentile(ttfts, 0.50),
        "p95_ttft_s": _percentile(ttfts, 0.95),
        "mean_tpot_s": _mean([
            (r.latency_s - r.ttft_s) / max(1.0, (r.output_tokens or 1.0) - 1.0)
            for r in successes
            if r.latency_s is not None and r.ttft_s is not None and r.output_tokens is not None
        ]),
        "max_vllm_running": max((s.get("running", 0.0) for s in samples), default=0.0),
        "max_vllm_waiting": max((s.get("waiting", 0.0) for s in samples), default=0.0),
        "max_kv_cache_usage": max((s.get("kv_cache_usage", 0.0) for s in samples), default=0.0),
        "preemption_events_delta": _counter_delta(samples, "preemptions_total"),
        "prefix_cache_queries_delta": _counter_delta(samples, "prefix_cache_queries_total"),
        "prefix_cache_hits_delta": _counter_delta(samples, "prefix_cache_hits_total"),
    }


def audit_environment(server_url: str, model: str) -> dict:
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "server_url": server_url,
        "git_head": _run(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "git_branch": _run(["git", "branch", "--show-current"], cwd=ROOT),
        "nvidia_smi_query": _run([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader",
        ]),
        "nvidia_smi": _run(["nvidia-smi"]),
        "vllm_http_version": None,
        "models_endpoint": None,
        "python_packages": {},
    }
    base_url = server_url.rstrip("/")
    for url, key in [(f"{base_url}/version", "vllm_http_version"), (f"{base_url}/v1/models", "models_endpoint")]:
        try:
            env[key] = _urlopen_text(url, timeout_s=3.0)
        except Exception as exc:  # noqa: BLE001
            env[key] = f"ERROR: {exc}"
    py = Path("/home/soroush/.venvs/vllm_baseline_pilot/bin/python")
    if py.exists():
        code = (
            "import importlib.metadata as m, json;"
            "pkgs=['vllm','torch','transformers','sarathi','sarathi-serve'];"
            "installed={d.metadata['Name'].lower():d.metadata['Name'] for d in m.distributions()};"
            "print(json.dumps({p:(m.version(installed[p]) if p in installed else None) for p in pkgs}))"
        )
        try:
            env["python_packages"] = json.loads(_run([str(py), "-c", code]) or "{}")
        except Exception:
            env["python_packages"] = {}
    return env


def write_outputs(out_dir: Path, env: dict, scenarios: list[RuntimeScenario], scenario_reports: list[dict], request_rows: list[RuntimeResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True))
    (out_dir / "scenario_results.json").write_text(json.dumps(scenario_reports, indent=2, sort_keys=True))
    with open(out_dir / "requests.jsonl", "w") as f:
        for row in request_rows:
            f.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    with open(out_dir / "scenario_summary.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in scenario_reports for k in r["runtime_summary"]})
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in scenario_reports:
            writer.writerow(r["runtime_summary"])
    summary = summarize_audit(scenario_reports)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "summary.md").write_text(_summary_md(summary, env, scenario_reports) + "\n")


def load_checkpoint(out_dir: Path) -> tuple[list[dict], list[RuntimeResult]]:
    reports_path = out_dir / "scenario_results.json"
    requests_path = out_dir / "requests.jsonl"
    reports = json.loads(reports_path.read_text()) if reports_path.exists() else []
    request_rows: list[RuntimeResult] = []
    if requests_path.exists():
        for line in requests_path.read_text().splitlines():
            if line.strip():
                request_rows.append(RuntimeResult(**json.loads(line)))
    return reports, request_rows


def make_calibration_profile(env: dict, summary: dict, reports: list[dict]) -> dict:
    prefill_points = [
        (r["runtime_summary"].get("mean_prompt_tokens"), r["runtime_summary"].get("mean_ttft_s"))
        for r in reports
        if r["runtime_summary"].get("mean_prompt_tokens") is not None
        and r["runtime_summary"].get("mean_ttft_s") is not None
    ]
    decode_points = [
        r["runtime_summary"].get("mean_tpot_s")
        for r in reports
        if r["runtime_summary"].get("mean_tpot_s") is not None
    ]
    intercept, slope = _linear_fit(prefill_points)
    warnings = []
    if summary.get("scenarios_with_vllm_waiting", 0) == 0:
        warnings.append("No waiting queue observed; saturation calibration remains incomplete.")
    if (summary.get("max_observed_kv_cache_usage") or 0.0) < 0.2:
        warnings.append("KV usage stayed below 20%; KV-pressure calibration remains incomplete.")
    return {
        "schema_version": "gpu_external_validity_calibration_profile_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": env.get("model"),
        "server_command": env.get("server_command"),
        "server_url": env.get("server_url"),
        "git_head": env.get("git_head"),
        "summary": summary,
        "prefill_latency_fit": {
            "form": "ttft_seconds ~= intercept_s + slope_s_per_prompt_token * prompt_tokens",
            "intercept_s": intercept,
            "slope_s_per_prompt_token": slope,
            "num_points": len(prefill_points),
        },
        "decode_tpot_summary": {
            "mean_s_per_token": _mean(decode_points),
            "median_s_per_token": statistics.median(decode_points) if decode_points else None,
            "num_points": len(decode_points),
        },
        "warnings": warnings,
        "historical_defaults_changed": False,
    }


def summarize_audit(scenario_reports: list[dict]) -> dict:
    runtime_lat = [r["runtime_summary"].get("mean_latency_s") for r in scenario_reports]
    runtime_ttft = [r["runtime_summary"].get("mean_ttft_s") for r in scenario_reports]
    vllm_sim_lat = [r["simulator_summary"]["vllm_faithful"].get("mean_latency_s") for r in scenario_reports]
    sarathi_sim_lat = [r["simulator_summary"]["sarathi_faithful"].get("mean_latency_s") for r in scenario_reports]
    return {
        "num_scenarios": len(scenario_reports),
        "num_requests": sum(r["runtime_summary"]["num_requests"] for r in scenario_reports),
        "num_success": sum(r["runtime_summary"]["num_success"] for r in scenario_reports),
        "runtime_mean_latency_s": _mean([x for x in runtime_lat if x is not None]),
        "runtime_mean_ttft_s": _mean([x for x in runtime_ttft if x is not None]),
        "sim_vllm_mean_latency_s": _mean([x for x in vllm_sim_lat if x is not None and not math.isnan(x)]),
        "sim_sarathi_mean_latency_s": _mean([x for x in sarathi_sim_lat if x is not None and not math.isnan(x)]),
        "vllm_runtime_vs_sim_latency_ratio_median": _median_ratio(runtime_lat, vllm_sim_lat),
        "sarathi_sim_vs_vllm_sim_latency_ratio_median": _median_ratio(sarathi_sim_lat, vllm_sim_lat),
        "scenarios_with_vllm_waiting": sum(1 for r in scenario_reports if r["runtime_summary"].get("max_vllm_waiting", 0.0) > 0.0),
        "max_observed_kv_cache_usage": max((r["runtime_summary"].get("max_kv_cache_usage", 0.0) for r in scenario_reports), default=0.0),
        "max_observed_vllm_running": max((r["runtime_summary"].get("max_vllm_running", 0.0) for r in scenario_reports), default=0.0),
        "preemption_events": sum((r["runtime_summary"].get("preemption_events_delta", 0.0) for r in scenario_reports), 0.0),
    }


def _summary_md(summary: dict, env: dict, reports: list[dict]) -> str:
    lines = [
        "# GPU External-Validity Audit",
        "",
        f"- Model: `{env.get('model')}`",
        f"- Server URL: `{env.get('server_url')}`",
        f"- vLLM HTTP version: `{env.get('vllm_http_version')}`",
        f"- Scenarios: {summary['num_scenarios']}",
        f"- Requests: {summary['num_requests']} ({summary['num_success']} success)",
        f"- Runtime mean latency: {summary['runtime_mean_latency_s']}",
        f"- Runtime mean TTFT: {summary['runtime_mean_ttft_s']}",
        f"- Simulator vLLM mean latency: {summary['sim_vllm_mean_latency_s']}",
        f"- Median runtime/sim-vLLM latency ratio: {summary['vllm_runtime_vs_sim_latency_ratio_median']}",
        f"- Median Sarathi-sim/vLLM-sim latency ratio: {summary['sarathi_sim_vs_vllm_sim_latency_ratio_median']}",
        f"- Scenarios with vLLM waiting >0: {summary['scenarios_with_vllm_waiting']}",
        f"- Max observed vLLM running sequences: {summary['max_observed_vllm_running']}",
        f"- Max observed vLLM KV-cache usage: {summary['max_observed_kv_cache_usage']}",
        f"- Preemption events: {summary['preemption_events']}",
        "",
        "## Scenario Table",
        "",
        "| Scenario | Runtime mean latency | Runtime mean TTFT | vLLM sim mean latency | Sarathi sim mean latency | max running | max waiting | max KV | preemptions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        rt = r["runtime_summary"]
        sim = r["simulator_summary"]
        lines.append(
            f"| {rt['scenario_name']} | {_fmt(rt.get('mean_latency_s'))} | {_fmt(rt.get('mean_ttft_s'))} | "
            f"{_fmt(sim['vllm_faithful'].get('mean_latency_s'))} | {_fmt(sim['sarathi_faithful'].get('mean_latency_s'))} | "
            f"{_fmt(rt.get('max_vllm_running'))} | {_fmt(rt.get('max_vllm_waiting'))} | {_fmt(rt.get('max_kv_cache_usage'))} | "
            f"{_fmt(rt.get('preemption_events_delta'))} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "stress", "xlong_stress"], default="smoke")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-live-server", action="store_true")
    parser.add_argument("--start-vllm-server", action="store_true")
    parser.add_argument("--vllm-executable", default=DEFAULT_VLLM_EXECUTABLE)
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--server-ready-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-log-requests", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-calibration-profile", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--no-real-traces", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_live_server and not args.mock and not args.start_vllm_server:
        print("ERROR: pass --allow-live-server, --start-vllm-server, or --mock", file=sys.stderr)
        return 2
    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        print("ERROR: choose at most one chunked-prefill mode", file=sys.stderr)
        return 2
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "experiments" / "gpu_external_validity" / f"vllm_{args.phase}_{timestamp}"
    if args.phase == "xlong_stress":
        scenarios = build_xlong_stress_scenarios(ROOT, include_real_traces=not args.no_real_traces)
    elif args.phase == "stress":
        scenarios = build_stress_scenarios(ROOT, include_real_traces=not args.no_real_traces)
    else:
        scenarios = build_scenarios(ROOT, include_real_traces=not args.no_real_traces)
    if args.limit_scenarios is not None:
        scenarios = scenarios[:args.limit_scenarios]

    server_proc: subprocess.Popen | None = None
    server_log_handle = None
    if args.start_vllm_server:
        args.server_url = _server_url_for_port(args.server_url, args.port)
        server_log_path = Path(args.server_log) if args.server_log else out_dir / "server.log"
        server_log_path.parent.mkdir(parents=True, exist_ok=True)
        server_log_handle = open(server_log_path, "a")
        server_cmd = _build_vllm_server_command(args)
        server_proc = _start_vllm_server(server_cmd, server_log_handle)
        _install_server_cleanup(server_proc)
        if not _wait_for_server(args.server_url, args.server_ready_timeout_seconds):
            print(f"ERROR: vLLM server did not become ready at {args.server_url}", file=sys.stderr)
            _stop_vllm_server(server_proc)
            return 1
    else:
        server_cmd = None

    try:
        env = audit_environment(args.server_url, args.model)
        env.update({
            "phase": args.phase,
            "server_command": " ".join(server_cmd) if server_cmd else None,
            "server_started_by_audit": bool(args.start_vllm_server),
            "scenario_count_planned": len(scenarios),
        })
        if args.resume:
            reports, request_rows = load_checkpoint(out_dir)
        else:
            reports, request_rows = [], []
        completed = {r["scenario"]["name"] for r in reports}
        for scenario in scenarios:
            if scenario.name in completed:
                print(f"skipping completed {scenario.name}", flush=True)
                continue
            print(f"running {scenario.name} ({len(scenario.requests)} requests)", flush=True)
            runtime_rows, runtime_summary = run_runtime_scenario(
                scenario,
                server_url=args.server_url,
                model=args.model,
                timeout_s=args.timeout_seconds,
                mock=args.mock,
            )
            simulator_summary = run_simulator_scenario(scenario)
            request_rows.extend(runtime_rows)
            reports.append({
                "scenario": {
                    "name": scenario.name,
                    "description": scenario.description,
                    "max_client_concurrency": scenario.max_client_concurrency,
                    "scenario_family": scenario.scenario_family,
                },
                "runtime_summary": runtime_summary,
                "simulator_summary": simulator_summary,
            })
            write_outputs(out_dir, env, scenarios, reports, request_rows)
        summary = summarize_audit(reports)
        if args.write_calibration_profile:
            (out_dir / "calibration_profile.json").write_text(
                json.dumps(make_calibration_profile(env, summary, reports), indent=2, sort_keys=True)
            )
        print(json.dumps({"output_dir": str(out_dir), **summary}, indent=2, sort_keys=True))
    finally:
        if server_proc is not None:
            _stop_vllm_server(server_proc)
        if server_log_handle is not None:
            server_log_handle.close()
    return 0


def _parse_vllm_metrics(text: str) -> dict[str, float]:
    out = {
        "running": 0.0,
        "waiting": 0.0,
        "kv_cache_usage": 0.0,
        "preemptions_total": 0.0,
        "prefix_cache_queries_total": 0.0,
        "prefix_cache_hits_total": 0.0,
    }
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            out["running"] += _metric_value(line)
        elif line.startswith("vllm:num_requests_waiting{"):
            out["waiting"] += _metric_value(line)
        elif line.startswith("vllm:kv_cache_usage_perc"):
            out["kv_cache_usage"] = max(out["kv_cache_usage"], _metric_value(line))
        elif line.startswith("vllm:num_preemptions_total"):
            out["preemptions_total"] += _metric_value(line)
        elif line.startswith("vllm:prefix_cache_queries_total"):
            out["prefix_cache_queries_total"] += _metric_value(line)
        elif line.startswith("vllm:prefix_cache_hits_total"):
            out["prefix_cache_hits_total"] += _metric_value(line)
    return out


def _server_url_for_port(server_url: str, port: int | None) -> str:
    if port is None:
        return server_url
    return f"http://127.0.0.1:{port}"


def _build_vllm_server_command(args: argparse.Namespace) -> list[str]:
    cmd = [args.vllm_executable, "serve", args.model]
    port = args.port
    if port is not None:
        cmd.extend(["--port", str(port)])
    flag_values = [
        ("--revision", args.model_revision),
        ("--gpu-memory-utilization", args.gpu_memory_utilization),
        ("--max-model-len", args.max_model_len),
        ("--max-num-seqs", args.max_num_seqs),
        ("--max-num-batched-tokens", args.max_num_batched_tokens),
        ("--block-size", args.block_size),
    ]
    for flag, value in flag_values:
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    if args.disable_chunked_prefill:
        cmd.append("--no-enable-chunked-prefill")
    if args.disable_prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.disable_log_requests:
        cmd.append("--disable-log-requests")
    return cmd


def _start_vllm_server(cmd: list[str], log_handle) -> subprocess.Popen:
    env = os.environ.copy()
    cuda_home = "/home/soroush/.venvs/vllm_baseline_pilot/lib/python3.12/site-packages/nvidia/cu13"
    env["CUDA_HOME"] = env.get("CUDA_HOME", cuda_home)
    env["PATH"] = f"{cuda_home}/bin:/home/soroush/.venvs/vllm_baseline_pilot/bin:{env.get('PATH', '')}"
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    print(f"starting vLLM server: {' '.join(cmd)}", flush=True)
    log_handle.write(f"\n# starting vLLM server: {' '.join(cmd)}\n")
    log_handle.flush()
    return subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=ROOT,
        start_new_session=True,
        text=True,
    )


def _wait_for_server(server_url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            _urlopen_text(f"{server_url.rstrip('/')}/version", timeout_s=5.0)
            _urlopen_text(f"{server_url.rstrip('/')}/v1/models", timeout_s=5.0)
            return True
        except Exception:
            time.sleep(5.0)
    return False


def _install_server_cleanup(proc: subprocess.Popen) -> None:
    def _handler(signum, _frame):
        _stop_vllm_server(proc)
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _stop_vllm_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _counter_delta(samples: list[dict[str, float]], key: str) -> float:
    vals = [s[key] for s in samples if key in s]
    if len(vals) < 2:
        return 0.0
    return max(0.0, vals[-1] - vals[0])


def _metric_value(line: str) -> float:
    try:
        return float(line.rsplit(" ", 1)[-1])
    except ValueError:
        return 0.0


def _urlopen_text(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def _mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else None


def _percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[idx]


def _median_ratio(numerators: Iterable[float | None], denominators: Iterable[float | None]) -> float | None:
    ratios = [
        n / d
        for n, d in zip(numerators, denominators)
        if n is not None and d is not None and not math.isnan(n) and not math.isnan(d) and d > 0
    ]
    return statistics.median(ratios) if ratios else None


def _linear_fit(points: Iterable[tuple[float | None, float | None]]) -> tuple[float | None, float | None]:
    vals = [
        (float(x), float(y))
        for x, y in points
        if x is not None and y is not None and not math.isnan(float(x)) and not math.isnan(float(y))
    ]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0][1], 0.0
    xs = [x for x, _ in vals]
    ys = [y for _, y in vals]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return y_mean, 0.0
    slope = sum((x - x_mean) * (y - y_mean) for x, y in vals) / denom
    intercept = y_mean - slope * x_mean
    return intercept, slope


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
