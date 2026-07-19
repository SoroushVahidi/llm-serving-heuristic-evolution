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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in scenario_reports:
            writer.writerow(r["runtime_summary"])
    summary = summarize_audit(scenario_reports)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "summary.md").write_text(_summary_md(summary, env, scenario_reports) + "\n")


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
        f"- Max observed vLLM KV-cache usage: {summary['max_observed_kv_cache_usage']}",
        "",
        "## Scenario Table",
        "",
        "| Scenario | Runtime mean latency | Runtime mean TTFT | vLLM sim mean latency | Sarathi sim mean latency | max waiting | max KV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        rt = r["runtime_summary"]
        sim = r["simulator_summary"]
        lines.append(
            f"| {rt['scenario_name']} | {_fmt(rt.get('mean_latency_s'))} | {_fmt(rt.get('mean_ttft_s'))} | "
            f"{_fmt(sim['vllm_faithful'].get('mean_latency_s'))} | {_fmt(sim['sarathi_faithful'].get('mean_latency_s'))} | "
            f"{_fmt(rt.get('max_vllm_waiting'))} | {_fmt(rt.get('max_kv_cache_usage'))} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-live-server", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--no-real-traces", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_live_server and not args.mock:
        print("ERROR: pass --allow-live-server or --mock", file=sys.stderr)
        return 2
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "experiments" / "gpu_external_validity" / f"vllm_qwen05b_{timestamp}"
    scenarios = build_scenarios(ROOT, include_real_traces=not args.no_real_traces)
    if args.limit_scenarios is not None:
        scenarios = scenarios[:args.limit_scenarios]

    env = audit_environment(args.server_url, args.model)
    request_rows: list[RuntimeResult] = []
    reports: list[dict] = []
    for scenario in scenarios:
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
    print(json.dumps({"output_dir": str(out_dir), **summarize_audit(reports)}, indent=2, sort_keys=True))
    return 0


def _parse_vllm_metrics(text: str) -> dict[str, float]:
    out = {"running": 0.0, "waiting": 0.0, "kv_cache_usage": 0.0}
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            out["running"] += _metric_value(line)
        elif line.startswith("vllm:num_requests_waiting{"):
            out["waiting"] += _metric_value(line)
        elif line.startswith("vllm:kv_cache_usage_perc"):
            out["kv_cache_usage"] = max(out["kv_cache_usage"], _metric_value(line))
    return out


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


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
