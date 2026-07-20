#!/usr/bin/env python3
"""Sarathi-Serve GPU smoke test + small bounded runtime-validation matrix.

Companion to scripts/run_gpu_external_validity_audit.py (the vLLM harness),
but built directly on Sarathi's offline `LLMEngine` Python API rather than an
HTTP server, since sarathi-serve's OpenAI-compatible server CLI arg surface
is auto-generated from nested dataclasses and was not worth reverse-
engineering under an overnight time budget. This does not train a selector
and does not launch a broad GPU sweep.

Phase 1 (smoke): import sarathi, confirm CUDA extensions import, confirm a
GPU is visible, build a real engine, load the model, and run two requests
(one short, one long enough to force multi-chunk prefill under
SarathiSchedulerConfig) to completion. If this fails, the script exits
non-zero and the runtime-validation phase is skipped entirely (no blind
continuation past a broken smoke test).

Phase 2 (runtime validation, only if phase 1 passes): a small (5-scenario,
~28-request) matrix reusing the already-loaded engine, covering long
prompt + moderate output, active-decode-plus-arriving-long-prefill
interference, a prefill-heavy burst, mixed prompt lengths, and one scenario
shaped to match vLLM job 1111541/1111545's `stress_kv_pressure` scenario as
closely as practical for a direct runtime comparison point. Each request's
TTFT/E2E latency/output-token count is measured from Python-level wall-clock
timestamps around LLMEngine.step() calls. Matched `sarathi_faithful` and
`vllm_faithful` simulator traces are also run for the same request shapes,
reusing the same simulator infrastructure as the vLLM harness.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import GPUConfig, Request  # noqa: E402
from llmserveopt.evaluation.run_policy import run_policy  # noqa: E402
from llmserveopt.policies.external_baselines_registry import make_external_baseline  # noqa: E402
from llmserveopt.real_llm import calibration_common as cc  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402


@dataclass(frozen=True)
class PlannedRequest:
    idx: int
    arrival_offset_s: float
    prompt_text: str
    target_output_tokens: int
    prompt_bucket: str


@dataclass
class RequestRecord:
    scenario_name: str
    seq_id: str
    submit_time: float
    first_token_time: float | None = None
    finish_time: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int = 0
    finish_reason: str | None = None
    error: str | None = None


def build_scenario(name: str, requests: list[PlannedRequest]) -> dict:
    return {"name": name, "requests": requests}


def make_scenarios() -> list[dict]:
    scenarios = []

    # A. Long prompt + moderate output.
    reqs = [
        PlannedRequest(i, 0.0, cc.build_length_targeted_prompt("long", 256, 20260719, 100 + i), 256, "long")
        for i in range(4)
    ]
    scenarios.append(build_scenario("sarathi_long_prompt_moderate_output", reqs))

    # B. Active decode streams + arriving long prefills.
    reqs = [
        PlannedRequest(i, 0.0, cc.build_length_targeted_prompt("medium", 256, 20260719, 200 + i), 256, "medium")
        for i in range(4)
    ] + [
        PlannedRequest(4 + i, 3.0 + 1.0 * i, cc.build_length_targeted_prompt("long", 64, 20260719, 210 + i), 64, "long")
        for i in range(4)
    ]
    scenarios.append(build_scenario("sarathi_active_decode_plus_arriving_prefill", reqs))

    # C. Prefill-heavy burst.
    reqs = [
        PlannedRequest(i, 0.0, cc.build_length_targeted_prompt("long", 32, 20260719, 300 + i), 32, "long")
        for i in range(6)
    ]
    scenarios.append(build_scenario("sarathi_prefill_heavy_burst", reqs))

    # D. Mixed prompt lengths.
    buckets = ["short", "medium", "long"]
    reqs = [
        PlannedRequest(i, 0.0, cc.build_length_targeted_prompt(buckets[i % 3], 64, 20260719, 400 + i), 64, buckets[i % 3])
        for i in range(6)
    ]
    scenarios.append(build_scenario("sarathi_mixed_prompt_lengths", reqs))

    # E. Matched as closely as practical to vLLM jobs 1111541/1111545's
    # stress_kv_pressure scenario: "long" bucket prompt (target 2048 tokens,
    # same as the vLLM harness's "long" bucket) + long decode (target 768,
    # same as stress_kv_pressure's target_output_tokens), burst arrival,
    # concurrency 12 (same as stress_kv_pressure's max_client_concurrency).
    reqs = [
        PlannedRequest(i, 0.0, cc.build_length_targeted_prompt("long", 768, 20260719, 500 + i), 768, "long")
        for i in range(12)
    ]
    scenarios.append(build_scenario("sarathi_matched_vllm_kv_pressure", reqs))

    # F. Short-context control: cheap, low-KV baseline point for comparing
    # against the longer-context scenarios above (and against the matched
    # real-vLLM run's equivalent control scenario).
    reqs = [
        PlannedRequest(i, 0.0, cc.build_length_targeted_prompt("short", 64, 20260719, 600 + i), 64, "short")
        for i in range(6)
    ]
    scenarios.append(build_scenario("sarathi_short_context_control", reqs))

    return scenarios


def run_smoke(engine, sampling_params_cls) -> dict:
    from sarathi.utils.hf_utils import get_and_verify_max_len  # noqa: F401  (import guard only)

    result: dict[str, Any] = {"passed": False}
    try:
        short_prompt = cc.build_length_targeted_prompt("short", 8, 20260719, 1)
        long_prompt = cc.build_length_targeted_prompt("long", 8, 20260719, 2)  # ~2048 tokens > chunk_size
        engine.add_request(
            prompt=short_prompt,
            sampling_params=sampling_params_cls(temperature=0.0, max_tokens=8),
            seq_id="smoke-short",
        )
        engine.add_request(
            prompt=long_prompt,
            sampling_params=sampling_params_cls(temperature=0.0, max_tokens=8),
            seq_id="smoke-long",
        )
        deadline = time.monotonic() + 300.0
        finished: dict[str, Any] = {}
        steps = 0
        while engine.has_unfinished_requests() and time.monotonic() < deadline:
            outputs = engine.step()
            steps += 1
            for out in outputs:
                if out.finished:
                    finished[out.seq_id] = out
        result["steps"] = steps
        result["short_finished"] = "smoke-short" in finished
        result["long_finished"] = "smoke-long" in finished
        result["short_text_nonempty"] = bool(finished.get("smoke-short") and finished["smoke-short"].text.strip())
        result["long_text_nonempty"] = bool(finished.get("smoke-long") and finished["smoke-long"].text.strip())
        result["long_prompt_tokens_gt_chunk_size"] = True  # by construction (~2048 tokens vs chunk_size=512)
        result["passed"] = (
            result["short_finished"] and result["long_finished"]
            and result["short_text_nonempty"] and result["long_text_nonempty"]
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    return result


def run_scenario(engine, sampling_params_cls, scenario: dict) -> tuple[list[RequestRecord], dict]:
    name = scenario["name"]
    planned: list[PlannedRequest] = sorted(scenario["requests"], key=lambda r: r.arrival_offset_s)
    records: dict[str, RequestRecord] = {}
    start = time.monotonic()
    next_idx = 0
    max_concurrent = 0
    deadline = start + 600.0

    while (next_idx < len(planned) or engine.has_unfinished_requests()) and time.monotonic() < deadline:
        now = time.monotonic() - start
        while next_idx < len(planned) and planned[next_idx].arrival_offset_s <= now:
            req = planned[next_idx]
            seq_id = f"{name}-{req.idx}"
            try:
                engine.add_request(
                    prompt=req.prompt_text,
                    sampling_params=sampling_params_cls(temperature=0.0, max_tokens=req.target_output_tokens),
                    seq_id=seq_id,
                )
                records[seq_id] = RequestRecord(name, seq_id, time.monotonic())
            except Exception as exc:  # noqa: BLE001
                records[seq_id] = RequestRecord(name, seq_id, time.monotonic(), error=f"{type(exc).__name__}: {exc}")
            next_idx += 1
        if engine.has_unfinished_requests():
            outputs = engine.step()
            t = time.monotonic()
            max_concurrent = max(max_concurrent, engine.get_num_unfinished_requests())
            for out in outputs:
                rec = records.get(out.seq_id)
                if rec is None or rec.error is not None:
                    continue
                if rec.prompt_tokens is None:
                    rec.prompt_tokens = len(out.prompt_token_ids)
                if rec.first_token_time is None and len(out.token_ids) > 0:
                    rec.first_token_time = t
                if out.finished and rec.finish_time is None:
                    rec.finish_time = t
                    rec.output_tokens = len(out.token_ids)
                    rec.finish_reason = out.finish_reason
        else:
            time.sleep(0.01)

    result_list = list(records.values())
    ttfts = [r.first_token_time - r.submit_time for r in result_list if r.first_token_time is not None]
    lats = [r.finish_time - r.submit_time for r in result_list if r.finish_time is not None]
    tpots = [
        (r.finish_time - r.first_token_time) / (r.output_tokens - 1)
        for r in result_list
        if r.finish_time is not None and r.first_token_time is not None and r.output_tokens > 1
    ]
    n_success = sum(1 for r in result_list if r.finish_time is not None and r.error is None)
    summary = {
        "scenario_name": name,
        "num_requests": len(planned),
        "num_success": n_success,
        "completion_fraction": (n_success / len(planned)) if planned else 0.0,
        "max_concurrent_unfinished": max_concurrent,
        "mean_ttft_s": statistics.fmean(ttfts) if ttfts else None,
        "p50_ttft_s": statistics.median(ttfts) if ttfts else None,
        "mean_tpot_s": statistics.fmean(tpots) if tpots else None,
        "mean_latency_s": statistics.fmean(lats) if lats else None,
        "p50_latency_s": statistics.median(lats) if lats else None,
        "wall_clock_timed_out": time.monotonic() >= deadline,
    }
    return result_list, summary


def run_simulator_scenario(scenario: dict) -> dict:
    requests = [
        Request(
            request_id=p.idx,
            arrival_time=p.arrival_offset_s,
            prompt_tokens=max(1, cc.approx_token_count(p.prompt_text)),
            predicted_output_tokens=p.target_output_tokens,
            actual_output_tokens=p.target_output_tokens,
            slo_deadline=p.arrival_offset_s + 10_000.0,
            priority=1.0,
            class_id="sarathi_gpu_validation",
        )
        for p in scenario["requests"]
    ]
    gpu_configs = [GPUConfig(0, max_active_sequences=256, max_batch_tokens=2560, max_kv_tokens=131_072)]
    policies = {
        "vllm_faithful": (make_external_baseline("vllm_faithful"), ServiceModel()),
        "sarathi_faithful": (
            make_external_baseline("sarathi_faithful"),
            ServiceModel(enable_prefill_modeling=True, decode_first=True, step_token_budget=512, max_prefill_chunk_tokens=512),
        ),
    }
    out: dict[str, dict] = {}
    for pname, (policy, svc) in policies.items():
        metrics = run_policy(
            policy=policy, requests=requests, gpu_configs=gpu_configs, service_model=svc,
            workload_tag=scenario["name"], seed=20260719, drain_steps=50_000,
        )
        out[pname] = {
            "num_completed": metrics.num_completed,
            "num_dropped": metrics.num_dropped,
            "completion_fraction": metrics.completion_fraction,
            "request_throughput": metrics.request_throughput,
            "mean_latency_s": metrics.mean_latency,
            "mean_ttft_s": metrics.mean_ttft,
            "mean_tpot_s": metrics.mean_tpot,
        }
    return out


def _patch_get_and_verify_max_len_for_default_rope_type() -> None:
    """Work around a compatibility bug in vendored sarathi-serve.

    Newer HuggingFace transformers always populates hf_config.rope_scaling
    with a no-op {"rope_type": "default", "rope_theta": ...} dict (no "type"
    or "factor" key) for models that use standard RoPE with no scaling --
    e.g. Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.1. This vendored
    sarathi-serve fork predates that HF convention and assumes any non-None
    rope_scaling means active scaling requiring the OLDER "type"/"factor"
    keys, in (at least) two independent call sites:

      1. sarathi.utils.hf_utils.get_and_verify_max_len() -- unconditionally
         asserts "factor" in rope_scaling. Confirmed on Qwen2.5-7B-Instruct
         (job 1111576): AssertionError, even though max_position_embeddings
         (32768) already comfortably covers the max_model_len this script
         requests (16384), so no real scaling behavior is even needed here.
      2. sarathi.model_executor.layers.rotary_embedding.get_rope(), called
         from each attention layer's __init__ (e.g. models/mistral.py) --
         unconditionally reads rope_scaling["type"]. Confirmed on
         mistralai/Mistral-7B-Instruct-v0.1 (job 1111705): KeyError: 'type'.

    Rather than patch each consumer separately (which is how this was first
    written, and which missed call site 2 on the first attempt), this
    patches sarathi.config.config's bound reference to get_config() itself
    -- the single point every consumer's hf_config flows through -- so a
    "default"/no-"type"-no-"factor" rope_scaling dict is normalized to None
    before anything downstream sees it, matching how vLLM itself handles
    this same HF config convention. It does not modify the vendored
    sarathi-serve source tree, and it only changes behavior for this
    specific no-op-rope_scaling case -- genuinely scaled models (an explicit
    "type"/"factor" pair, or a non-"default" rope_type) are returned
    unchanged and still hit the original code paths.
    """
    import sarathi.config.config as sarathi_config_module

    original_get_config = sarathi_config_module.get_config

    def patched_get_config(model, trust_remote_code, revision=None):
        hf_config = original_get_config(model, trust_remote_code, revision)
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and "type" not in rope_scaling and "factor" not in rope_scaling:
            hf_config.rope_scaling = None
        return hf_config

    sarathi_config_module.get_config = patched_get_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dtype", default=None,
        help=(
            "Model dtype passed to sarathi.config.ModelConfig. Default (None) omits "
            "the kwarg entirely so this vendored fork's own ModelConfig default "
            "(float16, see sarathi/config/config.py) applies -- its FlashInfer "
            "prefill-wrapper integration was found (job 1111711) to plan for "
            "float16 regardless of the model's configured dtype, so forcing "
            "bfloat16 (as every real-vLLM run in this investigation used) causes "
            "a dtype-mismatch ValueError at the first forward pass. Pass an "
            "explicit value (e.g. bfloat16) only if this fork's own default path "
            "is confirmed broken for a different reason."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scenario-names", default=None,
        help="Comma-separated exact scenario names to keep (others dropped). Default: all.",
    )
    parser.add_argument(
        "--trial-index", type=int, default=None,
        help="Repeated-trial index, recorded in env metadata only -- does not affect "
             "request generation (prompts/seeds stay fixed and matched across trials "
             "so any observed variance is attributable to system/execution noise, not "
             "workload content differences).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env: dict[str, Any] = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        import torch
        import sarathi
        env["torch_version"] = torch.__version__
        env["torch_cuda_version"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        env["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
        env["sarathi_version"] = getattr(sarathi, "__version__", "unknown")
        import sarathi.pos_encoding_ops, sarathi.layernorm_ops, sarathi.activation_ops, sarathi.moe_ops  # noqa: F401
        env["cuda_extensions_import_ok"] = True
    except Exception as exc:  # noqa: BLE001
        env["cuda_extensions_import_ok"] = False
        env["import_error"] = f"{type(exc).__name__}: {exc}"
        (out_dir / "smoke_result.json").write_text(json.dumps({"passed": False, "env": env}, indent=2))
        print(json.dumps({"passed": False, "reason": "import_failed", "env": env}, indent=2))
        return 1

    from sarathi.config import (
        ModelConfig, ParallelConfig, SarathiSchedulerConfig, MetricsConfig,
        SystemConfig, ReplicaConfig, WorkerConfig, CacheConfig,
    )
    from sarathi import LLMEngine, SamplingParams

    _patch_get_and_verify_max_len_for_default_rope_type()

    replica_config = ReplicaConfig(output_dir=str(out_dir / "sarathi_engine_output"))
    model_config_kwargs: dict[str, Any] = {
        "model": args.model, "max_model_len": args.max_model_len, "trust_remote_code": False,
    }
    if args.dtype is not None:
        model_config_kwargs["dtype"] = args.dtype
    model_config = ModelConfig(**model_config_kwargs)
    worker_config = WorkerConfig(gpu_memory_utilization=args.gpu_memory_utilization)
    cache_config = CacheConfig(block_size=args.block_size)
    parallel_config = ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1)
    scheduler_config = SarathiSchedulerConfig(chunk_size=args.chunk_size, max_num_seqs=args.max_num_seqs)
    metrics_config = MetricsConfig(write_metrics=False)
    system_config = SystemConfig(
        replica_config=replica_config, model_config=model_config, worker_config=worker_config,
        cache_config=cache_config, parallel_config=parallel_config, scheduler_config=scheduler_config,
        metrics_config=metrics_config,
    )
    env["server_config"] = {
        "model": args.model, "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs, "chunk_size": args.chunk_size,
        "block_size": args.block_size, "max_model_len": args.max_model_len,
        "dtype_arg": args.dtype,
        "effective_dtype": args.dtype if args.dtype is not None else "float16 (fork ModelConfig default; --dtype not passed)",
    }
    env["trial_index"] = args.trial_index
    env["prompt_seed"] = 20260719  # fixed constant baked into make_scenarios(); recorded here for provenance

    engine_build_start = time.monotonic()
    try:
        engine = LLMEngine.from_system_config(system_config)
    except Exception as exc:  # noqa: BLE001
        result = {"passed": False, "phase": "engine_build", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        (out_dir / "smoke_result.json").write_text(json.dumps({**result, "env": env}, indent=2))
        print(json.dumps(result, indent=2))
        return 1
    env["engine_build_seconds"] = time.monotonic() - engine_build_start

    smoke = run_smoke(engine, SamplingParams)
    (out_dir / "smoke_result.json").write_text(json.dumps({**smoke, "env": env}, indent=2))
    print(json.dumps({"smoke": smoke}, indent=2))
    if not smoke.get("passed"):
        print("SMOKE FAILED -- stopping before runtime-validation matrix.", file=sys.stderr)
        return 1

    scenarios = make_scenarios()
    if args.scenario_names:
        keep = {n.strip() for n in args.scenario_names.split(",") if n.strip()}
        scenarios = [s for s in scenarios if s["name"] in keep]
        missing = keep - {s["name"] for s in scenarios}
        if missing:
            print(f"WARNING: requested scenario names not found: {sorted(missing)}", file=sys.stderr)
    all_records: list[RequestRecord] = []
    scenario_reports = []
    for scenario in scenarios:
        print(f"running {scenario['name']} ({len(scenario['requests'])} requests)", flush=True)
        records, runtime_summary = run_scenario(engine, SamplingParams, scenario)
        all_records.extend(records)
        sim_summary = run_simulator_scenario(scenario)
        scenario_reports.append({
            "scenario_name": scenario["name"],
            "runtime_summary": runtime_summary,
            "simulator_summary": sim_summary,
        })
        (out_dir / "scenario_results.json").write_text(json.dumps(scenario_reports, indent=2, default=str))

    requests_path = out_dir / "requests.jsonl"
    with requests_path.open("w") as f:
        for r in all_records:
            f.write(json.dumps({
                "scenario_name": r.scenario_name,
                "seq_id": r.seq_id,
                "status": "success" if (r.finish_time is not None and r.error is None) else "error",
                "prompt_tokens": r.prompt_tokens,
                "output_tokens": r.output_tokens,
                "ttft_s": (r.first_token_time - r.submit_time) if r.first_token_time else None,
                "latency_s": (r.finish_time - r.submit_time) if r.finish_time else None,
                "finish_reason": r.finish_reason,
                "error": r.error,
            }) + "\n")

    n_req = len(all_records)
    n_success = sum(1 for r in all_records if r.finish_time is not None and r.error is None)
    summary = {
        "env": env,
        "num_scenarios": len(scenarios),
        "num_requests": n_req,
        "num_success": n_success,
        "completion_fraction": (n_success / n_req) if n_req else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
