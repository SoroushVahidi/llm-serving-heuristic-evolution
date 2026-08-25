#!/usr/bin/env python3
"""Apt-Serve official scheduler worker. Runs inside the isolated python 3.11 environment.

Receives state JSON via stdin, monkey-patches vLLM modules with the verified
official Apt-Serve files, constructs the compatibility vLLM objects, calls
Scheduler.schedule(), and writes the decision JSON to stdout.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import deque


def load_patched_module(module_name: str, file_path: str):
    """Dynamic monkey-patching of a stock vLLM package module with a patched file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True, help="Absolute path to the Apt-Serve checkout directory")
    args = parser.parse_args()

    # Verify checkout path exists
    if not os.path.exists(args.checkout):
        print(f"Checkout path does not exist: {args.checkout}", file=sys.stderr)
        return 1

    # Apply monkey-patching BEFORE any other imports of vllm occur
    try:
        load_patched_module("vllm.block", os.path.join(args.checkout, "additional_designs/aptserve_block.py"))
        load_patched_module("vllm.sequence", os.path.join(args.checkout, "additional_designs/aptserve_sequence.py"))
        load_patched_module("vllm.core.interfaces", os.path.join(args.checkout, "additional_designs/core/aptserve_interfaces.py"))
        load_patched_module("vllm.core.block_manager_v1", os.path.join(args.checkout, "additional_designs/core/aptserve_block_manager.py"))
        load_patched_module("vllm.core.scheduler", os.path.join(args.checkout, "additional_designs/core/aptserve_scheduler.py"))
    except Exception as e:
        print(f"Monkey-patching failed: {e}", file=sys.stderr)
        return 2

    # Now import stock packages from the isolated environment
    try:
        from vllm.config import CacheConfig, SchedulerConfig
        from vllm.core.scheduler import Scheduler
        from vllm.inputs import LLMInputs
        from vllm.sampling_params import SamplingParams
        from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
    except ImportError as e:
        print(f"Failed to import vllm classes: {e}", file=sys.stderr)
        return 3

    # Read exactly one JSON request line from stdin
    try:
        line = sys.stdin.readline()
        if not line:
            print("No input payload received on stdin.", file=sys.stderr)
            return 4
        payload = json.loads(line)
    except Exception as e:
        print(f"Malformed input JSON: {e}", file=sys.stderr)
        return 5

    schema_ver = payload.get("schema_version")
    if schema_ver != 1:
        print(f"Incompatible schema version: expected 1, got {schema_ver}", file=sys.stderr)
        return 6

    # Extract configs from inputs
    gpus = payload.get("gpus", [])
    gpu = gpus[0] if gpus else {}
    max_num_seqs = gpu.get("max_active_sequences", 16)
    max_num_batched_tokens = gpu.get("max_batch_tokens", 2048)
    num_gpu_blocks = gpu.get("max_kv_tokens", 1024) // 16

    # Build the scheduler
    try:
        sched_cfg = SchedulerConfig(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=2048,
        )
        cache_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4, cache_dtype="auto")
        cache_cfg.num_gpu_blocks = num_gpu_blocks
        cache_cfg.num_cpu_blocks = num_gpu_blocks
        
        sched = Scheduler(sched_cfg, cache_cfg, lora_config=None)
        
        # Inject custom SLO and hidden budgets
        sched.ttft_slo = gpu.get("apt_serve_ttft_slo", 1.0)
        sched.tbt_slo = gpu.get("apt_serve_tbt_slo", 1.0)
        sched.max_hidden_tokens = gpu.get("hidden_cache_capacity_blocks", 0) * 16
    except Exception as e:
        print(f"Failed to construct Scheduler: {e}", file=sys.stderr)
        return 7

    # Construct the requests
    sched.waiting = deque()
    sched.running = deque()

    all_groups: Dict[str, SequenceGroup] = {}

    try:
        # Populate queues
        # 1. Waiting Queue
        for r in payload.get("waiting_requests", []):
            prompt_len = r["prompt_tokens"]
            request_id = str(r["request_id"])
            arrival_time = r.get("arrival_time", 0.0)

            inputs = LLMInputs(prompt_token_ids=list(range(prompt_len)), prompt="x" * prompt_len)
            seq = Sequence(seq_id=hash(request_id) & 0xFFFFFFF, inputs=inputs, block_size=16)
            seq.status = SequenceStatus.WAITING

            seq_group = SequenceGroup(
                request_id=request_id,
                seqs=[seq],
                arrival_time=arrival_time,
                sampling_params=SamplingParams(max_tokens=r.get("predicted_output_tokens", 16))
            )
            seq_group.metrics.last_token_time = arrival_time
            if r.get("current_cache_tier") == "hidden":
                seq_group.set_use_hidden()

            sched.waiting.append(seq_group)
            all_groups[request_id] = seq_group

        # 2. Running Queue
        for r in payload.get("running_requests", []):
            prompt_len = r["prompt_tokens"]
            request_id = str(r["request_id"])
            arrival_time = r.get("arrival_time", 0.0)

            inputs = LLMInputs(prompt_token_ids=list(range(prompt_len)), prompt="x" * prompt_len)
            seq = Sequence(seq_id=hash(request_id) & 0xFFFFFFF, inputs=inputs, block_size=16)
            seq.status = SequenceStatus.RUNNING
            # Mark decode phase by adding at least 1 output token
            seq.data.output_token_ids = [0]

            seq_group = SequenceGroup(
                request_id=request_id,
                seqs=[seq],
                arrival_time=arrival_time,
                sampling_params=SamplingParams(max_tokens=r.get("predicted_output_tokens", 16))
            )
            # running duration tells us time elapsed
            running_dur = r.get("running_duration", 0.0)
            seq_group.metrics.last_token_time = payload.get("timestamp", 0.0) - running_dur
            if r.get("current_cache_tier") == "hidden":
                seq_group.set_use_hidden()

            sched.running.append(seq_group)
            all_groups[request_id] = seq_group
            
            # Pre-allocate blocks for active running sequences in scheduler block manager
            sched.block_manager.allocate(seq_group)

    except Exception as e:
        print(f"Failed to populate scheduler requests: {e}", file=sys.stderr)
        return 8

    # Invoke the Scheduler
    try:
        seq_group_metas, outputs = sched.schedule()
    except Exception as e:
        print(f"Scheduler.schedule() crashed: {e}", file=sys.stderr)
        return 9

    # Map outputs back to our project-owned response schema
    try:
        selected_ids = [int(m.request_id) for m in seq_group_metas]
        
        # Find cache tier assignments
        cache_assignments = {}
        for rid, g in all_groups.items():
            cache_assignments[rid] = "hidden" if g.use_hidden else "kv"

        # Evictions
        evictions = []
        if hasattr(outputs, "preempted"):
            evictions = [int(g.request_id) for g in outputs.preempted]

        # Deprioritized requests
        deprioritized = []

        # Value scores (if available, otherwise mock default)
        value_scores = {rid: 1.0 for rid in all_groups.keys()}

        response = {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "selected_request_ids": selected_ids,
            "cache_assignments": cache_assignments,
            "evictions": evictions,
            "deprioritized_requests": deprioritized,
            "value_scores": value_scores
        }

        # Write to stdout
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    except Exception as e:
        print(f"Failed to serialize scheduler outputs: {e}", file=sys.stderr)
        return 10

    return 0


if __name__ == "__main__":
    sys.exit(main())
