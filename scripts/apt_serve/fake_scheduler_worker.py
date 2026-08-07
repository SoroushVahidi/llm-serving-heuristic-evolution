#!/usr/bin/env python3
"""Fake Apt-Serve scheduler worker for CI/testing.

Simulates Apt-Serve's actual dual-tier value-based scheduling heuristics
deterministically based on SLO deadlines and KV pressure, without importing vLLM/torch.
"""
from __future__ import annotations

import json
import sys
import time
import math


def main() -> int:
    try:
        line = sys.stdin.readline()
        if not line:
            print("Fake worker: No input received.", file=sys.stderr)
            return 4
        payload = json.loads(line)
    except Exception as e:
        print(f"Fake worker malformed JSON input: {e}", file=sys.stderr)
        return 5

    schema_ver = payload.get("schema_version")
    if schema_ver != 1:
        print(f"Fake worker version mismatch: {schema_ver}", file=sys.stderr)
        return 6

    # Gather input request IDs and stats
    waiting_reqs = payload.get("waiting_requests", [])
    running_reqs = payload.get("running_requests", [])
    all_reqs = waiting_reqs + running_reqs
    input_ids = {r["request_id"] for r in all_reqs}

    # Deterministic test scenario bypasses
    # 1. Scenario 999999: malformed JSON response
    if 999999 in input_ids:
        sys.stdout.write("MALFORMED_NON_JSON_CONTENT\n")
        sys.stdout.flush()
        return 0

    # 2. Scenario 888888: crash
    if 888888 in input_ids:
        print("Fake worker simulated crash (exit 1).", file=sys.stderr)
        sys.exit(1)

    # 3. Scenario 777777: delay / timeout
    if 777777 in input_ids:
        time.sleep(20)
        return 0

    # 4. Scenario 666666: wrong schema version in response
    if 666666 in input_ids:
        response = {
            "schema_version": 2,
            "request_id": payload["request_id"],
            "selected_request_ids": [],
            "cache_assignments": {},
            "evictions": [],
            "deprioritized_requests": [],
            "value_scores": {}
        }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        return 0

    # 5. Scenario 555555: duplicate/invalid decision (selected ID not in input)
    if 555555 in input_ids:
        response = {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "selected_request_ids": [12345],
            "cache_assignments": {},
            "evictions": [],
            "deprioritized_requests": [],
            "value_scores": {}
        }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        return 0

    # 6. Scenario 444444: Force overcapacity KV allocation for rollback testing
    if 444444 in input_ids:
        response = {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "selected_request_ids": [444444],
            "cache_assignments": {"444444": "kv"},
            "evictions": [],
            "deprioritized_requests": [],
            "value_scores": {}
        }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        return 0

    # --- DUAL-TIER VALUE-BASED SCHEDULING HEURISTICS ---
    selected_ids = []
    cache_assignments = {}
    evictions = []
    deprioritized = []
    value_scores = {}

    # gpu config capacity
    gpus = payload.get("gpus", [])
    gpu = gpus[0] if gpus else {}
    raw_max_kv_blocks = gpu.get("max_kv_tokens", 1024) // 16
    raw_max_hidden_blocks = gpu.get("hidden_cache_capacity_blocks", 0)

    # Account for watermark blocks (vLLM reserves 1% of total blocks, minimum 1 block)
    watermark_kv = max(1, math.ceil(raw_max_kv_blocks * 0.01))
    max_kv_blocks = raw_max_kv_blocks - watermark_kv

    if raw_max_hidden_blocks > 0:
        watermark_hidden = max(1, math.ceil(raw_max_hidden_blocks * 0.01))
        max_hidden_blocks = raw_max_hidden_blocks - watermark_hidden
    else:
        max_hidden_blocks = 0

    # Pre-populate and initialize current exact allocations from current active tiers
    current_kv_blocks = 0
    current_hidden_blocks = 0
    for r in running_reqs:
        curr_tier = r.get("current_cache_tier", "kv")
        kv_b = -(-r["prompt_tokens"] // 16)
        if curr_tier == "kv":
            current_kv_blocks += kv_b
        elif curr_tier == "hidden":
            current_hidden_blocks += max(1, math.ceil(kv_b * 0.5))

    # First pass: analyze running requests and allocate memory
    # PRIORITIZE keeping KV requests on KV by sorting ("kv" first, "hidden" second)
    running_reqs_sorted = sorted(running_reqs, key=lambda r: 0 if r.get("current_cache_tier", "kv") == "kv" else 1)
    
    # We will log decisions to help audit
    log_messages = []
    log_messages.append(f"INIT: current_kv={current_kv_blocks}, current_hidden={current_hidden_blocks}, max_kv={max_kv_blocks}, max_hidden={max_hidden_blocks}")

    for r in running_reqs_sorted:
        rid = r["request_id"]
        is_relaxed = r.get("class_id") == "relaxed_long" or (r.get("slo_deadline", 0.0) - payload.get("timestamp", 0.0) > 5.0)
        curr_tier = r.get("current_cache_tier", "kv")
        
        kv_blocks = -(-r["prompt_tokens"] // 16)
        hidden_blocks = max(1, math.ceil(kv_blocks * 0.5))

        if is_relaxed and len(waiting_reqs) > 0:
            if curr_tier == "hidden":
                # Already on Hidden, keep it there
                cache_assignments[str(rid)] = "hidden"
                selected_ids.append(rid)
                log_messages.append(f"  Req {rid}: Keep Hidden (is_relaxed)")
            else:
                # Transition from KV to Hidden if Hidden has capacity
                if current_hidden_blocks + hidden_blocks <= max_hidden_blocks:
                    cache_assignments[str(rid)] = "hidden"
                    current_kv_blocks -= kv_blocks
                    current_hidden_blocks += hidden_blocks
                    selected_ids.append(rid)
                    log_messages.append(f"  Req {rid}: KV -> Hidden (is_relaxed)")
                else:
                    # Keep on KV
                    cache_assignments[str(rid)] = "kv"
                    selected_ids.append(rid)
                    log_messages.append(f"  Req {rid}: Keep KV (no Hidden capacity)")
        else:
            # We want to keep/restore to KV
            if curr_tier == "hidden":
                # Check if we can restore to KV
                if current_kv_blocks + kv_blocks <= max_kv_blocks:
                    cache_assignments[str(rid)] = "kv"
                    current_hidden_blocks -= hidden_blocks
                    current_kv_blocks += kv_blocks
                    selected_ids.append(rid)
                    log_messages.append(f"  Req {rid}: Hidden -> KV (restore)")
                else:
                    # Keep on Hidden
                    cache_assignments[str(rid)] = "hidden"
                    selected_ids.append(rid)
                    log_messages.append(f"  Req {rid}: Keep Hidden (no KV capacity for restore)")
            else:
                # Keep on KV
                cache_assignments[str(rid)] = "kv"
                selected_ids.append(rid)
                log_messages.append(f"  Req {rid}: Keep KV (non-relaxed)")

    # Second pass: schedule waiting requests within remaining KV blocks
    for r in waiting_reqs:
        rid = r["request_id"]
        kv_blocks = -(-r["prompt_tokens"] // 16)
        hidden_blocks = max(1, math.ceil(kv_blocks * 0.5))
        
        is_relaxed = r.get("class_id") == "relaxed_long" or (r.get("slo_deadline", 0.0) - payload.get("timestamp", 0.0) > 5.0)
        
        # If we have space on KV, admit
        if current_kv_blocks + kv_blocks <= max_kv_blocks:
            cache_assignments[str(rid)] = "kv"
            current_kv_blocks += kv_blocks
            selected_ids.append(rid)
            log_messages.append(f"  Admit {rid} to KV")
        else:
            # No space on KV. If it's relaxed and we can fit on Hidden
            if is_relaxed and (current_hidden_blocks + hidden_blocks <= max_hidden_blocks):
                cache_assignments[str(rid)] = "hidden"
                current_hidden_blocks += hidden_blocks
                selected_ids.append(rid)
                log_messages.append(f"  Admit {rid} to Hidden (is_relaxed)")
            else:
                # Urgent but no space on KV -> try to evict a relaxed running request to Hidden!
                relaxed_running = [g for g in running_reqs if cache_assignments.get(str(g["request_id"])) == "kv" and (g.get("class_id") == "relaxed_long" or g.get("slo_deadline", 0.0) - payload.get("timestamp", 0.0) > 5.0)]
                if relaxed_running:
                    evict_req = relaxed_running[-1]
                    evict_id = evict_req["request_id"]
                    evict_kv_blocks = -(-evict_req["prompt_tokens"] // 16)
                    evict_hidden_blocks = max(1, math.ceil(evict_kv_blocks * 0.5))
                    
                    if current_hidden_blocks + evict_hidden_blocks <= max_hidden_blocks:
                        # Evict from KV to Hidden
                        cache_assignments[str(evict_id)] = "hidden"
                        current_kv_blocks -= evict_kv_blocks
                        current_hidden_blocks += evict_hidden_blocks
                        
                        # Admit this urgent request to KV
                        cache_assignments[str(rid)] = "kv"
                        current_kv_blocks += kv_blocks
                        selected_ids.append(rid)
                        log_messages.append(f"  Preempt {evict_id} to Hidden, Admit {rid} to KV")
                    else:
                        # Keep in queue
                        cache_assignments[str(rid)] = "none"
                        deprioritized.append(rid)
                        log_messages.append(f"  Queue {rid} (no space & no Hidden space to evict)")
                else:
                    # Keep in queue
                    cache_assignments[str(rid)] = "none"
                    deprioritized.append(rid)
                    log_messages.append(f"  Queue {rid} (no space & nothing relaxed to evict)")

    # Save diagnostics
    with open("/tmp/fake_worker.log", "a") as f:
        f.write(f"--- STEP {payload['request_id']} ---\n")
        f.write("\n".join(log_messages) + "\n\n")

    # Value scores: priority / prompt_tokens
    for r in all_reqs:
        rid = str(r["request_id"])
        value_scores[rid] = r.get("priority", 1.0) / max(1, r["prompt_tokens"])

    response = {
        "schema_version": 1,
        "request_id": payload["request_id"],
        "selected_request_ids": selected_ids,
        "cache_assignments": cache_assignments,
        "evictions": evictions,
        "deprioritized_requests": deprioritized,
        "value_scores": value_scores
    }

    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
