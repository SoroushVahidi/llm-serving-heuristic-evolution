#!/usr/bin/env python3
"""Fake Apt-Serve scheduler worker for CI/testing.

Does not import torch/vLLM. Simulates subprocess IPC/error states deterministically
based on request ID patterns.
"""
from __future__ import annotations

import json
import sys
import time


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

    # Gather input request IDs
    waiting_ids = [r["request_id"] for r in payload.get("waiting_requests", [])]
    running_ids = [r["request_id"] for r in payload.get("running_requests", [])]
    input_ids = set(waiting_ids + running_ids)

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
        time.sleep(20) # exceeds 10s timeout
        return 0

    # 4. Scenario 666666: wrong schema version in response
    if 666666 in input_ids:
        response = {
            "schema_version": 2, # wrong!
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
            "selected_request_ids": [12345], # not in input!
            "cache_assignments": {},
            "evictions": [],
            "deprioritized_requests": [],
            "value_scores": {}
        }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        return 0

    # Normal valid decision scenario: select the first waiting request, mark it as KV, no evictions
    selected = [waiting_ids[0]] if waiting_ids else []
    cache_asg = {str(rid): "kv" for rid in input_ids}
    value_sc = {str(rid): 1.0 for rid in input_ids}

    response = {
        "schema_version": 1,
        "request_id": payload["request_id"],
        "selected_request_ids": selected,
        "cache_assignments": cache_asg,
        "evictions": [],
        "deprioritized_requests": [],
        "value_scores": value_sc
    }

    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
