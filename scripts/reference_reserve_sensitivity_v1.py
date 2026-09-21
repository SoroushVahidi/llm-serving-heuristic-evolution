#!/usr/bin/env python3
"""Execute the pre-specified post hoc reference-reserve sensitivity analysis."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(os.environ.get("FRESH_EXECUTION_ROOT", str(Path(__file__).resolve().parents[1])))
for extra in (ROOT, ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy

def run_smoke_test(reserve: float):
    logging.info(f"Running smoke test with reserve {reserve}")
    # Verify we can instantiate the policy with the custom reserve
    policy = KVConstrainedOnlinePolicy(target_kv_utilization=reserve)
    assert policy.target_kv_utilization == reserve, "Failed to configure reserve"
    logging.info("Smoke test passed.")

def run_experiment(reserve: float):
    logging.info(f"Starting reference_reserve_sensitivity_v1 for reserve: {reserve}")
    # End-to-end execution would be placed here.
    # Currently omitted as we stop before submitting to the cluster.
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", type=float, required=True, choices=[0.82, 0.90, 1.00])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(args.reserve)
    else:
        run_experiment(args.reserve)
