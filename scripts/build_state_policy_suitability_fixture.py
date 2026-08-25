#!/usr/bin/env python3
"""Build a small, genuinely discriminative state-policy fixture, covering
the full 27-policy Policy Library v2 registry.

Reuses the existing Selector Dataset v2 synthetic controlled-stress window
generators and discriminativeness classifier (single source of truth:
selector/dataset_v2/calibrated_targeted_pilot.py,
selector/dataset_v2/discriminativeness.py) -- extended here to run all 27
deployable policies per window instead of the 8-policy Option B pool, since
this fixture is for the joint state-policy suitability model, not Option B
Selector v2 training.

Small by design (a couple dozen windows): this is a local fixture for
correctness/signal evaluation, not a large simulator sweep.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.core.types import GPUConfig  # noqa: E402
from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES  # noqa: E402
from llmserveopt.selector.dataset_v2 import calibrated_targeted_pilot as p  # noqa: E402
from llmserveopt.selector.dataset_v2.discriminativeness import (  # noqa: E402
    PRIMARY_SELECTOR_OBJECTIVE, STANDARD_OBJECTIVES, compute_discriminativeness,
)
from llmserveopt.selector.dataset_v2.features import extract_selector_v2_features  # noqa: E402
from llmserveopt.selector.suitability.dataset import build_long_format_rows  # noqa: E402

from run_local_e2e_smoke import run_policy_library_v2_candidate_on_window  # noqa: E402

PRIMARY_OBJECTIVE = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)
DISCRIMINATIVE_CLASSES = ("STRONGLY_DISCRIMINATIVE", "MODERATELY_DISCRIMINATIVE")


def _split_for_index(idx: int, n_total_retained: int) -> str:
    # Deterministic, non-random split by generation order over the actual
    # retained count -- small fixture, so a simple chronological-style
    # split is sufficient and transparent.
    frac = idx / max(1, n_total_retained - 1)
    if frac < 0.6:
        return "TRAIN"
    if frac < 0.8:
        return "VALIDATION"
    return "TEST"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/state_policy_suitability_fixture/latest")
    parser.add_argument("--target-windows", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--drain-steps", type=int, default=4000)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    rng = random.Random(args.seed)
    shapes = list(p.FAMILY_GENERATORS.keys())

    window_rows: List[Dict] = []
    policy_rows: List[Dict] = []
    n_attempts = 0
    n_retained = 0
    n_discriminative = 0

    while n_retained < args.target_windows and n_attempts < args.max_attempts:
        shape = shapes[n_attempts % len(shapes)]
        window = p.synthetic_candidate_window(shape, rng, n_attempts)
        n_attempts += 1
        calibrated = p.calibrate_candidate_window(window, multiplier=2.0)
        if not calibrated:
            continue

        gpu_configs = [GPUConfig(
            0, max_active_sequences=window.max_active_sequences,
            max_batch_tokens=1_000_000, max_kv_tokens=window.max_kv_tokens,
        )]
        service_model = p._execution_service_model(window.budget, window.chunk)

        outcomes = []
        ok = True
        for policy in POLICY_LIBRARY_V2_NAMES:
            try:
                outcome = run_policy_library_v2_candidate_on_window(
                    policy, calibrated, gpu_configs, service_model,
                    workload_tag=f"fixture_{n_attempts}", seed=args.seed, drain_steps=args.drain_steps,
                )
            except Exception:
                ok = False
                break
            outcomes.append(outcome)
        if not ok or len(outcomes) != len(POLICY_LIBRARY_V2_NAMES):
            continue

        disc = compute_discriminativeness(outcomes, PRIMARY_OBJECTIVE)
        if disc.classification not in DISCRIMINATIVE_CLASSES:
            continue
        n_discriminative += 1

        features = extract_selector_v2_features(
            window_requests=calibrated, window_start_time=calibrated[0].arrival_time,
            gpu_configs=gpu_configs, topology_class="monolithic", step_token_budget=window.budget,
        )
        widx = n_retained
        wr = {"window_idx": widx, "split": "PENDING", "shape": shape}
        wr.update({f"feat_{k}": v for k, v in features.items()})
        window_rows.append(wr)
        for outcome in outcomes:
            row = {"window_idx": widx, "policy_name": outcome.policy_name}
            row.update(outcome.to_row_dict(prefix="metric"))
            policy_rows.append(row)
        n_retained += 1

    for wr in window_rows:
        wr["split"] = _split_for_index(wr["window_idx"], n_retained)

    long_rows = build_long_format_rows(
        window_rows, policy_rows, deployable_policies=POLICY_LIBRARY_V2_NAMES,
        source="synthetic_controlled_stress_fixture", trace_family="state_policy_suitability_fixture",
        seed=args.seed,
    )

    elapsed = time.perf_counter() - t0
    manifest = {
        "n_attempts": n_attempts,
        "n_retained_windows": n_retained,
        "n_discriminative_windows": n_discriminative,
        "n_long_rows": len(long_rows),
        "n_policies": len(POLICY_LIBRARY_V2_NAMES),
        "split_counts": {s: sum(1 for w in window_rows if w["split"] == s) for s in ("TRAIN", "VALIDATION", "TEST")},
        "seed": args.seed,
        "runtime_s": round(elapsed, 3),
        "discriminativeness_objective": PRIMARY_SELECTOR_OBJECTIVE,
        "discriminative_classes_kept": list(DISCRIMINATIVE_CLASSES),
    }
    (out_dir / "long_format_rows.json").write_text(json.dumps(long_rows, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
