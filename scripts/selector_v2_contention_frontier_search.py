#!/usr/bin/env python3
"""Selector v2 contention-validation pilot -- frontier-search follow-up.

Audits WHY the overnight pilot's 300-window specialization search
(experiments/selector_v2_overnight_20260720T045536Z/) tied 300/300, then
searches a much wider, bounded parameter space for the load/saturation
frontier where `enable_decode_prefill_contention` actually changes policy
outcomes. See docs/selector_v2_contention_frontier_search.md for the full
root-cause derivation this script's design is based on.

Two window "shapes", both built from a normal (non-injected) continuous
arrival trace -- no direct state seeding anywhere in this file:

* `admission_reorder`: 2-4 simultaneously-arriving, DIFFERENTLY-SIZED
  prefill-heavy requests (optionally plus a small decode-only runner
  burst shortly after). Exercises a mechanism NOT covered by
  contention_fixtures.py or the original random search: when an admitting
  policy's insertion order (used by `_advance_decode_protected`'s prefill
  loop) disagrees with strict (arrival_time, request_id) order (used by
  `_advance_shared_contention`'s), e.g. a shortest-job-first-style policy,
  the two execution models schedule the SAME simultaneously-prefilling
  requests in a genuinely different order -- a real, non-self-limiting
  divergence source (see docs/selector_v2_contention_frontier_search.md
  section "Root cause 2").
* `hog_runner_staggered`: a scaled-up, staggered-arrival generalization of
  the original contention_fixtures.py / overnight-search shape (one or
  more long-prefill "hogs", a trickle of short-prompt "runners" arriving
  over many steps) -- included to check at much larger scale (up to 40
  runners, multi-step trickles) whether that shape's documented
  self-limiting equilibrium (contention_fixtures.py's module docstring)
  actually holds, rather than assuming it from the smaller original
  search.

Diagnostic-only per-step signals (contention_diagnostics.py) are recorded
for every run so each window can be classified by whether the mechanism
was actually exercised, independent of whether the chosen objective
happened to change.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import GPUConfig, Request  # noqa: E402
from llmserveopt.evaluation.run_policy import run_policy  # noqa: E402
from llmserveopt.policies.registry import make_policy  # noqa: E402
from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy  # noqa: E402
from llmserveopt.policies.vllm_chunked_prefill_faithful import VLLMChunkedPrefillFaithfulPolicy  # noqa: E402
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy  # noqa: E402
from llmserveopt.selector.dataset_v2.builder import metrics_to_outcome_vector  # noqa: E402
from llmserveopt.selector.dataset_v2.discriminativeness import (  # noqa: E402
    STANDARD_OBJECTIVES, PRIMARY_SELECTOR_OBJECTIVE, compute_discriminativeness,
)
from llmserveopt.selector.dataset_v2.schema import PolicyOutcomeVector  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

ADMIT_CHUNK = 100_000
FAITHFUL_POLICIES = ["vllm_faithful", "sarathi_faithful", "vllm_chunked_prefill_faithful"]
CHEAP_HISTORICAL = ["fifo", "edf", "scorpio_style_slo_guard", "admission_control",
                     "weighted_shortest_processing", "estimated_service_time_first",
                     "best_fit", "multi_bin_batching"]
ALL_POLICIES = FAITHFUL_POLICIES + CHEAP_HISTORICAL
# Policies whose admission order can disagree with strict arrival order
# (see module docstring) -- used only for window-shape targeting, not to
# hand-pick winners.
ADMISSION_REORDERING_POLICIES = {"weighted_shortest_processing", "estimated_service_time_first",
                                  "scorpio_style_slo_guard", "edf"}


def _make_policy(name: str):
    if name == "sarathi_faithful":
        return SarathiFaithfulPolicy(chunk_size=ADMIT_CHUNK)
    if name == "vllm_chunked_prefill_faithful":
        return VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=ADMIT_CHUNK)
    if name == "vllm_faithful":
        return VLLMFaithfulPolicy(max_num_batched_tokens=ADMIT_CHUNK)
    return make_policy(name)


def _service_model_for_policy(policy_name: str, budget: int, chunk: int) -> ServiceModel:
    decode_first = policy_name != "vllm_chunked_prefill_faithful"
    return ServiceModel(
        enable_prefill_modeling=True, decode_first=decode_first,
        enable_decode_prefill_contention=True,
        step_token_budget=budget, max_prefill_chunk_tokens=chunk,
    )


# ---------------------------------------------------------------------------
# Window generators
# ---------------------------------------------------------------------------

def _admission_reorder_window(rng: random.Random) -> Dict:
    n_hogs = rng.randint(2, 4)
    base = rng.choice([500, 1000, 2000, 4000])
    disparity = rng.uniform(2.0, 20.0)
    sizes = [base, int(base * disparity)]
    for _ in range(n_hogs - 2):
        sizes.append(int(base * rng.uniform(1.0, disparity)))
    rng.shuffle(sizes)  # request_id order independent of size order
    budget = rng.choice([300, 400, 512, 600, 800, 1024, 1536, 2048])
    chunk = 512
    reqs = [
        Request(request_id=i, arrival_time=0.0, prompt_tokens=sizes[i],
                 predicted_output_tokens=1, actual_output_tokens=1,
                 slo_deadline=1000.0, priority=1.0, class_id="hog")
        for i in range(n_hogs)
    ]
    has_runner_burst = rng.random() < 0.5
    n_runners = rng.randint(2, 8) if has_runner_burst else 0
    runner_output = rng.choice([5, 10, 20, 40]) if has_runner_burst else 0
    arrival_gap = rng.choice([0.001, 0.002, 0.005]) if has_runner_burst else 0.0
    for i in range(n_runners):
        reqs.append(Request(
            request_id=n_hogs + i, arrival_time=arrival_gap, prompt_tokens=rng.randint(1, 40),
            predicted_output_tokens=runner_output, actual_output_tokens=runner_output,
            slo_deadline=1000.0, priority=1.0, class_id="runner",
        ))
    return dict(shape="admission_reorder", requests=reqs, budget=budget, chunk=chunk,
                n_hogs=n_hogs, hog_sizes=sizes, size_disparity=round(disparity, 2),
                n_runners=n_runners, runner_output=runner_output, arrival_gap=arrival_gap)


def _hog_runner_staggered_window(rng: random.Random) -> Dict:
    n_hogs = rng.randint(1, 3)
    hog_prompt = rng.choice([2000, 4000, 8000, 12000, 20000])
    n_runners = rng.randint(2, 40)
    runner_output = rng.choice([5, 10, 20, 40, 80])
    budget = 512 + rng.choice([1, 2, 3, 5, 8, 16, 32, 64])
    chunk = 512
    staggered = rng.random() < 0.6
    reqs = [
        Request(request_id=i, arrival_time=0.0, prompt_tokens=hog_prompt,
                 predicted_output_tokens=1, actual_output_tokens=1,
                 slo_deadline=1000.0, priority=1.0, class_id="hog")
        for i in range(n_hogs)
    ]
    for i in range(n_runners):
        arrival = (
            0.001 * (i + 1) if staggered else rng.choice([0.001, 0.002, 0.005])
        )
        reqs.append(Request(
            request_id=n_hogs + i, arrival_time=arrival, prompt_tokens=rng.randint(1, 40),
            predicted_output_tokens=runner_output, actual_output_tokens=runner_output,
            slo_deadline=1000.0, priority=1.0, class_id="runner",
        ))
    return dict(shape="hog_runner_staggered", requests=reqs, budget=budget, chunk=chunk,
                n_hogs=n_hogs, hog_prompt=hog_prompt, n_runners=n_runners,
                runner_output=runner_output, staggered=staggered)


# ---------------------------------------------------------------------------
# Per-window evaluation
# ---------------------------------------------------------------------------

def _run_window(window: Dict, search_seed: int, window_idx: int) -> Optional[Dict]:
    gpu_configs = [GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000, max_kv_tokens=200_000)]
    outcomes: List[PolicyOutcomeVector] = []
    diagnostics_by_policy: Dict[str, Dict] = {}
    raw_latencies_by_policy: Dict[str, List[Tuple[float, float, float]]] = {}
    for pname in ALL_POLICIES:
        sm = _service_model_for_policy(pname, window["budget"], window["chunk"])
        try:
            policy = _make_policy(pname)
        except Exception:
            continue
        sim = Simulator(SimulatorConfig(
            gpu_configs=gpu_configs, service_model=sm, drain_steps=5_000,
        ))
        sim.load_trace(list(window["requests"]))
        try:
            policy.reset()
        except Exception:
            pass
        try:
            m = run_policy(policy=policy, requests=list(window["requests"]), gpu_configs=gpu_configs,
                            service_model=sm, workload_tag=f"frontier_{window_idx}",
                            seed=search_seed + window_idx, drain_steps=5_000)
        except Exception:
            continue
        outcomes.append(metrics_to_outcome_vector(pname, m, {"admit": 0, "preempt": 0, "swap": 0, "migrate": 0},
                                                    gpu_count=1))
        # Re-run once more with instrumentation to collect diagnostics +
        # raw per-request latencies (run_policy() doesn't expose the
        # Simulator instance it built internally).
        sim2 = Simulator(SimulatorConfig(gpu_configs=gpu_configs, service_model=sm, drain_steps=5_000))
        sim2.load_trace(list(window["requests"]))
        policy2 = _make_policy(pname)
        try:
            policy2.reset()
        except Exception:
            pass
        try:
            sim2.run(policy=policy2, workload_tag=f"frontier_{window_idx}", seed=search_seed + window_idx)
        except Exception:
            continue
        diagnostics_by_policy[pname] = sim2.contention_diagnostics_summary()
        raw_latencies_by_policy[pname] = [
            (c.request.arrival_time, c.admission_time, c.completion_time) for c in sim2._completed
        ]

    if len(outcomes) < 2:
        return None
    return dict(window=window, outcomes=outcomes, diagnostics=diagnostics_by_policy,
                raw_latencies=raw_latencies_by_policy)


def _classify_window(result: Dict) -> str:
    diag = result["diagnostics"].get("vllm_chunked_prefill_faithful")
    if diag is None:
        return "UNKNOWN"
    completion_fractions = [
        o.completion_fraction for o in result["outcomes"] if o.completion_fraction is not None
    ]
    min_completion = min(completion_fractions) if completion_fractions else 1.0
    if min_completion < 0.5:
        return "PATHOLOGICAL_OVERLOAD"
    mechanism_active = diag["decode_stalled_steps"] > 0 or diag["prefill_stalled_steps"] > 0
    sat_frac = diag["budget_saturation_fraction"]
    if not mechanism_active and sat_frac < 0.05:
        return "UNDERLOADED"
    if sat_frac >= 0.5:
        return "SATURATED"
    if mechanism_active:
        return "CONTENTION_VISIBLE"
    return "UNDERLOADED"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-admission-reorder", type=int, default=450)
    parser.add_argument("--n-hog-runner", type=int, default=450)
    parser.add_argument("--search-seed", type=int, default=20260720)
    parser.add_argument("--output-dir", default="experiments/selector_v2_contention_frontier_search")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.search_seed)

    t0 = time.time()
    window_rows = []
    disc_rows = []
    primary_obj = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)

    generators = (
        [("admission_reorder", _admission_reorder_window)] * args.n_admission_reorder
        + [("hog_runner_staggered", _hog_runner_staggered_window)] * args.n_hog_runner
    )

    for idx, (shape_name, gen) in enumerate(generators):
        window = gen(rng)
        result = _run_window(window, args.search_seed, idx)
        if result is None:
            continue
        classification = _classify_window(result)
        diag = result["diagnostics"].get("vllm_chunked_prefill_faithful", {})

        disc_by_objective = {}
        for obj in STANDARD_OBJECTIVES:
            d = compute_discriminativeness(result["outcomes"], obj)
            disc_by_objective[obj.name] = d

        primary_disc = disc_by_objective.get(PRIMARY_SELECTOR_OBJECTIVE)

        row = {
            "window_idx": idx, "shape": shape_name, "classification": classification,
            "n_requests": len(window["requests"]),
            "budget": window["budget"], "chunk": window["chunk"],
            "decode_stalled_steps": diag.get("decode_stalled_steps", 0),
            "prefill_stalled_steps": diag.get("prefill_stalled_steps", 0),
            "budget_saturation_fraction": round(diag.get("budget_saturation_fraction", 0.0), 4),
            "max_waiting_queue": diag.get("max_waiting_queue", 0),
            "primary_objective_classification": primary_disc.classification if primary_disc else None,
            "primary_objective_best_policy": primary_disc.best_policy if primary_disc else None,
            "primary_objective_max_min_spread": round(primary_disc.max_min_spread, 6) if primary_disc else None,
        }
        for k, v in window.items():
            if k == "requests":
                continue
            row[f"param_{k}"] = v
        window_rows.append(row)

        for obj_name, d in disc_by_objective.items():
            if d is None:
                continue
            disc_rows.append(dict(window_idx=idx, shape=shape_name, window_classification=classification, **asdict(d)))

        # Persist raw latencies only for mechanism-active windows (Stage A
        # candidates) -- keeps output bounded.
        if classification in ("CONTENTION_VISIBLE", "SATURATED", "PATHOLOGICAL_OVERLOAD"):
            raw_path = out_dir / "raw_latencies" / f"window_{idx}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps({
                pname: lats for pname, lats in result["raw_latencies"].items()
            }, indent=2))

    elapsed = time.time() - t0

    windows_csv = out_dir / "frontier_windows.csv"
    with open(windows_csv, "w", newline="") as f:
        fieldnames = sorted({k for row in window_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in window_rows:
            w.writerow(row)

    disc_csv = out_dir / "frontier_discriminativeness.csv"
    with open(disc_csv, "w", newline="") as f:
        fieldnames = sorted({k for row in disc_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in disc_rows:
            w.writerow(row)

    from collections import Counter
    classification_counts = Counter(r["classification"] for r in window_rows)
    primary_class_counts = Counter(r["primary_objective_classification"] for r in window_rows)
    win_counts = Counter(r["primary_objective_best_policy"] for r in window_rows if r["primary_objective_best_policy"])

    summary = {
        "n_windows_attempted": len(generators),
        "n_windows_scored": len(window_rows),
        "elapsed_s": round(elapsed, 1),
        "classification_counts": dict(classification_counts),
        "primary_objective_classification_counts": dict(primary_class_counts),
        "primary_objective_win_distribution": dict(win_counts),
        "windows_csv": str(windows_csv.relative_to(ROOT)),
        "discriminativeness_csv": str(disc_csv.relative_to(ROOT)),
    }
    (out_dir / "frontier_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
