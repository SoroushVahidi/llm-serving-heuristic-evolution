#!/usr/bin/env python3
"""Faithful-baseline participation/loss/fairness audit (Selector v2).

Reproduces the exact 910-window SLO-calibrated search
(selector_v2_slo_calibrated_frontier_search.py --stage main) but records
the FULL per-policy outcome vector (not just best/second-best per
objective) plus explicit exception tracking, so the following can be
answered directly from data rather than re-derived from summaries:

* execution health: did every policy actually run, on every window,
  without a silently-swallowed exception?
* loss decomposition: on windows a faithful policy lost, which OTHER
  metrics (completion, rejection, SLO attainment, latency components)
  actually differed, and by how much?
* admission-control fairness: recomputing discriminativeness restricted
  to the 9 policies that never voluntarily reject a feasible-to-admit
  request (excluding admission_control/scorpio_style_slo_guard, the only
  two with laxity-based rejection) isolates how much of the historical-
  policy advantage is rejection vs. pure scheduling order.

Deterministic and window-index-compatible with the prior task's
slo_calibrated_windows.csv (same seed, same family-cycling order, same
real-trace loading) -- window_idx here matches that CSV's window_idx.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import GPUConfig  # noqa: E402
from llmserveopt.policies.registry import make_policy  # noqa: E402
from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy  # noqa: E402
from llmserveopt.policies.vllm_chunked_prefill_faithful import VLLMChunkedPrefillFaithfulPolicy  # noqa: E402
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy  # noqa: E402
from llmserveopt.selector.dataset_v2.builder import metrics_to_outcome_vector  # noqa: E402
from llmserveopt.selector.dataset_v2.discriminativeness import (  # noqa: E402
    STANDARD_OBJECTIVES, PRIMARY_SELECTOR_OBJECTIVE, compute_discriminativeness,
)
from llmserveopt.selector.dataset_v2.frontier_workload_families import FAMILY_GENERATORS  # noqa: E402
from llmserveopt.selector.dataset_v2.scenario_redesign import (  # noqa: E402
    local_real_trace_stress_specs, service_model as _default_service_model,
)
from llmserveopt.selector.dataset_v2.slo_calibration import calibrate_window_e2e  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

ADMIT_CHUNK = 100_000
FAITHFUL_POLICIES = ["vllm_faithful", "sarathi_faithful", "vllm_chunked_prefill_faithful"]
CHEAP_HISTORICAL = ["fifo", "edf", "scorpio_style_slo_guard", "admission_control",
                     "weighted_shortest_processing", "estimated_service_time_first",
                     "best_fit", "multi_bin_batching"]
ALL_POLICIES = FAITHFUL_POLICIES + CHEAP_HISTORICAL
# The only two historical policies with laxity-based VOLUNTARY rejection
# (skip a feasible-to-admit request because it is already SLO-hopeless).
# Verified by source inspection: admission_control.py / scorpio_style_
# slo_guard.py both filter the waiting queue by a laxity threshold before
# scheduling; edf/weighted_shortest_processing/estimated_service_time_
# first/fifo/best_fit/multi_bin_batching only ever REORDER the queue and
# admit greedily whenever `_feasible_on_gpu` -- see docs/selector_v2_
# faithful_baseline_scope_audit.md section on admission-control fairness.
REJECTING_POLICIES = {"admission_control", "scorpio_style_slo_guard"}
NON_REJECTING_POLICIES = [p for p in ALL_POLICIES if p not in REJECTING_POLICIES]

METRIC_FIELDS = [
    "arrival_normalized_weighted_goodput", "weighted_goodput", "completion_fraction",
    "rejection_fraction", "slo_attainment", "mean_latency", "mean_ttft", "mean_tpot",
    "request_throughput",
]


def _make_policy(name: str):
    if name == "sarathi_faithful":
        return SarathiFaithfulPolicy(chunk_size=ADMIT_CHUNK)
    if name == "vllm_chunked_prefill_faithful":
        return VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=ADMIT_CHUNK)
    if name == "vllm_faithful":
        return VLLMFaithfulPolicy(max_num_batched_tokens=ADMIT_CHUNK)
    return make_policy(name)


def _make_policy_real_defaults(name: str):
    """Every prior task in this thread (starting with the original
    contention-validation pilot) constructed the three faithful policies
    with `max_num_batched_tokens=ADMIT_CHUNK=100_000` -- a deliberate
    override, DOCUMENTED at the time, to "decouple policy-level admission
    chunking from execution-level contention" for THAT investigation. It
    has been silently inherited by every subsequent task, including the
    910-window SLO-calibrated search. This means `vllm_chunked_prefill_
    faithful`'s actual distinguishing admission mechanism (chunked
    admission, real default `max_num_batched_tokens=512`, vs.
    `vllm_faithful`'s real default all-or-nothing `2560`) was disabled in
    every window that search ever ran. This function uses each policy's
    TRUE real-world default instead, for the pairwise specialization
    search only (never for the health/decomposition reproduction stages
    above, which must stay bit-identical to the prior task's own search)."""
    if name == "sarathi_faithful":
        return SarathiFaithfulPolicy()  # real default chunk_size=512
    if name == "vllm_chunked_prefill_faithful":
        return VLLMChunkedPrefillFaithfulPolicy()  # real default max_num_batched_tokens=512
    if name == "vllm_faithful":
        return VLLMFaithfulPolicy()  # real default max_num_batched_tokens=2560
    return make_policy(name)


def _service_model_for_policy(policy_name: str, budget: int, chunk: int) -> ServiceModel:
    decode_first = policy_name != "vllm_chunked_prefill_faithful"
    return ServiceModel(
        enable_prefill_modeling=True, decode_first=decode_first,
        enable_decode_prefill_contention=True,
        step_token_budget=budget, max_prefill_chunk_tokens=chunk,
    )


def _reproduce_windows(search_seed: int, n_synthetic: int, n_real_trace_seeds: int) -> List[Dict]:
    """Bit-identical to stage_main's window construction in
    selector_v2_slo_calibrated_frontier_search.py."""
    rng = random.Random(search_seed)
    family_names = list(FAMILY_GENERATORS.keys())
    synthetic_windows = []
    for i in range(n_synthetic):
        fname = family_names[i % len(family_names)]
        w = FAMILY_GENERATORS[fname](rng)
        w["family_id"] = fname
        w["already_calibrated"] = False
        synthetic_windows.append(w)

    specs = local_real_trace_stress_specs(ROOT, max_requests=48)
    real_trace_windows = []
    for spec in specs:
        for seed in range(n_real_trace_seeds):
            try:
                reqs = spec.build(seed)
            except Exception:
                continue
            if not reqs:
                continue
            real_trace_windows.append(dict(
                shape=f"real_trace_stress__{spec.family_id}", requests=reqs,
                budget=spec.service_model.step_token_budget,
                chunk=spec.service_model.max_prefill_chunk_tokens,
                max_kv_tokens=spec.gpu_configs[0].max_kv_tokens,
                max_active_sequences=spec.gpu_configs[0].max_active_sequences,
                already_calibrated=True, source_trace=spec.source_trace, family_id=spec.family_id,
            ))
    return synthetic_windows + real_trace_windows


def _run_window_full(window: Dict, multiplier: float, search_seed: int, idx: int) -> Dict:
    """Runs every policy once, returning {policy_name: {"status": "ok"/"error",
    "outcome": PolicyOutcomeVector or None, "error": str or None,
    "diagnostics": dict}}. Exceptions are RECORDED, never silently swallowed."""
    if window.get("already_calibrated"):
        requests = window["requests"]
    else:
        sm_ref = _default_service_model(prefill=True, step_token_budget=window["budget"],
                                          max_prefill_chunk_tokens=window["chunk"])
        requests = calibrate_window_e2e(window["requests"], sm_ref, multiplier)

    gpu_configs = [GPUConfig(0, max_active_sequences=window.get("max_active_sequences", 64),
                              max_batch_tokens=1_000_000, max_kv_tokens=window.get("max_kv_tokens", 200_000))]
    results: Dict[str, Dict] = {}
    for pname in ALL_POLICIES:
        sm = _service_model_for_policy(pname, window["budget"], window["chunk"])
        try:
            policy = _make_policy(pname)
            policy.reset()
            sim = Simulator(SimulatorConfig(gpu_configs=gpu_configs, service_model=sm, drain_steps=5_000))
            sim.load_trace(list(requests))
            m = sim.run(policy=policy, workload_tag=f"audit_{idx}", seed=search_seed + idx)
            outcome = metrics_to_outcome_vector(pname, m, {"admit": 0, "preempt": 0, "swap": 0, "migrate": 0},
                                                  gpu_count=1)
            results[pname] = dict(status="ok", outcome=outcome, error=None,
                                   diagnostics=sim.contention_diagnostics_summary())
        except Exception as exc:
            results[pname] = dict(status="error", outcome=None,
                                    error=f"{type(exc).__name__}: {exc}", diagnostics=None)
    return results


# ---------------------------------------------------------------------------
# Stage: health (execution-health audit, task 18)
# ---------------------------------------------------------------------------

def stage_health(args) -> None:
    windows = _reproduce_windows(args.search_seed, args.n_synthetic, args.n_real_trace_seeds)
    per_policy_ok = {p: 0 for p in ALL_POLICIES}
    per_policy_errors: Dict[str, List[str]] = {p: [] for p in ALL_POLICIES}
    n_windows = min(len(windows), args.max_windows) if args.max_windows else len(windows)
    for idx in range(n_windows):
        results = _run_window_full(windows[idx], args.multiplier, args.search_seed, idx)
        for pname, r in results.items():
            if r["status"] == "ok":
                per_policy_ok[pname] += 1
            else:
                per_policy_errors[pname].append(f"window {idx}: {r['error']}")

    summary = {
        "n_windows_checked": n_windows,
        "per_policy_ok_count": per_policy_ok,
        "per_policy_ok_fraction": {p: round(v / n_windows, 4) for p, v in per_policy_ok.items()},
        "per_policy_error_samples": {p: errs[:5] for p, errs in per_policy_errors.items() if errs},
        "per_policy_error_count": {p: len(errs) for p, errs in per_policy_errors.items()},
    }
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "execution_health.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Stage: decompose (loss decomposition + restricted non-rejecting comparison, tasks 3+6)
# ---------------------------------------------------------------------------

def stage_decompose(args) -> None:
    windows = _reproduce_windows(args.search_seed, args.n_synthetic, args.n_real_trace_seeds)
    # Load the strongly/moderately-discriminative window indices from the prior task's CSV.
    prior_csv = ROOT / "experiments/selector_v2_slo_calibrated_frontier_search/slo_calibrated_windows.csv"
    target_indices = []
    with open(prior_csv) as f:
        for row in csv.DictReader(f):
            if row["primary_objective_classification"] in ("STRONGLY_DISCRIMINATIVE", "MODERATELY_DISCRIMINATIVE"):
                target_indices.append(int(row["window_idx"]))

    primary_obj = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)
    decomposition_rows = []
    restricted_disc_rows = []
    for idx in target_indices:
        if idx >= len(windows):
            continue
        results = _run_window_full(windows[idx], args.multiplier, args.search_seed, idx)
        ok_outcomes = {p: r["outcome"] for p, r in results.items() if r["status"] == "ok"}
        if len(ok_outcomes) < 2:
            continue

        full_disc = compute_discriminativeness(list(ok_outcomes.values()), primary_obj)
        if full_disc is None:
            continue
        winner = full_disc.best_policy

        for fpname in FAITHFUL_POLICIES:
            if fpname not in ok_outcomes or winner not in ok_outcomes:
                continue
            f_out, w_out = ok_outcomes[fpname], ok_outcomes[winner]
            row = dict(window_idx=idx, shape=windows[idx].get("shape"), winner=winner,
                        faithful_policy=fpname, winner_is_rejecting=winner in REJECTING_POLICIES)
            for metric in METRIC_FIELDS:
                row[f"winner_{metric}"] = getattr(w_out, metric, None)
                row[f"{fpname}_{metric}"] = getattr(f_out, metric, None)
            decomposition_rows.append(row)

        # Restricted non-rejecting-only discriminativeness (task 6).
        restricted_outcomes = [o for p, o in ok_outcomes.items() if p in NON_REJECTING_POLICIES]
        if len(restricted_outcomes) >= 2:
            rd = compute_discriminativeness(restricted_outcomes, primary_obj)
            if rd is not None:
                restricted_disc_rows.append(dict(
                    window_idx=idx, shape=windows[idx].get("shape"),
                    full_best_policy=winner, full_classification=full_disc.classification,
                    restricted_best_policy=rd.best_policy, restricted_classification=rd.classification,
                    restricted_max_min_spread=round(rd.max_min_spread, 6),
                ))

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    decomp_csv = out_dir / "faithful_loss_decomposition.csv"
    with open(decomp_csv, "w", newline="") as f:
        fieldnames = sorted({k for row in decomposition_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(decomposition_rows)
    restricted_csv = out_dir / "restricted_non_rejecting_discriminativeness.csv"
    with open(restricted_csv, "w", newline="") as f:
        fieldnames = sorted({k for row in restricted_disc_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(restricted_disc_rows)

    from collections import Counter
    winner_rejecting_counts = Counter(
        "rejecting" if any(r["winner_is_rejecting"] for r in decomposition_rows if r["window_idx"] == idx)
        else "non_rejecting"
        for idx in set(r["window_idx"] for r in decomposition_rows)
    )
    restricted_win_counts = Counter(r["restricted_best_policy"] for r in restricted_disc_rows)
    restricted_class_counts = Counter(r["restricted_classification"] for r in restricted_disc_rows)
    faithful_restricted_wins = {
        p: restricted_win_counts.get(p, 0) for p in FAITHFUL_POLICIES
    }

    summary = {
        "n_discriminative_windows_analyzed": len(set(r["window_idx"] for r in decomposition_rows)),
        "winner_rejecting_vs_non_rejecting": dict(winner_rejecting_counts),
        "restricted_non_rejecting_win_distribution": dict(restricted_win_counts),
        "restricted_non_rejecting_classification_counts": dict(restricted_class_counts),
        "faithful_wins_in_restricted_comparison": faithful_restricted_wins,
        "decomposition_csv": str(decomp_csv.relative_to(ROOT)),
        "restricted_csv": str(restricted_csv.relative_to(ROOT)),
    }
    (out_dir / "loss_decomposition_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Stage: pairwise (targeted specialization search, task 4/20, using each
# faithful policy's REAL default admission parameters -- not ADMIT_CHUNK)
# ---------------------------------------------------------------------------

import random  # noqa: E402
from llmserveopt.core.types import Request  # noqa: E402

RIVALS = ["weighted_shortest_processing", "edf", "scorpio_style_slo_guard", "admission_control"]


def _service_model_real(policy_name: str, budget: int, chunk: int) -> ServiceModel:
    decode_first = policy_name != "vllm_chunked_prefill_faithful"
    return ServiceModel(enable_prefill_modeling=True, decode_first=decode_first,
                         enable_decode_prefill_contention=True,
                         step_token_budget=budget, max_prefill_chunk_tokens=chunk)


def _run_pairwise_window(requests, budget, chunk, max_kv_tokens, max_active_sequences,
                           candidates, search_seed, idx):
    gpu_configs = [GPUConfig(0, max_active_sequences=max_active_sequences,
                              max_batch_tokens=1_000_000, max_kv_tokens=max_kv_tokens)]
    outcomes = []
    for pname in candidates:
        sm = _service_model_real(pname, budget, chunk)
        try:
            policy = _make_policy_real_defaults(pname)
            policy.reset()
            sim = Simulator(SimulatorConfig(gpu_configs=gpu_configs, service_model=sm, drain_steps=5_000))
            sim.load_trace(list(requests))
            m = sim.run(policy=policy, workload_tag=f"pairwise_{idx}", seed=search_seed + idx)
            outcomes.append(metrics_to_outcome_vector(pname, m, {"admit": 0, "preempt": 0, "swap": 0, "migrate": 0},
                                                        gpu_count=1))
        except Exception:
            continue
    return outcomes


def _long_context_burst_window(rng: random.Random) -> Dict:
    """Targets vllm_chunked_prefill_faithful: prompts well beyond
    vllm_faithful's real all-or-nothing admission budget (2560 tokens),
    all arriving in a burst -- exactly the runtime_validation_benchmark_
    pack's already-documented xlong_context_burst16 acceptance shape,
    generalized/randomized rather than copied verbatim."""
    n = rng.randint(4, 16)
    prompt = rng.randint(3000, 12000)  # > 2560 whole-prompt budget
    output = rng.choice([64, 128, 256, 512])
    reqs = [Request(request_id=i, arrival_time=0.0, prompt_tokens=prompt, predicted_output_tokens=output,
                      actual_output_tokens=output, slo_deadline=1.0, priority=1.0, class_id="long_ctx")
            for i in range(n)]
    return dict(requests=reqs, budget=4096, chunk=512, max_kv_tokens=n * prompt * 2, max_active_sequences=64)


def _decode_prefill_overlap_window(rng: random.Random) -> Dict:
    """Targets sarathi_faithful: an already-decoding cohort plus a
    fresh, moderately-long-prefill arrival shortly after -- prefill/
    decode overlap, evaluated against a TIGHT TPOT-oriented dual SLO
    rather than the single E2E deadline (sarathi's decode-protection
    principle is specifically a TPOT/stall-avoidance guarantee, which a
    single E2E deadline dilutes)."""
    n_decoding = rng.randint(4, 16)
    decode_output = rng.randint(20, 60)
    prefill_prompt = rng.randint(1000, 4000)
    reqs = [Request(request_id=i, arrival_time=0.0, prompt_tokens=rng.randint(50, 200),
                      predicted_output_tokens=decode_output, actual_output_tokens=decode_output,
                      slo_deadline=1.0, priority=1.0, class_id="decoding")
            for i in range(n_decoding)]
    reqs.append(Request(request_id=n_decoding, arrival_time=0.002, prompt_tokens=prefill_prompt,
                          predicted_output_tokens=1, actual_output_tokens=1,
                          slo_deadline=1.0, priority=1.0, class_id="arriving_prefill"))
    return dict(requests=reqs, budget=768, chunk=512, max_kv_tokens=200_000, max_active_sequences=64)


def _fcfs_friendly_window(rng: random.Random) -> Dict:
    """Targets vllm_faithful: low-heterogeneity, similar-size, similar-
    urgency requests -- the regime where reordering has nothing to
    exploit (FCFS should be indistinguishable from EDF/WSP/SCORPIO
    absent any admission-control benefit) and a moderate (not extreme)
    SLO multiplier, since the 4-2.0x grid already showed 2.0x is a
    high-pressure regime that favors deadline-aware policies."""
    n = rng.randint(4, 12)
    base_prompt = rng.randint(200, 2000)
    base_output = rng.randint(10, 60)
    reqs = [Request(request_id=i, arrival_time=0.0,
                      prompt_tokens=base_prompt + rng.randint(-20, 20),
                      predicted_output_tokens=base_output + rng.randint(-2, 2),
                      actual_output_tokens=base_output + rng.randint(-2, 2),
                      slo_deadline=1.0, priority=1.0, class_id="homogeneous")
            for i in range(n)]
    return dict(requests=reqs, budget=1024, chunk=512, max_kv_tokens=200_000, max_active_sequences=64)


PAIRWISE_TARGETS = {
    "vllm_chunked_prefill_faithful": (_long_context_burst_window, ["vllm_chunked_prefill_faithful"] + RIVALS),
    "sarathi_faithful": (_decode_prefill_overlap_window, ["sarathi_faithful"] + RIVALS),
    "vllm_faithful": (_fcfs_friendly_window, ["vllm_faithful"] + RIVALS),
}


def stage_pairwise(args) -> None:
    rng = random.Random(args.search_seed)
    primary_obj = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)
    from llmserveopt.selector.dataset_v2.slo_calibration import calibrate_window_e2e as _calib

    from collections import Counter
    results_by_target = {}
    for target_policy, (gen, candidates) in PAIRWISE_TARGETS.items():
        win_count = 0
        strong_or_moderate_win_count = 0
        n_scored = 0
        example_wins = []
        classification_counts = Counter()
        win_classification_counts = Counter()
        for i in range(args.n_per_target):
            w = gen(rng)
            sm_ref = _default_service_model(prefill=True, step_token_budget=w["budget"],
                                              max_prefill_chunk_tokens=w["chunk"])
            calibrated = _calib(w["requests"], sm_ref, args.multiplier)
            outcomes = _run_pairwise_window(calibrated, w["budget"], w["chunk"], w["max_kv_tokens"],
                                              w["max_active_sequences"], candidates, args.search_seed, i)
            if len(outcomes) < 2:
                continue
            n_scored += 1
            d = compute_discriminativeness(outcomes, primary_obj)
            if d is None:
                continue
            classification_counts[d.classification] += 1
            if d.best_policy == target_policy:
                win_count += 1
                win_classification_counts[d.classification] += 1
                if d.classification in ("STRONGLY_DISCRIMINATIVE", "MODERATELY_DISCRIMINATIVE"):
                    strong_or_moderate_win_count += 1
                    if len(example_wins) < 3:
                        example_wins.append(dict(window_idx=i, best_value=d.best_value,
                                                   second_best_policy=d.second_best_policy,
                                                   second_best_value=d.second_best_value))
        results_by_target[target_policy] = dict(
            n_scored=n_scored, win_count=win_count,
            strong_or_moderate_win_count=strong_or_moderate_win_count,
            win_fraction=round(win_count / n_scored, 4) if n_scored else None,
            classification_counts=dict(classification_counts),
            win_classification_counts=dict(win_classification_counts),
            example_strong_or_moderate_wins=example_wins,
        )

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairwise_specialization_search.json").write_text(json.dumps(results_by_target, indent=2))
    print(json.dumps(results_by_target, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["health", "decompose", "pairwise"], required=True)
    parser.add_argument("--search-seed", type=int, default=20260720)
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--n-synthetic", type=int, default=750)
    parser.add_argument("--n-real-trace-seeds", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--n-per-target", type=int, default=150)
    parser.add_argument("--output-dir", default="experiments/selector_v2_faithful_baseline_scope_audit")
    args = parser.parse_args()

    if args.stage == "health":
        stage_health(args)
    elif args.stage == "decompose":
        stage_decompose(args)
    else:
        stage_pairwise(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
