#!/usr/bin/env python3
"""SLO-calibrated frontier search (Selector v2, corrected-objective
follow-up to selector_v2_contention_frontier_search.py).

Three stages, run separately via --stage:

* `grid`: bounded diagnostic multiplier-grid sweep (section 4) over a
  small window sample, families A-E only (synthetic; real-trace loading
  is comparatively expensive and not needed to pick a default
  multiplier). Reports, per multiplier, all-success/all-fail fraction,
  ANWG/SLO discriminative fraction, oracle headroom, win diversity,
  completion fraction -- used to justify (not hand-pick) a default.
* `main`: the corrected targeted search (section 6) at ONE selected
  multiplier, across all 6 families (A-E synthetic + F real-trace stress,
  the latter via `scenario_redesign.local_real_trace_stress_specs`,
  unmodified, which already preserves BurstGPT/Azure provenance).
* `robustness`: SLO-scale-robustness check (section 8) -- re-evaluates
  every window found `main`-discriminative at the multiplier's immediate
  grid neighbors, classifying ROBUST_TO_SLO_SCALE / SENSITIVE_TO_SLO_SCALE
  / ARTIFACT_OF_THRESHOLD.

No calibration parameter here is ever chosen by which policy wins --
see docs/selector_v2_slo_calibrated_frontier_search.md for the full
justification and results.
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
from llmserveopt.selector.dataset_v2.slo_calibration import (  # noqa: E402
    CALIBRATION_MULTIPLIER_GRID, SLO_CALIBRATION_SCHEMA_VERSION, calibrate_window_e2e,
)
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

ADMIT_CHUNK = 100_000
FAITHFUL_POLICIES = ["vllm_faithful", "sarathi_faithful", "vllm_chunked_prefill_faithful"]
CHEAP_HISTORICAL = ["fifo", "edf", "scorpio_style_slo_guard", "admission_control",
                     "weighted_shortest_processing", "estimated_service_time_first",
                     "best_fit", "multi_bin_batching"]
ALL_POLICIES = FAITHFUL_POLICIES + CHEAP_HISTORICAL
PRACTICAL_EQUIVALENCE_ABS = 0.002


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


def _run_window_at_multiplier(window: Dict, multiplier: float, seed: int, window_idx: int) -> Optional[Dict]:
    """Runs every policy once on `window`'s requests, calibrated at
    `multiplier`, returning outcome vectors + diagnostics summaries."""
    gpu_configs = [GPUConfig(0, max_active_sequences=window.get("max_active_sequences", 64),
                              max_batch_tokens=1_000_000, max_kv_tokens=window.get("max_kv_tokens", 200_000))]
    outcomes = []
    diagnostics_by_policy: Dict[str, Dict] = {}
    for pname in ALL_POLICIES:
        sm = _service_model_for_policy(pname, window["budget"], window["chunk"])
        calibrated_requests = calibrate_window_e2e(window["requests"], sm, multiplier)
        try:
            policy = _make_policy(pname)
            policy.reset()
        except Exception:
            continue
        sim = Simulator(SimulatorConfig(gpu_configs=gpu_configs, service_model=sm, drain_steps=5_000))
        sim.load_trace(list(calibrated_requests))
        try:
            m = sim.run(policy=policy, workload_tag=f"slo_frontier_{window_idx}", seed=seed + window_idx)
        except Exception:
            continue
        outcomes.append(metrics_to_outcome_vector(pname, m, {"admit": 0, "preempt": 0, "swap": 0, "migrate": 0},
                                                    gpu_count=1))
        diagnostics_by_policy[pname] = sim.contention_diagnostics_summary()
    if len(outcomes) < 2:
        return None
    return dict(outcomes=outcomes, diagnostics=diagnostics_by_policy)


# ---------------------------------------------------------------------------
# Stage: grid
# ---------------------------------------------------------------------------

def stage_grid(args) -> None:
    rng = random.Random(args.search_seed)
    family_names = list(FAMILY_GENERATORS.keys())
    windows = []
    for i in range(args.grid_n_windows):
        fname = family_names[i % len(family_names)]
        windows.append((fname, FAMILY_GENERATORS[fname](rng)))

    rows = []
    primary_obj = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)
    for multiplier in CALIBRATION_MULTIPLIER_GRID:
        n_all_success = n_all_fail = n_anwg_disc = 0
        win_counts: Dict[str, int] = {}
        values_by_policy: Dict[str, List[float]] = {}
        n_scored = 0
        for idx, (fname, window) in enumerate(windows):
            result = _run_window_at_multiplier(window, multiplier, args.search_seed, idx)
            if result is None:
                continue
            n_scored += 1
            anwg_vals = [o.arrival_normalized_weighted_goodput for o in result["outcomes"]
                          if o.arrival_normalized_weighted_goodput is not None]
            if anwg_vals and min(anwg_vals) >= 0.999:
                n_all_success += 1
            if anwg_vals and max(anwg_vals) <= 0.001:
                n_all_fail += 1
            d = compute_discriminativeness(result["outcomes"], primary_obj)
            if d is not None:
                if d.classification != "ALL_COMPLETE_OR_EFFECTIVELY_TIED":
                    n_anwg_disc += 1
                win_counts[d.best_policy] = win_counts.get(d.best_policy, 0) + 1
            for o in result["outcomes"]:
                v = primary_obj.extractor(o)
                if v is not None:
                    values_by_policy.setdefault(o.policy_name, []).append(v)
        best_fixed_name, best_fixed_mean = None, None
        for name, vals in values_by_policy.items():
            m = sum(vals) / len(vals)
            if best_fixed_mean is None or m > best_fixed_mean:
                best_fixed_name, best_fixed_mean = name, m
        rows.append(dict(
            multiplier=multiplier, n_scored=n_scored,
            all_success_fraction=round(n_all_success / n_scored, 4) if n_scored else None,
            all_fail_fraction=round(n_all_fail / n_scored, 4) if n_scored else None,
            anwg_discriminative_fraction=round(n_anwg_disc / n_scored, 4) if n_scored else None,
            n_distinct_winners=len(win_counts), best_fixed_policy=best_fixed_name,
            best_fixed_mean_anwg=round(best_fixed_mean, 4) if best_fixed_mean is not None else None,
        ))

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "calibration_multiplier_grid.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))

    # Select default per section 4: avoid universal success/failure, then
    # maximize ANWG-discriminative fraction as secondary tiebreak.
    candidates = [r for r in rows if r["all_success_fraction"] is not None
                  and r["all_success_fraction"] < 0.95 and r["all_fail_fraction"] < 0.5]
    if not candidates:
        candidates = rows
    chosen = max(candidates, key=lambda r: r["anwg_discriminative_fraction"] or 0.0)
    (out_dir / "calibration_grid_selection.json").write_text(json.dumps({
        "chosen_multiplier": chosen["multiplier"],
        "reason": "avoids universal success/failure; maximizes ANWG-discriminative fraction among "
                  "the multipliers that do so -- never selected by which policy wins at that value",
        "all_rows": rows,
    }, indent=2))
    print(f"\nSelected default multiplier: {chosen['multiplier']}")


# ---------------------------------------------------------------------------
# Stage: main
# ---------------------------------------------------------------------------

def _load_real_trace_windows(max_windows_per_spec: int, n_seeds: int) -> List[Dict]:
    specs = local_real_trace_stress_specs(ROOT, max_requests=48)
    windows = []
    for spec in specs:
        for seed in range(n_seeds):
            try:
                reqs = spec.build(seed)
            except Exception:
                continue
            if not reqs:
                continue
            windows.append(dict(
                shape=f"real_trace_stress__{spec.family_id}", requests=reqs,
                budget=spec.service_model.step_token_budget,
                chunk=spec.service_model.max_prefill_chunk_tokens,
                max_kv_tokens=spec.gpu_configs[0].max_kv_tokens,
                max_active_sequences=spec.gpu_configs[0].max_active_sequences,
                already_calibrated=True,  # transform_requests already set slo_deadline via slo_scale
                source_trace=spec.source_trace, family_id=spec.family_id,
            ))
            if len(windows) >= max_windows_per_spec * len(specs):
                break
    return windows


def stage_main(args) -> None:
    rng = random.Random(args.search_seed)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    family_names = list(FAMILY_GENERATORS.keys())
    synthetic_windows = []
    for i in range(args.n_synthetic):
        fname = family_names[i % len(family_names)]
        w = FAMILY_GENERATORS[fname](rng)
        w["family_id"] = fname
        w["already_calibrated"] = False
        synthetic_windows.append(w)

    real_trace_windows = _load_real_trace_windows(max_windows_per_spec=args.n_real_trace_seeds,
                                                    n_seeds=args.n_real_trace_seeds)
    all_windows = synthetic_windows + real_trace_windows

    primary_obj = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)
    window_rows = []
    disc_rows = []
    per_window_anwg: List[Dict[str, float]] = []
    t0 = time.time()
    for idx, window in enumerate(all_windows):
        if window.get("already_calibrated"):
            requests = window["requests"]
            gpu_configs_kw = window
        else:
            sm_ref = _default_service_model(prefill=True, step_token_budget=window["budget"],
                                              max_prefill_chunk_tokens=window["chunk"])
            requests = calibrate_window_e2e(window["requests"], sm_ref, args.multiplier)
            gpu_configs_kw = window
        window_for_run = dict(window)
        window_for_run["requests"] = requests
        gpu_configs = [GPUConfig(0, max_active_sequences=window_for_run.get("max_active_sequences", 64),
                                  max_batch_tokens=1_000_000,
                                  max_kv_tokens=window_for_run.get("max_kv_tokens", 200_000))]
        outcomes = []
        diagnostics_by_policy: Dict[str, Dict] = {}
        for pname in ALL_POLICIES:
            sm = _service_model_for_policy(pname, window_for_run["budget"], window_for_run["chunk"])
            try:
                policy = _make_policy(pname)
                policy.reset()
            except Exception:
                continue
            sim = Simulator(SimulatorConfig(gpu_configs=gpu_configs, service_model=sm, drain_steps=5_000))
            sim.load_trace(list(requests))
            try:
                m = sim.run(policy=policy, workload_tag=f"slo_main_{idx}", seed=args.search_seed + idx)
            except Exception:
                continue
            outcomes.append(metrics_to_outcome_vector(pname, m, {"admit": 0, "preempt": 0, "swap": 0, "migrate": 0},
                                                        gpu_count=1))
            diagnostics_by_policy[pname] = sim.contention_diagnostics_summary()
        if len(outcomes) < 2:
            continue

        disc_by_objective = {obj.name: compute_discriminativeness(outcomes, obj) for obj in STANDARD_OBJECTIVES}
        primary_disc = disc_by_objective.get(PRIMARY_SELECTOR_OBJECTIVE)
        diag = diagnostics_by_policy.get("vllm_chunked_prefill_faithful", {})
        row = dict(
            window_idx=idx, shape=window.get("shape"), family_id=window.get("family_id"),
            source_trace=window.get("source_trace"),
            slo_schema_version=SLO_CALIBRATION_SCHEMA_VERSION,
            calibration_multiplier=(None if window.get("already_calibrated") else args.multiplier),
            n_requests=len(requests),
            decode_stalled_steps=diag.get("decode_stalled_steps", 0),
            prefill_stalled_steps=diag.get("prefill_stalled_steps", 0),
            budget_saturation_fraction=round(diag.get("budget_saturation_fraction", 0.0), 4),
            primary_objective_classification=primary_disc.classification if primary_disc else None,
            primary_objective_best_policy=primary_disc.best_policy if primary_disc else None,
            primary_objective_max_min_spread=round(primary_disc.max_min_spread, 6) if primary_disc else None,
        )
        window_rows.append(row)
        for obj_name, d in disc_by_objective.items():
            if d is None:
                continue
            disc_rows.append(dict(window_idx=idx, shape=window.get("shape"), **asdict(d)))
        per_window_anwg.append({
            o.policy_name: o.arrival_normalized_weighted_goodput for o in outcomes
            if o.arrival_normalized_weighted_goodput is not None
        })

    elapsed = time.time() - t0
    windows_csv = out_dir / "slo_calibrated_windows.csv"
    with open(windows_csv, "w", newline="") as f:
        fieldnames = sorted({k for row in window_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(window_rows)
    disc_csv = out_dir / "slo_calibrated_discriminativeness.csv"
    with open(disc_csv, "w", newline="") as f:
        fieldnames = sorted({k for row in disc_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(disc_rows)

    from collections import Counter
    primary_class_counts = Counter(r["primary_objective_classification"] for r in window_rows)
    win_counts = Counter(r["primary_objective_best_policy"] for r in window_rows if r["primary_objective_best_policy"])
    n_windows = len(window_rows)
    all_equiv = primary_class_counts.get("ALL_COMPLETE_OR_EFFECTIVELY_TIED", 0)
    strong = primary_class_counts.get("STRONGLY_DISCRIMINATIVE", 0)
    moderate = primary_class_counts.get("MODERATELY_DISCRIMINATIVE", 0)

    values_by_policy: Dict[str, List[float]] = {}
    for w_anwg in per_window_anwg:
        for pname, v in w_anwg.items():
            values_by_policy.setdefault(pname, []).append(v)
    best_fixed_name, best_fixed_mean = None, None
    for name, vals in values_by_policy.items():
        mean_v = sum(vals) / len(vals)
        if best_fixed_mean is None or mean_v > best_fixed_mean:
            best_fixed_name, best_fixed_mean = name, mean_v

    oracle_vals, best_fixed_vals = [], []
    disc_oracle_vals, disc_best_fixed_vals = [], []
    for w_anwg, row in zip(per_window_anwg, window_rows):
        if not w_anwg or best_fixed_name not in w_anwg:
            continue
        oracle_vals.append(max(w_anwg.values()))
        best_fixed_vals.append(w_anwg[best_fixed_name])
        if row["primary_objective_classification"] in ("STRONGLY_DISCRIMINATIVE", "MODERATELY_DISCRIMINATIVE"):
            disc_oracle_vals.append(max(w_anwg.values()))
            disc_best_fixed_vals.append(w_anwg[best_fixed_name])
    oracle_headroom = (
        (sum(oracle_vals) / len(oracle_vals)) - (sum(best_fixed_vals) / len(best_fixed_vals))
        if oracle_vals else None
    )
    discriminative_oracle_headroom = (
        (sum(disc_oracle_vals) / len(disc_oracle_vals)) - (sum(disc_best_fixed_vals) / len(disc_best_fixed_vals))
        if disc_oracle_vals else None
    )
    strong_win_counts = Counter(r["primary_objective_best_policy"] for r in window_rows
                                  if r["primary_objective_classification"] == "STRONGLY_DISCRIMINATIVE")
    top_strong_share = (
        max(strong_win_counts.values()) / sum(strong_win_counts.values()) if strong_win_counts else 0.0
    )

    summary = {
        "n_windows_attempted": len(all_windows), "n_windows_scored": n_windows,
        "n_synthetic": len(synthetic_windows), "n_real_trace": len(real_trace_windows),
        "multiplier_used_for_synthetic": args.multiplier,
        "elapsed_s": round(elapsed, 1),
        "primary_objective_classification_counts": dict(primary_class_counts),
        "all_equivalent_fraction": round(all_equiv / n_windows, 4) if n_windows else None,
        "strongly_discriminative_fraction": round(strong / n_windows, 4) if n_windows else None,
        "moderately_discriminative_fraction": round(moderate / n_windows, 4) if n_windows else None,
        "primary_objective_win_distribution": dict(win_counts),
        "strong_win_distribution": dict(strong_win_counts),
        "top_policy_strong_win_share": round(top_strong_share, 4),
        "best_fixed_policy": best_fixed_name,
        "best_fixed_policy_mean_anwg": round(best_fixed_mean, 4) if best_fixed_mean is not None else None,
        "oracle_headroom": round(oracle_headroom, 4) if oracle_headroom is not None else None,
        "discriminative_oracle_headroom": (
            round(discriminative_oracle_headroom, 4) if discriminative_oracle_headroom is not None else None
        ),
        "windows_csv": str(windows_csv.relative_to(ROOT)),
        "discriminativeness_csv": str(disc_csv.relative_to(ROOT)),
    }
    (out_dir / "slo_calibrated_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def stage_robustness(args) -> None:
    """Section 8: for each of a fresh synthetic-window sample, evaluate at
    the chosen default multiplier's neighbors (1.5, 2.0, 3.0 -- the
    nearest grid point below, the chosen default, and one point above the
    grid's own max, since 2.0 is the top of `CALIBRATION_MULTIPLIER_GRID`)
    and classify whether the ANWG winner is stable."""
    rng = random.Random(args.search_seed)
    family_names = list(FAMILY_GENERATORS.keys())
    windows = []
    for i in range(args.n_robustness_windows):
        fname = family_names[i % len(family_names)]
        windows.append((fname, FAMILY_GENERATORS[fname](rng)))

    primary_obj = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)
    neighbor_multipliers = [1.5, 2.0, 3.0]
    rows = []
    for idx, (fname, window) in enumerate(windows):
        winners_by_multiplier = {}
        for m in neighbor_multipliers:
            result = _run_window_at_multiplier(window, m, args.search_seed, idx)
            if result is None:
                continue
            d = compute_discriminativeness(result["outcomes"], primary_obj)
            if d is None:
                continue
            winners_by_multiplier[m] = (d.best_policy, d.classification)
        if 2.0 not in winners_by_multiplier:
            continue
        default_winner, default_class = winners_by_multiplier[2.0]
        if default_class == "ALL_COMPLETE_OR_EFFECTIVELY_TIED":
            continue  # not a discriminative window at the default -- nothing to classify
        agreeing = sum(1 for m, (w, _c) in winners_by_multiplier.items() if w == default_winner)
        total = len(winners_by_multiplier)
        if agreeing == total:
            robustness = "ROBUST_TO_SLO_SCALE"
        elif agreeing >= 2:
            robustness = "SENSITIVE_TO_SLO_SCALE"
        else:
            robustness = "ARTIFACT_OF_THRESHOLD"
        rows.append(dict(
            window_idx=idx, shape=fname, default_winner=default_winner,
            default_classification=default_class, robustness=robustness,
            winners_by_multiplier={str(m): w for m, (w, _c) in winners_by_multiplier.items()},
        ))

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    counts = Counter(r["robustness"] for r in rows)
    summary = {
        "n_discriminative_windows_at_default": len(rows),
        "robustness_counts": dict(counts),
        "robustness_fractions": {k: round(v / len(rows), 4) for k, v in counts.items()} if rows else {},
    }
    (out_dir / "slo_scale_robustness.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["grid", "main", "robustness"], required=True)
    parser.add_argument("--n-robustness-windows", type=int, default=200)
    parser.add_argument("--search-seed", type=int, default=20260720)
    parser.add_argument("--output-dir", default="experiments/selector_v2_slo_calibrated_frontier_search")
    parser.add_argument("--grid-n-windows", type=int, default=100)
    parser.add_argument("--n-synthetic", type=int, default=750)
    parser.add_argument("--n-real-trace-seeds", type=int, default=10)
    parser.add_argument("--multiplier", type=float, default=1.0)
    args = parser.parse_args()

    if args.stage == "grid":
        stage_grid(args)
    elif args.stage == "main":
        stage_main(args)
    else:
        stage_robustness(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
