#!/usr/bin/env python3
"""Selector Dataset v2 -- calibrated targeted pilot generator (Option B scope).

Generates a 250-500 RETAINED-window pilot over exactly the 8 approved
historical-monolithic-policy selector action space (see
docs/selector_v2_faithful_baseline_scope_audit.md), calibrated SLOs
(policy-independent reference-ServiceModel method, default multiplier 2.0),
broad synthetic + real-trace (BurstGPT/Azure) scenario diversity including a
genuinely newer, non-overlapping real-trace time slice reserved for OOD
evaluation, group-aware leakage-safe splits, and full 8-policy utility
vectors (never reduced to hard labels).

Adaptive generation: keeps drawing candidate windows (cycling synthetic
families A-E and real-trace base-file x transform x pool combinations) until
the retained-window target is reached or the bounded attempt cap is hit. A
window is RETAINED only if all 8 candidate policies produce a valid outcome
(see calibrated_targeted_pilot.run_window_all_candidates) -- every retained
row's 8-policy vector is therefore genuinely complete, never partial.

Checkpoints (retained_windows.csv, phase_status.json) are rewritten after
every batch so an interrupted run can be inspected or resumed without losing
already-retained windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.selector.dataset_v2 import calibrated_targeted_pilot as p  # noqa: E402
from llmserveopt.selector.dataset_v2.discriminativeness import (  # noqa: E402
    PRIMARY_SELECTOR_OBJECTIVE, STANDARD_OBJECTIVES, compute_discriminativeness,
)
from llmserveopt.selector.dataset_v2.features import extract_selector_v2_features  # noqa: E402
from llmserveopt.selector.dataset_v2.splits import (  # noqa: E402
    assign_group_aware_split,
    attach_leakage_safe_split_group_keys,
    leakage_safe_split_group_key,
    split_for_group,
    verify_group_atomicity,
    verify_no_cross_split_row_range_overlap,
    verify_ood_holdout,
)
from selector_v2_calibrated_pilot_gates import evaluate_quality_gates  # noqa: E402

SYNTHETIC_SHAPES = list(p.FAMILY_GENERATORS.keys())
PRIMARY_OBJECTIVE = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)

CHECKPOINT_FIELDS: List[str] = [
    "window_idx", "group_key", "split_group_key", "dataset_family", "source_trace", "shape",
    "time_slice_pool", "time_slice_row_start", "time_slice_row_end",
    "time_slice_arrival_start", "time_slice_arrival_end",
    "request_plan_ancestor_id", "n_requests", "calibration_multiplier", "split",
    "primary_objective_classification", "primary_objective_best_policy",
    "primary_objective_max_min_spread",
]


def _write_manifest(out_dir: Path, args: argparse.Namespace) -> None:
    manifest = {
        "pilot_name": "selector_v2_calibrated_targeted_pilot",
        "scope_decision": "OPTION B (docs/selector_v2_faithful_baseline_scope_audit.md)",
        "selector_action_space_8_policies": list(p.CANDIDATE_POLICIES),
        "excluded_faithful_baselines": list(p.EXCLUDED_FAITHFUL_BASELINES),
        "excluded_cross_topology_baselines": list(p.EXCLUDED_CROSS_TOPOLOGY_BASELINES),
        "primary_objective": PRIMARY_SELECTOR_OBJECTIVE,
        "slo_calibration_method": "v1_reference_service_model (policy-independent)",
        "slo_calibration_multiplier": args.multiplier,
        "slo_calibration_sensitivity_multipliers": [1.5, 2.0, 3.0],
        "target_retained_windows_min": args.target_min_retained,
        "target_retained_windows_max": args.target_max_retained,
        "synthetic_families": SYNTHETIC_SHAPES,
        "real_trace_base_files": [c[0] for c in p.REAL_TRACE_BASE_FILES],
        "real_trace_transforms": [t[0] for t in p.REAL_TRACE_TRANSFORMS],
        "real_trace_ood_reserved_fraction": p.OOD_RESERVED_FRACTION,
        "newer_time_slice_method": (
            "Each real-trace processed JSONL is chronologically sorted by arrival_time "
            "(verified). The last OOD_RESERVED_FRACTION of rows (chronologically later, "
            "relative trace time -- no calendar date exists in these sources beyond that) "
            "is reserved, disjoint by row-index construction, exclusively for OOD_TEST. "
            "The prior 910-window SLO-calibrated pilot only sampled seeds 0-9 from the "
            "HISTORICAL pool (first 1-fraction of rows); this reservation is structural, "
            "not merely non-collision-by-luck."
        ),
        "search_seed": args.search_seed,
        "max_attempts": args.max_attempts,
        "batch_size": args.batch_size,
        "drain_steps": args.drain_steps,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_provenance(out_dir: Path) -> None:
    provenance: Dict[str, Dict] = {}
    for base_name, source, rel_path in p.REAL_TRACE_BASE_FILES:
        path = ROOT / rel_path
        if not path.exists():
            provenance[base_name] = {"source": source, "rel_path": rel_path, "acquired": False}
            continue
        reqs, _meta = p.load_extended_jsonl(path)
        n = len(reqs)
        hist_lo, hist_hi = p.pool_row_range(n, p.HISTORICAL_POOL)
        ood_lo, ood_hi = p.pool_row_range(n, p.OOD_RESERVED_POOL)
        provenance[base_name] = {
            "source": source, "rel_path": rel_path, "acquired": True, "n_rows": n,
            "historical_pool_row_range": [hist_lo, hist_hi],
            "historical_pool_arrival_range": [reqs[hist_lo].arrival_time, reqs[hist_hi - 1].arrival_time],
            "ood_reserved_pool_row_range": [ood_lo, ood_hi],
            "ood_reserved_pool_arrival_range": [reqs[ood_lo].arrival_time, reqs[ood_hi - 1].arrival_time],
        }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


def _checkpoint_row(idx: int, window: "p.CandidateWindow", calibrated_n: int,
                     multiplier: float, split: str, disc) -> Dict:
    row_range = window.time_slice_row_range or (None, None)
    arrival_range = window.time_slice_arrival_range or (None, None)
    row = {
        "window_idx": idx, "group_key": window.group_key, "dataset_family": window.dataset_family,
        "source_trace": window.source_trace, "shape": window.shape,
        "time_slice_pool": window.time_slice_pool,
        "time_slice_row_start": row_range[0], "time_slice_row_end": row_range[1],
        "time_slice_arrival_start": arrival_range[0], "time_slice_arrival_end": arrival_range[1],
        "request_plan_ancestor_id": window.request_plan_ancestor_id, "n_requests": calibrated_n,
        "calibration_multiplier": multiplier, "split": split,
        "primary_objective_classification": disc.classification if disc else None,
        "primary_objective_best_policy": disc.best_policy if disc else None,
        "primary_objective_max_min_spread": round(disc.max_min_spread, 6) if disc else None,
    }
    attach_leakage_safe_split_group_keys([row])
    return row


def _split_group_key(window: "p.CandidateWindow") -> str:
    """Leakage-safe split atom for a `CandidateWindow` (builder-time convenience).

    Thin wrapper -- the actual grouping semantics live in exactly one place,
    `splits.py::leakage_safe_split_group_key`. Kept for tests/callers that
    have a `CandidateWindow` object rather than a checkpoint-row dict.
    """
    return leakage_safe_split_group_key({
        "dataset_family": window.dataset_family,
        "request_plan_ancestor_id": window.request_plan_ancestor_id,
        "time_slice_pool": window.time_slice_pool,
        "group_key": window.group_key,
    })


def _next_synthetic(rng: random.Random, idx: int) -> "p.CandidateWindow":
    shape = SYNTHETIC_SHAPES[idx % len(SYNTHETIC_SHAPES)]
    return p.synthetic_candidate_window(shape, rng, idx)


def _next_real_trace(combos: List, combo_cursor: int, seed: int) -> Optional["p.CandidateWindow"]:
    base_name, source, rel_path, transform_name, pool = combos[combo_cursor % len(combos)]
    return p.real_trace_candidate_window(ROOT, base_name, source, rel_path, transform_name, pool, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-min-retained", type=int, default=250)
    parser.add_argument("--target-max-retained", type=int, default=500)
    parser.add_argument("--min-real-trace-retained", type=int, default=60)
    parser.add_argument("--min-ood-reserved-retained", type=int, default=20)
    parser.add_argument("--real-trace-attempt-fraction", type=float, default=0.35)
    parser.add_argument("--ood-reserved-attempt-fraction", type=float, default=0.35,
                         help="Fraction of real-trace attempts (not overall) targeting the OOD-reserved pool.")
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--search-seed", type=int, default=20260720)
    parser.add_argument("--max-attempts", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--drain-steps", type=int, default=5000)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    _write_manifest(out_dir, args)
    _write_provenance(out_dir)

    rng = random.Random(args.search_seed)
    combos = p.all_real_trace_combinations()

    retained_rows: List[Dict] = []
    retained_features: List[Dict] = []          # window_idx -> feat_* dict
    retained_outcomes: List[Dict] = []           # window_idx -> {policy: PolicyOutcomeVector}
    retained_split_group_keys: List[str] = []
    per_window_anwg: List[Dict[str, float]] = []

    n_attempts = 0
    n_retained = 0
    n_retained_real_trace = 0
    n_retained_ood_reserved = 0
    n_discarded_incomplete = 0
    n_discarded_empty = 0
    synth_cursor = 0
    real_cursor = 0
    real_seed_cursor = 0
    t0 = time.time()

    def _checkpoint(status: str) -> None:
        csv_path = out_dir / "retained_windows.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CHECKPOINT_FIELDS)
            w.writeheader()
            w.writerows(retained_rows)
        status_obj = {
            "status": status, "n_attempts": n_attempts, "n_retained": n_retained,
            "n_retained_real_trace": n_retained_real_trace,
            "n_retained_ood_reserved": n_retained_ood_reserved,
            "n_discarded_incomplete": n_discarded_incomplete,
            "n_discarded_empty": n_discarded_empty,
            "elapsed_s": round(time.time() - t0, 1),
            "target_min_retained": args.target_min_retained,
            "target_max_retained": args.target_max_retained,
        }
        (out_dir / "phase_status.json").write_text(json.dumps(status_obj, indent=2))
        print(json.dumps(status_obj))

    while n_attempts < args.max_attempts and n_retained < args.target_max_retained:
        batch_start_attempts = n_attempts
        for _ in range(args.batch_size):
            if n_attempts >= args.max_attempts or n_retained >= args.target_max_retained:
                break
            n_attempts += 1
            is_real_trace_draw = (rng.random() < args.real_trace_attempt_fraction)
            window: Optional["p.CandidateWindow"]
            if is_real_trace_draw:
                target_ood = rng.random() < args.ood_reserved_attempt_fraction
                combo_list = [c for c in combos if (c[4] == p.OOD_RESERVED_POOL) == target_ood]
                idx_in_list = real_cursor % len(combo_list)
                base_name, source, rel_path, transform_name, pool = combo_list[idx_in_list]
                window = p.real_trace_candidate_window(
                    ROOT, base_name, source, rel_path, transform_name, pool, seed=real_seed_cursor,
                )
                real_cursor += 1
                if real_cursor % len(combo_list) == 0:
                    real_seed_cursor += 1
            else:
                window = _next_synthetic(rng, synth_cursor)
                synth_cursor += 1

            if window is None or not window.requests:
                n_discarded_empty += 1
                continue

            calibrated = p.calibrate_candidate_window(window, args.multiplier)
            outcomes = p.run_window_all_candidates(
                window, calibrated, seed=args.search_seed + n_attempts,
                workload_tag=f"calib_pilot_{n_attempts}", drain_steps=args.drain_steps,
            )
            if outcomes is None:
                n_discarded_incomplete += 1
                continue

            disc = compute_discriminativeness(outcomes, PRIMARY_OBJECTIVE)
            feats = extract_selector_v2_features(
                window_requests=calibrated, window_start_time=calibrated[0].arrival_time,
                gpu_configs=[], topology_class="monolithic",
                step_token_budget=window.budget,
            )

            idx = n_retained
            retained_rows.append(_checkpoint_row(idx, window, len(calibrated), args.multiplier, "PENDING", disc))
            retained_features.append(feats)
            retained_outcomes.append({o.policy_name: o for o in outcomes})
            retained_split_group_keys.append(retained_rows[-1]["split_group_key"])
            per_window_anwg.append({
                o.policy_name: o.arrival_normalized_weighted_goodput for o in outcomes
                if o.arrival_normalized_weighted_goodput is not None
            })
            n_retained += 1
            if window.dataset_family == "real_trace":
                n_retained_real_trace += 1
                if window.time_slice_pool == p.OOD_RESERVED_POOL:
                    n_retained_ood_reserved += 1

        _checkpoint("running")
        if (
            n_retained >= args.target_min_retained
            and n_retained_real_trace >= args.min_real_trace_retained
            and n_retained_ood_reserved >= args.min_ood_reserved_retained
        ):
            break
        if n_attempts == batch_start_attempts:
            break  # made no progress this batch -- avoid an infinite loop

    # ------------------------------------------------------------------
    # Split assignment (group-aware, OOD-forced for the reserved pool)
    # ------------------------------------------------------------------
    attach_leakage_safe_split_group_keys(retained_rows)
    retained_split_group_keys = [row["split_group_key"] for row in retained_rows]
    ood_forced_groups = {
        row["split_group_key"] for row in retained_rows if row["time_slice_pool"] == p.OOD_RESERVED_POOL
    }
    all_group_keys = sorted(set(retained_split_group_keys))
    split_assignment = assign_group_aware_split(
        all_group_keys, ood_group_keys=ood_forced_groups,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )
    for row, gk in zip(retained_rows, retained_split_group_keys):
        row["split"] = split_for_group(gk, split_assignment)

    verify_group_atomicity(retained_rows, group_key_field="split_group_key", split_field="split")
    verify_ood_holdout(retained_rows, group_key_field="split_group_key", ood_group_keys=ood_forced_groups, split_field="split")
    verify_no_cross_split_row_range_overlap(retained_rows)

    split_counts = Counter(row["split"] for row in retained_rows)
    (out_dir / "split_manifest.json").write_text(json.dumps({
        "split_counts": dict(split_counts),
        "n_groups": len(all_group_keys),
        "n_ood_forced_groups": len(ood_forced_groups),
        "ood_forced_groups": sorted(ood_forced_groups),
        "split_group_key_definition": (
            "synthetic: transform-specific synthetic group_key; real_trace: "
            "request_plan_ancestor_id + time_slice_pool, so all transforms and "
            "row slices from the same raw source pool are split atomically"
        ),
        "split_assignment": split_assignment,
        "train_frac": args.train_frac, "val_frac": args.val_frac,
    }, indent=2))

    # ------------------------------------------------------------------
    # Full policy vectors (one row per window x policy -- never collapsed)
    # ------------------------------------------------------------------
    full_vector_rows: List[Dict] = []
    for row, outcomes_by_policy in zip(retained_rows, retained_outcomes):
        for pname, outcome in outcomes_by_policy.items():
            frow = {
                "window_idx": row["window_idx"], "group_key": row["group_key"],
                "split_group_key": row["split_group_key"], "split": row["split"],
                "policy_name": pname,
            }
            frow.update(outcome.to_row_dict(prefix="metric"))
            full_vector_rows.append(frow)
    if full_vector_rows:
        fv_path = out_dir / "full_policy_vectors.csv"
        fieldnames = sorted({k for r in full_vector_rows for k in r.keys()})
        with open(fv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(full_vector_rows)

    # Features CSV (one row per window; joins to retained_windows.csv by window_idx).
    # Keys get the repo-standard "feat_" prefix (features.py's own
    # selector_v2_feature_columns() / schema.py's to_flat_rows() convention)
    # so this file is consumable by the same tooling as other v2 pilots.
    if retained_features:
        feat_path = out_dir / "window_features.csv"
        feat_rows = [
            {"window_idx": row["window_idx"], **{f"feat_{k}": v for k, v in feats.items()}}
            for row, feats in zip(retained_rows, retained_features)
        ]
        fieldnames = sorted({k for r in feat_rows for k in r.keys()})
        with open(feat_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(feat_rows)

    _checkpoint("generation_complete")

    # ------------------------------------------------------------------
    # Quality gates
    # ------------------------------------------------------------------
    gates_result = evaluate_quality_gates(
        retained_rows=retained_rows, per_window_anwg=per_window_anwg,
        n_retained_real_trace=n_retained_real_trace, n_retained_ood_reserved=n_retained_ood_reserved,
        split_counts=dict(split_counts),
    )
    (out_dir / "quality_gates.json").write_text(json.dumps(gates_result, indent=2))
    print(json.dumps(gates_result, indent=2))

    summary_lines = [
        "# Selector v2 calibrated targeted pilot -- final summary\n",
        f"Retained windows: {n_retained} (target {args.target_min_retained}-{args.target_max_retained})\n",
        f"Attempts: {n_attempts}; discarded (incomplete 8-policy vector): {n_discarded_incomplete}; "
        f"discarded (empty): {n_discarded_empty}\n",
        f"Real-trace retained: {n_retained_real_trace} (OOD-reserved: {n_retained_ood_reserved})\n",
        f"Split counts: {dict(split_counts)}\n",
        f"All quality gates passed: {gates_result['all_gates_passed']}\n",
    ]
    if not gates_result["all_gates_passed"]:
        summary_lines.append("Failed gates:\n")
        for name, g in gates_result["gates"].items():
            if not g["passed"]:
                summary_lines.append(f"- {name}: {g}\n")
    (out_dir / "final_summary.md").write_text("".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
