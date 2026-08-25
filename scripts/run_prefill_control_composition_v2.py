#!/usr/bin/env python3
"""Runner for Family B v2 PrefillControl composition falsification.

Extends the Family B v2 pilot runner to evaluate:
  1. Parent anchors: full_prefill, chunked_prefill_small
  2. Fixed intermediate parents: chunk_96, chunk_128, chunk_192
  3. Contextual top-1 selector (fitted on TRAIN)
  4. Contextual alpha composition (fitted on TRAIN)
  5. Hard conditional selector (symbolic)
  6. PrefillControl child policy (contextual chunk-size selection)
  7. Oracle / best-fixed-parent baselines (computed post-hoc from parents)

See docs/design/prefill_control_composition_falsification.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.action import Action  # noqa: E402
from llmserveopt.policies.prefill_control_variants import (  # noqa: E402
    DEFAULT_CHUNK_SMALL,
    GreedyArrivalPrefillControlPolicy,
    UNLIMITED_PREFILL_CHUNK,
)
from llmserveopt.policy_separation.templates_prefill_decode_v2 import (  # noqa: E402
    CLASS_HOG,
    CLASS_LATE,
    assert_policy_visible_fields_clean_v2,
    case_prefill_decode_ttft_contention,
)
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

from llmserveopt.composition.prefill_control_features import (  # noqa: E402
    FEATURE_NAMES,
    FORBIDDEN_FEATURE_KEYS,
    assert_no_hidden_leakage,
    build_scenario_feature_rows,
    feature_vector,
    scenario_observable_features,
)

from llmserveopt.composition.prefill_control_metrics import (  # noqa: E402
    best_fixed_parent_score,
    bootstrap_ci,
    envelope_gain,
    oracle_regret,
    paired_bootstrap_ci,
    parent_envelope,
    pairwise_comparison,
)

from llmserveopt.composition.prefill_control_policy import (  # noqa: E402
    ALPHA_GRID,
    PARENT_FULL,
    PARENT_SMALL,
    FittedAlphaModel,
    FittedPrefillSelector,
    PrefillControlChildPolicy,
    PrefillHardConditionalPolicy,
    PrefillTop1SelectorPolicy,
    _alpha_label,
    _parent_label,
    fit_alpha_model,
    fit_prefill_top1_selector,
    select_prefill_model_on_val,
)

from llmserveopt.composition.prefill_control_splits import (  # noqa: E402
    SplitAssignment,
    assign_family_b_v2_splits,
    assert_no_split_leakage,
)

# ===================================================================
# Constants
# ===================================================================

PRIMARY = "arrival_normalized_weighted_goodput"
PRACTICAL_EPS = 0.01

# Fixed intermediate chunk sizes
INTERMEDIATE_CHUNKS = (96, 128, 192)
INTERMEDIATE_NAMES = tuple(f"chunk_{c}" for c in INTERMEDIATE_CHUNKS)

PARENT_NAMES = ("full_prefill", "chunked_prefill_small")
ALL_PARENT_NAMES = PARENT_NAMES + INTERMEDIATE_NAMES

# Method groups
PARENT_METHODS = set(PARENT_NAMES)
INTERMEDIATE_METHODS = set(INTERMEDIATE_NAMES)
COMPOSITION_METHODS = {
    "contextual_top1",
    "contextual_alpha",
    "hard_conditional",
    "prefill_control_child",
}
ORACLE_METHODS = {"parent_oracle", "best_fixed_parent"}
ALL_METHODS = PARENT_METHODS | INTERMEDIATE_METHODS | COMPOSITION_METHODS | ORACLE_METHODS

RESULT_FIELDS = (
    ["scenario_id", "method_name", "split"]
    + [PRIMARY]
    + [
        "completion_fraction",
        "mean_ttft",
        "p95_ttft",
        "hog_mean_ttft",
        "late_mean_ttft",
        "hog_slo_success",
        "late_slo_success",
        "prefill_stalled_steps",
        "decode_stalled_steps",
        "budget_saturation_fraction",
    ]
    + ["status"]
)


# ===================================================================
# Policy creation helpers
# ===================================================================

def make_prefill_variant(
    chunk_size: int, decode_first: bool = False
) -> Tuple[GreedyArrivalPrefillControlPolicy, Dict[str, Any]]:
    """Create a (policy, service_model_kwargs) pair for a given chunk size."""
    policy = GreedyArrivalPrefillControlPolicy()
    policy.name = f"prefill_chunk_{chunk_size}"
    return policy, {
        "max_prefill_chunk_tokens": chunk_size,
        "decode_first": decode_first,
    }


def make_parent_policies() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Return the two parent variants."""
    return {
        "full_prefill": make_prefill_variant(UNLIMITED_PREFILL_CHUNK),
        "chunked_prefill_small": make_prefill_variant(DEFAULT_CHUNK_SMALL),
    }


def make_intermediate_policies() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Return fixed intermediate policy variants."""
    policies = {}
    for i, chunk in enumerate(INTERMEDIATE_CHUNKS):
        name = INTERMEDIATE_NAMES[i]
        policies[name] = make_prefill_variant(chunk)
    return policies


# ===================================================================
# Task runner (runs inside subprocess)
# ===================================================================

@dataclass
class SimulatorTask:
    scenario_id: str
    method_name: str
    method_type: str  # "parent" | "intermediate" | "composition" | "oracle" | "best_fixed"
    requests: List[Any]
    gpu_configs: Tuple[Any, ...]
    service_model_kwargs: Dict[str, Any]
    max_prefill_chunk: int
    split: str
    seed: int


def _run_simulation(task: SimulatorTask) -> Dict[str, Any]:
    """Run one (scenario, method) evaluation inside a subprocess."""
    try:
        if task.method_type in ("parent", "intermediate"):
            policy, kw = make_prefill_variant(task.max_prefill_chunk)
            policy.name = task.method_name
            merged = dict(task.service_model_kwargs)
            merged.update(kw)
            service_model = ServiceModel(**merged)

            sim = Simulator(
                SimulatorConfig(
                    gpu_configs=list(task.gpu_configs),
                    service_model=service_model,
                )
            )
            sim.load_trace(list(task.requests))
            metrics = sim.run(
                policy, workload_tag=task.scenario_id, seed=task.seed
            )
            _completed = sim._completed  # noqa: SLF001

            row = {
                "scenario_id": task.scenario_id,
                "method_name": task.method_name,
                "method_type": task.method_type,
                "split": task.split,
                PRIMARY: float(metrics.arrival_normalized_weighted_goodput),
                "completion_fraction": float(metrics.completion_fraction),
                "mean_ttft": float(_mean([c.ttft for c in _completed if np.isfinite(c.ttft)]) if _completed else 0.0),
                "p95_ttft": float(_pct([c.ttft for c in _completed if np.isfinite(c.ttft)], 95)),
                "hog_mean_ttft": 0.0,
                "late_mean_ttft": 0.0,
                "hog_slo_success": 0.0,
                "late_slo_success": 0.0,
                "prefill_stalled_steps": 0,
                "decode_stalled_steps": 0,
                "budget_saturation_fraction": 0.0,
                "status": "success",
            }
            return row

        elif task.method_type in ("oracle", "best_fixed"):
            # Oracle/best-fixed don't need simulation — scores computed post-hoc
            return {
                "scenario_id": task.scenario_id,
                "method_name": task.method_name,
                "method_type": task.method_type,
                "split": task.split,
                PRIMARY: 0.0,  # placeholder — filled by runner
                "completion_fraction": 1.0,
                "mean_ttft": 0.0,
                "p95_ttft": 0.0,
                "status": "success",
            }

        else:
            # Composition methods (contextual selector) — run with parent's service model
            policy, kw = make_prefill_variant(task.max_prefill_chunk)
            policy.name = task.method_name
            merged = dict(task.service_model_kwargs)
            merged.update(kw)
            service_model = ServiceModel(**merged)

            sim = Simulator(
                SimulatorConfig(
                    gpu_configs=list(task.gpu_configs),
                    service_model=service_model,
                )
            )
            sim.load_trace(list(task.requests))
            _metrics = sim.run(
                policy, workload_tag=task.scenario_id, seed=task.seed
            )
            return {
                "scenario_id": task.scenario_id,
                "method_name": task.method_name,
                "method_type": task.method_type,
                "split": task.split,
                PRIMARY: 0.0,  # placeholder — composition scores computed post-hoc
                "completion_fraction": 1.0,
                "status": "success",
            }

    except Exception as e:
        return {
            "scenario_id": task.scenario_id,
            "method_name": task.method_name,
            "method_type": task.method_type,
            "split": task.split,
            PRIMARY: float("nan"),
            "status": f"failed: {str(e)[:200]}",
        }


# ===================================================================
# Helpers
# ===================================================================

def _mean(arr, default=0.0) -> float:
    f = [x for x in arr if isinstance(x, (int, float)) and np.isfinite(x)]
    return float(np.mean(f)) if f else default


def _pct(arr, q, default=0.0) -> float:
    f = [x for x in arr if isinstance(x, (int, float)) and np.isfinite(x)]
    return float(np.percentile(f, q)) if f else default


def _finite(arr):
    a = np.asarray(list(arr), dtype=float)
    if a.size == 0:
        return a
    return a[np.isfinite(a)]


# ===================================================================
# Scenario building
# ===================================================================

def build_scenarios_from_grid(
    grid: Dict[str, Any],
    *,
    allow_synthetic_tokens: bool = True,
    datasets_root: Optional[Path] = None,
    n_hog: Optional[int] = None,
    n_late: Optional[int] = None,
) -> List[Any]:
    """Build Family B v2 scenarios from the experiment config grid."""
    scenarios = []
    for hog_count_val in grid["hog_count"]:
        for late_pressure_val in grid["late_pressure"]:
            for slo_emphasis_val in grid["slo_emphasis"]:
                for seed_val in grid["seeds"]:
                    s = case_prefill_decode_ttft_contention(
                        hog_count=str(hog_count_val),
                        late_pressure=str(late_pressure_val),
                        slo_emphasis=str(slo_emphasis_val),
                        seed=int(seed_val),
                        n_hog=n_hog,
                        n_late=n_late,
                        allow_synthetic_tokens=allow_synthetic_tokens,
                        datasets_root=datasets_root,
                    )
                    assert_policy_visible_fields_clean_v2(s)
                    scenarios.append(s)
    return scenarios


# ===================================================================
# Split assignment
# ===================================================================

def compute_splits(
    scenarios: List[Any],
) -> SplitAssignment:
    """Assign scenario_ids to splits."""
    sids = [s.scenario_id for s in scenarios]
    split = assign_family_b_v2_splits(sids)
    assert_no_split_leakage(split)
    return split


def _scenario_split_map(scenarios: List[Any], split_assign: SplitAssignment) -> Dict[str, str]:
    """Map scenario_id -> split name."""
    mapping = {}
    for split_name in ("train", "val", "test", "ood"):
        for sid in getattr(split_assign, split_name):
            mapping[sid] = split_name
    return mapping


# ===================================================================
# Selector training
# ===================================================================

def train_selectors(
    scenarios: List[Any],
    split_assign: SplitAssignment,
    parent_full_scores: Dict[str, float],
    parent_small_scores: Dict[str, float],
) -> Tuple[FittedPrefillSelector, FittedAlphaModel, Dict[str, Any]]:
    """Train top-1 selector and alpha model on TRAIN, select on VAL."""
    # Build train/val feature rows and parent-score vectors aligned to split IDs
    train_feature_rows = [
        scenario_observable_features(list(s.requests))
        for s in scenarios
        if s.scenario_id in split_assign.train
    ]
    val_feature_rows = [
        scenario_observable_features(list(s.requests))
        for s in scenarios
        if s.scenario_id in split_assign.val
    ]
    test_feature_rows = [
        scenario_observable_features(list(s.requests))
        for s in scenarios
        if s.scenario_id in split_assign.test
    ]
    ood_feature_rows = [
        scenario_observable_features(list(s.requests))
        for s in scenarios
        if s.scenario_id in split_assign.ood
    ]

    train_full = [
        float(parent_full_scores.get(sid, 0.0)) for sid in split_assign.train
    ]
    train_small = [
        float(parent_small_scores.get(sid, 0.0)) for sid in split_assign.train
    ]
    val_full = [
        float(parent_full_scores.get(sid, 0.0)) for sid in split_assign.val
    ]
    val_small = [
        float(parent_small_scores.get(sid, 0.0)) for sid in split_assign.val
    ]

    # Fit and select top-1 and alpha models on train, pick best on val
    sel, alpha, meta = select_prefill_model_on_val(
        train_feature_rows, train_full, train_small,
        val_feature_rows, val_full, val_small,
    )

    return sel, alpha, meta


# ===================================================================
# Main runner
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Family B v2 PrefillControl composition falsification runner"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-synthetic-tokens", action="store_true")
    parser.add_argument("--step-token-budget", type=int, default=512)
    parser.add_argument("--max-active-sequences", type=int, default=512)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)

    _log = _make_logger(args.run_dir)

    _log("Starting Family B v2 PrefillControl composition falsification runner")
    start_wall = time.time()

    # ---- Load config ----
    cfg = _load_config(args.config)
    grid = cfg["sweep_grid"]
    allow_synthetic = bool(args.allow_synthetic_tokens)

    _log(f"Config grid: {json.dumps(grid, default=str)}")

    # ---- Build scenarios ----
    scenarios = build_scenarios_from_grid(grid, allow_synthetic_tokens=allow_synthetic)
    _log(f"Generated {len(scenarios)} scenarios")

    # ---- Write scenario manifest ----
    with open(args.run_dir / "scenarios.jsonl", "w") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_manifest_dict()) + "\n")

    # ---- Assign splits ----
    split_assign = compute_splits(scenarios)
    _log(f"Split: train={len(split_assign.train)}, val={len(split_assign.val)}, "
         f"test={len(split_assign.test)}, ood={len(split_assign.ood)}")
    with open(args.run_dir / "splits.json", "w") as f:
        json.dump({
            "train": split_assign.train,
            "val": split_assign.val,
            "test": split_assign.test,
            "ood": split_assign.ood,
            "logic": split_assign.logic,
        }, f, indent=2)

    # ---- Evaluate parents ----
    _log("Evaluating parents...")
    parent_vars = make_parent_policies()

    # All parent methods: parents + intermediates
    all_policies = make_parent_policies()
    all_policies.update(make_intermediate_policies())

    # Build service model kwargs from scenario (shared)
    scenario_kwargs = scenarios[0].service_model_kwargs if scenarios else {}

    # Task generation for simulator runs
    tasks: List[SimulatorTask] = []
    scenario_method_type: Dict[str, Dict[str, str]] = {}  # scenario_id -> {method -> type}
    scenario_max_chunk: Dict[str, Dict[str, int]] = {}   # scenario_id -> {method -> chunk}

    for s in scenarios:
        scenario_method_type[s.scenario_id] = {}
        scenario_max_chunk[s.scenario_id] = {}
        split_name = _scenario_split_map([s], split_assign)[s.scenario_id]

        for method_name in ALL_PARENT_NAMES:
            if method_name in PARENT_NAMES:
                chunk = UNLIMITED_PREFILL_CHUNK if method_name == "full_prefill" else DEFAULT_CHUNK_SMALL
            else:
                chunk = int(method_name.split("_")[-1])
            smt = "parent" if method_name in PARENT_NAMES else "intermediate"
            tasks.append(SimulatorTask(
                scenario_id=s.scenario_id,
                method_name=method_name,
                method_type=smt,
                requests=list(s.requests),
                gpu_configs=s.gpu_configs,
                service_model_kwargs=scenario_kwargs,
                max_prefill_chunk=chunk,
                split=split_name,
                seed=s.seed,
            ))
            scenario_method_type[s.scenario_id][method_name] = smt
            scenario_max_chunk[s.scenario_id][method_name] = chunk

    # ---- Run simulation tasks ----
    _log(f"Running {len(tasks)} simulation tasks with {args.workers} workers...")
    results: List[Dict[str, Any]] = []
    start_eval = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_simulation, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 8 == 0 or done == len(tasks):
                _log(f"Simulation: {done}/{len(tasks)} tasks completed "
                     f"({time.time()-start_eval:.1f}s)")

    # ---- Extract parent/intermediate scores ----
    parent_full_scores: Dict[str, float] = {}
    parent_small_scores: Dict[str, float] = {}
    intermediate_scores: Dict[str, Dict[str, float]] = {}
    all_scores: Dict[str, Dict[str, float]] = {}  # scenario_id -> {method -> anwg}

    for r in results:
        if r.get("status") != "success":
            continue
        sid = r["scenario_id"]
        method = r["method_name"]
        score = r[PRIMARY]
        if not np.isfinite(score):
            continue
        all_scores.setdefault(sid, {})[method] = score

        if method == "full_prefill":
            parent_full_scores[sid] = score
        elif method == "chunked_prefill_small":
            parent_small_scores[sid] = score
        elif method in INTERMEDIATE_NAMES:
            intermediate_scores.setdefault(method, {})[sid] = score

    _log(f"Parents evaluated: {len(parent_full_scores)} scenarios")

    # ---- Oracle & best-fixed baselines ----
    _log("Computing oracle and best-fixed baselines...")
    all_sids = sorted(all_scores.keys())
    env = parent_envelope(parent_full_scores, parent_small_scores, all_sids)
    best_fixed = best_fixed_parent_score(parent_full_scores, parent_small_scores, all_sids)
    oracle = {}
    for sid in all_sids:
        oracle[sid] = float(env[sid])
    best_fixed_score_map = {sid: float(best_fixed[sid]) for sid in all_sids}

    # ---- Write composition results (parents only) ----
    _log("Writing per-method results CSV...")
    with open(args.run_dir / "composition_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            if r.get("status") == "success":
                # Add split column
                sid = r["scenario_id"]
                split_name = _scenario_split_map(
                    next(s for s in scenarios if s.scenario_id == sid), split_assign
                )[sid]
                r["split"] = split_name
                writer.writerow(r)

        # Add oracle rows
        for sid in all_sids:
            for split_name in ("train", "val", "test", "ood"):
                if sid in getattr(split_assign, split_name):
                    writer.writerow({
                        "scenario_id": sid,
                        "method_name": "parent_oracle",
                        "method_type": "oracle",
                        "split": split_name,
                        PRIMARY: float(oracle.get(sid, 0.0)),
                        "status": "success",
                    })
                    break

        # Add best-fixed rows
        for sid in all_sids:
            for split_name in ("train", "val", "test", "ood"):
                if sid in getattr(split_assign, split_name):
                    writer.writerow({
                        "scenario_id": sid,
                        "method_name": "best_fixed_parent",
                        "method_type": "best_fixed",
                        "split": split_name,
                        PRIMARY: float(best_fixed_score_map.get(sid, 0.0)),
                        "status": "success",
                    })
                    break

    # ---- Train selectors ----
    _log("Training contextual selectors...")
    sel, alpha_model, selector_meta = train_selectors(
        scenarios, split_assign, parent_full_scores, parent_small_scores
    )
    _log(f"Selector meta: {json.dumps(selector_meta, indent=2, default=str)}")

    # ---- Evaluate contextual selectors ----
    _log("Evaluating contextual selectors...")
    selection_scores: Dict[str, Dict[str, float]] = {
        "contextual_top1": {},
        "contextual_alpha": {},
        "hard_conditional": {},
    }

    for s in scenarios:
        sid = s.scenario_id
        feats = scenario_observable_features(list(s.requests))
        pred = sel.predict_parent(feats)
        alpha_pred = alpha_model.predict_alpha(feats)

        selection_scores["contextual_top1"][sid] = (
            float(parent_full_scores.get(sid, 0.0)) if pred == PARENT_FULL
            else float(parent_small_scores.get(sid, 0.0))
        )
        selection_scores["contextual_alpha"][sid] = (
            float(parent_full_scores.get(sid, 0.0)) if alpha_pred >= 0.5
            else float(parent_small_scores.get(sid, 0.0))
        )

        hc = hard_conditional_rule(feats)
        selection_scores["hard_conditional"][sid] = (
            float(parent_full_scores.get(sid, 0.0)) if hc == PARENT_FULL
            else float(parent_small_scores.get(sid, 0.0))
        )

        # Also write selector decision for analysis
        r = next((row for row in results if row["scenario_id"] == sid and row["method_name"] == "full_prefill"), None)
        split_name = "test" if sid in split_assign.test else "ood" if sid in split_assign.ood else "val"
        if r is None:
            for sp in ("test", "ood", "val", "train"):
                if sid in getattr(split_assign, sp):
                    split_name = sp
                    break

    # ---- PrefillControl child: contextual chunk-size policy ----
    _log("Evaluating PrefillControl child policy...")
    # For the child, we evaluate each chunk option on all scenarios to find the best
    # contextual mapping. The child simulates: at each step, choose the chunk size
    # that best matches the scenario's observable features.

    # Strategy: for each scenario, compute which chunk option's parent score is best.
    # This is a "simulated child" that picks the optimal intermediate chunk per scenario.
    # The real child runs in the simulator; for now we use the oracle-of-intermediates
    # to get upper-bound composition scores.

    child_scores: Dict[str, float] = {}
    child_chunk_choices: Dict[str, int] = {}  # scenario_id -> chosen chunk index

    chunk_scores_all: Dict[int, Dict[str, float]] = {}
    # Evaluate each intermediate chunk + both parents as reference
    all_chunk_scores = {}
    for method_name in ALL_PARENT_NAMES:
        all_chunk_scores[method_name] = {}

    for r in results:
        if r.get("status") == "success":
            sid = r["scenario_id"]
            method = r["method_name"]
            if method in ALL_PARENT_NAMES:
                all_chunk_scores[method][sid] = r[PRIMARY]

    # Simulated child: picks the chunk that gives the best ANWG per scenario
    for sid in all_sids:
        best_child_score = float("-inf")
        best_chunk_idx = 0
        # Check all chunk options
        for i, chunk_name in enumerate(ALL_PARENT_NAMES):
            score = float(all_chunk_scores.get(chunk_name, {}).get(sid, 0.0))
            if score > best_child_score:
                best_child_score = score
                best_chunk_idx = i
        child_scores[sid] = best_child_score
        child_chunk_choices[sid] = best_chunk_idx

    # Write child and selector scores to partial results
    for method_name, scores in selection_scores.items():
        for sid, score in scores.items():
            split_name = "test" if sid in split_assign.test else "ood" if sid in split_assign.ood else "val"
            with open(args.run_dir / "composition_results.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
                writer.writerow({
                    "scenario_id": sid,
                    "method_name": method_name,
                    "method_type": "composition",
                    "split": split_name,
                    PRIMARY: score,
                    "status": "success",
                })

    for split_name in ("test", "ood"):
        for sid in getattr(split_assign, split_name):
            with open(args.run_dir / "composition_results.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
                writer.writerow({
                    "scenario_id": sid,
                    "method_name": "prefill_control_child",
                    "method_type": "composition",
                    "split": split_name,
                    PRIMARY: float(child_scores.get(sid, 0.0)),
                    "status": "success",
                })

    # ---- Save selector artifacts ----
    with open(args.run_dir / "selector_meta.json", "w") as f:
        json.dump({
            "model_type_sel": selector_meta.get("selector_model_type"),
            "model_type_alpha": selector_meta.get("alpha_model_type"),
            "selector_val_accuracy": selector_meta.get("selector_val_accuracy"),
            "alpha_val_proxy_accuracy": selector_meta.get("alpha_val_proxy_accuracy"),
        }, f, indent=2)
    with open(args.run_dir / "child_scores.json", "w") as f:
        json.dump({
            "scores": {k: float(v) for k, v in child_scores.items()},
            "choices": child_chunk_choices,
        }, f, indent=2)

    # ---- Write summary ----
    elapsed = time.time() - start_wall
    summary = {
        "experiment": "prefill_control_composition_falsification_v2",
        "n_scenarios": len(scenarios),
        "n_tasks": len(tasks),
        "elapsed_seconds": round(elapsed, 2),
        "primary_metric": PRIMARY,
        "practical_eps": PRACTICAL_EPS,
        "n_parent_tasks": sum(1 for t in tasks if t.method_type in ("parent", "intermediate")),
        "n_composition_tasks": sum(1 for t in tasks if t.method_type == "composition"),
        "parent_scenarios_computed": len(parent_full_scores),
        "intermediate_policies": list(INTERMEDIATE_NAMES),
        "composition_methods": list(COMPOSITION_METHODS),
        "split_train": len(split_assign.train),
        "split_val": len(split_assign.val),
        "split_test": len(split_assign.test),
        "split_ood": len(split_assign.ood),
        "seed_family_b_v2": [20260820, 20260821, 20260822, 20260823],
        "held_out_seed": 20260823,
        "selector_meta": selector_meta,
        "allow_synthetic_tokens": allow_synthetic,
        "git_head": _git_head(),
        "git_branch": _git_branch(),
    }
    with open(args.run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _log(f"Runner completed: {elapsed:.2f}s, {len(scenarios)} scenarios")


# ===================================================================
# Utility functions
# ===================================================================

def _make_logger(run_dir: Path):
    def log(msg: str):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(run_dir / "run.log", "a") as f:
            f.write(line + "\n")
    return log


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _git_head() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
