#!/usr/bin/env python3
"""KV-aware composition falsification v1 runner.

See docs/design/KV_COMPOSITION_FALSIFICATION_V1.md. Evaluates, on the exact
same 72-scenario grid as the frozen KV v2 pairwise-separation pilot
(`case_kv_pressure_reserve_contention_v2`, unmodified):

  1. Parents: kv_constrained_online, least_laxity_first (unmodified)
  2. kv_adaptive_reserve_child (the falsification target; delegates every
     step to an unmodified parent instance, chosen from an online-observable
     trigger -- see kv_composition_policy.KVAdaptiveReserveChildPolicy)
  3. contextual_top1 (fitted selector), hard_conditional (symbolic rule),
     best_fixed_parent, parent_oracle, oracle_after_child -- all computed
     analytically from already-simulated parent/child scores, never re-run.

tau_urgent (the child's one free parameter) is fit on TRAIN, confirmed on
VAL, and frozen into child_threshold.json BEFORE any TEST/OOD scenario is
simulated for the child.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation.templates_kv_pressure_v2 import (  # noqa: E402
    assert_policy_visible_fields_clean_kv_v2,
    case_kv_pressure_reserve_contention_v2,
)
from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy  # noqa: E402
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

from llmserveopt.composition.kv_composition_features import (  # noqa: E402
    assert_no_hidden_leakage,
    scenario_observable_features,
)
from llmserveopt.composition.kv_composition_policy import (  # noqa: E402
    PARENT_KV,
    PARENT_LLF,
    TAU_URGENT_GRID,
    KVAdaptiveReserveChildPolicy,
    KVAdaptiveReserveHysteresisChildPolicy,
    hard_conditional_rule,
    select_kv_model_on_val,
)
from llmserveopt.composition.kv_composition_splits import (  # noqa: E402
    assert_no_split_leakage,
    assign_kv_composition_splits,
)
from llmserveopt.composition.kv_composition_metrics import (  # noqa: E402
    PRIMARY,
    best_fixed_parent_score,
    parent_envelope,
)

PARENT_METHODS = (PARENT_KV, PARENT_LLF)
CHILD_METHOD = "kv_adaptive_reserve_child"
CHILD_METHOD_HYSTERESIS = "kv_adaptive_reserve_hysteresis_child"


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def build_scenarios(cfg: dict, *, datasets_root) -> List[Any]:
    grid = cfg["sweep_grid"]
    scenarios = []
    for bulk_pressure in grid["bulk_pressure"]:
        for phase in grid["urgent_arrival_phase"]:
            for tightness in grid["urgent_tightness"]:
                for seed in grid["seeds"]:
                    s = case_kv_pressure_reserve_contention_v2(
                        bulk_pressure=str(bulk_pressure),
                        urgent_arrival_phase=str(phase),
                        urgent_tightness=str(tightness),
                        seed=int(seed),
                        max_kv_tokens=int(cfg.get("max_kv_tokens", 6000)),
                        max_active_sequences=int(cfg.get("max_active_sequences", 64)),
                        max_batch_tokens=int(cfg.get("max_batch_tokens", 64)),
                        allow_synthetic_tokens=False,
                        datasets_root=datasets_root,
                    )
                    assert_policy_visible_fields_clean_kv_v2(s)
                    scenarios.append(s)
    return scenarios


def _simulate(scenario: Any, policy: Any) -> Dict[str, Any]:
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(policy, workload_tag=scenario.scenario_id, seed=scenario.seed)
    completed = list(sim._completed)  # noqa: SLF001
    admission_time = {c.request.request_id: float(c.admission_time) for c in completed}
    
    gpu = sim._gpus[0]
    peak_tokens = max(gpu.step_kv_used) if gpu.step_kv_used else 0
    peak_kv = peak_tokens / gpu.config.max_kv_tokens

    return {
        "anwg": float(metrics.arrival_normalized_weighted_goodput),
        "completion_fraction": float(metrics.completion_fraction),
        "admission_time": admission_time,
        "n_steps": len(gpu.step_kv_used),  # noqa: SLF001
        "peak_kv": peak_kv,
    }


def _run_parent_task(args: Tuple[str, Any]) -> Dict[str, Any]:
    method_name, scenario = args
    try:
        policy_cls = {PARENT_KV: KVConstrainedOnlinePolicy, PARENT_LLF: LeastLaxityFirstPolicy}[method_name]
        result = _simulate(scenario, policy_cls())
        return {
            "scenario_id": scenario.scenario_id, "method_name": method_name,
            "status": "success", **result,
        }
    except Exception as e:  # noqa: BLE001
        return {"scenario_id": scenario.scenario_id, "method_name": method_name,
                "status": f"failed: {e}"}


def _run_child_task(args: Tuple[Any, int, str]) -> Dict[str, Any]:
    scenario, tau_urgent, method_name = args
    try:
        policy_cls = {
            CHILD_METHOD: KVAdaptiveReserveChildPolicy,
            CHILD_METHOD_HYSTERESIS: KVAdaptiveReserveHysteresisChildPolicy,
        }[method_name]
        policy = policy_cls(tau_urgent=tau_urgent)
        result = _simulate(scenario, policy)
        return {
            "scenario_id": scenario.scenario_id, "method_name": method_name,
            "status": "success",
            "n_llf_steps": policy.n_llf_steps,
            "n_reserve_steps": policy.n_reserve_steps,
            "transition_count": policy.transition_count,
            "kv_util_at_transition_mean": (
                float(np.mean(policy.kv_util_at_transition))
                if policy.kv_util_at_transition else 0.0
            ),
            **result,
        }
    except Exception as e:  # noqa: BLE001
        return {"scenario_id": scenario.scenario_id, "method_name": method_name,
                "status": f"failed: {e}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="KV-aware composition falsification v1 runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets-root", type=Path, default=Path(".local_data"))
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    _log(args.run_dir, "Starting KV-aware composition falsification v1 runner.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    scenarios = build_scenarios(cfg, datasets_root=args.datasets_root)
    _log(args.run_dir, f"Generated {len(scenarios)} scenarios.")
    by_id = {s.scenario_id: s for s in scenarios}

    split = assign_kv_composition_splits(list(by_id.keys()))
    assert_no_split_leakage(split)
    _log(args.run_dir, f"Split: train={len(split.train)} val={len(split.val)} "
                        f"test={len(split.test)} ood={len(split.ood)}")
    with open(args.run_dir / "splits.json", "w") as f:
        json.dump({"train": split.train, "val": split.val, "test": split.test,
                    "ood": split.ood, "logic": split.logic}, f, indent=2)

    def split_of(sid: str) -> str:
        for name in ("train", "val", "test", "ood"):
            if sid in getattr(split, name):
                return name
        raise KeyError(sid)

    # ---- Step 1: evaluate parents on ALL scenarios ----
    _log(args.run_dir, "Evaluating parents on all scenarios...")
    parent_results: Dict[str, Dict[str, Any]] = {}
    tasks = [(m, s) for s in scenarios for m in PARENT_METHODS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_parent_task, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            parent_results[(r["scenario_id"], r["method_name"])] = r
    n_parent_failed = sum(1 for r in parent_results.values() if r["status"] != "success")
    _log(args.run_dir, f"Parents done: {len(parent_results)} rows, {n_parent_failed} failed.")

    kv_scores = {sid: parent_results[(sid, PARENT_KV)]["anwg"]
                 for sid in by_id if parent_results.get((sid, PARENT_KV), {}).get("status") == "success"}
    llf_scores = {sid: parent_results[(sid, PARENT_LLF)]["anwg"]
                  for sid in by_id if parent_results.get((sid, PARENT_LLF), {}).get("status") == "success"}

    # ---- Step 2: fit tau_urgent on TRAIN, confirm on VAL, freeze ----
    _log(args.run_dir, "Fitting tau_urgent on TRAIN using original child...")
    train_scenarios = [by_id[sid] for sid in split.train]
    val_scenarios = [by_id[sid] for sid in split.val]

    tau_candidates: Dict[int, float] = {}
    tau_train_rows: Dict[int, List[Dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_child_task, (s, tau, CHILD_METHOD)): (s.scenario_id, tau)
                for tau in TAU_URGENT_GRID for s in train_scenarios}
        rows_by_tau: Dict[int, List[Dict[str, Any]]] = {t: [] for t in TAU_URGENT_GRID}
        for fut in as_completed(futs):
            sid, tau = futs[fut]
            r = fut.result()
            rows_by_tau[tau].append(r)
    for tau, rows in rows_by_tau.items():
        ok = [r["anwg"] for r in rows if r["status"] == "success"]
        tau_candidates[tau] = float(np.mean(ok)) if ok else float("-inf")
    best_tau = max(tau_candidates, key=tau_candidates.get)
    _log(args.run_dir, f"TRAIN mean ANWG by tau_urgent: {tau_candidates}; best={best_tau}")

    # Confirm on VAL: best_tau must not be worse than runner-up TRAIN candidate on VAL
    val_means: Dict[int, float] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_child_task, (s, tau, CHILD_METHOD)): tau
                for tau in TAU_URGENT_GRID for s in val_scenarios}
        rows_by_tau_val: Dict[int, List[Dict[str, Any]]] = {t: [] for t in TAU_URGENT_GRID}
        for fut in as_completed(futs):
            tau = futs[fut]
            rows_by_tau_val[tau].append(fut.result())
    for tau, rows in rows_by_tau_val.items():
        ok = [r["anwg"] for r in rows if r["status"] == "success"]
        val_means[tau] = float(np.mean(ok)) if ok else float("-inf")
    ranked_train = sorted(tau_candidates, key=tau_candidates.get, reverse=True)
    runner_up = ranked_train[1] if len(ranked_train) > 1 else best_tau
    val_confirmed = val_means[best_tau] >= val_means[runner_up] - 1e-9
    _log(args.run_dir, f"VAL mean ANWG by tau_urgent: {val_means}; "
                        f"best_tau={best_tau} confirmed={val_confirmed}")

    frozen_tau = {
        "tau_urgent": int(best_tau),
        "train_means": {int(k): v for k, v in tau_candidates.items()},
        "val_means": {int(k): v for k, v in val_means.items()},
        "val_confirmed_not_worse_than_runner_up": bool(val_confirmed),
        "runner_up_tau": int(runner_up),
    }
    with open(args.run_dir / "child_threshold.json", "w") as f:
        json.dump(frozen_tau, f, indent=2)
    _log(args.run_dir, f"FROZEN tau_urgent={best_tau}. Now evaluating child on VAL/TEST/OOD.")

    # ---- Step 3: evaluate frozen child on VAL+TEST+OOD (TRAIN already computed above) ----
    remaining = [by_id[sid] for sid in (split.test + split.ood)]
    child_rows: Dict[str, Dict[str, Any]] = {
        r["scenario_id"]: r for r in rows_by_tau[best_tau]  # TRAIN, already run
    }
    child_rows.update({r["scenario_id"]: r for r in rows_by_tau_val[best_tau]})  # VAL, already run
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_child_task, (s, best_tau, CHILD_METHOD)): s.scenario_id for s in remaining}
        for fut in as_completed(futs):
            r = fut.result()
            child_rows[r["scenario_id"]] = r
    n_child_failed = sum(1 for r in child_rows.values() if r["status"] != "success")
    _log(args.run_dir, f"Child done: {len(child_rows)} rows, {n_child_failed} failed.")

    child_scores = {sid: r["anwg"] for sid, r in child_rows.items() if r["status"] == "success"}

    # ---- Step 3.5: evaluate safety-refined hysteresis child on ALL scenarios with best_tau ----
    _log(args.run_dir, f"Evaluating hysteresis child on all scenarios with tau_urgent={best_tau}...")
    hysteresis_child_rows: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_child_task, (s, best_tau, CHILD_METHOD_HYSTERESIS)): s.scenario_id for s in scenarios}
        for fut in as_completed(futs):
            r = fut.result()
            hysteresis_child_rows[r["scenario_id"]] = r
    n_hyst_failed = sum(1 for r in hysteresis_child_rows.values() if r["status"] != "success")
    _log(args.run_dir, f"Hysteresis child done: {len(hysteresis_child_rows)} rows, {n_hyst_failed} failed.")

    hysteresis_scores = {sid: r["anwg"] for sid, r in hysteresis_child_rows.items() if r["status"] == "success"}

    # ---- Step 4: non-degeneracy / admission-disagreement diagnostic ----
    _log(args.run_dir, "Computing admission-disagreement diagnostic vs both parents...")
    disagree_counts: Dict[str, int] = {}
    disagree_counts_hyst: Dict[str, int] = {}
    
    for sid, crow in child_rows.items():
        if crow["status"] != "success":
            continue
        kv_row = parent_results.get((sid, PARENT_KV), {})
        llf_row = parent_results.get((sid, PARENT_LLF), {})
        if kv_row.get("status") != "success" or llf_row.get("status") != "success":
            continue
        child_adm = crow["admission_time"]
        kv_adm = kv_row["admission_time"]
        llf_adm = llf_row["admission_time"]
        n_diff = 0
        for rid, t_child in child_adm.items():
            t_kv = kv_adm.get(rid)
            t_llf = llf_adm.get(rid)
            if t_child != t_kv and t_child != t_llf:
                n_diff += 1
        all_ids = set(child_adm) | set(kv_adm) | set(llf_adm)
        for rid in all_ids:
            in_c, in_k, in_l = rid in child_adm, rid in kv_adm, rid in llf_adm
            if in_c != in_k and in_c != in_l:
                n_diff += 1
        disagree_counts[sid] = n_diff

    for sid, crow in hysteresis_child_rows.items():
        if crow["status"] != "success":
            continue
        kv_row = parent_results.get((sid, PARENT_KV), {})
        llf_row = parent_results.get((sid, PARENT_LLF), {})
        if kv_row.get("status") != "success" or llf_row.get("status") != "success":
            continue
        child_adm = crow["admission_time"]
        kv_adm = kv_row["admission_time"]
        llf_adm = llf_row["admission_time"]
        n_diff = 0
        for rid, t_child in child_adm.items():
            t_kv = kv_adm.get(rid)
            t_llf = llf_adm.get(rid)
            if t_child != t_kv and t_child != t_llf:
                n_diff += 1
        all_ids = set(child_adm) | set(kv_adm) | set(llf_adm)
        for rid in all_ids:
            in_c, in_k, in_l = rid in child_adm, rid in kv_adm, rid in llf_adm
            if in_c != in_k and in_c != in_l:
                n_diff += 1
        disagree_counts_hyst[sid] = n_diff

    # ---- Step 5: fit selectors on TRAIN, select on VAL (analytic) ----
    _log(args.run_dir, "Fitting contextual_top1 selector...")
    train_feats = [scenario_observable_features(list(by_id[sid].requests)) for sid in split.train]
    val_feats = [scenario_observable_features(list(by_id[sid].requests)) for sid in split.val]
    for f in train_feats + val_feats:
        assert_no_hidden_leakage(f)
    kv_train = [kv_scores.get(sid, 0.0) for sid in split.train]
    llf_train = [llf_scores.get(sid, 0.0) for sid in split.train]
    kv_val = [kv_scores.get(sid, 0.0) for sid in split.val]
    llf_val = [llf_scores.get(sid, 0.0) for sid in split.val]
    selector, selector_meta = select_kv_model_on_val(
        train_feats, kv_train, llf_train, val_feats, kv_val, llf_val
    )
    _log(args.run_dir, f"Selector meta: {selector_meta}")

    all_ids = list(by_id.keys())
    selector_scores: Dict[str, float] = {}
    hard_scores: Dict[str, float] = {}
    selector_choice: Dict[str, str] = {}
    for sid in all_ids:
        feats = scenario_observable_features(list(by_id[sid].requests))
        pred = selector.predict_parent(feats)
        selector_choice[sid] = pred
        selector_scores[sid] = kv_scores.get(sid, 0.0) if pred == PARENT_KV else llf_scores.get(sid, 0.0)
        hc = hard_conditional_rule(feats)
        hard_scores[sid] = kv_scores.get(sid, 0.0) if hc == PARENT_KV else llf_scores.get(sid, 0.0)

    # ---- Step 6: oracle / best-fixed (analytic, TRAIN-only fitting) ----
    envelope = parent_envelope(kv_scores, llf_scores, all_ids)
    best_fixed_scores, best_fixed_name = best_fixed_parent_score(
        kv_scores, llf_scores, split.train, all_ids
    )
    oracle_after_child = {
        sid: max(envelope.get(sid, 0.0), child_scores.get(sid, 0.0)) for sid in all_ids
    }
    oracle_after_hysteresis = {
        sid: max(envelope.get(sid, 0.0), hysteresis_scores.get(sid, 0.0)) for sid in all_ids
    }
    _log(args.run_dir, f"best_fixed_parent (TRAIN-selected) = {best_fixed_name}")

    # ---- Write per_policy_results.csv ----
    fieldnames = [
        "scenario_id", "method_name", "split", "status", PRIMARY,
        "completion_fraction", "n_steps", "peak_kv",
        "n_llf_steps", "n_reserve_steps", "transition_count",
        "kv_util_at_transition_mean", "n_admission_decisions_differ_from_both_parents",
        "tau_urgent",
    ]
    rows_out = []
    for sid in all_ids:
        sp = split_of(sid)
        for m in PARENT_METHODS:
            r = parent_results.get((sid, m), {})
            rows_out.append({
                "scenario_id": sid, "method_name": m, "split": sp,
                "status": r.get("status", "missing"), PRIMARY: r.get("anwg", float("nan")),
                "completion_fraction": r.get("completion_fraction", float("nan")),
                "n_steps": r.get("n_steps", 0),
                "peak_kv": r.get("peak_kv", float("nan")),
            })
        crow = child_rows.get(sid, {})
        rows_out.append({
            "scenario_id": sid, "method_name": CHILD_METHOD, "split": sp,
            "status": crow.get("status", "missing"), PRIMARY: crow.get("anwg", float("nan")),
            "completion_fraction": crow.get("completion_fraction", float("nan")),
            "n_steps": crow.get("n_steps", 0),
            "peak_kv": crow.get("peak_kv", float("nan")),
            "n_llf_steps": crow.get("n_llf_steps", 0),
            "n_reserve_steps": crow.get("n_reserve_steps", 0),
            "transition_count": crow.get("transition_count", 0),
            "kv_util_at_transition_mean": crow.get("kv_util_at_transition_mean", 0.0),
            "n_admission_decisions_differ_from_both_parents": disagree_counts.get(sid, 0),
            "tau_urgent": best_tau,
        })
        hrow = hysteresis_child_rows.get(sid, {})
        rows_out.append({
            "scenario_id": sid, "method_name": CHILD_METHOD_HYSTERESIS, "split": sp,
            "status": hrow.get("status", "missing"), PRIMARY: hrow.get("anwg", float("nan")),
            "completion_fraction": hrow.get("completion_fraction", float("nan")),
            "n_steps": hrow.get("n_steps", 0),
            "peak_kv": hrow.get("peak_kv", float("nan")),
            "n_llf_steps": hrow.get("n_llf_steps", 0),
            "n_reserve_steps": hrow.get("n_reserve_steps", 0),
            "transition_count": hrow.get("transition_count", 0),
            "kv_util_at_transition_mean": hrow.get("kv_util_at_transition_mean", 0.0),
            "n_admission_decisions_differ_from_both_parents": disagree_counts_hyst.get(sid, 0),
            "tau_urgent": best_tau,
        })
        for method, scores in (
            ("contextual_top1", selector_scores), ("hard_conditional", hard_scores),
            ("best_fixed_parent", best_fixed_scores), ("parent_oracle", envelope),
            ("oracle_after_child", oracle_after_child),
            ("oracle_after_hysteresis_child", oracle_after_hysteresis),
        ):
            rows_out.append({
                "scenario_id": sid, "method_name": method, "split": sp,
                "status": "success", PRIMARY: scores.get(sid, 0.0),
                "completion_fraction": float("nan"), "n_steps": 0,
                "peak_kv": float("nan"),
            })

    with open(args.run_dir / "per_policy_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    with open(args.run_dir / "selector_choice.json", "w") as f:
        json.dump(selector_choice, f, indent=2)

    n_failed_total = n_parent_failed + n_child_failed + n_hyst_failed
    summary = {
        "experiment": "kv_composition_falsification_v1",
        "n_scenarios": len(scenarios),
        "n_rows": len(rows_out),
        "n_failed": n_failed_total,
        "primary_metric": PRIMARY,
        "tau_urgent_frozen": int(best_tau),
        "best_fixed_parent_name": best_fixed_name,
        "selector_meta": selector_meta,
        "splits": {k: len(getattr(split, k)) for k in ("train", "val", "test", "ood")},
        "git_head": _git_head(),
        "git_branch": _git_branch(),
    }
    with open(args.run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log(args.run_dir, f"Done. {json.dumps(summary, default=str)}")


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
