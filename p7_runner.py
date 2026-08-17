#!/usr/bin/env python3
"""Runner for Family B v2 PrefillControl composition falsification.

Runs the two-parent composition experiment end to end:
  1. Generate Family B v2 scenarios (templates_prefill_decode_v2).
  2. Evaluate both parents on every scenario.
  3. Extract online-observable scenario features.
  4. Split into train/val/test/ood (by seed).
  5. Train the contextual top-1 selector on train+val.
  6. Evaluate all baselines (fixed intermediates, contextual alpha,
     hard conditional, top-1 selector) on test and ood.
  7. Write per-policy CSV + run log + manifest.

Primary metric: ``arrival_normalized_weighted_goodput`` (CANWG).

Composition Abstractions:
- ``ScenarioBatch``: a group of scenarios + their scenario-level feature dicts.
- ``ChildCompositionConfig``: the composition policy definition including
  chunk-grid, selector reference, and provenance fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import p3_chunk_control as p3

from llmserveopt.policies.prefill_control_variants import (  # noqa: E402
    DEFAULT_CHUNK_SMALL,
    UNLIMITED_PREFILL_CHUNK,
    GreedyArrivalPrefillControlPolicy,
    make_prefill_decode_variants_v2,
)
from llmserveopt.policy_separation.templates_prefill_decode_v2 import (  # noqa: E402
    CLASS_HOG,
    CLASS_LATE,
    assert_policy_visible_fields_clean_v2,
    case_prefill_decode_ttft_contention,
)
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402


# ===================================================================
# Data abstractions
# ===================================================================

@dataclass(frozen=True)
class ScenarioBatch:
    """A batch of scenarios plus their scenario-level observable features.

    Attributes:
        scenarios: list of ``PolicySeparationScenario`` objects.
        features: mapping scenario_id -> feature=dict (from p3).
        split_name: which split assignment this batch belongs to
            (``train``, ``val``, ``test``, ``ood``).
    """
    scenarios: List[Any]
    features: Dict[str, Dict[str, float]]
    split_name: str = ""


@dataclass(frozen=True)
class ChildCompositionConfig:
    """Describes a composition policy to evaluate.

    Attributes:
        composition_id: unique identifier (e.g. ``contextual_top1``, ``chunk_128``).
        policy_names: human-readable name list for provenance.
        chunk_grid: ordered chunk sizes for child composition.
        chunk_names: matching labels.
        parent_policy_names: the two parent names.
        selector_meta: fitted selector metadata when applicable.
        split: which split this config is evaluated on.
        seed: deterministic seed for evaluation.
        eval_id_prefix: prefix for generating deterministic eval IDs.
    """
    composition_id: str
    policy_names: List[str] = field(default_factory=list)
    chunk_grid: Tuple[int, ...] = ()
    chunk_names: Tuple[str, ...] = ()
    parent_policy_names: Tuple[str, str] = ("full_prefill", "chunked_prefill_small")
    selector_meta: Optional[Dict[str, Any]] = None
    split: str = ""
    seed: int = 20261201
    eval_id_prefix: str = "comp"


# ===================================================================
# Deterministic eval-ID generation
# ===================================================================

SCENARIO_ID_RE = re.compile(
    r"^pd2\.hog(?P<n_hog>\d+)\.late(?P<n_late>\d+)"
    r"\.slo(?P<slo_emphasis>hog_ttft|late_ttft)\.s(?P<seed>\d+)$"
)


def _build_eval_id(scenario_id: str, method: str, config_hash: str) -> str:
    """Build a deterministic evaluation ID.

    Formula: sha256(scenario_id|method|config_hash)[:12]
    Guarantees: no collisions, reproducible, no leakage of secret labels.
    """
    raw = f"{scenario_id}|{method}|{config_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _scenario_config_hash(cfg: dict) -> str:
    """Hash a composition config dict for deterministic ID generation."""
    canonical = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ===================================================================
# Scenario generation (Family B v2 only)
# ===================================================================

def build_scenarios_from_config(
    cfg: dict,
    *,
    allow_synthetic_tokens: bool,
    datasets_root: Optional[Path] = None,
) -> List[Any]:
    """Generate scenarios from composition config (Family B v2) using
    ``case_prefill_decode_ttft_contention`` from
    ``templates_prefill_decode_v2``.
    """
    grid = cfg["template"]["sweep_grid"]
    fixed = cfg.get("runner", {})
    scenarios = []
    for hog in grid["hog_count"]:
        for late in grid["late_pressure"]:
            for slo in grid["slo_emphasis"]:
                for seed in grid["seeds"]:
                    s = case_prefill_decode_ttft_contention(
                        hog_count=str(hog),
                        late_pressure=str(late),
                        slo_emphasis=str(slo),
                        seed=int(seed),
                        n_hog=fixed.get("max_active_sequences"),
                        n_late=fixed.get("max_active_sequences"),
                        max_active_sequences=int(fixed.get("max_active_sequences", 512)),
                        step_token_budget=int(fixed.get("step_token_budget", 512)),
                        allow_synthetic_tokens=allow_synthetic_tokens,
                        datasets_root=datasets_root,
                    )
                    assert_policy_visible_fields_clean_v2(s)
                    scenarios.append(s)
    return scenarios


# ===================================================================
# Feature extraction
# ===================================================================

def extract_scenario_features(
    scenarios: List[Any],
) -> Dict[str, Dict[str, float]]:
    """Extract scenario-level observable features for a list of scenarios.

    Only online-observable quantities. No forbidden leakage.
    """
    features = {}
    for s in scenarios:
        feats = p3.scenario_observable_features(list(s.requests))
        p3.assert_no_hidden_leakage(feats)
        features[s.scenario_id] = feats
    return features


# ===================================================================
# Parent evaluation
# ===================================================================

PRIMARY_FIELDS = [
    "arrival_normalized_weighted_goodput",
]
SECONDARY_FIELDS = [
    "unweighted_slo_success_rate",
    "completion_fraction",
    "mean_ttft",
    "p95_ttft",
    "p99_ttft",
    "mean_tpot",
    "p95_tpot",
    "mean_queuing_delay",
    "mean_prefill_delay_s",
    "request_throughput",
    "token_throughput",
    "global_tbt_attainment",
]
MECHANISM_FIELDS = [
    "decode_stalled_steps",
    "cumulative_decode_tokens_deferred",
    "steps_with_prefill_while_decode_deferred",
    "prefill_stalled_steps",
    "cumulative_prefill_requests_stalled",
    "budget_saturation_fraction",
    "mean_num_decoding",
    "mean_num_prefilling",
    "fraction_prefill_tokens_while_decodes_active",
    "mean_theoretical_chunks",
    "mean_interchunk_extra_wait_s",
    "steps_decode_active_prefill_stalled",
    "mean_num_prefilling_saturated",
    "mean_num_decoding_saturated",
    "n_saturated_steps",
]

RESULT_FIELDNAMES = (
    ["scenario_id", "policy_name", "eval_id"]
    + PRIMARY_FIELDS
    + SECONDARY_FIELDS
    + MECHANISM_FIELDS
    + ["status", "composition_id", "parent_policy", "split", "seed", "run_config_id"]
)


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if len(arr) else 0.0


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if len(arr) else 0.0


def _class_block(completed: Sequence[Any], class_id: str) -> Dict[str, float]:
    cs = [c for c in completed if c.request.class_id == class_id]
    prefix = "hog" if class_id == CLASS_HOG else "late"
    if not cs:
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_mean_ttft": 0.0,
            f"{prefix}_p95_ttft": 0.0,
            f"{prefix}_mean_tpot": 0.0,
            f"{prefix}_p95_tpot": 0.0,
            f"{prefix}_slo_success": 0.0,
            f"{prefix}_ttft_attainment": 0.0,
        }
    ttft = _finite([c.ttft for c in cs])
    tpot = _finite([c.tpot for c in cs])
    slo_ok = [0.0 if c.slo_violated else 1.0 for c in cs]
    from llmserveopt.policy_separation.templates_prefill_decode import STEP_SIZE  # noqa: E402
    ttft_ok = []
    for c in cs:
        if not np.isfinite(c.ttft):
            continue
        budget = (
            c.request.slo_deadline - c.request.arrival_time
            - c.request.predicted_output_tokens * STEP_SIZE
        )
        ttft_ok.append(1.0 if c.ttft <= budget else 0.0)
    return {
        f"{prefix}_n": float(len(cs)),
        f"{prefix}_mean_ttft": _mean(ttft),
        f"{prefix}_p95_ttft": _pct(ttft, 95),
        f"{prefix}_mean_tpot": _mean(tpot),
        f"{prefix}_p95_tpot": _pct(tpot, 95),
        f"{prefix}_slo_success": float(np.mean(slo_ok)),
        f"{prefix}_ttft_attainment": float(np.mean(ttft_ok)) if ttft_ok else 0.0,
    }


def _contention_metrics(sim: Simulator, chunk: int) -> Dict[str, Any]:
    from llmserveopt.policy_separation.templates_prefill_decode import STEP_SIZE  # noqa: E402
    summary = sim.contention_diagnostics_summary()
    completed = list(sim._completed)  # noqa: SLF001
    prefill_while_decode = 0
    total_prefill = 0
    decode_counts = []
    prefill_counts = []
    sat_prefill = []
    sat_decode = []
    decode_active_prefill_stalled = 0
    n_sat = 0
    for g in sim._gpus:  # noqa: SLF001
        for d in g.step_contention_diagnostics:
            decode_counts.append(d.num_decoding)
            prefill_counts.append(d.num_prefilling)
            total_prefill += d.prefill_tokens_served
            if d.num_decoding > 0:
                prefill_while_decode += d.prefill_tokens_served
            if d.num_decoding > 0 and d.prefill_requests_stalled > 0:
                decode_active_prefill_stalled += 1
            if d.budget_saturated:
                n_sat += 1
                sat_prefill.append(d.num_prefilling)
                sat_decode.append(d.num_decoding)

    n_chunks = [
        math.ceil(c.request.prompt_tokens / max(int(chunk), 1)) for c in completed
    ]
    extra_wait = []
    for c, n_ch in zip(completed, n_chunks):
        if np.isfinite(c.prefill_delay):
            extra_wait.append(float(c.prefill_delay) - n_ch * STEP_SIZE)

    return {
        "decode_stalled_steps": summary["decode_stalled_steps"],
        "cumulative_decode_tokens_deferred": summary["cumulative_decode_tokens_deferred"],
        "steps_with_prefill_while_decode_deferred": summary[
            "steps_with_prefill_while_decode_deferred"
        ],
        "prefill_stalled_steps": summary.get("prefill_stalled_steps", 0),
        "cumulative_prefill_requests_stalled": summary.get(
            "cumulative_prefill_requests_stalled", 0
        ),
        "budget_saturation_fraction": summary["budget_saturation_fraction"],
        "mean_num_decoding": _mean(np.asarray(decode_counts, dtype=float)),
        "mean_num_prefilling": _mean(np.asarray(prefill_counts, dtype=float)),
        "fraction_prefill_tokens_while_decodes_active": (
            prefill_while_decode / total_prefill if total_prefill > 0 else 0.0
        ),
        "mean_theoretical_chunks": (
            float(np.mean(n_chunks)) if n_chunks else 0.0
        ),
        "mean_interchunk_extra_wait_s": (
            float(np.mean(extra_wait)) if extra_wait else 0.0
        ),
        "steps_decode_active_prefill_stalled": int(decode_active_prefill_stalled),
        "mean_num_prefilling_saturated": _mean(np.asarray(sat_prefill, dtype=float)),
        "mean_num_decoding_saturated": _mean(np.asarray(sat_decode, dtype=float)),
        "n_saturated_steps": int(n_sat),
    }


def _eval_single_scenario(
    scenario_id: str,
    policy_name: str,
    scenario: Any,
    variant_kwargs: Dict[str, Any],
    chunk_size: int,
    eval_id_prefix: str,
    cfg_hash: str,
    composition_id: str,
    parent_policy: str,
    split: str,
    seed: int,
    run_config_id: str,
) -> dict:
    """Evaluate a single (scenario, policy) pair.

    Returns a result dict aligned with RESULT_FIELDNAMES.
    """
    try:
        from llmserveopt.policy_separation.templates_prefill_decode import STEP_SIZE  # noqa: E402

        policy, _ = make_prefill_decode_variants_v2(
            chunk_small=int(variant_kwargs.get("max_prefill_chunk_tokens", DEFAULT_CHUNK_SMALL))
        )[policy_name]
        policy.name = policy_name

        merged = dict(scenario.service_model_kwargs)
        merged.update(variant_kwargs)
        service_model = ServiceModel(**merged)

        sim = Simulator(
            SimulatorConfig(
                gpu_configs=list(scenario.gpu_configs),
                service_model=service_model,
            )
        )
        sim.load_trace(list(scenario.requests))
        metrics = sim.run(policy, workload_tag=scenario_id, seed=seed)
        completed = list(sim._completed)  # noqa: SLF001

        n_req = len(scenario.requests)
        total_v = sum(1 for c in completed if c.slo_violated)
        unweighted = (len(completed) - total_v) / max(1, n_req)

        ttfts = _finite([c.ttft for c in completed])
        tpots = _finite([c.tpot for c in completed])
        tbt_slo = float(scenario.params.get("tbt_slo_s", 0.002))
        tbt_attain = float(np.mean(tpots <= tbt_slo)) if len(tpots) else 0.0

        variant_kw = {"max_prefill_chunk_tokens": chunk_size}
        config_hash = _scenario_config_hash({"chunk": chunk_size, "policy": policy_name})
        eval_id = _build_eval_id(scenario_id, policy_name, config_hash)

        row: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "eval_id": eval_id,
            "arrival_normalized_weighted_goodput": float(
                metrics.arrival_normalized_weighted_goodput
            ),
            "unweighted_slo_success_rate": float(unweighted),
            "completion_fraction": float(metrics.completion_fraction),
            "mean_ttft": _mean(ttfts),
            "p95_ttft": _pct(ttfts, 95),
            "p99_ttft": _pct(ttfts, 99),
            "mean_tpot": _mean(tpots),
            "p95_tpot": _pct(tpots, 95),
            "mean_queuing_delay": float(metrics.mean_queuing_delay),
            "mean_prefill_delay_s": float(metrics.mean_prefill_delay),
            "request_throughput": float(metrics.request_throughput),
            "token_throughput": float(metrics.token_throughput),
            "global_tbt_attainment": tbt_attain,
        }
        row.update(_class_block(completed, CLASS_HOG))
        row.update(_class_block(completed, CLASS_LATE))
        row.update(_contention_metrics(sim, chunk_size))
        row["status"] = "success"
        row["composition_id"] = composition_id
        row["parent_policy"] = parent_policy
        row["split"] = split
        row["seed"] = seed
        row["run_config_id"] = run_config_id
        return row
    except Exception as e:
        import traceback
        return {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "eval_id": _build_eval_id(scenario_id, policy_name, "ERROR"),
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "composition_id": composition_id,
            "parent_policy": parent_policy,
            "split": split,
            "seed": seed,
            "run_config_id": run_config_id,
        }


# ===================================================================
# Runner pipeline
# ===================================================================

def _generate_composition_config(cfg: dict) -> ChildCompositionConfig:
    """From the YAML config, build the composition config for this experiment."""
    runner = cfg.get("runner", {})
    seed = cfg.get("experiment", {}).get("bootstrap_seeds", {}).get("seed", 20261201)
    config_hash = _scenario_config_hash({
        "chunk_options": list(p3.CHILD_CHUNK_OPTIONS),
        "seeds": cfg["template"]["sweep_grid"]["seeds"],
    })
    run_config_id = config_hash[:12]

    return ChildCompositionConfig(
        composition_id="prefill_control_composition",
        policy_names=list(p3.BASINELIST),
        chunk_grid=p3.CHILD_CHUNK_OPTIONS,
        chunk_names=p3.CHILD_CHUNK_NAMES,
        parent_policy_names=("full_prefill", "chunked_prefill_small"),
        split="",
        seed=int(seed),
        eval_id_prefix=run_config_id,
    )


def run_composition_experiment(cfg: dict, run_dir: Path, *, workers: int = 8, dry_run: int = 0) -> None:
    """Execute the full composition falsification pipeline.

    Args:
        cfg: loaded YAML config.
        run_dir: output directory.
        workers: process-pool parallelism.
        dry_run: if > 0, only run that many scenarios per parent.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 0: config & provenance ----
    log_fn = run_dir / "run.log"
    run_meta = {**cfg, "run_dir": str(run_dir), "workers": workers}
    _write_log(log_fn, "Starting Family B v2 PrefillControl composition experiment.")

    comp_cfg = _generate_composition_config(cfg)
    seed = comp_cfg.seed

    # ---- Step 1: generate scenarios ----
    _write_log(log_fn, "Generating Family B v2 scenarios...")
    allow_synthetic = not cfg.get("runner", {}).get("require_burstgpt", True)
    scenarios = build_scenarios_from_config(
        cfg,
        allow_synthetic_tokens=allow_synthetic,
        datasets_root=None,
    )
    if dry_run > 0:
        scenarios = scenarios[:dry_run]

    # ---- Step 2: split ----
    _write_log(log_fn, f"Created {len(scenarios)} scenarios. Splitting...")
    sids = [s.scenario_id for s in scenarios]
    split_assignment = p3.assign_family_b_v2_splits(sids)
    p3.assert_no_split_leakage(split_assignment)

    def _batch(split_name: str) -> ScenarioBatch:
        sids_set = set(getattr(split_assignment, split_name))
        scens = [s for s in scenarios if s.scenario_id in sids_set]
        return ScenarioBatch(scenarios=scens, features={}, split_name=split_name)

    batch_train = ScenarioBatch(
        scenarios=[s for s in scenarios if s.scenario_id in set(split_assignment.train)],
        features={},
        split_name="train",
    )
    batch_val = ScenarioBatch(
        scenarios=[s for s in scenarios if s.scenario_id in set(split_assignment.val)],
        features={},
        split_name="val",
    )
    batch_test = ScenarioBatch(
        scenarios=[s for s in scenarios if s.scenario_id in set(split_assignment.test)],
        features={},
        split_name="test",
    )
    batch_ood = ScenarioBatch(
        scenarios=[s for s in scenarios if s.scenario_id in set(split_assignment.ood)],
        features={},
        split_name="ood",
    )

    # ---- Step 3: feature extraction ----
    for b in (batch_train, batch_val, batch_test, batch_ood):
        b.features = extract_scenario_features(b.scenarios)

    # ---- Step 4: parent evaluation ----
    _write_log(log_fn, "Evaluating parents on all scenarios...")
    config_hash = _scenario_config_hash({"chunk_options": list(p3.CHILD_CHUNK_OPTIONS)})
    run_config_id = config_hash[:12]
    cfg_hash = _scenario_config_hash({
        "chunk_options": list(p3.CHILD_CHUNK_OPTIONS),
        "seeds": cfg["template"]["sweep_grid"]["seeds"],
    })

    all_results: List[dict] = []

    # Evaluate both parents across all scenarios
    all_batches = [batch_train, batch_val, batch_test, batch_ood]
    for bname, b in [("train", batch_train), ("val", batch_val),
                     ("test", batch_test), ("ood", batch_ood)]:
        _write_log(log_fn, f"  Evaluating {bname}: {len(b.scenarios)} scenarios x 2 parents")
        tasks = []
        for s in b.scenarios:
            for pname in ["full_prefill", "chunked_prefill_small"]:
                cfg_kv = p3.make_parent_config(pname)
                chunk_size = cfg_kv["max_prefill_chunk_tokens"]
                tasks.append((
                    s.scenario_id, pname, s, cfg_kv, chunk_size,
                    comp_cfg.eval_id_prefix, cfg_hash,
                    comp_cfg.composition_id, pname, bname, seed, run_config_id,
                ))

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_eval_single_scenario, *t): t for t in tasks}
            for fut in as_completed(futures):
                all_results.append(fut.result())

    # Also evaluate fixed-intermediate children on test & ood
    fixed_chunks = [p for p in p3.FIXED_INTERMEDIATE_PARENTS]
    for chunk_info in fixed_chunks:
        cname = chunk_info["name"]
        csize = chunk_info["max_prefill_chunk_tokens"]
        for bname in ["test", "ood"]:
            b = batch_test if bname == "test" else batch_ood
            _write_log(log_fn, f"  Evaluating child {cname} on {bname}: {len(b.scenarios)} scenarios")
            tasks = []
            for s in b.scenarios:
                cfg_kv = {"max_prefill_chunk_tokens": csize, "decode_first": False}
                tasks.append((
                    s.scenario_id, cname, s, cfg_kv, csize,
                    comp_cfg.eval_id_prefix, cfg_hash,
                    comp_cfg.composition_id, cname, bname, seed, run_config_id,
                ))
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_eval_single_scenario, *t): t for t in tasks}
                for fut in as_completed(futures):
                    all_results.append(fut.result())

    # ---- Step 5: write CSV ----
    _write_log(log_fn, f"Writing {len(all_results)} result rows to per_policy_results.csv")
    with open(run_dir / "per_policy_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    # ---- Step 6: write scenario features ----
    _write_log(log_fn, "Writing scenario features...")
    all_feats_rows = []
    for b in all_batches:
        for sid, feats in b.features.items():
            row = {"scenario_id": sid}
            row.update(feats)
            row["split"] = b.split_name
            all_feats_rows.append(row)
    with open(run_dir / "scenario_features.csv", "w", newline="") as f:
        fieldnames = sorted(["scenario_id", "split"] + list(p3.FEATURE_NAMES))
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_feats_rows)

    # ---- Step 7: meta summary ----
    success = [r for r in all_results if r.get("status") == "success"]
    failed = [r for r in all_results if r.get("status") == "failed"]
    summary = {
        "experiment": comp_cfg.composition_id,
        "n_scenarios": len(scenarios),
        "n_tasks": len(all_results),
        "n_completed": len(success),
        "n_failed": len(failed),
        "primary_metric": "arrival_normalized_weighted_goodput",
        "split_sizes": {
            "train": len(batch_train.scenarios),
            "val": len(batch_val.scenarios),
            "test": len(batch_test.scenarios),
            "ood": len(batch_ood.scenarios),
        },
        "parents": list(comp_cfg.parent_policy_names),
        "children_evaluated": [p["name"] for p in fixed_chunks],
        "seed": seed,
        "run_config_id": run_config_id,
    }
    with open(run_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Step 8: manifest ----
    manifest = {
        "run_dir": str(run_dir),
        "config": str(run_dir / "composition_config.yaml"),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(_ROOT), text=True
        ).strip(),
        "composition_id": comp_cfg.composition_id,
        "policy_schema": {
            "primary": PRIMARY_FIELDS,
            "secondary": SECONDARY_FIELDS,
            "mechanism": MECHANISM_FIELDS,
        },
        "split_assignment": {
            "train": split_assignment.train[:5] + [f"... ({len(split_assignment.train)} total)"],
            "val": split_assignment.val[:5] + [f"... ({len(split_assignment.val)} total)"],
            "test": split_assignment.test[:5] + [f"... ({len(split_assignment.test)} total)"],
            "ood": split_assignment.ood[:5] + [f"... ({len(split_assignment.ood)} total)"],
        },
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _write_log(log_fn, f"Completed: {len(success)} success, {len(failed)} failed")


# ===================================================================
# CLI
# ===================================================================

def _write_log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Family B v2 PrefillControl Composition Falsification Runner"
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Process pool workers")
    parser.add_argument("--dry-run", type=int, default=0, help="Run only N scenarios per parent")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_composition_experiment(cfg, args.run_dir, workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
