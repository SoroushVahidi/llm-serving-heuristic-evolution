#!/usr/bin/env python3
"""Runner for Policy Separation Family B v2 (TTFT-contention refinement).

Primary metric: canonical RunMetrics.arrival_normalized_weighted_goodput.
Policies: make_prefill_decode_variants_v2() — full_prefill vs
chunked_prefill_small only.

See docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md.
Production mode requires staged BurstGPT (``--require-burstgpt``, default True).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policies.prefill_control_variants import (  # noqa: E402
    DEFAULT_CHUNK_SMALL,
    make_prefill_decode_variants_v2,
)
from llmserveopt.policy_separation.templates_prefill_decode import (  # noqa: E402
    BurstGPTUnavailableError,
    STEP_SIZE,
    resolve_burstgpt_path,
)
from llmserveopt.policy_separation.templates_prefill_decode_v2 import (  # noqa: E402
    CLASS_HOG,
    CLASS_LATE,
    assert_policy_visible_fields_clean_v2,
    case_prefill_decode_ttft_contention,
)
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

# Metric schema. Keep primary / secondary / mechanism columns distinct.
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
    "hog_n",
    "late_n",
    "hog_mean_ttft",
    "hog_p95_ttft",
    "hog_mean_tpot",
    "hog_p95_tpot",
    "hog_slo_success",
    "hog_ttft_attainment",
    "late_mean_ttft",
    "late_p95_ttft",
    "late_mean_tpot",
    "late_p95_tpot",
    "late_slo_success",
    "late_ttft_attainment",
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
    ["scenario_id", "policy_name"]
    + PRIMARY_FIELDS
    + SECONDARY_FIELDS
    + MECHANISM_FIELDS
    + ["status"]
)


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def build_scenarios_from_config(
    cfg: dict,
    *,
    allow_synthetic_tokens: bool,
    datasets_root: Optional[Path] = None,
) -> List[Any]:
    grid = cfg["sweep_grid"]
    fixed = cfg.get("fixed", {})
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
                        n_hog=fixed.get("n_hog"),
                        n_late=fixed.get("n_late"),
                        max_active_sequences=int(
                            fixed.get("max_active_sequences", 512)
                        ),
                        step_token_budget=int(
                            fixed.get("step_token_budget", 512)
                        ),
                        allow_synthetic_tokens=allow_synthetic_tokens,
                        datasets_root=datasets_root,
                    )
                    assert_policy_visible_fields_clean_v2(s)
                    scenarios.append(s)
    return scenarios


def _mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if len(arr) else 0.0


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if len(arr) else 0.0


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _class_block(completed: Sequence[Any], class_id: str, step_size: float) -> Dict[str, float]:
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
    ttft_ok = []
    for c in cs:
        if not np.isfinite(c.ttft):
            continue
        budget = (
            c.request.slo_deadline
            - c.request.arrival_time
            - c.request.predicted_output_tokens * step_size
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


def _contention_metrics(sim: Simulator, completed: Sequence[Any], chunk: int) -> Dict[str, Any]:
    summary = sim.contention_diagnostics_summary()
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


def _run_one_task(args: Tuple[str, str, Any, Dict[str, Any], Dict[str, int]]) -> dict:
    scenario_id, policy_name, scenario, variant_kwargs, chunk_budgets = args
    try:
        from llmserveopt.policies.prefill_control_variants import (
            make_prefill_decode_variants_v2,
        )

        variants = make_prefill_decode_variants_v2(**chunk_budgets)
        policy, _ = variants[policy_name]
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
        metrics = sim.run(policy, workload_tag=scenario_id, seed=scenario.seed)
        completed = sim._completed  # noqa: SLF001

        n_req = len(scenario.requests)
        total_v = sum(1 for c in completed if c.slo_violated)
        unweighted = (len(completed) - total_v) / max(1, n_req)

        ttfts = _finite([c.ttft for c in completed])
        tpots = _finite([c.tpot for c in completed])
        tbt_slo = float(scenario.params.get("tbt_slo_s", 0.002))
        tbt_attain = float(np.mean(tpots <= tbt_slo)) if len(tpots) else 0.0

        chunk = int(variant_kwargs["max_prefill_chunk_tokens"])
        row = {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
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
        row.update(_class_block(completed, CLASS_HOG, STEP_SIZE))
        row.update(_class_block(completed, CLASS_LATE, STEP_SIZE))
        row.update(_contention_metrics(sim, completed, chunk))
        row["status"] = "success"
        return row
    except Exception as e:
        import traceback

        return {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, list):
        cfg = cfg[0]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-synthetic-tokens",
        action="store_true",
        help="Permit synthetic lengths (tests/local only). Production must omit.",
    )
    parser.add_argument(
        "--require-burstgpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast if BurstGPT cannot be resolved (default: true).",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help="Optional datasets root containing burstgpt_v2/raw/ (e.g. .local_data).",
    )
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(args.config)
    datasets_root = args.datasets_root
    require_burstgpt = bool(args.require_burstgpt) and not args.allow_synthetic_tokens
    if cfg.get("pilot_metadata", {}).get("require_burstgpt", True) and require_burstgpt:
        path = resolve_burstgpt_path(datasets_root=datasets_root)
        if path is None:
            raise SystemExit(
                "Family B v2 production requires BurstGPT; none resolved. "
                "Set LLM_SERVEOPT_BURSTGPT_CSV, pass --datasets-root, or stage "
                "burstgpt_v2/raw shards."
            )
        _log(args.run_dir, f"BurstGPT resolved: {path}")

    allow_synthetic = bool(args.allow_synthetic_tokens) and not require_burstgpt
    _log(
        args.run_dir,
        f"Starting Family B v2 (dry_run={args.dry_run}, "
        f"allow_synthetic_tokens={allow_synthetic})",
    )

    try:
        scenarios = build_scenarios_from_config(
            cfg,
            allow_synthetic_tokens=allow_synthetic,
            datasets_root=datasets_root,
        )
    except BurstGPTUnavailableError as e:
        raise SystemExit(str(e)) from e

    if args.dry_run:
        scenarios = scenarios[:4]

    with open(args.run_dir / "scenarios.jsonl", "w") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_manifest_dict()) + "\n")

    features_rows = []
    for s in scenarios:
        features_rows.append(
            {
                "scenario_id": s.scenario_id,
                **{
                    k: s.params.get(k)
                    for k in (
                        "hog_count",
                        "late_pressure",
                        "slo_emphasis",
                        "n_total_jobs",
                        "n_hog",
                        "n_late",
                        "step_token_budget",
                        "max_active_sequences",
                        "hog_prompt_median",
                        "late_prompt_median",
                        "output_median",
                        "late_start_s",
                        "slack_hog_s",
                        "slack_late_s",
                        "tbt_slo_s",
                        "arrival_shape",
                        "output_intervention",
                    )
                },
                "seed": s.seed,
                "stress_control_relationship": s.stress_control_relationship,
                "token_sources": json.dumps(s.params.get("token_sources")),
                "mean_e2e_slack_hog": float(
                    np.mean(
                        [
                            r.slo_deadline - r.arrival_time
                            for r in s.requests
                            if r.class_id == CLASS_HOG
                        ]
                    )
                ),
                "mean_e2e_slack_late": float(
                    np.mean(
                        [
                            r.slo_deadline - r.arrival_time
                            for r in s.requests
                            if r.class_id == CLASS_LATE
                        ]
                    )
                ),
            }
        )
    with open(args.run_dir / "scenario_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(features_rows[0].keys()))
        w.writeheader()
        w.writerows(features_rows)

    _log(args.run_dir, f"Generated {len(scenarios)} scenarios.")

    chunk_cfg = cfg.get("chunk_budgets", {})
    chunk_budgets = {
        "chunk_small": int(chunk_cfg.get("chunk_small", DEFAULT_CHUNK_SMALL)),
    }
    variants = make_prefill_decode_variants_v2(**chunk_budgets)
    policy_names = list(cfg["policies"])
    forbidden = {
        "chunked_prefill_large",
        "decode_priority_chunked",
        "adaptive_prefill_control",
    }
    if set(policy_names) & forbidden:
        raise SystemExit(f"Family B v2 forbids twin policies: {policy_names}")
    if set(policy_names) != {"full_prefill", "chunked_prefill_small"}:
        raise SystemExit(
            f"Family B v2 policy set must be the two anchors, got {policy_names}"
        )
    tasks: List[Tuple[str, str, Any, Dict[str, Any], Dict[str, int]]] = []
    for scen in scenarios:
        for name in policy_names:
            if name not in variants:
                raise KeyError(f"Unknown mechanism variant {name!r}")
            _, kwargs = variants[name]
            tasks.append((scen.scenario_id, name, scen, kwargs, chunk_budgets))
    _log(args.run_dir, f"Total tasks to run: {len(tasks)}")
    _log(args.run_dir, f"Chunk budgets: {chunk_budgets}")

    results: List[dict] = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one_task, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 8 == 0 or done == len(tasks):
                _log(args.run_dir, f"Completed {done}/{len(tasks)} tasks.")

    elapsed = time.time() - start
    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    _log(args.run_dir, f"Successes={len(success)} Failures={len(failed)} elapsed={elapsed:.2f}s")

    if success:
        with open(args.run_dir / "per_policy_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerows(success)
    if failed:
        with open(args.run_dir / "failures.jsonl", "w") as f:
            for r in failed:
                f.write(json.dumps(r) + "\n")

    token_sources = {
        (s.params.get("token_sources") or {}).get("hog_prompt") for s in scenarios
    }
    summary = {
        "experiment_name": cfg["pilot_metadata"]["experiment_name"],
        "generator_version": "prefill_decode_v2",
        "n_scenarios": len(scenarios),
        "n_tasks": len(tasks),
        "n_completed": len(success),
        "n_failed": len(failed),
        "elapsed_seconds": elapsed,
        "primary_metric": "arrival_normalized_weighted_goodput",
        "token_sources_observed": sorted(x for x in token_sources if x is not None),
        "require_burstgpt": require_burstgpt,
        "allow_synthetic_tokens": allow_synthetic,
        "policies": policy_names,
        "chunk_budgets": chunk_budgets,
        "output_intervention": "synthetic_short_output_for_ttft_isolation",
    }
    with open(args.run_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    import subprocess

    burst = resolve_burstgpt_path(datasets_root=datasets_root)
    manifest = {
        "run_dir": str(args.run_dir),
        "config": str(args.config),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
        "burstgpt_path": str(burst) if burst else None,
        "datasets_root": str(datasets_root) if datasets_root else None,
        "chunk_budgets": chunk_budgets,
        "metric_schema": {
            "primary": PRIMARY_FIELDS,
            "secondary": SECONDARY_FIELDS,
            "mechanism": MECHANISM_FIELDS,
        },
    }
    with open(args.run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _log(args.run_dir, "Completed Family B v2 execution.")


if __name__ == "__main__":
    main()
