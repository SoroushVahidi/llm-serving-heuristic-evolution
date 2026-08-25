#!/usr/bin/env python3
"""Runner for Policy Separation Family B v1 (prefill/decode chunk-control).

Primary metric: canonical RunMetrics.arrival_normalized_weighted_goodput.
Each (scenario, policy) task runs the Phase 1.5 simulator with a ServiceModel
built from the scenario's base kwargs merged with the mechanism variant's
kwargs (chunk size / decode-first), so the fixed variants A/B/C differ ONLY
in execution mechanism.

Production mode requires staged BurstGPT (``--require-burstgpt``, default True).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policies.prefill_control_variants import (
    DEFAULT_CHUNK_LARGE,
    DEFAULT_CHUNK_SMALL,
    DEFAULT_DECODE_PRIORITY_CHUNK,
    make_prefill_decode_variants,
)
from llmserveopt.policy_separation.templates_prefill_decode import (
    BurstGPTUnavailableError,
    assert_policy_visible_fields_clean,
    case_prefill_decode_interference,
    resolve_burstgpt_path,
)
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

RESULT_FIELDNAMES = [
    "scenario_id",
    "policy_name",
    "arrival_normalized_weighted_goodput",
    "unweighted_slo_success_rate",
    "completion_fraction",
    "mean_ttft",
    "p95_ttft",
    "p99_ttft",
    "mean_tpot",
    "p95_tpot",
    "mean_queuing_delay",
    "mean_prefill_delay_s",
    "ttft_attainment",
    "tbt_attainment",
    "request_throughput",
    "token_throughput",
    "decode_stalled_steps",
    "cumulative_decode_tokens_deferred",
    "steps_with_prefill_while_decode_deferred",
    "prefill_stalled_steps",
    "cumulative_prefill_requests_stalled",
    "budget_saturation_fraction",
    "mean_num_decoding",
    "mean_num_prefilling",
    "fraction_prefill_tokens_while_decodes_active",
    "status",
]


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
    for psize in grid["prefill_size_class"]:
        for occ in grid["decode_occupancy"]:
            for slo in grid["slo_regime"]:
                for load in grid["offered_load"]:
                    for seed in grid["seeds"]:
                        s = case_prefill_decode_interference(
                            prefill_size_class=str(psize),
                            decode_occupancy=str(occ),
                            slo_regime=str(slo),
                            offered_load=str(load),
                            seed=int(seed),
                            n_decode=fixed.get("n_decode"),
                            n_prefill=fixed.get("n_prefill"),
                            max_active_sequences=int(
                                fixed.get("max_active_sequences", 512)
                            ),
                            step_token_budget=int(
                                fixed.get("step_token_budget", 512)
                            ),
                            allow_synthetic_tokens=allow_synthetic_tokens,
                            datasets_root=datasets_root,
                        )
                        assert_policy_visible_fields_clean(s)
                        scenarios.append(s)
    return scenarios


def _contention_metrics(sim: Simulator) -> Dict[str, Any]:
    summary = sim.contention_diagnostics_summary()
    # Per-step diagnostics for occupancy + prefill-while-decode-attribution.
    prefill_while_decode = 0
    total_prefill = 0
    decode_counts = []
    prefill_counts = []
    n_steps = 0
    for g in sim._gpus:  # noqa: SLF001
        for d in g.step_contention_diagnostics:
            decode_counts.append(d.num_decoding)
            prefill_counts.append(d.num_prefilling)
            n_steps += 1
            total_prefill += d.prefill_tokens_served
            if d.num_decoding > 0:
                prefill_while_decode += d.prefill_tokens_served
    return {
        "decode_stalled_steps": summary["decode_stalled_steps"],
        "cumulative_decode_tokens_deferred": summary["cumulative_decode_tokens_deferred"],
        "steps_with_prefill_while_decode_deferred": summary["steps_with_prefill_while_decode_deferred"],
        "prefill_stalled_steps": summary["prefill_stalled_steps"],
        "cumulative_prefill_requests_stalled": summary["cumulative_prefill_requests_stalled"],
        "budget_saturation_fraction": summary["budget_saturation_fraction"],
        "mean_num_decoding": float(np.mean(decode_counts)) if decode_counts else 0.0,
        "mean_num_prefilling": float(np.mean(prefill_counts)) if prefill_counts else 0.0,
        "fraction_prefill_tokens_while_decodes_active": (
            prefill_while_decode / total_prefill if total_prefill > 0 else 0.0
        ),
    }


def _run_one_task(args: Tuple[str, str, Any, Dict[str, Any], Dict[str, int]]) -> dict:
    scenario_id, policy_name, scenario, variant_kwargs, chunk_budgets = args
    try:
        from llmserveopt.policies.prefill_control_variants import make_prefill_decode_variants

        variants = make_prefill_decode_variants(**chunk_budgets)
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

        ttfts = np.array([c.ttft for c in completed], dtype=float)
        valid_ttft = ttfts[~np.isnan(ttfts)]
        tpots = np.array([c.tpot for c in completed], dtype=float)
        valid_tpot = tpots[~np.isnan(tpots)]

        ttft_slo_s = float(scenario.params["ttft_slo_s"])
        tbt_slo_s = float(scenario.params["tbt_slo_s"])
        ttft_attain = float(np.mean(valid_ttft <= ttft_slo_s)) if len(valid_ttft) else 0.0
        tbt_attain = float(np.mean(valid_tpot <= tbt_slo_s)) if len(valid_tpot) else 0.0

        pct = lambda arr, q: (  # noqa: E731
            float(np.percentile(arr, q)) if len(arr) else 0.0
        )
        mean_of = lambda arr: float(np.mean(arr)) if len(arr) else 0.0  # noqa: E731

        row = {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "arrival_normalized_weighted_goodput": float(
                metrics.arrival_normalized_weighted_goodput
            ),
            "unweighted_slo_success_rate": float(unweighted),
            "completion_fraction": float(metrics.completion_fraction),
            "mean_ttft": mean_of(valid_ttft),
            "p95_ttft": pct(valid_ttft, 95),
            "p99_ttft": pct(valid_ttft, 99),
            "mean_tpot": mean_of(valid_tpot),
            "p95_tpot": pct(valid_tpot, 95),
            "mean_queuing_delay": float(metrics.mean_queuing_delay),
            "mean_prefill_delay_s": float(metrics.mean_prefill_delay),
            "ttft_attainment": ttft_attain,
            "tbt_attainment": tbt_attain,
            "request_throughput": float(metrics.request_throughput),
            "token_throughput": float(metrics.token_throughput),
        }
        row.update(_contention_metrics(sim))
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
                "Family B v1 production requires BurstGPT; none resolved. "
                "Set LLM_SERVEOPT_BURSTGPT_CSV, pass --datasets-root, or stage "
                "burstgpt_v2/raw shards."
            )
        _log(args.run_dir, f"BurstGPT resolved: {path}")

    allow_synthetic = bool(args.allow_synthetic_tokens) and not require_burstgpt
    _log(
        args.run_dir,
        f"Starting Family B v1 (dry_run={args.dry_run}, "
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
                **{k: s.params.get(k) for k in (
                    "prefill_size_class",
                    "decode_occupancy",
                    "slo_regime",
                    "offered_load",
                    "n_total_jobs",
                    "n_prefill",
                    "n_decode",
                    "step_token_budget",
                    "max_active_sequences",
                    "prefill_prompt_median",
                    "decode_output_median",
                    "decode_start_s",
                    "slack_prefill_s",
                    "decode_margin_s",
                    "ttft_slo_s",
                    "tbt_slo_s",
                    "arrival_shape",
                )},
                "seed": s.seed,
                "stress_control_relationship": s.stress_control_relationship,
                "token_sources": json.dumps(s.params.get("token_sources")),
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
        "chunk_large": int(chunk_cfg.get("chunk_large", DEFAULT_CHUNK_LARGE)),
        "decode_priority_chunk": int(
            chunk_cfg.get("decode_priority_chunk", DEFAULT_DECODE_PRIORITY_CHUNK)
        ),
    }
    variants = make_prefill_decode_variants(**chunk_budgets)
    policy_names = list(cfg["policies"])
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
            if done % 25 == 0:
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
        s.params.get("token_sources", {}).get("prefill_prompt")
        for s in scenarios
    }
    summary = {
        "experiment_name": cfg["pilot_metadata"]["experiment_name"],
        "generator_version": "prefill_decode_v1",
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
    }
    with open(args.run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _log(args.run_dir, "Completed Family B v1 pilot execution.")


if __name__ == "__main__":
    main()