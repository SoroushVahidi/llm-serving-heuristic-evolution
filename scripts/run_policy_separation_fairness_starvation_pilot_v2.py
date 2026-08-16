#!/usr/bin/env python3
"""Runner for Policy Separation Family A v2 (fairness vs size) pilot.

Primary metric: canonical RunMetrics.arrival_normalized_weighted_goodput.
Production mode requires BurstGPT (``--require-burstgpt``, default True).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policies.registry import make_policy_library_v2
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (
    BurstGPTUnavailableError,
    case_fairness_vs_size_v2,
    resolve_burstgpt_path,
)
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

RESULT_FIELDNAMES = [
    "scenario_id",
    "policy_name",
    "arrival_normalized_weighted_goodput",
    "unweighted_slo_success_rate",
    "completion_fraction",
    "favored_violations",
    "favored_total",
    "other_violations",
    "other_total",
    "jains_fairness_index",
    "mean_ttft",
    "status",
]


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def jains_index(g1: float, g2: float) -> float:
    num = (g1 + g2) ** 2
    denom = 2 * (g1 ** 2 + g2 ** 2)
    return num / denom if denom > 0 else 0.0


def build_scenarios_from_config(
    cfg: dict, *, allow_synthetic_tokens: bool
) -> List[Any]:
    grid = cfg["sweep_grid"]
    fixed = cfg.get("fixed", {})
    scenarios = []
    for fav in grid["favored_tenant_size"]:
        for skew in grid["tenant_weight_skew"]:
            for util in grid["target_utilization"]:
                for noise in grid["prediction_noise_sigma"]:
                    for seed in grid["seeds"]:
                        scenarios.append(
                            case_fairness_vs_size_v2(
                                target_utilization=float(util),
                                tenant_weight_skew=float(skew),
                                favored_tenant_size=str(fav),
                                prediction_noise_sigma=float(noise),
                                seed=int(seed),
                                n_total_jobs=int(fixed.get("n_total_jobs", 120)),
                                max_active_sequences=int(
                                    fixed.get("max_active_sequences", 2)
                                ),
                                favored_slo_slack_s=float(
                                    fixed.get("favored_slo_slack_s", 2.0)
                                ),
                                other_slo_slack_s=float(
                                    fixed.get("other_slo_slack_s", 12.0)
                                ),
                                allow_synthetic_tokens=allow_synthetic_tokens,
                            )
                        )
    return scenarios


def _run_one_task(args: Tuple[str, str, Any]) -> dict:
    scenario_id, policy_name, scenario = args
    try:
        policy = make_policy_library_v2(policy_name)
        sim = Simulator(SimulatorConfig(gpu_configs=list(scenario.gpu_configs)))
        sim.load_trace(list(scenario.requests))
        metrics = sim.run(policy, workload_tag=scenario_id)
        completed = sim._completed  # noqa: SLF001

        favored = [c for c in completed if c.request.class_id == "tenant_favored"]
        other = [c for c in completed if c.request.class_id == "tenant_other"]
        fav_v = sum(1 for c in favored if c.completion_time > c.request.slo_deadline)
        oth_v = sum(1 for c in other if c.completion_time > c.request.slo_deadline)
        fav_n = len(favored)
        oth_n = len(other)
        fav_gp = (fav_n - fav_v) / max(1, fav_n)
        oth_gp = (oth_n - oth_v) / max(1, oth_n)
        total_v = fav_v + oth_v
        unweighted = (len(completed) - total_v) / max(1, len(scenario.requests))
        ttfts = [c.ttft for c in completed if c.first_token_time >= 0]
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0

        return {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "arrival_normalized_weighted_goodput": float(
                metrics.arrival_normalized_weighted_goodput
            ),
            "unweighted_slo_success_rate": float(unweighted),
            "completion_fraction": float(metrics.completion_fraction),
            "favored_violations": fav_v,
            "favored_total": fav_n,
            "other_violations": oth_v,
            "other_total": oth_n,
            "jains_fairness_index": jains_index(fav_gp, oth_gp),
            "mean_ttft": mean_ttft,
            "status": "success",
        }
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
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(args.config)
    require_burstgpt = bool(args.require_burstgpt) and not args.allow_synthetic_tokens
    if cfg.get("pilot_metadata", {}).get("require_burstgpt", True) and require_burstgpt:
        path = resolve_burstgpt_path()
        if path is None:
            raise SystemExit(
                "Family A v2 production requires BurstGPT; none resolved. "
                "Set LLM_SERVEOPT_BURSTGPT_CSV or stage burstgpt_v2/raw shards."
            )
        _log(args.run_dir, f"BurstGPT resolved: {path}")

    allow_synthetic = bool(args.allow_synthetic_tokens) and not require_burstgpt
    _log(
        args.run_dir,
        f"Starting Family A v2 (dry_run={args.dry_run}, "
        f"allow_synthetic_tokens={allow_synthetic})",
    )

    try:
        scenarios = build_scenarios_from_config(
            cfg, allow_synthetic_tokens=allow_synthetic
        )
    except BurstGPTUnavailableError as e:
        raise SystemExit(str(e)) from e

    if args.dry_run:
        scenarios = scenarios[:4]

    # Write scenario manifests (no raw requests)
    with open(args.run_dir / "scenarios.jsonl", "w") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_manifest_dict()) + "\n")

    features_rows = []
    for s in scenarios:
        features_rows.append(
            {
                "scenario_id": s.scenario_id,
                **{k: s.params.get(k) for k in (
                    "target_utilization",
                    "tenant_weight_skew",
                    "favored_tenant_size",
                    "other_tenant_size",
                    "prediction_noise_sigma",
                    "token_length_source",
                    "burstgpt_path",
                    "size_priority_alignment",
                    "max_active_sequences",
                )},
                "seed": s.seed,
                "stress_control_relationship": s.stress_control_relationship,
            }
        )
    with open(args.run_dir / "scenario_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(features_rows[0].keys()))
        w.writeheader()
        w.writerows(features_rows)

    _log(args.run_dir, f"Generated {len(scenarios)} scenarios.")
    policies = list(cfg["policies"])
    tasks = [(scen.scenario_id, p, scen) for scen in scenarios for p in policies]
    _log(args.run_dir, f"Total tasks to run: {len(tasks)}")

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

    token_sources = sorted({str(s.params.get("token_length_source")) for s in scenarios})
    summary = {
        "experiment_name": cfg["pilot_metadata"]["experiment_name"],
        "generator_version": "fairness_starvation_v2",
        "n_scenarios": len(scenarios),
        "n_tasks": len(tasks),
        "n_completed": len(success),
        "n_failed": len(failed),
        "elapsed_seconds": elapsed,
        "primary_metric": "arrival_normalized_weighted_goodput",
        "token_length_sources_observed": token_sources,
        "require_burstgpt": require_burstgpt,
        "allow_synthetic_tokens": allow_synthetic,
        "policies": policies,
    }
    with open(args.run_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Provenance / git
    import subprocess

    manifest = {
        "run_dir": str(args.run_dir),
        "config": str(args.config),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
        "burstgpt_path": str(resolve_burstgpt_path()) if resolve_burstgpt_path() else None,
    }
    with open(args.run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    _log(args.run_dir, "Completed Family A v2 pilot execution.")


if __name__ == "__main__":
    main()
