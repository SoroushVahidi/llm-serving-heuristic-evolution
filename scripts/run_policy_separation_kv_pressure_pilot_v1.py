#!/usr/bin/env python3
"""Family C KV-pressure reserve pairwise-separation pilot runner (v1 & v2).

See docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md and _V2.md. This
is a pairwise-separation pilot, NOT a composition falsification: exactly two
policies (`kv_constrained_online`, `least_laxity_first`) are evaluated on
every scenario -- no selector is fit, no child policy is run.

--template-version {v1,v2} selects the scenario generator (default v1, so
existing v1 configs/invocations are unaffected). v2 configs may set
`held_out_seeds` (a list) to tag rows for the v2 held-out replication check
(design doc v2 SS7) -- ignored, and must be omitted, for v1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation.templates_kv_pressure import (  # noqa: E402
    CLASS_BULK,
    CLASS_URGENT,
    assert_policy_visible_fields_clean_kv_v1,
    case_kv_pressure_reserve_contention,
)
from llmserveopt.policy_separation.templates_kv_pressure_v2 import (  # noqa: E402
    assert_policy_visible_fields_clean_kv_v2,
    case_kv_pressure_reserve_contention_v2,
)
from llmserveopt.policy_separation.templates_prefill_decode import (  # noqa: E402
    resolve_burstgpt_path,
)
from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy  # noqa: E402
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402


class InstrumentedKVConstrainedOnlinePolicy(KVConstrainedOnlinePolicy):
    """Diagnostic wrapper: counts admission-deferral-due-to-reserve events
    (H2/G3) without changing KVConstrainedOnlinePolicy's own logic. A
    "deferral" is any candidate the gate would have admitted on hard KV
    capacity alone but blocked because it is non-urgent and would push
    utilization past `target_kv_utilization`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_reserve_deferrals = 0

    def _admit_filter(self, req, gpu, admitted, now):
        post_util = (gpu.current_kv_tokens + req.prompt_tokens) / max(gpu.max_kv_tokens, 1)
        from llmserveopt.policies.policy_library_v2_helpers import laxity_seconds
        urgent = laxity_seconds(req, now, self.step_size, self.alpha, self.beta) <= self.urgent_laxity_seconds
        would_defer = post_util > self.target_kv_utilization and not urgent
        if would_defer:
            self.n_reserve_deferrals += 1
        return super()._admit_filter(req, gpu, admitted, now)


POLICIES = {
    "kv_constrained_online": InstrumentedKVConstrainedOnlinePolicy,
    "least_laxity_first": LeastLaxityFirstPolicy,
}

RESULT_FIELDNAMES = [
    "scenario_id", "policy_name", "bulk_pressure", "urgent_arrival_phase",
    "urgent_tightness", "seed", "held_out", "status",
    "arrival_normalized_weighted_goodput", "unweighted_slo_success_rate",
    "completion_fraction",
    "bulk_n", "bulk_slo_success_rate",
    "urgent_n", "urgent_slo_success_rate",
    "peak_kv_utilization", "steps_over_reserve_threshold", "n_steps",
    "n_reserve_deferrals",
]


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def _class_success(completed, class_id: str) -> Tuple[int, float]:
    rows = [c for c in completed if c.request.class_id == class_id]
    if not rows:
        return 0, 0.0
    ok = sum(1 for c in rows if not c.slo_violated)
    return len(rows), ok / len(rows)


def _run_one(args: Tuple[str, str, Any, int, bool]) -> dict:
    scenario_id, policy_name, scenario, max_kv_tokens, held_out = args
    try:
        policy_cls = POLICIES[policy_name]
        policy = policy_cls()
        policy.name = policy_name

        sim = Simulator(SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**scenario.service_model_kwargs),
        ))
        sim.load_trace(list(scenario.requests))
        metrics = sim.run(policy, workload_tag=scenario_id, seed=scenario.seed)
        completed = list(sim._completed)  # noqa: SLF001

        n_req = len(scenario.requests)
        n_violated = sum(1 for c in completed if c.slo_violated)
        unweighted = (len(completed) - n_violated) / max(1, n_req)

        bulk_n, bulk_ok = _class_success(completed, CLASS_BULK)
        urgent_n, urgent_ok = _class_success(completed, CLASS_URGENT)

        kv_hist = sim._gpus[0].step_kv_used  # noqa: SLF001
        peak_util = (max(kv_hist) / max_kv_tokens) if kv_hist else 0.0
        n_over = sum(1 for k in kv_hist if k / max_kv_tokens > 0.82)

        n_deferrals = getattr(policy, "n_reserve_deferrals", 0)

        row = {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "bulk_pressure": scenario.params["bulk_pressure"],
            "urgent_arrival_phase": scenario.params["urgent_arrival_phase"],
            "urgent_tightness": scenario.params["urgent_tightness"],
            "seed": scenario.seed,
            "held_out": held_out,
            "status": "success",
            "arrival_normalized_weighted_goodput": float(
                metrics.arrival_normalized_weighted_goodput
            ),
            "unweighted_slo_success_rate": float(unweighted),
            "completion_fraction": float(metrics.completion_fraction),
            "bulk_n": bulk_n,
            "bulk_slo_success_rate": bulk_ok,
            "urgent_n": urgent_n,
            "urgent_slo_success_rate": urgent_ok,
            "peak_kv_utilization": float(peak_util),
            "steps_over_reserve_threshold": n_over,
            "n_steps": len(kv_hist),
            "n_reserve_deferrals": n_deferrals,
        }
        return row
    except Exception as e:  # noqa: BLE001
        import traceback
        return {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "held_out": held_out,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def build_scenarios(
    cfg: dict, *, template_version: str, allow_synthetic_tokens: bool, datasets_root,
) -> List[Any]:
    grid = cfg["sweep_grid"]
    max_kv_tokens = int(cfg.get("max_kv_tokens", 8000))
    max_active_sequences = int(cfg.get("max_active_sequences", 64))
    max_batch_tokens = int(cfg.get("max_batch_tokens", 64))
    if template_version == "v1":
        build_fn = case_kv_pressure_reserve_contention
        leakage_guard = assert_policy_visible_fields_clean_kv_v1
        if cfg.get("held_out_seeds"):
            raise ValueError("held_out_seeds is a v2-only config field; --template-version v1 given")
    elif template_version == "v2":
        build_fn = case_kv_pressure_reserve_contention_v2
        leakage_guard = assert_policy_visible_fields_clean_kv_v2
    else:
        raise ValueError(f"--template-version must be v1 or v2, got {template_version!r}")

    scenarios = []
    for bulk_pressure in grid["bulk_pressure"]:
        for phase in grid["urgent_arrival_phase"]:
            for tightness in grid["urgent_tightness"]:
                for seed in grid["seeds"]:
                    s = build_fn(
                        bulk_pressure=str(bulk_pressure),
                        urgent_arrival_phase=str(phase),
                        urgent_tightness=str(tightness),
                        seed=int(seed),
                        max_kv_tokens=max_kv_tokens,
                        max_active_sequences=max_active_sequences,
                        max_batch_tokens=max_batch_tokens,
                        allow_synthetic_tokens=allow_synthetic_tokens,
                        datasets_root=datasets_root,
                    )
                    leakage_guard(s)
                    scenarios.append(s)
    return scenarios


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    """Additive provenance helper. Never raises; returns None on any failure
    (missing file, permissions, etc.) rather than aborting the run."""
    if path is None:
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", None))
    except Exception:
        return None


def _collect_provenance(
    *,
    config_path: Path,
    dataset_path: Optional[Path],
    seeds: List[int],
    template_version: str,
    policy_names: List[str],
) -> Dict[str, Any]:
    """Additive, null-safe run-provenance metadata for FUTURE runs. Never
    raises; unavailable fields become None/"unknown". Purely observational
    over already-resolved inputs and already-written outputs -- does not
    affect scenario generation, RNG order, policy execution, or metrics.
    `result_csv_sha256` is added by the caller after the CSV write."""
    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "command": " ".join(sys.argv),
        "config_path": str(config_path.resolve()) if config_path else None,
        "config_sha256": _sha256_file(config_path),
        "dataset_path": str(dataset_path.resolve()) if dataset_path else None,
        "dataset_sha256": _sha256_file(dataset_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": _pkg_version("numpy"),
        "pandas_version": _pkg_version("pandas"),
        "scipy_version": _pkg_version("scipy"),
        "sklearn_version": _pkg_version("sklearn"),
        "seeds": sorted(seeds),
        "template_version": template_version,
        "policy_names": list(policy_names),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--template-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--allow-synthetic-tokens", action="store_true")
    parser.add_argument("--datasets-root", type=Path, default=None)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _log(args.run_dir, f"Starting Family C {args.template_version} KV-pressure reserve pairwise-separation pilot.")
    scenarios = build_scenarios(
        cfg, template_version=args.template_version,
        allow_synthetic_tokens=args.allow_synthetic_tokens,
        datasets_root=args.datasets_root,
    )
    max_kv_tokens = int(cfg.get("max_kv_tokens", 8000))
    held_out_seeds = set(int(s) for s in cfg.get("held_out_seeds", []))
    _log(args.run_dir, f"Generated {len(scenarios)} scenarios. Held-out seeds: {sorted(held_out_seeds) or 'none'}.")

    tasks = []
    for s in scenarios:
        held_out = s.seed in held_out_seeds
        for policy_name in POLICIES:
            tasks.append((s.scenario_id, policy_name, s, max_kv_tokens, held_out))

    all_results: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            all_results.append(fut.result())

    results_csv_path = args.run_dir / "per_policy_results.csv"
    with open(results_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    success = [r for r in all_results if r.get("status") == "success"]
    failed = [r for r in all_results if r.get("status") == "failed"]

    # Additive forward-looking provenance (docs/audits/kv_v2_reproducibility_forensic_20260817.md
    # SS9): resolved AFTER scenario generation and the CSV write so
    # result_csv_sha256 reflects the actual written bytes. Read-only,
    # null-safe, does not affect scenario generation/RNG/policy behavior.
    try:
        dataset_path = resolve_burstgpt_path(datasets_root=args.datasets_root)
    except Exception:
        dataset_path = None
    provenance = _collect_provenance(
        config_path=args.config,
        dataset_path=dataset_path,
        seeds=[int(s) for s in cfg.get("sweep_grid", {}).get("seeds", [])],
        template_version=args.template_version,
        policy_names=list(POLICIES.keys()),
    )
    provenance["result_csv_sha256"] = _sha256_file(results_csv_path)

    summary = {
        "template_version": args.template_version,
        "n_scenarios": len(scenarios),
        "n_tasks": len(all_results),
        "n_completed": len(success),
        "n_failed": len(failed),
        "primary_metric": "arrival_normalized_weighted_goodput",
        "max_kv_tokens": max_kv_tokens,
        "policies": list(POLICIES.keys()),
        "held_out_seeds": sorted(held_out_seeds),
        "provenance": provenance,
    }
    with open(args.run_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log(args.run_dir, f"Completed: {len(success)} success, {len(failed)} failed")


if __name__ == "__main__":
    main()
