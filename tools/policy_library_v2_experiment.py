#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path("/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution")
PRIMARY = "arrival_normalized_weighted_goodput"
METRIC_PREFIX = "metric_"
WSP = "weighted_shortest_processing"
SCORPIO = "scorpio_style_slo_guard"
N_CONFIGS = 2048

sys.path.insert(0, str(REPO / "src"))

from llmserveopt.core.types import GPUConfig, Request  # noqa: E402
from llmserveopt.policies.registry import (  # noqa: E402
    BASELINE_NAMES,
    POLICY_LIBRARY_V2_NAMES,
    POLICY_LIBRARY_V2_NEW_NAMES,
    make_policy_library_v2,
)
from llmserveopt.selector.dataset_v2.calibrated_targeted_pilot import _execution_service_model  # noqa: E402
from llmserveopt.selector.dataset_v2.features import extract_selector_v2_features  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402


RANGES = {
    "base_arrival_rate": (0.8, 32.0),
    "burst_multiplier": (1.0, 8.0),
    "burst_duration_fraction": (0.05, 0.45),
    "interarrival_cv": (0.35, 2.8),
    "prompt_mean": (128.0, 4096.0),
    "prompt_cv": (0.25, 1.8),
    "long_context_fraction": (0.0, 0.35),
    "output_mean": (24.0, 768.0),
    "output_cv": (0.25, 2.2),
    "decode_heavy_fraction": (0.0, 0.45),
    "prediction_noise": (0.0, 0.75),
    "slo_tightness": (0.45, 3.6),
    "urgent_fraction": (0.0, 0.5),
    "priority_skew": (0.0, 1.0),
    "gpu_sequence_capacity": (4.0, 16.0),
    "kv_token_budget": (3500.0, 24000.0),
    "step_token_budget": (256.0, 1536.0),
    "n_requests": (64.0, 144.0),
}

IMPLEMENTED = list(POLICY_LIBRARY_V2_NEW_NAMES)
DEFERRED = [
    "cache_prefix_reuse_aware",
    "cache_loading_aware",
    "disaggregated_prefill_decode_routing",
    "request_splitting_micro_request_scheduling",
    "heterogeneous_gpu_affinity_routing",
]
MODEST_EXTENSION = [
    "gate_and_route_phase_control",
    "decode_deadline_guard",
    "prefill_budget_controller",
    "phase_load_balance",
]

COVERAGE_DIMENSIONS = [
    "shortest_work_bias",
    "deadline_awareness",
    "laxity_slo_awareness",
    "explicit_admission_control",
    "overload_stability",
    "phase_awareness",
    "prefill_decode_interference",
    "chunked_prefill",
    "kv_cache_pressure",
    "resource_packing",
    "fairness_aging",
    "multi_tenant_fairness",
    "placement_awareness",
    "preemption_support",
    "cache_reuse_awareness",
    "disaggregated_routing",
    "heterogeneous_hardware_routing",
]


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default))


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNKNOWN: {exc}"


def status(root: Path, stage: str, state: str, payload: dict[str, Any] | None = None) -> None:
    row = {
        "stage": stage,
        "status": state,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }
    if payload:
        row.update(payload)
    write_json(root / "manifests" / f"{stage}_status.json", row)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_behavior_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tags: dict[str, set[str]] = {
        "fifo": set(),
        "edf": {"deadline_awareness"},
        "shortest_output_first": {"shortest_work_bias"},
        "shortest_prompt_first": {"shortest_work_bias", "kv_cache_pressure"},
        "greedy_token_fill": {"resource_packing"},
        "least_loaded": {"placement_awareness"},
        "multi_bin_batching": {"resource_packing", "kv_cache_pressure"},
        "random_feasible": set(),
        "first_fit": {"placement_awareness"},
        "best_fit": {"resource_packing", "placement_awareness", "kv_cache_pressure"},
        "orca_style": {"resource_packing"},
        "vllm_style_token_budget": {"resource_packing", "kv_cache_pressure"},
        "sarathi_style": {"phase_awareness", "prefill_decode_interference"},
        "splitfuse_style": {"phase_awareness", "prefill_decode_interference"},
        "slo_slack_score": {"deadline_awareness", "laxity_slo_awareness"},
        "weighted_shortest_processing": {"shortest_work_bias"},
        "least_laxity_first": {"deadline_awareness", "laxity_slo_awareness"},
        "estimated_service_time_first": {"shortest_work_bias"},
        "admission_control": {"deadline_awareness", "laxity_slo_awareness", "explicit_admission_control"},
        "scorpio_style_slo_guard": {"deadline_awareness", "laxity_slo_awareness", "explicit_admission_control", "overload_stability", "kv_cache_pressure"},
        "sola_style_state_aware": {"shortest_work_bias", "deadline_awareness", "laxity_slo_awareness", "overload_stability", "kv_cache_pressure", "placement_awareness"},
        "slai_style_phase_aware": {"phase_awareness", "prefill_decode_interference"},
        "flow_control_stability": {"explicit_admission_control", "overload_stability"},
        "kv_constrained_online": {"explicit_admission_control", "kv_cache_pressure"},
        "adaptive_chunked_prefill": {"explicit_admission_control", "phase_awareness", "prefill_decode_interference", "chunked_prefill", "kv_cache_pressure"},
        "aging_priority": {"fairness_aging"},
        "weighted_fair_share": {"fairness_aging", "multi_tenant_fairness"},
    }
    for name in POLICY_LIBRARY_V2_NAMES:
        row = {
            "policy_name": name,
            "library": "v2_new" if name in POLICY_LIBRARY_V2_NEW_NAMES else "v1_historical",
        }
        present = tags.get(name, set())
        for dim in COVERAGE_DIMENSIONS:
            row[dim] = int(dim in present)
        rows.append(row)
    return rows


def audit_policy_library(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for d in ["design", "shards", "combined", "models", "diagnostics", "reports", "manifests", "logs", "sbatch"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    coverage = policy_behavior_matrix()
    write_csv(root / "diagnostics" / "behavioral_coverage_matrix.csv", coverage)
    audit = {
        "POLICY_LIBRARY_V1_COUNT": len(BASELINE_NAMES),
        "POLICY_LIBRARY_V2_COUNT": len(POLICY_LIBRARY_V2_NAMES),
        "implemented_now": IMPLEMENTED,
        "implement_after_modest_extension": MODEST_EXTENSION,
        "deferred_policy_families": DEFERRED,
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "HEAD"]),
        "worktree": git(["status", "--short", "--branch"]),
        "coverage_matrix": str(root / "diagnostics" / "behavioral_coverage_matrix.csv"),
        "methodology": "V2 extends historical registry through POLICY_LIBRARY_V2_NAMES without changing BASELINE_NAMES.",
    }
    write_json(root / "manifests" / "policy_library_audit.json", audit)
    status(root, "policy_library_audit", "PASS", audit)


def latin_hypercube(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for name, (lo, hi) in RANGES.items():
        bins = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(bins)
        vals = lo + bins * (hi - lo)
        if name in {"gpu_sequence_capacity", "n_requests", "step_token_budget"}:
            vals = np.round(vals).astype(int)
        data[name] = vals
    df = pd.DataFrame(data)
    df["config_id"] = [f"plv2_{i:06d}" for i in range(n)]
    df["seed"] = seed + np.arange(n) * 23
    df["split_group"] = np.where(np.arange(n) % 10 < 6, "train", np.where(np.arange(n) % 10 < 8, "validation", "id_test"))
    high_shift = (df["base_arrival_rate"] > df["base_arrival_rate"].quantile(0.75)) & (df["long_context_fraction"] > df["long_context_fraction"].quantile(0.65))
    df.loc[high_shift, "split_group"] = "synthetic_ood"
    df["parameter_provenance"] = "Policy Library v2 Latin hypercube within BurstGPT/Azure-plausible workload ranges"
    return df


def frontier_design(root: Path) -> None:
    df = latin_hypercube(N_CONFIGS, 525252)
    df.to_csv(root / "design" / "policy_library_v2_design.csv", index=False)
    manifest = {
        "configurations": len(df),
        "ranges": RANGES,
        "v1_policies": list(BASELINE_NAMES),
        "v2_policies": list(POLICY_LIBRARY_V2_NAMES),
        "v2_new_policies": list(POLICY_LIBRARY_V2_NEW_NAMES),
        "split_counts": df["split_group"].value_counts().to_dict(),
    }
    write_json(root / "design" / "policy_library_v2_design_manifest.json", manifest)
    status(root, "frontier_design", "PASS", manifest)


def lognormal_int(rng: np.random.Generator, mean: float, cv: float, size: int, low: int, high: int) -> np.ndarray:
    sigma2 = math.log(max(cv * cv + 1.0, 1.0001))
    sigma = math.sqrt(sigma2)
    mu = math.log(max(mean, 1.0)) - sigma2 / 2.0
    vals = rng.lognormal(mu, sigma, size=size)
    return np.clip(np.round(vals), low, high).astype(int)


def make_requests(params: dict[str, Any]) -> tuple[list[Request], GPUConfig, dict[str, float]]:
    rng = np.random.default_rng(int(params["seed"]))
    n = int(params["n_requests"])
    rate = float(params["base_arrival_rate"])
    cv = float(params["interarrival_cv"])
    shape = 1.0 / max(cv * cv, 1e-6)
    scale = (1.0 / max(rate, 1e-6)) / shape
    inter = rng.gamma(shape, scale, size=n)
    burst_start = int(rng.uniform(0.15, 0.65) * n)
    burst_len = max(2, int(float(params["burst_duration_fraction"]) * n))
    inter[burst_start:burst_start + burst_len] /= float(params["burst_multiplier"])
    arrivals = np.cumsum(inter)
    arrivals -= arrivals.min()

    prompt = lognormal_int(rng, float(params["prompt_mean"]), float(params["prompt_cv"]), n, 8, 32768)
    out_pred = lognormal_int(rng, float(params["output_mean"]), float(params["output_cv"]), n, 4, 4096)
    long_mask = rng.random(n) < float(params["long_context_fraction"])
    prompt[long_mask] = np.clip(prompt[long_mask] * rng.integers(3, 8, size=long_mask.sum()), 8, 32768)
    decode_mask = rng.random(n) < float(params["decode_heavy_fraction"])
    out_pred[decode_mask] = np.clip(out_pred[decode_mask] * rng.integers(2, 5, size=decode_mask.sum()), 4, 4096)
    noise = rng.lognormal(mean=-0.5 * float(params["prediction_noise"]) ** 2, sigma=float(params["prediction_noise"]), size=n)
    out_actual = np.clip(np.round(out_pred * noise), 1, 8192).astype(int)

    urgent = rng.random(n) < float(params["urgent_fraction"])
    medium = (~urgent) & (rng.random(n) < 0.45)
    class_ids = np.where(urgent, "tight", np.where(medium, "medium", "loose"))
    priority = np.where(urgent, 3.0 + 2.0 * float(params["priority_skew"]), np.where(medium, 2.0, 1.0))
    seq_cap = int(params["gpu_sequence_capacity"])
    kv = int(params["kv_token_budget"])
    step_budget = int(params["step_token_budget"])
    service_est = prompt / max(step_budget, 1) + out_pred / max(seq_cap * 24.0, 1.0)
    class_mult = np.where(urgent, 0.75, np.where(medium, 1.6, 3.2))
    deadlines = arrivals + service_est * class_mult * float(params["slo_tightness"]) + 0.05
    requests = [
        Request(
            request_id=i,
            arrival_time=float(arrivals[i]),
            prompt_tokens=int(prompt[i]),
            predicted_output_tokens=int(out_pred[i]),
            actual_output_tokens=int(out_actual[i]),
            slo_deadline=float(deadlines[i]),
            priority=float(priority[i]),
            class_id=str(class_ids[i]),
        )
        for i in range(n)
    ]
    gpu = GPUConfig(0, max_active_sequences=seq_cap, max_batch_tokens=1_000_000, max_kv_tokens=kv)
    duration = max(float(arrivals.max() - arrivals.min()), 1e-6)
    derived = {
        "duration": duration,
        "offered_request_rate": n / duration,
        "prefill_decode_token_ratio": float(np.sum(prompt) / max(np.sum(out_pred), 1)),
        "prompt_p95": float(np.percentile(prompt, 95)),
        "output_p95": float(np.percentile(out_pred, 95)),
        "approx_load_ratio": float((n / duration) / max(seq_cap * step_budget / max(float(np.mean(out_pred)), 1.0), 1e-9)),
    }
    return requests, gpu, derived


def dynamic_features(requests: list[Request], gpu: GPUConfig, params: dict[str, Any], derived: dict[str, float]) -> dict[str, float]:
    decision_time = float(np.percentile([r.arrival_time for r in requests], 30))
    obs = [r for r in requests if r.arrival_time <= decision_time]
    prompt = np.array([r.prompt_tokens for r in obs], dtype=float)
    out = np.array([r.predicted_output_tokens for r in obs], dtype=float)
    slack = np.array([r.slo_deadline - decision_time for r in obs], dtype=float)
    feats: dict[str, float] = {}
    for q in [10, 25, 50, 75, 90, 95]:
        feats[f"plv2_prompt_p{q}"] = float(np.percentile(prompt, q)) if len(prompt) else 0.0
        feats[f"plv2_output_p{q}"] = float(np.percentile(out, q)) if len(out) else 0.0
        feats[f"plv2_slack_p{q}"] = float(np.percentile(slack, q)) if len(slack) else 0.0
    for horizon in [1.0, 5.0, 20.0, 60.0]:
        recent = [r for r in obs if decision_time - r.arrival_time <= horizon]
        feats[f"plv2_arrival_rate_{int(horizon)}s"] = len(recent) / horizon
        feats[f"plv2_work_rate_{int(horizon)}s"] = sum(r.prompt_tokens + r.predicted_output_tokens for r in recent) / horizon
    feats["plv2_queue_length_now"] = float(len(obs))
    feats["plv2_queue_growth_short_minus_medium"] = feats["plv2_arrival_rate_5s"] - feats["plv2_arrival_rate_20s"]
    feats["plv2_fraction_negative_slack"] = float(np.mean(slack < 0.0)) if len(slack) else 0.0
    feats["plv2_fraction_slack_lt_1s"] = float(np.mean(slack < 1.0)) if len(slack) else 0.0
    feats["plv2_estimated_backlog_work"] = float(np.sum(prompt + out)) if len(obs) else 0.0
    feats["plv2_estimated_work_per_slot"] = feats["plv2_estimated_backlog_work"] / max(gpu.max_active_sequences, 1)
    feats["plv2_kv_pressure"] = feats["plv2_estimated_backlog_work"] / max(gpu.max_kv_tokens, 1)
    for k in RANGES:
        feats[f"param_{k}"] = float(params[k])
    feats.update({f"derived_{k}": float(v) for k, v in derived.items()})
    return feats


def run_policy(policy_name: str, requests: list[Request], gpu: GPUConfig, budget: int, seed: int, tag: str) -> dict[str, Any]:
    policy = make_policy_library_v2(policy_name, seed=seed)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=_execution_service_model(budget, budget), drain_steps=20_000))
    sim.load_trace(requests)
    metrics = sim.run(policy, workload_tag=tag, seed=seed)
    row = asdict(metrics)
    return {f"{METRIC_PREFIX}{k}": v for k, v in row.items()}


def run_config(params: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config_id = str(params["config_id"])
    requests, gpu, derived = make_requests(params)
    budget = int(params["step_token_budget"])
    features = extract_selector_v2_features(
        window_requests=requests,
        window_start_time=float(np.percentile([r.arrival_time for r in requests], 30)),
        gpu_configs=[gpu],
        topology_class="monolithic",
        step_token_budget=budget,
        active_sequence_count=0,
        aggregate_kv_utilization=0.0,
        active_batch_size=0.0,
    )
    features.update(dynamic_features(requests, gpu, params, derived))
    feature_row = {"config_id": config_id, "split_group": params.get("split_group", ""), **{f"feat_{k}": v for k, v in features.items()}}
    vectors: list[dict[str, Any]] = []
    rewards: dict[str, float] = {}
    for policy_name in POLICY_LIBRARY_V2_NAMES:
        metric_row = run_policy(policy_name, requests, gpu, budget, int(params["seed"]), config_id)
        reward = float(metric_row.get(f"{METRIC_PREFIX}{PRIMARY}", 0.0) or 0.0)
        rewards[policy_name] = reward
        vectors.append({"config_id": config_id, "split_group": params.get("split_group", ""), "policy_name": policy_name, "library": "v2_new" if policy_name in POLICY_LIBRARY_V2_NEW_NAMES else "v1", **metric_row})
    sorted_v1 = sorted(BASELINE_NAMES, key=lambda p: rewards[p], reverse=True)
    sorted_v2 = sorted(POLICY_LIBRARY_V2_NAMES, key=lambda p: rewards[p], reverse=True)
    summary = {
        "config_id": config_id,
        "split_group": params.get("split_group", ""),
        "seed": int(params["seed"]),
        "best_v1_policy": sorted_v1[0],
        "best_v2_policy": sorted_v2[0],
        "v1_oracle_anwg": rewards[sorted_v1[0]],
        "v2_oracle_anwg": rewards[sorted_v2[0]],
        "oracle_envelope_gain": rewards[sorted_v2[0]] - rewards[sorted_v1[0]],
        "new_policy_best": int(sorted_v2[0] in POLICY_LIBRARY_V2_NEW_NAMES),
        "new_policy_meaningful_unique_win": int(sorted_v2[0] in POLICY_LIBRARY_V2_NEW_NAMES and rewards[sorted_v2[0]] - rewards[sorted_v1[0]] >= 0.002),
        "best_second_margin_v2": rewards[sorted_v2[0]] - rewards[sorted_v2[1]],
        "wsp_reward": rewards.get(WSP, 0.0),
        "scorpio_reward": rewards.get(SCORPIO, 0.0),
        "delta_scorpio_wsp": rewards.get(SCORPIO, 0.0) - rewards.get(WSP, 0.0),
        **{k: params[k] for k in RANGES},
        **derived,
    }
    return summary, vectors, feature_row


def expanded_frontier_array(root: Path) -> None:
    design = pd.read_csv(root / "design" / "policy_library_v2_design.csv")
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    subset = design.iloc[[i for i in range(len(design)) if i % count == task]]
    summaries, vectors, features = [], [], []
    started = time.time()
    errors: list[dict[str, str]] = []
    for _, row in subset.iterrows():
        try:
            summary, vec, feat = run_config(row.to_dict())
        except Exception as exc:
            errors.append({"config_id": str(row.get("config_id", "")), "error": repr(exc)})
            continue
        summaries.append(summary)
        vectors.extend(vec)
        features.append(feat)
    out = root / "shards" / f"task_{task:04d}"
    write_csv(out / "workload_summaries.csv", summaries)
    write_csv(out / "policy_vectors.csv", vectors)
    write_csv(out / "features.csv", features)
    if errors:
        write_csv(out / "errors.csv", errors)
    manifest = {
        "task": task,
        "attempted": len(subset),
        "retained": len(summaries),
        "errors": len(errors),
        "runtime_s": time.time() - started,
        "new_policy_best_count": int(sum(r["new_policy_best"] for r in summaries)),
        "oracle_gain_sum": float(sum(r["oracle_envelope_gain"] for r in summaries)),
    }
    write_json(out / "manifest.json", manifest)
    status(root, "expanded_frontier_array", "PASS" if not errors else "PARTIAL", manifest)


def combine_frontier_v2(root: Path) -> None:
    frames = {"summary": [], "vectors": [], "features": []}
    errors = []
    for task_dir in sorted((root / "shards").glob("task_*")):
        for key, filename in [("summary", "workload_summaries.csv"), ("vectors", "policy_vectors.csv"), ("features", "features.csv")]:
            path = task_dir / filename
            if path.exists() and path.stat().st_size > 0:
                frames[key].append(pd.read_csv(path, dtype={"config_id": str}))
        err = task_dir / "errors.csv"
        if err.exists() and err.stat().st_size > 0:
            errors.append(pd.read_csv(err, dtype={"config_id": str}))
    out = root / "combined"
    result = {"error_rows": int(sum(len(e) for e in errors))}
    for key, parts in frames.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        df.to_csv(out / f"{key}.csv", index=False)
        result[f"{key}_rows"] = int(len(df))
    if frames["summary"]:
        summary = pd.concat(frames["summary"], ignore_index=True)
        result.update({
            "windows": int(len(summary)),
            "meaningful_windows": int((summary["best_second_margin_v2"] >= 0.002).sum()),
            "v1_oracle_anwg": float(summary["v1_oracle_anwg"].mean()),
            "v2_oracle_anwg": float(summary["v2_oracle_anwg"].mean()),
            "oracle_envelope_gain": float(summary["oracle_envelope_gain"].mean()),
            "new_policy_best_window_fraction": float(summary["new_policy_best"].mean()),
            "new_policy_meaningful_unique_win_count": int(summary["new_policy_meaningful_unique_win"].sum()),
            "best_new_policy_counts": summary.loc[summary["best_v2_policy"].isin(POLICY_LIBRARY_V2_NEW_NAMES), "best_v2_policy"].value_counts().to_dict(),
            "split_counts": summary["split_group"].value_counts().to_dict(),
        })
    write_json(out / "combine_manifest.json", result)
    status(root, "combine_frontier_v2", "PASS", result)


def policy_complementarity(root: Path) -> None:
    summary = pd.read_csv(root / "combined" / "summary.csv", dtype={"config_id": str})
    vectors = pd.read_csv(root / "combined" / "vectors.csv", dtype={"config_id": str})
    pivot = vectors.pivot_table(index="config_id", columns="policy_name", values=f"{METRIC_PREFIX}{PRIMARY}", aggfunc="first")
    rows = []
    for policy in POLICY_LIBRARY_V2_NEW_NAMES:
        rewards = pivot[policy]
        best = summary["best_v2_policy"].eq(policy)
        v1_gap = rewards.values - summary["v1_oracle_anwg"].values
        rows.append({
            "policy_name": policy,
            "win_count": int(best.sum()),
            "unique_win_count": int((best & (summary["oracle_envelope_gain"] > 1e-9)).sum()),
            "meaningful_unique_win_count": int((best & (summary["oracle_envelope_gain"] >= 0.002)).sum()),
            "mean_gain_vs_v1_oracle_when_best": float(summary.loc[best, "oracle_envelope_gain"].mean()) if best.any() else 0.0,
            "mean_regret_when_not_best": float((summary.loc[~best, "v2_oracle_anwg"] - rewards.loc[summary.loc[~best, "config_id"]].values).mean()) if (~best).any() else 0.0,
            "disagreement_rate_with_wsp": float((rewards.round(9) != pivot[WSP].round(9)).mean()),
            "disagreement_rate_with_scorpio": float((rewards.round(9) != pivot[SCORPIO].round(9)).mean()),
            "correlation_with_wsp": float(rewards.corr(pivot[WSP])),
            "correlation_with_scorpio": float(rewards.corr(pivot[SCORPIO])),
        })
    write_csv(root / "diagnostics" / "new_policy_complementarity.csv", rows)
    pivot.corr().to_csv(root / "diagnostics" / "regret_profile_similarity.csv")
    status(root, "policy_complementarity", "PASS", {"new_policy_rows": len(rows)})


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_")]


def selector_v1_v2_compare(root: Path) -> None:
    from sklearn.ensemble import RandomForestRegressor

    summary = pd.read_csv(root / "combined" / "summary.csv", dtype={"config_id": str})
    features = pd.read_csv(root / "combined" / "features.csv", dtype={"config_id": str})
    vectors = pd.read_csv(root / "combined" / "vectors.csv", dtype={"config_id": str})
    df = summary.merge(features, on=["config_id", "split_group"], how="inner")
    cols = _feature_cols(df)
    vec = vectors.pivot_table(index="config_id", columns="policy_name", values=f"{METRIC_PREFIX}{PRIMARY}", aggfunc="first")
    rows = []
    for label, policies in [("selector_v1", list(BASELINE_NAMES)), ("selector_v2", list(POLICY_LIBRARY_V2_NAMES))]:
        train_ids = set(df.loc[df["split_group"].isin(["train", "validation"]), "config_id"])
        X_train = df.loc[df["config_id"].isin(train_ids), cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        models = {}
        for policy in policies:
            y = vec.loc[X_train.index.map(lambda i: df.loc[i, "config_id"]), policy].values
            model = RandomForestRegressor(n_estimators=160, max_depth=12, random_state=42, n_jobs=-1)
            model.fit(X_train.values, y)
            models[policy] = model
        for split in ["id_test", "synthetic_ood"]:
            part = df[df["split_group"].eq(split)]
            if part.empty:
                continue
            X = part[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            preds = np.vstack([models[p].predict(X.values) for p in policies]).T
            choices = [policies[int(i)] for i in preds.argmax(axis=1)]
            rewards = []
            best_fixed = {p: float(vec.loc[part["config_id"], p].mean()) for p in policies}
            for config_id, choice in zip(part["config_id"], choices):
                rewards.append(float(vec.loc[config_id, choice]))
            rows.append({
                "selector_name": label,
                "split": split,
                "n": len(part),
                "selector_anwg": float(np.mean(rewards)),
                "best_fixed_policy": max(best_fixed, key=best_fixed.get),
                "best_fixed_anwg": float(max(best_fixed.values())),
                "oracle_anwg": float(vec.loc[part["config_id"], policies].max(axis=1).mean()),
                "oracle_gap_closed_vs_wsp": float((np.mean(rewards) - best_fixed.get(WSP, 0.0)) / max(vec.loc[part["config_id"], policies].max(axis=1).mean() - best_fixed.get(WSP, 0.0), 1e-9)),
            })
    write_csv(root / "models" / "selector_v1_v2_compare.csv", rows)
    status(root, "selector_v1_v2_compare", "PASS", {"rows": len(rows)})


def composition_readiness(root: Path) -> None:
    primitives = []
    mapping = {
        "sola_style_state_aware": ["priority", "load", "kv", "laxity"],
        "slai_style_phase_aware": ["phase", "priority"],
        "flow_control_stability": ["admission", "load"],
        "kv_constrained_online": ["admission", "kv", "laxity"],
        "adaptive_chunked_prefill": ["admission", "phase", "kv"],
        "aging_priority": ["fairness", "priority"],
        "weighted_fair_share": ["fairness", "priority"],
    }
    for policy, parts in mapping.items():
        primitives.append({"policy_name": policy, "reusable_primitives": ",".join(parts)})
    write_csv(root / "diagnostics" / "composition_readiness.csv", primitives)
    status(root, "composition_readiness", "PASS", {"policies": len(primitives)})


def final_policy_library_report(root: Path) -> None:
    combine = read_json(root / "combined" / "combine_manifest.json", {})
    selector = pd.read_csv(root / "models" / "selector_v1_v2_compare.csv") if (root / "models" / "selector_v1_v2_compare.csv").exists() else pd.DataFrame()
    comp = pd.read_csv(root / "diagnostics" / "new_policy_complementarity.csv") if (root / "diagnostics" / "new_policy_complementarity.csv").exists() else pd.DataFrame()
    best_new_policy = None
    if combine.get("best_new_policy_counts"):
        best_new_policy = max(combine["best_new_policy_counts"], key=combine["best_new_policy_counts"].get)
    v1_id = selector.query("selector_name == 'selector_v1' and split == 'id_test'")["selector_anwg"].iloc[0] if not selector.empty and len(selector.query("selector_name == 'selector_v1' and split == 'id_test'")) else None
    v2_id = selector.query("selector_name == 'selector_v2' and split == 'id_test'")["selector_anwg"].iloc[0] if not selector.empty and len(selector.query("selector_name == 'selector_v2' and split == 'id_test'")) else None
    v1_ood = selector.query("selector_name == 'selector_v1' and split == 'synthetic_ood'")["selector_anwg"].iloc[0] if not selector.empty and len(selector.query("selector_name == 'selector_v1' and split == 'synthetic_ood'")) else None
    v2_ood = selector.query("selector_name == 'selector_v2' and split == 'synthetic_ood'")["selector_anwg"].iloc[0] if not selector.empty and len(selector.query("selector_name == 'selector_v2' and split == 'synthetic_ood'")) else None
    gain = float(combine.get("oracle_envelope_gain", 0.0) or 0.0)
    unique = int(combine.get("new_policy_meaningful_unique_win_count", 0) or 0)
    new_frac = float(combine.get("new_policy_best_window_fraction", 0.0) or 0.0)
    if gain < 0.001 and unique < 10:
        status_value = "SUFFICIENT"
    elif gain >= 0.002 and new_frac >= 0.03:
        status_value = "EXPANSION_HELPED"
    elif unique > 0:
        status_value = "STILL_INCOMPLETE"
    else:
        status_value = "SIMULATOR_LIMITED"
    fields = {
        "POLICY_LIBRARY_STATUS": status_value,
        "POLICY_LIBRARY_V1_COUNT": len(BASELINE_NAMES),
        "POLICY_LIBRARY_V2_COUNT": len(POLICY_LIBRARY_V2_NAMES),
        "V1_ORACLE_ANWG": combine.get("v1_oracle_anwg"),
        "V2_ORACLE_ANWG": combine.get("v2_oracle_anwg"),
        "ORACLE_ENVELOPE_GAIN": combine.get("oracle_envelope_gain"),
        "NEW_POLICY_BEST_WINDOW_FRACTION": combine.get("new_policy_best_window_fraction"),
        "NEW_POLICY_MEANINGFUL_UNIQUE_WIN_COUNT": unique,
        "BEST_NEW_POLICY": best_new_policy or "UNKNOWN",
        "SELECTOR_V1_ID_ANWG": v1_id,
        "SELECTOR_V2_ID_ANWG": v2_id,
        "SELECTOR_V1_OOD_ANWG": v1_ood,
        "SELECTOR_V2_OOD_ANWG": v2_ood,
        "MAIN_NEW_BEHAVIORAL_DIMENSION": "overload stability, phase awareness, KV reserve control, aging/fair-share behavior",
        "MAIN_REMAINING_POLICY_GAP": "cache reuse, true disaggregated routing, request splitting, and heterogeneous hardware affinity remain simulator/action-space limited",
        "NEXT_RECOMMENDED_ACTION": "Use Policy Frontier Cartography plus this V2 oracle-gain report to decide whether to add simulator primitives or move to policy composition.",
    }
    write_json(root / "reports" / "final_policy_library_fields.json", fields)
    lines = ["# Policy Library v2 Expanded-Library Evaluation", ""]
    for k, v in fields.items():
        lines.append(f"{k} = {v}")
    lines += [
        "",
        "## Artifacts",
        "- diagnostics/behavioral_coverage_matrix.csv",
        "- combined/summary.csv",
        "- combined/vectors.csv",
        "- diagnostics/new_policy_complementarity.csv",
        "- diagnostics/regret_profile_similarity.csv",
        "- models/selector_v1_v2_compare.csv",
        "- diagnostics/composition_readiness.csv",
    ]
    if not comp.empty:
        top = comp.sort_values("meaningful_unique_win_count", ascending=False).head(10)
        lines += ["", "## New Policy Complementarity", "```", top.to_string(index=False), "```"]
    (root / "reports" / "FINAL_POLICY_LIBRARY_REPORT.md").write_text("\n".join(lines) + "\n")
    status(root, "final_policy_library_report", "PASS", fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    stages = {
        "policy_library_audit": audit_policy_library,
        "frontier_design": frontier_design,
        "expanded_frontier_array": expanded_frontier_array,
        "combine_frontier_v2": combine_frontier_v2,
        "policy_complementarity": policy_complementarity,
        "selector_v1_v2_compare": selector_v1_v2_compare,
        "composition_readiness": composition_readiness,
        "final_policy_library_report": final_policy_library_report,
    }
    try:
        stages[args.stage](args.run_root)
    except Exception as exc:
        status(args.run_root, args.stage, "FAIL", {"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
