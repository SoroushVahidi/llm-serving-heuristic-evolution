#!/usr/bin/env python
"""Joint multi-mechanism workload generalization experiment.

This is a CPU-only, fixed-policy evaluation. It does not train/select/search
for any scheduler. The generator specification and scenario manifest are
written and hashed before any policy utility is evaluated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmserveopt.core.metrics import metrics_to_dict
from llmserveopt.core.types import GPUConfig
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policy_separation.builders import req
from llmserveopt.policy_separation.schema import PolicySeparationScenario
from llmserveopt.policy_separation.unified_utility_matrix import (
    CANONICAL_ANCHOR_IDS,
    _build_policy,
)
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "joint_multimechanism_generalization_v1"
FIG = OUT / "figures"

SEED = 20260824
N_SCENARIOS = 240
ROBUSTNESS_BOOTSTRAP_SEED = 20260825
PRACTICAL_EPSILON = 0.01

POLICIES = [
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
]


class RecordingPolicy(BasePolicy):
    """Transparent wrapper that records admitted request IDs per step."""

    def __init__(self, inner: BasePolicy) -> None:
        self.inner = inner
        self.name = inner.name
        self.records: list[dict[str, Any]] = []

    def select_action(self, state):
        action = self.inner.select_action(state)
        admitted = sorted(rid for ids in action.admit.values() for rid in ids)
        if admitted:
            self.records.append(
                {
                    "step": int(state.step),
                    "time": float(state.time),
                    "waiting": len(state.waiting_queue),
                    "running": int(sum(len(g.active_request_ids) for g in state.gpu_states)),
                    "admitted_ids": admitted,
                }
            )
        return action


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(json_safe(obj), sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(stable_json_dumps(obj))


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()

    info: dict[str, Any] = {}
    for key, args in {
        "branch": ["git", "branch", "--show-current"],
        "head": ["git", "rev-parse", "HEAD"],
        "upstream": ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
    }.items():
        try:
            info[key] = run(args)
        except Exception:
            info[key] = None
    try:
        info["ahead_behind"] = run(["git", "rev-list", "--left-right", "--count", "@{u}...HEAD"])
    except Exception:
        info["ahead_behind"] = None
    try:
        info["status_short"] = run(["git", "status", "--short", "--branch"])
    except Exception:
        info["status_short"] = None
    return info


def generator_spec() -> dict[str, Any]:
    return {
        "experiment": "joint_multimechanism_generalization_v1",
        "version": "joint_multimechanism_generator_v1.0.0",
        "seed": SEED,
        "n_scenarios": N_SCENARIOS,
        "policy_outcome_independence": (
            "All distributions and interpretation thresholds are fixed before "
            "policy evaluation; no selector/model/search is run."
        ),
        "scenario_schema": "single joint synthetic TRAIN-compatible workload schema",
        "sampling": {
            "offered_load": "uniform[0.75,1.35]",
            "burstiness": "uniform[0,1], controls cluster strength and arrival compression",
            "long_fraction": "uniform[0.15,0.85]",
            "prompt_scale": "log-uniform median tokens [128,2048]",
            "prompt_heterogeneity_sigma": "uniform[0.25,0.95]",
            "output_scale": "log-uniform median tokens [32,256]",
            "output_heterogeneity_sigma": "uniform[0.25,0.9]",
            "tenant_weight_skew": "log-uniform [1,5]",
            "class_share_skew": "uniform[0.25,0.75]",
            "slo_tightness": "uniform[0.65,2.4] multiplier on estimated isolated service",
            "prediction_noise_sigma": "uniform[0,0.45]",
            "kv_pressure_target": "uniform[0.35,1.35], converted to max_kv_tokens with feasibility floor",
            "late_pressure": "uniform[0,1], controls fraction and density of late wave",
            "late_phase": "uniform[0.10,0.72] of early span",
            "max_active_sequences": "integer sampled from {4,8,16,32}",
            "step_token_budget": "integer sampled from {256,512,1024}",
        },
        "service_model": {
            "step_size": 0.001,
            "enable_prefill_modeling": True,
            "prefill_cost_per_token": 1.0,
            "enable_decode_prefill_contention": True,
            "decode_first": False,
            "allow_chunked_prefill": True,
            "base_max_prefill_chunk_tokens": 512,
        },
        "policies": POLICIES,
        "primary_metric": "arrival_normalized_weighted_goodput",
        "practical_epsilon": PRACTICAL_EPSILON,
        "hypotheses": {
            "H1_COMPLEMENTARITY": "meaningful six-policy complementarity persists",
            "H2_ORACLE_HEADROOM": "six-policy oracle retains positive practical headroom over best fixed",
            "H3_MULTI_MECHANISM": "some oracle gain occurs with >=2 elevated mechanism pressures",
            "H4_NON_CONCENTRATION": "oracle gain is not dominated by a tiny scenario subset",
        },
        "runtime_feasibility_adjustment": {
            "original_target": "approximately 300 scenarios",
            "final_count": N_SCENARIOS,
            "reason": (
                "Initial CPU execution attempt showed trace/storage and per-cell "
                "runtime cost too high before any utility result was written. "
                "The count was reduced within the preregistered 200-400 range "
                "using runtime feasibility only."
            ),
        },
        "verdict_thresholds": {
            "strong_min_headroom_anwg": 0.01,
            "strong_min_unique_winner_fraction": 0.10,
            "strong_min_nontrivial_spread_fraction": 0.25,
            "strong_min_gain_share_in_multi_pressure": 0.30,
            "strong_max_top10_gain_share": 0.50,
            "partial_min_headroom_anwg": 0.003,
        },
        "decision_disagreement_metric": (
            "Closed-loop action-set overlap proxy from transparent policy wrappers. "
            "Because trajectories can diverge, this is diagnostic only, not a selector target."
        ),
    }


def sample_params(rng: np.random.Generator, idx: int) -> dict[str, Any]:
    logu = lambda lo, hi: float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    max_active = int(rng.choice([4, 8, 16, 32], p=[0.25, 0.35, 0.25, 0.15]))
    return {
        "joint_id": f"joint_mm_{idx:04d}",
        "seed": int(SEED + idx),
        "offered_load": float(rng.uniform(0.75, 1.35)),
        "burstiness": float(rng.uniform(0.0, 1.0)),
        "long_fraction": float(rng.uniform(0.15, 0.85)),
        "prompt_scale": logu(160, 4096),
        "prompt_heterogeneity_sigma": float(rng.uniform(0.25, 0.95)),
        "output_scale": logu(48, 512),
        "output_heterogeneity_sigma": float(rng.uniform(0.25, 0.9)),
        "tenant_weight_skew": logu(1.0, 5.0),
        "class_share_skew": float(rng.uniform(0.25, 0.75)),
        "slo_tightness": float(rng.uniform(0.65, 2.4)),
        "prediction_noise_sigma": float(rng.uniform(0.0, 0.45)),
        "kv_pressure_target": float(rng.uniform(0.35, 1.35)),
        "late_pressure": float(rng.uniform(0.0, 1.0)),
        "late_phase": float(rng.uniform(0.10, 0.72)),
        "max_active_sequences": max_active,
        "step_token_budget": int(rng.choice([256, 512, 1024], p=[0.25, 0.50, 0.25])),
        "n_requests": int(rng.integers(36, 73)),
    }


def build_scenario(p: dict[str, Any]) -> PolicySeparationScenario:
    rng = np.random.default_rng(int(p["seed"]))
    n = int(p["n_requests"])
    late_frac = 0.10 + 0.35 * float(p["late_pressure"])
    n_late = max(6, min(n - 12, int(round(n * late_frac))))
    n_early = n - n_late

    is_long = rng.random(n) < float(p["long_fraction"])
    prompt_median = float(p["prompt_scale"])
    out_median = float(p["output_scale"])
    prompt_mult = np.where(is_long, 2.0, 0.45)
    output_mult = np.where(is_long, 1.65, 0.55)
    prompts = rng.lognormal(
        mean=np.log(prompt_median * prompt_mult),
        sigma=float(p["prompt_heterogeneity_sigma"]),
    )
    actual_outputs = rng.lognormal(
        mean=np.log(out_median * output_mult),
        sigma=float(p["output_heterogeneity_sigma"]),
    )
    prompts = np.clip(np.rint(prompts), 32, 4096).astype(int)
    actual_outputs = np.clip(np.rint(actual_outputs), 8, 512).astype(int)
    pred_noise = rng.lognormal(mean=0.0, sigma=float(p["prediction_noise_sigma"]), size=n)
    predicted_outputs = np.clip(np.rint(actual_outputs * pred_noise), 1, 2048).astype(int)

    step_size = 0.001
    step_budget = int(p["step_token_budget"])
    prefill_est = prompts / max(step_budget, 1) * step_size
    decode_est = predicted_outputs * step_size
    total_work = float(np.sum(prefill_est + decode_est))
    horizon = max(0.05, total_work / max(float(p["offered_load"]) * max(int(p["max_active_sequences"]), 1), 1e-6))

    early_span = max(0.02, horizon * (0.70 - 0.35 * float(p["burstiness"])))
    early_arrivals = np.sort(rng.uniform(0, early_span, size=n_early))
    if float(p["burstiness"]) > 0.15:
        n_clusters = 2 + int(float(p["burstiness"]) * 4)
        centers = rng.uniform(0, early_span, size=n_clusters)
        mask = rng.random(n_early) < float(p["burstiness"])
        early_arrivals[mask] = rng.choice(centers, size=int(mask.sum())) + rng.normal(
            0.0, 0.004 * (1.1 - float(p["burstiness"])), size=int(mask.sum())
        )
        early_arrivals = np.sort(np.clip(early_arrivals, 0, early_span))

    late_start = float(p["late_phase"]) * early_span
    late_density = max(0.0008, 0.006 * (1.0 - 0.80 * float(p["late_pressure"])))
    late_arrivals = late_start + np.cumsum(rng.exponential(late_density, size=n_late))

    arrivals = np.concatenate([early_arrivals, late_arrivals])
    order = np.argsort(arrivals, kind="mergesort")
    prompts = prompts[order]
    actual_outputs = actual_outputs[order]
    predicted_outputs = predicted_outputs[order]
    arrivals = arrivals[order]
    is_late = np.array([False] * n_early + [True] * n_late)[order]
    is_long = is_long[order]

    high_class = rng.random(n) < float(p["class_share_skew"])
    base_tight = float(p["slo_tightness"])
    requests = []
    for rid in range(n):
        priority = float(p["tenant_weight_skew"]) if high_class[rid] else 1.0
        class_id = "tenant_high" if high_class[rid] else "tenant_low"
        if is_late[rid]:
            class_id = f"{class_id}_late"
        # Keep deadline pressure online-observable but not impossible by
        # construction. Late and high-priority requests are slightly tighter.
        class_factor = 0.82 if high_class[rid] else 1.12
        late_factor = 0.78 if is_late[rid] else 1.0
        slack = max(
            0.004,
            base_tight * class_factor * late_factor * float(prefill_est[order][rid] + decode_est[order][rid]),
        )
        requests.append(
            req(
                request_id=rid,
                arrival_time=float(arrivals[rid]),
                prompt_tokens=int(prompts[rid]),
                predicted_output_tokens=int(predicted_outputs[rid]),
                actual_output_tokens=int(actual_outputs[rid]),
                slo_deadline=float(arrivals[rid]) + slack,
                priority=priority,
                class_id=class_id,
            )
        )

    prompt_sum = int(np.sum(prompts))
    max_prompt = int(np.max(prompts))
    active = int(p["max_active_sequences"])
    kv_by_pressure = int(max(prompt_sum * 0.20, np.quantile(prompts, 0.90) * active * 0.75) / max(float(p["kv_pressure_target"]), 0.05))
    max_kv_tokens = int(max(max_prompt + 512, min(600_000, kv_by_pressure)))
    gpu = GPUConfig(
        gpu_id=0,
        max_active_sequences=active,
        max_batch_tokens=active,
        max_kv_tokens=max_kv_tokens,
    )
    service_model_kwargs = {
        "step_size": step_size,
        "enable_prefill_modeling": True,
        "prefill_cost_per_token": 1.0,
        "step_token_budget": step_budget,
        "max_prefill_chunk_tokens": 512,
        "enable_decode_prefill_contention": True,
        "decode_first": False,
        "allow_chunked_prefill": True,
    }
    params = dict(p)
    params.update(
        {
            "n_early": n_early,
            "n_late": n_late,
            "prompt_median_realized": float(np.median(prompts)),
            "output_median_realized": float(np.median(actual_outputs)),
            "long_fraction_realized": float(np.mean(is_long)),
            "late_fraction_realized": float(np.mean(is_late)),
            "high_priority_fraction_realized": float(np.mean(high_class)),
            "arrival_span_s": float(np.max(arrivals) - np.min(arrivals)),
            "max_kv_tokens": max_kv_tokens,
        }
    )
    return PolicySeparationScenario(
        scenario_id=str(p["joint_id"]),
        family="joint_multimechanism_v1",
        template_name="joint_continuous_mixed_mechanism_v1",
        generator_version="joint_multimechanism_generator_v1.0.0",
        seed=int(p["seed"]),
        params=params,
        requests=tuple(sorted(requests, key=lambda r: (r.arrival_time, r.request_id))),
        gpu_configs=(gpu,),
        service_model_kwargs=service_model_kwargs,
        target_policy_family="six_policy_portfolio_generalization",
        target_mechanism="joint_continuous_multi_mechanism",
        expected_qualitative_hypothesis=(
            "Policy winners should vary across jointly sampled fairness, service, "
            "prefill/decode, urgency, burst, and KV pressure, leaving positive "
            "six-policy oracle headroom over the best fixed policy."
        ),
        stress_control_relationship=None,
    )


def pressure_indicators(s: PolicySeparationScenario) -> dict[str, Any]:
    p = s.params
    prompts = np.array([r.prompt_tokens for r in s.requests], dtype=float)
    outputs = np.array([r.actual_output_tokens for r in s.requests], dtype=float)
    arrivals = np.array([r.arrival_time for r in s.requests], dtype=float)
    priorities = np.array([r.priority for r in s.requests], dtype=float)
    late = np.array(["late" in r.class_id for r in s.requests], dtype=bool)
    deadlines = np.array([r.slo_deadline - r.arrival_time for r in s.requests], dtype=float)
    service_est = (prompts / max(int(p["step_token_budget"]), 1) + outputs) * 0.001
    kv_cap = int(p["max_kv_tokens"])
    span = max(float(np.max(arrivals) - np.min(arrivals)), 1e-9)
    pressures = {
        "fairness_pressure": min(1.0, (float(np.max(priorities)) - 1.0) / 4.0 * (1.0 + abs(float(np.mean(priorities > 1.0)) - 0.5))),
        "service_heterogeneity": min(1.0, float(np.std(outputs) / max(np.mean(outputs), 1.0))),
        "prefill_decode_pressure": min(1.0, float(np.mean(late)) * 1.5 + float(np.quantile(prompts, 0.85) / 8192) * 0.7),
        "kv_pressure": min(1.0, float(np.quantile(prompts, 0.90) * int(p["max_active_sequences"]) / max(kv_cap, 1))),
        "urgency_pressure": min(1.0, float(np.mean(service_est / np.maximum(deadlines, 1e-9)))),
        "burst_pressure": min(1.0, float(p["burstiness"]) * 0.65 + (len(arrivals) / span > 300.0) * 0.35),
    }
    elevated = {f"high_{k}": bool(v >= 0.60) for k, v in pressures.items()}
    return {
        **{k: round(float(v), 6) for k, v in pressures.items()},
        **elevated,
        "n_elevated_mechanisms": int(sum(elevated.values())),
    }


def run_policy_on_scenario(s: PolicySeparationScenario, policy_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    policy, sm_override = _build_policy(policy_id)
    rec = RecordingPolicy(policy)
    merged_sm = dict(s.service_model_kwargs)
    merged_sm.update(sm_override)
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(s.gpu_configs),
            service_model=ServiceModel(**merged_sm),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(s.requests))
    metrics = sim.run(rec, workload_tag=s.scenario_id, seed=s.seed)
    row = metrics_to_dict(metrics)
    row.update(
        {
            "scenario_id": s.scenario_id,
            "policy_id": policy_id,
            "primary_utility_anwg": metrics.arrival_normalized_weighted_goodput,
            "status": "success",
        }
    )
    return row, rec.records, sim.contention_diagnostics_summary()


def coverage_summary(scenarios: list[PolicySeparationScenario]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for s in scenarios:
        p = s.params
        row = {
            "scenario_id": s.scenario_id,
            "seed": s.seed,
            "num_requests": len(s.requests),
            "offered_load": p["offered_load"],
            "burstiness": p["burstiness"],
            "long_fraction": p["long_fraction_realized"],
            "prompt_median": p["prompt_median_realized"],
            "output_median": p["output_median_realized"],
            "tenant_weight_skew": p["tenant_weight_skew"],
            "class_share_skew": p["class_share_skew"],
            "slo_tightness": p["slo_tightness"],
            "prediction_noise_sigma": p["prediction_noise_sigma"],
            "kv_pressure_target": p["kv_pressure_target"],
            "late_pressure": p["late_pressure"],
            "late_fraction": p["late_fraction_realized"],
            "late_phase": p["late_phase"],
            "max_active_sequences": p["max_active_sequences"],
            "step_token_budget": p["step_token_budget"],
            "max_kv_tokens": p["max_kv_tokens"],
        }
        row.update(pressure_indicators(s))
        rows.append(row)
    df = pd.DataFrame(rows)
    dims = [
        "offered_load",
        "burstiness",
        "long_fraction",
        "prompt_median",
        "output_median",
        "tenant_weight_skew",
        "slo_tightness",
        "kv_pressure",
        "late_pressure",
    ]
    summary: dict[str, Any] = {
        "n_scenarios": len(df),
        "dimension_summary": {},
        "mechanism_pressure_counts": df["n_elevated_mechanisms"].value_counts().sort_index().to_dict(),
        "multi_mechanism_scenarios_ge2": int((df["n_elevated_mechanisms"] >= 2).sum()),
        "multi_mechanism_fraction_ge2": float((df["n_elevated_mechanisms"] >= 2).mean()),
        "triple_plus_mechanism_scenarios": int((df["n_elevated_mechanisms"] >= 3).sum()),
        "pressure_means": {k: float(df[k].mean()) for k in [
            "fairness_pressure",
            "service_heterogeneity",
            "prefill_decode_pressure",
            "kv_pressure",
            "urgency_pressure",
            "burst_pressure",
        ]},
    }
    for d in dims:
        summary["dimension_summary"][d] = {
            "min": float(df[d].min()),
            "p10": float(df[d].quantile(0.10)),
            "median": float(df[d].median()),
            "p90": float(df[d].quantile(0.90)),
            "max": float(df[d].max()),
        }
    corr_cols = [
        "offered_load",
        "burstiness",
        "long_fraction",
        "prompt_median",
        "output_median",
        "tenant_weight_skew",
        "slo_tightness",
        "kv_pressure",
        "late_pressure",
    ]
    summary["pairwise_correlations"] = df[corr_cols].corr().round(4).to_dict()
    return df, summary


def summarize_utilities(wide: pd.DataFrame, cov: pd.DataFrame) -> dict[str, Any]:
    utility_cols = [f"anwg__{p}" for p in POLICIES]
    vals = wide[utility_cols]
    winner = vals.idxmax(axis=1).str.replace("anwg__", "", regex=False)
    wide["winner"] = winner
    wide["oracle"] = vals.max(axis=1)
    mean_by_policy = {p: float(wide[f"anwg__{p}"].mean()) for p in POLICIES}
    best_fixed_policy = max(mean_by_policy, key=mean_by_policy.get)
    wide["best_fixed_utility"] = wide[f"anwg__{best_fixed_policy}"]
    wide["oracle_gain_over_best_fixed"] = wide["oracle"] - wide["best_fixed_utility"]
    wide["policy_range"] = vals.max(axis=1) - vals.min(axis=1)
    eps = PRACTICAL_EPSILON
    unique_winners = {}
    for p in POLICIES:
        others = [f"anwg__{q}" for q in POLICIES if q != p]
        unique_winners[p] = int((wide[f"anwg__{p}"] >= wide[others].max(axis=1) + eps).sum())
    cov2 = cov.set_index("scenario_id")
    merged = wide.join(cov2, on="scenario_id", rsuffix="_cov")
    multi = merged["n_elevated_mechanisms"] >= 2
    total_gain = float(wide["oracle_gain_over_best_fixed"].clip(lower=0).sum())
    top_gain = wide["oracle_gain_over_best_fixed"].clip(lower=0).sort_values(ascending=False)
    top10_share = float(top_gain.head(max(1, math.ceil(0.10 * len(top_gain)))).sum() / total_gain) if total_gain > 0 else 0.0
    return {
        "mean_by_policy": mean_by_policy,
        "best_fixed_policy": best_fixed_policy,
        "best_fixed_mean_utility": float(mean_by_policy[best_fixed_policy]),
        "oracle_mean_utility": float(wide["oracle"].mean()),
        "oracle_headroom_anwg": float(wide["oracle"].mean() - mean_by_policy[best_fixed_policy]),
        "winner_counts": winner.value_counts().reindex(POLICIES, fill_value=0).astype(int).to_dict(),
        "winner_fractions": (winner.value_counts(normalize=True).reindex(POLICIES, fill_value=0.0)).to_dict(),
        "unique_winner_counts_epsilon": unique_winners,
        "unique_winner_fraction": float(sum(unique_winners.values()) / len(wide)),
        "nontrivial_policy_spread_fraction": float((wide["policy_range"] >= eps).mean()),
        "policy_range_summary": {
            "mean": float(wide["policy_range"].mean()),
            "median": float(wide["policy_range"].median()),
            "p90": float(wide["policy_range"].quantile(0.90)),
            "max": float(wide["policy_range"].max()),
        },
        "oracle_gain_summary": {
            "mean": float(wide["oracle_gain_over_best_fixed"].mean()),
            "median": float(wide["oracle_gain_over_best_fixed"].median()),
            "p75": float(wide["oracle_gain_over_best_fixed"].quantile(0.75)),
            "p90": float(wide["oracle_gain_over_best_fixed"].quantile(0.90)),
            "max": float(wide["oracle_gain_over_best_fixed"].max()),
            "positive_fraction": float((wide["oracle_gain_over_best_fixed"] > 0).mean()),
            "epsilon_positive_fraction": float((wide["oracle_gain_over_best_fixed"] >= eps).mean()),
        },
        "gain_share_multi_mechanism_ge2": float(merged.loc[multi, "oracle_gain_over_best_fixed"].clip(lower=0).sum() / total_gain) if total_gain > 0 else 0.0,
        "gain_share_triple_plus_ge3": float(merged.loc[merged["n_elevated_mechanisms"] >= 3, "oracle_gain_over_best_fixed"].clip(lower=0).sum() / total_gain) if total_gain > 0 else 0.0,
        "top10_percent_gain_share": top10_share,
        "top5_scenario_gain_share": float(top_gain.head(5).sum() / total_gain) if total_gain > 0 else 0.0,
        "win_by_elevated_mechanism_count": merged.groupby("n_elevated_mechanisms")["winner"].value_counts().unstack(fill_value=0).to_dict(),
    }


def summarize_action_disagreement(
    traces: dict[tuple[str, str], list[dict[str, Any]]],
    wide: pd.DataFrame,
) -> dict[str, Any]:
    pair_rows = []
    per_scenario = []
    for sid in wide["scenario_id"]:
        scenario_dis = []
        for i, p in enumerate(POLICIES):
            for q in POLICIES[i + 1 :]:
                a = {r["step"]: set(r["admitted_ids"]) for r in traces[(sid, p)] if r["admitted_ids"]}
                b = {r["step"]: set(r["admitted_ids"]) for r in traces[(sid, q)] if r["admitted_ids"]}
                steps = sorted(set(a) | set(b))
                if not steps:
                    overlap = 1.0
                else:
                    vals = []
                    for st in steps:
                        sa, sb = a.get(st, set()), b.get(st, set())
                        union = sa | sb
                        vals.append(1.0 if not union else len(sa & sb) / len(union))
                    overlap = float(np.mean(vals))
                dis = 1.0 - overlap
                scenario_dis.append(dis)
                pair_rows.append({"scenario_id": sid, "policy_a": p, "policy_b": q, "action_set_disagreement": dis})
        per_scenario.append({"scenario_id": sid, "mean_pairwise_action_set_disagreement": float(np.mean(scenario_dis))})
    pair_df = pd.DataFrame(pair_rows)
    scen_df = pd.DataFrame(per_scenario)
    merged = wide[["scenario_id", "policy_range"]].merge(scen_df, on="scenario_id")
    corr = float(merged["mean_pairwise_action_set_disagreement"].corr(merged["policy_range"])) if len(merged) > 2 else float("nan")
    return {
        "available": True,
        "metric": "closed_loop_action_set_disagreement_proxy",
        "caveat": "Policy trajectories can diverge; this is diagnostic, not a same-state selector disagreement estimate.",
        "pairwise_mean": pair_df.groupby(["policy_a", "policy_b"])["action_set_disagreement"].mean().round(6).to_dict(),
        "overall_mean_pairwise_disagreement": float(pair_df["action_set_disagreement"].mean()),
        "scenario_disagreement_policy_range_correlation": corr,
        "scenario_summary": {
            "median": float(scen_df["mean_pairwise_action_set_disagreement"].median()),
            "p90": float(scen_df["mean_pairwise_action_set_disagreement"].quantile(0.90)),
            "max": float(scen_df["mean_pairwise_action_set_disagreement"].max()),
        },
    }


def local_winner_structure(wide: pd.DataFrame, cov: pd.DataFrame) -> dict[str, Any]:
    cols = [
        "fairness_pressure",
        "service_heterogeneity",
        "prefill_decode_pressure",
        "kv_pressure",
        "urgency_pressure",
        "burst_pressure",
    ]
    df = wide[["scenario_id", "winner"]].merge(cov[["scenario_id"] + cols], on="scenario_id")
    x = df[cols].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-9)
    k = min(10, len(df) - 1)
    consistencies = []
    entropies = []
    for i in range(len(df)):
        d = np.linalg.norm(x - x[i], axis=1)
        nn = np.argsort(d)[1 : k + 1]
        wins = df.iloc[nn]["winner"].tolist()
        same = sum(1 for w in wins if w == df.iloc[i]["winner"]) / k
        counts = np.array(list(pd.Series(wins).value_counts(normalize=True)), dtype=float)
        ent = float(-(counts * np.log2(np.maximum(counts, 1e-12))).sum())
        consistencies.append(same)
        entropies.append(ent)
    out = {
        "feature_axes": cols,
        "k": k,
        "nearest_neighbor_winner_consistency_mean": float(np.mean(consistencies)),
        "nearest_neighbor_winner_consistency_median": float(np.median(consistencies)),
        "local_winner_entropy_mean_bits": float(np.mean(entropies)),
        "local_winner_entropy_median_bits": float(np.median(entropies)),
    }
    bin_df = df.copy()
    for c in cols:
        bin_df[f"{c}_bin"] = pd.cut(bin_df[c], bins=[-0.001, 0.333, 0.667, 1.001], labels=["low", "mid", "high"])
    out["winner_by_pressure_bin"] = {}
    for c in cols:
        out["winner_by_pressure_bin"][c] = bin_df.groupby(f"{c}_bin", observed=False)["winner"].value_counts().unstack(fill_value=0).to_dict()
    return out


def old_family_comparison() -> dict[str, Any]:
    path = ROOT / "experiments" / "unified_utility_matrix_v2" / "unified_utility_matrix_wide_v2.csv"
    if not path.exists():
        return {"available": False, "reason": str(path)}
    old = pd.read_csv(path)
    utility_cols = [f"anwg__{p}" for p in POLICIES]
    old = old.dropna(subset=utility_cols)
    old["winner"] = old[utility_cols].idxmax(axis=1).str.replace("anwg__", "", regex=False)
    rows = {}
    for fam, g in old.groupby("mechanism_family"):
        mean_by_policy = {p: float(g[f"anwg__{p}"].mean()) for p in POLICIES}
        best = max(mean_by_policy, key=mean_by_policy.get)
        oracle = g[utility_cols].max(axis=1)
        gain = oracle - g[f"anwg__{best}"]
        rows[fam] = {
            "n_scenarios": int(len(g)),
            "winner_counts": g["winner"].value_counts().reindex(POLICIES, fill_value=0).astype(int).to_dict(),
            "best_fixed_policy": best,
            "best_fixed_mean": mean_by_policy[best],
            "oracle_mean": float(oracle.mean()),
            "oracle_headroom": float(oracle.mean() - mean_by_policy[best]),
            "oracle_gain_p90": float(gain.quantile(0.90)),
        }
    return {"available": True, "source": str(path.relative_to(ROOT)), "families": rows}


def bootstrap_robustness(wide: pd.DataFrame, n_boot: int = 1000) -> dict[str, Any]:
    rng = np.random.default_rng(ROBUSTNESS_BOOTSTRAP_SEED)
    utility_cols = [f"anwg__{p}" for p in POLICIES]
    heads = []
    bests = []
    n = len(wide)
    vals = wide[utility_cols].to_numpy(dtype=float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = vals[idx]
        means = sample.mean(axis=0)
        best_fixed = float(means.max())
        oracle = float(sample.max(axis=1).mean())
        heads.append(oracle - best_fixed)
        bests.append(POLICIES[int(np.argmax(means))])
    return {
        "method": "scenario bootstrap from frozen joint population",
        "n_bootstrap": n_boot,
        "seed": ROBUSTNESS_BOOTSTRAP_SEED,
        "headroom_ci95": [float(np.quantile(heads, 0.025)), float(np.quantile(heads, 0.975))],
        "headroom_mean": float(np.mean(heads)),
        "best_fixed_policy_bootstrap_counts": pd.Series(bests).value_counts().reindex(POLICIES, fill_value=0).astype(int).to_dict(),
    }


def make_figures(cov: pd.DataFrame, wide: pd.DataFrame) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    winner_counts = wide["winner"].value_counts().reindex(POLICIES, fill_value=0)
    plt.figure(figsize=(8, 4))
    winner_counts.plot(kind="bar")
    plt.ylabel("scenarios")
    plt.title("Joint workload six-policy winner distribution")
    plt.tight_layout()
    p = FIG / "winner_distribution.png"
    plt.savefig(p, dpi=180)
    plt.close()
    made.append(str(p.relative_to(ROOT)))

    plt.figure(figsize=(6, 4))
    wide["oracle_gain_over_best_fixed"].plot(kind="hist", bins=30)
    plt.xlabel("oracle gain over best fixed ANWG")
    plt.title("Joint workload oracle-gain distribution")
    plt.tight_layout()
    p = FIG / "oracle_gain_histogram.png"
    plt.savefig(p, dpi=180)
    plt.close()
    made.append(str(p.relative_to(ROOT)))

    m = cov[[
        "fairness_pressure",
        "service_heterogeneity",
        "prefill_decode_pressure",
        "kv_pressure",
        "urgency_pressure",
        "burst_pressure",
    ]].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(m, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="correlation")
    plt.xticks(range(len(m.columns)), m.columns, rotation=45, ha="right")
    plt.yticks(range(len(m.columns)), m.columns)
    plt.title("Joint workload pressure-axis correlations")
    plt.tight_layout()
    p = FIG / "mechanism_pressure_correlation.png"
    plt.savefig(p, dpi=180)
    plt.close()
    made.append(str(p.relative_to(ROOT)))

    colors = {p: i for i, p in enumerate(POLICIES)}
    plot_df = wide[["scenario_id", "winner"]].merge(cov, on="scenario_id")
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        plot_df["prefill_decode_pressure"],
        plot_df["kv_pressure"],
        c=[colors[w] for w in plot_df["winner"]],
        cmap="tab10",
        s=18,
        alpha=0.75,
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=pol, markerfacecolor=plt.get_cmap("tab10")(i), markersize=6)
        for pol, i in colors.items()
    ]
    plt.legend(handles=handles, fontsize=7, loc="best")
    plt.xlabel("prefill/decode pressure")
    plt.ylabel("KV pressure")
    plt.title("Winner map over two mechanism axes")
    plt.tight_layout()
    p = FIG / "winner_map_prefill_kv.png"
    plt.savefig(p, dpi=180)
    plt.close()
    made.append(str(p.relative_to(ROOT)))
    return made


def decide(summary: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    th = generator_spec()["verdict_thresholds"]
    gates = {
        "meaningful_complementarity": summary["unique_winner_fraction"] >= th["strong_min_unique_winner_fraction"]
        and summary["nontrivial_policy_spread_fraction"] >= th["strong_min_nontrivial_spread_fraction"],
        "positive_headroom": summary["oracle_headroom_anwg"] >= th["strong_min_headroom_anwg"],
        "multi_mechanism_gain": summary["gain_share_multi_mechanism_ge2"] >= th["strong_min_gain_share_in_multi_pressure"],
        "not_top10_concentrated": summary["top10_percent_gain_share"] <= th["strong_max_top10_gain_share"],
    }
    if all(gates.values()):
        return "JOINT_GENERALIZATION_STRONG", gates
    if (
        summary["oracle_headroom_anwg"] >= th["partial_min_headroom_anwg"]
        and (summary["unique_winner_fraction"] > 0 or summary["nontrivial_policy_spread_fraction"] > 0)
    ):
        return "JOINT_GENERALIZATION_PARTIAL", gates
    return "JOINT_GENERALIZATION_NO_GO", gates


def write_report(
    *,
    cov_summary: dict[str, Any],
    winner_summary: dict[str, Any],
    disagreement: dict[str, Any],
    local_structure: dict[str, Any],
    old_cmp: dict[str, Any],
    robustness: dict[str, Any],
    decision: dict[str, Any],
    figures: list[str],
    wall_s: float,
) -> None:
    report = ROOT / "docs" / "current" / "joint_multimechanism_generalization_v1_analysis_20260824.md"
    lines = [
        "# joint_multimechanism_generalization_v1 Analysis (2026-08-24)",
        "",
        "CPU-only six-policy workload-breadth experiment. No selector, search, DEV, TEST, FINAL, GPU, vLLM, or Wulver work was used.",
        "",
        "## Verdict",
        f"- Scientific verdict: `{decision['verdict']}`",
        f"- Best fixed policy: `{winner_summary['best_fixed_policy']}`",
        f"- Best fixed mean ANWG: {winner_summary['best_fixed_mean_utility']:.6f}",
        f"- Six-policy oracle mean ANWG: {winner_summary['oracle_mean_utility']:.6f}",
        f"- Oracle headroom: {winner_summary['oracle_headroom_anwg']:.6f}",
        f"- Unique-winner fraction at epsilon={PRACTICAL_EPSILON}: {winner_summary['unique_winner_fraction']:.3f}",
        f"- Nontrivial policy-spread fraction: {winner_summary['nontrivial_policy_spread_fraction']:.3f}",
        f"- Multi-mechanism gain share (>=2 elevated pressures): {winner_summary['gain_share_multi_mechanism_ge2']:.3f}",
        f"- Top-10% scenario gain share: {winner_summary['top10_percent_gain_share']:.3f}",
        "",
        "## Workload Coverage",
        f"- Scenarios: {cov_summary['n_scenarios']}",
        f"- >=2 elevated mechanisms: {cov_summary['multi_mechanism_scenarios_ge2']} ({cov_summary['multi_mechanism_fraction_ge2']:.3f})",
        f"- >=3 elevated mechanisms: {cov_summary['triple_plus_mechanism_scenarios']}",
        f"- Elevated-mechanism counts: `{cov_summary['mechanism_pressure_counts']}`",
        "",
        "## Winner Distribution",
        f"- Winner counts: `{winner_summary['winner_counts']}`",
        f"- Unique winner counts: `{winner_summary['unique_winner_counts_epsilon']}`",
        "",
        "## Decision Disagreement",
        f"- Metric: {disagreement['metric']}",
        f"- Caveat: {disagreement['caveat']}",
        f"- Overall mean pairwise disagreement: {disagreement['overall_mean_pairwise_disagreement']:.6f}",
        f"- Correlation with policy utility range: {disagreement['scenario_disagreement_policy_range_correlation']:.6f}",
        "",
        "## Local Winner Structure",
        f"- kNN winner consistency mean: {local_structure['nearest_neighbor_winner_consistency_mean']:.6f}",
        f"- Local winner entropy mean bits: {local_structure['local_winner_entropy_mean_bits']:.6f}",
        "",
        "## Robustness",
        f"- Bootstrap headroom mean: {robustness['headroom_mean']:.6f}",
        f"- Bootstrap 95% CI: {robustness['headroom_ci95']}",
        f"- Best-fixed bootstrap counts: `{robustness['best_fixed_policy_bootstrap_counts']}`",
        "",
        "## A/B/C Comparison",
        f"- Source: `{old_cmp.get('source')}`",
    ]
    if old_cmp.get("available"):
        for fam, row in old_cmp["families"].items():
            lines.append(
                f"- {fam}: n={row['n_scenarios']}, best_fixed={row['best_fixed_policy']}, "
                f"headroom={row['oracle_headroom']:.6f}, winners={row['winner_counts']}"
            )
    lines += [
        "",
        "## Figures",
        *[f"- `{f}`" for f in figures],
        "",
        "## Claim Safety",
        "- Safe: complementarity persists in this broader jointly varying synthetic workload distribution, if stated with the reported bounds.",
        "- Safe: oracle headroom remains positive in scenarios combining multiple stress mechanisms.",
        "- Unsafe: this proves complementarity in arbitrary production traffic.",
        "- Unsafe: this proves an adaptive scheduler can exploit the oracle headroom.",
        "",
        "## Runtime",
        f"- End-to-end experiment wall time: {wall_s:.2f}s",
        "",
    ]
    report.write_text("\n".join(lines))


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    design = {
        "git": git_info(),
        "question": (
            "Do six-policy complementarity, best-fixed-vs-oracle headroom, and "
            "decision disagreement persist under a broader continuous workload "
            "distribution in which multiple scheduling mechanisms vary simultaneously?"
        ),
        "not_in_scope": [
            "selector training",
            "policy search",
            "DEV/TEST/FINAL",
            "GPU/vLLM/Wulver",
            "new scheduler synthesis",
        ],
        "hypotheses": generator_spec()["hypotheses"],
        "verdict_thresholds": generator_spec()["verdict_thresholds"],
    }
    write_json(OUT / "design.json", design)
    spec = generator_spec()
    write_json(OUT / "generator_spec.json", spec)
    (OUT / "generator_spec.sha256").write_text(sha256_file(OUT / "generator_spec.json") + "\n")

    rng = np.random.default_rng(SEED)
    scenarios = [build_scenario(sample_params(rng, i)) for i in range(N_SCENARIOS)]
    cov_df, cov_summary = coverage_summary(scenarios)
    cov_df.to_csv(OUT / "scenario_manifest.csv", index=False)
    (OUT / "scenario_manifest.sha256").write_text(sha256_file(OUT / "scenario_manifest.csv") + "\n")
    write_json(OUT / "coverage_summary.json", cov_summary)

    long_rows: list[dict[str, Any]] = []
    traces: dict[tuple[str, str], list[dict[str, Any]]] = {}
    contention_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # Policy evaluation starts only after generator and manifest hashes exist.
    for s in scenarios:
        for policy_id in POLICIES:
            try:
                row, recs, contention = run_policy_on_scenario(s, policy_id)
                long_rows.append(row)
                traces[(s.scenario_id, policy_id)] = recs
                contention_rows.append({"scenario_id": s.scenario_id, "policy_id": policy_id, **contention})
            except Exception as e:  # noqa: BLE001
                failures.append({"scenario_id": s.scenario_id, "policy_id": policy_id, "error": repr(e)})

    if failures:
        write_json(OUT / "run_integrity.json", {"status": "failed", "failures": failures})
        raise SystemExit("Policy evaluation failure; see run_integrity.json")

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT / "utility_matrix_long.csv", index=False)
    wide = long_df.pivot(index="scenario_id", columns="policy_id", values="primary_utility_anwg").reset_index()
    wide.columns = ["scenario_id"] + [f"anwg__{c}" for c in wide.columns[1:]]
    wide = wide[["scenario_id"] + [f"anwg__{p}" for p in POLICIES]]

    winner_summary = summarize_utilities(wide, cov_df)
    # summarize_utilities mutates wide with winner/oracle columns.
    wide.to_csv(OUT / "utility_matrix_wide.csv", index=False)
    write_json(OUT / "winner_summary.json", {k: winner_summary[k] for k in [
        "mean_by_policy",
        "winner_counts",
        "winner_fractions",
        "unique_winner_counts_epsilon",
        "unique_winner_fraction",
        "nontrivial_policy_spread_fraction",
        "policy_range_summary",
    ]})
    write_json(OUT / "oracle_summary.json", {k: winner_summary[k] for k in [
        "best_fixed_policy",
        "best_fixed_mean_utility",
        "oracle_mean_utility",
        "oracle_headroom_anwg",
        "oracle_gain_summary",
        "top10_percent_gain_share",
        "top5_scenario_gain_share",
    ]})

    disagreement = summarize_action_disagreement(traces, wide)
    local_structure = local_winner_structure(wide, cov_df)
    write_json(OUT / "decision_disagreement_summary.json", {**disagreement, "local_winner_structure": local_structure})
    mixed = {
        "gain_share_multi_mechanism_ge2": winner_summary["gain_share_multi_mechanism_ge2"],
        "gain_share_triple_plus_ge3": winner_summary["gain_share_triple_plus_ge3"],
        "win_by_elevated_mechanism_count": winner_summary["win_by_elevated_mechanism_count"],
        "coverage_pressure_means": cov_summary["pressure_means"],
    }
    write_json(OUT / "mixed_mechanism_summary.json", mixed)

    robustness = bootstrap_robustness(wide)
    write_json(OUT / "robustness_summary.json", robustness)
    old_cmp = old_family_comparison()
    write_json(OUT / "old_family_comparison.json", old_cmp)
    pd.DataFrame(contention_rows).to_csv(OUT / "contention_diagnostics_summary.csv", index=False)

    verdict, gates = decide(winner_summary)
    decision = {
        "verdict": verdict,
        "gates": gates,
        "preregistered_categories": [
            "JOINT_GENERALIZATION_STRONG",
            "JOINT_GENERALIZATION_PARTIAL",
            "JOINT_GENERALIZATION_NO_GO",
        ],
        "next_task_recommendation": "stop science and begin manuscript production"
        if verdict == "JOINT_GENERALIZATION_STRONG"
        else "review whether one narrowly justified validity follow-up is needed before writing",
        "safety": {
            "cpu_only": True,
            "no_selector_training": True,
            "no_policy_search": True,
            "no_dev_test_final": True,
            "no_gpu_vllm_wulver": True,
        },
    }
    write_json(OUT / "decision.json", decision)

    figures = make_figures(cov_df, wide)
    write_report(
        cov_summary=cov_summary,
        winner_summary=winner_summary,
        disagreement=disagreement,
        local_structure=local_structure,
        old_cmp=old_cmp,
        robustness=robustness,
        decision=decision,
        figures=figures,
        wall_s=time.perf_counter() - t0,
    )

    print(stable_json_dumps({
        "status": "complete",
        "verdict": verdict,
        "n_scenarios": N_SCENARIOS,
        "n_policy_cells": len(long_df),
        "best_fixed_policy": winner_summary["best_fixed_policy"],
        "oracle_headroom_anwg": winner_summary["oracle_headroom_anwg"],
        "unique_winner_fraction": winner_summary["unique_winner_fraction"],
        "multi_mechanism_gain_share": winner_summary["gain_share_multi_mechanism_ge2"],
        "top10_gain_share": winner_summary["top10_percent_gain_share"],
        "output_dir": str(OUT.relative_to(ROOT)),
    }))


if __name__ == "__main__":
    main()
