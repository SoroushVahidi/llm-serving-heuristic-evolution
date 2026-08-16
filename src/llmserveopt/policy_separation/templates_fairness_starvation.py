"""Fairness, weight skew, and aging starvation scenario templates (Family A).

A hybrid real-trace-anchored or synthetic-fallback workload construction
designed to measure the exact crossover boundary where size-based/throughput-
oriented scheduling (ESTF) degrades due to tenant starvation, and fairness/
starvation-aware scheduling (Weighted Fair Share, Aging Priority) becomes
advantageous.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.types import GPUConfig, Request
from .builders import req, generous_gpu
from .schema import PolicySeparationScenario

GENERATOR_VERSION = "fairness_starvation_v1"
STEP_SIZE = 0.001  # ServiceModel default step_size (1ms)

# Wolverine cluster staging root for raw datasets
DATASETS_ROOT = Path("/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets")


def _get_staged_burstgpt_path() -> Optional[Path]:
    """Return path to staged BurstGPT csv if available on the cluster."""
    p = DATASETS_ROOT / "burstgpt_v2" / "raw" / "BurstGPT_without_fails.csv"
    return p if p.exists() else None


def _get_staged_azure_path() -> Optional[Path]:
    """Return path to staged Azure csv if available on the cluster."""
    p = DATASETS_ROOT / "azure_llm_2023" / "raw" / "AzureLLMInferenceTrace_conv_2023.csv"
    return p if p.exists() else None


def sample_trace_token_lengths(
    rng: np.random.Generator,
    count: int,
    use_bulk: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample prompt and output token lengths from staged real traces if present,
    falling back to synthetic lognormal distributions if raw files are missing
    (e.g., in local developer environments/tests)."""
    
    burstgpt_path = _get_staged_burstgpt_path()
    if burstgpt_path is not None:
        try:
            # Load a random chunk of the CSV to avoid in-memory overhead
            # We sample from the prompt/response tokens columns
            df = pd.read_csv(burstgpt_path, nrows=5000)
            p_col = "Request Token" if "Request Token" in df.columns else "request_token"
            r_col = "Response Token" if "Response Token" in df.columns else "response_token"
            
            p_vals = df[p_col].dropna().values
            r_vals = df[r_col].dropna().values
            
            if len(p_vals) > 0 and len(r_vals) > 0:
                if use_bulk:
                    # Filter for bulk-like requests (longer prompt/output)
                    p_bulk = p_vals[p_vals >= np.percentile(p_vals, 50)]
                    r_bulk = r_vals[r_vals >= np.percentile(r_vals, 50)]
                    prompts = rng.choice(p_bulk if len(p_bulk) > 0 else p_vals, size=count)
                    outputs = rng.choice(r_bulk if len(r_bulk) > 0 else r_vals, size=count)
                else:
                    # Filter for interactive-like requests (shorter prompt/output)
                    p_inter = p_vals[p_vals <= np.percentile(p_vals, 50)]
                    r_inter = r_vals[r_vals <= np.percentile(r_vals, 50)]
                    prompts = rng.choice(p_inter if len(p_inter) > 0 else p_vals, size=count)
                    outputs = rng.choice(r_inter if len(r_inter) > 0 else r_vals, size=count)
                
                # Sane bounds
                return np.clip(prompts, 16, 2048), np.clip(outputs, 8, 1024)
        except Exception:
            pass # Fall through to synthetic lognormal fallback

    # Highly robust synthetic lognormal fallback (identical statistics to repaired VTC sweep)
    if use_bulk:
        prompts = rng.lognormal(mean=np.log(500.0), sigma=0.5, size=count)
        outputs = rng.lognormal(mean=np.log(300.0), sigma=0.4, size=count)
    else:
        prompts = rng.lognormal(mean=np.log(200.0), sigma=0.5, size=count)
        outputs = rng.lognormal(mean=np.log(100.0), sigma=0.4, size=count)
        
    return np.clip(prompts, 16, 2048).astype(int), np.clip(outputs, 8, 1024).astype(int)


def case4_fairness_starvation(
    target_utilization: float,
    tenant_weight_skew: float,
    interactive_volume_fraction: float,
    seed: int,
    n_total_jobs: int = 120,
) -> PolicySeparationScenario:
    """One scenario cell of the Fairness and Starvation pilot.

    Constructs a dual-tenant workload:
    - tenant_interactive: interactive requests (short prompts/outputs, tight SLO, priority scaled by weight skew)
    - tenant_bulk: bulk requests (long prompts/outputs, loose SLO, priority = 1.0)
    
    The `tenant_weight_skew` modifies the relative priority weight of the interactive tenant
    requests vs bulk tenant requests, allowing priority-weighted fair policies (such as
    weighted_fair_share and aging_priority) to act.
    """
    rng = np.random.default_rng(seed)
    
    # Calculate interactive and bulk request counts
    n_interactive = int(round(interactive_volume_fraction * n_total_jobs))
    n_bulk = n_total_jobs - n_interactive
    
    # Sane bounds
    n_interactive = max(5, min(n_total_jobs - 5, n_interactive))
    n_bulk = n_total_jobs - n_interactive

    # Sample prompt and output lengths from trace or synthetic fallback
    p_inter, o_inter = sample_trace_token_lengths(rng, n_interactive, use_bulk=False)
    p_bulk, o_bulk = sample_trace_token_lengths(rng, n_bulk, use_bulk=True)

    # Combine service times to calculate mean and arrival rates
    # Service time in this simulator is strictly output_tokens * STEP_SIZE (prefill is 0s)
    service_s_inter = o_inter * STEP_SIZE
    service_s_bulk = o_bulk * STEP_SIZE
    mean_service_s = float(np.mean(np.concatenate([service_s_inter, service_s_bulk])))
    
    # max_active_sequences = 1 forces strict queue ordering to induce clean contention
    max_active_sequences = 1
    capacity_per_s = max_active_sequences / max_service_capacity_factor(mean_service_s)
    rate = target_utilization * capacity_per_s

    # Poisson arrival process for both tenants via exponential inter-arrival times
    rate_inter = rate * interactive_volume_fraction
    rate_bulk = rate * (1.0 - interactive_volume_fraction)
    arrival_inter = np.cumsum(rng.exponential(1.0 / max(rate_inter, 1e-9), size=n_interactive))
    arrival_bulk = np.cumsum(rng.exponential(1.0 / max(rate_bulk, 1e-9), size=n_bulk))

    # Build individual Request objects
    requests: List[Request] = []
    rid = 0

    # Interactive tenant: high priority (skew), tight SLO (1.0s slack)
    # SLO slack is calibrated to produce a genuine crossover boundary
    for arr, p, o in zip(arrival_inter, p_inter, o_inter):
        requests.append(req(
            request_id=rid,
            arrival_time=float(arr),
            prompt_tokens=int(p),
            predicted_output_tokens=int(o),
            slo_deadline=float(arr) + 1.0,
            priority=float(tenant_weight_skew),
            class_id="tenant_interactive",
        ))
        rid += 1

    # Bulk tenant: low priority (1.0), loose SLO (15.0s slack)
    for arr, p, o in zip(arrival_bulk, p_bulk, o_bulk):
        requests.append(req(
            request_id=rid,
            arrival_time=float(arr),
            prompt_tokens=int(p),
            predicted_output_tokens=int(o),
            slo_deadline=float(arr) + 15.0,
            priority=1.0,
            class_id="tenant_bulk",
        ))
        rid += 1

    # Sort all requests by arrival time
    requests_sorted = tuple(sorted(requests, key=lambda r: r.arrival_time))
    
    # Sane GPU capacity configuration with max_active_sequences=1 to force sequential slot contention
    gpu_configs = (generous_gpu(max_active_sequences=max_active_sequences),)

    role = "control" if tenant_weight_skew == 1.0 else "stress"

    hypothesis = (
        "Under high load and high weight skew, size-based throughput-oriented rules (ESTF) "
        "will starve interactive requests in favor of bulk requests, violating tight interactive "
        "deadlines and causing a drop in ANWG. Introducing fairness/starvation-aware rules (Weighted Fair Share) "
        "will restore interactive SLOs via priority-weighted allocation, outperforming ESTF."
    )

    params = {
        "target_utilization": target_utilization,
        "tenant_weight_skew": tenant_weight_skew,
        "interactive_volume_fraction": interactive_volume_fraction,
        "generator_family": "sobol_family_a_fairness_starvation",
    }

    return PolicySeparationScenario(
        scenario_id=f"fs.util{target_utilization:.4f}.skew{tenant_weight_skew:.4f}.vol{interactive_volume_fraction:.4f}.s{seed}",
        family="sobol_family_a_fairness_starvation",
        template_name="case4_fairness_starvation",
        generator_version=GENERATOR_VERSION,
        seed=seed,
        params=params,
        requests=requests_sorted,
        gpu_configs=gpu_configs,
        target_policy_family="fairness_starvation",
        target_mechanism="tenant_fairness_and_aging",
        expected_qualitative_hypothesis=hypothesis,
        stress_control_relationship=role,
        pair_id=f"fs.util{target_utilization:.4f}.vol{interactive_volume_fraction:.4f}",
        changed_parameters=("tenant_weight_skew",),
    )


def max_service_capacity_factor(mean_service_s: float) -> float:
    """Convenience helper to map mean service time to capacity bounds."""
    return max(mean_service_s, 1e-9)
