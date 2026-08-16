"""Family A v2: fairness vs size with orthogonal size×priority treatments.

See docs/design/POLICY_SEPARATION_FAMILY_A_V2.md.

Production generation requires staged BurstGPT token shapes. Synthetic
lognormal sampling is allowed only when ``allow_synthetic_tokens=True``
(unit tests / local smoke without cluster datasets).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.types import Request
from .builders import generous_gpu, req
from .schema import PolicySeparationScenario
from .templates_fairness_starvation import (
    DATASETS_ROOT,
    STEP_SIZE,
    _get_staged_burstgpt_path,
    max_service_capacity_factor,
)

GENERATOR_VERSION = "fairness_starvation_v2"
FAMILY_NAME = "family_a_fairness_starvation_v2"
TEMPLATE_NAME = "case_fairness_vs_size_v2"

# Calibrated defaults (may be overridden by caller/config).
DEFAULT_N_JOBS = 120
DEFAULT_MAX_ACTIVE = 1
DEFAULT_FAVORED_SLO_SLACK_S = 1.0
DEFAULT_OTHER_SLO_SLACK_S = 8.0


class BurstGPTUnavailableError(RuntimeError):
    """Raised when production generation requires BurstGPT but none is found."""


@lru_cache(maxsize=4)
def _load_burstgpt_arrays(path_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load prompt/output token arrays from a staged BurstGPT CSV.

    Uses the shared BurstGPT schema detector so both ``Request tokens``
    (release v2) and legacy ``Request Token`` / ``request_token`` headers work.
    """
    from ..workloads.burstgpt import detect_burstgpt_schema

    df = pd.read_csv(path_str, nrows=20000)
    schema = detect_burstgpt_schema(list(df.columns))
    p_col = schema["request_tokens"]
    r_col = schema["response_tokens"]
    assert p_col is not None and r_col is not None
    prompts = pd.to_numeric(df[p_col], errors="coerce").dropna().to_numpy(dtype=float)
    outputs = pd.to_numeric(df[r_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(prompts) < 100 or len(outputs) < 100:
        raise BurstGPTUnavailableError(f"BurstGPT CSV too small or unusable: {path_str}")
    return prompts, outputs


def resolve_burstgpt_path(datasets_root: Optional[Path] = None) -> Optional[Path]:
    return _get_staged_burstgpt_path(datasets_root=datasets_root)


def sample_size_class_lengths(
    rng: np.random.Generator,
    count: int,
    *,
    size_class: str,
    allow_synthetic_tokens: bool,
    datasets_root: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, str, Optional[str]]:
    """Sample (prompt, output) lengths for short or long size class.

    Returns prompts, outputs, token_length_source, burstgpt_path_str_or_None.
    """
    if size_class not in {"short", "long"}:
        raise ValueError(f"size_class must be 'short' or 'long', got {size_class!r}")

    path = resolve_burstgpt_path(datasets_root=datasets_root)
    if path is not None:
        prompts_pool, outputs_pool = _load_burstgpt_arrays(str(path))
        p_cut = float(np.percentile(prompts_pool, 50))
        o_cut = float(np.percentile(outputs_pool, 50))
        if size_class == "long":
            p_pool = prompts_pool[prompts_pool >= p_cut]
            o_pool = outputs_pool[outputs_pool >= o_cut]
        else:
            p_pool = prompts_pool[prompts_pool <= p_cut]
            o_pool = outputs_pool[outputs_pool <= o_cut]
        if len(p_pool) == 0:
            p_pool = prompts_pool
        if len(o_pool) == 0:
            o_pool = outputs_pool
        prompts = rng.choice(p_pool, size=count)
        outputs = rng.choice(o_pool, size=count)
        return (
            np.clip(prompts, 16, 2048).astype(int),
            np.clip(outputs, 8, 1024).astype(int),
            "burstgpt_staged",
            str(path),
        )

    if not allow_synthetic_tokens:
        root = Path(datasets_root) if datasets_root is not None else DATASETS_ROOT
        raise BurstGPTUnavailableError(
            "Family A v2 production mode requires staged BurstGPT under "
            f"{root / 'burstgpt_v2' / 'raw'} (or LLM_SERVEOPT_BURSTGPT_CSV). "
            "Refusing silent synthetic fallback."
        )

    if size_class == "long":
        prompts = rng.lognormal(mean=np.log(500.0), sigma=0.5, size=count)
        outputs = rng.lognormal(mean=np.log(300.0), sigma=0.4, size=count)
    else:
        prompts = rng.lognormal(mean=np.log(200.0), sigma=0.5, size=count)
        outputs = rng.lognormal(mean=np.log(100.0), sigma=0.4, size=count)
    return (
        np.clip(prompts, 16, 2048).astype(int),
        np.clip(outputs, 8, 1024).astype(int),
        "synthetic_lognormal_fallback",
        None,
    )


def apply_prediction_noise(
    rng: np.random.Generator,
    actual_output: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Multiplicative lognormal noise on predicted output lengths.

    sigma=0 => predicted == actual (accurate control).
    """
    actual_output = np.asarray(actual_output, dtype=int)
    if sigma <= 0.0:
        return actual_output.copy()
    noise = rng.lognormal(mean=0.0, sigma=float(sigma), size=len(actual_output))
    predicted = np.rint(actual_output * noise).astype(int)
    return np.clip(predicted, 1, 4096)


def case_fairness_vs_size_v2(
    *,
    target_utilization: float,
    tenant_weight_skew: float,
    favored_tenant_size: str,
    prediction_noise_sigma: float,
    seed: int,
    n_total_jobs: int = DEFAULT_N_JOBS,
    max_active_sequences: int = DEFAULT_MAX_ACTIVE,
    favored_slo_slack_s: float = DEFAULT_FAVORED_SLO_SLACK_S,
    other_slo_slack_s: float = DEFAULT_OTHER_SLO_SLACK_S,
    allow_synthetic_tokens: bool = False,
    datasets_root: Optional[Path] = None,
) -> PolicySeparationScenario:
    """Build one Family A v2 scenario with orthogonal size×priority treatment."""
    if favored_tenant_size not in {"short", "long"}:
        raise ValueError(f"favored_tenant_size must be short|long, got {favored_tenant_size!r}")
    if n_total_jobs < 20 or n_total_jobs % 2 != 0:
        raise ValueError("n_total_jobs must be even and >= 20")

    rng = np.random.default_rng(seed)
    n_favored = n_total_jobs // 2
    n_other = n_total_jobs - n_favored
    other_size = "long" if favored_tenant_size == "short" else "short"

    p_fav, o_fav, src_f, burst_path = sample_size_class_lengths(
        rng,
        n_favored,
        size_class=favored_tenant_size,
        allow_synthetic_tokens=allow_synthetic_tokens,
        datasets_root=datasets_root,
    )
    p_oth, o_oth, src_o, burst_path_o = sample_size_class_lengths(
        rng,
        n_other,
        size_class=other_size,
        allow_synthetic_tokens=allow_synthetic_tokens,
        datasets_root=datasets_root,
    )
    if src_f != src_o:
        # Should not happen under consistent allow_synthetic / BurstGPT availability.
        token_source = "mixed_inconsistent"
    else:
        token_source = src_f
    burstgpt_path = burst_path or burst_path_o

    pred_fav = apply_prediction_noise(rng, o_fav, prediction_noise_sigma)
    pred_oth = apply_prediction_noise(rng, o_oth, prediction_noise_sigma)

    service = np.concatenate([o_fav, o_oth]).astype(float) * STEP_SIZE
    mean_service_s = float(np.mean(service))
    capacity_per_s = max_active_sequences / max_service_capacity_factor(mean_service_s)
    rate = target_utilization * capacity_per_s
    rate_each = rate * 0.5
    arr_fav = np.cumsum(rng.exponential(1.0 / max(rate_each, 1e-9), size=n_favored))
    arr_oth = np.cumsum(rng.exponential(1.0 / max(rate_each, 1e-9), size=n_other))

    requests: List[Request] = []
    rid = 0
    for arr, p, actual, pred in zip(arr_fav, p_fav, o_fav, pred_fav):
        requests.append(
            req(
                request_id=rid,
                arrival_time=float(arr),
                prompt_tokens=int(p),
                predicted_output_tokens=int(pred),
                actual_output_tokens=int(actual),
                slo_deadline=float(arr) + float(favored_slo_slack_s),
                priority=float(tenant_weight_skew),
                class_id="tenant_favored",
            )
        )
        rid += 1
    for arr, p, actual, pred in zip(arr_oth, p_oth, o_oth, pred_oth):
        requests.append(
            req(
                request_id=rid,
                arrival_time=float(arr),
                prompt_tokens=int(p),
                predicted_output_tokens=int(pred),
                actual_output_tokens=int(actual),
                slo_deadline=float(arr) + float(other_slo_slack_s),
                priority=1.0,
                class_id="tenant_other",
            )
        )
        rid += 1

    requests_sorted = tuple(sorted(requests, key=lambda r: r.arrival_time))
    gpu_configs = (generous_gpu(max_active_sequences=max_active_sequences),)

    role = "control" if float(tenant_weight_skew) == 1.0 else "stress"
    hypothesis = (
        "When favored_tenant_size=long and skew>1, ESTF prefers short tenant_other "
        "jobs while WFS prefers high-priority long tenant_favored jobs, producing "
        "bidirectional ESTF↔WFS niches on canonical ANWG. Aging should reduce "
        "starvation without universally dominating. FIFO should degrade under overload."
    )

    params: Dict[str, Any] = {
        "target_utilization": float(target_utilization),
        "tenant_weight_skew": float(tenant_weight_skew),
        "favored_tenant_size": favored_tenant_size,
        "other_tenant_size": other_size,
        "prediction_noise_sigma": float(prediction_noise_sigma),
        "n_total_jobs": int(n_total_jobs),
        "max_active_sequences": int(max_active_sequences),
        "favored_slo_slack_s": float(favored_slo_slack_s),
        "other_slo_slack_s": float(other_slo_slack_s),
        "generator_family": FAMILY_NAME,
        "token_length_source": token_source,
        "burstgpt_path": burstgpt_path,
        "allow_synthetic_tokens": bool(allow_synthetic_tokens),
        # Explicit orthogonality bookkeeping (hidden metadata).
        "size_priority_alignment": (
            "aligned_short_high"
            if favored_tenant_size == "short"
            else "conflict_long_high"
        ),
    }

    scenario_id = (
        f"fs2.util{target_utilization:.4f}"
        f".skew{tenant_weight_skew:.4f}"
        f".fav{favored_tenant_size}"
        f".noise{prediction_noise_sigma:.2f}"
        f".s{seed}"
    )

    return PolicySeparationScenario(
        scenario_id=scenario_id,
        family=FAMILY_NAME,
        template_name=TEMPLATE_NAME,
        generator_version=GENERATOR_VERSION,
        seed=int(seed),
        params=params,
        requests=requests_sorted,
        gpu_configs=gpu_configs,
        target_policy_family="fairness_starvation",
        target_mechanism="orthogonal_size_vs_priority",
        expected_qualitative_hypothesis=hypothesis,
        stress_control_relationship=role,
        pair_id=(
            f"fs2.util{target_utilization:.4f}"
            f".fav{favored_tenant_size}"
            f".noise{prediction_noise_sigma:.2f}"
        ),
        changed_parameters=("tenant_weight_skew",),
    )


def assert_size_priority_orthogonality(scenario: PolicySeparationScenario) -> None:
    """Check mean actual output length ordering matches favored_tenant_size."""
    fav = [r for r in scenario.requests if r.class_id == "tenant_favored"]
    oth = [r for r in scenario.requests if r.class_id == "tenant_other"]
    mean_fav = float(np.mean([r.actual_output_tokens for r in fav]))
    mean_oth = float(np.mean([r.actual_output_tokens for r in oth]))
    favored_size = scenario.params["favored_tenant_size"]
    if favored_size == "long" and not (mean_fav > mean_oth):
        raise AssertionError(
            f"expected favored longer than other, got {mean_fav} vs {mean_oth}"
        )
    if favored_size == "short" and not (mean_fav < mean_oth):
        raise AssertionError(
            f"expected favored shorter than other, got {mean_fav} vs {mean_oth}"
        )
