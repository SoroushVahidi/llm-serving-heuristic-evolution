"""Family B v1: prefill/decode interference + chunk-control scenario templates.

See docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md.

Scientific question: "When should a scheduler process a prefill
aggressively/full, and when should it chunk or defer prefill work to protect
active decode progress?"

Execution-model fidelity
------------------------
Under ``enable_decode_prefill_contention=True, decode_first=False`` the
simulator consumes one shared per-step budget in FCFS-by-**arrival** order.
An earlier-arriving large prefill chunk can therefore starve later-arriving
work for that step; a later-arriving prefill cannot stall earlier-arriving
decodes. Consequently the discriminative arrival shape for this family is a
**prefill convoy first**, with decode-tenant arrivals overlapping the convoy
(not "steady decode then late prefill burst").

That is still the scientifically intended interference: full/unchunked
prefill finishes early tenants faster (TTFT) but blocks later decode
tenants; chunked / decode-priority leaves budget crumbs so later decode
tenants can start. ``decode_stalled_steps`` may stay near zero on greedy
natural traces (entering decode while an earlier full prefill still runs is
structurally hard); primary separation signals are ANWG, per-class TTFT,
prefill-stall diagnostics, and decode-tenant SLO attainment.

Field provenance is documented explicitly (real-trace-anchored vs derived vs
synthetic intervention). Production generation requires staged BurstGPT;
``allow_synthetic_tokens=True`` is permitted for unit tests / local smoke only.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.types import GPUConfig, Request
from .builders import req
from .schema import PolicySeparationScenario
from .templates_fairness_starvation_v2 import (
    BurstGPTUnavailableError,
    _get_staged_burstgpt_path,
)

GENERATOR_VERSION = "prefill_decode_v1"
FAMILY_NAME = "family_b_prefill_decode_v1"
TEMPLATE_NAME = "case_prefill_decode_interference_v1"

STEP_SIZE = 0.001  # ServiceModel default step size (1ms)

# Prompt-length medians per prefill_size_class (controlled intervention).
PREFILL_PROMPT_MEDIAN = {
    "short": 512,
    "medium": 2048,
    "long": 8192,
}
DECODE_PROMPT_MEDIAN = 128
PREFILL_OUTPUT_MEDIAN = 128
# Decode-tenant output median (controlled; drives e2e decode work).
DECODE_OUTPUT_MEDIAN = {
    "low": 400,
    "medium": 1200,
    "high": 1500,
}

# Prefill-convoy size and decode-overlap intensity.
N_PREFILL = {
    "moderate": {"short": 40, "medium": 30, "long": 30, "mixed": 30},
    "high": {"short": 60, "medium": 45, "long": 40, "mixed": 40},
}
N_DECODE = {
    "low": 12,
    "medium": 50,
    "high": 80,
}
# Decode arrivals start while the convoy is still running (except low,
# which starts after a short delay so overlap is thin).
DECODE_START_S = {
    "low": 0.50,
    "medium": 0.02,
    "high": 0.02,
}
DECODE_ARRIVAL_DT = {
    "low": 0.10,
    "medium": 0.015,
    "high": 0.010,
}
PREFILL_ARRIVAL_DT = 0.002

# End-to-end deadline slacks (seconds). Decode slack sits on top of
# nominal decode time (output * step_size); it is the TTFT/queue margin
# that makes convoy blocking bite under decode-tight regimes.
SLO_REGIME_SLACK = {
    "ttft_tight": {"slack_prefill_s": 0.08, "decode_margin_s": 1.00},
    "tbt_tight": {"slack_prefill_s": 3.00, "decode_margin_s": 0.15},
    "balanced": {"slack_prefill_s": 0.45, "decode_margin_s": 0.35},
}

# Analysis-side SLO bounds (hidden from policies; post-hoc TTFT/TBT only).
TTFT_SLO_BOUND = {"ttft_tight": 0.05, "tbt_tight": 0.40, "balanced": 0.15}
TBT_SLO_BOUND = {"ttft_tight": 0.004, "tbt_tight": 0.0016, "balanced": 0.0025}

DEFAULT_MAX_ACTIVE = 512
DEFAULT_STEP_TOKEN_BUDGET = 512

PREFILL_SIZE_CLASSES = ("short", "medium", "long", "mixed")


@lru_cache(maxsize=4)
def _load_burstgpt_arrays(path_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load prompt/output token arrays from a staged BurstGPT CSV."""
    from ..workloads.burstgpt import detect_burstgpt_schema

    df = pd.read_csv(path_str, nrows=30000)
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


def _log_sigma(pool: Optional[np.ndarray], fallback: float) -> float:
    if pool is None or len(pool) < 10:
        return fallback
    positive = pool[pool > 0]
    if len(positive) < 10:
        return fallback
    return float(np.std(np.log(positive)))


def _sample_lengths(
    rng: np.random.Generator,
    pool: Optional[np.ndarray],
    count: int,
    *,
    median: int,
    sigma_anchor: float,
    lo: int,
    hi: int,
    prefer_real: bool = True,
) -> Tuple[np.ndarray, str]:
    """Sample lengths with explicit provenance.

    ``prefer_real=False`` forces a controlled lognormal around ``median``
    (BurstGPT shape-anchored when a pool exists). Used for decode outputs
    and other occupancy-/deadline-critical fields.
    """
    if prefer_real and pool is not None:
        real = pool[(pool >= lo) & (pool < hi)]
        if len(real) >= max(4, count // 4):
            return rng.choice(real, size=count).astype(int), "burstgpt_staged"
    if pool is not None:
        source = "burstgpt_anchored"
        sigma = _log_sigma(pool, sigma_anchor)
    else:
        source = "synthetic_lognormal"
        sigma = sigma_anchor
    vals = rng.lognormal(mean=np.log(max(median, 1)), sigma=sigma, size=count)
    return np.clip(vals, lo, hi).astype(int), source


def _prompt_window(prefill_size_class: str) -> Tuple[int, int]:
    return {
        "short": (256, 1024),
        "medium": (1024, 4096),
        "long": (4096, 16384),
        "mixed": (256, 16384),
    }[prefill_size_class]


def _sample_prefill_prompts(
    rng: np.random.Generator,
    pool: Optional[np.ndarray],
    count: int,
    *,
    prefill_size_class: str,
) -> Tuple[np.ndarray, str]:
    if prefill_size_class == "mixed":
        half = count // 2
        short_vals, s1 = _sample_lengths(
            rng, pool, half, median=512, sigma_anchor=0.6, lo=256, hi=1024
        )
        long_vals, s2 = _sample_lengths(
            rng, pool, count - half, median=8192, sigma_anchor=0.6, lo=4096, hi=16384
        )
        vals = _interleave(short_vals, long_vals)
        source = s1 if s1 == s2 else "mixed_sources"
        return vals, source
    lo, hi = _prompt_window(prefill_size_class)
    return _sample_lengths(
        rng,
        pool,
        count,
        median=PREFILL_PROMPT_MEDIAN[prefill_size_class],
        sigma_anchor=0.6,
        lo=lo,
        hi=hi,
    )


def _interleave(short_vals: np.ndarray, long_vals: np.ndarray) -> np.ndarray:
    out = []
    i = j = 0
    n = len(short_vals) + len(long_vals)
    for k in range(n):
        if k % 2 == 0 and i < len(short_vals):
            out.append(int(short_vals[i]))
            i += 1
        elif j < len(long_vals):
            out.append(int(long_vals[j]))
            j += 1
        else:
            out.append(int(short_vals[i]))
            i += 1
    return np.asarray(out, dtype=int)


def case_prefill_decode_interference(
    *,
    prefill_size_class: str,
    decode_occupancy: str,
    slo_regime: str,
    offered_load: str,
    seed: int,
    n_decode: Optional[int] = None,
    n_prefill: Optional[int] = None,
    max_active_sequences: int = DEFAULT_MAX_ACTIVE,
    step_token_budget: int = DEFAULT_STEP_TOKEN_BUDGET,
    allow_synthetic_tokens: bool = False,
    datasets_root: Optional[Path] = None,
) -> PolicySeparationScenario:
    """Build one Family B v1 scenario cell (prefill convoy + overlapping decodes)."""
    if prefill_size_class not in PREFILL_SIZE_CLASSES:
        raise ValueError(
            f"prefill_size_class must be one of {PREFILL_SIZE_CLASSES}, "
            f"got {prefill_size_class!r}"
        )
    if decode_occupancy not in N_DECODE:
        raise ValueError(
            f"decode_occupancy must be one of {sorted(N_DECODE)}, "
            f"got {decode_occupancy!r}"
        )
    if slo_regime not in SLO_REGIME_SLACK:
        raise ValueError(
            f"slo_regime must be one of {sorted(SLO_REGIME_SLACK)}, got {slo_regime!r}"
        )
    if offered_load not in N_PREFILL:
        raise ValueError(
            f"offered_load must be one of {sorted(N_PREFILL)}, got {offered_load!r}"
        )

    rng = np.random.default_rng(seed)

    path = resolve_burstgpt_path(datasets_root=datasets_root)
    pool = None
    if path is not None:
        pool_p, pool_o = _load_burstgpt_arrays(str(path))
        pool = (pool_p, pool_o)
    elif not allow_synthetic_tokens:
        raise BurstGPTUnavailableError(
            "Family B v1 production mode requires staged BurstGPT "
            "(LLM_SERVEOPT_BURSTGPT_CSV or staged shards). Refusing silent "
            "synthetic fallback."
        )

    p_pool = pool[0] if pool is not None else None
    o_pool = pool[1] if pool is not None else None

    n_prefill_ = (
        int(n_prefill)
        if n_prefill is not None
        else int(N_PREFILL[offered_load][prefill_size_class])
    )
    n_decode_ = int(n_decode) if n_decode is not None else int(N_DECODE[decode_occupancy])
    n_total = n_decode_ + n_prefill_

    prefill_prompts, src_pp = _sample_prefill_prompts(
        rng, p_pool, n_prefill_, prefill_size_class=prefill_size_class
    )
    prefill_outputs, src_po = _sample_lengths(
        rng,
        o_pool,
        n_prefill_,
        median=PREFILL_OUTPUT_MEDIAN,
        sigma_anchor=0.4,
        lo=32,
        hi=512,
    )
    decode_prompts, src_dp = _sample_lengths(
        rng,
        p_pool,
        n_decode_,
        median=DECODE_PROMPT_MEDIAN,
        sigma_anchor=0.5,
        lo=32,
        hi=256,
    )
    # Occupancy-/deadline-critical: controlled median (shape-anchored).
    decode_out_median = int(DECODE_OUTPUT_MEDIAN[decode_occupancy])
    decode_outputs, src_do = _sample_lengths(
        rng,
        o_pool,
        n_decode_,
        median=decode_out_median,
        sigma_anchor=0.35,
        lo=max(64, decode_out_median // 4),
        hi=min(8192, decode_out_median * 3),
        prefer_real=False,
    )

    # --- Arrivals: prefill convoy first, decode overlap by occupancy ---
    arrivals_prefill = np.array(
        [i * PREFILL_ARRIVAL_DT for i in range(n_prefill_)], dtype=float
    )
    # Light jitter so ties are seed-stable but not perfectly periodic.
    arrivals_prefill = arrivals_prefill + rng.uniform(0.0, PREFILL_ARRIVAL_DT * 0.25, size=n_prefill_)
    arrivals_prefill.sort()

    decode_start = float(DECODE_START_S[decode_occupancy])
    decode_dt = float(DECODE_ARRIVAL_DT[decode_occupancy])
    arrivals_decode = decode_start + np.cumsum(
        rng.exponential(decode_dt, size=n_decode_)
    )

    slack_p = float(SLO_REGIME_SLACK[slo_regime]["slack_prefill_s"])
    decode_margin = float(SLO_REGIME_SLACK[slo_regime]["decode_margin_s"])

    requests: List[Request] = []
    rid = 0
    for arr, p, o in zip(arrivals_prefill, prefill_prompts, prefill_outputs):
        requests.append(
            req(
                request_id=rid,
                arrival_time=float(arr),
                prompt_tokens=int(p),
                predicted_output_tokens=int(o),
                slo_deadline=float(arr) + slack_p,
                priority=1.0,
                class_id="tenant_prefill",
            )
        )
        rid += 1
    for arr, p, o in zip(arrivals_decode, decode_prompts, decode_outputs):
        slack_d = float(o) * STEP_SIZE + decode_margin
        requests.append(
            req(
                request_id=rid,
                arrival_time=float(arr),
                prompt_tokens=int(p),
                predicted_output_tokens=int(o),
                slo_deadline=float(arr) + slack_d,
                priority=1.0,
                class_id="tenant_decode",
            )
        )
        rid += 1

    requests_sorted = tuple(sorted(requests, key=lambda r: (r.arrival_time, r.request_id)))

    gpu_configs = (
        GPUConfig(
            gpu_id=0,
            max_active_sequences=int(max_active_sequences),
            max_batch_tokens=int(max_active_sequences),
            max_kv_tokens=8_000_000,
        ),
    )

    service_model_kwargs: Dict[str, Any] = {
        "step_size": STEP_SIZE,
        "enable_prefill_modeling": True,
        "prefill_cost_per_token": 1.0,
        "step_token_budget": int(step_token_budget),
        "enable_decode_prefill_contention": True,
        "decode_first": False,  # overridden per policy variant by the runner
    }

    hypothesis = (
        f"When a {prefill_size_class} prefill convoy overlaps {decode_occupancy} "
        f"late decode arrivals under {slo_regime} SLOs, full/unchunked prefill "
        "should win TTFT-tight low-overlap regimes while chunked or "
        "decode-priority prefill should win decode-deadline-tight high-overlap "
        "regimes; fixed chunking and full prefill should each be non-universal."
    )

    token_sources = {
        "prefill_prompt": src_pp,
        "prefill_output": src_po,
        "decode_prompt": src_dp,
        "decode_output": src_do,
        "burstgpt_path": str(path) if path is not None else None,
    }

    params: Dict[str, Any] = {
        "prefill_size_class": prefill_size_class,
        "decode_occupancy": decode_occupancy,
        "slo_regime": slo_regime,
        "offered_load": offered_load,
        "n_total_jobs": int(n_total),
        "n_prefill": int(n_prefill_),
        "n_decode": int(n_decode_),
        "max_active_sequences": int(max_active_sequences),
        "step_token_budget": int(step_token_budget),
        "prefill_prompt_median": int(np.median(prefill_prompts)),
        "decode_output_median": int(np.median(decode_outputs)),
        "decode_start_s": decode_start,
        "slack_prefill_s": slack_p,
        "decode_margin_s": decode_margin,
        "ttft_slo_s": TTFT_SLO_BOUND[slo_regime],
        "tbt_slo_s": TBT_SLO_BOUND[slo_regime],
        "generator_family": FAMILY_NAME,
        "allow_synthetic_tokens": bool(allow_synthetic_tokens),
        "token_sources": token_sources,
        "arrival_shape": "prefill_convoy_then_overlapping_decode",
    }

    scenario_id = (
        f"pd1.psize{prefill_size_class}"
        f".occ{decode_occupancy}"
        f".slo{slo_regime}"
        f".load{offered_load}"
        f".s{seed}"
    )

    role = (
        "stress"
        if decode_occupancy == "high"
        else ("control" if decode_occupancy == "low" else None)
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
        service_model_kwargs=service_model_kwargs,
        target_policy_family="prefill_decode_control",
        target_mechanism="prefill_decode_interference_chunk_control",
        expected_qualitative_hypothesis=hypothesis,
        stress_control_relationship=role,
        pair_id=(
            f"pd1.psize{prefill_size_class}"
            f".occ{decode_occupancy}"
            f".slo{slo_regime}"
            f".load{offered_load}"
        ),
        changed_parameters=("seed",),
    )


def assert_policy_visible_fields_clean(scenario: PolicySeparationScenario) -> None:
    """Anti-leakage guard for policy-visible Request fields."""
    for r in scenario.requests:
        if r.class_id not in {"tenant_prefill", "tenant_decode"}:
            raise AssertionError(
                f"scenario {scenario.scenario_id}: illegal class_id {r.class_id!r}"
            )
    visible_str = {"class_id"}
    for r in scenario.requests:
        for field in visible_str:
            val = str(getattr(r, field)).lower()
            for label in (
                "short",
                "medium",
                "long",
                "mixed",
                "low",
                "high",
                "ttft_tight",
                "tbt_tight",
                "balanced",
                "moderate",
            ):
                if label in val:
                    raise AssertionError(
                        f"scenario {scenario.scenario_id}: visible field "
                        f"{field}={val!r} encodes factor label {label!r}"
                    )
    for r in scenario.requests:
        if not (0 < r.prompt_tokens <= 32768):
            raise AssertionError(f"scenario {scenario.scenario_id}: prompt out of range")
        if r.priority != 1.0:
            raise AssertionError(f"scenario {scenario.scenario_id}: unexpected priority")
