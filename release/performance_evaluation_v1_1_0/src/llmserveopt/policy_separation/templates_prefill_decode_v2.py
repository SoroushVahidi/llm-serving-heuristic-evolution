"""Family B v2: TTFT-contention prefill/decode scenario templates.

See docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md.

Scientific question (refinement, not composition):

    Does prefill/decode control contain at least two stable, mechanistically
    distinct, online-observable policy niches that could justify a future
    composition/synthesis experiment?

v1 diagnosis (frozen; not re-interpreted): the five-policy family collapsed
to one pairwise contrast (full vs small-chunk). decode_priority ≡ small on
clean admission traces because of arrival-FCFS leftover equilibrium;
decode_stalled_steps was 0 because late tenants were blocked in prefill.
v2 keeps only the two anchors and retargets the workload at the TTFT-class
tradeoff that actually moved ANWG.

Workload geometry
-----------------
Long-prompt hog convoy first, short-prompt late tenants overlapping mid-convoy.
Both classes use *short* outputs so e2e SLO is TTFT-dominated. Short outputs
are a labeled synthetic intervention (v1 long decode outputs diluted TTFT
into e2e and saturated TBT).

Policies never see scenario_id, seed, slo_emphasis, hog_count, late_pressure,
or intended winner. class_id is only ``tenant_prefill`` / ``tenant_late``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import GPUConfig
from .builders import req
from .schema import PolicySeparationScenario
from .templates_prefill_decode import (
    BurstGPTUnavailableError,
    STEP_SIZE,
    _load_burstgpt_arrays,
    _sample_lengths,
    resolve_burstgpt_path,
)

GENERATOR_VERSION = "prefill_decode_v2"
FAMILY_NAME = "family_b_prefill_decode_v2"
TEMPLATE_NAME = "case_prefill_decode_ttft_contention_v2"

HOG_PROMPT_MEDIAN = 8192
HOG_PROMPT_LO, HOG_PROMPT_HI = 4096, 16384
LATE_PROMPT_MEDIAN = 128
LATE_PROMPT_LO, LATE_PROMPT_HI = 64, 256

# Synthetic intervention: short outputs so e2e SLO isolates TTFT contention.
OUTPUT_MEDIAN = 80
OUTPUT_LO, OUTPUT_HI = 48, 128

HOG_COUNT = {"low": 12, "high": 24}
LATE_PRESSURE = {"low": 12, "high": 40}

HOG_ARRIVAL_DT = 0.003
LATE_ARRIVAL_DT = 0.004
LATE_START_FRAC = 0.25

# e2e slack: tight class gets a small margin on top of nominal decode time
# (output * step_size). Prefill time is *not* in the deadline, so convoy
# blocking bites. The loose class is unconstrained.
SLO_EMPHASIS = {
    "hog_ttft": {"slack_hog_s": 0.05, "slack_late_s": 2.0},
    "late_ttft": {"slack_hog_s": 2.0, "slack_late_s": 0.08},
}

# Analysis-side TBT bound (hidden). TBT is not the v2 separator; recorded
# so we can confirm it saturates.
TBT_SLO_S = 0.002

DEFAULT_MAX_ACTIVE = 512
DEFAULT_STEP_TOKEN_BUDGET = 512

CLASS_HOG = "tenant_prefill"
CLASS_LATE = "tenant_late"

ALLOWED_CLASS_IDS = {CLASS_HOG, CLASS_LATE}


def case_prefill_decode_ttft_contention(
    *,
    hog_count: str,
    late_pressure: str,
    slo_emphasis: str,
    seed: int,
    n_hog: Optional[int] = None,
    n_late: Optional[int] = None,
    max_active_sequences: int = DEFAULT_MAX_ACTIVE,
    step_token_budget: int = DEFAULT_STEP_TOKEN_BUDGET,
    allow_synthetic_tokens: bool = False,
    datasets_root: Optional[Path] = None,
) -> PolicySeparationScenario:
    """Build one Family B v2 cell: hog convoy + mid-overlap late tenants."""
    if hog_count not in HOG_COUNT:
        raise ValueError(
            f"hog_count must be one of {sorted(HOG_COUNT)}, got {hog_count!r}"
        )
    if late_pressure not in LATE_PRESSURE:
        raise ValueError(
            f"late_pressure must be one of {sorted(LATE_PRESSURE)}, got {late_pressure!r}"
        )
    if slo_emphasis not in SLO_EMPHASIS:
        raise ValueError(
            f"slo_emphasis must be one of {sorted(SLO_EMPHASIS)}, got {slo_emphasis!r}"
        )

    rng = np.random.default_rng(seed)

    path = resolve_burstgpt_path(datasets_root=datasets_root)
    pool = None
    if path is not None:
        pool_p, pool_o = _load_burstgpt_arrays(str(path))
        pool = (pool_p, pool_o)
    elif not allow_synthetic_tokens:
        raise BurstGPTUnavailableError(
            "Family B v2 production mode requires staged BurstGPT "
            "(LLM_SERVEOPT_BURSTGPT_CSV or staged shards). Refusing silent "
            "synthetic fallback."
        )

    p_pool = pool[0] if pool is not None else None
    o_pool = pool[1] if pool is not None else None

    n_hog_ = int(n_hog) if n_hog is not None else int(HOG_COUNT[hog_count])
    n_late_ = int(n_late) if n_late is not None else int(LATE_PRESSURE[late_pressure])
    n_total = n_hog_ + n_late_

    hog_prompts, src_hp = _sample_lengths(
        rng,
        p_pool,
        n_hog_,
        median=HOG_PROMPT_MEDIAN,
        sigma_anchor=0.6,
        lo=HOG_PROMPT_LO,
        hi=HOG_PROMPT_HI,
        prefer_real=True,
    )
    late_prompts, src_lp = _sample_lengths(
        rng,
        p_pool,
        n_late_,
        median=LATE_PROMPT_MEDIAN,
        sigma_anchor=0.5,
        lo=LATE_PROMPT_LO,
        hi=LATE_PROMPT_HI,
        prefer_real=True,
    )
    # Short outputs: labeled synthetic intervention (prefer_real=False).
    hog_outputs, src_ho = _sample_lengths(
        rng,
        o_pool,
        n_hog_,
        median=OUTPUT_MEDIAN,
        sigma_anchor=0.35,
        lo=OUTPUT_LO,
        hi=OUTPUT_HI,
        prefer_real=False,
    )
    late_outputs, src_lo = _sample_lengths(
        rng,
        o_pool,
        n_late_,
        median=OUTPUT_MEDIAN,
        sigma_anchor=0.35,
        lo=OUTPUT_LO,
        hi=OUTPUT_HI,
        prefer_real=False,
    )

    arrivals_hog = np.array(
        [i * HOG_ARRIVAL_DT for i in range(n_hog_)], dtype=float
    )
    arrivals_hog = arrivals_hog + rng.uniform(
        0.0, HOG_ARRIVAL_DT * 0.25, size=n_hog_
    )
    arrivals_hog.sort()

    convoy_span = float((n_hog_ - 1) * HOG_ARRIVAL_DT) if n_hog_ > 1 else 0.0
    late_start = LATE_START_FRAC * convoy_span
    arrivals_late = late_start + np.cumsum(
        rng.exponential(LATE_ARRIVAL_DT, size=n_late_)
    )

    slack_hog = float(SLO_EMPHASIS[slo_emphasis]["slack_hog_s"])
    slack_late = float(SLO_EMPHASIS[slo_emphasis]["slack_late_s"])

    requests: List = []
    rid = 0
    for arr, p, o in zip(arrivals_hog, hog_prompts, hog_outputs):
        requests.append(
            req(
                request_id=rid,
                arrival_time=float(arr),
                prompt_tokens=int(p),
                predicted_output_tokens=int(o),
                slo_deadline=float(arr) + float(o) * STEP_SIZE + slack_hog,
                priority=1.0,
                class_id=CLASS_HOG,
            )
        )
        rid += 1
    for arr, p, o in zip(arrivals_late, late_prompts, late_outputs):
        requests.append(
            req(
                request_id=rid,
                arrival_time=float(arr),
                prompt_tokens=int(p),
                predicted_output_tokens=int(o),
                slo_deadline=float(arr) + float(o) * STEP_SIZE + slack_late,
                priority=1.0,
                class_id=CLASS_LATE,
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
        "decode_first": False,  # overridden per policy by the runner
    }

    hypothesis = (
        f"Under {slo_emphasis} SLO emphasis with hog_count={hog_count} and "
        f"late_pressure={late_pressure}, uninterrupted full prefill should "
        "win when hog-class e2e slack is tight, and small-chunk prefill "
        "should win when late-class e2e slack is tight; the direction should "
        "track the observable per-class deadline mix, not hidden scenario id."
    )

    token_sources = {
        "hog_prompt": src_hp,
        "late_prompt": src_lp,
        "hog_output": src_ho,
        "late_output": src_lo,
        "output_intervention": "synthetic_short_output_for_ttft_isolation",
        "burstgpt_path": str(path) if path is not None else None,
    }

    params: Dict[str, Any] = {
        "hog_count": hog_count,
        "late_pressure": late_pressure,
        "slo_emphasis": slo_emphasis,
        "n_total_jobs": int(n_total),
        "n_hog": int(n_hog_),
        "n_late": int(n_late_),
        "max_active_sequences": int(max_active_sequences),
        "step_token_budget": int(step_token_budget),
        "hog_prompt_median": int(np.median(hog_prompts)),
        "late_prompt_median": int(np.median(late_prompts)),
        "output_median": int(np.median(np.concatenate([hog_outputs, late_outputs]))),
        "late_start_s": float(late_start),
        "slack_hog_s": slack_hog,
        "slack_late_s": slack_late,
        "tbt_slo_s": TBT_SLO_S,
        "generator_family": FAMILY_NAME,
        "allow_synthetic_tokens": bool(allow_synthetic_tokens),
        "token_sources": token_sources,
        "arrival_shape": "hog_convoy_midoverlap_late_tenants",
        "output_intervention": "synthetic_short_output_for_ttft_isolation",
    }

    n_hog_tag = n_hog_
    n_late_tag = n_late_
    scenario_id = (
        f"pd2.hog{n_hog_tag}"
        f".late{n_late_tag}"
        f".slo{slo_emphasis}"
        f".s{seed}"
    )

    role = "stress" if late_pressure == "high" else "control"

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
        target_mechanism="ttft_class_contention_chunk_control",
        expected_qualitative_hypothesis=hypothesis,
        stress_control_relationship=role,
        pair_id=(
            f"pd2.hog{n_hog_tag}"
            f".late{n_late_tag}"
            f".slo{slo_emphasis}"
        ),
        changed_parameters=("seed",),
    )


def assert_policy_visible_fields_clean_v2(scenario: PolicySeparationScenario) -> None:
    """Anti-leakage guard for policy-visible Request fields (Family B v2)."""
    forbidden = (
        "hog_ttft",
        "late_ttft",
        "slo_emphasis",
        "late_pressure",
        "hog_count",
        "intended",
        "winner",
        "full_prefill",
        "chunked",
        "scenario",
        "pd2.",
    )
    for r in scenario.requests:
        if r.class_id not in ALLOWED_CLASS_IDS:
            raise AssertionError(
                f"scenario {scenario.scenario_id}: illegal class_id {r.class_id!r}"
            )
        val = str(r.class_id).lower()
        for label in forbidden:
            if label in val:
                raise AssertionError(
                    f"scenario {scenario.scenario_id}: visible field "
                    f"class_id={val!r} encodes factor label {label!r}"
                )
        if not (0 < r.prompt_tokens <= 32768):
            raise AssertionError(
                f"scenario {scenario.scenario_id}: prompt out of range"
            )
        if r.priority != 1.0:
            raise AssertionError(
                f"scenario {scenario.scenario_id}: unexpected priority"
            )
        # Seed / scenario identity must not appear on the request.
        if str(scenario.seed) in r.class_id:
            raise AssertionError("seed leaked into class_id")
        if scenario.scenario_id.lower() in r.class_id.lower():
            raise AssertionError("scenario_id leaked into class_id")
