"""Family C v1: KV-pressure admission-reserve scenario templates.

See docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md.

Scientific question: does a KV-occupancy-aware admission reserve
(`kv_constrained_online`) create a niche relative to a KV-blind laxity-ranked
policy (`least_laxity_first`) whose relative advantage can plausibly change
*within* a single scenario's trajectory (as `current_kv_tokens` rises and
falls), rather than only scenario-to-scenario?

Two tenant classes:
- `tenant_bulk`: long-prompt, loose-deadline background load whose sole
  purpose is to drive KV occupancy up. Never itself SLO-relevant.
- `tenant_urgent`: the SLO-critical population. `urgent_arrival_phase`
  controls whether it arrives before or after KV pressure has built up;
  `urgent_tightness` controls whether its own deadline is tight enough to
  trigger `kv_constrained_online`'s urgent-override bypass.

`enable_prefill_modeling` is left at its default (False, instant prefill) so
admission timing -- not prefill-chunk contention (already characterized by
Family B) -- is the sole mechanism under test.

Field provenance, observable/hidden feature split, and the preregistered
hypotheses/gates are documented in the design doc above, not repeated here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..core.types import GPUConfig
from .builders import req, kv_scarce_gpu
from .schema import PolicySeparationScenario
from .templates_prefill_decode import (
    _load_burstgpt_arrays,
    _sample_lengths,
    resolve_burstgpt_path,
)
from .templates_fairness_starvation_v2 import BurstGPTUnavailableError

GENERATOR_VERSION = "kv_pressure_v1"
FAMILY_NAME = "family_c_kv_pressure_v1"
TEMPLATE_NAME = "case_kv_pressure_reserve_contention"

CLASS_BULK = "tenant_bulk"
CLASS_URGENT = "tenant_urgent"
ALLOWED_CLASS_IDS = {CLASS_BULK, CLASS_URGENT}

# --- Bulk tenant (KV-pressure driver) ---
# BULK_PROMPT_HI is capped well below DEFAULT_MAX_KV_TOKENS (calibrated,
# see design doc SS5): the original [2048, 8192] window could sample a
# single bulk request whose prompt_tokens alone exceeded max_kv_tokens,
# making it permanently unadmittable under either policy and corrupting the
# whole scenario's step count/metrics (a workload bug, not a finding).
N_BULK = {"low": 6, "high": 14}
BULK_PROMPT_MEDIAN = 2048
BULK_PROMPT_LO, BULK_PROMPT_HI = 1024, 3072
BULK_OUTPUT_MEDIAN = 300
BULK_OUTPUT_LO, BULK_OUTPUT_HI = 100, 600
BULK_ARRIVAL_DT = 0.05
# 1.5 is a calibrated value (see design doc SS5 calibration note): the
# original 30.0 ("never itself SLO-relevant") target made the reserve's
# admission-deferral cost invisible to bulk tenants -- kv_constrained_online
# then never lost a single smoke cell, an artificial one-sided result, not a
# genuine bidirectional trade-off.
BULK_SLACK_S = 1.5

# --- Urgent tenant (SLO-critical population) ---
N_URGENT = 6  # fixed; not swept, to isolate the two swept timing/tightness factors
URGENT_PROMPT_MEDIAN = 1024
URGENT_PROMPT_LO, URGENT_PROMPT_HI = 512, 2048
URGENT_OUTPUT_MEDIAN = 150  # labeled synthetic intervention, isolates admission latency
URGENT_OUTPUT_LO, URGENT_OUTPUT_HI = 50, 400
URGENT_ARRIVAL_DT = 0.03
# tight=0.55 is a calibrated value (see design doc §5 calibration note):
# the original 0.9 target produced real KV-occupancy differences between the
# two policies but zero outcome difference (slack fully absorbed even the
# largest observed admission delays).
URGENT_SLACK_S = {"loose": 3.0, "tight": 0.55}
URGENT_ARRIVAL_PHASE_FRACTION = {"early": 0.0, "late": 0.7}

# --- GPU: KV capacity is the binding constraint (see builders.kv_scarce_gpu) ---
# 6000 is a calibrated value (see design doc SS5 calibration note, round 3,
# after BULK_PROMPT_HI was capped below max_kv_tokens to eliminate
# permanently-infeasible bulk requests). Earlier rounds (24_000; then 8_000/
# 7_000 before the prompt-window fix) are documented in the design doc for
# provenance.
DEFAULT_MAX_KV_TOKENS = 6_000
DEFAULT_MAX_ACTIVE_SEQUENCES = 64
DEFAULT_MAX_BATCH_TOKENS = 64


def case_kv_pressure_reserve_contention(
    *,
    bulk_pressure: str,
    urgent_arrival_phase: str,
    urgent_tightness: str,
    seed: int,
    n_bulk: Optional[int] = None,
    n_urgent: Optional[int] = None,
    max_kv_tokens: int = DEFAULT_MAX_KV_TOKENS,
    max_active_sequences: int = DEFAULT_MAX_ACTIVE_SEQUENCES,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    allow_synthetic_tokens: bool = False,
    datasets_root: Optional[Path] = None,
) -> PolicySeparationScenario:
    """Build one Family C v1 cell: bulk KV-pressure convoy + a fixed-size
    urgent-tenant population whose arrival timing and deadline tightness are
    swept factors (see module docstring / design doc)."""
    if bulk_pressure not in N_BULK:
        raise ValueError(f"bulk_pressure must be one of {sorted(N_BULK)}, got {bulk_pressure!r}")
    if urgent_arrival_phase not in URGENT_ARRIVAL_PHASE_FRACTION:
        raise ValueError(
            f"urgent_arrival_phase must be one of {sorted(URGENT_ARRIVAL_PHASE_FRACTION)}, "
            f"got {urgent_arrival_phase!r}"
        )
    if urgent_tightness not in URGENT_SLACK_S:
        raise ValueError(
            f"urgent_tightness must be one of {sorted(URGENT_SLACK_S)}, got {urgent_tightness!r}"
        )

    rng = np.random.default_rng(seed)

    path = resolve_burstgpt_path(datasets_root=datasets_root)
    pool = None
    if path is not None:
        pool = _load_burstgpt_arrays(str(path))
    elif not allow_synthetic_tokens:
        raise BurstGPTUnavailableError(
            "Family C v1 production mode requires staged BurstGPT "
            "(LLM_SERVEOPT_BURSTGPT_CSV or staged shards). Refusing silent "
            "synthetic fallback."
        )
    p_pool = pool[0] if pool is not None else None
    o_pool = pool[1] if pool is not None else None

    n_bulk_ = int(n_bulk) if n_bulk is not None else int(N_BULK[bulk_pressure])
    n_urgent_ = int(n_urgent) if n_urgent is not None else N_URGENT

    bulk_prompts, src_bp = _sample_lengths(
        rng, p_pool, n_bulk_, median=BULK_PROMPT_MEDIAN, sigma_anchor=0.6,
        lo=BULK_PROMPT_LO, hi=BULK_PROMPT_HI, prefer_real=True,
    )
    bulk_outputs, src_bo = _sample_lengths(
        rng, o_pool, n_bulk_, median=BULK_OUTPUT_MEDIAN, sigma_anchor=0.4,
        lo=BULK_OUTPUT_LO, hi=BULK_OUTPUT_HI, prefer_real=False,
    )
    urgent_prompts, src_up = _sample_lengths(
        rng, p_pool, n_urgent_, median=URGENT_PROMPT_MEDIAN, sigma_anchor=0.5,
        lo=URGENT_PROMPT_LO, hi=URGENT_PROMPT_HI, prefer_real=True,
    )
    urgent_outputs, src_uo = _sample_lengths(
        rng, o_pool, n_urgent_, median=URGENT_OUTPUT_MEDIAN, sigma_anchor=0.35,
        lo=URGENT_OUTPUT_LO, hi=URGENT_OUTPUT_HI, prefer_real=False,
    )

    arrivals_bulk = np.array([i * BULK_ARRIVAL_DT for i in range(n_bulk_)], dtype=float)
    arrivals_bulk = arrivals_bulk + rng.uniform(0.0, BULK_ARRIVAL_DT * 0.25, size=n_bulk_)
    arrivals_bulk.sort()

    bulk_span = float((n_bulk_ - 1) * BULK_ARRIVAL_DT) if n_bulk_ > 1 else 0.0
    urgent_start = URGENT_ARRIVAL_PHASE_FRACTION[urgent_arrival_phase] * bulk_span
    arrivals_urgent = urgent_start + np.cumsum(
        rng.exponential(URGENT_ARRIVAL_DT, size=n_urgent_)
    )

    slack_urgent = float(URGENT_SLACK_S[urgent_tightness])

    requests = []
    rid = 0
    for arr, p, o in zip(arrivals_bulk, bulk_prompts, bulk_outputs):
        requests.append(req(
            request_id=rid, arrival_time=float(arr), prompt_tokens=int(p),
            predicted_output_tokens=int(o),
            slo_deadline=float(arr) + BULK_SLACK_S,
            priority=1.0, class_id=CLASS_BULK,
        ))
        rid += 1
    for arr, p, o in zip(arrivals_urgent, urgent_prompts, urgent_outputs):
        requests.append(req(
            request_id=rid, arrival_time=float(arr), prompt_tokens=int(p),
            predicted_output_tokens=int(o),
            slo_deadline=float(arr) + slack_urgent,
            priority=1.0, class_id=CLASS_URGENT,
        ))
        rid += 1

    requests.sort(key=lambda r: (r.arrival_time, r.request_id))

    gpu = kv_scarce_gpu(
        max_kv_tokens=max_kv_tokens,
        max_active_sequences=max_active_sequences,
        max_batch_tokens=max_batch_tokens,
    )

    scenario_id = (
        f"kvp1.bulk{n_bulk_}"
        f".phase{urgent_arrival_phase}"
        f".tight{urgent_tightness}"
        f".s{seed}"
    )

    params = {
        "bulk_pressure": bulk_pressure,
        "urgent_arrival_phase": urgent_arrival_phase,
        "urgent_tightness": urgent_tightness,
        "seed": seed,
        "n_bulk": n_bulk_,
        "n_urgent": n_urgent_,
        "max_kv_tokens": max_kv_tokens,
        "max_active_sequences": max_active_sequences,
        "max_batch_tokens": max_batch_tokens,
        "allow_synthetic_tokens": allow_synthetic_tokens,
        "bulk_prompt_source": src_bp,
        "bulk_output_source": src_bo,
        "urgent_prompt_source": src_up,
        "urgent_output_source": src_uo,
        "output_intervention": "synthetic_short_urgent_output_for_admission_latency_isolation",
    }

    return PolicySeparationScenario(
        scenario_id=scenario_id,
        family=FAMILY_NAME,
        template_name=TEMPLATE_NAME,
        generator_version=GENERATOR_VERSION,
        seed=int(seed),
        params=params,
        requests=tuple(requests),
        gpu_configs=(gpu,),
        service_model_kwargs={},
        target_policy_family="kv_pressure_reserve",
        target_mechanism="kv_occupancy_admission_reserve",
        expected_qualitative_hypothesis=(
            "kv_constrained_online's urgent-tenant SLO advantage over "
            "least_laxity_first is larger when urgent tenants arrive after "
            "KV pressure has built up (urgent_arrival_phase=late) than "
            "before (early), and larger under urgent_tightness=tight than "
            "loose."
        ),
        stress_control_relationship=(
            "stress" if (bulk_pressure == "high" and urgent_arrival_phase == "late") else "control"
        ),
        pair_id=f"kvp1.tight{urgent_tightness}.s{seed}",
        changed_parameters=("bulk_pressure", "urgent_arrival_phase"),
    )


def assert_policy_visible_fields_clean_kv_v1(scenario: PolicySeparationScenario) -> None:
    """Anti-leakage guard for policy-visible Request fields (Family C v1)."""
    forbidden = (
        "bulk_pressure", "urgent_arrival_phase", "urgent_tightness",
        "phaseearly", "phaselate", "tighttight", "tightloose",
        "intended", "winner", "kv_constrained", "least_laxity",
        "scenario", "kvp1.",
    )
    for r in scenario.requests:
        if r.class_id not in ALLOWED_CLASS_IDS:
            raise AssertionError(f"scenario {scenario.scenario_id}: illegal class_id {r.class_id!r}")
        val = str(r.class_id).lower()
        for label in forbidden:
            if label in val:
                raise AssertionError(
                    f"scenario {scenario.scenario_id}: visible field "
                    f"class_id={val!r} encodes factor label {label!r}"
                )
        if not (0 < r.prompt_tokens <= 32768):
            raise AssertionError(f"scenario {scenario.scenario_id}: prompt out of range")
        if r.priority != 1.0:
            raise AssertionError(f"scenario {scenario.scenario_id}: unexpected priority")
        if str(scenario.seed) in r.class_id:
            raise AssertionError("seed leaked into class_id")
        if scenario.scenario_id.lower() in r.class_id.lower():
            raise AssertionError("scenario_id leaked into class_id")
