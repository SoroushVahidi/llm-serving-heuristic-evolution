"""Family C v2: KV-pressure admission-reserve scenario templates (refinement).

See docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md. Extends v1
(`templates_kv_pressure.py`, frozen -- its module constants and the v1 CSV
output are untouched by this file) to address two things v1's diagnosis
found:

1. **ANWG resolution.** v1 used only 20 requests/scenario, giving ANWG only
   9 achievable discrete values across the whole pilot and making "tie"
   mostly mean "identical success count", not "genuinely similar
   performance". v2 roughly doubles the population (more achievable ANWG
   values -> finer resolution).
2. **Accidental bulk-tenant urgency.** v1's `BULK_SLACK_S=1.5` put the
   *median* bulk tenant's own laxity (~0.176s) below
   `KVConstrainedOnlinePolicy.urgent_laxity_seconds=0.25` -- bulk tenants
   were frequently bypassing the reserve gate via the policy's own urgent
   override, undermining the intended "deferrable background load, never
   itself urgent" role. v2 raises `BULK_SLACK_S` so the median bulk tenant
   is clearly non-urgent while remaining tight enough for the Round-2
   bidirectionality property (deferring bulk tenants still has a real cost)
   to hold -- verified empirically in the v2 design doc's calibration log,
   not assumed.

Also adds a third `urgent_arrival_phase` level (`middle`) for finer
trajectory resolution of the within-scenario timing hypothesis (H3).

Shares tenant classes, GPU-config helper, and BurstGPT provenance machinery
with v1 -- only the calibrated constants and the phase grid are new.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .builders import req, kv_scarce_gpu
from .schema import PolicySeparationScenario
from .templates_prefill_decode import (
    _load_burstgpt_arrays,
    _sample_lengths,
    resolve_burstgpt_path,
)
from .templates_fairness_starvation_v2 import BurstGPTUnavailableError
from .templates_kv_pressure import CLASS_BULK, CLASS_URGENT, ALLOWED_CLASS_IDS

GENERATOR_VERSION = "kv_pressure_v2"
FAMILY_NAME = "family_c_kv_pressure_v2"
TEMPLATE_NAME = "case_kv_pressure_reserve_contention_v2"

# --- Bulk tenant (KV-pressure driver) ---
# Population roughly doubled vs v1 (ANWG-resolution fix, see module
# docstring). Prompt window unchanged (already safely below any candidate
# max_kv_tokens; the v1 Round-3 infeasibility bug was in the window bound,
# not the population size).
N_BULK_V2 = {"low": 10, "high": 24}
BULK_PROMPT_MEDIAN = 2048
BULK_PROMPT_LO, BULK_PROMPT_HI = 1024, 3072
BULK_OUTPUT_MEDIAN = 300
BULK_OUTPUT_LO, BULK_OUTPUT_HI = 100, 600
BULK_ARRIVAL_DT = 0.05
# 1.65 is the v2-calibrated value (see design doc v2 calibration log):
# 2.0 (median bulk laxity=0.676s, safely non-urgent) fixed the accidental-
# urgency confound but made bulk tenants so safe under delay that
# bidirectionality collapsed (32-vs-1 wins at the larger v2 population).
# 1.65 keeps the median bulk tenant non-urgent (laxity=1.65-1.324=0.326s >
# urgent_laxity_seconds=0.25) while restoring genuine bulk-tenant risk
# under sustained deferral (empirically verified: 36-vs-6 wins, tie rate
# 12.5%, gate G8 "no universal dominant parent" satisfied).
BULK_SLACK_S = 1.65

# --- Urgent tenant (SLO-critical population) ---
# Population roughly doubled vs v1 (ANWG-resolution fix).
N_URGENT_V2 = 10  # fixed; not swept, to isolate the swept timing/tightness factors
URGENT_PROMPT_MEDIAN = 1024
URGENT_PROMPT_LO, URGENT_PROMPT_HI = 512, 2048
URGENT_OUTPUT_MEDIAN = 150
URGENT_OUTPUT_LO, URGENT_OUTPUT_HI = 50, 400
URGENT_ARRIVAL_DT = 0.03
URGENT_SLACK_S = {"loose": 3.0, "tight": 0.55}  # unchanged from v1
# Third "middle" level added for finer within-scenario timing resolution
# (H3): fraction of the bulk convoy's arrival span at which urgent tenants
# start arriving.
URGENT_ARRIVAL_PHASE_FRACTION = {"early": 0.0, "middle": 0.35, "late": 0.7}

# --- GPU: KV capacity is the binding constraint (see builders.kv_scarce_gpu) ---
DEFAULT_MAX_KV_TOKENS = 6_000  # unchanged from v1; re-verified in v2 calibration
DEFAULT_MAX_ACTIVE_SEQUENCES = 64
DEFAULT_MAX_BATCH_TOKENS = 64


def case_kv_pressure_reserve_contention_v2(
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
    """Build one Family C v2 cell. See module docstring for what changed
    from v1 and why."""
    if bulk_pressure not in N_BULK_V2:
        raise ValueError(f"bulk_pressure must be one of {sorted(N_BULK_V2)}, got {bulk_pressure!r}")
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
            "Family C v2 production mode requires staged BurstGPT "
            "(LLM_SERVEOPT_BURSTGPT_CSV or staged shards). Refusing silent "
            "synthetic fallback."
        )
    p_pool = pool[0] if pool is not None else None
    o_pool = pool[1] if pool is not None else None

    n_bulk_ = int(n_bulk) if n_bulk is not None else int(N_BULK_V2[bulk_pressure])
    n_urgent_ = int(n_urgent) if n_urgent is not None else N_URGENT_V2

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
        f"kvp2.bulk{n_bulk_}"
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
        "generator_version": GENERATOR_VERSION,
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
            "least_laxity_first is materially larger at urgent_arrival_phase "
            "middle/late than at early (calibration showed peak effect at "
            "middle, not necessarily late -- sustained mid-convoy pressure "
            "can exceed end-of-convoy pressure once earlier admissions have "
            "started completing), most strongly under urgent_tightness=tight, "
            "and this pattern replicates on held-out seeds not used in v2 "
            "calibration."
        ),
        stress_control_relationship=(
            "stress" if (bulk_pressure == "high" and urgent_arrival_phase == "late") else "control"
        ),
        pair_id=f"kvp2.tight{urgent_tightness}.s{seed}",
        changed_parameters=("bulk_pressure", "urgent_arrival_phase"),
    )


def assert_policy_visible_fields_clean_kv_v2(scenario: PolicySeparationScenario) -> None:
    """Anti-leakage guard for policy-visible Request fields (Family C v2)."""
    forbidden = (
        "bulk_pressure", "urgent_arrival_phase", "urgent_tightness",
        "phaseearly", "phasemiddle", "phaselate", "tighttight", "tightloose",
        "intended", "winner", "kv_constrained", "least_laxity",
        "scenario", "kvp2.",
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
