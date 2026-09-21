"""Public Replay Load Scaling v1 -- preregistered arrival-rate scaling over
the frozen 60-window public trace replay corpus.

Addresses the external-validity criticism that `public_trace_replay_v1`
(see `docs/current/public_trace_replay_v1_analysis_20260820.md`) is
effectively unloaded (p99 active requests ~5/512, max KV utilization
~0.0038) and therefore cannot expose scheduler differences.

This module implements a SINGLE, ISOLATED manipulation: the inter-arrival
timing of each of the 60 canonical augmented-view windows is compressed by a
preregistered load factor lambda, drawn from a fixed geometric grid. Nothing
else about the workload changes:

  - request ordering is preserved (the transform is a strictly increasing
    affine map of arrival_time, so relative order is invariant by
    construction)
  - prompt_tokens / actual_output_tokens / predicted_output_tokens are
    untouched
  - class_id / priority are untouched
  - GPU capacity (max_active_sequences, max_batch_tokens, max_kv_tokens) is
    UNCHANGED from the base replay (512 / 512 / 8,000,000) -- this is the
    key difference from the prior `public_trace_stress_v1` protocol (see
    `docs/current/external_baseline_stress_protocol_20260824.md`), which
    confounded arrival compression (M) with a simultaneous capacity cut (C).
    That prior protocol is a valid, separately-frozen artifact; it answers a
    different question (find a stress point) and is not reused or
    superseded by this experiment.
  - the deadline-arrival slack (slo_deadline - arrival_time) of each request
    is preserved in absolute seconds -- i.e. slo_deadline moves together
    with arrival_time so that per-request SLO tightness, a non-arrival
    workload attribute, is not accidentally changed by the arrival-timing
    manipulation. This is the same convention already frozen and used by
    `run_public_trace_stress_p6.py` / `run_public_trace_stress_external.py`.

Transform (frozen, matches docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md):

    t'_i = t_start + (t_i - t_start) / lambda

where t_start is the minimum arrival_time within that window (0.0 for every
canonical window in this corpus, but computed generically).

Policy portfolio (Pext, frozen; see docs/current/external_baseline_fidelity_ledger_20260824.md
Pass-4 "Final common Pext definition"):

    P6 = full_prefill, chunked_prefill_small, estimated_service_time_first,
         weighted_fair_share, least_laxity_first, kv_constrained_online
    + official_vtc_joint_token_budget_remap  (frozen, OFFICIAL_CODE_ADAPTED)
    + vllm_style_continuous_batching          (frozen simulator PROXY, not
                                                native vLLM)

SOLA and vLLM-LTR are excluded (not in Pext_common; see fidelity ledger
Pass-3/Pass-4) -- this experiment does not implement any new external
scheduler.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ..core.metrics import metrics_to_dict  # noqa: E402
from ..core.types import GPUConfig  # noqa: E402
from ..policies.base import BasePolicy  # noqa: E402
from ..simulator.service_model import ServiceModel  # noqa: E402
from ..simulator.simulator import Simulator, SimulatorConfig  # noqa: E402
from . import public_trace_replay_v1 as ptr  # noqa: E402
from .schema import PolicySeparationScenario  # noqa: E402
from .unified_utility_matrix import _build_policy as _build_p6_policy  # noqa: E402

BUILDER_VERSION = "public_replay_load_scaling_v1.0.0"
WORKLOAD_ID = "public_replay_load_scaling_v1"

# -- frozen preregistered constants (docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md) --
LOAD_FACTORS: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)

P6_POLICIES: Tuple[str, ...] = (
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
)
EXTERNAL_POLICIES: Tuple[str, ...] = (
    "official_vtc_joint_token_budget_remap",
    "vllm_style_continuous_batching",
)
PEXT_POLICIES: Tuple[str, ...] = P6_POLICIES + EXTERNAL_POLICIES
assert len(PEXT_POLICIES) == 8

EXPECTED_N_WINDOWS = 60
EXPECTED_N_WINDOWS_PER_SOURCE = 20
EXPECTED_N_LOAD_FACTORS = 8
EXPECTED_N_POLICIES = 8
EXPECTED_N_CELLS = EXPECTED_N_WINDOWS * EXPECTED_N_LOAD_FACTORS * EXPECTED_N_POLICIES  # 3840

SIM_MAX_STEPS = 200_000
SIM_DRAIN_STEPS = 50_000


# ---------------------------------------------------------------------------
# Canonical window selection (reuses public_trace_replay_v1's frozen Layer-2
# augmented-view scenarios verbatim -- no new window sampling here)
# ---------------------------------------------------------------------------

def get_canonical_windows() -> List[Dict[str, Any]]:
    """The 60 canonical augmented-view (controlled-annotation) windows from
    `public_trace_replay_v1`, sorted deterministically by canonical_scenario_id.

    Each record has the same shape as `public_trace_replay_v1.build_all_scenarios()`
    records restricted to `scenario_evidence_class == AUGMENTED`.
    """
    all_records = ptr.build_all_scenarios()
    aug = [r for r in all_records if r["scenario_evidence_class"] == ptr.AUGMENTED]
    aug.sort(key=lambda r: r["canonical_scenario_id"])
    if len(aug) != EXPECTED_N_WINDOWS:
        raise AssertionError(f"expected {EXPECTED_N_WINDOWS} augmented windows, got {len(aug)}")
    per_source: Dict[str, int] = {}
    for r in aug:
        per_source[r["source_dataset"]] = per_source.get(r["source_dataset"], 0) + 1
    for src, n in per_source.items():
        if n != EXPECTED_N_WINDOWS_PER_SOURCE:
            raise AssertionError(
                f"expected {EXPECTED_N_WINDOWS_PER_SOURCE} windows for source {src!r}, got {n}"
            )
    if set(per_source) != set(ptr.SOURCES):
        raise AssertionError(f"expected sources {ptr.SOURCES}, got {sorted(per_source)}")
    return aug


# ---------------------------------------------------------------------------
# Arrival-only scaling transform
# ---------------------------------------------------------------------------

def transform_arrival_only(scenario: PolicySeparationScenario, lam: float) -> PolicySeparationScenario:
    """Scale ONLY inter-arrival timing by load factor `lam`. GPU capacity,
    token lengths, class/priority, and per-request slack are preserved.

    t'_i = t_start + (t_i - t_start) / lam,  slack_i = deadline_i - t_i preserved.
    """
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    reqs = list(scenario.requests)
    if not reqs:
        raise ValueError("scenario has no requests")
    t_start = min(float(r.arrival_time) for r in reqs)
    new_reqs = []
    for r in reqs:
        arr = t_start + (float(r.arrival_time) - t_start) / float(lam)
        slack = max(0.0, float(r.slo_deadline) - float(r.arrival_time))
        new_reqs.append(replace(r, arrival_time=arr, slo_deadline=arr + slack))
    new_reqs.sort(key=lambda x: (x.arrival_time, x.request_id))

    # Capacity is explicitly UNCHANGED (identity passthrough) -- this is the
    # defining difference from public_trace_stress_v1's M x C protocol.
    gpu_configs = scenario.gpu_configs

    params = dict(scenario.params)
    params["load_scaling_lambda"] = float(lam)
    params["load_scaling_t_start"] = float(t_start)

    lam_int = int(lam) if float(lam).is_integer() else lam
    return PolicySeparationScenario(
        scenario_id=f"{scenario.scenario_id}__load_scaling_lambda{lam_int}",
        family=scenario.family,
        template_name=scenario.template_name,
        generator_version=scenario.generator_version,
        seed=scenario.seed,
        params=params,
        requests=tuple(new_reqs),
        gpu_configs=gpu_configs,
        service_model_kwargs=dict(scenario.service_model_kwargs),
        target_policy_family=scenario.target_policy_family,
        target_mechanism=WORKLOAD_ID,
        expected_qualitative_hypothesis=scenario.expected_qualitative_hypothesis,
        stress_control_relationship="control" if float(lam) == 1.0 else "stress",
    )


# ---------------------------------------------------------------------------
# Policy construction (reuses frozen builders; no new scheduler implemented)
# ---------------------------------------------------------------------------

def build_pext_policy(policy_id: str, scenario: PolicySeparationScenario):
    """Return (policy_instance, service_model_kwargs_override)."""
    if policy_id in P6_POLICIES:
        return _build_p6_policy(policy_id)
    if policy_id == "vllm_style_continuous_batching":
        from ..policies.vllm_faithful import VLLMFaithfulPolicy  # noqa: E402

        return VLLMFaithfulPolicy(), {"allow_chunked_prefill": False, "decode_first": True}
    if policy_id == "official_vtc_joint_token_budget_remap":
        from baselines.vtc.adapter.simulator_policy import VTCFairnessPolicy  # noqa: E402

        tenants = sorted({r.class_id for r in scenario.requests})
        max_prompt = max(int(r.prompt_tokens) for r in scenario.requests)
        step_budget = int(scenario.service_model_kwargs.get("step_token_budget", 512))
        budget = max(step_budget, max_prompt)
        return VTCFairnessPolicy(known_tenants=tenants, batch_token_budget_override=budget), {}
    raise KeyError(f"Unknown Pext policy_id {policy_id!r}")


# ---------------------------------------------------------------------------
# Pressure telemetry -- lightweight per-step sampling wrapper. Delegates
# every scheduling decision unchanged to the wrapped policy; only observes.
# ---------------------------------------------------------------------------

class _TelemetryWrapPolicy(BasePolicy):
    """Wraps an arbitrary policy, recording per-step system-pressure samples
    without altering any scheduling decision."""

    def __init__(self, inner: BasePolicy) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "wrapped")
        self.samples: List[Dict[str, float]] = []

    def reset(self) -> None:
        self.samples = []
        self.inner.reset()

    def select_action(self, state):
        gpu = state.gpu_states[0]
        self.samples.append(
            {
                "waiting": float(len(state.waiting_queue)),
                "active": float(len(gpu.active_request_ids)),
                "kv_util": float(gpu.current_kv_tokens) / max(float(gpu.max_kv_tokens), 1.0),
                "active_util": float(len(gpu.active_request_ids)) / max(float(gpu.max_active_sequences), 1.0),
            }
        )
        return self.inner.select_action(state)


def summarize_pressure(samples: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not samples:
        return {
            "n_steps": 0,
            "queue_length_mean": 0.0,
            "queue_length_p95": 0.0,
            "queue_length_p99": 0.0,
            "queue_length_max": 0.0,
            "frac_steps_queue_positive": 0.0,
            "active_mean": 0.0,
            "active_p95": 0.0,
            "active_p99": 0.0,
            "active_max": 0.0,
            "active_util_max": 0.0,
            "active_util_p95": 0.0,
            "active_util_p99": 0.0,
            "kv_utilization_mean": 0.0,
            "kv_utilization_p95": 0.0,
            "kv_utilization_p99": 0.0,
            "kv_utilization_max": 0.0,
        }
    wait = np.array([s["waiting"] for s in samples], dtype=float)
    act = np.array([s["active"] for s in samples], dtype=float)
    kv = np.array([s["kv_util"] for s in samples], dtype=float)
    act_u = np.array([s["active_util"] for s in samples], dtype=float)
    return {
        "n_steps": int(len(samples)),
        "queue_length_mean": float(np.mean(wait)),
        "queue_length_p95": float(np.quantile(wait, 0.95)),
        "queue_length_p99": float(np.quantile(wait, 0.99)),
        "queue_length_max": float(np.max(wait)),
        "frac_steps_queue_positive": float(np.mean(wait > 0)),
        "active_mean": float(np.mean(act)),
        "active_p95": float(np.quantile(act, 0.95)),
        "active_p99": float(np.quantile(act, 0.99)),
        "active_max": float(np.max(act)),
        "active_util_max": float(np.max(act_u)),
        "active_util_p95": float(np.quantile(act_u, 0.95)),
        "active_util_p99": float(np.quantile(act_u, 0.99)),
        "kv_utilization_mean": float(np.mean(kv)),
        "kv_utilization_p95": float(np.quantile(kv, 0.95)),
        "kv_utilization_p99": float(np.quantile(kv, 0.99)),
        "kv_utilization_max": float(np.max(kv)),
    }


# ---------------------------------------------------------------------------
# Single-cell evaluation
# ---------------------------------------------------------------------------

def evaluate_cell(record: Dict[str, Any], lam: float, policy_id: str) -> Dict[str, Any]:
    """Evaluate one (window, lambda, policy) cell. Never raises: failures are
    captured in the returned row's status/error fields."""
    base_scenario = record["scenario"]
    row: Dict[str, Any] = {
        "workload_id": WORKLOAD_ID,
        "source_dataset": record["source_dataset"],
        "window_index": record["window_index"],
        "canonical_scenario_id": record["canonical_scenario_id"],
        "load_factor_lambda": float(lam),
        "policy_id": policy_id,
        "n_requests_base": len(base_scenario.requests),
    }
    try:
        s = transform_arrival_only(base_scenario, lam)
        row["scenario_id"] = s.scenario_id
        row["seed"] = int(s.seed)
        row["n_requests_scaled"] = len(s.requests)
        row["gpu_max_active_sequences"] = s.gpu_configs[0].max_active_sequences
        row["gpu_max_batch_tokens"] = s.gpu_configs[0].max_batch_tokens
        row["gpu_max_kv_tokens"] = s.gpu_configs[0].max_kv_tokens

        policy, sm_override = build_pext_policy(policy_id, s)
        sm = dict(s.service_model_kwargs)
        sm.update(sm_override)
        wrapped = _TelemetryWrapPolicy(policy)

        sim = Simulator(
            SimulatorConfig(
                gpu_configs=list(s.gpu_configs),
                service_model=ServiceModel(**sm),
                max_steps=SIM_MAX_STEPS,
                drain_steps=SIM_DRAIN_STEPS,
            )
        )
        sim.load_trace(list(s.requests))
        metrics = sim.run(wrapped, workload_tag=s.scenario_id, seed=s.seed)
        md = metrics_to_dict(metrics)

        row.update(
            {
                "status": "success",
                "error": "",
                "anwg": float(metrics.arrival_normalized_weighted_goodput),
                "weighted_goodput": md.get("weighted_goodput"),
                "completion_fraction": float(metrics.completion_fraction),
                "weighted_completion_fraction": md.get("weighted_completion_fraction"),
                "slo_violation_rate": md.get("slo_violation_rate"),
                "num_completed": int(md.get("num_completed") or 0),
                "num_dropped": int(md.get("num_dropped") or 0),
                "num_total": int(md.get("num_total") or 0),
                "mean_ttft": md.get("mean_ttft"),
                "p95_ttft": md.get("p95_ttft"),
                "p99_ttft": md.get("p99_ttft"),
                "mean_latency": md.get("mean_latency"),
                "p95_latency": md.get("p95_latency"),
                "p99_latency": md.get("p99_latency"),
                "request_throughput": md.get("request_throughput"),
                "token_throughput": md.get("token_throughput"),
                "sim_duration": md.get("sim_duration"),
            }
        )
        row.update(summarize_pressure(wrapped.samples))
    except Exception as e:  # noqa: BLE001
        import traceback

        row.update(
            {
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "anwg": None,
            }
        )
    return row


def cell_key(canonical_scenario_id: str, lam: float, policy_id: str) -> str:
    lam_int = int(lam) if float(lam).is_integer() else lam
    return f"{canonical_scenario_id}::lambda{lam_int}::{policy_id}"


def expected_cell_keys() -> List[str]:
    windows = get_canonical_windows()
    keys = []
    for r in windows:
        for lam in LOAD_FACTORS:
            for pid in PEXT_POLICIES:
                keys.append(cell_key(r["canonical_scenario_id"], lam, pid))
    if len(keys) != EXPECTED_N_CELLS:
        raise AssertionError(f"expected {EXPECTED_N_CELLS} cells, got {len(keys)}")
    if len(set(keys)) != len(keys):
        raise AssertionError("duplicate cell keys detected")
    return keys
