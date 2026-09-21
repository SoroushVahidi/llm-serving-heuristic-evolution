"""
Unified evaluation harness for the five external baselines (see
src/llmserveopt/policies/external_baselines_registry.py,
src/llmserveopt/evaluation/external_baseline_configs.py, and
docs/external_baseline_integration.md).

Responsibilities (see docs/external_baseline_integration.md §6):
  - validate topology before execution, failing clearly and early on any
    config that violates a baseline's own structural requirements
    (min GPU count, required roles, distserve_faithful's exact 1+1),
  - run the SAME workload trace across baselines when scientifically
    appropriate (the caller supplies one `requests` sequence used for
    every baseline it evaluates),
  - record resource allocation (TopologyDescription, reconstructed
    directly from the gpu_configs actually used -- never trusted
    separately from what was really run),
  - record baseline provenance (pinned source, reference doc, fidelity
    class) and every constructor parameter used,
  - track policy-level events (admit/preempt/swap/migrate counts) via a
    counting wrapper around select_action, without modifying any policy,
  - preserve deterministic seeds,
  - produce a JSON-serializable result (`ExternalBaselineRunResult.to_dict()`).

This module does NOT force every baseline through a shared abstraction
that changes its algorithm -- each policy's own `select_action` runs
completely unmodified; only its RETURN VALUE (the Action) is inspected.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.metrics import RunMetrics
from ..core.types import GPUConfig, Request
from ..policies.external_baselines_registry import (
    ExternalBaselineSpec,
    TopologyClass,
    get_external_baseline_spec,
)
from ..simulator.service_model import ServiceModel
from .external_baseline_configs import TopologyDescription
from .run_policy import run_policy


class TopologyValidationError(ValueError):
    """Raised when a gpu_configs list does not satisfy a baseline's own
    structural requirements -- always raised BEFORE the simulator runs,
    never discovered mid-run."""


def describe_topology(gpu_configs: Sequence[GPUConfig], topology_class: Optional[TopologyClass] = None) -> TopologyDescription:
    """Reconstruct a TopologyDescription directly from the gpu_configs
    actually being used -- never trust a caller-supplied description that
    might not match what is really run.

    `topology_class` disambiguates the one case gpu_configs alone cannot
    (role=None GPUs are used by BOTH MONOLITHIC and
    MULTI_INSTANCE_MIGRATORY -- the difference is in the POLICY's own
    resource-sharing semantics, not anything visible on GPUConfig itself).
    Callers with a target baseline in hand (run_external_baseline) should
    always pass it; omitted only for ad-hoc/exploratory use."""
    prefill = [g for g in gpu_configs if g.role == "prefill"]
    decode = [g for g in gpu_configs if g.role == "decode"]
    none_role = [g for g in gpu_configs if g.role is None]

    if prefill or decode:
        return TopologyDescription(
            topology_class=TopologyClass.DISAGGREGATED_PREFILL_DECODE.value,
            total_gpus=len(gpu_configs),
            num_prefill_gpus=len(prefill), num_decode_gpus=len(decode),
            total_kv_tokens=sum(g.max_kv_tokens for g in gpu_configs),
            prefill_kv_tokens=sum(g.max_kv_tokens for g in prefill),
            decode_kv_tokens=sum(g.max_kv_tokens for g in decode),
        )
    resolved_class = topology_class.value if topology_class is not None else "monolithic_or_multi_instance"
    return TopologyDescription(
        topology_class=resolved_class,
        total_gpus=len(gpu_configs), num_instances=len(none_role),
        total_kv_tokens=sum(g.max_kv_tokens for g in gpu_configs),
    )


def validate_topology(spec: ExternalBaselineSpec, gpu_configs: Sequence[GPUConfig]) -> None:
    """Raise TopologyValidationError with a clear, specific message if
    `gpu_configs` does not satisfy `spec`'s structural requirements.
    Never silently proceeds with an incompatible config."""
    prefill = [g for g in gpu_configs if g.role == "prefill"]
    decode = [g for g in gpu_configs if g.role == "decode"]
    none_role = [g for g in gpu_configs if g.role is None]

    if spec.topology_class == TopologyClass.DISAGGREGATED_PREFILL_DECODE:
        if not prefill or not decode:
            raise TopologyValidationError(
                f"{spec.name} requires at least one role='prefill' and one role='decode' "
                f"GPU; got {len(prefill)} prefill, {len(decode)} decode, {len(none_role)} role=None."
            )
        if spec.min_role_counts is not None:
            min_p, min_d = spec.min_role_counts
            if len(prefill) < min_p or len(decode) < min_d:
                raise TopologyValidationError(
                    f"{spec.name} requires >= {min_p} prefill and >= {min_d} decode GPUs; "
                    f"got {len(prefill)} prefill, {len(decode)} decode."
                )
        if spec.name == "distserve_faithful" and (len(prefill) != 1 or len(decode) != 1):
            raise TopologyValidationError(
                f"distserve_faithful requires EXACTLY 1 prefill + 1 decode GPU "
                f"(its own hard single-worker-per-stage constraint); got "
                f"{len(prefill)} prefill, {len(decode)} decode."
            )
    else:
        if prefill or decode:
            raise TopologyValidationError(
                f"{spec.name} uses only role=None GPUs (topology_class={spec.topology_class.value}); "
                f"got {len(prefill)} role='prefill' and {len(decode)} role='decode' GPUs."
            )
        if len(none_role) < spec.min_gpu_count:
            raise TopologyValidationError(
                f"{spec.name} requires >= {spec.min_gpu_count} GPU(s); got {len(none_role)}."
            )


@dataclass
class ExternalBaselineRunResult:
    baseline_name: str
    fidelity_class: str
    topology_class: str
    pinned_source: str
    reference_doc: str
    policy_params: Dict[str, Any]
    seed: int
    workload_tag: str
    topology: TopologyDescription
    metrics: RunMetrics
    num_admit_events: int
    num_preempt_events: int
    num_swap_events: int
    num_migrate_events: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # RunMetrics/TopologyDescription are already plain dataclasses;
        # asdict() recurses into them automatically.
        return d


def run_external_baseline(
    name: str,
    requests: Sequence[Request],
    gpu_configs: Sequence[GPUConfig],
    service_model: Optional[ServiceModel] = None,
    workload_tag: str = "unknown",
    seed: int = 0,
    drain_steps: int = 50_000,
    policy_kwargs: Optional[Dict[str, Any]] = None,
) -> ExternalBaselineRunResult:
    spec = get_external_baseline_spec(name)
    validate_topology(spec, gpu_configs)
    policy_kwargs = dict(policy_kwargs or {})
    policy = spec.factory(**policy_kwargs)

    counts = {"admit": 0, "preempt": 0, "swap": 0, "migrate": 0}
    orig_select_action = policy.select_action

    def counting_select_action(state):
        action = orig_select_action(state)
        counts["admit"] += sum(len(v) for v in action.admit.values())
        counts["preempt"] += sum(len(v) for v in action.preempt.values())
        counts["swap"] += sum(len(v) for v in action.swap.values())
        counts["migrate"] += sum(len(v) for v in action.migrate.values())
        return action

    policy.select_action = counting_select_action

    metrics = run_policy(
        policy=policy, requests=requests, gpu_configs=list(gpu_configs),
        service_model=service_model, workload_tag=workload_tag, seed=seed,
        drain_steps=drain_steps,
    )

    notes: List[str] = []
    if spec.topology_class == TopologyClass.DISAGGREGATED_PREFILL_DECODE:
        notes.append(
            "mean_gpu_utilization/mean_active_batch_size average across BOTH "
            "prefill and decode roles; not separately reported per role."
        )
    if spec.preemption_mode.value == "admission_avoidance":
        notes.append(
            "num_preempt_events/num_swap_events are always 0 for this baseline "
            "by design (admission-time avoidance, never runtime eviction) -- "
            "this is a genuine 0, not a missing/unobserved metric."
        )
    notes.append("KV utilization (per-GPU occupancy over time) is not tracked by this harness pass; see docs/external_baseline_integration.md §7.")

    return ExternalBaselineRunResult(
        baseline_name=name,
        fidelity_class=spec.fidelity_class.value,
        topology_class=spec.topology_class.value,
        pinned_source=spec.pinned_source,
        reference_doc=spec.reference_doc,
        policy_params=policy_kwargs,
        seed=seed,
        workload_tag=workload_tag,
        topology=describe_topology(gpu_configs, topology_class=spec.topology_class),
        metrics=metrics,
        num_admit_events=counts["admit"],
        num_preempt_events=counts["preempt"],
        num_swap_events=counts["swap"],
        num_migrate_events=counts["migrate"],
        notes=notes,
    )


def results_to_json(results: Sequence[ExternalBaselineRunResult], indent: int = 2) -> str:
    return json.dumps([r.to_dict() for r in results], indent=indent, default=str)
