"""
Selector Dataset v2 builder: orchestrates scenario generation, window
construction, causal feature extraction, unified (historical + external-
baseline) per-policy execution, and full-outcome-vector row assembly.
See docs/selector_dataset_v2.md.

Two-pass design
----------------
Pass 1 (`_run_scenario`): for each scenario, build windows, extract causal
(leakage-free) features, and run every candidate policy on every window,
producing a `PolicyOutcomeVector` list per window -- no regret-to-best-
fixed yet (that is a dataset-wide statistic).

Pass 2 (`build_selector_dataset_v2`): after every scenario has been run,
compute each objective's best-FIXED (single policy, best on average
across the WHOLE dataset) value, then compute per-window discriminativeness
and regret (including regret-to-best-fixed) for every objective, assign a
group-aware split, and assemble the final `WindowRecordV2` list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.metrics import RunMetrics
from ...core.types import GPUConfig, Request
from ...evaluation.run_policy import run_policy
from ...simulator.service_model import ServiceModel
from .candidates import fidelity_class_of, make_candidate_policy
from .discriminativeness import (
    STANDARD_OBJECTIVES,
    best_fixed_policy_and_value,
    compute_discriminativeness,
    compute_regrets,
)
from .schema import PolicyOutcomeVector, ScenarioIdentifiers, WindowRecordV2
from .scenario_families import ScenarioFamilySpec
from .features import extract_selector_v2_features
from .splits import assign_group_aware_split, split_for_group
from ..windows import make_windows  # re-exported below for convenience

__all__ = [
    "make_windows",
    "run_candidate_policy_on_window",
    "metrics_to_outcome_vector",
    "build_selector_dataset_v2",
    "build_selector_dataset_v2_trials",
]


def metrics_to_outcome_vector(
    policy_name: str, metrics: RunMetrics, event_counts: Dict[str, int], gpu_count: int,
) -> PolicyOutcomeVector:
    """Convert a RunMetrics + event-count dict into a PolicyOutcomeVector,
    tracking exactly which fields were populated (`available_metrics`) so
    a `None` downstream is never confused with an unpopulated 0."""
    def _nan_to_none(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                return None
        except TypeError:
            pass
        return v

    slo_violation_rate = _nan_to_none(metrics.slo_violation_rate)
    slo_attainment = (1.0 - slo_violation_rate) if slo_violation_rate is not None else None
    completion_fraction = _nan_to_none(metrics.completion_fraction)
    weighted_goodput = _nan_to_none(metrics.weighted_goodput)
    weighted_completion_fraction = _nan_to_none(metrics.weighted_completion_fraction)
    arrival_normalized_wg = _nan_to_none(metrics.arrival_normalized_weighted_goodput)
    if weighted_goodput is None and metrics.num_total > 0 and metrics.num_completed == 0:
        weighted_goodput = 0.0
    if completion_fraction is None and metrics.num_total > 0:
        completion_fraction = metrics.num_completed / metrics.num_total
    if weighted_completion_fraction is None and completion_fraction is not None:
        weighted_completion_fraction = completion_fraction
    if arrival_normalized_wg is None and weighted_completion_fraction is not None and weighted_goodput is not None:
        arrival_normalized_wg = weighted_completion_fraction * weighted_goodput
    request_throughput = _nan_to_none(metrics.request_throughput)
    slo_success_throughput = (
        request_throughput * arrival_normalized_wg
        if request_throughput is not None and arrival_normalized_wg is not None
        else None
    )
    rejection_fraction = (
        metrics.num_dropped / metrics.num_total
        if metrics.num_total else None
    )

    fields = {
        "weighted_goodput": weighted_goodput,
        "arrival_normalized_weighted_goodput": arrival_normalized_wg,
        "weighted_completion_fraction": weighted_completion_fraction,
        "completion_fraction": completion_fraction,
        "slo_violation_rate": slo_violation_rate,
        "slo_attainment": slo_attainment,
        "request_throughput": request_throughput,
        "token_throughput": _nan_to_none(metrics.token_throughput),
        "slo_success_throughput": slo_success_throughput,
        "mean_latency": _nan_to_none(metrics.mean_latency),
        "median_latency": _nan_to_none(metrics.median_latency),
        "p95_latency": _nan_to_none(metrics.p95_latency),
        "p99_latency": _nan_to_none(metrics.p99_latency),
        "mean_ttft": _nan_to_none(metrics.mean_ttft),
        "p50_ttft": None,
        "p95_ttft": _nan_to_none(metrics.p95_ttft),
        "p99_ttft": _nan_to_none(metrics.p99_ttft),
        "mean_tpot": _nan_to_none(metrics.mean_tpot),
        "p50_tpot": None,
        "p95_tpot": _nan_to_none(metrics.p95_tpot),
        "p99_tpot": None,
        "mean_tbt": _nan_to_none(metrics.mean_tpot),
        "p50_tbt": None,
        "p95_tbt": _nan_to_none(metrics.p95_tpot),
        "p99_tbt": None,
        "admission_rate": (
            event_counts.get("admit") / metrics.num_total
            if metrics.num_total else None
        ),
        "rejection_rate": (
            rejection_fraction
        ),
        "rejection_fraction": rejection_fraction,
        "num_total": metrics.num_total,
        "num_completed": metrics.num_completed,
        "num_dropped": metrics.num_dropped,
        "num_admit_events": event_counts.get("admit"),
        "num_preempt_events": event_counts.get("preempt"),
        "num_swap_events": event_counts.get("swap"),
        "num_migrate_events": event_counts.get("migrate"),
        "policy_decision_overhead_s": _nan_to_none(metrics.mean_policy_time_s),
        "simulation_wall_time_s": _nan_to_none(metrics.wall_clock_s),
        "resource_gpu_count": gpu_count,
        "prefill_gpu_utilization": None,
        "decode_gpu_utilization": None,
        "prefill_queue_mean": None,
        "prefill_queue_p95": None,
        "decode_queue_mean": None,
        "decode_queue_p95": None,
        "bridge_queue_mean": None,
        "bridge_queue_p95": None,
    }
    available = [k for k, v in fields.items() if v is not None]
    return PolicyOutcomeVector(
        policy_name=policy_name, fidelity_class=fidelity_class_of(policy_name),
        available_metrics=available, **fields,
    )


def run_candidate_policy_on_window(
    policy_name: str, requests: Sequence[Request], gpu_configs: List[GPUConfig],
    service_model: Optional[ServiceModel], workload_tag: str, seed: int, drain_steps: int,
) -> PolicyOutcomeVector:
    """Unified runner for BOTH historical (registry.py) and external
    (external_baselines_registry.py) candidate policies -- resolved
    transparently via candidates.make_candidate_policy. Wraps
    select_action in a non-invasive counting closure (same technique as
    evaluation/external_baseline_harness.py) to populate admit/preempt/
    swap/migrate event counts for every candidate uniformly."""
    policy = make_candidate_policy(policy_name, seed=seed)
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
        policy=policy, requests=requests, gpu_configs=gpu_configs, service_model=service_model,
        workload_tag=workload_tag, seed=seed, drain_steps=drain_steps,
    )
    return metrics_to_outcome_vector(policy_name, metrics, counts, gpu_count=len(gpu_configs))


@dataclass
class _RawWindow:
    identifiers: ScenarioIdentifiers
    features: Dict[str, Optional[float]]
    outcomes: List[PolicyOutcomeVector]


def _run_scenario(
    spec: ScenarioFamilySpec, seed: int, gpu_configs: List[GPUConfig],
    service_model: Optional[ServiceModel], candidate_policies: List[str],
    window_size: int, drain_steps: int, topology_class: str,
) -> List[_RawWindow]:
    requests = spec.build(seed)
    if not requests:
        return []
    effective_gpus = spec.effective_gpu_configs(gpu_configs)
    effective_service_model = spec.effective_service_model(service_model)
    scenario_id = f"{spec.family_id}__seed{seed}"
    service_part = ""
    if effective_service_model is not None:
        service_part = (
            f"_prefill{int(effective_service_model.enable_prefill_modeling)}"
            f"_tb{effective_service_model.step_token_budget}"
        )
    resource_configuration_id = (
        f"gpus{len(effective_gpus)}"
        f"_seq{sum(g.max_active_sequences for g in effective_gpus)}"
        f"_kv{sum(g.max_kv_tokens for g in effective_gpus)}"
        f"{service_part}"
    )

    windows = make_windows(requests, trace_id=scenario_id, window_size=window_size)
    reqs_list = list(requests)
    raw_windows: List[_RawWindow] = []

    for w in windows:
        prefix = reqs_list[:w.start_request_index]
        feats = extract_selector_v2_features(
            window_requests=w.requests, window_start_time=w.start_time,
            prefix_requests=prefix, gpu_configs=effective_gpus,
            topology_class=topology_class,
            step_token_budget=(
                effective_service_model.step_token_budget
                if effective_service_model is not None else None
            ),
        )

        outcomes = [
            run_candidate_policy_on_window(
                pname, w.requests, effective_gpus, effective_service_model,
                workload_tag=f"{scenario_id}_w{w.window_id}", seed=seed, drain_steps=drain_steps,
            )
            for pname in candidate_policies
        ]
        identifiers = ScenarioIdentifiers(
            scenario_id=scenario_id, scenario_family_id=spec.family_id, dataset_family=spec.dataset_family,
            source_trace=spec.source_trace, seed=seed, topology_class=topology_class,
            resource_configuration_id=resource_configuration_id, window_id=w.window_id,
            temporal_block_id=spec.temporal_block_id,
            request_plan_ancestor_id=spec.request_plan_ancestor_id or spec.family_id,
            scenario_pool=spec.scenario_pool,
            bottleneck_class=spec.bottleneck_class,
        )
        raw_windows.append(_RawWindow(identifiers=identifiers, features=feats, outcomes=outcomes))

    return raw_windows


def build_selector_dataset_v2(
    scenario_specs: List[ScenarioFamilySpec], seeds: List[int], gpu_configs: List[GPUConfig],
    candidate_policies: List[str], service_model: Optional[ServiceModel] = None,
    window_size: int = 30, drain_steps: int = 20_000, topology_class: str = "monolithic",
    verbose: bool = False,
) -> List[WindowRecordV2]:
    """Build the full Selector Dataset v2 for one topology class. Returns
    a list of `WindowRecordV2`, each carrying the FULL per-policy outcome
    vector (never reduced to a winner label), per-objective
    discriminativeness, and per-objective regret (including regret to the
    dataset-wide best-fixed policy). Split assignment is a SEPARATE step
    (see `assemble_dataset_rows`) applied when flattening to rows, so the
    same built records can be re-split under different train/val
    fractions without re-running any simulation."""
    trials = [(spec, seed) for spec in scenario_specs for seed in seeds]
    return build_selector_dataset_v2_trials(
        trials=trials,
        gpu_configs=gpu_configs,
        candidate_policies=candidate_policies,
        service_model=service_model,
        window_size=window_size,
        drain_steps=drain_steps,
        topology_class=topology_class,
        verbose=verbose,
    )


def build_selector_dataset_v2_trials(
    trials: List[Tuple[ScenarioFamilySpec, int]], gpu_configs: List[GPUConfig],
    candidate_policies: List[str], service_model: Optional[ServiceModel] = None,
    window_size: int = 30, drain_steps: int = 20_000, topology_class: str = "monolithic",
    verbose: bool = False,
) -> List[WindowRecordV2]:
    """Build v2 records for an explicit list of (scenario spec, seed) trials."""
    all_raw_windows: List[_RawWindow] = []
    for spec, seed in trials:
        if verbose:
            print(f"  running {spec.family_id} seed={seed} ...")
        all_raw_windows.extend(_run_scenario(
            spec, seed, gpu_configs, service_model, candidate_policies,
            window_size, drain_steps, topology_class,
        ))

    # Pass 2: dataset-wide best-fixed-policy value per objective.
    all_outcomes_lists = [rw.outcomes for rw in all_raw_windows]
    best_fixed_values: Dict[str, float] = {}
    for objective in STANDARD_OBJECTIVES:
        result = best_fixed_policy_and_value(all_outcomes_lists, objective)
        if result is not None:
            best_fixed_values[objective.name] = result[1]

    records: List[WindowRecordV2] = []
    for rw in all_raw_windows:
        discs = []
        regrets = []
        for objective in STANDARD_OBJECTIVES:
            disc = compute_discriminativeness(rw.outcomes, objective)
            if disc is not None:
                discs.append(disc)
            regrets.extend(compute_regrets(
                rw.outcomes, objective, best_fixed_policy_value=best_fixed_values.get(objective.name),
            ))
        records.append(WindowRecordV2(
            identifiers=rw.identifiers, features=rw.features, outcomes=rw.outcomes,
            discriminativeness=discs, regrets=regrets,
        ))

    return records


def assemble_dataset_rows(
    records: List[WindowRecordV2], group_key_field: str = "scenario_family_id",
    ood_scenario_family_ids: Optional[set] = None, train_frac: float = 0.6, val_frac: float = 0.2,
) -> List[Dict]:
    """Flatten `records` to one row per (window, policy) pair (via
    `WindowRecordV2.to_flat_rows`) and attach a group-aware `split` column
    -- the group-aware split assignment is computed ONCE across every
    record's `scenario_family_id`, then applied uniformly, guaranteeing
    group atomicity (see splits.verify_group_atomicity)."""
    rows: List[Dict] = []
    for record in records:
        rows.extend(record.to_flat_rows())

    family_ids = sorted({row[group_key_field] for row in rows})
    split_assignment = assign_group_aware_split(
        family_ids, ood_group_keys=ood_scenario_family_ids, train_frac=train_frac, val_frac=val_frac,
    )
    for row in rows:
        row["split"] = split_for_group(row[group_key_field], split_assignment)
    return rows
