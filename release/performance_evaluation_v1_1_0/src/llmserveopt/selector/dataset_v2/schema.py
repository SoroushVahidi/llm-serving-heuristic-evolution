"""
Selector Dataset v2 schema: scenario/window x topology x policy, preserving
the FULL per-policy outcome vector (never reduced to a single winner label
at construction time). See docs/selector_dataset_v2.md.

Design rules this schema encodes (see that doc's §3 for the full
rationale):
  - Every row identifies exactly one (scenario, topology, resource
    configuration, policy) combination -- never aggregated across policies
    at write time.
  - `actual_output_tokens` never appears anywhere in this module or in any
    downstream feature; only `predicted_output_tokens` (this project's
    long-established non-oracle field) is ever read from a Request when
    building features.
  - Metrics genuinely unavailable for a given policy/topology are recorded
    as `None` plus an explicit availability flag -- never silently 0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ScenarioIdentifiers:
    """Required identifiers for a single scenario/window row (see
    docs/selector_dataset_v2.md §3)."""
    scenario_id: str
    scenario_family_id: str
    dataset_family: str          # "real_trace" | "real_distribution_synthetic" | "controlled_stress"
    source_trace: str            # e.g. "sharegpt", "burstgpt", "synthetic_poisson"
    seed: int
    topology_class: str
    resource_configuration_id: str
    window_id: int
    temporal_block_id: Optional[str] = None  # only meaningful for real-trace temporal slices
    request_plan_ancestor_id: Optional[str] = None
    scenario_pool: str = "REPRESENTATIVE_POOL"
    bottleneck_class: Optional[str] = None


@dataclass(frozen=True)
class PolicyOutcomeVector:
    """Full per-policy outcome for one scenario/window -- see
    docs/selector_dataset_v2.md §4. Every field is `Optional`; a `None`
    means genuinely unavailable/not-applicable for this policy or
    topology, NEVER a silently-substituted 0. `available_metrics` lists
    which of the non-identifier fields were actually populated, so a
    downstream consumer never has to guess whether a 0 is real."""
    policy_name: str
    fidelity_class: str           # "historical" | "faithful" | "paper_reimplementation"

    weighted_goodput: Optional[float] = None
    arrival_normalized_weighted_goodput: Optional[float] = None
    weighted_completion_fraction: Optional[float] = None
    completion_fraction: Optional[float] = None
    slo_violation_rate: Optional[float] = None
    slo_attainment: Optional[float] = None   # 1 - slo_violation_rate, kept explicit for clarity
    request_throughput: Optional[float] = None
    token_throughput: Optional[float] = None
    slo_success_throughput: Optional[float] = None

    mean_latency: Optional[float] = None
    median_latency: Optional[float] = None
    p95_latency: Optional[float] = None
    p99_latency: Optional[float] = None

    mean_ttft: Optional[float] = None
    p50_ttft: Optional[float] = None
    p95_ttft: Optional[float] = None
    p99_ttft: Optional[float] = None

    mean_tpot: Optional[float] = None
    p50_tpot: Optional[float] = None
    p95_tpot: Optional[float] = None
    p99_tpot: Optional[float] = None
    mean_tbt: Optional[float] = None
    p50_tbt: Optional[float] = None
    p95_tbt: Optional[float] = None
    p99_tbt: Optional[float] = None

    admission_rate: Optional[float] = None
    rejection_rate: Optional[float] = None
    rejection_fraction: Optional[float] = None

    num_total: Optional[int] = None
    num_completed: Optional[int] = None
    num_dropped: Optional[int] = None

    num_admit_events: Optional[int] = None
    num_preempt_events: Optional[int] = None
    num_swap_events: Optional[int] = None
    num_migrate_events: Optional[int] = None

    policy_decision_overhead_s: Optional[float] = None   # mean_policy_time_s
    simulation_wall_time_s: Optional[float] = None

    resource_gpu_count: Optional[int] = None

    prefill_gpu_utilization: Optional[float] = None
    decode_gpu_utilization: Optional[float] = None
    prefill_queue_mean: Optional[float] = None
    prefill_queue_p95: Optional[float] = None
    decode_queue_mean: Optional[float] = None
    decode_queue_p95: Optional[float] = None
    bridge_queue_mean: Optional[float] = None
    bridge_queue_p95: Optional[float] = None

    available_metrics: List[str] = field(default_factory=list)

    def to_row_dict(self, prefix: str) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("available_metrics")
        d.pop("policy_name")
        d.pop("fidelity_class")
        return {f"{prefix}_{k}": v for k, v in d.items()}


@dataclass(frozen=True)
class ObjectiveBest:
    """Best-policy-by-objective record, part of the multi-objective
    performance-target set (docs/selector_dataset_v2.md §5). `higher_is_better`
    records the comparison direction used so regret is never silently
    computed with the wrong sign for a lower-is-better metric (e.g. p95
    latency)."""
    objective_name: str
    higher_is_better: bool
    best_policy: str
    best_value: float


@dataclass(frozen=True)
class DiscriminativenessResult:
    """See docs/selector_dataset_v2.md §6. Computed per-objective (a
    window can be STRONGLY_DISCRIMINATIVE for weighted_goodput but
    NEAR_TIE for p95 latency)."""
    objective_name: str
    best_policy: str
    best_value: float
    second_best_policy: str
    second_best_value: float
    absolute_winner_margin: float
    relative_winner_margin: float   # margin / |best_value|, NaN-safe
    max_min_spread: float
    tie_set: List[str]              # policies within a small epsilon of best_value
    classification: str             # STRONGLY_DISCRIMINATIVE | MODERATELY_DISCRIMINATIVE | NEAR_TIE | ALL_COMPLETE_OR_EFFECTIVELY_TIED


@dataclass(frozen=True)
class RegretRecord:
    """regret(s, p) = score(best compatible policy for scenario s) - score(policy p),
    for one objective, one policy (docs/selector_dataset_v2.md §5)."""
    objective_name: str
    policy_name: str
    regret: float
    regret_to_best_fixed: float     # vs. the single policy best on AVERAGE across the whole dataset, not just this window


@dataclass(frozen=True)
class WindowRecordV2:
    """One fully-assembled Selector Dataset v2 unit: identifiers + causal
    features + the FULL per-policy outcome table + discriminativeness/regret
    for every objective. This is the in-memory unit `builder.py` produces;
    `to_flat_rows()` explodes it into one CSV row per (window, policy) pair
    -- the "scenario/window x topology x policy" unit the task requires,
    never collapsed to a single winner-label row."""
    identifiers: ScenarioIdentifiers
    features: Dict[str, Optional[float]]
    outcomes: List[PolicyOutcomeVector]
    discriminativeness: List[DiscriminativenessResult]
    regrets: List[RegretRecord]

    def to_flat_rows(self) -> List[Dict[str, Any]]:
        disc_by_objective = {d.objective_name: d for d in self.discriminativeness}
        regret_by_policy_objective = {(r.policy_name, r.objective_name): r for r in self.regrets}

        base: Dict[str, Any] = {**asdict(self.identifiers)}
        for fname, fval in self.features.items():
            base[f"feat_{fname}"] = fval

        rows: List[Dict[str, Any]] = []
        for outcome in self.outcomes:
            row = dict(base)
            row["policy_name"] = outcome.policy_name
            row["fidelity_class"] = outcome.fidelity_class
            row.update(outcome.to_row_dict(prefix="metric"))
            row["available_metrics"] = ",".join(outcome.available_metrics)
            for objective_name, disc in disc_by_objective.items():
                row[f"disc_{objective_name}_classification"] = disc.classification
                row[f"disc_{objective_name}_best_policy"] = disc.best_policy
                row[f"disc_{objective_name}_absolute_margin"] = disc.absolute_winner_margin
                row[f"disc_{objective_name}_relative_margin"] = disc.relative_winner_margin
                row[f"disc_{objective_name}_max_min_spread"] = disc.max_min_spread
                row[f"disc_{objective_name}_tie_set"] = "|".join(disc.tie_set)
                key = (outcome.policy_name, objective_name)
                if key in regret_by_policy_objective:
                    r = regret_by_policy_objective[key]
                    row[f"regret_{objective_name}"] = r.regret
                    row[f"regret_to_best_fixed_{objective_name}"] = r.regret_to_best_fixed
            rows.append(row)
        return rows


@dataclass
class DatasetManifestV2:
    """Machine-readable dataset-level manifest (docs/selector_dataset_v2.md §14)."""
    dataset_name: str
    schema_version: str
    topology_class: str
    candidate_policies: List[str]
    feature_names: List[str]
    objectives: List[str]
    num_scenarios: int
    num_windows: int
    num_policy_evaluations: int
    scenario_family_ids: List[str]
    source_traces: List[str]
    seeds: List[int]
    split_group_key: str
    split_counts: Dict[str, int]
    quality_gate_results: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
