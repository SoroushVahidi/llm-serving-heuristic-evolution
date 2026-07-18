"""
Scenario-family generators for Selector Dataset v2. See
docs/selector_dataset_v2.md §8.

Reuses this project's existing `workloads.synthetic.WorkloadConfig`/
`generate_workload` (real-distribution-preserving lognormal prompt/output
generation, seeded, with configurable arrival process, prediction noise,
and SLO-class mixture) and `workloads.burstgpt` (real trace ingestion) --
nothing here reimplements request generation from scratch.

Coverage-aware, not a Cartesian product: this module defines one
`ScenarioFamilySpec` per named family (documented parameter choices, not
arbitrary), and `all_scenario_family_specs()` returns the full curated
list actually used by the pilot builder -- adding a new family means
adding one spec, not multiplying every existing axis by every other axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from ...core.types import Request
from ...workloads.synthetic import SLOClass, WorkloadConfig, generate_workload


@dataclass(frozen=True)
class ScenarioFamilySpec:
    family_id: str
    dataset_family: str          # "controlled_stress" | "real_distribution_synthetic" | "real_trace"
    description: str
    build: Callable[[int], List[Request]]   # (seed) -> requests
    source_trace: str = "synthetic"
    temporal_block_id: Optional[str] = None


def _tight_slo_classes() -> List[SLOClass]:
    return [
        SLOClass("tight", slo_slack=0.3, priority=3.0, weight=0.7),
        SLOClass("medium", slo_slack=1.0, priority=2.0, weight=0.2),
        SLOClass("loose", slo_slack=5.0, priority=1.0, weight=0.1),
    ]


def _mixed_priority_classes() -> List[SLOClass]:
    return [
        SLOClass("critical", slo_slack=1.0, priority=10.0, weight=0.1),
        SLOClass("tight", slo_slack=0.5, priority=3.0, weight=0.2),
        SLOClass("medium", slo_slack=2.0, priority=2.0, weight=0.4),
        SLOClass("loose", slo_slack=10.0, priority=1.0, weight=0.3),
    ]


# ---------------------------------------------------------------------------
# Controlled stress families (docs/selector_dataset_v2.md §8.C) -- every
# parameter below is this project's own disclosed choice, not paper-sourced.
# ---------------------------------------------------------------------------

_CONTROLLED_STRESS_CONFIGS = {
    "low_load": WorkloadConfig(arrival_rate=4.0, duration=25.0, tag="low_load"),
    "moderate_load": WorkloadConfig(arrival_rate=12.0, duration=25.0, tag="moderate_load"),
    "near_saturation": WorkloadConfig(arrival_rate=22.0, duration=25.0, tag="near_saturation"),
    "overload": WorkloadConfig(arrival_rate=40.0, duration=20.0, tag="overload"),
    "burst_overload": WorkloadConfig(
        arrival_process="bursty", arrival_rate=15.0, duration=20.0,
        burst_factor=8.0, burst_fraction=0.25, tag="burst_overload",
    ),
    "kv_pressure": WorkloadConfig(
        arrival_rate=18.0, duration=20.0, prompt_mean=400.0, prompt_sigma=0.6,
        output_mean=300.0, output_sigma=0.6, tag="kv_pressure",
    ),
    "prefill_heavy": WorkloadConfig(
        arrival_rate=10.0, duration=20.0, prompt_mean=600.0, prompt_sigma=0.5,
        output_mean=20.0, output_sigma=0.4, tag="prefill_heavy",
    ),
    "decode_heavy": WorkloadConfig(
        arrival_rate=10.0, duration=20.0, prompt_mean=30.0, prompt_sigma=0.4,
        output_mean=500.0, output_sigma=0.5, tag="decode_heavy",
    ),
    "mixed_short_long": WorkloadConfig(
        arrival_rate=14.0, duration=25.0, prompt_dist="pareto", prompt_mean=150.0,
        output_dist="pareto", output_mean=150.0, tag="mixed_short_long",
    ),
    "extreme_prediction_noise": WorkloadConfig(
        arrival_rate=12.0, duration=20.0, prediction_noise_rel=0.75, tag="extreme_prediction_noise",
    ),
    "tight_slos": WorkloadConfig(
        arrival_rate=14.0, duration=20.0, slo_classes=_tight_slo_classes(), tag="tight_slos",
    ),
    "mixed_priorities": WorkloadConfig(
        arrival_rate=14.0, duration=20.0, slo_classes=_mixed_priority_classes(), tag="mixed_priorities",
    ),
}


def _controlled_stress_builder(config: WorkloadConfig) -> Callable[[int], List[Request]]:
    def _build(seed: int) -> List[Request]:
        return generate_workload(config, seed=seed)
    return _build


def _rapid_workload_shift_builder(seed: int) -> List[Request]:
    """Concatenates two contrasting regimes back-to-back (low load then
    burst overload) to produce a within-trace regime shift -- not covered
    by any single WorkloadConfig, since WorkloadConfig itself models one
    stationary regime for its whole duration."""
    first = generate_workload(
        WorkloadConfig(arrival_rate=4.0, duration=12.0, tag="shift_low"), seed=seed,
    )
    second_raw = generate_workload(
        WorkloadConfig(arrival_process="bursty", arrival_rate=30.0, duration=12.0,
                        burst_factor=6.0, burst_fraction=0.3, tag="shift_burst"), seed=seed + 1,
    )
    offset = first[-1].arrival_time if first else 0.0
    next_id = (first[-1].request_id + 1) if first else 0
    shifted_second = []
    for i, r in enumerate(second_raw):
        shifted_second.append(Request(
            request_id=next_id + i, arrival_time=r.arrival_time + offset + 0.01,
            prompt_tokens=r.prompt_tokens, predicted_output_tokens=r.predicted_output_tokens,
            actual_output_tokens=r.actual_output_tokens,
            slo_deadline=r.slo_deadline + offset + 0.01, priority=r.priority, class_id=r.class_id,
        ))
    return first + shifted_second


def all_scenario_family_specs() -> List[ScenarioFamilySpec]:
    specs = []
    for name, config in _CONTROLLED_STRESS_CONFIGS.items():
        specs.append(ScenarioFamilySpec(
            family_id=f"controlled_stress__{name}",
            dataset_family="controlled_stress",
            description=f"Controlled stress family '{name}': {config}",
            build=_controlled_stress_builder(config),
        ))
    specs.append(ScenarioFamilySpec(
        family_id="controlled_stress__rapid_workload_shift",
        dataset_family="controlled_stress",
        description="Two contrasting regimes (low load -> burst overload) concatenated to force a within-trace load shift.",
        build=_rapid_workload_shift_builder,
    ))
    # Real-distribution synthetic: this project's own existing "realistic"
    # presets (lognormal prompt/output, calibrated to look like ShareGPT-
    # style conversational traffic) -- see workloads/synthetic.py.
    specs.append(ScenarioFamilySpec(
        family_id="real_distribution_synthetic__medium",
        dataset_family="real_distribution_synthetic",
        description="workloads.synthetic default lognormal prompt/output distribution at moderate load.",
        build=_controlled_stress_builder(WorkloadConfig(arrival_rate=10.0, duration=25.0, tag="real_dist_medium")),
    ))
    specs.append(ScenarioFamilySpec(
        family_id="real_distribution_synthetic__heavy_tail",
        dataset_family="real_distribution_synthetic",
        description="Heavier-tailed (pareto) prompt/output lengths, moderate load.",
        build=_controlled_stress_builder(WorkloadConfig(
            arrival_rate=10.0, duration=25.0, prompt_dist="pareto", output_dist="pareto", tag="real_dist_heavy_tail",
        )),
    ))
    return specs


def load_burstgpt_real_trace_scenario(
    path: str, seed: int = 0, num_requests: Optional[int] = None,
) -> Optional[List[Request]]:
    """Real-trace family (docs/selector_dataset_v2.md §8.A): loads a slice
    of the locally-available BurstGPT CSV via the existing
    `workloads.burstgpt` conversion pipeline. Returns None if the file is
    not present -- callers must treat this as "real trace unavailable in
    this environment," never silently substitute synthetic data and label
    it as real."""
    from pathlib import Path

    from ...workloads.burstgpt import BurstGPTConversionConfig, load_burstgpt_trace

    if not Path(path).exists():
        return None
    config = BurstGPTConversionConfig() if num_requests is None else BurstGPTConversionConfig(max_requests=num_requests)
    requests, _report = load_burstgpt_trace(path, config=config, seed=seed)
    return requests
