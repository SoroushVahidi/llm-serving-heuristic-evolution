"""Bottleneck-oriented scenario families for Selector Dataset v2 redesign."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from ...core.types import GPUConfig, Request
from ...simulator.service_model import ServiceModel
from ...workloads.synthetic import SLOClass, WorkloadConfig, generate_workload
from ...workloads.trace_io_extended import load_extended_jsonl
from .scenario_families import ScenarioFamilySpec


REPRESENTATIVE_POOL = "REPRESENTATIVE_POOL"
DISCRIMINATIVE_POOL = "DISCRIMINATIVE_POOL"


def scarcity_gpu(
    *,
    max_active_sequences: int = 8,
    max_batch_tokens: int = 8,
    max_kv_tokens: int = 6000,
) -> tuple[GPUConfig, ...]:
    return (
        GPUConfig(
            gpu_id=0,
            max_active_sequences=max_active_sequences,
            max_batch_tokens=max_batch_tokens,
            max_kv_tokens=max_kv_tokens,
        ),
    )


def service_model(
    *,
    prefill: bool = False,
    step_token_budget: int = 4096,
    max_prefill_chunk_tokens: int = 512,
    decode_first: bool = False,
) -> ServiceModel:
    return ServiceModel(
        enable_prefill_modeling=prefill,
        step_token_budget=step_token_budget,
        max_prefill_chunk_tokens=max_prefill_chunk_tokens,
        decode_first=decode_first,
    )


def slo_classes(
    tight_fraction: float,
    high_priority_fraction: float = 0.2,
    tight_slack: float = 0.08,
    medium_slack: float = 0.35,
    loose_slack: float = 1.5,
) -> list[SLOClass]:
    high = min(max(high_priority_fraction, 0.0), tight_fraction)
    tight = max(tight_fraction - high, 0.0)
    loose = max(1.0 - high - tight, 0.0)
    return [
        SLOClass("critical", slo_slack=tight_slack, priority=10.0, weight=high),
        SLOClass("tight", slo_slack=tight_slack, priority=3.0, weight=tight),
        SLOClass("medium", slo_slack=medium_slack, priority=2.0, weight=0.5 * loose),
        SLOClass("loose", slo_slack=loose_slack, priority=1.0, weight=0.5 * loose),
    ]


def transform_requests(
    requests: Sequence[Request],
    *,
    time_scale: float = 1.0,
    slo_scale: float = 1.0,
    prediction_noise_rel: float = 0.0,
    prediction_bias: float = 1.0,
    burst_amplification: float = 1.0,
    seed: int = 0,
) -> list[Request]:
    """Deterministically stress a trace while preserving provenance.

    ``time_scale < 1`` compresses arrivals. ``burst_amplification`` compresses
    the middle third of inter-arrivals more than the edges, creating a transient
    overload without changing request order.
    """
    rng = np.random.default_rng(seed)
    reqs = list(requests)
    if not reqs:
        return []
    arrivals = np.array([r.arrival_time for r in reqs], dtype=float)
    arrivals = arrivals - arrivals[0]
    if len(arrivals) > 1:
        gaps = np.diff(arrivals) * time_scale
        if burst_amplification > 1.0:
            lo = len(gaps) // 3
            hi = 2 * len(gaps) // 3
            gaps[lo:hi] = gaps[lo:hi] / burst_amplification
        arrivals = np.concatenate([[0.0], np.cumsum(gaps)])

    out: list[Request] = []
    for i, r in enumerate(reqs):
        true_out = r.actual_output_tokens
        if prediction_noise_rel > 0.0:
            noise = rng.lognormal(mean=0.0, sigma=prediction_noise_rel)
        else:
            noise = 1.0
        pred = max(1, int(round(true_out * prediction_bias * noise)))
        original_slack = max(r.slo_deadline - r.arrival_time, 0.001)
        slack = max(0.001, original_slack * slo_scale)
        out.append(Request(
            request_id=i,
            arrival_time=float(arrivals[i]),
            prompt_tokens=r.prompt_tokens,
            predicted_output_tokens=pred,
            actual_output_tokens=r.actual_output_tokens,
            slo_deadline=float(arrivals[i] + slack),
            priority=r.priority,
            class_id=r.class_id,
        ))
    return out


def _synthetic_spec(
    *,
    family_id: str,
    bottleneck_class: str,
    config: WorkloadConfig,
    gpu: tuple[GPUConfig, ...],
    service: ServiceModel,
    pool: str = DISCRIMINATIVE_POOL,
    ancestor_id: Optional[str] = None,
) -> ScenarioFamilySpec:
    def _build(seed: int, _config=config) -> list[Request]:
        return generate_workload(_config, seed=seed)

    return ScenarioFamilySpec(
        family_id=family_id,
        dataset_family="controlled_stress",
        source_trace="synthetic",
        request_plan_ancestor_id=ancestor_id or family_id,
        scenario_pool=pool,
        bottleneck_class=bottleneck_class,
        description=f"{bottleneck_class}: {_config_summary(config)}",
        build=_build,
        gpu_configs=gpu,
        service_model=service,
    )


def _config_summary(config: WorkloadConfig) -> str:
    return (
        f"arrival={config.arrival_rate}, duration={config.duration}, "
        f"arrival_process={config.arrival_process}, prompt={config.prompt_dist}/"
        f"{config.prompt_mean}, output={config.output_dist}/{config.output_mean}, "
        f"noise={config.prediction_noise_rel}"
    )


def bottleneck_taxonomy_specs() -> list[ScenarioFamilySpec]:
    """Hand-authored bottleneck taxonomy covering regimes A-H."""
    specs: list[ScenarioFamilySpec] = []
    specs.extend([
        _synthetic_spec(
            family_id="admission_pressure__tight_overload",
            bottleneck_class="admission_pressure",
            config=WorkloadConfig(
                arrival_rate=260.0, duration=1.2, prompt_mean=96.0,
                output_mean=80.0, prediction_noise_rel=0.2,
                slo_classes=slo_classes(0.75, 0.25, tight_slack=0.045, medium_slack=0.18),
                tag="admission_pressure_tight_overload",
            ),
            gpu=scarcity_gpu(max_active_sequences=8, max_batch_tokens=8, max_kv_tokens=7000),
            service=service_model(),
        ),
        _synthetic_spec(
            family_id="admission_pressure__priority_mixed",
            bottleneck_class="admission_pressure",
            config=WorkloadConfig(
                arrival_rate=210.0, duration=1.5, prompt_mean=128.0,
                output_mean=96.0, prediction_noise_rel=0.25,
                slo_classes=slo_classes(0.55, 0.35, tight_slack=0.06, medium_slack=0.22),
                tag="admission_pressure_priority_mixed",
            ),
            gpu=scarcity_gpu(max_active_sequences=10, max_batch_tokens=10, max_kv_tokens=8500),
            service=service_model(),
        ),
    ])
    specs.extend([
        _synthetic_spec(
            family_id="kv_pressure__long_decode_low_kv",
            bottleneck_class="kv_pressure",
            config=WorkloadConfig(
                arrival_rate=95.0, duration=2.2, prompt_mean=80.0,
                output_dist="pareto", output_mean=520.0, output_high=1800,
                prediction_noise_rel=0.15, slo_classes=slo_classes(0.45, 0.2, tight_slack=0.3, medium_slack=0.8),
                tag="kv_pressure_long_decode_low_kv",
            ),
            gpu=scarcity_gpu(max_active_sequences=10, max_batch_tokens=10, max_kv_tokens=2600),
            service=service_model(),
        ),
        _synthetic_spec(
            family_id="kv_pressure__high_variance_outputs",
            bottleneck_class="kv_pressure",
            config=WorkloadConfig(
                arrival_rate=120.0, duration=1.8, prompt_mean=64.0,
                output_dist="pareto", output_mean=360.0, output_high=2200,
                prediction_noise_rel=0.1, slo_classes=slo_classes(0.35, 0.15, tight_slack=0.25, medium_slack=0.7),
                tag="kv_pressure_high_variance_outputs",
            ),
            gpu=scarcity_gpu(max_active_sequences=12, max_batch_tokens=12, max_kv_tokens=2200),
            service=service_model(),
        ),
    ])
    specs.extend([
        _synthetic_spec(
            family_id="prefill_heavy__chunking_advantage",
            bottleneck_class="prefill_heavy",
            config=WorkloadConfig(
                arrival_rate=75.0, duration=1.6, prompt_mean=2400.0,
                prompt_sigma=0.55, prompt_high=6000, output_mean=18.0,
                output_sigma=0.35, output_high=80, prediction_noise_rel=0.1,
                slo_classes=slo_classes(0.45, 0.2, tight_slack=0.10, medium_slack=0.32),
                tag="prefill_heavy_chunking_advantage",
            ),
            gpu=scarcity_gpu(max_active_sequences=12, max_batch_tokens=12, max_kv_tokens=30000),
            service=service_model(prefill=True, step_token_budget=256, max_prefill_chunk_tokens=512, decode_first=True),
        ),
        _synthetic_spec(
            family_id="prefill_heavy__token_budget_constrained",
            bottleneck_class="prefill_heavy",
            config=WorkloadConfig(
                arrival_rate=95.0, duration=1.4, prompt_mean=1800.0,
                prompt_sigma=0.65, prompt_high=5000, output_mean=24.0,
                output_sigma=0.4, output_high=96, prediction_noise_rel=0.15,
                slo_classes=slo_classes(0.6, 0.25, tight_slack=0.08, medium_slack=0.28),
                tag="prefill_heavy_token_budget_constrained",
            ),
            gpu=scarcity_gpu(max_active_sequences=10, max_batch_tokens=10, max_kv_tokens=24000),
            service=service_model(prefill=True, step_token_budget=192, max_prefill_chunk_tokens=512, decode_first=True),
        ),
    ])
    specs.extend([
        _synthetic_spec(
            family_id="decode_heavy__long_running_occupancy",
            bottleneck_class="decode_heavy",
            config=WorkloadConfig(
                arrival_rate=105.0, duration=2.0, prompt_mean=32.0,
                output_dist="pareto", output_mean=700.0, output_high=2500,
                prediction_noise_rel=0.2, slo_classes=slo_classes(0.35, 0.15, tight_slack=0.4, medium_slack=1.0),
                tag="decode_heavy_long_running_occupancy",
            ),
            gpu=scarcity_gpu(max_active_sequences=8, max_batch_tokens=8, max_kv_tokens=5000),
            service=service_model(),
        ),
        _synthetic_spec(
            family_id="decode_heavy__mixed_short_long",
            bottleneck_class="decode_heavy",
            config=WorkloadConfig(
                arrival_rate=125.0, duration=1.8, prompt_mean=40.0,
                output_dist="pareto", output_mean=420.0, output_high=2200,
                prediction_noise_rel=0.15, slo_classes=slo_classes(0.5, 0.2, tight_slack=0.25, medium_slack=0.8),
                tag="decode_heavy_mixed_short_long",
            ),
            gpu=scarcity_gpu(max_active_sequences=10, max_batch_tokens=10, max_kv_tokens=4200),
            service=service_model(),
        ),
    ])
    specs.extend([
        _synthetic_spec(
            family_id="slo_heterogeneous__deadline_mix",
            bottleneck_class="slo_heterogeneous",
            config=WorkloadConfig(
                arrival_rate=180.0, duration=1.6, prompt_mean=96.0,
                output_mean=110.0, prediction_noise_rel=0.2,
                slo_classes=slo_classes(0.65, 0.15, tight_slack=0.06, medium_slack=0.35, loose_slack=2.0),
                tag="slo_heterogeneous_deadline_mix",
            ),
            gpu=scarcity_gpu(max_active_sequences=10, max_batch_tokens=10, max_kv_tokens=8000),
            service=service_model(),
        ),
        _synthetic_spec(
            family_id="prediction_noise__biased_underestimate",
            bottleneck_class="prediction_noise",
            config=WorkloadConfig(
                arrival_rate=145.0, duration=1.8, prompt_mean=100.0,
                output_dist="pareto", output_mean=260.0, output_high=1500,
                prediction_noise_rel=0.85, slo_classes=slo_classes(0.45, 0.2, tight_slack=0.18, medium_slack=0.65),
                tag="prediction_noise_biased_underestimate",
            ),
            gpu=scarcity_gpu(max_active_sequences=10, max_batch_tokens=10, max_kv_tokens=5200),
            service=service_model(),
        ),
        _synthetic_spec(
            family_id="bursty_transient__shock_recovery",
            bottleneck_class="bursty_transient",
            config=WorkloadConfig(
                arrival_process="bursty", arrival_rate=95.0, duration=2.6,
                burst_factor=12.0, burst_fraction=0.18, prompt_mean=96.0,
                output_mean=120.0, prediction_noise_rel=0.2,
                slo_classes=slo_classes(0.55, 0.2, tight_slack=0.08, medium_slack=0.35),
                tag="bursty_transient_shock_recovery",
            ),
            gpu=scarcity_gpu(max_active_sequences=10, max_batch_tokens=10, max_kv_tokens=7600),
            service=service_model(),
        ),
        _synthetic_spec(
            family_id="resource_scarcity__sequence_cap",
            bottleneck_class="resource_scarcity",
            config=WorkloadConfig(
                arrival_rate=160.0, duration=1.6, prompt_mean=90.0,
                output_mean=110.0, prediction_noise_rel=0.15,
                slo_classes=slo_classes(0.45, 0.2, tight_slack=0.08, medium_slack=0.35),
                tag="resource_scarcity_sequence_cap",
            ),
            gpu=scarcity_gpu(max_active_sequences=4, max_batch_tokens=4, max_kv_tokens=9000),
            service=service_model(),
        ),
    ])
    return specs


def sampled_bottleneck_specs(seed: int, count: int = 48) -> list[ScenarioFamilySpec]:
    """Deterministic random-search candidates around the taxonomy."""
    rng = np.random.default_rng(seed)
    bottlenecks = [
        "admission_pressure",
        "kv_pressure",
        "prefill_heavy",
        "decode_heavy",
        "slo_heterogeneous",
        "prediction_noise",
        "bursty_transient",
        "resource_scarcity",
    ]
    specs: list[ScenarioFamilySpec] = []
    for idx in range(count):
        b = bottlenecks[idx % len(bottlenecks)]
        arrival = float(rng.uniform(80.0, 280.0))
        tight = float(rng.uniform(0.25, 0.85))
        high_pri = float(rng.uniform(0.05, min(0.45, tight)))
        prompt_mean = float(rng.choice([48.0, 96.0, 160.0, 512.0, 1400.0, 2400.0]))
        output_mean = float(rng.choice([32.0, 96.0, 180.0, 420.0, 750.0]))
        prefill = b == "prefill_heavy" or prompt_mean >= 1000.0
        prompt_high = 6000 if prefill else 2048
        output_high = 2600 if output_mean >= 420 else 1024
        output_dist = "pareto" if b in {"kv_pressure", "decode_heavy", "prediction_noise"} else "lognormal"
        arrival_process = "bursty" if b == "bursty_transient" else "poisson"
        seq_cap = int(rng.choice([4, 6, 8, 10, 12]))
        kv_cap = int(rng.choice([2200, 3200, 5200, 8000, 16000, 28000]))
        if prefill:
            kv_cap = max(kv_cap, int(prompt_mean * 8))
        token_budget = int(rng.choice([128, 192, 256, 384, 512]))
        config = WorkloadConfig(
            arrival_process=arrival_process,
            arrival_rate=arrival,
            duration=float(rng.uniform(1.2, 2.4)),
            prompt_mean=prompt_mean,
            prompt_sigma=float(rng.uniform(0.35, 0.8)),
            prompt_high=prompt_high,
            output_dist=output_dist,
            output_mean=output_mean,
            output_sigma=float(rng.uniform(0.35, 0.9)),
            output_high=output_high,
            prediction_noise_rel=float(rng.choice([0.0, 0.15, 0.35, 0.7])),
            burst_factor=float(rng.uniform(6.0, 14.0)),
            burst_fraction=float(rng.uniform(0.12, 0.28)),
            slo_classes=slo_classes(
                tight,
                high_pri,
                tight_slack=float(rng.uniform(0.035, 0.18)),
                medium_slack=float(rng.uniform(0.18, 0.8)),
                loose_slack=float(rng.uniform(1.2, 3.0)),
            ),
            tag=f"sampled_{b}_{idx:03d}",
        )
        specs.append(_synthetic_spec(
            family_id=f"adaptive_sample__{b}__{idx:03d}",
            bottleneck_class=b,
            config=config,
            gpu=scarcity_gpu(
                max_active_sequences=seq_cap,
                max_batch_tokens=seq_cap,
                max_kv_tokens=kv_cap,
            ),
            service=service_model(
                prefill=prefill,
                step_token_budget=token_budget if prefill else 4096,
                max_prefill_chunk_tokens=512,
                decode_first=prefill,
            ),
            ancestor_id=f"adaptive_sample__{b}",
        ))
    return specs


def targeted_counterexample_specs(seed: int, count_per_target: int = 36) -> list[ScenarioFamilySpec]:
    """Deterministic counterexample candidates for underrepresented policies.

    These are still ordinary workload/resource configurations. The policy
    names in the family ids describe the *hypothesized* specialization region
    being searched, not a forced label or retention rule.
    """
    rng = np.random.default_rng(seed)
    specs: list[ScenarioFamilySpec] = []
    specs.extend(_sarathi_counterexample_specs(rng, count_per_target))
    specs.extend(_vllm_counterexample_specs(rng, count_per_target))
    specs.extend(_deadline_counterexample_specs(rng, count_per_target))
    return specs


def _sarathi_counterexample_specs(
    rng: np.random.Generator,
    count: int,
) -> list[ScenarioFamilySpec]:
    specs: list[ScenarioFamilySpec] = []
    for idx in range(count):
        arrival = float(rng.uniform(18.0, 95.0))
        prompt_mean = float(rng.choice([900.0, 1400.0, 2200.0, 3200.0, 4200.0]))
        output_mean = float(rng.choice([24.0, 48.0, 96.0, 160.0]))
        token_budget = int(rng.choice([64, 96, 128, 192, 256, 384]))
        chunk = int(rng.choice([128, 256, 512, 768]))
        seq_cap = int(rng.choice([8, 10, 12, 16, 20]))
        # Keep KV realistic for long prompts; scarcity is in the per-step
        # prefill token budget, not impossible memory.
        kv_cap = int(max(prompt_mean * rng.uniform(8.0, 18.0), 18_000))
        tight = float(rng.uniform(0.10, 0.55))
        config = WorkloadConfig(
            arrival_process="poisson",
            arrival_rate=arrival,
            duration=float(rng.uniform(2.4, 4.4)),
            prompt_mean=prompt_mean,
            prompt_sigma=float(rng.uniform(0.35, 0.75)),
            prompt_high=int(max(4096, prompt_mean * 2.4)),
            output_mean=output_mean,
            output_sigma=float(rng.uniform(0.35, 0.85)),
            output_high=int(max(256, output_mean * 5)),
            prediction_noise_rel=float(rng.choice([0.0, 0.08, 0.18])),
            slo_classes=slo_classes(
                tight,
                high_priority_fraction=float(rng.uniform(0.02, min(0.18, tight))),
                tight_slack=float(rng.uniform(0.22, 0.75)),
                medium_slack=float(rng.uniform(1.0, 2.5)),
                loose_slack=float(rng.uniform(4.0, 8.0)),
            ),
            tag=f"counterexample_sarathi_{idx:03d}",
        )
        specs.append(_synthetic_spec(
            family_id=f"counterexample__sarathi_faithful__{idx:03d}",
            bottleneck_class="prefill_heavy",
            config=config,
            gpu=scarcity_gpu(
                max_active_sequences=seq_cap,
                max_batch_tokens=seq_cap,
                max_kv_tokens=kv_cap,
            ),
            service=service_model(
                prefill=True,
                step_token_budget=token_budget,
                max_prefill_chunk_tokens=chunk,
                decode_first=True,
            ),
            ancestor_id="counterexample__sarathi_faithful",
        ))
    return specs


def _vllm_counterexample_specs(
    rng: np.random.Generator,
    count: int,
) -> list[ScenarioFamilySpec]:
    specs: list[ScenarioFamilySpec] = []
    for idx in range(count):
        output_mean = float(rng.choice([180.0, 320.0, 520.0, 760.0, 1050.0]))
        prompt_mean = float(rng.choice([24.0, 48.0, 96.0, 160.0]))
        seq_cap = int(rng.choice([12, 16, 20, 24, 32]))
        kv_cap = int(rng.choice([3600, 5200, 7600, 10_000, 14_000]))
        arrival = float(rng.uniform(45.0, 155.0))
        config = WorkloadConfig(
            arrival_process="poisson",
            arrival_rate=arrival,
            duration=float(rng.uniform(2.0, 4.0)),
            prompt_mean=prompt_mean,
            prompt_sigma=float(rng.uniform(0.25, 0.55)),
            prompt_high=512,
            output_dist="pareto",
            output_mean=output_mean,
            output_sigma=float(rng.uniform(0.55, 1.0)),
            output_high=int(max(1800, output_mean * 4)),
            prediction_noise_rel=float(rng.choice([0.0, 0.1, 0.25])),
            slo_classes=slo_classes(
                tight_fraction=float(rng.uniform(0.05, 0.35)),
                high_priority_fraction=float(rng.uniform(0.0, 0.08)),
                tight_slack=float(rng.uniform(0.55, 1.4)),
                medium_slack=float(rng.uniform(2.0, 5.0)),
                loose_slack=float(rng.uniform(8.0, 16.0)),
            ),
            tag=f"counterexample_vllm_{idx:03d}",
        )
        specs.append(_synthetic_spec(
            family_id=f"counterexample__vllm_faithful__{idx:03d}",
            bottleneck_class="decode_heavy",
            config=config,
            gpu=scarcity_gpu(
                max_active_sequences=seq_cap,
                max_batch_tokens=seq_cap,
                max_kv_tokens=kv_cap,
            ),
            service=service_model(prefill=False),
            ancestor_id="counterexample__vllm_faithful",
        ))
    return specs


def _deadline_counterexample_specs(
    rng: np.random.Generator,
    count: int,
) -> list[ScenarioFamilySpec]:
    specs: list[ScenarioFamilySpec] = []
    for idx in range(count):
        target = str(rng.choice([
            "edf",
            "slo_slack_score",
            "weighted_shortest_processing",
            "estimated_service_time_first",
            "admission_control",
        ]))
        prompt_mean = float(rng.choice([48.0, 96.0, 180.0, 320.0]))
        output_mean = float(rng.choice([48.0, 110.0, 220.0, 360.0]))
        arrival = float(rng.uniform(70.0, 185.0))
        config = WorkloadConfig(
            arrival_process="bursty" if idx % 4 == 0 else "poisson",
            arrival_rate=arrival,
            duration=float(rng.uniform(2.0, 3.6)),
            burst_factor=float(rng.uniform(3.0, 8.0)),
            burst_fraction=float(rng.uniform(0.08, 0.18)),
            prompt_mean=prompt_mean,
            prompt_sigma=float(rng.uniform(0.35, 0.75)),
            prompt_high=2048,
            output_dist="pareto" if output_mean >= 220 else "lognormal",
            output_mean=output_mean,
            output_sigma=float(rng.uniform(0.35, 0.9)),
            output_high=int(max(512, output_mean * 5)),
            prediction_noise_rel=float(rng.choice([0.0, 0.1, 0.3, 0.55])),
            slo_classes=slo_classes(
                tight_fraction=float(rng.uniform(0.25, 0.65)),
                high_priority_fraction=float(rng.uniform(0.0, 0.20)),
                tight_slack=float(rng.uniform(0.16, 0.55)),
                medium_slack=float(rng.uniform(0.8, 2.2)),
                loose_slack=float(rng.uniform(4.0, 9.0)),
            ),
            tag=f"counterexample_deadline_{target}_{idx:03d}",
        )
        seq_cap = int(rng.choice([8, 10, 12, 16]))
        specs.append(_synthetic_spec(
            family_id=f"counterexample__deadline_policy__{target}__{idx:03d}",
            bottleneck_class="slo_heterogeneous",
            config=config,
            gpu=scarcity_gpu(
                max_active_sequences=seq_cap,
                max_batch_tokens=seq_cap,
                max_kv_tokens=int(rng.choice([5500, 7500, 10_000, 14_000])),
            ),
            service=service_model(prefill=False),
            ancestor_id=f"counterexample__deadline_policy__{target}",
        ))
    return specs


def local_real_trace_stress_specs(
    root: Path,
    *,
    max_requests: int = 144,
) -> list[ScenarioFamilySpec]:
    candidates = [
        ("burstgpt_scaled_moderate", "burstgpt", "data/processed/burstgpt/burstgpt_scaled_moderate_10k.jsonl"),
        ("burstgpt_scaled_high", "burstgpt", "data/processed/burstgpt/burstgpt_scaled_high_10k.jsonl"),
        ("azure_2023_code", "azure_llm_2023", "data/processed/azure/azure_llm_2023_code.jsonl"),
        ("azure_2023_conv", "azure_llm_2023", "data/processed/azure/azure_llm_2023_conv.jsonl"),
    ]
    specs: list[ScenarioFamilySpec] = []
    for base_name, source, rel_path in candidates:
        path = root / rel_path
        if not path.exists():
            continue
        specs.extend(_real_trace_variants(path, base_name, source, max_requests))
    return specs


def _real_trace_variants(
    path: Path,
    base_name: str,
    source: str,
    max_requests: int,
) -> list[ScenarioFamilySpec]:
    transforms = [
        ("representative", REPRESENTATIVE_POOL, "real_trace_representative", 1.0, 1.0, 0.0, 1.0, 1.0,
         scarcity_gpu(max_active_sequences=12, max_batch_tokens=12, max_kv_tokens=12000), service_model()),
        ("compressed_tight", DISCRIMINATIVE_POOL, "admission_pressure", 0.08, 0.25, 0.2, 1.0, 1.0,
         scarcity_gpu(max_active_sequences=8, max_batch_tokens=8, max_kv_tokens=7000), service_model()),
        ("burst_kv", DISCRIMINATIVE_POOL, "kv_pressure", 0.12, 0.45, 0.25, 1.0, 5.0,
         scarcity_gpu(max_active_sequences=8, max_batch_tokens=8, max_kv_tokens=3500), service_model()),
        ("noise_underpredict", DISCRIMINATIVE_POOL, "prediction_noise", 0.12, 0.45, 0.7, 0.55, 2.5,
         scarcity_gpu(max_active_sequences=8, max_batch_tokens=8, max_kv_tokens=5000), service_model()),
    ]

    def _load_slice(seed: int) -> list[Request]:
        reqs, _metadata = load_extended_jsonl(path)
        if len(reqs) <= max_requests:
            return reqs
        start = (seed * 7919) % (len(reqs) - max_requests + 1)
        return reqs[start:start + max_requests]

    specs: list[ScenarioFamilySpec] = []
    for name, pool, bottleneck, time_scale, slo_scale, noise, bias, burst_amp, gpu, svc in transforms:
        def _build(
            seed: int,
            _time_scale=time_scale,
            _slo_scale=slo_scale,
            _noise=noise,
            _bias=bias,
            _burst_amp=burst_amp,
        ) -> list[Request]:
            return transform_requests(
                _load_slice(seed),
                time_scale=_time_scale,
                slo_scale=_slo_scale,
                prediction_noise_rel=_noise,
                prediction_bias=_bias,
                burst_amplification=_burst_amp,
                seed=seed,
            )

        specs.append(ScenarioFamilySpec(
            family_id=f"real_trace_stress__{base_name}__{name}",
            dataset_family="real_trace",
            source_trace=source,
            temporal_block_id=name,
            request_plan_ancestor_id=f"real_trace__{base_name}",
            scenario_pool=pool,
            bottleneck_class=bottleneck,
            description=f"Real trace {base_name} transform {name} from {path}",
            build=_build,
            gpu_configs=gpu,
            service_model=svc,
        ))
    return specs


def representative_easy_specs() -> list[ScenarioFamilySpec]:
    return [
        _synthetic_spec(
            family_id="representative_easy__moderate_load",
            bottleneck_class="representative_easy",
            pool=REPRESENTATIVE_POOL,
            config=WorkloadConfig(
                arrival_rate=45.0, duration=2.0, prompt_mean=128.0,
                output_mean=80.0, prediction_noise_rel=0.15,
                slo_classes=slo_classes(0.25, 0.05, tight_slack=0.3, medium_slack=1.0, loose_slack=3.0),
                tag="representative_easy_moderate_load",
            ),
            gpu=scarcity_gpu(max_active_sequences=16, max_batch_tokens=16, max_kv_tokens=16000),
            service=service_model(),
        )
    ]


def with_pool(spec: ScenarioFamilySpec, pool: str) -> ScenarioFamilySpec:
    return replace(spec, scenario_pool=pool)
