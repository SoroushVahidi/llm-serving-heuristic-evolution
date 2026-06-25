"""
Multi-regime evaluation: run evaluate_candidates across multiple workload regimes
and aggregate results with train/validation split.

Train regimes are used for candidate selection. Validation regimes check
generalization. Test regimes are held out and NOT used here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..workloads.synthetic import WorkloadConfig, SLOClass
from .evaluation import CandidateResult, EvaluationConfig, evaluate_candidates


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

@dataclass
class RegimeSpec:
    name: str
    split: str               # "train" | "validation" | "test"
    workload: WorkloadConfig
    # Optional override for gpu config
    max_active_sequences: int = 4
    max_batch_tokens: int = 512
    max_kv_tokens: int = 8192
    seed: int = 42


def _tight_slo_classes():
    return [
        SLOClass("tight",  slo_slack=0.3,  priority=3.0, weight=0.3),
        SLOClass("medium", slo_slack=1.5,  priority=2.0, weight=0.5),
        SLOClass("loose",  slo_slack=8.0,  priority=1.0, weight=0.2),
    ]


def _mixed_slo_classes():
    return [
        SLOClass("tight",  slo_slack=0.5,  priority=3.0, weight=0.2),
        SLOClass("medium", slo_slack=2.0,  priority=2.0, weight=0.5),
        SLOClass("loose",  slo_slack=10.0, priority=1.0, weight=0.3),
    ]


TRAIN_REGIMES: List[RegimeSpec] = [
    RegimeSpec(
        name="train_poisson_moderate",
        split="train",
        workload=WorkloadConfig(
            arrival_process="poisson", arrival_rate=15.0, duration=60.0,
            prompt_mean=256.0, output_mean=128.0,
            prediction_noise_rel=0.15,
            tag="train_poisson_moderate",
        ),
        seed=42,
    ),
    RegimeSpec(
        name="train_bursty_moderate",
        split="train",
        workload=WorkloadConfig(
            arrival_process="bursty", arrival_rate=15.0, duration=60.0,
            prompt_mean=256.0, output_mean=128.0,
            burst_factor=4.0, burst_fraction=0.2,
            prediction_noise_rel=0.15,
            tag="train_bursty_moderate",
        ),
        seed=43,
    ),
    RegimeSpec(
        name="train_overloaded",
        split="train",
        workload=WorkloadConfig(
            arrival_process="poisson", arrival_rate=25.0, duration=60.0,
            prompt_mean=256.0, output_mean=128.0,
            prediction_noise_rel=0.15,
            slo_classes=_tight_slo_classes(),
            tag="train_overloaded",
        ),
        seed=44,
    ),
    RegimeSpec(
        name="train_mixed_slo",
        split="train",
        workload=WorkloadConfig(
            arrival_process="poisson", arrival_rate=15.0, duration=60.0,
            prompt_mean=256.0, output_mean=128.0,
            prediction_noise_rel=0.15,
            slo_classes=_mixed_slo_classes(),
            tag="train_mixed_slo",
        ),
        seed=45,
    ),
]

VALIDATION_REGIMES: List[RegimeSpec] = [
    RegimeSpec(
        name="val_prefill_heavy",
        split="validation",
        workload=WorkloadConfig(
            arrival_process="poisson", arrival_rate=10.0, duration=60.0,
            prompt_mean=512.0, output_mean=64.0,
            prompt_sigma=0.6, output_sigma=0.6,
            prediction_noise_rel=0.15,
            tag="val_prefill_heavy",
        ),
        seed=100,
    ),
    RegimeSpec(
        name="val_decode_heavy",
        split="validation",
        workload=WorkloadConfig(
            arrival_process="poisson", arrival_rate=10.0, duration=60.0,
            prompt_mean=128.0, output_mean=384.0,
            prompt_sigma=0.6, output_sigma=0.8,
            prediction_noise_rel=0.15,
            tag="val_decode_heavy",
        ),
        seed=101,
    ),
    RegimeSpec(
        name="val_noisy_predictions",
        split="validation",
        workload=WorkloadConfig(
            arrival_process="poisson", arrival_rate=15.0, duration=60.0,
            prompt_mean=256.0, output_mean=128.0,
            prediction_noise_rel=0.35,   # high noise
            slo_classes=_mixed_slo_classes(),
            tag="val_noisy_predictions",
        ),
        seed=102,
    ),
]

TEST_REGIMES: List[RegimeSpec] = [
    # Held-out: very high load + high noise (hardest overload condition)
    RegimeSpec(
        name="test_very_overloaded",
        split="test",
        workload=WorkloadConfig(
            arrival_process="poisson", arrival_rate=35.0, duration=60.0,
            prompt_mean=256.0, output_mean=128.0,
            prediction_noise_rel=0.30,
            slo_classes=_tight_slo_classes(),
            tag="test_very_overloaded",
        ),
        seed=200,
    ),
    # Held-out: extreme burst factor (far beyond train_bursty burst_factor=4)
    RegimeSpec(
        name="test_extreme_bursty",
        split="test",
        workload=WorkloadConfig(
            arrival_process="bursty", arrival_rate=25.0, duration=60.0,
            prompt_mean=256.0, output_mean=128.0,
            burst_factor=8.0, burst_fraction=0.25,
            prediction_noise_rel=0.25,
            slo_classes=_mixed_slo_classes(),
            tag="test_extreme_bursty",
        ),
        seed=201,
    ),
    # Held-out: very high prediction noise (0.50 vs 0.35 max in train/val)
    RegimeSpec(
        name="test_high_noise",
        split="test",
        workload=WorkloadConfig(
            arrival_process="poisson", arrival_rate=15.0, duration=60.0,
            prompt_mean=256.0, output_mean=128.0,
            prediction_noise_rel=0.50,
            slo_classes=_mixed_slo_classes(),
            tag="test_high_noise",
        ),
        seed=202,
    ),
]

DEFAULT_REGIMES = TRAIN_REGIMES + VALIDATION_REGIMES

DEFAULT_BASELINES = [
    "fifo", "edf", "least_laxity_first", "estimated_service_time_first",
    "shortest_output_first", "slo_slack_score", "vllm_style_token_budget",
    "sarathi_style", "splitfuse_style", "best_fit",
]

ALL_BASELINES = [
    "fifo", "edf", "shortest_output_first", "shortest_prompt_first",
    "greedy_token_fill", "least_loaded", "multi_bin_batching", "random_feasible",
    "first_fit", "best_fit", "orca_style", "vllm_style_token_budget",
    "sarathi_style", "splitfuse_style", "slo_slack_score",
    "weighted_shortest_processing", "least_laxity_first", "estimated_service_time_first",
]


# ---------------------------------------------------------------------------
# Multi-regime evaluation config
# ---------------------------------------------------------------------------

@dataclass
class MultiRegimeConfig:
    regimes: List[RegimeSpec] = field(default_factory=lambda: DEFAULT_REGIMES)
    baseline_names: List[str] = field(default_factory=lambda: list(DEFAULT_BASELINES))
    drain_steps: int = 50_000
    verbose: bool = True
    include_oracle: bool = False


# ---------------------------------------------------------------------------
# Per-regime aggregated result
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    regime_name: str
    split: str
    heuristics: List[CandidateResult]
    baselines: List[CandidateResult]


# ---------------------------------------------------------------------------
# Aggregated candidate stats across regimes
# ---------------------------------------------------------------------------

@dataclass
class AggregatedCandidateResult:
    name: str
    source: str
    train_mean_wg: float
    val_mean_wg: float
    overall_mean_wg: float
    train_val_gap: float        # val - train (negative = overfit)
    worst_regime_wg: float
    worst_regime_name: str
    train_violation_rate: float
    val_violation_rate: float
    train_p95_ttft: float
    val_p95_ttft: float
    regimes_beating_best_fixed: int
    regimes_beating_slo_slack: int
    regimes_beating_estf: int
    n_train_regimes: int
    n_val_regimes: int
    per_regime: Dict[str, float]   # regime_name -> wg
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate_multi_regime(
    candidate_records: List[Dict[str, Any]],
    cfg: Optional[MultiRegimeConfig] = None,
    *,
    verbose: bool = True,
) -> List[RegimeResult]:
    """Run evaluate_candidates for each regime and return per-regime results."""
    if cfg is None:
        cfg = MultiRegimeConfig()

    results = []
    for regime in cfg.regimes:
        if verbose:
            print(f"\n  [REGIME: {regime.name} ({regime.split})]")
        eval_cfg = EvaluationConfig(
            arrival_rate=regime.workload.arrival_rate,
            duration=regime.workload.duration,
            seed=regime.seed,
            max_active_sequences=regime.max_active_sequences,
            max_batch_tokens=regime.max_batch_tokens,
            max_kv_tokens=regime.max_kv_tokens,
            baseline_names=cfg.baseline_names,
            drain_steps=cfg.drain_steps,
        )
        r = _evaluate_regime(candidate_records, eval_cfg, regime,
                             include_oracle=cfg.include_oracle)
        results.append(RegimeResult(
            regime_name=regime.name,
            split=regime.split,
            heuristics=r["heuristics"],
            baselines=r["baselines"],
        ))
    return results


def _evaluate_regime(
    candidate_records: List[Dict[str, Any]],
    eval_cfg: EvaluationConfig,
    regime: RegimeSpec,
    *,
    include_oracle: bool = False,
) -> Dict[str, List[CandidateResult]]:
    """Evaluate candidates on a single regime using the workload spec."""
    from ..core.types import GPUConfig
    from ..evaluation.run_policy import run_policy
    from ..core.metrics import metrics_to_dict
    from ..workloads.synthetic import generate_workload
    from ..heuristics import build_heuristic_policy
    from ..heuristics.compiler import CompilationError
    from ..policies.registry import BASELINE_NAMES, make_policy, make_oracle_policy

    requests = generate_workload(regime.workload, seed=regime.seed)
    gpu = GPUConfig(
        gpu_id=0,
        max_active_sequences=regime.max_active_sequences,
        max_batch_tokens=regime.max_batch_tokens,
        max_kv_tokens=regime.max_kv_tokens,
    )

    if eval_cfg.verbose if hasattr(eval_cfg, 'verbose') else True:
        print(f"    {len(requests)} requests, rate={regime.workload.arrival_rate:.0f}/s")

    def _run(policy):
        try:
            m = run_policy(policy, requests, [gpu], seed=regime.seed)
            return CandidateResult(
                name=policy.name, source="unknown", policy_name=policy.name,
                weighted_goodput=m.weighted_goodput,
                priority_weighted_slo_goodput=m.priority_weighted_slo_goodput,
                slo_violation_rate=m.slo_violation_rate,
                p95_ttft=m.p95_ttft if m.p95_ttft == m.p95_ttft else 0.0,
                p95_latency=m.p95_latency if m.p95_latency == m.p95_latency else 0.0,
                request_throughput=m.request_throughput,
                num_completed=m.num_completed,
                raw_metrics=metrics_to_dict(m),
            )
        except Exception as e:
            return CandidateResult(
                name=policy.name, source="unknown", policy_name=policy.name,
                weighted_goodput=float("nan"),
                priority_weighted_slo_goodput=float("nan"),
                slo_violation_rate=float("nan"),
                p95_ttft=float("nan"), p95_latency=float("nan"),
                request_throughput=float("nan"), num_completed=0,
                error=str(e),
            )

    heuristic_results = []
    for rec in candidate_records:
        cand = rec.get("candidate", {})
        name = cand.get("name", "unnamed")
        try:
            policy = build_heuristic_policy(cand)
            r = _run(policy)
            r.source = "heuristic"
            r.name = name
        except CompilationError as e:
            r = CandidateResult(
                name=name, source="heuristic", policy_name=name,
                weighted_goodput=float("nan"),
                priority_weighted_slo_goodput=float("nan"),
                slo_violation_rate=float("nan"),
                p95_ttft=float("nan"), p95_latency=float("nan"),
                request_throughput=float("nan"), num_completed=0,
                error=f"CompilationError: {e}",
            )
        heuristic_results.append(r)

    baseline_results = []
    for bname in eval_cfg.baseline_names:
        if bname not in BASELINE_NAMES:
            continue
        policy = make_policy(bname)
        r = _run(policy)
        r.source = "baseline"
        r.name = bname
        baseline_results.append(r)

    if include_oracle:
        oracle_policy = make_oracle_policy("oracle_srtf", requests)
        r = _run(oracle_policy)
        r.source = "oracle"
        r.name = "oracle_srtf"
        baseline_results.append(r)

    return {"heuristics": heuristic_results, "baselines": baseline_results}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _nanmean(vals: List[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def _nanmin(vals: List[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return min(clean) if clean else float("nan")


def aggregate_regime_results(
    regime_results: List[RegimeResult],
    *,
    candidate_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, AggregatedCandidateResult]:
    """Aggregate per-regime CandidateResults into per-candidate summaries."""
    # Collect all candidate names from heuristics
    all_names = {}   # name -> source
    for rr in regime_results:
        for r in rr.heuristics:
            all_names[r.name] = "heuristic"
        for r in rr.baselines:
            all_names[r.name] = "baseline"

    # Build per-candidate, per-regime wg dict
    # {candidate_name: {regime_name: CandidateResult}}
    cand_by_regime: Dict[str, Dict[str, CandidateResult]] = {n: {} for n in all_names}
    for rr in regime_results:
        for r in rr.heuristics + rr.baselines:
            cand_by_regime[r.name][rr.regime_name] = r

    # Per-regime best-fixed baselines for comparison
    best_fixed_per_regime: Dict[str, float] = {}
    slo_slack_per_regime: Dict[str, float] = {}
    estf_per_regime: Dict[str, float] = {}
    for rr in regime_results:
        wgs = [r.priority_weighted_slo_goodput for r in rr.baselines
               if not math.isnan(r.priority_weighted_slo_goodput)]
        best_fixed_per_regime[rr.regime_name] = max(wgs) if wgs else float("nan")
        for r in rr.baselines:
            if r.name == "slo_slack_score":
                slo_slack_per_regime[rr.regime_name] = r.priority_weighted_slo_goodput
            if r.name == "estimated_service_time_first":
                estf_per_regime[rr.regime_name] = r.priority_weighted_slo_goodput

    aggregated: Dict[str, AggregatedCandidateResult] = {}
    for name, source in all_names.items():
        regime_map = cand_by_regime[name]
        train_wgs, val_wgs, all_wgs = [], [], []
        train_vr, val_vr = [], []
        train_ttft, val_ttft = [], []
        per_regime_wg = {}
        worst_wg, worst_name = float("inf"), ""
        beats_best_fixed = 0
        beats_slo_slack = 0
        beats_estf = 0

        for rr in regime_results:
            r = regime_map.get(rr.regime_name)
            if r is None:
                continue
            wg = r.priority_weighted_slo_goodput
            per_regime_wg[rr.regime_name] = wg
            if not math.isnan(wg):
                all_wgs.append(wg)
                if wg < worst_wg:
                    worst_wg, worst_name = wg, rr.regime_name
                if rr.split == "train":
                    train_wgs.append(wg)
                    if not math.isnan(r.slo_violation_rate):
                        train_vr.append(r.slo_violation_rate)
                    if not math.isnan(r.p95_ttft):
                        train_ttft.append(r.p95_ttft)
                elif rr.split == "validation":
                    val_wgs.append(wg)
                    if not math.isnan(r.slo_violation_rate):
                        val_vr.append(r.slo_violation_rate)
                    if not math.isnan(r.p95_ttft):
                        val_ttft.append(r.p95_ttft)
                # Count beats
                bf = best_fixed_per_regime.get(rr.regime_name, float("nan"))
                if not math.isnan(bf) and wg > bf:
                    beats_best_fixed += 1
                ss = slo_slack_per_regime.get(rr.regime_name, float("nan"))
                if not math.isnan(ss) and wg > ss:
                    beats_slo_slack += 1
                ef = estf_per_regime.get(rr.regime_name, float("nan"))
                if not math.isnan(ef) and wg > ef:
                    beats_estf += 1

        train_mean = _nanmean(train_wgs)
        val_mean = _nanmean(val_wgs)
        overall_mean = _nanmean(all_wgs)
        gap = val_mean - train_mean if not (math.isnan(val_mean) or math.isnan(train_mean)) else float("nan")

        n_train = sum(1 for rr in regime_results if rr.split == "train")
        n_val = sum(1 for rr in regime_results if rr.split == "validation")

        aggregated[name] = AggregatedCandidateResult(
            name=name,
            source=source,
            train_mean_wg=train_mean,
            val_mean_wg=val_mean,
            overall_mean_wg=overall_mean,
            train_val_gap=gap,
            worst_regime_wg=worst_wg if worst_wg < float("inf") else float("nan"),
            worst_regime_name=worst_name,
            train_violation_rate=_nanmean(train_vr),
            val_violation_rate=_nanmean(val_vr),
            train_p95_ttft=_nanmean(train_ttft),
            val_p95_ttft=_nanmean(val_ttft),
            regimes_beating_best_fixed=beats_best_fixed,
            regimes_beating_slo_slack=beats_slo_slack,
            regimes_beating_estf=beats_estf,
            n_train_regimes=n_train,
            n_val_regimes=n_val,
            per_regime=per_regime_wg,
        )

    return aggregated
