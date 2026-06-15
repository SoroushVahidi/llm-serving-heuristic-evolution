"""
Compile verified heuristics and evaluate them in the simulator.

Fitness oracle: priority_weighted_slo_goodput (= weighted_goodput)
The selector is NOT the fitness oracle — it is an adaptive baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..core.metrics import RunMetrics, metrics_to_dict
from ..core.types import GPUConfig, Request
from ..evaluation.run_policy import run_policy
from ..heuristics import build_heuristic_policy
from ..heuristics.compiler import CompilationError
from ..policies.registry import BASELINE_NAMES, make_policy
from ..workloads.synthetic import WorkloadConfig, generate_workload


@dataclass
class EvaluationConfig:
    # Smoke workload parameters
    arrival_rate: float = 15.0
    duration: float = 30.0
    seed: int = 42
    # GPU config (stressed, same as selector build)
    max_active_sequences: int = 4
    max_batch_tokens: int = 512
    max_kv_tokens: int = 8192
    # Baselines to compare against (subset of BASELINE_NAMES)
    baseline_names: List[str] = field(default_factory=lambda: [
        "fifo", "edf",
        "least_laxity_first", "estimated_service_time_first",
        "slo_slack_score", "vllm_style_token_budget", "sarathi_style", "best_fit",
    ])
    drain_steps: int = 50_000


@dataclass
class CandidateResult:
    name: str
    source: str           # "heuristic" | "baseline"
    policy_name: str
    weighted_goodput: float
    priority_weighted_slo_goodput: float
    slo_violation_rate: float
    p95_ttft: float
    p95_latency: float
    request_throughput: float
    num_completed: int
    error: Optional[str] = None
    raw_metrics: Optional[Dict] = field(default=None, repr=False)


def _make_gpu(cfg: EvaluationConfig) -> GPUConfig:
    return GPUConfig(
        gpu_id=0,
        max_active_sequences=cfg.max_active_sequences,
        max_batch_tokens=cfg.max_batch_tokens,
        max_kv_tokens=cfg.max_kv_tokens,
    )


def _run_and_collect(policy, requests: List[Request], gpu: GPUConfig, seed: int) -> CandidateResult:
    try:
        m = run_policy(policy, requests, [gpu], seed=seed)
        return CandidateResult(
            name=policy.name,
            source="unknown",
            policy_name=policy.name,
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
            name=policy.name,
            source="unknown",
            policy_name=policy.name,
            weighted_goodput=float("nan"),
            priority_weighted_slo_goodput=float("nan"),
            slo_violation_rate=float("nan"),
            p95_ttft=float("nan"),
            p95_latency=float("nan"),
            request_throughput=float("nan"),
            num_completed=0,
            error=str(e),
        )


def evaluate_candidates(
    candidate_records: List[Dict[str, Any]],
    cfg: Optional[EvaluationConfig] = None,
) -> Dict[str, List[CandidateResult]]:
    """Evaluate verified heuristic candidates and baselines.

    Parameters
    ----------
    candidate_records : list of dicts with "candidate" key (heuristic JSON).
    cfg : EvaluationConfig.

    Returns
    -------
    dict with keys "heuristics" and "baselines", each a list of CandidateResult.
    """
    if cfg is None:
        cfg = EvaluationConfig()

    wcfg = WorkloadConfig(
        arrival_process="poisson",
        arrival_rate=cfg.arrival_rate,
        duration=cfg.duration,
        prompt_mean=256.0,
        output_mean=128.0,
    )
    requests = generate_workload(wcfg, seed=cfg.seed)
    gpu = _make_gpu(cfg)

    print(f"Evaluation: {len(requests)} requests, GPU max_seq={cfg.max_active_sequences}, "
          f"max_kv={cfg.max_kv_tokens}")

    heuristic_results: List[CandidateResult] = []
    baseline_results: List[CandidateResult] = []

    # Evaluate heuristic candidates
    for rec in candidate_records:
        cand = rec.get("candidate", {})
        name = cand.get("name", "unnamed")
        print(f"  [heuristic] {name}...")
        try:
            policy = build_heuristic_policy(cand)
            r = _run_and_collect(policy, requests, gpu, cfg.seed)
            r.source = "heuristic"
            r.name = name
            heuristic_results.append(r)
        except CompilationError as e:
            heuristic_results.append(CandidateResult(
                name=name, source="heuristic", policy_name=name,
                weighted_goodput=float("nan"),
                priority_weighted_slo_goodput=float("nan"),
                slo_violation_rate=float("nan"),
                p95_ttft=float("nan"), p95_latency=float("nan"),
                request_throughput=float("nan"), num_completed=0,
                error=f"CompilationError: {e}",
            ))

    # Evaluate baselines
    for bname in cfg.baseline_names:
        if bname not in BASELINE_NAMES:
            print(f"  [WARN] '{bname}' not in BASELINE_NAMES — skipping")
            continue
        print(f"  [baseline] {bname}...")
        policy = make_policy(bname)
        r = _run_and_collect(policy, requests, gpu, cfg.seed)
        r.source = "baseline"
        r.name = bname
        baseline_results.append(r)

    return {"heuristics": heuristic_results, "baselines": baseline_results}
