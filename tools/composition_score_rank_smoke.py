#!/usr/bin/env python3
"""Small correctness smoke for score aggregation, reciprocal-rank aggregation,
and decision instrumentation.

This is not a scientific performance claim: it runs a single tiny fixed-seed
synthetic scenario through a handful of native policies and composed
variants, records their decisions, and writes a compact comparison artifact.
See docs/current/COMPOSITION_IMPLEMENTATION_STATUS.md for the broader
(also-not-yet-decisive) composition harness this complements, and
docs/current/WOLVERINE_ORACLE_MIXTURE_HANDOFF.md for the large-scale sweep
this is meant to prepare the ground for.

Usage:
    python tools/composition_score_rank_smoke.py [--output PATH] [--trace-jsonl PATH]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.composition import RankExpertSpec, StaticRankEnsemblePolicy
from llmserveopt.policies.instrumentation import DecisionTraceSink, InstrumentedPolicy
from llmserveopt.policies.registry import make_policy
from llmserveopt.policies.score_aggregation import NormalizationMode, ScoreExpertSpec, StaticScoreEnsemblePolicy
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

SEED = 20260724
PARENT_POLICIES = ["weighted_shortest_processing", "edf", "shortest_prompt_first"]


def make_tiny_trace(seed: int = SEED, n: int = 40) -> list[Request]:
    """Deterministic small synthetic trace: mixed prompt/output lengths and
    deadlines, arriving in a tight burst so a 2-slot GPU sees real queuing
    contention (otherwise every policy trivially completes everything and
    the comparison proves nothing)."""
    rng = random.Random(seed)
    requests = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(1.0 / 0.01)
        prompt = rng.choice([32, 64, 128, 256, 512])
        output = rng.choice([16, 32, 64, 128, 256])
        tight = rng.random() < 0.3
        deadline = t + (prompt + output) * (0.001 if tight else 0.01) + rng.uniform(0.01, 0.05)
        requests.append(
            Request(
                request_id=i,
                arrival_time=round(t, 4),
                prompt_tokens=prompt,
                predicted_output_tokens=output,
                actual_output_tokens=output,
                slo_deadline=round(deadline, 4),
                priority=rng.choice([1.0, 1.0, 1.0, 2.0]),
                class_id="tight" if tight else "medium",
            )
        )
    return requests


def make_gpu() -> GPUConfig:
    return GPUConfig(gpu_id=0, max_active_sequences=2, max_batch_tokens=64, max_kv_tokens=768)


def run_policy(name: str, policy, trace: list[Request], sink: DecisionTraceSink | None = None) -> dict:
    sim = Simulator(SimulatorConfig(gpu_configs=[make_gpu()], drain_steps=2000))
    sim.load_trace(trace)
    to_run = InstrumentedPolicy(policy, sink) if sink is not None else policy
    metrics = sim.run(to_run, workload_tag="composition_score_rank_smoke", seed=SEED)
    return {
        "policy": name,
        "arrival_normalized_weighted_goodput": metrics.arrival_normalized_weighted_goodput,
        "completion_fraction": metrics.completion_fraction,
        "slo_violation_rate": metrics.num_slo_violated / max(metrics.num_completed, 1),
        "num_completed": metrics.num_completed,
        "num_dropped": metrics.num_dropped,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/composition_score_rank_smoke/smoke_result.json"))
    parser.add_argument("--trace-jsonl", type=Path, default=Path("results/composition_score_rank_smoke/decision_trace.jsonl"))
    args = parser.parse_args()

    trace = make_tiny_trace()

    parent_results = []
    for name in PARENT_POLICIES:
        parent_results.append(run_policy(name, make_policy(name), trace))

    borda = StaticRankEnsemblePolicy(
        [RankExpertSpec(name, 1.0) for name in PARENT_POLICIES], method="borda"
    )
    rrf = StaticRankEnsemblePolicy(
        [RankExpertSpec(name, 1.0) for name in PARENT_POLICIES], method="reciprocal_rank"
    )
    score_minmax = StaticScoreEnsemblePolicy(
        [ScoreExpertSpec("weighted_shortest_processing", 1.0), ScoreExpertSpec("edf", 1.0)],
        normalization=NormalizationMode.MIN_MAX,
    )
    score_robust = StaticScoreEnsemblePolicy(
        [ScoreExpertSpec("weighted_shortest_processing", 1.0), ScoreExpertSpec("edf", 1.0)],
        normalization=NormalizationMode.ROBUST_MAD,
    )

    sink = DecisionTraceSink(enabled=True, scenario_id="composition_score_rank_smoke")
    composed_results = [
        run_policy("composition_static_rank_borda", borda, trace, sink=sink),
        run_policy("composition_static_rank_reciprocal", rrf, trace, sink=sink),
        run_policy("composition_static_score_minmax", score_minmax, trace, sink=sink),
        run_policy("composition_static_score_robust_mad", score_robust, trace, sink=sink),
    ]

    all_results = parent_results + composed_results
    best_parent = max(parent_results, key=lambda r: r["arrival_normalized_weighted_goodput"])
    best_composed = max(composed_results, key=lambda r: r["arrival_normalized_weighted_goodput"])

    result = {
        "status": "PASS",
        "scientific_claim": "correctness smoke only -- not a decisive composition-vs-best-policy comparison",
        "seed": SEED,
        "trace_size": len(trace),
        "parent_policies": PARENT_POLICIES,
        "results": all_results,
        "best_parent_policy": best_parent["policy"],
        "best_parent_anwg": best_parent["arrival_normalized_weighted_goodput"],
        "best_composed_method": best_composed["policy"],
        "best_composed_anwg": best_composed["arrival_normalized_weighted_goodput"],
        "composed_beats_best_parent": best_composed["arrival_normalized_weighted_goodput"] > best_parent["arrival_normalized_weighted_goodput"],
        "decision_trace_records": len(sink.records),
        "decision_trace_path": str(args.trace_jsonl),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sink.write_jsonl(args.trace_jsonl)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
