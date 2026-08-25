#!/usr/bin/env python3
"""Build a CPU-only Selector Dataset v2 pilot.

HISTORICAL ENTRY POINT. This was the first Selector Dataset v2 pilot
generator; retained for reproducibility of that early pilot's results. It
was superseded by scripts/build_selector_dataset_v2_redesigned_pilot.py and
then by scripts/build_selector_dataset_v2_calibrated_targeted_pilot.py
(current, Option B scope). See docs/selector_dataset_v2.md for the full
generation lineage.

This script intentionally does not train a selector. It generates a small
topology-aware full-outcome dataset plus summary/manifests suitable for deciding
whether large-scale generation is warranted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.selector.dataset_v2.builder import (
    assemble_dataset_rows,
    build_selector_dataset_v2,
)
from llmserveopt.selector.dataset_v2.candidates import candidate_policies_for_topology
from llmserveopt.selector.dataset_v2.discriminativeness import STANDARD_OBJECTIVES
from llmserveopt.selector.dataset_v2.scenario_families import (
    ScenarioFamilySpec,
    all_scenario_family_specs,
)
from llmserveopt.selector.dataset_v2.schema import DatasetManifestV2, WindowRecordV2
from llmserveopt.selector.dataset_v2.splits import verify_group_atomicity, verify_ood_holdout
from llmserveopt.selector.dataset_v2.workload_sources import WORKLOAD_SOURCE_MANIFEST
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.trace_io_extended import load_extended_jsonl


def _load_jsonl_slice(path: Path, max_requests: int, offset_seed: int) -> List[Request]:
    requests, _metadata = load_extended_jsonl(path)
    if len(requests) <= max_requests:
        return requests
    max_start = len(requests) - max_requests
    start = (offset_seed * 9973) % (max_start + 1)
    chunk = requests[start:start + max_requests]
    t0 = chunk[0].arrival_time
    return [
        Request(
            request_id=i,
            arrival_time=r.arrival_time - t0,
            prompt_tokens=r.prompt_tokens,
            predicted_output_tokens=r.predicted_output_tokens,
            actual_output_tokens=r.actual_output_tokens,
            slo_deadline=r.slo_deadline - t0,
            priority=r.priority,
            class_id=r.class_id,
        )
        for i, r in enumerate(chunk)
    ]


def _local_trace_spec(
    family_id: str,
    source_trace: str,
    path: str,
    max_requests: int,
    temporal_block_id: str,
) -> ScenarioFamilySpec | None:
    p = ROOT / path
    if not p.exists():
        return None

    def _build(seed: int, _path=p, _max=max_requests) -> List[Request]:
        return _load_jsonl_slice(_path, _max, seed)

    return ScenarioFamilySpec(
        family_id=family_id,
        dataset_family="real_trace",
        source_trace=source_trace,
        temporal_block_id=temporal_block_id,
        description=f"Local processed real trace slice: {path}",
        build=_build,
    )


def _capped_spec(spec: ScenarioFamilySpec, max_requests: int) -> ScenarioFamilySpec:
    def _build(seed: int, _spec=spec, _max=max_requests) -> List[Request]:
        return _spec.build(seed)[:_max]

    return ScenarioFamilySpec(
        family_id=spec.family_id,
        dataset_family=spec.dataset_family,
        source_trace=spec.source_trace,
        temporal_block_id=spec.temporal_block_id,
        description=f"{spec.description} (pilot-capped at {max_requests} requests)",
        build=_build,
    )


def pilot_scenario_specs(max_real_requests: int, max_synthetic_requests: int) -> List[ScenarioFamilySpec]:
    specs = [_capped_spec(spec, max_synthetic_requests) for spec in all_scenario_family_specs()]
    for maybe in [
        _local_trace_spec(
            "real_trace__burstgpt_scaled_moderate",
            "burstgpt",
            "data/processed/burstgpt/burstgpt_scaled_moderate_10k.jsonl",
            max_real_requests,
            "seeded_slice",
        ),
        _local_trace_spec(
            "real_trace__azure_2023_code",
            "azure_llm_2023",
            "data/processed/azure/azure_llm_2023_code.jsonl",
            max_real_requests,
            "seeded_slice",
        ),
        _local_trace_spec(
            "real_trace__azure_2023_conv",
            "azure_llm_2023",
            "data/processed/azure/azure_llm_2023_conv.jsonl",
            max_real_requests,
            "seeded_slice",
        ),
    ]:
        if maybe is not None:
            specs.append(maybe)
    return specs


def summarize_records(records: List[WindowRecordV2]) -> dict:
    wg_name = "weighted_goodput"
    win_counts = Counter()
    strong_win_counts = Counter()
    class_counts = Counter()
    all_complete = 0
    policy_values = defaultdict(list)
    oracle_values = []

    for record in records:
        by_policy = {o.policy_name: o.weighted_goodput for o in record.outcomes if o.weighted_goodput is not None}
        for policy, value in by_policy.items():
            policy_values[policy].append(value)
        disc = next((d for d in record.discriminativeness if d.objective_name == wg_name), None)
        if disc is not None:
            win_counts[disc.best_policy] += 1
            class_counts[disc.classification] += 1
            if disc.classification == "STRONGLY_DISCRIMINATIVE":
                strong_win_counts[disc.best_policy] += 1
            oracle_values.append(disc.best_value)
        completion_values = [o.completion_fraction for o in record.outcomes if o.completion_fraction is not None]
        if completion_values and all(abs(v - 1.0) <= 1e-12 for v in completion_values):
            all_complete += 1

    means = {
        policy: (sum(vals) / len(vals))
        for policy, vals in policy_values.items()
        if vals
    }
    if means:
        global_best_fixed_policy = max(means, key=means.get)
        global_best_fixed_score = means[global_best_fixed_policy]
    else:
        global_best_fixed_policy = None
        global_best_fixed_score = None

    oracle_score = sum(oracle_values) / len(oracle_values) if oracle_values else None
    headroom = (
        oracle_score - global_best_fixed_score
        if oracle_score is not None and global_best_fixed_score is not None
        else None
    )
    total = len(records)
    strongly = class_counts["STRONGLY_DISCRIMINATIVE"] / total if total else 0.0
    strong_total = sum(strong_win_counts.values())
    strong_max_share = (
        max(strong_win_counts.values()) / strong_total
        if strong_total else 0.0
    )
    near = class_counts["NEAR_TIE"] / total if total else 0.0
    all_complete_fraction = all_complete / total if total else 0.0

    return {
        "num_windows": total,
        "weighted_goodput_discriminativeness": dict(class_counts),
        "strongly_discriminative_fraction": strongly,
        "near_tie_fraction": near,
        "all_complete_fraction": all_complete_fraction,
        "policy_win_distribution": dict(win_counts),
        "strong_window_policy_win_distribution": dict(strong_win_counts),
        "strong_window_top_policy_share": strong_max_share,
        "global_best_fixed_policy": global_best_fixed_policy,
        "global_best_fixed_score": global_best_fixed_score,
        "per_scenario_oracle_score": oracle_score,
        "oracle_headroom": headroom,
    }


def _write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/selector_dataset_v2/pilot")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 17])
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--max-synthetic-requests", type=int, default=80)
    parser.add_argument("--max-real-requests", type=int, default=80)
    parser.add_argument("--max-windows", type=int, default=260)
    parser.add_argument("--topology-class", default="monolithic")
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--drain-steps", type=int, default=20_000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    policies = args.policies or candidate_policies_for_topology(args.topology_class)
    gpu_configs = [
        GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=20_000),
    ]

    specs = pilot_scenario_specs(args.max_real_requests, args.max_synthetic_requests)
    t0 = time.perf_counter()
    records = build_selector_dataset_v2(
        scenario_specs=specs,
        seeds=args.seeds,
        gpu_configs=gpu_configs,
        candidate_policies=policies,
        service_model=ServiceModel(),
        window_size=args.window_size,
        drain_steps=args.drain_steps,
        topology_class=args.topology_class,
        verbose=args.verbose,
    )
    if args.max_windows and len(records) > args.max_windows:
        records = records[: args.max_windows]

    present_families = {record.identifiers.scenario_family_id for record in records}
    if "real_trace__azure_2023_conv" in present_families:
        ood = {"real_trace__azure_2023_conv"}
    elif "real_trace__azure_2023_code" in present_families:
        ood = {"real_trace__azure_2023_code"}
    else:
        ood = set()
    rows = assemble_dataset_rows(records, ood_scenario_family_ids=ood)
    verify_group_atomicity(rows, "scenario_family_id")
    if ood:
        verify_ood_holdout(rows, "scenario_family_id", ood)

    summary = summarize_records(records)
    elapsed = time.perf_counter() - t0
    split_counts = Counter(row["split"] for row in rows)
    manifest = DatasetManifestV2(
        dataset_name="selector_dataset_v2_pilot",
        schema_version="2.0",
        topology_class=args.topology_class,
        candidate_policies=policies,
        feature_names=sorted(k.removeprefix("feat_") for k in rows[0] if k.startswith("feat_")) if rows else [],
        objectives=[o.name for o in STANDARD_OBJECTIVES],
        num_scenarios=len({r.identifiers.scenario_id for r in records}),
        num_windows=len(records),
        num_policy_evaluations=len(rows),
        scenario_family_ids=sorted({r.identifiers.scenario_family_id for r in records}),
        source_traces=sorted({r.identifiers.source_trace for r in records}),
        seeds=args.seeds,
        split_group_key="scenario_family_id",
        split_counts=dict(split_counts),
        quality_gate_results={
            "at_least_3_policies_with_meaningful_wins": len(summary["policy_win_distribution"]) >= 3,
            "single_policy_strong_window_dominance_lte_85pct": summary["strong_window_top_policy_share"] <= 0.85,
            "substantial_oracle_headroom": (summary["oracle_headroom"] or 0.0) >= 0.01,
            "real_trace_representation": any(r.identifiers.dataset_family == "real_trace" for r in records),
            "controlled_stress_representation": any(r.identifiers.dataset_family == "controlled_stress" for r in records),
            "ood_split_defined": bool(ood),
            "all_complete_fraction_below_50pct": summary["all_complete_fraction"] < 0.5,
        },
        generation_config={
            "window_size": args.window_size,
            "max_real_requests": args.max_real_requests,
            "max_synthetic_requests": args.max_synthetic_requests,
            "max_windows": args.max_windows,
            "drain_steps": args.drain_steps,
            "elapsed_seconds": round(elapsed, 3),
        },
        notes=[
            "CPU-only pilot; not a final selector-training dataset.",
            "No selector model is trained by this script.",
            "Rows are scenario/window x topology x policy.",
        ],
    )

    _write_csv(rows, out_dir / "selector_dataset_v2_pilot.csv")
    _write_json(manifest.to_dict(), out_dir / "manifest.json")
    _write_json(summary, out_dir / "pilot_summary.json")
    _write_json([asdict(s) for s in WORKLOAD_SOURCE_MANIFEST], out_dir / "workload_source_manifest.json")
    print(json.dumps({
        "output_dir": str(out_dir),
        "windows": len(records),
        "policy_evaluations": len(rows),
        "elapsed_seconds": round(elapsed, 3),
        **summary,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
