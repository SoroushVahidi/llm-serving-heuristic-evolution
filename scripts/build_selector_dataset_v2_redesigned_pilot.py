#!/usr/bin/env python3
"""Build the redesigned, bottleneck-oriented Selector Dataset v2 pilot."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import GPUConfig
from llmserveopt.selector.dataset_v2.builder import (
    assemble_dataset_rows,
    build_selector_dataset_v2,
    build_selector_dataset_v2_trials,
)
from llmserveopt.selector.dataset_v2.candidates import candidate_policies_for_topology
from llmserveopt.selector.dataset_v2.discriminativeness import STANDARD_OBJECTIVES
from llmserveopt.selector.dataset_v2.scenario_redesign import (
    DISCRIMINATIVE_POOL,
    REPRESENTATIVE_POOL,
    bottleneck_taxonomy_specs,
    local_real_trace_stress_specs,
    representative_easy_specs,
    sampled_bottleneck_specs,
    targeted_counterexample_specs,
)
from llmserveopt.selector.dataset_v2.scenario_search import (
    DISCRIMINATIVE_CLASSES,
    attach_retention,
    diversity_aware_retained_pool_for_trial,
    retained_pool_for_trial,
    spec_with_retained_pool,
    summarize_trial,
)
from llmserveopt.selector.dataset_v2.schema import DatasetManifestV2, WindowRecordV2
from llmserveopt.selector.dataset_v2.splits import verify_group_atomicity, verify_ood_holdout
from llmserveopt.simulator.service_model import ServiceModel


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


def _wg_disc(record: WindowRecordV2):
    return next(d for d in record.discriminativeness if d.objective_name == "weighted_goodput")


def summarize_records(records: List[WindowRecordV2], policies: List[str]) -> dict:
    win_counts = Counter()
    discriminative_win_counts = Counter()
    strong_win_counts = Counter()
    class_counts = Counter()
    pool_counts = Counter()
    bottleneck_counts = Counter()
    all_complete = 0
    policy_values = defaultdict(list)
    oracle_values = []
    discr_oracle_values = []
    discr_policy_values = defaultdict(list)
    random_values = []
    rule_values = []

    for record in records:
        disc = _wg_disc(record)
        class_counts[disc.classification] += 1
        win_counts[disc.best_policy] += 1
        if disc.classification in DISCRIMINATIVE_CLASSES:
            discriminative_win_counts[disc.best_policy] += 1
        if disc.classification == "STRONGLY_DISCRIMINATIVE":
            strong_win_counts[disc.best_policy] += 1
        pool_counts[record.identifiers.scenario_pool] += 1
        bottleneck_counts[record.identifiers.bottleneck_class or "unknown"] += 1
        oracle_values.append(disc.best_value)

        by_policy = {o.policy_name: o.weighted_goodput for o in record.outcomes if o.weighted_goodput is not None}
        for policy, value in by_policy.items():
            policy_values[policy].append(value)
        if disc.classification in DISCRIMINATIVE_CLASSES:
            discr_oracle_values.append(disc.best_value)
            for policy, value in by_policy.items():
                discr_policy_values[policy].append(value)
        completions = [o.completion_fraction for o in record.outcomes if o.completion_fraction is not None]
        if completions and all(abs(v - 1.0) <= 1e-12 for v in completions):
            all_complete += 1

        random_values.append(sum(by_policy.values()) / len(by_policy) if by_policy else 0.0)
        rule_policy = _rule_policy_for_record(record)
        rule_values.append(by_policy.get(rule_policy, by_policy.get("scorpio_style_slo_guard", 0.0)))

    means = {policy: sum(vals) / len(vals) for policy, vals in policy_values.items() if vals}
    best_fixed_policy = max(means, key=means.get) if means else None
    best_fixed_score = means[best_fixed_policy] if best_fixed_policy else None
    oracle = sum(oracle_values) / len(oracle_values) if oracle_values else None
    headroom = oracle - best_fixed_score if oracle is not None and best_fixed_score is not None else None

    discr_means = {policy: sum(vals) / len(vals) for policy, vals in discr_policy_values.items() if vals}
    discr_best_fixed = max(discr_means.values()) if discr_means else None
    discr_oracle = sum(discr_oracle_values) / len(discr_oracle_values) if discr_oracle_values else None
    discr_headroom = (
        discr_oracle - discr_best_fixed
        if discr_oracle is not None and discr_best_fixed is not None
        else None
    )
    total = len(records)
    strong_total = sum(strong_win_counts.values())
    strong_top_share = max(strong_win_counts.values()) / strong_total if strong_total else 0.0

    return {
        "num_windows": total,
        "weighted_goodput_discriminativeness": dict(class_counts),
        "all_complete_fraction": class_counts["ALL_COMPLETE_OR_EFFECTIVELY_TIED"] / total if total else 0.0,
        "near_tie_fraction": class_counts["NEAR_TIE"] / total if total else 0.0,
        "moderately_discriminative_fraction": class_counts["MODERATELY_DISCRIMINATIVE"] / total if total else 0.0,
        "strongly_discriminative_fraction": class_counts["STRONGLY_DISCRIMINATIVE"] / total if total else 0.0,
        "all_policies_complete_fraction": all_complete / total if total else 0.0,
        "policy_win_distribution": dict(win_counts),
        "discriminative_policy_win_distribution": dict(discriminative_win_counts),
        "strong_window_policy_win_distribution": dict(strong_win_counts),
        "strong_window_top_policy_share": strong_top_share,
        "pool_window_counts": dict(pool_counts),
        "bottleneck_window_counts": dict(bottleneck_counts),
        "global_best_fixed_policy": best_fixed_policy,
        "global_best_fixed_score": best_fixed_score,
        "per_scenario_oracle_score": oracle,
        "oracle_headroom": headroom,
        "discriminative_oracle_score": discr_oracle,
        "discriminative_oracle_headroom": discr_headroom,
        "random_policy_baseline_score": sum(random_values) / len(random_values) if random_values else None,
        "simple_rule_selector_score": sum(rule_values) / len(rule_values) if rule_values else None,
        "faithful_baseline_wins": {
            "vllm_faithful": win_counts.get("vllm_faithful", 0),
            "sarathi_faithful": win_counts.get("sarathi_faithful", 0),
        },
        "faithful_baseline_discriminative_wins": {
            "vllm_faithful": discriminative_win_counts.get("vllm_faithful", 0),
            "sarathi_faithful": discriminative_win_counts.get("sarathi_faithful", 0),
        },
        "secondary_objective_winners": secondary_objective_winners(records),
    }


def secondary_objective_winners(records: List[WindowRecordV2]) -> dict:
    """Objective-sensitivity audit without changing the primary objective."""
    objectives = [
        "weighted_goodput",
        "arrival_normalized_weighted_goodput",
        "slo_attainment",
        "request_throughput",
    ]
    out: dict[str, dict] = {}
    for objective in objectives:
        wins = Counter()
        strong = Counter()
        oracle_values = []
        policy_values = defaultdict(list)
        for record in records:
            disc = next((d for d in record.discriminativeness if d.objective_name == objective), None)
            if disc is None:
                continue
            wins[disc.best_policy] += 1
            if disc.classification == "STRONGLY_DISCRIMINATIVE":
                strong[disc.best_policy] += 1
            oracle_values.append(disc.best_value)
            for outcome in record.outcomes:
                value = getattr(outcome, objective, None)
                if value is not None:
                    policy_values[outcome.policy_name].append(value)
        means = {p: sum(v) / len(v) for p, v in policy_values.items() if v}
        best_fixed = max(means, key=means.get) if means else None
        oracle = sum(oracle_values) / len(oracle_values) if oracle_values else None
        out[objective] = {
            "win_distribution": dict(wins),
            "strong_win_distribution": dict(strong),
            "best_fixed_policy": best_fixed,
            "best_fixed_score": means.get(best_fixed) if best_fixed else None,
            "oracle_score": oracle,
            "oracle_headroom": (
                oracle - means[best_fixed]
                if oracle is not None and best_fixed is not None else None
            ),
        }

    out["completion_adjusted_weighted_goodput"] = _derived_objective_summary(
        records,
        lambda o: (
            o.weighted_goodput * o.completion_fraction
            if o.weighted_goodput is not None and o.completion_fraction is not None
            else None
        ),
        higher_is_better=True,
    )
    out["p95_latency_constrained_goodput"] = _derived_objective_summary(
        records,
        _p95_latency_constrained_goodput,
        higher_is_better=True,
    )
    return out


def _p95_latency_constrained_goodput(outcome) -> float | None:
    if outcome.weighted_goodput is None or outcome.p95_latency is None:
        return None
    if outcome.slo_attainment is None:
        return outcome.weighted_goodput
    return outcome.weighted_goodput * outcome.slo_attainment / max(outcome.p95_latency, 1e-9)


def _derived_objective_summary(records: List[WindowRecordV2], extractor, *, higher_is_better: bool) -> dict:
    sign = 1.0 if higher_is_better else -1.0
    wins = Counter()
    strong = Counter()
    oracle_values = []
    policy_values = defaultdict(list)
    for record in records:
        values = {
            outcome.policy_name: extractor(outcome)
            for outcome in record.outcomes
        }
        values = {p: v for p, v in values.items() if v is not None}
        if len(values) < 2:
            continue
        ranked = sorted(values.items(), key=lambda item: -sign * item[1])
        best_policy, best = ranked[0]
        second = ranked[1][1]
        margin = sign * (best - second)
        wins[best_policy] += 1
        if margin >= 0.02:
            strong[best_policy] += 1
        oracle_values.append(best)
        for policy, value in values.items():
            policy_values[policy].append(value)
    means = {p: sum(v) / len(v) for p, v in policy_values.items() if v}
    best_fixed = max(means, key=lambda p: sign * means[p]) if means else None
    oracle = sum(oracle_values) / len(oracle_values) if oracle_values else None
    return {
        "win_distribution": dict(wins),
        "strong_win_distribution": dict(strong),
        "best_fixed_policy": best_fixed,
        "best_fixed_score": means.get(best_fixed) if best_fixed else None,
        "oracle_score": oracle,
        "oracle_headroom": (
            sign * (oracle - means[best_fixed])
            if oracle is not None and best_fixed is not None else None
        ),
    }


def _rule_policy_for_record(record: WindowRecordV2) -> str:
    b = record.identifiers.bottleneck_class or ""
    if b == "prefill_heavy":
        return "sarathi_faithful"
    if b in {"kv_pressure", "decode_heavy"}:
        return "weighted_shortest_processing"
    if b in {"slo_heterogeneous", "admission_pressure", "bursty_transient"}:
        return "scorpio_style_slo_guard"
    if b == "prediction_noise":
        return "edf"
    return "scorpio_style_slo_guard"


def specialization_report(records: List[WindowRecordV2]) -> dict:
    rows_by_winner = defaultdict(list)
    for record in records:
        disc = _wg_disc(record)
        rows_by_winner[disc.best_policy].append((record, disc))

    report = {}
    for policy, items in rows_by_winner.items():
        features = defaultdict(list)
        bottlenecks = Counter()
        classes = Counter()
        for record, disc in items:
            bottlenecks[record.identifiers.bottleneck_class or "unknown"] += 1
            classes[disc.classification] += 1
            for name in [
                "arrival_rate_prefix",
                "saturation_load_estimate",
                "prompt_mean",
                "pred_output_mean",
                "pred_output_p95",
                "resource_kv_capacity",
                "resource_token_budget",
                "p10_slack",
                "tight_slo_fraction",
                "burstiness_cv",
            ]:
                value = record.features.get(name)
                if value is not None:
                    features[name].append(value)
        report[policy] = {
            "windows_won": len(items),
            "bottleneck_counts": dict(bottlenecks),
            "discriminativeness_counts": dict(classes),
            "feature_region_summary": {
                name: _feature_summary(vals)
                for name, vals in features.items()
            },
        }
    return report


def _feature_summary(values: list[float]) -> dict:
    vals = sorted(values)
    if not vals:
        return {}
    return {
        "min": vals[0],
        "p50": vals[len(vals) // 2],
        "p90": vals[int(0.9 * (len(vals) - 1))],
        "max": vals[-1],
    }


def run_adaptive_search(args, policies: List[str]) -> tuple[list[tuple], list[dict]]:
    default_gpus = [GPUConfig(0, 16, 16, 16000)]
    default_service = ServiceModel()
    search_policies = args.search_policies or [
        "vllm_faithful",
        "sarathi_faithful",
        "scorpio_style_slo_guard",
        "weighted_shortest_processing",
        "edf",
        "admission_control",
    ]
    candidates = []
    candidates.extend(representative_easy_specs())
    candidates.extend(local_real_trace_stress_specs(ROOT, max_requests=args.max_real_requests))
    candidates.extend(bottleneck_taxonomy_specs())
    if args.include_counterexamples:
        candidates.extend(targeted_counterexample_specs(args.search_seed + 17, count_per_target=args.counterexamples_per_target))
    candidates.extend(sampled_bottleneck_specs(args.search_seed, count=args.sampled_candidates))

    retained: list[tuple] = []
    trace: list[dict] = []
    winner_counts: Counter[str] = Counter()
    strong_winner_counts: Counter[str] = Counter()
    representative_windows = 0
    discriminative_windows = 0
    target_windows = args.target_windows

    for spec in candidates:
        for seed in args.seeds:
            if args.verbose:
                print(f"search trial {spec.family_id} seed={seed}", flush=True)
            trial_records = build_selector_dataset_v2(
                scenario_specs=[spec],
                seeds=[seed],
                gpu_configs=default_gpus,
                candidate_policies=search_policies,
                service_model=default_service,
                window_size=args.window_size,
                drain_steps=args.search_drain_steps,
                topology_class=args.topology_class,
                verbose=False,
            )
            summary = summarize_trial(spec, seed, trial_records)
            if args.diversity_aware_retention:
                pool, reason = diversity_aware_retained_pool_for_trial(
                    summary,
                    winner_counts,
                    strong_winner_counts,
                    representative_windows,
                    discriminative_windows,
                    target_policies=set(args.target_policies),
                    max_representative_fraction=args.max_representative_fraction,
                    max_single_strong_winner_share=args.max_search_winner_share,
                )
            else:
                pool, reason = retained_pool_for_trial(
                    summary,
                    winner_counts,
                    representative_windows,
                    discriminative_windows,
                    max_representative_fraction=args.max_representative_fraction,
                )
            kept_summary = attach_retention(summary, pool, reason)
            trace.append(kept_summary.to_dict())
            if pool is None:
                if args.verbose:
                    print(f"  skip {reason}: {summary.class_counts}", flush=True)
                continue
            if pool == DISCRIMINATIVE_POOL and not _adds_winner_diversity(summary, winner_counts):
                primary_winner, _count = Counter(summary.winner_counts).most_common(1)[0]
                retained_disc_wins = sum(winner_counts.values())
                if retained_disc_wins and (
                    winner_counts[primary_winner] / retained_disc_wins
                ) >= args.max_search_winner_share:
                    kept_summary = attach_retention(summary, None, "skipped_dominant_redundant_winner")
                    trace[-1] = kept_summary.to_dict()
                    if args.verbose:
                        print(f"  skip dominant {primary_winner}: {summary.class_counts}", flush=True)
                    continue
            retained_spec = spec_with_retained_pool(spec, pool)
            retained.append((retained_spec, seed))
            for policy, count in summary.winner_counts.items():
                winner_counts[policy] += count
            strong_trial_counts = _strong_winners_for_records(trial_records)
            for policy, count in strong_trial_counts.items():
                strong_winner_counts[policy] += count
            if pool == REPRESENTATIVE_POOL:
                representative_windows += summary.num_windows
            else:
                discriminative_windows += summary.num_windows
            if args.verbose:
                print(
                    f"  keep {pool}: windows={summary.num_windows} "
                    f"rep={representative_windows} disc={discriminative_windows} "
                    f"winners={dict(summary.winner_counts)} classes={summary.class_counts}",
                    flush=True,
                )
            if representative_windows + discriminative_windows >= target_windows:
                return retained, trace
    return retained, trace


def _adds_winner_diversity(summary, winner_counts: Counter[str]) -> bool:
    return any(winner_counts.get(policy, 0) == 0 for policy in summary.winner_counts)


def _strong_winners_for_records(records: List[WindowRecordV2]) -> Counter[str]:
    counts = Counter()
    for record in records:
        disc = _wg_disc(record)
        if disc.classification == "STRONGLY_DISCRIMINATIVE":
            counts[disc.best_policy] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/selector_dataset_v2/redesigned_pilot")
    parser.add_argument("--topology-class", default="monolithic")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 17])
    parser.add_argument("--search-seed", type=int, default=20260718)
    parser.add_argument("--sampled-candidates", type=int, default=56)
    parser.add_argument("--include-counterexamples", action="store_true", default=True)
    parser.add_argument("--no-counterexamples", dest="include_counterexamples", action="store_false")
    parser.add_argument("--counterexamples-per-target", type=int, default=36)
    parser.add_argument("--diversity-aware-retention", action="store_true", default=True)
    parser.add_argument("--legacy-retention", dest="diversity_aware_retention", action="store_false")
    parser.add_argument(
        "--target-policies",
        nargs="+",
        default=[
            "sarathi_faithful",
            "vllm_faithful",
            "edf",
            "slo_slack_score",
            "admission_control",
            "weighted_shortest_processing",
            "estimated_service_time_first",
        ],
    )
    parser.add_argument("--target-windows", type=int, default=720)
    parser.add_argument("--max-real-requests", type=int, default=144)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--drain-steps", type=int, default=30_000)
    parser.add_argument("--search-drain-steps", type=int, default=5_000)
    parser.add_argument("--max-representative-fraction", type=float, default=0.30)
    parser.add_argument("--max-search-winner-share", type=float, default=0.45)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--search-policies", nargs="*", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    policies = args.policies or candidate_policies_for_topology(args.topology_class)
    default_gpus = [GPUConfig(0, 16, 16, 16000)]
    default_service = ServiceModel()
    t0 = time.perf_counter()

    retained_trials, search_trace = run_adaptive_search(args, policies)
    records = build_selector_dataset_v2_trials(
        trials=retained_trials,
        gpu_configs=default_gpus,
        candidate_policies=policies,
        service_model=default_service,
        window_size=args.window_size,
        drain_steps=args.drain_steps,
        topology_class=args.topology_class,
        verbose=args.verbose,
    )

    ancestors = {record.identifiers.request_plan_ancestor_id for record in records}
    ood = (
        {"real_trace__azure_2023_conv"}
        if "real_trace__azure_2023_conv" in ancestors
        else {"real_trace__azure_2023_code"} if "real_trace__azure_2023_code" in ancestors else set()
    )
    rows = assemble_dataset_rows(
        records,
        group_key_field="request_plan_ancestor_id",
        ood_scenario_family_ids=ood,
    )
    verify_group_atomicity(rows, "request_plan_ancestor_id")
    if ood:
        verify_ood_holdout(rows, "request_plan_ancestor_id", ood)

    summary = summarize_records(records, policies)
    specialization = specialization_report(records)
    elapsed = time.perf_counter() - t0
    split_counts = Counter(row["split"] for row in rows)
    quality_gates = {
        "all_complete_fraction_below_40pct": summary["all_complete_fraction"] < 0.40,
        "moderate_or_strong_fraction_substantially_increased": (
            summary["moderately_discriminative_fraction"] + summary["strongly_discriminative_fraction"]
        ) >= 0.35,
        "at_least_3_policies_win_discriminative_windows": len(summary["discriminative_policy_win_distribution"]) >= 3,
        "single_policy_strong_window_dominance_lte_85pct": summary["strong_window_top_policy_share"] <= 0.85,
        "nontrivial_oracle_headroom": (summary["oracle_headroom"] or 0.0) >= 0.01,
        "nontrivial_discriminative_oracle_headroom": (summary["discriminative_oracle_headroom"] or 0.0) >= 0.02,
        "faithful_external_baseline_wins": sum(summary["faithful_baseline_discriminative_wins"].values()) >= 10,
        "real_trace_representation": any(r.identifiers.dataset_family == "real_trace" for r in records),
        "ood_split_defined": bool(ood),
    }

    manifest = DatasetManifestV2(
        dataset_name="selector_dataset_v2_redesigned_pilot",
        schema_version="2.1",
        topology_class=args.topology_class,
        candidate_policies=policies,
        feature_names=sorted(k.removeprefix("feat_") for k in rows[0] if k.startswith("feat_")) if rows else [],
        objectives=[o.name for o in STANDARD_OBJECTIVES],
        num_scenarios=len({r.identifiers.scenario_id for r in records}),
        num_windows=len(records),
        num_policy_evaluations=len(rows),
        scenario_family_ids=sorted({r.identifiers.scenario_family_id for r in records}),
        source_traces=sorted({r.identifiers.source_trace for r in records}),
        seeds=sorted({r.identifiers.seed for r in records}),
        split_group_key="request_plan_ancestor_id",
        split_counts=dict(split_counts),
        quality_gate_results=quality_gates,
        generation_config={
            "window_size": args.window_size,
            "target_windows": args.target_windows,
            "sampled_candidates": args.sampled_candidates,
            "search_seed": args.search_seed,
            "drain_steps": args.drain_steps,
            "elapsed_seconds": round(elapsed, 3),
        },
        notes=[
            "Redesigned CPU-only pilot; not a final selector-training dataset.",
            "Adaptive search retains representative and discriminative pools separately.",
            "No RF/DT final selector is trained by this script.",
        ],
    )

    _write_csv(rows, out_dir / "selector_dataset_v2_redesigned_pilot.csv")
    _write_json(manifest.to_dict(), out_dir / "manifest.json")
    _write_json(summary, out_dir / "pilot_summary.json")
    _write_json(search_trace, out_dir / "search_trace.json")
    _write_json(specialization, out_dir / "policy_specialization.json")
    print(json.dumps({
        "output_dir": str(out_dir),
        "elapsed_seconds": round(elapsed, 3),
        "retained_trials": len(retained_trials),
        "windows": len(records),
        "policy_evaluations": len(rows),
        **summary,
        "quality_gates": quality_gates,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
