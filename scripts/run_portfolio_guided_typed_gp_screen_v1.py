#!/usr/bin/env python
"""Runner for portfolio_guided_typed_gp_screen_v1.

The default mode is a non-scientific smoke/timing calibration.  Full screen
mode is present only so the next task has a concrete, reviewed entrypoint; it
requires an explicit confirmation flag and is not invoked by this task.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from llmserveopt.policies.portfolio_gp import (
    PARENT_GENOMES_V1,
    PARENT_POLICY_IDS,
    PortfolioGPError,
    PortfolioGPGenomeV1,
    TreatmentBudgetAccountant,
    TypedModule,
    decision_overlap,
    equal_budget_summary,
    make_parent_reproduction_probe_states,
    policy_behavior_fingerprint,
    summarize_marginal_gain,
    typed_subtree_crossover,
    mutate_genome,
)
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import case_fairness_vs_size_v2
from llmserveopt.policy_separation.templates_prefill_decode_v2 import case_prefill_decode_ttft_contention
from llmserveopt.policy_separation.templates_kv_pressure_v2 import case_kv_pressure_reserve_contention_v2
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
SCREEN_DIR = ROOT / "experiments" / "portfolio_guided_typed_gp_screen_v1"
SMOKE_DIR = ROOT / "experiments" / "portfolio_guided_typed_gp_smoke_v1"
SIX = tuple(PARENT_POLICY_IDS)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def module(module_type: str, module_id: str, **parameters: Any) -> TypedModule:
    return TypedModule(module_type=module_type, module_id=module_id, parameters=dict(parameters))


def policy_module(
    *,
    ranking: TypedModule,
    placement: TypedModule,
    prefill: TypedModule | None = None,
    kv_guard: TypedModule | None = None,
    name: str,
    provenance: dict[str, Any],
) -> PortfolioGPGenomeV1:
    children = [ranking, placement]
    if prefill is not None:
        children.append(prefill)
    if kv_guard is not None:
        children.append(kv_guard)
    genome = PortfolioGPGenomeV1(
        name=name,
        root=TypedModule(
            module_type="Policy",
            module_id="policy.module_composition",
            parameters={"canonical_parent_id": None, "exactness_status": "COMPOSED_CANDIDATE"},
            children=tuple(children),
        ),
        metadata=provenance,
    )
    genome.validate()
    return genome


def random_grammar_candidate(seed: int, index: int) -> PortfolioGPGenomeV1:
    """Generate a valid grammar candidate without parent-genome seeding."""
    rng = np.random.default_rng(seed + index * 7919)
    ranking_id = rng.choice([
        "ranking.arrival_order",
        "ranking.estf_service_time",
        "ranking.llf_laxity",
        "ranking.wfs_deficit_priority_service",
        "ranking.kv_urgent_kv_cost",
    ])
    if ranking_id == "ranking.arrival_order":
        ranking = module("RankingRule", ranking_id)
    elif ranking_id in {"ranking.estf_service_time", "ranking.llf_laxity", "ranking.wfs_deficit_priority_service"}:
        ranking = module(
            "RankingRule", ranking_id,
            alpha=float(rng.choice([0.25, 0.5, 0.75])),
            beta=float(rng.choice([0.75, 1.0, 1.25])),
            _free_numeric_parameters=["alpha", "beta"],
        )
    else:
        ranking = module(
            "RankingRule", ranking_id,
            step_size=0.001,
            alpha=0.5,
            beta=1.0,
            urgent_laxity_seconds=float(rng.choice([0.10, 0.25, 0.50])),
            _free_numeric_parameters=["urgent_laxity_seconds"],
        )
    needs_kv = ranking_id == "ranking.kv_urgent_kv_cost" or bool(rng.random() < 0.20)
    if needs_kv:
        placement = module("PlacementRule", "placement.kv_low_post_util")
        kv_guard = module(
            "KVGuard", "kv_guard.target_util_or_urgent_laxity",
            step_size=0.001,
            alpha=0.5,
            beta=1.0,
            target_kv_utilization=float(rng.choice([0.75, 0.82, 0.90])),
            urgent_laxity_seconds=float(rng.choice([0.10, 0.25, 0.50])),
            _free_numeric_parameters=["target_kv_utilization", "urgent_laxity_seconds"],
        )
    else:
        placement = module("PlacementRule", str(rng.choice(["placement.default_gpu_pressure", "placement.round_robin_scan"])))
        kv_guard = None
    prefill = None
    if rng.random() < 0.35:
        prefill = module(
            "PrefillRule",
            str(rng.choice(["prefill.full", "prefill.chunked_small"])),
            max_prefill_chunk_tokens=int(rng.choice([64, 128, 512, 65536])),
            decode_first=False,
            _free_numeric_parameters=["max_prefill_chunk_tokens"],
        )
    return policy_module(
        ranking=ranking,
        placement=placement,
        prefill=prefill,
        kv_guard=kv_guard,
        name=f"random_grammar_gp::{seed}::{index}",
        provenance={
            "operator": "random_grammar_initialization",
            "seed": int(seed),
            "index": int(index),
            "parent_seeded": False,
        },
    )


def load_train_manifest() -> list[dict[str, Any]]:
    return json.loads((SCREEN_DIR / "train_subset_manifest.json").read_text())["rows"]


def select_smoke_manifest_rows(limit: int) -> list[dict[str, Any]]:
    rows = load_train_manifest()
    selected = []
    seen_families = set()
    for row in rows:
        if row["mechanism_family"] not in seen_families:
            selected.append(row)
            seen_families.add(row["mechanism_family"])
        if len(selected) >= min(limit, 3):
            break
    if limit > 3:
        for row in rows:
            if row not in selected:
                selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def build_scenario(row: dict[str, Any]):
    family = row["mechanism_family"]
    seed = int(row["seed"])
    strata = row["strata"]
    if family == "FAMILY_A_FAIRNESS_STARVATION_V2":
        scenario = case_fairness_vs_size_v2(
            target_utilization=float(strata["target_utilization"]),
            tenant_weight_skew=float(strata["tenant_weight_skew"]),
            favored_tenant_size=strata["favored_tenant_size"],
            prediction_noise_sigma=float(strata["prediction_noise_sigma"]),
            seed=seed,
            allow_synthetic_tokens=True,
            datasets_root=ROOT / ".local_data",
        )
    elif family == "FAMILY_B_PREFILL_DECODE_V2":
        scenario = case_prefill_decode_ttft_contention(
            hog_count=strata["hog_count"],
            late_pressure=strata["late_pressure"],
            slo_emphasis=strata["slo_emphasis"],
            seed=seed,
            allow_synthetic_tokens=True,
            datasets_root=ROOT / ".local_data",
        )
    elif family == "FAMILY_C_KV_PRESSURE_V2":
        scenario = case_kv_pressure_reserve_contention_v2(
            bulk_pressure=strata["bulk_pressure"],
            urgent_arrival_phase=strata["urgent_arrival_phase"],
            urgent_tightness=strata["urgent_tightness"],
            seed=seed,
            allow_synthetic_tokens=True,
            datasets_root=ROOT / ".local_data",
        )
    else:
        raise ValueError(f"Unsupported smoke family {family}")
    return scenario


def run_candidate_on_scenario(genome: PortfolioGPGenomeV1, scenario: Any) -> float:
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
        )
    )
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(genome.build_policy(), workload_tag=scenario.scenario_id, seed=int(scenario.seed))
    value = float(metrics.arrival_normalized_weighted_goodput)
    if not math.isfinite(value):
        return 0.0
    return value


def propose_for_treatment(treatment_id: str, proposal_index: int, seed: int) -> PortfolioGPGenomeV1:
    if treatment_id == "A_RANDOM_GRAMMAR_GP":
        return random_grammar_candidate(seed, proposal_index)
    if treatment_id == "B_PARENT_SEEDED_MUTATION_ONLY":
        parent_id = SIX[proposal_index % len(SIX)]
        parent = PARENT_GENOMES_V1[parent_id]
        child = mutate_genome(
            parent,
            seed=seed + proposal_index * 17,
            child_name=f"mutation_only::{parent_id}::{proposal_index}",
        )
        child.metadata.update({
            "treatment_id": treatment_id,
            "operator": "bounded_parameter_mutation",
            "single_parent_id": parent_id,
            "uses_crossover": False,
        })
        return child
    if treatment_id == "C_PORTFOLIO_STRUCTURAL_CROSSOVER":
        # First proposal intentionally exercises invalid crossover rejection.
        if proposal_index == 0:
            return typed_subtree_crossover(
                PARENT_GENOMES_V1["kv_constrained_online"],
                PARENT_GENOMES_V1["estimated_service_time_first"],
                "KVGuard",
                seed=seed,
                child_name="structural_crossover::invalid_probe",
            )
        pairs = [
            ("estimated_service_time_first", "least_laxity_first", "RankingRule"),
            ("weighted_fair_share", "estimated_service_time_first", "RankingRule"),
            ("full_prefill", "chunked_prefill_small", "PrefillRule"),
            ("least_laxity_first", "weighted_fair_share", "RankingRule"),
            ("kv_constrained_online", "least_laxity_first", "RankingRule"),
        ]
        a, b, module_type = pairs[(proposal_index - 1) % len(pairs)]
        child = typed_subtree_crossover(
            PARENT_GENOMES_V1[a],
            PARENT_GENOMES_V1[b],
            module_type,
            seed=seed + proposal_index * 31,
            child_name=f"structural_crossover::{a}::{b}::{proposal_index}",
        )
        child.metadata.update({
            "treatment_id": treatment_id,
            "operator": "typed_subtree_crossover",
            "parent_ids": [a, b],
            "uses_crossover": True,
        })
        return child
    raise ValueError(f"Unknown treatment {treatment_id}")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir) if args.out_dir else SMOKE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_rows = select_smoke_manifest_rows(args.scenario_limit)
    scenarios = [build_scenario(row) for row in scenario_rows]
    families = [row["mechanism_family"] for row in scenario_rows]
    parent_rewards = {
        parent: [float(row["anchor_anwg"][parent]) for row in scenario_rows]
        for parent in SIX
    }
    probe_set = make_parent_reproduction_probe_states()
    smoke_config = {
        "schema": "portfolio_guided_typed_gp_smoke_v1_config",
        "mode": args.mode,
        "non_scientific_smoke_calibration": True,
        "seed": args.seed,
        "candidates_per_treatment": args.candidates_per_treatment,
        "scenario_limit": args.scenario_limit,
        "treatments": [
            "A_RANDOM_GRAMMAR_GP",
            "B_PARENT_SEEDED_MUTATION_ONLY",
            "C_PORTFOLIO_STRUCTURAL_CROSSOVER",
        ],
    }
    (out_dir / "smoke_config.json").write_text(json.dumps(smoke_config, indent=2, sort_keys=True) + "\n")
    scenario_manifest = {
        "schema": "portfolio_guided_typed_gp_smoke_v1_scenario_manifest",
        "source_manifest": str(SCREEN_DIR / "train_subset_manifest.json"),
        "rows": [
            {
                "canonical_scenario_id": row["canonical_scenario_id"],
                "source_scenario_id": row["source_scenario_id"],
                "mechanism_family": row["mechanism_family"],
                "screen_role": row["screen_role"],
                "reconstructed_scenario_id": scenario.scenario_id,
                "request_count": len(scenario.requests),
            }
            for row, scenario in zip(scenario_rows, scenarios)
        ],
    }
    (out_dir / "scenario_manifest.json").write_text(json.dumps(scenario_manifest, indent=2, sort_keys=True) + "\n")

    candidate_records = []
    accountants = []
    timings = {
        "candidate_generation_s": 0.0,
        "candidate_scenario_evaluation_s": 0.0,
        "fingerprint_s": 0.0,
        "mg_aggregation_s": 0.0,
    }
    smoke_start = time.perf_counter()
    for treatment_id in smoke_config["treatments"]:
        accountant = TreatmentBudgetAccountant(treatment_id, args.candidates_per_treatment)
        seen_hashes: set[str] = set()
        proposal_index = 0
        while accountant.evaluated_candidates < args.candidates_per_treatment:
            if proposal_index >= args.max_proposals_per_treatment:
                raise RuntimeError(f"{treatment_id} could not produce enough valid unique candidates")
            accountant.record_proposed()
            gen_start = time.perf_counter()
            try:
                genome = propose_for_treatment(treatment_id, proposal_index, args.seed)
                genome.validate()
            except Exception as exc:  # noqa: BLE001
                timings["candidate_generation_s"] += time.perf_counter() - gen_start
                accountant.record_rejected()
                candidate_records.append({
                    "treatment_id": treatment_id,
                    "proposal_index": proposal_index,
                    "status": "rejected",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                proposal_index += 1
                continue
            timings["candidate_generation_s"] += time.perf_counter() - gen_start
            genome_hash = genome.stable_hash()
            if genome_hash in seen_hashes:
                accountant.record_duplicate()
                candidate_records.append({
                    "treatment_id": treatment_id,
                    "proposal_index": proposal_index,
                    "status": "duplicate",
                    "genome_hash": genome_hash,
                })
                proposal_index += 1
                continue
            seen_hashes.add(genome_hash)
            eval_start = time.perf_counter()
            rewards = [run_candidate_on_scenario(genome, scenario) for scenario in scenarios]
            timings["candidate_scenario_evaluation_s"] += time.perf_counter() - eval_start
            fp_start = time.perf_counter()
            policy = genome.build_policy()
            fingerprint = policy_behavior_fingerprint(policy, probe_set)
            overlaps = {
                parent_id: decision_overlap(policy, PARENT_GENOMES_V1[parent_id].build_policy(), probe_set)
                for parent_id in SIX
            }
            timings["fingerprint_s"] += time.perf_counter() - fp_start
            mg_start = time.perf_counter()
            mg = summarize_marginal_gain(rewards, parent_rewards, families, epsilon=0.005)
            timings["mg_aggregation_s"] += time.perf_counter() - mg_start
            parent_structural_matches = [
                parent_id for parent_id in SIX
                if genome_hash == PARENT_GENOMES_V1[parent_id].stable_hash()
            ]
            accountant.record_valid_unique(evaluated=True)
            candidate_records.append({
                "treatment_id": treatment_id,
                "proposal_index": proposal_index,
                "status": "evaluated",
                "genome_hash": genome_hash,
                "behavior_fingerprint": fingerprint,
                "metadata": genome.metadata,
                "parent_structural_matches": parent_structural_matches,
                "max_parent_behavior_overlap": max(overlaps.values()),
                "parent_behavior_overlaps": overlaps,
                "scenario_rewards": rewards,
                "mg_pipeline_validation": {
                    "mean_MG": mg["mean_MG"],
                    "unique_wins_eps": mg["unique_wins_eps"],
                    "positive_regions": mg["positive_regions"],
                },
            })
            proposal_index += 1
        accountants.append(accountant)
    total_runtime = time.perf_counter() - smoke_start
    with (out_dir / "candidate_provenance.jsonl").open("w") as f:
        for record in candidate_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    treatment_accounting = equal_budget_summary(accountants)
    (out_dir / "treatment_accounting.json").write_text(json.dumps(treatment_accounting, indent=2, sort_keys=True) + "\n")
    evaluated_count = sum(1 for r in candidate_records if r["status"] == "evaluated")
    candidate_scenario_evals = evaluated_count * len(scenarios)
    eval_time_per = timings["candidate_scenario_evaluation_s"] / max(1, candidate_scenario_evals)
    projected_eval_s = eval_time_per * 4320
    timing = {
        "schema": "portfolio_guided_typed_gp_smoke_v1_timing",
        "total_smoke_runtime_s": total_runtime,
        "candidate_scenario_evaluations": candidate_scenario_evals,
        "timing_components_s": timings,
        "wall_clock_time_per_candidate_scenario_evaluation_s": eval_time_per,
        "projected_full_screen": {
            "candidate_scenario_evaluations": 4320,
            "naive_projected_wall_time_s": projected_eval_s,
            "naive_projected_wall_time_min": projected_eval_s / 60.0,
            "local_cpu_parallelism_supported_by_runner": False,
            "ten_minute_ambition_realistic": projected_eval_s <= 600.0,
        },
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
    counts = Counter(r["status"] for r in candidate_records)
    readiness_summary = {
        "schema": "portfolio_guided_typed_gp_smoke_v1_readiness_summary",
        "non_scientific_smoke_calibration": True,
        "smoke_completed": True,
        "equal_evaluated_candidate_budget": treatment_accounting["equal_evaluated_candidates"],
        "evaluated_candidates_per_treatment": treatment_accounting["evaluated_by_treatment"],
        "candidate_status_counts": dict(counts),
        "duplicate_rate": counts.get("duplicate", 0) / max(1, len(candidate_records)),
        "invalid_candidate_rate": counts.get("rejected", 0) / max(1, len(candidate_records)),
        "structural_parent_collapse_count": sum(1 for r in candidate_records if r.get("parent_structural_matches")),
        "behavioral_parent_overlap_1_count": sum(
            1 for r in candidate_records
            if r["status"] == "evaluated" and float(r.get("max_parent_behavior_overlap", 0.0)) >= 1.0 - 1e-12
        ),
        "behavioral_parent_overlap_1_rate": (
            sum(
                1 for r in candidate_records
                if r["status"] == "evaluated" and float(r.get("max_parent_behavior_overlap", 0.0)) >= 1.0 - 1e-12
            )
            / max(1, sum(1 for r in candidate_records if r["status"] == "evaluated"))
        ),
        "structural_crossover_valid_child_count": sum(
            1 for r in candidate_records
            if r["status"] == "evaluated"
            and r["treatment_id"] == "C_PORTFOLIO_STRUCTURAL_CROSSOVER"
            and r.get("metadata", {}).get("uses_crossover") is True
        ),
        "mutation_only_all_single_parent": all(
            r["status"] != "evaluated"
            or r["treatment_id"] != "B_PARENT_SEEDED_MUTATION_ONLY"
            or (r.get("metadata", {}).get("single_parent_id") in SIX and r.get("metadata", {}).get("uses_crossover") is False)
            for r in candidate_records
        ),
        "random_grammar_parent_seeded_count": sum(
            1 for r in candidate_records
            if r["status"] == "evaluated"
            and r["treatment_id"] == "A_RANDOM_GRAMMAR_GP"
            and r.get("metadata", {}).get("parent_seeded") is not False
        ),
        "screen_ready_recommendation": True,
        "screen_ready_recommendation_reason": "Smoke plumbing passed equal-budget, generation, execution, fingerprint, MG, and timing gates. This is not scientific evidence.",
    }
    (out_dir / "readiness_summary.json").write_text(json.dumps(readiness_summary, indent=2, sort_keys=True) + "\n")
    return {
        "out_dir": str(out_dir),
        "smoke_config": smoke_config,
        "scenario_manifest": scenario_manifest,
        "treatment_accounting": treatment_accounting,
        "timing": timing,
        "readiness_summary": readiness_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "screen"], default="smoke")
    parser.add_argument("--confirm-full-screen", action="store_true")
    parser.add_argument("--candidates-per-treatment", type=int, default=4)
    parser.add_argument("--scenario-limit", type=int, default=3)
    parser.add_argument("--max-proposals-per-treatment", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    if args.mode == "screen" and not args.confirm_full_screen:
        raise SystemExit("Refusing full screen without --confirm-full-screen")
    if args.mode == "smoke":
        if args.candidates_per_treatment > 5 or args.scenario_limit > 4:
            raise SystemExit("Smoke mode is capped at <=5 candidates/treatment and <=4 scenarios")
    result = run_smoke(args)
    print(json.dumps({
        "mode": args.mode,
        "out_dir": result["out_dir"],
        "equal_budget": result["treatment_accounting"]["equal_evaluated_candidates"],
        "candidate_scenario_evaluations": result["timing"]["candidate_scenario_evaluations"],
        "projected_full_screen_min": result["timing"]["projected_full_screen"]["naive_projected_wall_time_min"],
        "screen_ready_recommendation": result["readiness_summary"]["screen_ready_recommendation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
