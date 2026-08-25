#!/usr/bin/env python3
"""First comparative evaluation of the PARS offline-scored baseline,
following the same protocol as
``scripts/run_vllm_ltr_first_comparative_evaluation.py`` (imports and
reuses that script's ``SelectorDispatchPolicy``, ``load_selector_artifact``,
and ``compute_bootstrap_ci`` rather than duplicating them), extended to run
across multiple workloads: the real WildChat-1M control AND the 7 accepted
synthetic families from the canonical benchmark suite
(``benchmarks/canonical_suite/``, see
``docs/audits/canonical_benchmark_suite_design_20260804.md``).

Policies compared (identical set to the vLLM-LTR evaluation, with
``pars_semantic_reference`` in place of ``vllm_ltr_semantic_reference``):
  1. fifo                         6. rule_based_selector (hard selector)
  2. edf                          7. scorpio_style_slo_guard (best fixed)
  3. estimated_service_time_first 8. regression_anwg_selector (best global composition)
  4. shortest_output_first        9. pars_semantic_reference (offline-scored, evaluation-only)
  5. weighted_shortest_processing 10. oracle_srtf (non-deployable hindsight ceiling)

PARS is evaluation-only for this run: not registered in any policy
registry, selector-candidate list, or CC4/CC5 training data (see
baselines/pars/adapter/simulator_policy.py's own SELECTOR_ELIGIBLE = False).

Each canonical-suite family runs under its OWN gpu/service_model
configuration (recorded in that family's manifest.json at generation time,
e.g. max_active_sequences=2 for most families vs. 8 for the WildChat
control) -- not the WildChat-tuned defaults, so each workload's own
carefully-calibrated headroom (see the canonical-suite design doc) is
preserved rather than flattened.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from baselines.pars.adapter.offline_scoring import load_score_cache, scores_only
from baselines.pars.adapter.simulator_policy import PARSSemanticReferencePolicy
from llmserveopt.core.metrics import RunMetrics, metrics_to_dict
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.oracle import build_oracle
from llmserveopt.policies.registry import make_policy
from llmserveopt.policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA, predicted_service_proxy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from llmserveopt.workloads.sharegpt import ShareGPTConversionConfig, convert_sharegpt_to_requests

# Reuse (do not duplicate) the vLLM-LTR eval script's selector-dispatch
# plumbing -- identical objective/artifact-validation logic must be used
# for both baselines' comparisons to be methodologically comparable.
_VLLM_LTR_EVAL_SPEC = importlib.util.spec_from_file_location(
    "run_vllm_ltr_first_comparative_evaluation",
    Path(__file__).parent / "run_vllm_ltr_first_comparative_evaluation.py",
)
_vllm_ltr_eval = importlib.util.module_from_spec(_VLLM_LTR_EVAL_SPEC)
sys.modules["run_vllm_ltr_first_comparative_evaluation"] = _vllm_ltr_eval
_VLLM_LTR_EVAL_SPEC.loader.exec_module(_vllm_ltr_eval)

SelectorDispatchPolicy = _vllm_ltr_eval.SelectorDispatchPolicy
load_selector_artifact = _vllm_ltr_eval.load_selector_artifact
compute_bootstrap_ci = _vllm_ltr_eval.compute_bootstrap_ci
SelectorArtifactError = _vllm_ltr_eval.SelectorArtifactError

FIXED_POLICIES = ["fifo", "edf", "estimated_service_time_first", "shortest_output_first", "weighted_shortest_processing"]
BEST_FIXED_POLICY = "scorpio_style_slo_guard"

# WildChat control: identical config to the vLLM-LTR evaluation, for
# comparability with that already-completed run.
WILDCHAT_GPU_CONFIGS = [GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=8192, max_kv_tokens=131072)]
WILDCHAT_SERVICE_MODEL = ServiceModel(
    step_size=0.001, enable_prefill_modeling=True, prefill_cost_per_token=1.0,
    max_prefill_chunk_tokens=512, step_token_budget=8192, decode_first=False,
)
DRAIN_STEPS = 200_000
CANONICAL_SUITE_DIR = "benchmarks/canonical_suite"


@dataclass
class SeedRunResult:
    metrics: RunMetrics
    rows: List[dict]


def run_policy_with_rows(policy: BasePolicy, requests: List[Request], workload_tag: str,
                          seed: int, sim_cfg: SimulatorConfig) -> SeedRunResult:
    sim = Simulator(sim_cfg)
    sim.load_trace(list(requests))
    policy.reset()
    metrics = sim.run(policy, workload_tag=workload_tag, seed=seed)

    completed_by_id = {c.request.request_id: c for c in sim._completed}
    rows = []
    for r in requests:
        c = completed_by_id.get(r.request_id)
        if c is not None:
            rows.append({"request_id": r.request_id, "seed": seed, "priority": r.priority,
                         "class_id": r.class_id, "status": "success", "slo_violated": bool(c.slo_violated)})
        else:
            rows.append({"request_id": r.request_id, "seed": seed, "priority": r.priority,
                         "class_id": r.class_id, "status": "dropped", "slo_violated": True})
    return SeedRunResult(metrics=metrics, rows=rows)


def _rank_correlation(a: List[float], b: List[float]) -> float:
    a_ranks = np.argsort(np.argsort(a)).astype(float)
    b_ranks = np.argsort(np.argsort(b)).astype(float)
    if np.std(a_ranks) == 0 or np.std(b_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(a_ranks, b_ranks)[0, 1])


def compute_ranking_agreement_record(seed: int, requests: List[Request], pars_scores: Dict[int, float]) -> dict:
    req_by_id = {r.request_id: r for r in requests}
    ids_sorted = sorted(req_by_id.keys())
    # pars_scores: HIGHER = predicted LONGER response -- negate so
    # "higher = higher scheduling priority" for a consistent sign
    # convention against est_order/sof_order below (which are already
    # negated to that same convention).
    pars_order = [-pars_scores[i] for i in ids_sorted]
    est_order = [-predicted_service_proxy(req_by_id[i], alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA) for i in ids_sorted]
    sof_order = [-req_by_id[i].predicted_output_tokens for i in ids_sorted]
    return {
        "seed": seed,
        "spearman_pars_vs_estimated_service_time_first": _rank_correlation(pars_order, est_order),
        "spearman_pars_vs_shortest_output_first": _rank_correlation(pars_order, sof_order),
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _wildchat_requests_for_seed(pairs_path: str, seed: int, tokenizer_name: str, max_requests: Optional[int]) -> List[Request]:
    from llmserveopt.workloads.sharegpt import load_sharegpt_raw
    from llmserveopt.workloads.augmentation import AugmentationConfig

    records = load_sharegpt_raw(pairs_path)
    config = ShareGPTConversionConfig(arrival_mode="poisson", arrival_rate=10.0,
                                       tokenizer_name=tokenizer_name, fallback_whitespace=False)
    requests, report = convert_sharegpt_to_requests(records, config=config, seed=seed, aug_config=AugmentationConfig())
    if report.rows_retained != len(records):
        warnings.warn(f"seed={seed}: {report.rows_retained}/{len(records)} rows retained after filtering.")
    if max_requests is not None:
        requests = sorted(requests, key=lambda r: r.request_id)[:max_requests]
    return requests


def _canonical_requests_for_seed(family: str, seed: int) -> Tuple[List[Request], dict, dict]:
    family_dir = os.path.join(CANONICAL_SUITE_DIR, family)
    with open(os.path.join(family_dir, f"seed_{seed}.json")) as f:
        rows = json.load(f)
    requests = [
        Request(
            request_id=r["request_id"], arrival_time=r["arrival_time"], prompt_tokens=r["prompt_tokens"],
            predicted_output_tokens=r["predicted_output_tokens"], actual_output_tokens=r["actual_output_tokens"],
            slo_deadline=r["slo_deadline"], priority=r["priority"], class_id=r["class_id"],
        )
        for r in rows
    ]
    with open(os.path.join(family_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return requests, manifest["gpu"], manifest["service_model"]


def _build_sim_config(gpu_overrides: dict, service_model_overrides: dict) -> SimulatorConfig:
    gpu_cfg = dict(max_active_sequences=8, max_batch_tokens=8192, max_kv_tokens=131072)
    gpu_cfg.update(gpu_overrides or {})
    sm_cfg = dict(step_size=0.001, enable_prefill_modeling=True, prefill_cost_per_token=1.0,
                  max_prefill_chunk_tokens=512, step_token_budget=8192, decode_first=False)
    sm_cfg.update(service_model_overrides or {})
    return SimulatorConfig(gpu_configs=[GPUConfig(gpu_id=0, **gpu_cfg)], service_model=ServiceModel(**sm_cfg), drain_steps=DRAIN_STEPS)


def run_one_workload(
    workload_name: str,
    requests_by_seed: Dict[int, List[Request]],
    sim_cfg: SimulatorConfig,
    pars_scores_by_seed: Dict[int, Dict[int, float]],
    rule_selector, regression_selector,
    output_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    rule_selector_dispatch = SelectorDispatchPolicy(rule_selector, name="rule_based_selector")
    regression_selector_dispatch = SelectorDispatchPolicy(regression_selector, name="regression_anwg_selector")

    all_metrics: List[RunMetrics] = []
    rows_by_policy: Dict[str, List[dict]] = {}
    ranking_agreement_records = []

    for seed, requests in requests_by_seed.items():
        pars_scores = pars_scores_by_seed[seed]
        missing = [r.request_id for r in requests if r.request_id not in pars_scores]
        if missing:
            raise RuntimeError(f"{workload_name} seed={seed}: {len(missing)} requests missing PARS scores "
                                f"(first: {missing[:5]})")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            oracle = build_oracle(requests)

        rule_selector_dispatch.start_seed(seed)
        regression_selector_dispatch.start_seed(seed)

        policies: List[BasePolicy] = [make_policy(p) for p in FIXED_POLICIES]
        policies.append(make_policy(BEST_FIXED_POLICY))
        policies.append(rule_selector_dispatch)
        policies.append(regression_selector_dispatch)
        policies.append(PARSSemanticReferencePolicy(scores=pars_scores))
        policies.append(oracle)

        for policy in policies:
            print(f"  workload={workload_name} seed={seed} policy={policy.name} n_req={len(requests)}")
            result = run_policy_with_rows(policy, requests, workload_tag=workload_name, seed=seed, sim_cfg=sim_cfg)
            all_metrics.append(result.metrics)
            for row in result.rows:
                row["policy"] = policy.name
            rows_by_policy.setdefault(policy.name, []).extend(result.rows)

        ranking_agreement_records.append(compute_ranking_agreement_record(seed, requests, pars_scores))

    reference_policy = "pars_semantic_reference"
    ci_results = compute_bootstrap_ci(rows_by_policy, reference_policy=reference_policy)

    metrics_path = os.path.join(output_dir, "run_metrics.csv")
    import csv
    fieldnames = list(metrics_to_dict(all_metrics[0]).keys())
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(metrics_to_dict(m))

    outcomes_path = os.path.join(output_dir, "request_level_outcomes.csv")
    with open(outcomes_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "seed", "request_id", "priority", "class_id", "status", "slo_violated"])
        writer.writeheader()
        for rows in rows_by_policy.values():
            for row in rows:
                writer.writerow(row)

    ci_path = os.path.join(output_dir, "bootstrap_confidence_intervals.json")
    with open(ci_path, "w", encoding="utf-8") as f:
        json.dump(ci_results, f, indent=2)

    ranking_path = os.path.join(output_dir, "ranking_agreement.json")
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(ranking_agreement_records, f, indent=2)

    def _entropy_bits(counts: Dict[str, int]) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        ps = np.array([c / total for c in counts.values()], dtype=float)
        return float(-np.sum(ps * np.log2(ps)))

    def _dispatch_summary(dispatch) -> dict:
        totals: Dict[str, int] = {}
        for seed_counts in dispatch.dispatch_counts_by_seed.values():
            for pname, c in seed_counts.items():
                totals[pname] = totals.get(pname, 0) + c
        return {"dispatch_counts_by_seed": dispatch.dispatch_counts_by_seed, "dispatch_counts_total": totals,
                "num_distinct_subpolicies_dispatched": len(totals), "entropy_bits": _entropy_bits(totals)}

    behavioral_diversity = {
        "rule_based_selector": _dispatch_summary(rule_selector_dispatch),
        "regression_anwg_selector": _dispatch_summary(regression_selector_dispatch),
        "ranking_agreement": ranking_agreement_records,
    }
    diversity_path = os.path.join(output_dir, "behavioral_diversity.json")
    with open(diversity_path, "w", encoding="utf-8") as f:
        json.dump(behavioral_diversity, f, indent=2)

    completion_accounting = [
        {"policy": m.policy_name, "seed": m.seed, "num_total": m.num_total, "num_completed": m.num_completed,
         "num_dropped": m.num_dropped, "num_slo_violated": m.num_slo_violated,
         "completion_fraction": m.completion_fraction, "weighted_completion_fraction": m.weighted_completion_fraction,
         "slo_violation_rate": m.slo_violation_rate}
        for m in all_metrics
    ]
    completion_path = os.path.join(output_dir, "completion_accounting.json")
    with open(completion_path, "w", encoding="utf-8") as f:
        json.dump(completion_accounting, f, indent=2)

    print(f"\n=== {workload_name}: mean ANWG across seeds ===")
    by_policy_anwg: Dict[str, List[float]] = {}
    for m in all_metrics:
        by_policy_anwg.setdefault(m.policy_name, []).append(m.arrival_normalized_weighted_goodput)
    for pname, vals in sorted(by_policy_anwg.items(), key=lambda kv: -np.mean(kv[1])):
        ci = ci_results.get(pname, {})
        print(f"  {pname:35s} ANWG_mean={np.mean(vals):.4f}  bootstrap_point={ci.get('point', float('nan')):.4f}  "
              f"CI=[{ci.get('ci_lo', float('nan')):.4f}, {ci.get('ci_hi', float('nan')):.4f}]")

    return {
        "workload": workload_name,
        "outputs": {
            "run_metrics_csv": metrics_path, "request_level_outcomes_csv": outcomes_path,
            "bootstrap_confidence_intervals_json": ci_path, "ranking_agreement_json": ranking_path,
            "behavioral_diversity_json": diversity_path, "completion_accounting_json": completion_path,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", default="data/processed/wildchat/wildchat_eval_sharegpt_shaped.json")
    parser.add_argument("--wildchat-score-cache", default="results/pars_official/wildchat_score_cache.json")
    parser.add_argument("--canonical-score-dir", default="results/pars_official/canonical_suite_matched")
    parser.add_argument("--selector-artifact", default="results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib")
    parser.add_argument("--tokenizer", default="facebook/opt-125m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", default="results/pars_first_comparative_evaluation")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--workloads", nargs="+", default=None,
                         help="'wildchat' and/or canonical family names (default: wildchat + all accepted).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rule_selector = _vllm_ltr_eval.RuleBasedSelector()
    regression_selector = load_selector_artifact(args.selector_artifact)

    workloads = args.workloads
    if workloads is None:
        with open(os.path.join(CANONICAL_SUITE_DIR, "suite_manifest.json")) as f:
            accepted = json.load(f)["accepted"]
        workloads = ["wildchat"] + accepted

    per_workload_manifests = []
    input_hashes = {}

    for wname in workloads:
        if wname == "wildchat":
            with open("data/processed/wildchat/wildchat_eval_prompts_by_id.json") as f:
                id_to_prompt = {int(k): v for k, v in json.load(f).items()}
            score_cache = load_score_cache(args.wildchat_score_cache)
            pars_scores_flat = scores_only(score_cache, id_to_prompt=id_to_prompt)
            requests_by_seed = {seed: _wildchat_requests_for_seed(args.pairs_path, seed, args.tokenizer, args.max_requests)
                                 for seed in args.seeds}
            sim_cfg = SimulatorConfig(gpu_configs=WILDCHAT_GPU_CONFIGS, service_model=WILDCHAT_SERVICE_MODEL, drain_steps=DRAIN_STEPS)
            input_hashes["wildchat_score_cache"] = {"path": args.wildchat_score_cache, "sha256": _sha256_file(args.wildchat_score_cache)}
        else:
            score_dir = os.path.join(args.canonical_score_dir, wname)
            requests_by_seed = {}
            gpu_overrides = service_model_overrides = None
            for seed in args.seeds:
                reqs, gpu_overrides, service_model_overrides = _canonical_requests_for_seed(wname, seed)
                if args.max_requests is not None:
                    reqs = sorted(reqs, key=lambda r: r.request_id)[: args.max_requests]
                requests_by_seed[seed] = reqs
            sim_cfg = _build_sim_config(gpu_overrides, service_model_overrides)
            per_seed_scores = {}
            for seed in args.seeds:
                cache_path = os.path.join(score_dir, f"seed_{seed}_score_cache.json")
                per_seed_scores[seed] = scores_only(load_score_cache(cache_path))
                input_hashes[f"{wname}_seed_{seed}_score_cache"] = {"path": cache_path, "sha256": _sha256_file(cache_path)}
            pars_scores_by_seed_this = per_seed_scores

        if wname == "wildchat":
            pars_scores_by_seed = {seed: pars_scores_flat for seed in args.seeds}
        else:
            pars_scores_by_seed = pars_scores_by_seed_this

        out_dir = os.path.join(args.output_dir, wname)
        result = run_one_workload(wname, requests_by_seed, sim_cfg, pars_scores_by_seed,
                                   rule_selector, regression_selector, out_dir)
        per_workload_manifests.append(result)

    manifest = {
        "command": " ".join(sys.argv),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": args.seeds,
        "max_requests": args.max_requests,
        "pairs_path": args.pairs_path,
        "tokenizer": args.tokenizer,
        "workloads": workloads,
        "policies_compared": FIXED_POLICIES + [BEST_FIXED_POLICY, "rule_based_selector", "regression_anwg_selector", "pars_semantic_reference", "oracle_srtf"],
        "selector_artifact": args.selector_artifact,
        "input_hashes": input_hashes,
        "per_workload_outputs": per_workload_manifests,
    }
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
