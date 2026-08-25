#!/usr/bin/env python3
"""First scientifically fair comparative evaluation of the vLLM-LTR
offline-scored baseline against the existing policy library, run entirely
on the discrete-event simulator (``src/llmserveopt/simulator``).

Workload: real prompt text from WildChat-1M (ingested by
``scripts/ingest_wildchat_eval_dataset.py``), tokenized with the exact
vLLM-LTR checkpoint tokenizer, offline-scored by the exact official
checkpoint (``scripts/score_vllm_ltr_eval_dataset.py``). Arrival timing,
predicted-output-token noise, and SLO-class assignment are synthetic
(``llmserveopt.workloads.sharegpt.convert_sharegpt_to_requests`` +
``llmserveopt.workloads.augmentation.augment_trace``), exactly mirroring
how this repo already treats ShareGPT.

Every policy in this comparison runs on the IDENTICAL request list per
seed (same ``Request`` objects, same admission order opportunities) --
``compare_policies``-style fairness, verified by construction since all
policies iterate the same ``requests`` list built once per seed.

Policies compared (see docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md):
  1. fifo                         6. RuleBasedSelector (current hard selector)
  2. edf                          7. scorpio_style_slo_guard (best fixed policy)
  3. estimated_service_time_first 8. PerPolicyRegressionAnwgSelector (best global composition)
  4. shortest_output_first        9. vllm_ltr_semantic_reference (offline-scored)
  5. weighted_shortest_processing 10. oracle_srtf (non-deployable hindsight ceiling)

vLLM-LTR is used here ONLY for this one-off evaluation run: it is not added
to any registry, selector-candidate list, or CC4/CC5 training data by this
script (see baselines/vllm_ltr/adapter/simulator_policy.py's own
SELECTOR_ELIGIBLE = False and docs/audits/vllm_ltr_baseline_audit_20260804.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from baselines.vllm_ltr.adapter.offline_scoring import load_score_cache, scores_only
from baselines.vllm_ltr.adapter.simulator_policy import VLLMLTRSemanticReferencePolicy
from llmserveopt.core.metrics import RunMetrics, compute_metrics
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.oracle import build_oracle
from llmserveopt.policies.registry import make_policy
from llmserveopt.policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA, predicted_service_proxy
from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FeatureMode, extract_features
from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector, RuleBasedSelector
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from llmserveopt.workloads.sharegpt import ShareGPTConversionConfig, convert_sharegpt_to_requests
from llmserveopt.workloads.augmentation import AugmentationConfig

FIXED_POLICIES = [
    "fifo",
    "edf",
    "estimated_service_time_first",
    "shortest_output_first",
    "weighted_shortest_processing",
]
BEST_FIXED_POLICY = "scorpio_style_slo_guard"

# Matches configs/sharegpt_poisson_comparison.yaml -- reused unchanged so
# this evaluation's hardware/capacity assumptions are consistent with this
# repo's existing ShareGPT-based comparisons, not a bespoke config.
GPU_CONFIGS = [
    GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=8192, max_kv_tokens=131072)
]
SERVICE_MODEL = ServiceModel(
    step_size=0.001,
    enable_prefill_modeling=True,
    prefill_cost_per_token=1.0,
    max_prefill_chunk_tokens=512,
    step_token_budget=8192,
    decode_first=False,
)
DRAIN_STEPS = 200_000


class SelectorArtifactError(Exception):
    """Raised when the persisted selector artifact/manifest is missing,
    unreadable, or was not trained under the corrected objective."""


class SelectorDispatchPolicy(BasePolicy):
    """Wraps a feature-based selector (``RuleBasedSelector`` or
    ``PerPolicyRegressionAnwgSelector``) as a ``BasePolicy`` for the
    discrete-event simulator: at every decision step, compute the 18
    selector features from the CURRENT ``ObservableState`` (causal mode --
    only what a real online policy could see), ask the selector which
    candidate policy to use, and delegate ``select_action`` to that
    candidate.

    Honest limitation (documented, not hidden): ``recent_slo_violation_rate``
    is not observable from ``ObservableState`` alone (it isn't tracked
    anywhere the simulator exposes to policies) -- this wrapper passes
    ``recent_violation_available=False``, exactly the same "honest
    placeholder" pattern ``scripts/run_vllm_external_baseline_comparison.py``
    already uses for ``kv_utilization_available`` under its own
    observability limits. Every OTHER feature (queue, KV utilization, free
    sequence ratio, prompt/output-token stats, SLO slack, arrival rate) is
    computed for real from the live simulator state, which is strictly
    MORE observable here than in that HTTP-harness's client-side view.
    """

    def __init__(self, selector, name: str):
        self._selector = selector
        self.name = name
        self._subpolicy_cache: Dict[str, BasePolicy] = {}
        # Dispatch histogram (behavioral-diversity accounting): how many
        # decision steps this run routed to each candidate sub-policy.
        # NOT reset by reset() -- reset() is called once per (policy, seed)
        # simulator run, and callers read/accumulate counts across seeds
        # via dispatch_counts_by_seed.
        self.dispatch_counts_by_seed: Dict[int, Dict[str, int]] = {}
        self._current_seed: Optional[int] = None

    def start_seed(self, seed: int) -> None:
        self._current_seed = seed
        self.dispatch_counts_by_seed[seed] = {}

    def reset(self) -> None:
        self._subpolicy_cache.clear()

    def _get_subpolicy(self, policy_name: str) -> BasePolicy:
        if policy_name not in self._subpolicy_cache:
            self._subpolicy_cache[policy_name] = make_policy(policy_name)
        return self._subpolicy_cache[policy_name]

    def select_action(self, state):
        active_sequence_count = sum(len(g.active_request_ids) for g in state.gpu_states)
        concurrency = sum(g.max_active_sequences for g in state.gpu_states)
        total_kv = sum(g.max_kv_tokens for g in state.gpu_states)
        used_kv = sum(g.current_kv_tokens for g in state.gpu_states)
        kv_utilization = used_kv / total_kv if total_kv > 0 else 0.0
        free_sequence_ratio = 1.0 - (active_sequence_count / concurrency if concurrency > 0 else 0.0)

        features = extract_features(
            window_requests=state.waiting_queue,
            window_start_time=state.time,
            mode=FeatureMode.CAUSAL,
            prefix_requests=None,
            recent_violation_rate=0.0,
            recent_violation_available=False,
            active_sequence_count=active_sequence_count,
            kv_utilization=kv_utilization,
            kv_utilization_available=True,
            free_sequence_ratio=free_sequence_ratio,
            free_sequence_ratio_available=True,
        )
        policy_name = self._selector.predict_one(features)
        if self._current_seed is not None:
            counts = self.dispatch_counts_by_seed[self._current_seed]
            counts[policy_name] = counts.get(policy_name, 0) + 1
        return self._get_subpolicy(policy_name).select_action(state)


def load_selector_artifact(artifact_path: str) -> PerPolicyRegressionAnwgSelector:
    """Load the persisted 'best global composition' selector, refusing
    anything not explicitly declared as trained under the corrected
    arrival_normalized_wg objective (mirrors
    scripts/run_vllm_external_baseline_comparison.py's
    load_and_validate_selector_artifact -- reimplemented locally since
    scripts/ is not an importable package)."""
    if not os.path.exists(artifact_path):
        raise SelectorArtifactError(f"Selector artifact not found: {artifact_path}")
    manifest_path = os.path.join(os.path.dirname(artifact_path), "manifest.json")
    if not os.path.exists(manifest_path):
        raise SelectorArtifactError(f"No manifest.json next to {artifact_path}.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    objective = manifest.get("objective_definition", {}).get("name")
    if objective != "arrival_normalized_wg":
        raise SelectorArtifactError(
            f"Selector manifest declares objective_definition.name={objective!r}, "
            "expected 'arrival_normalized_wg'."
        )
    return PerPolicyRegressionAnwgSelector.load(artifact_path)


@dataclass
class SeedRunResult:
    metrics: RunMetrics
    rows: List[dict]  # per-request rows for bootstrap CI


def run_policy_with_rows(
    policy: BasePolicy,
    requests: List[Request],
    workload_tag: str,
    seed: int,
) -> SeedRunResult:
    sim_cfg = SimulatorConfig(
        gpu_configs=GPU_CONFIGS, service_model=SERVICE_MODEL, drain_steps=DRAIN_STEPS
    )
    sim = Simulator(sim_cfg)
    sim.load_trace(list(requests))
    policy.reset()
    metrics = sim.run(policy, workload_tag=workload_tag, seed=seed)

    completed_by_id = {c.request.request_id: c for c in sim._completed}
    rows = []
    for r in requests:
        c = completed_by_id.get(r.request_id)
        if c is not None:
            rows.append({
                "request_id": r.request_id,
                "seed": seed,
                "priority": r.priority,
                "class_id": r.class_id,
                "status": "success",
                "slo_violated": bool(c.slo_violated),
            })
        else:
            rows.append({
                "request_id": r.request_id,
                "seed": seed,
                "priority": r.priority,
                "class_id": r.class_id,
                "status": "dropped",
                "slo_violated": True,
            })
    return SeedRunResult(metrics=metrics, rows=rows)


def compute_bootstrap_ci(
    rows_by_policy: Dict[str, List[dict]],
    reference_policy: str,
    n_boot: int = 2000,
    seed: int = 20260804,
) -> Dict[str, dict]:
    """Paired bootstrap over (seed, request_id): for each policy, resample
    (seed, request_id) pairs with replacement (paired across policies,
    since every policy ran the identical per-seed request list) and
    recompute arrival-normalized WG each replicate. Also reports the
    paired difference (reference_policy - each other policy)."""
    rng = np.random.default_rng(seed)
    by_policy: Dict[str, Dict[Tuple[int, int], dict]] = {
        pname: {(r["seed"], r["request_id"]): r for r in rows}
        for pname, rows in rows_by_policy.items()
    }
    all_keys = sorted(next(iter(by_policy.values())).keys())
    n = len(all_keys)

    def _anwg_for_keys(policy_rows: Dict[Tuple[int, int], dict], keys) -> float:
        num, den = 0.0, 0.0
        for k in keys:
            row = policy_rows[k]
            w = row["priority"] if row["priority"] > 0 else 1.0
            den += w
            if row["status"] == "success" and not row["slo_violated"]:
                num += w
        return num / den if den > 0 else 0.0

    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    results: Dict[str, dict] = {}
    ref_replicates = None
    for pname, policy_rows in by_policy.items():
        replicates = np.array([
            _anwg_for_keys(policy_rows, [all_keys[i] for i in idx_matrix[b]])
            for b in range(n_boot)
        ])
        point = _anwg_for_keys(policy_rows, all_keys)
        lo, hi = np.percentile(replicates, [2.5, 97.5])
        results[pname] = {"point": point, "ci_lo": float(lo), "ci_hi": float(hi)}
        if pname == reference_policy:
            ref_replicates = replicates

    if ref_replicates is not None:
        for pname, policy_rows in by_policy.items():
            if pname == reference_policy:
                continue
            other_replicates = np.array([
                _anwg_for_keys(policy_rows, [all_keys[i] for i in idx_matrix[b]])
                for b in range(n_boot)
            ])
            diff = ref_replicates - other_replicates
            lo, hi = np.percentile(diff, [2.5, 97.5])
            results[pname][f"{reference_policy}_minus_this_ci"] = [float(lo), float(hi)]
    return results


def _rank_correlation(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation, implemented locally (no scipy dependency,
    mirroring this repo's policy of not adding hard deps for one metric)."""
    a_ranks = np.argsort(np.argsort(a)).astype(float)
    b_ranks = np.argsort(np.argsort(b)).astype(float)
    if np.std(a_ranks) == 0 or np.std(b_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(a_ranks, b_ranks)[0, 1])


def compute_ranking_agreement_record(
    seed: int,
    requests: List[Request],
    ltr_scores: Dict[int, float],
) -> dict:
    """Spearman agreement between vLLM-LTR's score order and each SJF-proxy
    policy's own real ranking rule, over the request set (identical
    regardless of seed's arrival/SLO augmentation -- only real-text-derived
    fields are used).

    ``estimated_service_time_first`` ranks by ``predicted_service_proxy``
    (``alpha*prompt_tokens + beta*predicted_output_tokens`` -- the actual
    sort key ``EstimatedServiceTimeFirstPolicy._sort_key`` uses), which
    differs from ``shortest_output_first`` (``predicted_output_tokens``
    alone) whenever prompt length varies independently of predicted output
    length. Using the same formula for both would silently report two
    copies of one correlation instead of independently measuring agreement
    with each policy -- see
    docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md.
    """
    req_by_id = {r.request_id: r for r in requests}
    ids_sorted = sorted(req_by_id.keys())
    ltr_order = [ltr_scores[i] for i in ids_sorted]
    # Negated so "higher = higher priority" for the Spearman correlation,
    # matching ltr_scores' own higher-is-better convention.
    est_order = [
        -predicted_service_proxy(req_by_id[i], alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA)
        for i in ids_sorted
    ]
    sof_order = [-req_by_id[i].predicted_output_tokens for i in ids_sorted]
    return {
        "seed": seed,
        "spearman_ltr_vs_estimated_service_time_first": _rank_correlation(ltr_order, est_order),
        "spearman_ltr_vs_shortest_output_first": _rank_correlation(ltr_order, sof_order),
    }


def build_requests_for_seed(pairs_path: str, seed: int, tokenizer_name: str) -> List[Request]:
    from llmserveopt.workloads.sharegpt import load_sharegpt_raw

    records = load_sharegpt_raw(pairs_path)
    config = ShareGPTConversionConfig(
        arrival_mode="poisson",
        arrival_rate=10.0,
        tokenizer_name=tokenizer_name,
        fallback_whitespace=False,
    )
    requests, report = convert_sharegpt_to_requests(
        records, config=config, seed=seed, aug_config=AugmentationConfig()
    )
    if report.rows_retained != len(records):
        warnings.warn(
            f"seed={seed}: {report.rows_retained}/{len(records)} rows retained "
            "after token-count filtering."
        )
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", default="data/processed/wildchat/wildchat_eval_sharegpt_shaped.json")
    parser.add_argument("--prompts-path", default="data/processed/wildchat/wildchat_eval_prompts_by_id.json")
    parser.add_argument("--score-cache-path", default="data/processed/wildchat/vllm_ltr_score_cache.json")
    parser.add_argument("--selector-artifact", default="results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib")
    parser.add_argument("--tokenizer", default="facebook/opt-125m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", default="results/vllm_ltr_first_comparative_evaluation")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="If set, deterministically truncate each seed's request list to "
             "the first N by ascending request_id -- for a fast pilot run "
             "before committing to the full sample size.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.prompts_path, "r", encoding="utf-8") as f:
        id_to_prompt = {int(k): v for k, v in json.load(f).items()}
    score_cache = load_score_cache(args.score_cache_path)
    ltr_scores = scores_only(score_cache, id_to_prompt=id_to_prompt)
    print(f"Loaded {len(ltr_scores)} vLLM-LTR scores (hash-verified against current prompts).")

    rule_selector = RuleBasedSelector()
    regression_selector = load_selector_artifact(args.selector_artifact)
    # Created once (not per-seed) so dispatch_counts_by_seed accumulates
    # across every seed in this run -- behavioral-diversity accounting.
    rule_selector_dispatch = SelectorDispatchPolicy(rule_selector, name="rule_based_selector")
    regression_selector_dispatch = SelectorDispatchPolicy(regression_selector, name="regression_anwg_selector")

    all_metrics: List[RunMetrics] = []
    rows_by_policy: Dict[str, List[dict]] = {}
    ranking_agreement_records: List[dict] = []

    for seed in args.seeds:
        requests = build_requests_for_seed(args.pairs_path, seed, args.tokenizer)
        if args.max_requests is not None:
            requests = sorted(requests, key=lambda r: r.request_id)[: args.max_requests]
        missing = [r.request_id for r in requests if r.request_id not in ltr_scores]
        if missing:
            raise RuntimeError(
                f"seed={seed}: {len(missing)} requests have no vLLM-LTR score "
                f"(first few missing ids: {missing[:5]}). Run "
                "scripts/score_vllm_ltr_eval_dataset.py against the same "
                "prompts file first."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # oracle's own documented warning
            oracle = build_oracle(requests)

        rule_selector_dispatch.start_seed(seed)
        regression_selector_dispatch.start_seed(seed)

        policies: List[BasePolicy] = [make_policy(p) for p in FIXED_POLICIES]
        policies.append(make_policy(BEST_FIXED_POLICY))
        policies.append(rule_selector_dispatch)
        policies.append(regression_selector_dispatch)
        policies.append(VLLMLTRSemanticReferencePolicy(scores=ltr_scores))
        policies.append(oracle)

        for policy in policies:
            print(f"  seed={seed} policy={policy.name} n_req={len(requests)}")
            result = run_policy_with_rows(policy, requests, workload_tag="wildchat_eval", seed=seed)
            all_metrics.append(result.metrics)
            for row in result.rows:
                row["policy"] = policy.name
            rows_by_policy.setdefault(policy.name, []).extend(result.rows)

        ranking_agreement_records.append(
            compute_ranking_agreement_record(seed, requests, ltr_scores)
        )

    reference_policy = "vllm_ltr_semantic_reference"
    ci_results = compute_bootstrap_ci(rows_by_policy, reference_policy=reference_policy)

    metrics_path = os.path.join(args.output_dir, "run_metrics.csv")
    import csv
    from llmserveopt.core.metrics import metrics_to_dict
    fieldnames = list(metrics_to_dict(all_metrics[0]).keys())
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(metrics_to_dict(m))
    print(f"Wrote {metrics_path}")

    # Raw per-request outcome rows (policy, seed, request_id, priority,
    # status, slo_violated) -- the same rows compute_bootstrap_ci() uses
    # internally, persisted so independent verification (task step 10) can
    # recompute ANWG/completion fraction/bootstrap CIs from genuinely raw,
    # unaggregated data rather than trusting run_metrics.csv's own
    # aggregation code.
    outcomes_path = os.path.join(args.output_dir, "request_level_outcomes.csv")
    with open(outcomes_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy", "seed", "request_id", "priority", "class_id", "status", "slo_violated",
            ],
        )
        writer.writeheader()
        for pname, rows in rows_by_policy.items():
            for row in rows:
                writer.writerow(row)
    print(f"Wrote {outcomes_path}")

    ci_path = os.path.join(args.output_dir, "bootstrap_confidence_intervals.json")
    with open(ci_path, "w", encoding="utf-8") as f:
        json.dump(ci_results, f, indent=2)
    print(f"Wrote {ci_path}")

    ranking_path = os.path.join(args.output_dir, "ranking_agreement.json")
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(ranking_agreement_records, f, indent=2)
    print(f"Wrote {ranking_path}")

    def _entropy_bits(counts: Dict[str, int]) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        ps = np.array([c / total for c in counts.values()], dtype=float)
        return float(-np.sum(ps * np.log2(ps)))

    def _dispatch_summary(dispatch: "SelectorDispatchPolicy") -> dict:
        totals: Dict[str, int] = {}
        for seed_counts in dispatch.dispatch_counts_by_seed.values():
            for pname, c in seed_counts.items():
                totals[pname] = totals.get(pname, 0) + c
        return {
            "dispatch_counts_by_seed": dispatch.dispatch_counts_by_seed,
            "dispatch_counts_total": totals,
            "num_distinct_subpolicies_dispatched": len(totals),
            "entropy_bits": _entropy_bits(totals),
            "max_entropy_bits_at_this_diversity": (
                float(np.log2(len(totals))) if len(totals) > 0 else 0.0
            ),
        }

    behavioral_diversity = {
        "rule_based_selector": _dispatch_summary(rule_selector_dispatch),
        "regression_anwg_selector": _dispatch_summary(regression_selector_dispatch),
        "ranking_agreement": ranking_agreement_records,
    }
    diversity_path = os.path.join(args.output_dir, "behavioral_diversity.json")
    with open(diversity_path, "w", encoding="utf-8") as f:
        json.dump(behavioral_diversity, f, indent=2)
    print(f"Wrote {diversity_path}")

    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    completion_accounting = []
    for m in all_metrics:
        completion_accounting.append({
            "policy": m.policy_name,
            "seed": m.seed,
            "num_total": m.num_total,
            "num_completed": m.num_completed,
            "num_dropped": m.num_dropped,
            "num_slo_violated": m.num_slo_violated,
            "completion_fraction": m.completion_fraction,
            "weighted_completion_fraction": m.weighted_completion_fraction,
            "slo_violation_rate": m.slo_violation_rate,
        })
    completion_path = os.path.join(args.output_dir, "completion_accounting.json")
    with open(completion_path, "w", encoding="utf-8") as f:
        json.dump(completion_accounting, f, indent=2)
    print(f"Wrote {completion_path}")

    input_hashes = {}
    for label, path in [
        ("pairs_path", args.pairs_path),
        ("prompts_path", args.prompts_path),
        ("score_cache_path", args.score_cache_path),
        ("selector_artifact", args.selector_artifact),
    ]:
        input_hashes[label] = {"path": path, "sha256": _sha256_file(path)}

    manifest = {
        "command": " ".join(sys.argv),
        "replay_command": (
            f"python3 scripts/run_vllm_ltr_first_comparative_evaluation.py "
            f"--pairs-path {args.pairs_path} --prompts-path {args.prompts_path} "
            f"--score-cache-path {args.score_cache_path} "
            f"--selector-artifact {args.selector_artifact} --tokenizer {args.tokenizer} "
            f"--seeds {' '.join(str(s) for s in args.seeds)} --output-dir {args.output_dir}"
            + (f" --max-requests {args.max_requests}" if args.max_requests is not None else "")
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": args.seeds,
        "max_requests": args.max_requests,
        "policies_compared": [p.name for p in policies],
        "num_ltr_scores_loaded": len(ltr_scores),
        "input_hashes": input_hashes,
        "outputs": {
            "run_metrics_csv": metrics_path,
            "request_level_outcomes_csv": outcomes_path,
            "bootstrap_confidence_intervals_json": ci_path,
            "ranking_agreement_json": ranking_path,
            "behavioral_diversity_json": diversity_path,
            "completion_accounting_json": completion_path,
        },
    }
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")

    print("\n=== Summary (mean arrival_normalized_weighted_goodput across seeds) ===")
    by_policy_anwg: Dict[str, List[float]] = {}
    for m in all_metrics:
        by_policy_anwg.setdefault(m.policy_name, []).append(m.arrival_normalized_weighted_goodput)
    for pname, vals in sorted(by_policy_anwg.items(), key=lambda kv: -np.mean(kv[1])):
        ci = ci_results.get(pname, {})
        print(
            f"  {pname:35s} ANWG_mean={np.mean(vals):.4f}  "
            f"bootstrap_point={ci.get('point', float('nan')):.4f}  "
            f"CI=[{ci.get('ci_lo', float('nan')):.4f}, {ci.get('ci_hi', float('nan')):.4f}]"
        )


if __name__ == "__main__":
    main()
