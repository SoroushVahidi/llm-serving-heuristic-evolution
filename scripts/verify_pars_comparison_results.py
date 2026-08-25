#!/usr/bin/env python3
"""Independent re-verification of a PARS comparative-evaluation run's
outputs, across every workload it was run on (WildChat control + accepted
canonical-suite families).

Reuses (imports, does not duplicate) the generic, workload-agnostic
recomputation functions already written and proven for the vLLM-LTR
evaluation's independent verifier
(``scripts/verify_vllm_ltr_comparison_results.py`` --
``recompute_completion_fraction``, ``win_tie_loss``,
``oracle_envelope_contribution``, ``paired_request_bootstrap_ci``,
``per_regime_breakdown``, ``completion_violation_counts``,
``load_request_level_outcomes``, ``recompute_anwg_from_request_rows``,
``cross_check_anwg_against_run_metrics``, ``independent_rank_correlation``)
-- none of those functions assume anything about vLLM-LTR specifically,
only about the shared ``run_metrics.csv``/``request_level_outcomes.csv``
schema both eval scripts produce. Only the ranking-agreement cross-check
is PARS-specific (different score cache layout/ranking-sign convention),
reimplemented fresh here rather than reused.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_vllm_ltr_comparison_results",
    Path(__file__).parent / "verify_vllm_ltr_comparison_results.py",
)
_v = importlib.util.module_from_spec(_VERIFY_SPEC)
sys.modules["verify_vllm_ltr_comparison_results"] = _v
_VERIFY_SPEC.loader.exec_module(_v)

_read_csv_rows = _v._read_csv_rows
_to_float = _v._to_float
recompute_completion_fraction = _v.recompute_completion_fraction
win_tie_loss = _v.win_tie_loss
oracle_envelope_contribution = _v.oracle_envelope_contribution
load_request_level_outcomes = _v.load_request_level_outcomes
recompute_anwg_from_request_rows = _v.recompute_anwg_from_request_rows
cross_check_anwg_against_run_metrics = _v.cross_check_anwg_against_run_metrics
paired_request_bootstrap_ci = _v.paired_request_bootstrap_ci
per_regime_breakdown = _v.per_regime_breakdown
completion_violation_counts = _v.completion_violation_counts
independent_rank_correlation = _v.independent_rank_correlation


def cross_check_pars_ranking_agreement(workload_output_dir: str, requests_by_seed: Dict[int, list],
                                        input_hashes: dict) -> dict:
    """Independent Spearman recomputation for PARS-vs-EST/SOF agreement,
    reading the run's own ranking_agreement.json and recomputing from raw
    Request fields (predicted_output_tokens for SOF,
    llmserveopt.policies.scoring.predicted_service_proxy for EST) plus the
    PARS score cache -- mirrors
    verify_vllm_ltr_comparison_results.cross_check_ranking_agreement's
    approach but reads PARS's own cache format."""
    ranking_path = os.path.join(workload_output_dir, "ranking_agreement.json")
    if not os.path.exists(ranking_path):
        return {"error": f"{ranking_path} not found"}
    with open(ranking_path) as f:
        recorded = json.load(f)

    from llmserveopt.policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA, predicted_service_proxy

    workload_name = os.path.basename(workload_output_dir.rstrip("/"))
    mismatches = []
    recomputed_records = []
    for rec in recorded:
        seed = rec["seed"]
        requests = requests_by_seed[seed]
        req_by_id = {r.request_id: r for r in requests}
        ids_sorted = sorted(req_by_id.keys())

        from baselines.pars.adapter.offline_scoring import load_score_cache, scores_only

        if workload_name == "wildchat":
            cache_path = input_hashes["wildchat_score_cache"]["path"]
        else:
            cache_path = input_hashes[f"{workload_name}_seed_{seed}_score_cache"]["path"]
        pars_scores = scores_only(load_score_cache(cache_path))

        pars_order = [-pars_scores[i] for i in ids_sorted]
        est_order = [-predicted_service_proxy(req_by_id[i], alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA) for i in ids_sorted]
        sof_order = [-req_by_id[i].predicted_output_tokens for i in ids_sorted]

        est_corr = independent_rank_correlation(pars_order, est_order)
        sof_corr = independent_rank_correlation(pars_order, sof_order)
        recomputed_records.append({"seed": seed, "spearman_pars_vs_est": est_corr, "spearman_pars_vs_sof": sof_corr})

        def _close(a, b, tol=5e-3):
            if math.isnan(a) and math.isnan(b):
                return True
            if math.isnan(a) or math.isnan(b):
                return False
            return abs(a - b) <= tol

        if not _close(est_corr, rec["spearman_pars_vs_estimated_service_time_first"]):
            mismatches.append({"seed": seed, "field": "est", "recomputed": est_corr,
                                "recorded": rec["spearman_pars_vs_estimated_service_time_first"]})
        if not _close(sof_corr, rec["spearman_pars_vs_shortest_output_first"]):
            mismatches.append({"seed": seed, "field": "sof", "recomputed": sof_corr,
                                "recorded": rec["spearman_pars_vs_shortest_output_first"]})

    return {"recomputed": recomputed_records, "recorded": recorded, "mismatches": mismatches,
            "all_match": len(mismatches) == 0}


def verify_workload(workload_output_dir: str, workload_name: str, n_prompts_expected: "int | None",
                     reference_policy: str = "pars_semantic_reference") -> dict:
    metrics_path = os.path.join(workload_output_dir, "run_metrics.csv")
    rows = _read_csv_rows(metrics_path)
    if not rows:
        return {"error": f"no rows in {metrics_path}"}

    completion_check = recompute_completion_fraction(rows)
    n_completion_mismatches = sum(1 for v in completion_check.values()
                                   if not v["completion_fraction_matches"] or not v["slo_violation_rate_matches"])
    n_identity_violations = sum(1 for v in completion_check.values() if not v["accounting_identity_holds"])

    n_seeds = len({int(r["seed"]) for r in rows})
    n_policies = len({r["policy"] for r in rows})

    outcomes_path = os.path.join(workload_output_dir, "request_level_outcomes.csv")
    by_policy = load_request_level_outcomes(outcomes_path)
    n_raw_rows = sum(len(v) for v in by_policy.values())
    if n_prompts_expected is None:
        # Canonical-suite families use Poisson arrivals, so num_total can
        # legitimately differ per seed -- sum each seed's own num_total
        # rather than assuming a uniform per-seed count (a uniform count
        # only happens to hold for wildchat, where n_req=300 every seed).
        num_total_by_seed = {}
        for r in rows:
            num_total_by_seed.setdefault(int(r["seed"]), int(r["num_total"]))
        expected_raw_rows = sum(num_total_by_seed.values()) * n_policies
    else:
        expected_raw_rows = n_prompts_expected * n_seeds * n_policies

    request_level_anwg = recompute_anwg_from_request_rows(by_policy)
    anwg_mismatches = cross_check_anwg_against_run_metrics(request_level_anwg, rows)
    ci = paired_request_bootstrap_ci(by_policy, reference_policy=reference_policy)
    wtl = win_tie_loss(request_level_anwg)
    envelope = oracle_envelope_contribution(request_level_anwg)
    regime = per_regime_breakdown(by_policy)
    violations = completion_violation_counts(by_policy)

    return {
        "workload": workload_name,
        "row_count_check": {"expected": expected_raw_rows, "actual": n_raw_rows, "matches": n_raw_rows == expected_raw_rows},
        "anwg_cross_check_mismatches": len(anwg_mismatches),
        "n_completion_mismatches": n_completion_mismatches,
        "n_identity_violations": n_identity_violations,
        "paired_bootstrap_ci": ci,
        "win_tie_loss": wtl,
        "oracle_envelope_contribution": envelope,
        "per_regime_anwg": regime,
        "completion_violation_counts": violations,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/pars_first_comparative_evaluation")
    parser.add_argument("--n-prompts", type=int, default=None,
                         help="Prompts per seed per workload. If omitted, inferred per-workload from run_metrics.csv's num_total.")
    parser.add_argument("--skip-ranking-cross-check", action="store_true")
    args = parser.parse_args()

    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    report = {"workloads": {}}
    any_hard_failure = False

    for wname in manifest["workloads"]:
        wdir = os.path.join(args.output_dir, wname)
        result = verify_workload(wdir, wname, args.n_prompts)
        report["workloads"][wname] = result

        print(f"=== {wname} ===")
        print(f"  row count: expected={result['row_count_check']['expected']} actual={result['row_count_check']['actual']} "
              f"matches={result['row_count_check']['matches']}")
        print(f"  ANWG cross-check mismatches: {result['anwg_cross_check_mismatches']}")
        print(f"  completion mismatches: {result['n_completion_mismatches']}  identity violations: {result['n_identity_violations']}")

        if not args.skip_ranking_cross_check:
            max_requests = manifest.get("max_requests")
            if wname == "wildchat":
                from run_pars_first_comparative_evaluation import _wildchat_requests_for_seed
                requests_by_seed = {seed: _wildchat_requests_for_seed(
                    manifest["pairs_path"], seed, manifest["tokenizer"], max_requests
                ) for seed in manifest["seeds"]}
            else:
                from run_pars_first_comparative_evaluation import _canonical_requests_for_seed
                requests_by_seed = {}
                for seed in manifest["seeds"]:
                    reqs = _canonical_requests_for_seed(wname, seed)[0]
                    if max_requests is not None:
                        reqs = sorted(reqs, key=lambda r: r.request_id)[:max_requests]
                    requests_by_seed[seed] = reqs
            ranking_result = cross_check_pars_ranking_agreement(wdir, requests_by_seed, manifest["input_hashes"])
            report["workloads"][wname]["ranking_agreement_cross_check"] = ranking_result
            if "error" in ranking_result:
                print(f"  ranking cross-check: ERROR ({ranking_result['error']})")
            else:
                print(f"  ranking cross-check: all_match={ranking_result['all_match']} ({len(ranking_result['mismatches'])} mismatches)")

        hard_fail = (
            not result["row_count_check"]["matches"] or result["anwg_cross_check_mismatches"] > 0
            or result["n_completion_mismatches"] > 0 or result["n_identity_violations"] > 0
        )
        any_hard_failure = any_hard_failure or hard_fail
        print()

    out_path = os.path.join(args.output_dir, "independent_verification_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    sys.exit(1 if any_hard_failure else 0)


if __name__ == "__main__":
    main()
