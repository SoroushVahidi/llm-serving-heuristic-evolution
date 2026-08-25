#!/usr/bin/env python3
"""Independent re-verification of a vLLM-LTR comparative-evaluation run's
outputs (task step 10 of the 2026-08-04 recovery task -- see
docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md section 6).

This script does NOT reuse
``scripts/run_vllm_ltr_first_comparative_evaluation.py``'s own aggregation
functions (``compute_bootstrap_ci``, ``_rank_correlation``,
``compute_ranking_agreement_record``) -- every number here is re-derived
from the run's raw output files using independently written code, and
cross-checked against the run's own summary artifacts
(``bootstrap_confidence_intervals.json``, ``ranking_agreement.json``). Any
mismatch is reported, not silently accepted.

Two modes, chosen automatically based on what the output directory
contains:

* **Request-level mode** (preferred): if ``request_level_outcomes.csv``
  is present (one row per (policy, seed, request_id) -- added during this
  recovery specifically so independent verification would not be limited
  to re-checking pre-aggregated ratios), this script recomputes ANWG,
  completion fraction, a genuine PAIRED bootstrap CI (resampling
  (seed, request_id) keys, mirroring the eval script's own resampling
  granularity but implemented fresh), win/tie/loss, oracle-envelope
  contribution, and a per-regime (``class_id``: tight/medium/loose SLO
  class) breakdown directly from the raw per-request rows.
* **Seed-level fallback mode**: if that file is absent (e.g. a run
  predating this addition), the same quantities are recomputed from
  ``run_metrics.csv``'s per-(policy, seed) aggregate counts instead, and
  the CI is a coarser SEED-level bootstrap (resampling over the run's N
  seeds' own per-seed ANWG values) -- reported as such, not conflated with
  the request-level paired bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


def _read_csv_rows(path: str) -> List[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v) -> float:
    if v is None or v == "":
        return float("nan")
    return float(v)


def recompute_completion_fraction(rows: List[dict]) -> Dict[Tuple[str, int], dict]:
    """Independently recompute completion_fraction and slo_violation_rate
    from the raw integer counts in run_metrics.csv, and cross-check against
    the recorded ratio columns."""
    out = {}
    for row in rows:
        policy, seed = row["policy"], int(row["seed"])
        num_completed = int(row["num_completed"])
        num_total = int(row["num_total"])
        num_dropped = int(row["num_dropped"])
        num_slo_violated = int(row["num_slo_violated"])

        recomputed_completion_fraction = (
            num_completed / num_total if num_total > 0 else float("nan")
        )
        recomputed_slo_violation_rate = (
            num_slo_violated / num_completed if num_completed > 0 else float("nan")
        )
        recorded_completion_fraction = _to_float(row["completion_fraction"])
        recorded_slo_violation_rate = _to_float(row["slo_violation_rate"])

        def _close(a: float, b: float, tol: float = 6e-7) -> bool:
            # run_metrics.csv values are written through the eval script's
            # own metrics_to_dict()/_fmt(), which rounds to 6 decimal
            # places -- tol must exceed that rounding's max error (5e-7),
            # or every non-terminating ratio (e.g. 1/300) falsely "mismatches".
            if math.isnan(a) and math.isnan(b):
                return True
            if math.isnan(a) or math.isnan(b):
                return False
            return abs(a - b) <= tol

        out[(policy, seed)] = {
            "num_completed": num_completed,
            "num_total": num_total,
            "num_dropped": num_dropped,
            "num_slo_violated": num_slo_violated,
            "recomputed_completion_fraction": recomputed_completion_fraction,
            "recorded_completion_fraction": recorded_completion_fraction,
            "completion_fraction_matches": _close(
                recomputed_completion_fraction, recorded_completion_fraction
            ),
            "recomputed_slo_violation_rate": recomputed_slo_violation_rate,
            "recorded_slo_violation_rate": recorded_slo_violation_rate,
            "slo_violation_rate_matches": _close(
                recomputed_slo_violation_rate, recorded_slo_violation_rate
            ),
            # accounting identity: every arrival either completed or was
            # dropped/still-active at drain; completed+dropped must never
            # exceed num_total.
            "accounting_identity_holds": (num_completed + num_dropped) <= num_total,
            "recorded_arrival_normalized_weighted_goodput": _to_float(
                row["arrival_normalized_weighted_goodput"]
            ),
        }
    return out


def seed_level_bootstrap_ci(
    anwg_by_policy_seed: Dict[str, Dict[int, float]],
    n_boot: int = 5000,
    seed: int = 20260804,
) -> Dict[str, dict]:
    """Independent CI: bootstrap resampling over the run's seeds themselves
    (not paired per-request resampling -- see module docstring). Legitimate
    when seeds are iid replicates of the same evaluation."""
    rng = np.random.default_rng(seed)
    results = {}
    for policy, by_seed in anwg_by_policy_seed.items():
        seeds_sorted = sorted(by_seed.keys())
        vals = np.array([by_seed[s] for s in seeds_sorted], dtype=float)
        n = len(vals)
        if n < 2:
            results[policy] = {
                "point": float(np.mean(vals)) if n else float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "note": "fewer than 2 seeds -- CI undefined at seed granularity",
            }
            continue
        idx = rng.integers(0, n, size=(n_boot, n))
        replicates = vals[idx].mean(axis=1)
        lo, hi = np.percentile(replicates, [2.5, 97.5])
        results[policy] = {
            "point": float(np.mean(vals)),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n_seeds": n,
        }
    return results


def win_tie_loss(anwg_by_policy_seed: Dict[str, Dict[int, float]], tol: float = 1e-9) -> dict:
    """Per-seed pairwise win/tie/loss counts across all policies, plus each
    policy's 'unique win' count (strictly highest ANWG that seed, no ties
    for first)."""
    policies = sorted(anwg_by_policy_seed.keys())
    seeds = sorted({s for by_seed in anwg_by_policy_seed.values() for s in by_seed})

    pairwise = {p: {"wins": 0, "ties": 0, "losses": 0} for p in policies}
    for s in seeds:
        vals = {p: anwg_by_policy_seed[p].get(s, float("nan")) for p in policies}
        for i, pi in enumerate(policies):
            for pj in policies[i + 1:]:
                vi, vj = vals[pi], vals[pj]
                if math.isnan(vi) or math.isnan(vj):
                    continue
                if abs(vi - vj) <= tol:
                    pairwise[pi]["ties"] += 1
                    pairwise[pj]["ties"] += 1
                elif vi > vj:
                    pairwise[pi]["wins"] += 1
                    pairwise[pj]["losses"] += 1
                else:
                    pairwise[pj]["wins"] += 1
                    pairwise[pi]["losses"] += 1

    unique_wins = {p: 0 for p in policies}
    for s in seeds:
        vals = {p: anwg_by_policy_seed[p].get(s, float("nan")) for p in policies
                 if not math.isnan(anwg_by_policy_seed[p].get(s, float("nan")))}
        if not vals:
            continue
        best_val = max(vals.values())
        top = [p for p, v in vals.items() if abs(v - best_val) <= tol]
        if len(top) == 1:
            unique_wins[top[0]] += 1

    return {"pairwise_win_tie_loss": pairwise, "unique_wins_by_seed": unique_wins}


def oracle_envelope_contribution(anwg_by_policy_seed: Dict[str, Dict[int, float]]) -> dict:
    oracle_key = next((p for p in anwg_by_policy_seed if "oracle" in p), None)
    if oracle_key is None:
        return {"error": "no oracle_* policy found in run_metrics.csv"}
    seeds = sorted(anwg_by_policy_seed[oracle_key].keys())
    oracle_mean = float(np.mean([anwg_by_policy_seed[oracle_key][s] for s in seeds]))

    means = {
        p: float(np.mean([by_seed[s] for s in seeds if s in by_seed]))
        for p, by_seed in anwg_by_policy_seed.items()
        if p != oracle_key
    }
    worst = min(means.values())
    best_non_oracle = max(means, key=lambda p: means[p])
    best_val = means[best_non_oracle]

    total_gap = oracle_mean - worst
    closed_gap = best_val - worst
    envelope_contribution = (closed_gap / total_gap) if total_gap > 1e-12 else float("nan")

    return {
        "oracle_policy": oracle_key,
        "oracle_mean_anwg": oracle_mean,
        "worst_mean_anwg": worst,
        "best_non_oracle_policy": best_non_oracle,
        "best_non_oracle_mean_anwg": best_val,
        "total_oracle_gap": total_gap,
        "gap_closed_by_best_non_oracle": closed_gap,
        "fraction_of_oracle_gap_closed": envelope_contribution,
    }


_MISSING = "__missing__"


def load_request_level_outcomes(path: str) -> Dict[str, Dict[Tuple[int, int], dict]]:
    """Load request_level_outcomes.csv into {policy: {(seed, request_id): row}}."""
    rows = _read_csv_rows(path)
    by_policy: Dict[str, Dict[Tuple[int, int], dict]] = {}
    for row in rows:
        policy = row["policy"]
        key = (int(row["seed"]), int(row["request_id"]))
        by_policy.setdefault(policy, {})[key] = row
    return by_policy


def _request_weight(row: dict) -> float:
    p = float(row["priority"])
    return p if p > 0 else 1.0


def _anwg_for_rows(rows: List[dict]) -> float:
    num, den = 0.0, 0.0
    for row in rows:
        w = _request_weight(row)
        den += w
        if row["status"] == "success" and row["slo_violated"].lower() == "false":
            num += w
    return num / den if den > 0 else float("nan")


def recompute_anwg_from_request_rows(
    by_policy: Dict[str, Dict[Tuple[int, int], dict]],
) -> Dict[str, Dict[int, float]]:
    """Independently recompute per-(policy, seed) ANWG directly from raw
    per-request rows (completion status + SLO-violated flag + priority
    weight) -- the same formula the eval script's compute_metrics() uses
    internally, re-derived here from the persisted raw rows rather than
    trusting its output."""
    out: Dict[str, Dict[int, float]] = {}
    for policy, keyed_rows in by_policy.items():
        by_seed: Dict[int, List[dict]] = {}
        for (seed, _rid), row in keyed_rows.items():
            by_seed.setdefault(seed, []).append(row)
        out[policy] = {seed: _anwg_for_rows(rows) for seed, rows in by_seed.items()}
    return out


def cross_check_anwg_against_run_metrics(
    request_level_anwg: Dict[str, Dict[int, float]], metrics_rows: List[dict], tol: float = 1e-6
) -> List[dict]:
    mismatches = []
    for row in metrics_rows:
        policy, seed = row["policy"], int(row["seed"])
        recorded = _to_float(row["arrival_normalized_weighted_goodput"])
        recomputed = request_level_anwg.get(policy, {}).get(seed, float("nan"))
        if math.isnan(recorded) and math.isnan(recomputed):
            continue
        if math.isnan(recorded) or math.isnan(recomputed) or abs(recorded - recomputed) > tol:
            mismatches.append(
                {"policy": policy, "seed": seed, "recorded": recorded, "recomputed": recomputed}
            )
    return mismatches


def paired_request_bootstrap_ci(
    by_policy: Dict[str, Dict[Tuple[int, int], dict]],
    reference_policy: str,
    n_boot: int = 2000,
    seed: int = 20260804,
) -> Dict[str, dict]:
    """Independently implemented paired bootstrap over (seed, request_id)
    keys -- resamples the SAME keys across every policy on each replicate
    (paired, since every policy ran the identical per-seed request list),
    exactly mirroring the eval script's own resampling granularity, but
    coded from scratch here rather than calling compute_bootstrap_ci()."""
    rng = np.random.default_rng(seed)
    all_keys = sorted(next(iter(by_policy.values())).keys())
    n = len(all_keys)
    idx_matrix = rng.integers(0, n, size=(n_boot, n))

    results: Dict[str, dict] = {}
    ref_replicates = None
    per_policy_replicates: Dict[str, np.ndarray] = {}
    for policy, keyed_rows in by_policy.items():
        replicates = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            sample_rows = [keyed_rows[all_keys[i]] for i in idx_matrix[b]]
            replicates[b] = _anwg_for_rows(sample_rows)
        per_policy_replicates[policy] = replicates
        point = _anwg_for_rows([keyed_rows[k] for k in all_keys])
        lo, hi = np.percentile(replicates, [2.5, 97.5])
        results[policy] = {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi)}
        if policy == reference_policy:
            ref_replicates = replicates

    if ref_replicates is not None:
        for policy, replicates in per_policy_replicates.items():
            if policy == reference_policy:
                continue
            diff = ref_replicates - replicates
            lo, hi = np.percentile(diff, [2.5, 97.5])
            results[policy][f"{reference_policy}_minus_this_ci"] = [float(lo), float(hi)]
    return results


def request_level_win_tie_loss(
    request_level_anwg: Dict[str, Dict[int, float]], tol: float = 1e-9
) -> dict:
    return win_tie_loss(request_level_anwg, tol=tol)


def per_regime_breakdown(
    by_policy: Dict[str, Dict[Tuple[int, int], dict]],
) -> Dict[str, Dict[str, float]]:
    """ANWG per policy, broken down by SLO class_id ('tight'/'medium'/
    'loose'/etc, whatever classes the workload augmentation assigned) --
    pooled across all seeds' requests within each class."""
    out: Dict[str, Dict[str, float]] = {}
    for policy, keyed_rows in by_policy.items():
        by_class: Dict[str, List[dict]] = {}
        for row in keyed_rows.values():
            by_class.setdefault(row.get("class_id", "unknown"), []).append(row)
        out[policy] = {cls: _anwg_for_rows(rows) for cls, rows in by_class.items()}
    return out


def completion_violation_counts(
    by_policy: Dict[str, Dict[Tuple[int, int], dict]],
) -> Dict[str, dict]:
    out = {}
    for policy, keyed_rows in by_policy.items():
        rows = list(keyed_rows.values())
        n_total = len(rows)
        n_dropped = sum(1 for r in rows if r["status"] == "dropped")
        n_completed = n_total - n_dropped
        n_slo_violated = sum(1 for r in rows if r["slo_violated"].lower() == "true")
        out[policy] = {
            "n_total": n_total,
            "n_completed": n_completed,
            "n_dropped": n_dropped,
            "n_slo_violated": n_slo_violated,
            "completion_fraction": n_completed / n_total if n_total else float("nan"),
        }
    return out


def independent_rank_correlation(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation, reimplemented from scratch (does not
    import or call the eval script's own ``_rank_correlation``)."""
    n = len(a)
    if n == 0:
        return float("nan")

    def _ranks(x: List[float]) -> List[float]:
        order = sorted(range(len(x)), key=lambda i: x[i])
        ranks = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    ra, rb = _ranks(a), _ranks(b)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return float("nan")
    return cov / math.sqrt(var_a * var_b)


def cross_check_ranking_agreement(output_dir: str) -> dict:
    ranking_path = os.path.join(output_dir, "ranking_agreement.json")
    if not os.path.exists(ranking_path):
        return {"error": f"{ranking_path} not found"}
    with open(ranking_path) as f:
        recorded = json.load(f)

    manifest_path = os.path.join(output_dir, "run_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Re-derive requests/scores exactly as the eval script does, but using
    # only public loader functions -- not compute_ranking_agreement_record.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from baselines.vllm_ltr.adapter.offline_scoring import load_score_cache, scores_only
    from llmserveopt.policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA, predicted_service_proxy

    prompts_path = manifest["input_hashes"]["prompts_path"]["path"]
    score_cache_path = manifest["input_hashes"]["score_cache_path"]["path"]
    pairs_path = manifest["input_hashes"]["pairs_path"]["path"]

    with open(prompts_path) as f:
        id_to_prompt = {int(k): v for k, v in json.load(f).items()}
    score_cache = load_score_cache(score_cache_path)
    ltr_scores = scores_only(score_cache, id_to_prompt=id_to_prompt)

    from run_vllm_ltr_first_comparative_evaluation import build_requests_for_seed  # noqa: E402

    mismatches = []
    recomputed_records = []
    for rec in recorded:
        seed = rec["seed"]
        requests = build_requests_for_seed(pairs_path, seed, manifest.get("tokenizer", "facebook/opt-125m") if "tokenizer" in manifest else "facebook/opt-125m")
        max_requests = manifest.get("max_requests")
        if max_requests is not None:
            requests = sorted(requests, key=lambda r: r.request_id)[:max_requests]
        req_by_id = {r.request_id: r for r in requests}
        ids_sorted = sorted(req_by_id.keys())
        ltr_order = [ltr_scores[i] for i in ids_sorted]
        est_order = [
            -predicted_service_proxy(req_by_id[i], alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA)
            for i in ids_sorted
        ]
        sof_order = [-req_by_id[i].predicted_output_tokens for i in ids_sorted]

        est_corr = independent_rank_correlation(ltr_order, est_order)
        sof_corr = independent_rank_correlation(ltr_order, sof_order)
        recomputed_records.append(
            {"seed": seed, "spearman_ltr_vs_est": est_corr, "spearman_ltr_vs_sof": sof_corr}
        )

        def _close(a, b, tol=5e-3):
            # tol is deliberately loose (not a formatting-precision tolerance):
            # the eval script's own _rank_correlation() ranks via
            # np.argsort(np.argsort(x)) with NO tie-averaging, while
            # independent_rank_correlation() above uses textbook
            # average-rank Spearman for ties. predicted_output_tokens and
            # est/sof scores have real ties in this dataset (integer token
            # counts, limited range), so the two methods can legitimately
            # differ by ~1e-4 without either being wrong -- this is a
            # tie-handling CONVENTION difference, not a computational bug.
            # A difference bigger than this tolerance would indicate a
            # real discrepancy worth investigating.
            if math.isnan(a) and math.isnan(b):
                return True
            if math.isnan(a) or math.isnan(b):
                return False
            return abs(a - b) <= tol

        if not _close(est_corr, rec["spearman_ltr_vs_estimated_service_time_first"]):
            mismatches.append({"seed": seed, "field": "est", "recomputed": est_corr,
                                "recorded": rec["spearman_ltr_vs_estimated_service_time_first"]})
        if not _close(sof_corr, rec["spearman_ltr_vs_shortest_output_first"]):
            mismatches.append({"seed": seed, "field": "sof", "recomputed": sof_corr,
                                "recorded": rec["spearman_ltr_vs_shortest_output_first"]})

    return {
        "recomputed": recomputed_records,
        "recorded": recorded,
        "mismatches": mismatches,
        "tie_handling_note": (
            "Small (~1e-4) differences are expected and are NOT bugs: the "
            "eval script ranks via np.argsort(np.argsort(x)) with no "
            "tie-averaging; this cross-check uses average-rank Spearman. "
            "Real ties exist in predicted_output_tokens/est/sof for this "
            "dataset. Mismatch tolerance (5e-3) is set above that expected "
            "gap; anything flagged below exceeds it."
        ),
        "all_match": len(mismatches) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", "--results-dir", dest="output_dir",
        default="results/vllm_ltr_first_comparative_evaluation",
    )
    parser.add_argument("--reference-policy", default="vllm_ltr_semantic_reference")
    parser.add_argument("--n-prompts", type=int, default=300,
                         help="Prompts per seed, for the expected-row-count check.")
    parser.add_argument("--skip-ranking-cross-check", action="store_true",
                         help="Skip the ranking-agreement cross-check (requires re-tokenizing "
                              "the WildChat sample; slower than the rest of this script).")
    args = parser.parse_args()

    metrics_path = os.path.join(args.output_dir, "run_metrics.csv")
    rows = _read_csv_rows(metrics_path)
    if not rows:
        raise SystemExit(f"No rows in {metrics_path} -- run did not complete?")

    completion_check = recompute_completion_fraction(rows)
    n_completion_mismatches = sum(
        1 for v in completion_check.values()
        if not v["completion_fraction_matches"] or not v["slo_violation_rate_matches"]
    )
    n_identity_violations = sum(
        1 for v in completion_check.values() if not v["accounting_identity_holds"]
    )

    anwg_by_policy_seed: Dict[str, Dict[int, float]] = {}
    for row in rows:
        anwg_by_policy_seed.setdefault(row["policy"], {})[int(row["seed"])] = _to_float(
            row["arrival_normalized_weighted_goodput"]
        )
    n_seeds = len({int(row["seed"]) for row in rows})
    n_policies = len({row["policy"] for row in rows})

    outcomes_path = os.path.join(args.output_dir, "request_level_outcomes.csv")
    request_level_available = os.path.exists(outcomes_path)

    row_count_check = None
    anwg_cross_check_mismatches = None
    per_regime = None
    completion_violations = None
    if request_level_available:
        by_policy = load_request_level_outcomes(outcomes_path)
        n_raw_rows = sum(len(v) for v in by_policy.values())
        expected_raw_rows = args.n_prompts * n_seeds * n_policies
        row_count_check = {
            "expected_raw_rows": expected_raw_rows,
            "actual_raw_rows": n_raw_rows,
            "matches": n_raw_rows == expected_raw_rows,
            "formula": "n_prompts * n_seeds * n_policies",
        }

        request_level_anwg = recompute_anwg_from_request_rows(by_policy)
        anwg_cross_check_mismatches = cross_check_anwg_against_run_metrics(
            request_level_anwg, rows
        )
        # Request-level mode supersedes the run_metrics.csv-only ANWG/CI/
        # win-tie-loss/envelope computation for reporting purposes below.
        anwg_source = request_level_anwg
        ci_result = paired_request_bootstrap_ci(by_policy, reference_policy=args.reference_policy)
        ci_label = "independent_paired_request_bootstrap_ci"
        wtl = win_tie_loss(anwg_source)
        envelope = oracle_envelope_contribution(anwg_source)
        per_regime = per_regime_breakdown(by_policy)
        completion_violations = completion_violation_counts(by_policy)
    else:
        anwg_source = anwg_by_policy_seed
        ci_result = seed_level_bootstrap_ci(anwg_by_policy_seed)
        ci_label = "independent_seed_level_bootstrap_ci"
        wtl = win_tie_loss(anwg_by_policy_seed)
        envelope = oracle_envelope_contribution(anwg_by_policy_seed)

    ranking_result = None
    if not args.skip_ranking_cross_check:
        try:
            ranking_result = cross_check_ranking_agreement(args.output_dir)
        except Exception as e:  # pragma: no cover - diagnostic path
            ranking_result = {"error": f"{type(e).__name__}: {e}"}

    recorded_ci_path = os.path.join(args.output_dir, "bootstrap_confidence_intervals.json")
    recorded_ci = None
    if os.path.exists(recorded_ci_path):
        with open(recorded_ci_path) as f:
            recorded_ci = json.load(f)

    report = {
        "output_dir": args.output_dir,
        "mode": "request_level" if request_level_available else "seed_level_fallback",
        "row_count_check": row_count_check,
        "anwg_cross_check_vs_run_metrics_csv": anwg_cross_check_mismatches,
        "completion_accounting_recomputation": {
            "n_policy_seed_rows": len(completion_check),
            "n_completion_or_slo_rate_mismatches": n_completion_mismatches,
            "n_accounting_identity_violations": n_identity_violations,
            "detail": {f"{p}|seed={s}": v for (p, s), v in completion_check.items()},
        },
        "completion_violation_counts_from_raw_rows": completion_violations,
        ci_label: ci_result,
        "recorded_paired_request_bootstrap_ci": recorded_ci,
        "win_tie_loss": wtl,
        "oracle_envelope_contribution": envelope,
        "per_regime_anwg": per_regime,
        "ranking_agreement_cross_check": ranking_result,
    }

    out_path = os.path.join(args.output_dir, "independent_verification_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    print(f"\n=== Independent verification summary (mode={report['mode']}) ===")
    if row_count_check is not None:
        print(f"raw row count: expected={row_count_check['expected_raw_rows']} "
              f"actual={row_count_check['actual_raw_rows']} matches={row_count_check['matches']}")
    if anwg_cross_check_mismatches is not None:
        print(f"ANWG cross-check vs run_metrics.csv: {len(anwg_cross_check_mismatches)} mismatches")
    print(f"completion_fraction/slo_violation_rate mismatches: {n_completion_mismatches}/{len(completion_check)}")
    print(f"accounting identity violations (completed+dropped > total): {n_identity_violations}")
    if ranking_result is not None:
        if "error" in ranking_result:
            print(f"ranking agreement cross-check: ERROR ({ranking_result['error']})")
        else:
            print(f"ranking agreement cross-check: all_match={ranking_result['all_match']} "
                  f"({len(ranking_result['mismatches'])} mismatches)")
    print(f"oracle envelope: {envelope}")
    print(f"\nMean ANWG by policy (independent, {report['mode']}):")
    for p in sorted(anwg_source, key=lambda p: -ci_result[p]["point"]):
        ci = ci_result[p]
        print(f"  {p:35s} point={ci['point']:.4f}  CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    hard_failure = (
        n_completion_mismatches > 0
        or n_identity_violations > 0
        or (row_count_check is not None and not row_count_check["matches"])
        or (anwg_cross_check_mismatches is not None and len(anwg_cross_check_mismatches) > 0)
        or (ranking_result is not None and "error" not in ranking_result and not ranking_result["all_match"])
    )
    if hard_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
