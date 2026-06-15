#!/usr/bin/env python3
"""
Evaluate generated heuristic candidates via simulator and rank by
priority_weighted_slo_goodput.

Usage:
    python scripts/evaluate_generated_heuristics.py \\
        --candidates-dir results/phase2b2_llm_generation/mock_candidates \\
        --output-dir    results/phase2b2_llm_generation/mock_evaluation

    python scripts/evaluate_generated_heuristics.py \\
        --candidates-dir results/phase2b2_llm_generation/real_api_candidates \\
        --output-dir    results/phase2b2_llm_generation/real_api_evaluation \\
        --arrival-rate 20 --duration 30

NOTE: oracle_srtf is NOT included as a deployable baseline.
      RF Selector from Phase 2A.3 was trained on 16-policy set; rerun needed
      before final paper evaluation with 18 policies.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.llm_generation.candidate_io import load_verified_candidates
from llmserveopt.llm_generation.evaluation import EvaluationConfig, evaluate_candidates
from llmserveopt.llm_generation.ranking import (
    build_summary_md,
    rank_candidates,
    save_ranking_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated heuristics and rank by priority_weighted_slo_goodput"
    )
    parser.add_argument("--candidates-dir", required=True,
                        help="Directory containing generated candidate archives")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arrival-rate", type=float, default=15.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-active-sequences", type=int, default=4)
    parser.add_argument("--max-kv-tokens", type=int, default=8192)
    parser.add_argument("--baselines", default="fifo,edf,least_laxity_first,"
                        "estimated_service_time_first,slo_slack_score,"
                        "vllm_style_token_budget,sarathi_style,best_fit",
                        help="Comma-separated baseline policy names")
    args = parser.parse_args()

    candidates_dir = Path(args.candidates_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading candidates from: {candidates_dir}")
    candidate_records = load_verified_candidates(candidates_dir)
    print(f"Found {len(candidate_records)} verified candidate(s)")

    baseline_names = [b.strip() for b in args.baselines.split(",") if b.strip()]

    cfg = EvaluationConfig(
        arrival_rate=args.arrival_rate,
        duration=args.duration,
        seed=args.seed,
        max_active_sequences=args.max_active_sequences,
        max_kv_tokens=args.max_kv_tokens,
        baseline_names=baseline_names,
    )

    print()
    results = evaluate_candidates(candidate_records, cfg)

    all_results = results["heuristics"] + results["baselines"]
    ranked_all = rank_candidates(all_results)
    ranked_h = rank_candidates(results["heuristics"])
    ranked_b = rank_candidates(results["baselines"])

    # Load index for generation stats
    index_path = candidates_dir / "index.csv"
    n_gen = n_ver = n_rep = n_fail = 0
    if index_path.exists():
        with open(index_path) as f:
            for row in csv.DictReader(f):
                n_gen += 1
                if row.get("verification_ok", "").lower() == "true":
                    n_ver += 1
                    if int(row.get("repair_attempts", 0)) > 0:
                        n_rep += 1
                else:
                    n_fail += 1

    # Save outputs
    save_ranking_csv(ranked_all, output_dir / "ranking.csv")

    # Separate CSVs
    save_ranking_csv(ranked_h, output_dir / "candidate_metrics.csv")
    save_ranking_csv(ranked_b, output_dir / "baseline_metrics.csv")

    # Markdown summary
    md = build_summary_md(ranked_all, n_gen, n_ver, n_rep, n_fail)
    (output_dir / "evaluation_summary.md").write_text(md, encoding="utf-8")

    # Print results
    print()
    print("=" * 60)
    print("Ranking: priority_weighted_slo_goodput (all, best first)")
    print("=" * 60)
    for rank, r in enumerate(ranked_all, 1):
        wg = f"{r.priority_weighted_slo_goodput:.4f}" if r.priority_weighted_slo_goodput == r.priority_weighted_slo_goodput else "  nan"
        err = f"  ERROR: {r.error[:60]}" if r.error else ""
        print(f"  {rank:2d}. [{r.source:9s}] {r.name:<40s} WG={wg}{err}")

    print()
    print(f"Outputs written to: {output_dir}")

    # Best heuristic vs best baseline
    if ranked_h and ranked_b:
        best_h = ranked_h[0]
        best_b = ranked_b[0]
        bh_wg = best_h.priority_weighted_slo_goodput
        bb_wg = best_b.priority_weighted_slo_goodput
        if bh_wg == bh_wg and bb_wg == bb_wg:
            delta = bh_wg - bb_wg
            sign = "+" if delta >= 0 else ""
            print()
            print(f"Best heuristic:  {best_h.name} (WG={bh_wg:.4f})")
            print(f"Best baseline:   {best_b.name} (WG={bb_wg:.4f})")
            print(f"Delta:           {sign}{delta:.4f}")


if __name__ == "__main__":
    main()
