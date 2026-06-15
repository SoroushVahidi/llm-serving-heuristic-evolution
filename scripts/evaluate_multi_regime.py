#!/usr/bin/env python3
"""
Multi-regime evaluation of LLM-generated heuristic candidates.

Evaluates verified candidates across multiple synthetic train/validation regimes
and produces search rankings. Test regimes are NOT used here.

Usage:
    python scripts/evaluate_multi_regime.py \\
        --candidates-dir results/phase2b3_llm_search/candidates_main \\
        --output-dir     results/phase2b3_llm_search/evaluation_train_validation

NOTE: oracle_srtf is excluded from all deployable comparisons.
      RF Selector from Phase 2A.3 trained on 16 policies; rerun needed for final paper.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.llm_generation.candidate_io import load_verified_candidates
from llmserveopt.llm_generation.diversity import deduplicate_candidates
from llmserveopt.llm_generation.multi_regime_evaluation import (
    MultiRegimeConfig,
    TRAIN_REGIMES,
    VALIDATION_REGIMES,
    DEFAULT_REGIMES,
    DEFAULT_BASELINES,
    aggregate_regime_results,
    evaluate_multi_regime,
)
from llmserveopt.llm_generation.search_ranking import (
    build_search_summary_md,
    rank_search_results,
    save_per_regime_csv,
    save_search_ranking_csv,
)
from llmserveopt.llm_generation.ranking import rank_candidates, save_ranking_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-regime evaluation of LLM-generated heuristics"
    )
    parser.add_argument("--candidates-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    candidates_dir = Path(args.candidates_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    # Load verified candidates
    print(f"Loading candidates from: {candidates_dir}")
    records = load_verified_candidates(candidates_dir)
    print(f"Found {len(records)} verified candidate(s)")

    # Dedup
    if not args.no_dedup:
        records, removed = deduplicate_candidates(records, verbose=verbose)
        if removed:
            print(f"Deduplicated: removed {len(removed)} exact duplicate(s)")

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    cfg = MultiRegimeConfig(
        regimes=DEFAULT_REGIMES,
        baseline_names=baselines,
        verbose=verbose,
    )

    print(f"\nRunning evaluation on {len(cfg.regimes)} regimes "
          f"({len(TRAIN_REGIMES)} train, {len(VALIDATION_REGIMES)} validation)...")
    regime_results = evaluate_multi_regime(records, cfg, verbose=verbose)

    # Aggregate
    all_records = records  # include baselines implicitly via evaluate_multi_regime
    agg = aggregate_regime_results(regime_results, candidate_records=records)

    regime_names = [rr.regime_name for rr in regime_results]
    train_regime_names = [rr.regime_name for rr in regime_results if rr.split == "train"]
    val_regime_names = [rr.regime_name for rr in regime_results if rr.split == "validation"]

    ranked_all = rank_search_results(agg)
    ranked_h = rank_search_results(agg, source_filter="heuristic")
    ranked_b = rank_search_results(agg, source_filter="baseline")

    # Save all outputs
    save_search_ranking_csv(ranked_all, output_dir / "ranking_overall.csv")
    save_search_ranking_csv(ranked_h, output_dir / "ranking_heuristics.csv")
    save_search_ranking_csv(ranked_b, output_dir / "ranking_baselines.csv")
    save_per_regime_csv(ranked_all, regime_names, output_dir / "candidate_metrics_by_regime.csv")

    # Per-regime flat CSVs (all results in one CSV)
    _save_per_regime_flat(regime_results, output_dir / "candidate_metrics_by_regime_flat.csv")
    _save_per_regime_flat(regime_results, output_dir / "baseline_metrics_by_regime_flat.csv",
                          source_filter="baseline")

    # Load generation stats from index
    n_gen = n_ver = n_rep = n_fail = n_dup = 0
    index_path = candidates_dir / "index.csv"
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

    # Markdown summary
    md = build_search_summary_md(
        ranked_all=ranked_all,
        ranked_heuristics=ranked_h,
        ranked_baselines=ranked_b,
        n_generated=n_gen,
        n_verified=n_ver,
        n_repaired=n_rep,
        n_failed=n_fail,
        n_duplicates=len(removed) if not args.no_dedup else 0,
        regime_names=regime_names,
        train_regime_names=train_regime_names,
        val_regime_names=val_regime_names,
    )
    (output_dir / "evaluation_summary.md").write_text(md, encoding="utf-8")

    # Top candidates folder
    top_dir = output_dir / "top_candidates"
    top_dir.mkdir(exist_ok=True)
    _save_top_candidates(ranked_h[:5], candidates_dir, top_dir, agg, regime_results, baselines)

    # Print results
    print()
    print("=" * 70)
    print("Ranking: validation priority_weighted_slo_goodput (heuristics)")
    print("=" * 70)
    for rank, r in enumerate(ranked_h[:10], 1):
        wg = f"{r.val_mean_wg:.4f}" if r.val_mean_wg == r.val_mean_wg else "  nan"
        gap = f"{r.train_val_gap:+.4f}" if r.train_val_gap == r.train_val_gap else "   nan"
        beats = r.regimes_beating_best_fixed
        total_r = r.n_train_regimes + r.n_val_regimes
        print(f"  {rank:2d}. {r.name:<40s} Val-WG={wg} Gap={gap} Beats={beats}/{total_r}")

    if ranked_h and ranked_b:
        best_h = ranked_h[0]
        best_b = ranked_b[0]
        if best_h.val_mean_wg == best_h.val_mean_wg and best_b.val_mean_wg == best_b.val_mean_wg:
            delta = best_h.val_mean_wg - best_b.val_mean_wg
            print(f"\nBest heuristic:  {best_h.name} (val WG={best_h.val_mean_wg:.4f})")
            print(f"Best baseline:   {best_b.name} (val WG={best_b.val_mean_wg:.4f})")
            print(f"Delta:           {delta:+.4f}")

    print(f"\nOutputs: {output_dir}")


def _save_per_regime_flat(
    regime_results,
    output_path: Path,
    *,
    source_filter: str = None,
) -> None:
    fields = ["regime", "split", "name", "source", "priority_weighted_slo_goodput",
              "slo_violation_rate", "p95_ttft", "p95_latency", "request_throughput",
              "num_completed", "error"]
    import math
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rr in regime_results:
            results = rr.heuristics + rr.baselines
            for r in results:
                if source_filter and r.source != source_filter:
                    continue
                def fmt(v):
                    return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
                w.writerow({
                    "regime": rr.regime_name,
                    "split": rr.split,
                    "name": r.name,
                    "source": r.source,
                    "priority_weighted_slo_goodput": fmt(r.priority_weighted_slo_goodput),
                    "slo_violation_rate": fmt(r.slo_violation_rate),
                    "p95_ttft": fmt(r.p95_ttft),
                    "p95_latency": fmt(r.p95_latency),
                    "request_throughput": fmt(r.request_throughput),
                    "num_completed": r.num_completed,
                    "error": r.error or "",
                })


def _save_top_candidates(
    top_ranked, candidates_dir: Path, top_dir: Path,
    agg, regime_results, baselines,
) -> None:
    import shutil, math
    for rank, r in enumerate(top_ranked, 1):
        # Find candidate JSON in archive
        for d in sorted(candidates_dir.iterdir()):
            if not d.is_dir():
                continue
            cand_path = d / "candidate.json"
            if not cand_path.exists():
                continue
            try:
                cand = json.loads(cand_path.read_text())
                if cand.get("name") == r.name:
                    dest = top_dir / f"rank{rank:02d}_{r.name[:40].replace(' ', '_')}.json"
                    shutil.copy2(cand_path, dest)
                    # Write markdown summary
                    _write_candidate_summary(
                        top_dir / f"rank{rank:02d}_{r.name[:40].replace(' ', '_')}.md",
                        rank, r, cand, d, agg, regime_results, baselines,
                    )
                    break
            except Exception:
                continue


def _write_candidate_summary(
    path: Path, rank: int, agg_r, cand: dict,
    source_dir: Path, agg, regime_results, baselines,
) -> None:
    import math
    meta_path = source_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        f"# Candidate Rank {rank}: {agg_r.name}",
        "",
        "## Metadata",
        f"- Provider: {meta.get('provider', 'unknown')}",
        f"- Model: {meta.get('model', 'unknown')}",
        f"- Design target: {meta.get('design_target', meta.get('extra', {}).get('design_target', 'unknown') if isinstance(meta.get('extra', {}), dict) else 'unknown')}",
        f"- Temperature: {meta.get('temperature', 'unknown')}",
        f"- Verification OK: {meta.get('verification_ok', 'unknown')}",
        f"- Repair attempts: {meta.get('repair_attempt_count', 0)}",
        "",
        "## Aggregate Performance",
        f"- Val mean WG: {fmt(agg_r.val_mean_wg)}",
        f"- Train mean WG: {fmt(agg_r.train_mean_wg)}",
        f"- Train/val gap: {fmt(agg_r.train_val_gap)}",
        f"- Worst regime WG: {fmt(agg_r.worst_regime_wg)} ({agg_r.worst_regime_name})",
        f"- Regimes beating best fixed: {agg_r.regimes_beating_best_fixed}",
        f"- Regimes beating slo_slack_score: {agg_r.regimes_beating_slo_slack}",
        f"- Regimes beating estimated_service_time_first: {agg_r.regimes_beating_estf}",
        "",
        "## Per-Regime WG",
    ]
    for rname, wg in sorted(agg_r.per_regime.items()):
        lines.append(f"- {rname}: {fmt(wg)}")
    lines += [
        "",
        "## DSL",
        "```json",
        json.dumps(cand, indent=2)[:2000],
        "```",
    ]
    if len(json.dumps(cand)) > 2000:
        lines.append("...(truncated)")
    lines += [
        "",
        "## Overfitting concern",
    ]
    gap = agg_r.train_val_gap
    if not math.isnan(gap) and gap < -0.02:
        lines.append(f"- WARNING: train_val_gap={fmt(gap)} — possible overfitting to train regimes")
    else:
        lines.append("- None detected.")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
