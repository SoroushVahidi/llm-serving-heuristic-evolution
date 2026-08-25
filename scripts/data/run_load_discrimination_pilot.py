#!/usr/bin/env python3
"""
Bounded load-discrimination pilot over constructed real/busy/scaled/synthetic windows.

Policies (registry-verified, monolithic-compatible):
  fifo, edf, estimated_service_time_first, scorpio_style_slo_guard, vllm_style_token_budget
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import GPUConfig
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.registry import make_policy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.real_window_construction import (
    WINDOW_ORIGIN_BUSY,
    WINDOW_ORIGIN_NATURAL,
    WINDOW_ORIGIN_SCALED,
    WINDOW_ORIGIN_SYNTHETIC,
    load_window_jsonl,
)

PILOT_POLICIES = [
    "fifo",
    "edf",
    "estimated_service_time_first",
    "scorpio_style_slo_guard",
    "vllm_style_token_budget",
]
DATASETS = [
    "burstgpt_v2",
    "azure_llm_2023",
    "azure_llm_2024",
    "bailian_qwen",
    "mooncake",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def select_windows(run_root: Path) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for ds in DATASETS:
        cat_path = run_root / "windows" / ds / "window_catalog.json"
        if not cat_path.exists():
            raise FileNotFoundError(f"missing catalog: {cat_path}")
        cat = json.loads(cat_path.read_text())
        by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for w in cat["windows"]:
            by_origin[w["window_origin"]].append(w)
        # Stratified caps
        selected.extend(by_origin.get(WINDOW_ORIGIN_NATURAL, [])[:12])
        selected.extend(by_origin.get(WINDOW_ORIGIN_BUSY, [])[:12])
        # up to 8 per load factor among scaled
        scaled = by_origin.get(WINDOW_ORIGIN_SCALED, [])
        per_factor: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for w in scaled:
            per_factor[int(w.get("load_factor", 1))].append(w)
        for _k, rows in sorted(per_factor.items()):
            selected.extend(rows[:8])
        # synthetic
        synth_report = run_root / "windows" / ds / "synthetic_calibration_report.json"
        if synth_report.exists():
            syn = json.loads(synth_report.read_text()).get("windows", [])
            for w in syn[:8]:
                selected.append(
                    {
                        "window_id": w["window_id"],
                        "path": w["path"],
                        "window_origin": WINDOW_ORIGIN_SYNTHETIC,
                        "chronological_split": "train_fit_only",
                        "source_family": "synthetic",
                        "load_factor": 1,
                        "dataset": ds,
                    }
                )
        for w in selected:
            w.setdefault("dataset", ds)
    # Hard cap ~200
    return selected[:200]


def evaluate_window(path: Path, policies: List[str], seed: int = 17) -> Dict[str, Any]:
    meta, reqs = load_window_jsonl(path)
    # Bounded capacity so natural rates can discriminate without needing huge fleets.
    gpus = [
        GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=65536),
        GPUConfig(gpu_id=1, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=65536),
    ]
    sm = ServiceModel(step_size=0.01)
    results = {}
    for pname in policies:
        policy = make_policy(pname, seed=seed)
        metrics = run_policy(
            policy,
            reqs,
            gpus,
            service_model=sm,
            workload_tag=meta.get("window_id", path.stem),
            seed=seed,
            drain_steps=20_000,
        )
        results[pname] = {
            "anwg": float(metrics.arrival_normalized_weighted_goodput),
            "completion_fraction": float(metrics.completion_fraction),
            "num_completed": int(metrics.num_completed),
            "num_dropped": int(metrics.num_dropped),
        }
    anwgs = {p: results[p]["anwg"] for p in policies}
    ordered = sorted(anwgs.items(), key=lambda kv: (-kv[1], kv[0]))
    best, best_v = ordered[0]
    second_v = ordered[1][1] if len(ordered) > 1 else best_v
    margin = best_v - second_v
    exact_tie = abs(best_v - second_v) <= 1e-12
    near_tie = margin <= 0.01
    saturated = all(results[p]["completion_fraction"] >= 0.999 for p in policies) or all(
        math.isnan(results[p]["anwg"]) for p in policies
    )
    return {
        "window_meta": {
            "window_id": meta.get("window_id"),
            "window_origin": meta.get("window_origin"),
            "chronological_split": meta.get("chronological_split"),
            "source_family": meta.get("source_family"),
            "load_factor": meta.get("load_factor"),
            "n_requests": len(reqs),
        },
        "policy_results": results,
        "best_policy": best,
        "best_anwg": best_v,
        "second_anwg": second_v,
        "best_second_margin": margin,
        "exact_tie": exact_tie,
        "near_tie": near_tie,
        "saturated": saturated,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    winners = Counter(r["best_policy"] for r in rows)
    margins = [r["best_second_margin"] for r in rows]
    return {
        "n": len(rows),
        "mean_best_anwg": float(np.mean([r["best_anwg"] for r in rows])),
        "mean_margin": float(np.mean(margins)),
        "exact_tie_rate": float(np.mean([r["exact_tie"] for r in rows])),
        "near_tie_rate": float(np.mean([r["near_tie"] for r in rows])),
        "saturated_rate": float(np.mean([r["saturated"] for r in rows])),
        "n_effective_winner_classes": len(winners),
        "winner_counts": dict(winners),
        "policy_mean_anwg": {
            p: float(np.nanmean([r["policy_results"][p]["anwg"] for r in rows]))
            for p in PILOT_POLICIES
        },
    }


def decide(summary_all: Dict[str, Any], by_group: Dict[str, Any]) -> str:
    sat = summary_all.get("saturated_rate", 1.0)
    winners = summary_all.get("n_effective_winner_classes", 0)
    near = summary_all.get("near_tie_rate", 1.0)
    if summary_all.get("n", 0) < 20:
        return "INVALID_WINDOWS"
    if sat >= 0.85:
        return "STILL_SATURATED"
    if winners >= 2 and near <= 0.75:
        return "READY_FOR_FULL_FINGERPRINT_SWEEP"
    if winners >= 2 or sat < 0.85:
        return "PARTIALLY_READY"
    return "STILL_SATURATED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    out_dir = run_root / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify registries
    for p in PILOT_POLICIES:
        make_policy(p)

    selected = select_windows(run_root)
    results = []
    for i, w in enumerate(selected):
        path = Path(w["path"])
        print(f"[{i+1}/{len(selected)}] {w.get('dataset')} {w['window_id']}", flush=True)
        ev = evaluate_window(path, PILOT_POLICIES)
        ev["dataset"] = w.get("dataset")
        ev["path"] = str(path)
        results.append(ev)

    by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_origin[str(r["window_meta"]["window_origin"])].append(r)
        by_dataset[str(r["dataset"])].append(r)
        by_split[str(r["window_meta"]["chronological_split"])].append(r)

    summary_all = summarize(results)
    report = {
        "utc": utc_now(),
        "git_sha": args.git_sha,
        "policies": PILOT_POLICIES,
        "n_windows_evaluated": len(results),
        "summary_all": summary_all,
        "by_origin": {k: summarize(v) for k, v in by_origin.items()},
        "by_dataset": {k: summarize(v) for k, v in by_dataset.items()},
        "by_split": {k: summarize(v) for k, v in by_split.items()},
        "rows": results,
    }
    decision = decide(summary_all, report["by_origin"])
    report["LOAD_DISCRIMINATION_PILOT"] = decision
    (out_dir / "pilot_results.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# Load discrimination pilot",
        "",
        f"- utc: {report['utc']}",
        f"- n_windows: {report['n_windows_evaluated']}",
        f"- policies: {', '.join(PILOT_POLICIES)}",
        f"- LOAD_DISCRIMINATION_PILOT = {decision}",
        "",
        "## Overall",
        f"- mean_best_anwg: {summary_all.get('mean_best_anwg')}",
        f"- mean_margin: {summary_all.get('mean_margin')}",
        f"- exact_tie_rate: {summary_all.get('exact_tie_rate')}",
        f"- near_tie_rate: {summary_all.get('near_tie_rate')}",
        f"- saturated_rate: {summary_all.get('saturated_rate')}",
        f"- winner_classes: {summary_all.get('n_effective_winner_classes')}",
        f"- winners: {summary_all.get('winner_counts')}",
        "",
        "## By origin",
    ]
    for k, v in report["by_origin"].items():
        md.append(f"- {k}: n={v.get('n')} sat={v.get('saturated_rate')} winners={v.get('winner_counts')}")
    md.append("")
    md.append("## By dataset")
    for k, v in report["by_dataset"].items():
        md.append(f"- {k}: n={v.get('n')} sat={v.get('saturated_rate')} winners={v.get('winner_counts')}")
    (out_dir / "PILOT_REPORT.md").write_text("\n".join(md) + "\n")
    print(f"LOAD_DISCRIMINATION_PILOT = {decision}")


if __name__ == "__main__":
    main()
