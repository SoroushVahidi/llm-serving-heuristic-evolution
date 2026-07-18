#!/usr/bin/env python3
"""Analyze a flattened Selector Dataset v2 pilot CSV."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def _f(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _summary(values: list[float]) -> dict:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": vals[0],
        "p25": vals[int(0.25 * (len(vals) - 1))],
        "p50": median(vals),
        "p75": vals[int(0.75 * (len(vals) - 1))],
        "p90": vals[int(0.90 * (len(vals) - 1))],
        "max": vals[-1],
        "mean": mean(vals),
    }


def analyze_rows(rows: list[dict]) -> dict:
    by_window: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_window[(row["scenario_id"], row["window_id"])].append(row)

    class_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    all_complete_by_family: Counter[str] = Counter()
    strong_by_family: Counter[str] = Counter()
    differentiated_metrics: Counter[str] = Counter()
    window_records: list[dict] = []
    scorpio_strong_gap_records: list[dict] = []

    for (_scenario_id, _window_id), wrs in by_window.items():
        first = wrs[0]
        wg_class = first.get("disc_weighted_goodput_classification", "")
        winner = first.get("disc_weighted_goodput_best_policy", "")
        class_counts[wg_class] += 1
        winner_counts[winner] += 1
        family = first["scenario_family_id"]
        family_counts[family] += 1
        if wg_class == "ALL_COMPLETE_OR_EFFECTIVELY_TIED":
            all_complete_by_family[family] += 1
        if wg_class == "STRONGLY_DISCRIMINATIVE":
            strong_by_family[family] += 1

        for objective in [
            "weighted_goodput",
            "arrival_normalized_weighted_goodput",
            "p95_latency",
            "slo_attainment",
            "request_throughput",
        ]:
            spread = _f(first.get(f"disc_{objective}_max_min_spread"))
            if not math.isnan(spread) and spread > 0.002:
                differentiated_metrics[objective] += 1

        completions = [_f(r.get("metric_completion_fraction")) for r in wrs]
        weighted_goodput = [_f(r.get("metric_weighted_goodput")) for r in wrs]
        p95_latency = [_f(r.get("metric_p95_latency")) for r in wrs]
        by_policy = {r["policy_name"]: r for r in wrs}
        window_records.append({
            "family": family,
            "pool": first.get("scenario_pool"),
            "bottleneck": first.get("bottleneck_class"),
            "source_trace": first.get("source_trace"),
            "class": wg_class,
            "winner": winner,
            "winner_margin": _f(first.get("disc_weighted_goodput_absolute_margin")),
            "top2_gap": _f(first.get("disc_weighted_goodput_absolute_margin")),
            "max_min_spread": _f(first.get("disc_weighted_goodput_max_min_spread")),
            "offered_load_estimate": _f(first.get("feat_saturation_load_estimate")),
            "realized_arrival_rate": _f(first.get("feat_arrival_rate_prefix")),
            "queue_buildup": _f(first.get("feat_recent_queue_growth_rate")),
            "kv_pressure": _safe_ratio(_f(first.get("feat_pred_output_p95")), _f(first.get("feat_resource_kv_capacity"))),
            "token_budget_pressure": _safe_ratio(_f(first.get("feat_prompt_p95")), _f(first.get("feat_resource_token_budget"))),
            "slo_tightness": _f(first.get("feat_p10_slack")),
            "arrival_burstiness": _f(first.get("feat_burstiness_cv")),
            "prompt_mean": _f(first.get("feat_prompt_mean")),
            "pred_output_mean": _f(first.get("feat_pred_output_mean")),
            "min_completion_fraction": min((v for v in completions if not math.isnan(v)), default=math.nan),
            "max_completion_fraction": max((v for v in completions if not math.isnan(v)), default=math.nan),
            "min_weighted_goodput": min((v for v in weighted_goodput if not math.isnan(v)), default=math.nan),
            "max_weighted_goodput": max((v for v in weighted_goodput if not math.isnan(v)), default=math.nan),
            "max_p95_latency": max((v for v in p95_latency if not math.isnan(v)), default=math.nan),
        })
        if wg_class == "STRONGLY_DISCRIMINATIVE" and winner == "scorpio_style_slo_guard":
            scorpio = by_policy.get("scorpio_style_slo_guard")
            alternatives = [r for r in wrs if r["policy_name"] != "scorpio_style_slo_guard"]
            if scorpio and alternatives:
                second = max(alternatives, key=lambda r: _f(r.get("metric_weighted_goodput")))
                best_completion = max(alternatives, key=lambda r: _f(r.get("metric_completion_fraction")))
                scorpio_strong_gap_records.append(_policy_gap_record(
                    first,
                    scorpio,
                    second,
                    best_completion,
                ))

    total = len(window_records)
    feature_fields = [
        "offered_load_estimate",
        "realized_arrival_rate",
        "queue_buildup",
        "kv_pressure",
        "token_budget_pressure",
        "slo_tightness",
        "arrival_burstiness",
        "prompt_mean",
        "pred_output_mean",
        "winner_margin",
        "max_min_spread",
        "min_completion_fraction",
        "max_p95_latency",
    ]
    by_class = {}
    for cls in sorted(class_counts):
        rows_for_class = [r for r in window_records if r["class"] == cls]
        by_class[cls] = {
            "windows": len(rows_for_class),
            "feature_summaries": {
                field: _summary([r[field] for r in rows_for_class])
                for field in feature_fields
            },
            "winner_counts": dict(Counter(r["winner"] for r in rows_for_class)),
            "family_counts": dict(Counter(r["family"] for r in rows_for_class)),
        }

    return {
        "num_windows": total,
        "class_counts": dict(class_counts),
        "class_fractions": {k: v / total for k, v in class_counts.items()} if total else {},
        "policy_win_distribution": dict(winner_counts),
        "family_counts": dict(family_counts),
        "all_complete_by_family": dict(all_complete_by_family),
        "strongly_discriminative_by_family": dict(strong_by_family),
        "differentiated_metrics": dict(differentiated_metrics),
        "overall_feature_summaries": {
            field: _summary([r[field] for r in window_records])
            for field in feature_fields
        },
        "by_discriminativeness": by_class,
        "failure_cause_evidence": _failure_cause_evidence(window_records),
        "scorpio_strong_win_diagnostics": _scorpio_diagnostics(scorpio_strong_gap_records),
    }


def _safe_ratio(num: float, denom: float) -> float:
    if math.isnan(num) or math.isnan(denom) or denom <= 0:
        return math.nan
    return num / denom


def _failure_cause_evidence(window_records: list[dict]) -> dict:
    all_complete = [r for r in window_records if r["class"] == "ALL_COMPLETE_OR_EFFECTIVELY_TIED"]
    non_all = [r for r in window_records if r["class"] != "ALL_COMPLETE_OR_EFFECTIVELY_TIED"]
    return {
        "all_complete_windows": len(all_complete),
        "non_all_complete_windows": len(non_all),
        "all_complete_min_completion_fraction": _summary([r["min_completion_fraction"] for r in all_complete]),
        "all_complete_slo_tightness": _summary([r["slo_tightness"] for r in all_complete]),
        "all_complete_token_budget_pressure": _summary([r["token_budget_pressure"] for r in all_complete]),
        "all_complete_kv_pressure": _summary([r["kv_pressure"] for r in all_complete]),
        "non_all_slo_tightness": _summary([r["slo_tightness"] for r in non_all]),
        "non_all_token_budget_pressure": _summary([r["token_budget_pressure"] for r in non_all]),
        "non_all_kv_pressure": _summary([r["kv_pressure"] for r in non_all]),
    }


def _policy_gap_record(window: dict, scorpio: dict, second: dict, best_completion: dict) -> dict:
    def gap(field: str, other: dict = second) -> float:
        return _f(scorpio.get(field)) - _f(other.get(field))

    return {
        "scenario_family_id": window.get("scenario_family_id"),
        "source_trace": window.get("source_trace"),
        "bottleneck_class": window.get("bottleneck_class"),
        "second_policy": second.get("policy_name"),
        "best_completion_policy": best_completion.get("policy_name"),
        "offered_load_estimate": _f(window.get("feat_saturation_load_estimate")),
        "realized_arrival_rate": _f(window.get("feat_arrival_rate_prefix")),
        "queue_growth": _f(window.get("feat_recent_queue_growth_rate")),
        "kv_pressure": _safe_ratio(_f(window.get("feat_pred_output_p95")), _f(window.get("feat_resource_kv_capacity"))),
        "token_budget_pressure": _safe_ratio(_f(window.get("feat_prompt_p95")), _f(window.get("feat_resource_token_budget"))),
        "p10_slack": _f(window.get("feat_p10_slack")),
        "burstiness_cv": _f(window.get("feat_burstiness_cv")),
        "prompt_mean": _f(window.get("feat_prompt_mean")),
        "pred_output_mean": _f(window.get("feat_pred_output_mean")),
        "weighted_goodput_gap_vs_second": gap("metric_weighted_goodput"),
        "completion_fraction_gap_vs_second": gap("metric_completion_fraction"),
        "completion_fraction_gap_vs_best_completion": gap("metric_completion_fraction", best_completion),
        "admission_rate_gap_vs_second": gap("metric_admission_rate"),
        "rejection_rate_gap_vs_second": gap("metric_rejection_rate"),
        "slo_attainment_gap_vs_second": gap("metric_slo_attainment"),
        "p95_latency_gap_vs_second": gap("metric_p95_latency"),
        "p95_ttft_gap_vs_second": gap("metric_p95_ttft"),
        "p95_tpot_gap_vs_second": gap("metric_p95_tpot"),
        "preemption_gap_vs_second": gap("metric_num_preempt_events"),
    }


def _scorpio_diagnostics(records: list[dict]) -> dict:
    if not records:
        return {"windows": 0}
    fields = [
        "offered_load_estimate",
        "realized_arrival_rate",
        "queue_growth",
        "kv_pressure",
        "token_budget_pressure",
        "p10_slack",
        "burstiness_cv",
        "prompt_mean",
        "pred_output_mean",
        "weighted_goodput_gap_vs_second",
        "completion_fraction_gap_vs_second",
        "completion_fraction_gap_vs_best_completion",
        "admission_rate_gap_vs_second",
        "rejection_rate_gap_vs_second",
        "slo_attainment_gap_vs_second",
        "p95_latency_gap_vs_second",
        "p95_ttft_gap_vs_second",
        "p95_tpot_gap_vs_second",
        "preemption_gap_vs_second",
    ]
    return {
        "windows": len(records),
        "second_policy_counts": dict(Counter(r["second_policy"] for r in records)),
        "best_completion_policy_counts": dict(Counter(r["best_completion_policy"] for r in records)),
        "bottleneck_counts": dict(Counter(r["bottleneck_class"] for r in records)),
        "field_summaries": {
            field: _summary([r[field] for r in records])
            for field in fields
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.input)))
    report = analyze_rows(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps({
        "input": args.input,
        "output": args.output,
        "num_windows": report["num_windows"],
        "class_fractions": report["class_fractions"],
        "policy_win_distribution": report["policy_win_distribution"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
