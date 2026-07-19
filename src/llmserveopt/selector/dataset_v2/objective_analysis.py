"""Objective-audit utilities for Selector Dataset v2.

These helpers operate on flattened Dataset v2 rows so existing generated pilot
CSVs can be re-scored without rerunning simulations. Historical
``weighted_goodput`` semantics are preserved; corrected objectives are added as
separate names.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence


HISTORICAL_CONDITIONAL_OBJECTIVE = "weighted_goodput"
ARRIVAL_NORMALIZED_OBJECTIVE = "arrival_normalized_weighted_goodput"
COMPLETION_ADJUSTED_OBJECTIVE = "completion_adjusted_weighted_goodput"
SLO_SUCCESS_THROUGHPUT_OBJECTIVE = "slo_success_throughput"
CONSTRAINED_ARRIVAL_NORMALIZED_OBJECTIVE = "constrained_arrival_normalized_weighted_goodput"


@dataclass(frozen=True)
class ConstrainedRankingConfig:
    min_completion_fraction: float | None = None
    max_rejection_fraction: float | None = None

    @property
    def key(self) -> str:
        beta = "none" if self.min_completion_fraction is None else f"{self.min_completion_fraction:.2f}"
        rho = "none" if self.max_rejection_fraction is None else f"{self.max_rejection_fraction:.2f}"
        return f"cf_ge_{beta}__rej_le_{rho}"


def constrained_policy_scores(
    policy_rows: Sequence[Mapping[str, object]],
    config: ConstrainedRankingConfig,
    *,
    score_field: str = f"metric_{ARRIVAL_NORMALIZED_OBJECTIVE}",
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in policy_rows:
        completion = _f(row.get("metric_completion_fraction"))
        rejection = _rejection_fraction(row)
        if config.min_completion_fraction is not None:
            if math.isnan(completion) or completion < config.min_completion_fraction:
                continue
        if config.max_rejection_fraction is not None:
            if math.isnan(rejection) or rejection > config.max_rejection_fraction:
                continue
        value = _f(row.get(score_field))
        if not math.isnan(value):
            scores[str(row["policy_name"])] = value
    return scores


def constrained_winner(
    policy_rows: Sequence[Mapping[str, object]],
    config: ConstrainedRankingConfig,
) -> tuple[str | None, float | None]:
    scores = constrained_policy_scores(policy_rows, config)
    if not scores:
        return None, None
    winner = max(scores, key=scores.get)
    return winner, scores[winner]


def sensitivity_grid(
    rows: Sequence[Mapping[str, object]],
    *,
    completion_minimums: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    maximum_rejections: Sequence[float] = (0.5, 0.3, 0.2, 0.1),
) -> dict[str, dict]:
    windows = _group_windows(rows)
    out: dict[str, dict] = {}
    for beta in completion_minimums:
        for rho in maximum_rejections:
            config = ConstrainedRankingConfig(beta, rho)
            wins: Counter[str] = Counter()
            eligible_counts: list[int] = []
            oracle_values: list[float] = []
            for policy_rows in windows.values():
                scores = constrained_policy_scores(policy_rows, config)
                eligible_counts.append(len(scores))
                if not scores:
                    continue
                winner = max(scores, key=scores.get)
                wins[winner] += 1
                oracle_values.append(scores[winner])
            out[config.key] = {
                "winner_counts": dict(wins),
                "windows_with_eligible_policy": sum(1 for c in eligible_counts if c > 0),
                "mean_eligible_policy_count": (
                    sum(eligible_counts) / len(eligible_counts) if eligible_counts else 0.0
                ),
                "oracle_score": (
                    sum(oracle_values) / len(oracle_values) if oracle_values else None
                ),
            }
    return out


def objective_summary(
    rows: Sequence[Mapping[str, object]],
    objective_name: str,
    *,
    config: ConstrainedRankingConfig | None = None,
    practical_epsilon: float = 0.002,
    strong_abs_margin: float = 0.02,
) -> dict:
    windows = _group_windows(rows)
    winner_counts: Counter[str] = Counter()
    strong_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    oracle_values: list[float] = []
    winner_completions: list[float] = []
    winner_rejections: list[float] = []
    policy_values: dict[str, list[float]] = defaultdict(list)

    for policy_rows in windows.values():
        scores = _scores_for_objective(policy_rows, objective_name, config=config)
        if len(scores) < 1:
            continue
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        winner, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else best
        margin = best - second
        spread = max(scores.values()) - min(scores.values())
        cls = _classify_margin(best, margin, spread, practical_epsilon, strong_abs_margin)
        class_counts[cls] += 1
        winner_counts[winner] += 1
        if cls == "STRONGLY_DISCRIMINATIVE":
            strong_counts[winner] += 1
        oracle_values.append(best)
        winner_row = next(r for r in policy_rows if r["policy_name"] == winner)
        completion = _f(winner_row.get("metric_completion_fraction"))
        rejection = _rejection_fraction(winner_row)
        if not math.isnan(completion):
            winner_completions.append(completion)
        if not math.isnan(rejection):
            winner_rejections.append(rejection)
        for policy, value in scores.items():
            policy_values[policy].append(value)

    total = sum(class_counts.values())
    if objective_name == CONSTRAINED_ARRIVAL_NORMALIZED_OBJECTIVE:
        means = {
            p: sum(vals) / len(vals)
            for p, vals in policy_values.items()
            if vals and len(vals) == total
        }
    else:
        means = {p: sum(vals) / len(vals) for p, vals in policy_values.items() if vals}
    best_fixed = max(means, key=means.get) if means else None
    oracle = sum(oracle_values) / len(oracle_values) if oracle_values else None
    return {
        "objective": objective_name if config is None else f"{objective_name}:{config.key}",
        "window_count": total,
        "global_best_fixed_policy": best_fixed,
        "global_best_fixed_score": means.get(best_fixed) if best_fixed else None,
        "per_window_oracle_score": oracle,
        "oracle_headroom": (
            oracle - means[best_fixed] if oracle is not None and best_fixed is not None else None
        ),
        "class_counts": dict(class_counts),
        "class_fractions": {k: v / total for k, v in class_counts.items()} if total else {},
        "policy_win_distribution": dict(winner_counts),
        "strong_policy_win_distribution": dict(strong_counts),
        "faithful_baseline_wins": {
            "vllm_faithful": winner_counts.get("vllm_faithful", 0),
            "sarathi_faithful": winner_counts.get("sarathi_faithful", 0),
        },
        "mean_winner_completion_fraction": (
            sum(winner_completions) / len(winner_completions) if winner_completions else None
        ),
        "mean_winner_rejection_fraction": (
            sum(winner_rejections) / len(winner_rejections) if winner_rejections else None
        ),
    }


def selective_service_advantages(
    rows: Sequence[Mapping[str, object]],
    *,
    historical_margin_epsilon: float = 0.002,
    corrected_tie_epsilon: float = 0.002,
    completion_gap: float = 0.05,
    rejection_gap: float = 0.05,
    max_examples: int = 12,
) -> dict:
    windows = _group_windows(rows)
    examples: list[dict] = []
    count = 0
    by_policy: Counter[str] = Counter()
    for key, policy_rows in windows.items():
        for a in policy_rows:
            for b in policy_rows:
                if a["policy_name"] == b["policy_name"]:
                    continue
                hist_a = _f(a.get(f"metric_{HISTORICAL_CONDITIONAL_OBJECTIVE}"))
                hist_b = _f(b.get(f"metric_{HISTORICAL_CONDITIONAL_OBJECTIVE}"))
                corr_a = _f(a.get(f"metric_{ARRIVAL_NORMALIZED_OBJECTIVE}"))
                corr_b = _f(b.get(f"metric_{ARRIVAL_NORMALIZED_OBJECTIVE}"))
                if any(math.isnan(v) for v in [hist_a, hist_b, corr_a, corr_b]):
                    continue
                if hist_a <= hist_b + historical_margin_epsilon:
                    continue
                comp_a = _f(a.get("metric_completion_fraction"))
                comp_b = _f(b.get("metric_completion_fraction"))
                rej_a = _rejection_fraction(a)
                rej_b = _rejection_fraction(b)
                selective = (
                    (not math.isnan(comp_a) and not math.isnan(comp_b) and comp_a + completion_gap < comp_b)
                    or (not math.isnan(rej_a) and not math.isnan(rej_b) and rej_a > rej_b + rejection_gap)
                )
                if not selective:
                    continue
                if corr_a <= corr_b + corrected_tie_epsilon:
                    count += 1
                    by_policy[str(a["policy_name"])] += 1
                    if len(examples) < max_examples:
                        examples.append({
                            "scenario_id": key[0],
                            "window_id": key[1],
                            "historical_winner": a["policy_name"],
                            "alternative_policy": b["policy_name"],
                            "historical_score_gap": hist_a - hist_b,
                            "arrival_normalized_gap": corr_a - corr_b,
                            "completion_fraction_gap": comp_a - comp_b,
                            "rejection_fraction_gap": rej_a - rej_b,
                            "bottleneck_class": a.get("bottleneck_class"),
                            "source_trace": a.get("source_trace"),
                        })
    return {
        "SELECTIVE_SERVICE_ADVANTAGE": count,
        "by_advantaged_policy": dict(by_policy),
        "examples": examples,
    }


def _scores_for_objective(
    policy_rows: Sequence[Mapping[str, object]],
    objective_name: str,
    *,
    config: ConstrainedRankingConfig | None = None,
) -> dict[str, float]:
    if objective_name == CONSTRAINED_ARRIVAL_NORMALIZED_OBJECTIVE:
        if config is None:
            raise ValueError("constrained objective requires a ConstrainedRankingConfig")
        return constrained_policy_scores(policy_rows, config)

    scores: dict[str, float] = {}
    for row in policy_rows:
        value = _objective_value(row, objective_name)
        if not math.isnan(value):
            scores[str(row["policy_name"])] = value
    return scores


def _objective_value(row: Mapping[str, object], objective_name: str) -> float:
    if objective_name == COMPLETION_ADJUSTED_OBJECTIVE:
        return _f(row.get(f"metric_{HISTORICAL_CONDITIONAL_OBJECTIVE}")) * _f(row.get("metric_completion_fraction"))
    if objective_name == SLO_SUCCESS_THROUGHPUT_OBJECTIVE:
        return _f(row.get("metric_request_throughput")) * _f(row.get(f"metric_{HISTORICAL_CONDITIONAL_OBJECTIVE}"))
    return _f(row.get(f"metric_{objective_name}"))


def _group_windows(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], list[Mapping[str, object]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario_id"]), str(row["window_id"]))].append(row)
    return grouped


def _classify_margin(
    best: float,
    margin: float,
    spread: float,
    practical_epsilon: float,
    strong_abs_margin: float,
) -> str:
    if spread <= practical_epsilon:
        return "ALL_COMPLETE_OR_EFFECTIVELY_TIED"
    if margin <= practical_epsilon:
        return "NEAR_TIE"
    rel = margin / max(abs(best), 1e-9)
    if margin >= strong_abs_margin or rel >= 0.03:
        return "STRONGLY_DISCRIMINATIVE"
    return "MODERATELY_DISCRIMINATIVE"


def _rejection_fraction(row: Mapping[str, object]) -> float:
    value = _f(row.get("metric_rejection_fraction"))
    if not math.isnan(value):
        return value
    return _f(row.get("metric_rejection_rate"))


def _f(value: object) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
