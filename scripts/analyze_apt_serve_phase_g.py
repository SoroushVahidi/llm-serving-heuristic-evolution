#!/usr/bin/env python3
"""Deterministic post-hoc analysis for Apt-Serve Phase G.

This script analyzes an already-collected Phase G run directory. It does
not run simulations, modify collection artifacts, or update status docs.
Outputs are written incrementally and atomically so the job can be resumed
after interruption.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_TRANSITION_COST = "1x"
EXPECTED_BASELINES = [
    "fifo",
    "edf",
    "weighted_shortest_processing",
    "least_laxity_first",
    "estimated_service_time_first",
    "scorpio_style_slo_guard",
    "vllm_style_token_budget",
    "sarathi_style",
    "orca_style",
    "shortest_output_first",
    "slo_slack_score",
    "admission_control",
]
EXPECTED_TRANSITION_COSTS = ["0x_idealized", "0.5x", "1x", "2x", "4x"]
EPSILONS = [0.0, 0.005, 0.01, 0.05]
STAGES = [
    "dataset_validation",
    "global_policy_summary",
    "regime_summary",
    "marginal_contribution_summary",
    "stage1_stage2_replication",
    "transition_cost_analysis",
    "mechanism_analysis",
    "grouped_bootstrap_results",
    "final_summary",
]


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass(frozen=True)
class UnitSummary:
    stage: str
    regime_id: str
    seed: int
    kv_pressure: str
    slo_pattern: str
    length_pattern: str
    arrival_pattern: str
    cache_use_structure: str
    n_requests: int
    best_baseline_policy: str
    best_baseline_anwg: float
    apt_primary_anwg: float
    gap_vs_best_baseline: float
    marginal_contribution: float
    apt_unique_winner: bool
    apt_completion_fraction: float
    apt_slo_violation_rate: float
    total_transitions: float
    transitions_per_completed_request: float
    evictions: float
    recomputations: float
    switch_latency_paid: float
    restore_latency_paid: float
    wall_time_sec: float


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def git_state_text() -> str:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT).decode().strip()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT).decode()
        upstream = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        upstream_sha = subprocess.check_output(["git", "rev-parse", "@{u}"], cwd=REPO_ROOT).decode().strip()
        ahead_behind = subprocess.check_output(
            ["git", "rev-list", "--left-right", "--count", "HEAD...@{u}"],
            cwd=REPO_ROOT,
        ).decode().strip()
        return (
            f"branch={branch}\nHEAD={head}\nupstream={upstream}\n"
            f"upstream_sha={upstream_sha}\nahead_behind={ahead_behind}\nstatus:\n{status}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"git_state unavailable: {exc}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), p))


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def cell_key(cell: dict[str, Any]) -> tuple[str, str]:
    return str(cell.get("policy_label")), str(cell.get("transition_cost"))


def validate_dataset(records: list[dict[str, Any]], source_run_dir: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    unit_keys = [(r.get("stage"), r.get("regime_id"), r.get("seed")) for r in records]
    duplicates = [key for key, count in Counter(unit_keys).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate unit keys: {duplicates[:10]}")

    expected_cell_keys = {(p, "na") for p in EXPECTED_BASELINES}
    expected_cell_keys.update({("apt_serve_faithful", tc) for tc in EXPECTED_TRANSITION_COSTS})
    failures = 0
    critical_failures = 0
    malformed_cells = 0
    nan_inf = 0
    impossible_values = 0
    stage_counts = Counter()
    regime_counts_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    seed_sets_by_stage: dict[str, set[int]] = defaultdict(set)
    policy_counts = Counter()
    transition_counts = Counter()

    for rec in records:
        stage = str(rec.get("stage"))
        regime_id = str(rec.get("regime_id"))
        seed = int(rec.get("seed"))
        stage_counts[stage] += 1
        regime_counts_by_stage[stage][regime_id] += 1
        seed_sets_by_stage[stage].add(seed)
        if rec.get("failures"):
            failures += len(rec["failures"])
        if rec.get("critical_failure") is not None:
            critical_failures += 1
        cells = rec.get("cells")
        if not isinstance(cells, list):
            errors.append(f"{stage}/{regime_id}/{seed}: cells is not a list")
            continue
        labels = [cell_key(c) for c in cells]
        if set(labels) != expected_cell_keys:
            malformed_cells += 1
        if len(labels) != len(set(labels)):
            malformed_cells += 1
        for cell in cells:
            policy_counts[str(cell.get("policy_label"))] += 1
            transition_counts[str(cell.get("transition_cost"))] += 1
            completed = cell.get("num_completed")
            dropped = cell.get("num_dropped")
            total = cell.get("num_total")
            if completed is None or dropped is None or total is None or completed + dropped != total:
                impossible_values += 1
            for metric in (
                "completion_fraction",
                "arrival_normalized_weighted_goodput",
                "weighted_goodput_completed_only",
                "slo_violation_rate",
                "mean_latency",
                "p95_latency",
                "mean_ttft",
                "p95_ttft",
                "request_throughput",
                "token_throughput",
            ):
                value = cell.get(metric)
                if value is None or not finite(value):
                    nan_inf += 1
            cf = cell.get("completion_fraction")
            slo = cell.get("slo_violation_rate")
            anwg = cell.get("arrival_normalized_weighted_goodput")
            if finite(cf) and not (0.0 <= float(cf) <= 1.0):
                impossible_values += 1
            if finite(slo) and not (0.0 <= float(slo) <= 1.0):
                impossible_values += 1
            if finite(anwg) and float(anwg) < -1e-12:
                impossible_values += 1

    for stage, regime_counts in regime_counts_by_stage.items():
        seed_count = len(seed_sets_by_stage[stage])
        expected = len(regime_counts) * seed_count
        actual = stage_counts[stage]
        if expected != actual:
            errors.append(f"{stage}: expected cartesian {expected} units from regimes x seeds, found {actual}")

    source_summary: dict[str, Any] = {}
    if source_run_dir is not None and (source_run_dir / "results.jsonl").exists():
        source_records = load_jsonl(source_run_dir / "results.jsonl")
        source_keys = {(r["stage"], r["regime_id"], r["seed"]) for r in source_records}
        run_keys = {(r["stage"], r["regime_id"], r["seed"]) for r in records}
        source_critical = [r for r in source_records if r.get("critical_failure") is not None]
        source_summary = {
            "source_run_dir": str(source_run_dir.resolve()),
            "source_units": len(source_records),
            "source_keys": len(source_keys),
            "source_critical_failures": len(source_critical),
            "source_keys_present_in_run": len(source_keys & run_keys),
            "source_keys_not_present_in_run": len(source_keys - run_keys),
        }
    else:
        warnings.append("source run directory unavailable; provenance boundary not cross-checked")

    valid = not errors and failures == 0 and critical_failures == 0 and malformed_cells == 0 and nan_inf == 0 and impossible_values == 0
    return {
        "valid": valid,
        "classification": "STRUCTURALLY_VALID" if valid else "INVALID",
        "errors": errors,
        "warnings": warnings,
        "total_units": len(records),
        "stage_counts": dict(stage_counts),
        "regime_counts_by_stage": {k: dict(v) for k, v in regime_counts_by_stage.items()},
        "seed_ranges_by_stage": {
            stage: {
                "min": min(seeds) if seeds else None,
                "max": max(seeds) if seeds else None,
                "count": len(seeds),
            }
            for stage, seeds in seed_sets_by_stage.items()
        },
        "duplicate_unit_keys": len(duplicates),
        "failures": failures,
        "critical_failures": critical_failures,
        "malformed_units": malformed_cells,
        "nan_inf_values": nan_inf,
        "impossible_values": impossible_values,
        "policy_counts": dict(policy_counts),
        "transition_counts": dict(transition_counts),
        "source_summary": source_summary,
    }


def build_unit_summaries(records: list[dict[str, Any]]) -> list[UnitSummary]:
    summaries: list[UnitSummary] = []
    for rec in records:
        cells = rec["cells"]
        baselines = [c for c in cells if c["policy_kind"] == "baseline"]
        apt_primary = [
            c for c in cells
            if c["policy_kind"] == "apt_serve" and c["transition_cost"] == PRIMARY_TRANSITION_COST
        ]
        if not baselines or len(apt_primary) != 1:
            continue
        best = max(baselines, key=lambda c: c["arrival_normalized_weighted_goodput"])
        apt = apt_primary[0]
        best_val = float(best["arrival_normalized_weighted_goodput"])
        apt_val = float(apt["arrival_normalized_weighted_goodput"])
        stats = apt.get("apt_stats") or {}
        total_transitions = float(stats.get("kv_to_hidden_transitions", 0) + stats.get("hidden_to_kv_transitions", 0))
        completed = float(apt["num_completed"])
        regime = rec["regime"]
        summaries.append(UnitSummary(
            stage=rec["stage"],
            regime_id=rec["regime_id"],
            seed=int(rec["seed"]),
            kv_pressure=regime["kv_pressure"],
            slo_pattern=regime["slo_pattern"],
            length_pattern=regime["length_pattern"],
            arrival_pattern=regime["arrival_pattern"],
            cache_use_structure=regime["cache_use_structure"],
            n_requests=int(rec["n_requests"]),
            best_baseline_policy=best["policy_label"],
            best_baseline_anwg=best_val,
            apt_primary_anwg=apt_val,
            gap_vs_best_baseline=apt_val - best_val,
            marginal_contribution=max(0.0, apt_val - best_val),
            apt_unique_winner=apt_val > best_val,
            apt_completion_fraction=float(apt["completion_fraction"]),
            apt_slo_violation_rate=float(apt["slo_violation_rate"]),
            total_transitions=total_transitions,
            transitions_per_completed_request=total_transitions / completed if completed else float("nan"),
            evictions=float(stats.get("evictions", 0)),
            recomputations=float(stats.get("recomputations", 0)),
            switch_latency_paid=float(stats.get("switch_latency_paid", 0.0)),
            restore_latency_paid=float(stats.get("restore_latency_paid", 0.0)),
            wall_time_sec=float(rec.get("wall_time_sec", 0.0)),
        ))
    return summaries


def flatten_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        regime = rec["regime"]
        for cell in rec["cells"]:
            rows.append({
                "stage": rec["stage"],
                "regime_id": rec["regime_id"],
                "seed": rec["seed"],
                "policy_label": cell["policy_label"],
                "policy_kind": cell["policy_kind"],
                "transition_cost": cell["transition_cost"],
                "kv_pressure": regime["kv_pressure"],
                "slo_pattern": regime["slo_pattern"],
                "length_pattern": regime["length_pattern"],
                "arrival_pattern": regime["arrival_pattern"],
                "cache_use_structure": regime["cache_use_structure"],
                "num_completed": cell["num_completed"],
                "num_dropped": cell["num_dropped"],
                "num_total": cell["num_total"],
                "completion_fraction": float(cell["completion_fraction"]),
                "arrival_normalized_weighted_goodput": float(cell["arrival_normalized_weighted_goodput"]),
                "weighted_goodput_completed_only": float(cell["weighted_goodput_completed_only"]),
                "slo_violation_rate": float(cell["slo_violation_rate"]),
                "mean_latency": float(cell["mean_latency"]),
                "p95_latency": float(cell["p95_latency"]),
                "mean_ttft": float(cell["mean_ttft"]),
                "p95_ttft": float(cell["p95_ttft"]),
                "request_throughput": float(cell["request_throughput"]),
                "token_throughput": float(cell["token_throughput"]),
            })
    return rows


def summarize_values(prefix: str, values: list[float]) -> dict[str, Any]:
    return {
        f"{prefix}_n": len(values),
        f"{prefix}_mean": mean(values),
        f"{prefix}_median": median(values),
        f"{prefix}_std": stdev(values),
        f"{prefix}_p05": percentile(values, 5),
        f"{prefix}_p95": percentile(values, 95),
    }


def global_policy_summary(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        stages = [row["stage"], "all"]
        for stage in stages:
            grouped[(stage, row["policy_label"], row["transition_cost"])].append(row)

    out: list[dict[str, Any]] = []
    for (stage, policy, tc), rows in sorted(grouped.items()):
        anwg = [r["arrival_normalized_weighted_goodput"] for r in rows]
        completion = [r["completion_fraction"] for r in rows]
        slo = [r["slo_violation_rate"] for r in rows]
        out.append({
            "stage": stage,
            "policy_label": policy,
            "transition_cost": tc,
            "n": len(rows),
            "mean_anwg": mean(anwg),
            "median_anwg": median(anwg),
            "std_anwg": stdev(anwg),
            "mean_completion_fraction": mean(completion),
            "mean_slo_violation_rate": mean(slo),
            "mean_num_completed": mean([float(r["num_completed"]) for r in rows]),
            "mean_num_dropped": mean([float(r["num_dropped"]) for r in rows]),
        })
    return out


def regime_summary(units: list[UnitSummary]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[UnitSummary]] = defaultdict(list)
    for unit in units:
        grouped[(unit.stage, unit.regime_id)].append(unit)

    out: list[dict[str, Any]] = []
    for (stage, regime_id), rows in sorted(grouped.items()):
        gaps = [r.gap_vs_best_baseline for r in rows]
        best_policy_counts = Counter(r.best_baseline_policy for r in rows)
        row: dict[str, Any] = {
            "stage": stage,
            "regime_id": regime_id,
            "n_seeds": len(rows),
            "kv_pressure": rows[0].kv_pressure,
            "slo_pattern": rows[0].slo_pattern,
            "length_pattern": rows[0].length_pattern,
            "arrival_pattern": rows[0].arrival_pattern,
            "cache_use_structure": rows[0].cache_use_structure,
            "mean_apt_gap_vs_best_baseline": mean(gaps),
            "median_apt_gap_vs_best_baseline": median(gaps),
            "mean_apt_anwg": mean([r.apt_primary_anwg for r in rows]),
            "mean_best_baseline_anwg": mean([r.best_baseline_anwg for r in rows]),
            "best_baseline_mode": best_policy_counts.most_common(1)[0][0],
        }
        for eps in EPSILONS:
            suffix = str(eps).replace(".", "p")
            row[f"wins_eps_{suffix}"] = sum(1 for gap in gaps if gap > eps)
            row[f"ties_eps_{suffix}"] = sum(1 for gap in gaps if abs(gap) <= eps)
            row[f"losses_eps_{suffix}"] = sum(1 for gap in gaps if gap < -eps)
        out.append(row)
    return out


def marginal_contribution_summary(units: list[UnitSummary]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[UnitSummary]] = defaultdict(list)
    for unit in units:
        groups[("all", "all")].append(unit)
        groups[("stage", unit.stage)].append(unit)
        groups[("kv_pressure", unit.kv_pressure)].append(unit)
        groups[("cache_use_structure", unit.cache_use_structure)].append(unit)
        groups[("arrival_pattern", unit.arrival_pattern)].append(unit)
        groups[("slo_pattern", unit.slo_pattern)].append(unit)

    out: list[dict[str, Any]] = []
    for (group_type, group_value), rows in sorted(groups.items()):
        mc = [r.marginal_contribution for r in rows]
        row = {
            "group_type": group_type,
            "group_value": group_value,
            "n": len(rows),
            "mean_marginal_contribution": mean(mc),
            "median_marginal_contribution": median(mc),
            "max_marginal_contribution": max(mc) if mc else float("nan"),
            "fraction_gt_0": mean([1.0 if v > 0.0 else 0.0 for v in mc]),
            "fraction_gt_0p005": mean([1.0 if v > 0.005 else 0.0 for v in mc]),
            "fraction_gt_0p01": mean([1.0 if v > 0.01 else 0.0 for v in mc]),
            "fraction_gt_0p05": mean([1.0 if v > 0.05 else 0.0 for v in mc]),
            "fraction_unique_winner": mean([1.0 if r.apt_unique_winner else 0.0 for r in rows]),
        }
        out.append(row)
    return out


def stage1_stage2_replication(units: list[UnitSummary]) -> list[dict[str, Any]]:
    by_stage_regime: dict[tuple[str, str], list[UnitSummary]] = defaultdict(list)
    for unit in units:
        by_stage_regime[(unit.stage, unit.regime_id)].append(unit)
    confirmation_regimes = sorted({u.regime_id for u in units if u.stage == "confirmation"})
    out: list[dict[str, Any]] = []
    for regime in confirmation_regimes:
        s1 = by_stage_regime.get(("screening", regime), [])
        s2 = by_stage_regime.get(("confirmation", regime), [])
        s1_gaps = [u.gap_vs_best_baseline for u in s1]
        s2_gaps = [u.gap_vs_best_baseline for u in s2]
        s1_mean = mean(s1_gaps)
        s2_mean = mean(s2_gaps)
        out.append({
            "regime_id": regime,
            "screening_n": len(s1),
            "confirmation_n": len(s2),
            "screening_mean_gap": s1_mean,
            "confirmation_mean_gap": s2_mean,
            "confirmation_minus_screening": s2_mean - s1_mean,
            "same_sign": (s1_mean == 0.0 and s2_mean == 0.0) or (s1_mean > 0 and s2_mean > 0) or (s1_mean < 0 and s2_mean < 0),
            "screening_wins_eps005": sum(1 for v in s1_gaps if v > 0.005),
            "confirmation_wins_eps005": sum(1 for v in s2_gaps if v > 0.005),
            "screening_losses_eps005": sum(1 for v in s1_gaps if v < -0.005),
            "confirmation_losses_eps005": sum(1 for v in s2_gaps if v < -0.005),
        })
    return out


def transition_cost_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        baselines = [
            c["arrival_normalized_weighted_goodput"]
            for c in rec["cells"]
            if c["policy_kind"] == "baseline"
        ]
        best_baseline = max(float(v) for v in baselines)
        for c in rec["cells"]:
            if c["policy_kind"] != "apt_serve":
                continue
            stats = c.get("apt_stats") or {}
            total_transitions = float(stats.get("kv_to_hidden_transitions", 0) + stats.get("hidden_to_kv_transitions", 0))
            completed = float(c["num_completed"])
            rows.append({
                "stage": rec["stage"],
                "transition_cost": c["transition_cost"],
                "anwg": float(c["arrival_normalized_weighted_goodput"]),
                "gap_vs_best_baseline": float(c["arrival_normalized_weighted_goodput"]) - best_baseline,
                "transitions_per_completed_request": total_transitions / completed if completed else float("nan"),
                "evictions": float(stats.get("evictions", 0)),
                "recomputations": float(stats.get("recomputations", 0)),
                "switch_latency_paid": float(stats.get("switch_latency_paid", 0.0)),
                "restore_latency_paid": float(stats.get("restore_latency_paid", 0.0)),
            })

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[("all", row["transition_cost"])].append(row)
        grouped[(row["stage"], row["transition_cost"])].append(row)

    out: list[dict[str, Any]] = []
    for (stage, tc), group_rows in sorted(grouped.items()):
        out.append({
            "stage": stage,
            "transition_cost": tc,
            "n": len(group_rows),
            "mean_anwg": mean([r["anwg"] for r in group_rows]),
            "mean_gap_vs_best_baseline": mean([r["gap_vs_best_baseline"] for r in group_rows]),
            "mean_transitions_per_completed_request": mean([r["transitions_per_completed_request"] for r in group_rows]),
            "mean_evictions": mean([r["evictions"] for r in group_rows]),
            "mean_recomputations": mean([r["recomputations"] for r in group_rows]),
            "mean_switch_latency_paid": mean([r["switch_latency_paid"] for r in group_rows]),
            "mean_restore_latency_paid": mean([r["restore_latency_paid"] for r in group_rows]),
        })
    return out


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mechanism_analysis(units: list[UnitSummary]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[UnitSummary]] = defaultdict(list)
    for unit in units:
        groups[("all", "all")].append(unit)
        groups[("stage", unit.stage)].append(unit)
        groups[("kv_pressure", unit.kv_pressure)].append(unit)
        groups[("cache_use_structure", unit.cache_use_structure)].append(unit)
        groups[("slo_pattern", unit.slo_pattern)].append(unit)

    out: list[dict[str, Any]] = []
    for (group_type, group_value), rows in sorted(groups.items()):
        gaps = [r.gap_vs_best_baseline for r in rows]
        out.append({
            "group_type": group_type,
            "group_value": group_value,
            "n": len(rows),
            "mean_gap_vs_best_baseline": mean(gaps),
            "median_gap_vs_best_baseline": median(gaps),
            "mean_total_transitions": mean([r.total_transitions for r in rows]),
            "mean_transitions_per_completed_request": mean([r.transitions_per_completed_request for r in rows]),
            "mean_evictions": mean([r.evictions for r in rows]),
            "mean_recomputations": mean([r.recomputations for r in rows]),
            "mean_restore_latency_paid": mean([r.restore_latency_paid for r in rows]),
            "corr_gap_total_transitions": corr(gaps, [r.total_transitions for r in rows]),
            "corr_gap_transitions_per_completed": corr(gaps, [r.transitions_per_completed_request for r in rows]),
            "corr_gap_restore_latency": corr(gaps, [r.restore_latency_paid for r in rows]),
        })
    return out


def grouped_bootstrap_ci(
    rows: list[UnitSummary],
    value_fn: Callable[[UnitSummary], float],
    *,
    group_fn: Callable[[UnitSummary], str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "n_groups": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[group_fn(row)].append(float(value_fn(row)))
    group_values = {k: np.asarray(v, dtype=float) for k, v in grouped.items()}
    labels = np.asarray(sorted(group_values), dtype=object)
    group_means = np.asarray([group_values[str(label)].mean() for label in labels], dtype=float)
    point = mean([float(value_fn(r)) for r in rows])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled = rng.integers(0, len(labels), size=len(labels))
        boots[i] = float(group_means[sampled].mean())
    return {
        "n": len(rows),
        "n_groups": len(labels),
        "mean": point,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


def grouped_bootstrap_results(
    units: list[UnitSummary],
    *,
    n_bootstrap: int,
    seed: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    subsets: list[tuple[str, str, list[UnitSummary]]] = [("all", "all", units)]
    for stage in sorted({u.stage for u in units}):
        subsets.append(("stage", stage, [u for u in units if u.stage == stage]))
    for kv in sorted({u.kv_pressure for u in units}):
        subsets.append(("kv_pressure", kv, [u for u in units if u.kv_pressure == kv]))
    for cache in sorted({u.cache_use_structure for u in units}):
        subsets.append(("cache_use_structure", cache, [u for u in units if u.cache_use_structure == cache]))

    metrics: list[tuple[str, Callable[[UnitSummary], float]]] = [
        ("apt_gap_vs_best_baseline", lambda u: u.gap_vs_best_baseline),
        ("marginal_contribution", lambda u: u.marginal_contribution),
        ("apt_primary_anwg", lambda u: u.apt_primary_anwg),
        ("best_baseline_anwg", lambda u: u.best_baseline_anwg),
    ]
    total = len(subsets) * len(metrics)
    out: list[dict[str, Any]] = []
    task = 0
    for subset_type, subset_value, subset_rows in subsets:
        for metric_name, value_fn in metrics:
            task += 1
            if progress_callback:
                progress_callback({
                    "bootstrap_task": task,
                    "bootstrap_total_tasks": total,
                    "bootstrap_subset_type": subset_type,
                    "bootstrap_subset_value": subset_value,
                    "bootstrap_metric": metric_name,
                })
            boot = grouped_bootstrap_ci(
                subset_rows,
                value_fn,
                group_fn=lambda u: f"{u.stage}|{u.regime_id}",
                n_bootstrap=n_bootstrap,
                seed=seed + task,
            )
            out.append({
                "subset_type": subset_type,
                "subset_value": subset_value,
                "metric": metric_name,
                "n_bootstrap": n_bootstrap,
                **boot,
                "ci_excludes_zero": (
                    bool(boot["ci_low"] > 0.0 or boot["ci_high"] < 0.0)
                    if math.isfinite(float(boot["ci_low"])) and math.isfinite(float(boot["ci_high"]))
                    else False
                ),
            })
    return out


def write_progress(out_dir: Path, *, current_stage: str, completed_stages: list[str], extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "updated_at": time.time(),
        "current_stage": current_stage,
        "total_stages": len(STAGES),
        "completed_stages": completed_stages,
        "completed_stage_count": len(completed_stages),
        "pending_stages": [s for s in STAGES if s not in completed_stages],
    }
    if extra:
        payload.update(extra)
    atomic_write_json(out_dir / "progress.json", payload)


def should_skip(out_dir: Path, stage: str, output_name: str, resume: bool) -> bool:
    if not resume:
        return False
    if stage == "final_summary":
        return False
    return (out_dir / output_name).exists()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_analysis(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    results_path = dataset_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"missing dataset results.jsonl: {results_path}")

    log(f"starting Apt-Serve Phase G analysis: dataset={dataset_dir} output={out_dir}")
    manifest = {
        "started_at": time.time(),
        "repository": str(REPO_ROOT),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(out_dir),
        "python": sys.executable,
        "command": " ".join([sys.executable, *sys.argv]),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "stages": STAGES,
        "resume": args.resume,
    }
    atomic_write_json(out_dir / "analysis_manifest.json", manifest)
    (out_dir / "git_state.txt").write_text(git_state_text())

    completed: list[str] = []
    records = load_jsonl(results_path)
    log(f"loaded {len(records)} experiment units")
    units = build_unit_summaries(records)
    cell_rows = flatten_cells(records)
    log(f"built {len(units)} unit summaries and {len(cell_rows)} policy-cell rows")

    write_progress(out_dir, current_stage="dataset_validation", completed_stages=completed)
    validation_path = out_dir / "dataset_validation.json"
    if not should_skip(out_dir, "dataset_validation", "dataset_validation.json", args.resume):
        log("stage dataset_validation started")
        validation = validate_dataset(records, Path(args.source_run_dir).resolve() if args.source_run_dir else None)
        atomic_write_json(validation_path, validation)
        if not validation["valid"]:
            log("stage dataset_validation failed")
            write_progress(out_dir, current_stage="dataset_validation_failed", completed_stages=completed, extra={"valid": False})
            return 2
        log("stage dataset_validation completed")
    else:
        log("stage dataset_validation skipped via --resume")
    completed.append("dataset_validation")

    stage_outputs: list[tuple[str, str, Callable[[], None]]] = [
        (
            "global_policy_summary",
            "global_policy_summary.csv",
            lambda: atomic_write_csv(
                out_dir / "global_policy_summary.csv",
                [
                    "stage", "policy_label", "transition_cost", "n", "mean_anwg", "median_anwg",
                    "std_anwg", "mean_completion_fraction", "mean_slo_violation_rate",
                    "mean_num_completed", "mean_num_dropped",
                ],
                global_policy_summary(cell_rows),
            ),
        ),
        (
            "regime_summary",
            "regime_summary.csv",
            lambda: atomic_write_csv(
                out_dir / "regime_summary.csv",
                [
                    "stage", "regime_id", "n_seeds", "kv_pressure", "slo_pattern", "length_pattern",
                    "arrival_pattern", "cache_use_structure", "mean_apt_gap_vs_best_baseline",
                    "median_apt_gap_vs_best_baseline", "mean_apt_anwg", "mean_best_baseline_anwg",
                    "best_baseline_mode", "wins_eps_0p0", "ties_eps_0p0", "losses_eps_0p0",
                    "wins_eps_0p005", "ties_eps_0p005", "losses_eps_0p005",
                    "wins_eps_0p01", "ties_eps_0p01", "losses_eps_0p01",
                    "wins_eps_0p05", "ties_eps_0p05", "losses_eps_0p05",
                ],
                regime_summary(units),
            ),
        ),
        (
            "marginal_contribution_summary",
            "marginal_contribution_summary.csv",
            lambda: atomic_write_csv(
                out_dir / "marginal_contribution_summary.csv",
                [
                    "group_type", "group_value", "n", "mean_marginal_contribution",
                    "median_marginal_contribution", "max_marginal_contribution", "fraction_gt_0",
                    "fraction_gt_0p005", "fraction_gt_0p01", "fraction_gt_0p05",
                    "fraction_unique_winner",
                ],
                marginal_contribution_summary(units),
            ),
        ),
        (
            "stage1_stage2_replication",
            "stage1_stage2_replication.csv",
            lambda: atomic_write_csv(
                out_dir / "stage1_stage2_replication.csv",
                [
                    "regime_id", "screening_n", "confirmation_n", "screening_mean_gap",
                    "confirmation_mean_gap", "confirmation_minus_screening", "same_sign",
                    "screening_wins_eps005", "confirmation_wins_eps005",
                    "screening_losses_eps005", "confirmation_losses_eps005",
                ],
                stage1_stage2_replication(units),
            ),
        ),
        (
            "transition_cost_analysis",
            "transition_cost_analysis.csv",
            lambda: atomic_write_csv(
                out_dir / "transition_cost_analysis.csv",
                [
                    "stage", "transition_cost", "n", "mean_anwg", "mean_gap_vs_best_baseline",
                    "mean_transitions_per_completed_request", "mean_evictions", "mean_recomputations",
                    "mean_switch_latency_paid", "mean_restore_latency_paid",
                ],
                transition_cost_analysis(records),
            ),
        ),
        (
            "mechanism_analysis",
            "mechanism_analysis.csv",
            lambda: atomic_write_csv(
                out_dir / "mechanism_analysis.csv",
                [
                    "group_type", "group_value", "n", "mean_gap_vs_best_baseline",
                    "median_gap_vs_best_baseline", "mean_total_transitions",
                    "mean_transitions_per_completed_request", "mean_evictions",
                    "mean_recomputations", "mean_restore_latency_paid",
                    "corr_gap_total_transitions", "corr_gap_transitions_per_completed",
                    "corr_gap_restore_latency",
                ],
                mechanism_analysis(units),
            ),
        ),
    ]

    for stage, output_name, fn in stage_outputs:
        write_progress(out_dir, current_stage=stage, completed_stages=completed)
        if not should_skip(out_dir, stage, output_name, args.resume):
            log(f"stage {stage} started")
            fn()
            log(f"stage {stage} completed")
        else:
            log(f"stage {stage} skipped via --resume")
        completed.append(stage)

    write_progress(out_dir, current_stage="grouped_bootstrap_results", completed_stages=completed)
    if not should_skip(out_dir, "grouped_bootstrap_results", "grouped_bootstrap_results.csv", args.resume):
        log(f"stage grouped_bootstrap_results started n_bootstrap={args.n_bootstrap}")
        def progress_callback(extra: dict[str, Any]) -> None:
            write_progress(out_dir, current_stage="grouped_bootstrap_results", completed_stages=completed, extra=extra)
            log(
                "bootstrap task "
                f"{extra['bootstrap_task']}/{extra['bootstrap_total_tasks']} "
                f"{extra['bootstrap_subset_type']}={extra['bootstrap_subset_value']} "
                f"metric={extra['bootstrap_metric']}"
            )

        boot_rows = grouped_bootstrap_results(
            units,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            progress_callback=progress_callback,
        )
        atomic_write_csv(
            out_dir / "grouped_bootstrap_results.csv",
            [
                "subset_type", "subset_value", "metric", "n_bootstrap", "n", "n_groups",
                "mean", "ci_low", "ci_high", "ci_excludes_zero",
            ],
            boot_rows,
        )
        log("stage grouped_bootstrap_results completed")
    else:
        log("stage grouped_bootstrap_results skipped via --resume")
    completed.append("grouped_bootstrap_results")

    write_progress(out_dir, current_stage="final_summary", completed_stages=completed)
    log("stage final_summary started")
    validation = json.loads(validation_path.read_text())
    global_rows = read_csv_rows(out_dir / "global_policy_summary.csv")
    marginal_rows = read_csv_rows(out_dir / "marginal_contribution_summary.csv")
    bootstrap_rows = read_csv_rows(out_dir / "grouped_bootstrap_results.csv")
    apt_primary = [
        r for r in global_rows
        if r["stage"] == "all" and r["policy_label"] == "apt_serve_faithful" and r["transition_cost"] == PRIMARY_TRANSITION_COST
    ][0]
    best_fixed = max(
        (r for r in global_rows if r["stage"] == "all" and r["transition_cost"] == "na"),
        key=lambda r: float(r["mean_anwg"]),
    )
    overall_mc = [r for r in marginal_rows if r["group_type"] == "all" and r["group_value"] == "all"][0]
    final = {
        "status": "COMPLETE",
        "completed_at": time.time(),
        "dataset_validation": validation["classification"],
        "total_units": validation["total_units"],
        "apt_primary_transition_cost": PRIMARY_TRANSITION_COST,
        "apt_primary_mean_anwg": float(apt_primary["mean_anwg"]),
        "best_fixed_policy": best_fixed["policy_label"],
        "best_fixed_mean_anwg": float(best_fixed["mean_anwg"]),
        "apt_minus_best_fixed_mean_anwg": float(apt_primary["mean_anwg"]) - float(best_fixed["mean_anwg"]),
        "mean_marginal_contribution": float(overall_mc["mean_marginal_contribution"]),
        "median_marginal_contribution": float(overall_mc["median_marginal_contribution"]),
        "bootstrap_rows": len(bootstrap_rows),
        "note": (
            "Generated by analyze_apt_serve_phase_g.py from completed Phase G artifacts. "
            "This file is an analysis artifact only; project status documents remain authoritative "
            "only after a separate interpretation/reconciliation task reviews these outputs."
        ),
    }
    atomic_write_json(out_dir / "final_summary.json", final)
    completed.append("final_summary")
    write_progress(out_dir, current_stage="complete", completed_stages=completed, extra={"status": "COMPLETE"})
    log("stage final_summary completed")
    log("analysis complete")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Completed Phase G run directory")
    parser.add_argument("--output-dir", required=True, help="Dedicated analysis output directory")
    parser.add_argument("--source-run-dir", default=None, help="Original pre-fix run directory for provenance cross-check")
    parser.add_argument("--n-bootstrap", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--resume", action="store_true", help="Skip completed stage outputs where possible")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
