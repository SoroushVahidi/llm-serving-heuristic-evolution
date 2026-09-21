"""Read-only scientific analysis helpers for Public Trace Replay v1.

The functions in this module never run replay simulation and never mutate the
canonical replay outputs. They consume the completed Layer-2/3/4 artifacts and
derive deterministic summaries for the public-trace analysis report.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from llmserveopt.policy_separation import public_trace_replay_v1 as ptr

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPLAY_DIR = REPO_ROOT / "experiments" / "public_trace_replay_v1"
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "public_trace_corpus_v1"
DEFAULT_UNIFIED_MATRIX = (
    REPO_ROOT / "experiments" / "unified_utility_matrix_v2" / "unified_utility_matrix_long_v2.csv"
)

FAITHFUL = ptr.FAITHFUL
AUGMENTED = ptr.AUGMENTED
POLICIES = tuple(ptr.CANONICAL_ANCHOR_IDS)
FAITHFUL_POLICIES = tuple(ptr.FAITHFUL_POLICIES)
PRACTICAL_EPS = 0.01  # inherited from prior composition reports; exact ties remain primary.

_SCENARIO_RE = re.compile(r"^PUBLIC_TRACE::(?P<source>.+)::w(?P<window>\d+)::(?P<view>faithful|augmented)$")


def _as_float(x: Any) -> float:
    if x is None:
        return float("nan")
    return float(x)


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.25, 0.5, 0.75, 0.9, 0.95)) -> Dict[str, float]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(arr, q)) for q in qs}


def basic_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"),
        }
    out = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    out.update(quantiles(arr, (0.25, 0.75, 0.9, 0.95, 0.99)))
    return out


def parse_public_scenario_id(scenario_id: str) -> Dict[str, Any]:
    m = _SCENARIO_RE.match(scenario_id)
    if not m:
        raise ValueError(f"unrecognized public trace scenario id: {scenario_id}")
    d = m.groupdict()
    return {
        "source_dataset": d["source"],
        "window_index": int(d["window"]),
        "view_short": d["view"],
        "base_window_id": f"{d['source']}::w{d['window']}",
    }


def load_checkpoint_df(replay_dir: Path = DEFAULT_REPLAY_DIR) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(replay_dir / "layer3_checkpoint.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    parsed = df["canonical_scenario_id"].map(parse_public_scenario_id).apply(pd.Series)
    for col in ("window_index", "view_short", "base_window_id"):
        df[col] = parsed[col]
    numeric_cols = [
        "primary_utility_anwg",
        "secondary_completion_fraction",
        "secondary_weighted_completion_fraction",
        "weighted_goodput",
        "mean_latency",
        "median_latency",
        "p95_latency",
        "p99_latency",
        "mean_queuing_delay",
        "mean_ttft",
        "mean_tpot",
        "slo_violation_rate",
        "num_total",
        "num_completed",
        "num_dropped",
        "num_slo_violated",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_manifest(replay_dir: Path = DEFAULT_REPLAY_DIR) -> Dict[str, Any]:
    with open(replay_dir / "layer2_scenario_manifest.json") as f:
        return json.load(f)


def load_integrity(replay_dir: Path = DEFAULT_REPLAY_DIR) -> Dict[str, Any]:
    with open(replay_dir / "layer3_checkpoint_integrity_report.json") as f:
        return json.load(f)


def verify_public_trace_inputs(replay_dir: Path = DEFAULT_REPLAY_DIR) -> Dict[str, Any]:
    manifest = load_manifest(replay_dir)
    integrity = load_integrity(replay_dir)
    df = load_checkpoint_df(replay_dir)
    traj_count = len(list((replay_dir / "trajectories").glob("*.parquet")))
    scenario_ids = manifest.get("scenario_ids", [])
    scenario_counts = pd.Series(scenario_ids).map(parse_public_scenario_id).apply(pd.Series)
    out = {
        "manifest_n_base_windows": int(manifest.get("n_base_windows")),
        "manifest_n_scenario_records": int(manifest.get("n_scenario_records")),
        "manifest_expected_cells": int(manifest.get("n_expected_layer3_cells")),
        "manifest_sources": sorted(manifest.get("sources", [])),
        "manifest_faithful_scenarios": int((scenario_counts["view_short"] == "faithful").sum()),
        "manifest_augmented_scenarios": int((scenario_counts["view_short"] == "augmented").sum()),
        "checkpoint_rows": int(len(df)),
        "checkpoint_success": int((df["status"] == "success").sum()),
        "checkpoint_failed": int((df["status"] == "failed").sum()),
        "checkpoint_faithful_cells": int((df["scenario_evidence_class"] == FAITHFUL).sum()),
        "checkpoint_augmented_cells": int((df["scenario_evidence_class"] == AUGMENTED).sum()),
        "trajectory_file_count": int(traj_count),
        "integrity": integrity,
        "ok": bool(
            manifest.get("n_base_windows") == 60
            and manifest.get("n_scenario_records") == 120
            and manifest.get("n_expected_layer3_cells") == 480
            and (scenario_counts["view_short"] == "faithful").sum() == 60
            and (scenario_counts["view_short"] == "augmented").sum() == 60
            and len(df) == 480
            and (df["status"] == "success").sum() == 480
            and (df["status"] == "failed").sum() == 0
            and (df["scenario_evidence_class"] == FAITHFUL).sum() == 120
            and (df["scenario_evidence_class"] == AUGMENTED).sum() == 360
            and traj_count == 480
            and integrity.get("ok") is True
        ),
    }
    return out


def load_corpus_distribution(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> Dict[str, Any]:
    with open(corpus_dir / "distribution_stats.json") as f:
        return json.load(f)


def source_characterization(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> Dict[str, Any]:
    stats = load_corpus_distribution(corpus_dir)
    records = ptr.build_all_scenarios()
    seen: set[Tuple[str, int]] = set()
    per_source: Dict[str, List[Dict[str, float]]] = {s: [] for s in ptr.SOURCES}
    for r in records:
        key = (r["source_dataset"], r["window_index"])
        if r["scenario_evidence_class"] != FAITHFUL or key in seen:
            continue
        seen.add(key)
        reqs = r["scenario"].requests
        arrivals = np.asarray([q.arrival_time for q in reqs], dtype=float)
        prompts = np.asarray([q.prompt_tokens for q in reqs], dtype=float)
        outputs = np.asarray([q.actual_output_tokens for q in reqs], dtype=float)
        inter = np.diff(arrivals)
        duration = float(arrivals[-1] - arrivals[0]) if len(arrivals) else 0.0
        per_source[r["source_dataset"]].append({
            "request_count": float(len(reqs)),
            "duration": duration,
            "prompt_mean": float(np.mean(prompts)),
            "prompt_p50": float(np.quantile(prompts, 0.5)),
            "prompt_p90": float(np.quantile(prompts, 0.9)),
            "output_mean": float(np.mean(outputs)),
            "output_p50": float(np.quantile(outputs, 0.5)),
            "output_p90": float(np.quantile(outputs, 0.9)),
            "interarrival_mean": float(np.mean(inter)) if inter.size else float("nan"),
            "interarrival_p50": float(np.quantile(inter, 0.5)) if inter.size else float("nan"),
            "interarrival_p90": float(np.quantile(inter, 0.9)) if inter.size else float("nan"),
            "interarrival_p99": float(np.quantile(inter, 0.99)) if inter.size else float("nan"),
            "zero_interarrival_fraction": float(np.mean(inter == 0.0)) if inter.size else float("nan"),
            "burstiness_p90_over_mean": (
                float(np.quantile(inter, 0.9) / np.mean(inter))
                if inter.size and np.mean(inter) > 0 else float("nan")
            ),
        })
    out: Dict[str, Any] = {}
    for source, rows in per_source.items():
        df = pd.DataFrame(rows)
        out[source] = {
            "n_base_windows": int(len(df)),
            "request_count_distribution": basic_stats(df["request_count"].tolist()),
            "window_duration_seconds": basic_stats(df["duration"].tolist()),
            "selected_window_prompt_mean": basic_stats(df["prompt_mean"].tolist()),
            "selected_window_output_mean": basic_stats(df["output_mean"].tolist()),
            "selected_window_interarrival_mean": basic_stats(df["interarrival_mean"].tolist()),
            "selected_window_zero_interarrival_fraction": basic_stats(df["zero_interarrival_fraction"].tolist()),
            "selected_window_burstiness_p90_over_mean": basic_stats(df["burstiness_p90_over_mean"].tolist()),
            "full_source_distribution": stats[source],
        }
    return out


def _metric_matrix(df: pd.DataFrame, evidence_class: str, metric: str = "primary_utility_anwg") -> pd.DataFrame:
    sub = df[df["scenario_evidence_class"] == evidence_class]
    return sub.pivot(index="canonical_scenario_id", columns="canonical_policy_id", values=metric).sort_index()


def paired_bootstrap_ci(values: Sequence[float], n_boot: int = 5000, seed: int = 20260820) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def faithful_two_policy_summary(df: pd.DataFrame) -> Dict[str, Any]:
    m = _metric_matrix(df, FAITHFUL)
    full = m["full_prefill"]
    chunked = m["chunked_prefill_small"]
    diff = chunked - full
    source_by_scenario = df[df["scenario_evidence_class"] == FAITHFUL].drop_duplicates("canonical_scenario_id").set_index("canonical_scenario_id")["source_dataset"]
    per_source: Dict[str, Any] = {}
    for source, idx in source_by_scenario.groupby(source_by_scenario).groups.items():
        d = diff.loc[list(idx)]
        per_source[source] = {
            "full_prefill_mean": float(full.loc[list(idx)].mean()),
            "chunked_prefill_small_mean": float(chunked.loc[list(idx)].mean()),
            "paired_diff_chunked_minus_full_mean": float(d.mean()),
            "paired_diff_ci95": paired_bootstrap_ci(d.tolist()),
            "wins_chunked_ties_losses": {
                "wins": int((d > 0).sum()),
                "ties": int((d == 0).sum()),
                "losses": int((d < 0).sum()),
            },
        }
    latency = {}
    for metric in ("mean_latency", "p95_latency", "mean_queuing_delay", "mean_ttft", "slo_violation_rate"):
        mat = _metric_matrix(df, FAITHFUL, metric)
        latency[f"{metric}_diff_chunked_minus_full_mean"] = float((mat["chunked_prefill_small"] - mat["full_prefill"]).mean())
    return {
        "n_windows": int(len(m)),
        "full_prefill": basic_stats(full.tolist()),
        "chunked_prefill_small": basic_stats(chunked.tolist()),
        "paired_diff_chunked_minus_full": basic_stats(diff.tolist()),
        "paired_diff_ci95": paired_bootstrap_ci(diff.tolist()),
        "wins_chunked_ties_losses": {
            "wins": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "losses": int((diff < 0).sum()),
        },
        "exact_tie_fraction": float((diff == 0).mean()),
        "near_tie_fraction_eps_0_01": float((diff.abs() <= PRACTICAL_EPS).mean()),
        "per_source": per_source,
        "completion_fraction_by_policy": {
            p: basic_stats(_metric_matrix(df, FAITHFUL, "secondary_completion_fraction")[p].tolist())
            for p in FAITHFUL_POLICIES
        },
        "latency_resource_differences": latency,
    }


def winner_summary(matrix: pd.DataFrame) -> Dict[str, Any]:
    winner_sets: List[List[str]] = []
    fractional_counts = {p: 0.0 for p in matrix.columns}
    unique_winners: List[str] = []
    tie_multiplicities: List[int] = []
    for _, row in matrix.iterrows():
        mx = row.max()
        winners = [p for p, v in row.items() if v == mx]
        winner_sets.append(winners)
        tie_multiplicities.append(len(winners))
        for p in winners:
            fractional_counts[p] += 1.0 / len(winners)
        if len(winners) == 1:
            unique_winners.append(winners[0])
    probs = np.asarray([v for v in fractional_counts.values() if v > 0], dtype=float)
    probs = probs / probs.sum() if probs.size else probs
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
    return {
        "n_windows": int(len(matrix)),
        "n_unique_winner_windows": int(sum(m == 1 for m in tie_multiplicities)),
        "unique_winner_fraction": float(np.mean([m == 1 for m in tie_multiplicities])) if tie_multiplicities else float("nan"),
        "n_tie_windows": int(sum(m > 1 for m in tie_multiplicities)),
        "tie_fraction": float(np.mean([m > 1 for m in tie_multiplicities])) if tie_multiplicities else float("nan"),
        "tie_multiplicity_distribution": {str(k): int(v) for k, v in pd.Series(tie_multiplicities).value_counts().sort_index().items()},
        "winner_fractional_counts": {k: float(v) for k, v in sorted(fractional_counts.items())},
        "unique_winner_counts": {k: int(v) for k, v in pd.Series(unique_winners).value_counts().sort_index().items()},
        "n_distinct_winning_policies_fractional": int(sum(v > 0 for v in fractional_counts.values())),
        "n_distinct_unique_winners": int(len(set(unique_winners))),
        "winner_entropy_bits_fractional": entropy,
        "winner_sets": winner_sets,
    }


def policy_performance_summary(df: pd.DataFrame, evidence_class: str) -> Dict[str, Any]:
    sub = df[df["scenario_evidence_class"] == evidence_class]
    matrix = _metric_matrix(df, evidence_class)
    by_policy: Dict[str, Any] = {}
    win = winner_summary(matrix)
    source_map = sub.drop_duplicates("canonical_scenario_id").set_index("canonical_scenario_id")["source_dataset"]
    for policy in matrix.columns:
        source_means = {
            source: float(matrix.loc[list(idx), policy].mean())
            for source, idx in source_map.groupby(source_map).groups.items()
        }
        by_policy[policy] = {
            **basic_stats(matrix[policy].tolist()),
            "source_mean": source_means,
            "unique_win_count": int(win["unique_winner_counts"].get(policy, 0)),
            "fractional_win_count": float(win["winner_fractional_counts"].get(policy, 0.0)),
            "rank_distribution": {
                str(k): int(v)
                for k, v in matrix.rank(axis=1, ascending=False, method="min")[policy]
                .astype(int).value_counts().sort_index().items()
            },
        }
    return {"by_policy": by_policy, "winner_summary": win}


def best_fixed_and_envelope(df: pd.DataFrame, evidence_class: str = AUGMENTED) -> Dict[str, Any]:
    m = _metric_matrix(df, evidence_class)
    means = m.mean(axis=0).sort_values(ascending=False)
    best_policy = str(means.index[0])
    best_series = m[best_policy]
    envelope = m.max(axis=1)
    gain = envelope - best_series
    source_map = df[df["scenario_evidence_class"] == evidence_class].drop_duplicates("canonical_scenario_id").set_index("canonical_scenario_id")["source_dataset"]
    per_source = {}
    for source, idx in source_map.groupby(source_map).groups.items():
        sub = m.loc[list(idx)]
        sub_means = sub.mean(axis=0).sort_values(ascending=False)
        sub_best = str(sub_means.index[0])
        sub_gain = sub.max(axis=1) - sub[sub_best]
        global_gain = sub.max(axis=1) - sub[best_policy]
        per_source[source] = {
            "best_fixed_policy": sub_best,
            "best_fixed_mean": float(sub_means.iloc[0]),
            "global_best_fixed_mean_on_source": float(sub[best_policy].mean()),
            "envelope_mean": float(sub.max(axis=1).mean()),
            "envelope_gain_over_source_best_mean": float(sub_gain.mean()),
            "envelope_gain_over_global_best_mean": float(global_gain.mean()),
            "positive_gain_over_source_best_fraction": float((sub_gain > 0).mean()),
        }
    return {
        "policy_means": {k: float(v) for k, v in means.items()},
        "best_fixed_policy": best_policy,
        "best_fixed_mean": float(means.iloc[0]),
        "envelope_mean": float(envelope.mean()),
        "envelope_gain_over_best_fixed": basic_stats(gain.tolist()),
        "envelope_gain_ci95": paired_bootstrap_ci(gain.tolist()),
        "positive_gain_fraction": float((gain > 0).mean()),
        "gain_gt_0_01_fraction": float((gain > PRACTICAL_EPS).mean()),
        "per_source": per_source,
    }


def pairwise_policy_separation(df: pd.DataFrame, evidence_class: str = AUGMENTED) -> Dict[str, Any]:
    m = _metric_matrix(df, evidence_class)
    out: Dict[str, Any] = {}
    for a, b in itertools.combinations(m.columns, 2):
        diff = m[a] - m[b]
        key = f"{a}__vs__{b}"
        out[key] = {
            "mean_abs_diff": float(diff.abs().mean()),
            "median_abs_diff": float(diff.abs().median()),
            "max_abs_diff": float(diff.abs().max()),
            "nonzero_fraction": float((diff != 0).mean()),
            f"{a}_wins": int((diff > 0).sum()),
            f"{b}_wins": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "mean_signed_diff_a_minus_b": float(diff.mean()),
        }
    if out:
        strongest = max(out.items(), key=lambda kv: (kv[1]["mean_abs_diff"], kv[1]["max_abs_diff"]))
        weakest = min(out.items(), key=lambda kv: (kv[1]["mean_abs_diff"], kv[1]["max_abs_diff"]))
    else:
        strongest = weakest = ("", {})
    return {
        "pairs": out,
        "strongest_pair_by_mean_abs": {"pair": strongest[0], **strongest[1]},
        "most_redundant_pair_by_mean_abs": {"pair": weakest[0], **weakest[1]},
    }


def source_specific_summary(df: pd.DataFrame) -> Dict[str, Any]:
    m = _metric_matrix(df, AUGMENTED)
    source_map = df[df["scenario_evidence_class"] == AUGMENTED].drop_duplicates("canonical_scenario_id").set_index("canonical_scenario_id")["source_dataset"]
    out: Dict[str, Any] = {}
    pairwise = pairwise_policy_separation
    for source, idx in source_map.groupby(source_map).groups.items():
        ids = list(idx)
        sub_df = df[(df["scenario_evidence_class"] == AUGMENTED) & (df["canonical_scenario_id"].isin(ids))]
        sub_m = m.loc[ids]
        env = best_fixed_and_envelope(sub_df, AUGMENTED)
        wins = winner_summary(sub_m)
        sep = pairwise(sub_df, AUGMENTED)
        out[source] = {
            "n_windows": int(len(ids)),
            "policy_means_desc": {k: float(v) for k, v in sub_m.mean(axis=0).sort_values(ascending=False).items()},
            "best_fixed_policy": env["best_fixed_policy"],
            "envelope_gain_over_best_fixed_mean": env["envelope_gain_over_best_fixed"]["mean"],
            "tie_fraction": wins["tie_fraction"],
            "unique_winner_fraction": wins["unique_winner_fraction"],
            "winner_fractional_counts": wins["winner_fractional_counts"],
            "strongest_pair": sep["strongest_pair_by_mean_abs"],
        }
    return out


def mf_psd_comparison(unified_matrix_path: Path = DEFAULT_UNIFIED_MATRIX) -> Dict[str, Any]:
    df = pd.read_csv(unified_matrix_path)
    m = df.pivot(index="canonical_scenario_id", columns="canonical_policy_id", values="primary_utility_anwg")
    win = winner_summary(m)
    means = m.mean(axis=0).sort_values(ascending=False)
    best = means.index[0]
    gain = m.max(axis=1) - m[best]
    pair = pairwise_policy_separation(
        df.assign(scenario_evidence_class=AUGMENTED),
        AUGMENTED,
    )
    return {
        "n_scenarios": int(len(m)),
        "best_fixed_policy": str(best),
        "best_fixed_mean": float(means.iloc[0]),
        "envelope_mean": float(m.max(axis=1).mean()),
        "envelope_gain_over_best_fixed": basic_stats(gain.tolist()),
        "positive_gain_fraction": float((gain > 0).mean()),
        "gain_gt_0_01_fraction": float((gain > PRACTICAL_EPS).mean()),
        "winner_summary": {k: v for k, v in win.items() if k != "winner_sets"},
        "strongest_pair_by_mean_abs": pair["strongest_pair_by_mean_abs"],
        "most_redundant_pair_by_mean_abs": pair["most_redundant_pair_by_mean_abs"],
    }


def _trajectory_paths(replay_dir: Path) -> List[Path]:
    return sorted((replay_dir / "trajectories").glob("*.parquet"))


def trajectory_contention_summary(replay_dir: Path = DEFAULT_REPLAY_DIR) -> Dict[str, Any]:
    paths = _trajectory_paths(replay_dir)
    cols = [
        "canonical_scenario_id", "source_dataset", "scenario_evidence_class", "policy_id",
        "queue_length", "active_request_count", "mean_kv_utilization", "max_kv_utilization",
        "admitted_count",
    ]
    frames = [pd.read_parquet(path, columns=cols) for path in paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    df["active_capacity_fraction"] = df["active_request_count"] / ptr.GPU_CONFIG_KWARGS["max_active_sequences"]
    df["queue_gt0"] = df["queue_length"] > 0
    df["active_at_capacity"] = df["active_request_count"] >= ptr.GPU_CONFIG_KWARGS["max_active_sequences"]
    df["active_ge_90pct_capacity"] = df["active_request_count"] >= 0.9 * ptr.GPU_CONFIG_KWARGS["max_active_sequences"]
    cell = df.groupby(["canonical_scenario_id", "policy_id"], as_index=False).agg(
        source_dataset=("source_dataset", "first"),
        scenario_evidence_class=("scenario_evidence_class", "first"),
        n_steps=("queue_length", "size"),
        frac_queue_gt0=("queue_gt0", "mean"),
        max_queue=("queue_length", "max"),
        p99_queue=("queue_length", lambda s: float(np.quantile(s, 0.99))),
        max_active=("active_request_count", "max"),
        p99_active=("active_request_count", lambda s: float(np.quantile(s, 0.99))),
        max_active_capacity_fraction=("active_capacity_fraction", "max"),
        frac_active_at_capacity=("active_at_capacity", "mean"),
        frac_active_ge_90pct_capacity=("active_ge_90pct_capacity", "mean"),
        max_kv=("max_kv_utilization", "max"),
        p99_kv=("max_kv_utilization", lambda s: float(np.quantile(s, 0.99))),
        frac_admitting=("admitted_count", lambda s: float((s > 0).mean())),
    )
    out: Dict[str, Any] = {
        "n_trajectory_files": int(len(paths)),
        "n_step_rows": int(len(df)),
        "capacity": dict(ptr.GPU_CONFIG_KWARGS),
        "overall_step_weighted": {
            "frac_queue_gt0": float(df["queue_gt0"].mean()) if len(df) else float("nan"),
            "queue": basic_stats(df["queue_length"].tolist()),
            "active_count": basic_stats(df["active_request_count"].tolist()),
            "active_capacity_fraction": basic_stats(df["active_capacity_fraction"].tolist()),
            "max_kv_utilization": basic_stats(df["max_kv_utilization"].tolist()),
            "frac_active_at_capacity": float(df["active_at_capacity"].mean()) if len(df) else float("nan"),
            "frac_active_ge_90pct_capacity": float(df["active_ge_90pct_capacity"].mean()) if len(df) else float("nan"),
        },
        "cell_level": {
            "frac_cells_ever_queue_gt0": float((cell["max_queue"] > 0).mean()) if len(cell) else float("nan"),
            "frac_cells_active_at_capacity": float((cell["max_active"] >= ptr.GPU_CONFIG_KWARGS["max_active_sequences"]).mean()) if len(cell) else float("nan"),
            "max_queue": basic_stats(cell["max_queue"].tolist()),
            "max_active": basic_stats(cell["max_active"].tolist()),
            "max_active_capacity_fraction": basic_stats(cell["max_active_capacity_fraction"].tolist()),
            "max_kv": basic_stats(cell["max_kv"].tolist()),
        },
        "by_source_view": {},
        "by_source": {},
    }
    for keys, group in df.groupby(["source_dataset", "scenario_evidence_class"]):
        source, view = keys
        gcell = cell[(cell["source_dataset"] == source) & (cell["scenario_evidence_class"] == view)]
        out["by_source_view"][f"{source}|{view}"] = {
            "n_cells": int(len(gcell)),
            "n_steps": int(len(group)),
            "frac_queue_gt0_step_weighted": float((group["queue_length"] > 0).mean()),
            "queue": basic_stats(group["queue_length"].tolist()),
            "active_count": basic_stats(group["active_request_count"].tolist()),
            "active_capacity_fraction": basic_stats((group["active_request_count"] / ptr.GPU_CONFIG_KWARGS["max_active_sequences"]).tolist()),
            "max_kv_utilization": basic_stats(group["max_kv_utilization"].tolist()),
            "frac_cells_ever_queue_gt0": float((gcell["max_queue"] > 0).mean()),
            "frac_cells_active_at_capacity": float((gcell["max_active"] >= ptr.GPU_CONFIG_KWARGS["max_active_sequences"]).mean()),
            "frac_cells_active_ge_90pct_capacity": float((gcell["max_active"] >= 0.9 * ptr.GPU_CONFIG_KWARGS["max_active_sequences"]).mean()),
        }
    for source, group in df.groupby("source_dataset"):
        gcell = cell[cell["source_dataset"] == source]
        out["by_source"][source] = {
            "n_cells": int(len(gcell)),
            "frac_queue_gt0_step_weighted": float((group["queue_length"] > 0).mean()),
            "queue": basic_stats(group["queue_length"].tolist()),
            "active_count": basic_stats(group["active_request_count"].tolist()),
            "max_kv_utilization": basic_stats(group["max_kv_utilization"].tolist()),
            "frac_cells_ever_queue_gt0": float((gcell["max_queue"] > 0).mean()),
            "max_queue_cell": basic_stats(gcell["max_queue"].tolist()),
            "max_active_cell": basic_stats(gcell["max_active"].tolist()),
            "max_kv_cell": basic_stats(gcell["max_kv"].tolist()),
        }
    return out


def _safe_trajectory_path(replay_dir: Path, scenario_id: str, policy_id: str) -> Path:
    return replay_dir / "trajectories" / f"{scenario_id.replace('::', '__')}__{policy_id}.parquet"


def action_trace_comparison(df: pd.DataFrame, replay_dir: Path = DEFAULT_REPLAY_DIR) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for evidence_class in (FAITHFUL, AUGMENTED):
        sub = df[df["scenario_evidence_class"] == evidence_class]
        matrix = _metric_matrix(df, evidence_class)
        policies = list(matrix.columns)
        pair_counts: Dict[str, Dict[str, int]] = {
            f"{a}__vs__{b}": {
                "same_action_trace": 0,
                "different_action_trace": 0,
                "same_utility_and_different_action_trace": 0,
                "compared_windows": 0,
            }
            for a, b in itertools.combinations(policies, 2)
        }
        for scenario_id in sorted(sub["canonical_scenario_id"].unique()):
            traces: Dict[str, List[Tuple[int, Tuple[Any, ...]]]] = {}
            for policy in policies:
                path = _safe_trajectory_path(replay_dir, scenario_id, policy)
                tdf = pd.read_parquet(path, columns=["step", "admitted_request_ids"])
                seq = [
                    (int(row.step), tuple(row.admitted_request_ids) if isinstance(row.admitted_request_ids, (list, tuple, np.ndarray)) else tuple())
                    for row in tdf.itertuples(index=False)
                ]
                traces[policy] = seq
            for a, b in itertools.combinations(policies, 2):
                key = f"{a}__vs__{b}"
                same = traces[a] == traces[b]
                same_utility = matrix.loc[scenario_id, a] == matrix.loc[scenario_id, b]
                pair_counts[key]["compared_windows"] += 1
                pair_counts[key]["same_action_trace" if same else "different_action_trace"] += 1
                if (not same) and same_utility:
                    pair_counts[key]["same_utility_and_different_action_trace"] += 1
        out[evidence_class] = pair_counts
    return out


def annotation_sensitivity() -> Dict[str, Any]:
    rows: List[Dict[str, float | str]] = []
    for r in ptr.build_all_scenarios():
        if r["scenario_evidence_class"] != AUGMENTED:
            continue
        actual = np.asarray([q.actual_output_tokens for q in r["scenario"].requests], dtype=float)
        pred = np.asarray([q.predicted_output_tokens for q in r["scenario"].requests], dtype=float)
        prompt = np.asarray([q.prompt_tokens for q in r["scenario"].requests], dtype=float)
        service_est = ptr.DEFAULT_ALPHA * prompt + ptr.DEFAULT_BETA * pred
        deadline_minus_arrival = np.asarray([q.slo_deadline - q.arrival_time for q in r["scenario"].requests], dtype=float)
        rows.extend(
            {
                "source_dataset": r["source_dataset"],
                "predicted_output_tokens": float(pred[i]),
                "actual_output_tokens": float(actual[i]),
                "prediction_abs_error": float(abs(pred[i] - actual[i])),
                "prediction_ratio": float(pred[i] / actual[i]) if actual[i] else float("nan"),
                "deadline_minus_arrival": float(deadline_minus_arrival[i]),
                "deadline_service_multiple": float(deadline_minus_arrival[i] / service_est[i]) if service_est[i] else float("nan"),
            }
            for i in range(len(actual))
        )
    df = pd.DataFrame(rows)
    out = {
        "priority": {"unique_values": [1.0], "interpretation": "uniform controlled annotation"},
        "class_id": {"rule": "source_dataset", "unique_classes_per_single_source_window": 1},
        "prediction_noise_sigma": ptr.PREDICTION_NOISE_SIGMA,
        "slack_multiplier": ptr.SLACK_MULTIPLIER,
        "overall": {
            "predicted_output_tokens": basic_stats(df["predicted_output_tokens"].tolist()),
            "actual_output_tokens": basic_stats(df["actual_output_tokens"].tolist()),
            "prediction_abs_error": basic_stats(df["prediction_abs_error"].tolist()),
            "prediction_ratio": basic_stats(df["prediction_ratio"].tolist()),
            "deadline_minus_arrival": basic_stats(df["deadline_minus_arrival"].tolist()),
            "deadline_service_multiple": basic_stats(df["deadline_service_multiple"].tolist()),
        },
        "by_source": {},
    }
    for source, g in df.groupby("source_dataset"):
        out["by_source"][source] = {
            "prediction_abs_error": basic_stats(g["prediction_abs_error"].tolist()),
            "prediction_ratio": basic_stats(g["prediction_ratio"].tolist()),
            "deadline_minus_arrival": basic_stats(g["deadline_minus_arrival"].tolist()),
        }
    return out


def public_trace_science_summary(
    replay_dir: Path = DEFAULT_REPLAY_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    unified_matrix_path: Path = DEFAULT_UNIFIED_MATRIX,
    include_trajectories: bool = True,
    include_action_traces: bool = True,
) -> Dict[str, Any]:
    df = load_checkpoint_df(replay_dir)
    integrity = verify_public_trace_inputs(replay_dir)
    if not integrity["ok"]:
        return {"classification": "PUBLIC_TRACE_RESULT_INTEGRITY_PROBLEM", "integrity": integrity}

    faithful = faithful_two_policy_summary(df)
    annotated = policy_performance_summary(df, AUGMENTED)
    envelope = best_fixed_and_envelope(df, AUGMENTED)
    pairwise = pairwise_policy_separation(df, AUGMENTED)
    source_summary = source_specific_summary(df)
    mf = mf_psd_comparison(unified_matrix_path)
    utility_matrix = _metric_matrix(df, AUGMENTED)
    utility_positive_window_fraction = float(((utility_matrix.max(axis=1) - utility_matrix.min(axis=1)) > 0).mean())
    summary = {
        "classification": "PUBLIC_TRACE_NEAR_DEGENERACY",
        "integrity": integrity,
        "source_characterization": source_characterization(corpus_dir),
        "faithful_view": faithful,
        "annotated_view": annotated,
        "best_fixed_and_envelope": envelope,
        "pairwise_policy_separation": pairwise,
        "source_specific": source_summary,
        "mf_psd_comparison": mf,
        "utility_positive_window_fraction": utility_positive_window_fraction,
        "annotation_sensitivity": annotation_sensitivity(),
    }
    if include_trajectories:
        summary["trajectory_contention"] = trajectory_contention_summary(replay_dir)
    if include_action_traces:
        summary["action_trace_comparison"] = action_trace_comparison(df, replay_dir)
    return summary


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
