#!/usr/bin/env python3
"""Small local end-to-end selector prototype.

Pipeline:
raw BurstGPT CSV or canonical extended JSONL
-> canonical Request JSONL
-> chronological request windows
-> causal feature extraction
-> simulator evaluation of a small policy portfolio
-> oracle/best-fixed utility matrix
-> per-policy reward-regression selector
-> validation/test comparison against baselines

This is a smoke test for pipeline correctness, not a real-serving benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Portable default (relative to repo root). Cluster paths are NEVER hardcoded:
# set LLMSERVEOPT_BURSTGPT_CSV to an absolute CSV, or LLMSERVEOPT_DATA_ROOT to a
# shared data root containing the overnight-scale BurstGPT layout below.
DEFAULT_BURSTGPT_TRACE_REL = "data/raw/burstgpt/BurstGPT_1.csv"
ENV_BURSTGPT_CSV = "LLMSERVEOPT_BURSTGPT_CSV"
ENV_DATA_ROOT = "LLMSERVEOPT_DATA_ROOT"
CLUSTER_BURSTGPT_REL = (
    "selector_v2_overnight_20260720T235405/raw/burstgpt/BurstGPT_1.csv"
)


def _as_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def resolve_burstgpt_trace_path(explicit: str | Path | None = None) -> Path | None:
    """Resolve a BurstGPT CSV without baking in machine-specific paths.

    Precedence:
    1. ``explicit`` path when it exists (CLI ``--trace-path``).
    2. If ``explicit`` is a *non-default* path and missing → ``None``
       (do not silently substitute another file).
    3. ``LLMSERVEOPT_BURSTGPT_CSV`` when set (must exist; fail closed if missing).
    4. Portable in-repo ``data/raw/burstgpt/BurstGPT_1.csv``.
    5. ``$LLMSERVEOPT_DATA_ROOT/`` + overnight-scale relative layout (optional).

    Returns ``None`` when nothing usable is found (callers skip or raise).
    """
    default_portable = ROOT / DEFAULT_BURSTGPT_TRACE_REL

    if explicit is not None:
        path = _as_repo_path(explicit)
        if path.exists():
            return path
        explicit_key = str(Path(explicit))
        is_default = explicit_key in {
            DEFAULT_BURSTGPT_TRACE_REL,
            str(default_portable),
        }
        if not is_default:
            return None

    env_csv = os.environ.get(ENV_BURSTGPT_CSV)
    if env_csv:
        env_path = Path(env_csv).expanduser()
        return env_path if env_path.exists() else None

    if default_portable.exists():
        return default_portable

    data_root = os.environ.get(ENV_DATA_ROOT)
    if data_root:
        cluster = Path(data_root).expanduser() / CLUSTER_BURSTGPT_REL
        if cluster.exists():
            return cluster

    return None


from llmserveopt.core.types import GPUConfig  # noqa: E402
from llmserveopt.evaluation.run_policy import run_policy  # noqa: E402
from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES, make_policy_library_v2  # noqa: E402
from llmserveopt.selector.advanced import (  # noqa: E402
    PolicyRewardRegressorSelector,
    anwg_column,
    validate_feature_columns,
)
from llmserveopt.selector.dataset_v2.builder import metrics_to_outcome_vector  # noqa: E402
from llmserveopt.selector.dataset_v2.features import extract_selector_v2_features  # noqa: E402
from llmserveopt.selector.windows import make_windows  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.workloads.burstgpt import (  # noqa: E402
    BurstGPTConversionConfig,
    conversion_report_to_dict,
    convert_burstgpt_to_requests,
    load_burstgpt_raw,
)
from llmserveopt.workloads.trace_io_extended import load_extended_jsonl, save_extended_jsonl  # noqa: E402


DEFAULT_POLICIES = (
    "fifo",
    "edf",
    "scorpio_style_slo_guard",
    "weighted_shortest_processing",
)

# The full 27-policy deployable registry (20 historical + 7 Policy Library
# v2), for the integrated smoke test -- see test_local_e2e_smoke.py.
FULL_POLICY_LIBRARY_V2 = tuple(POLICY_LIBRARY_V2_NAMES)


def run_policy_library_v2_candidate_on_window(
    policy_name: str,
    requests: Sequence,
    gpu_configs: List[GPUConfig],
    service_model: ServiceModel,
    workload_tag: str,
    seed: int,
    drain_steps: int,
):
    """Same shape as builder.run_candidate_policy_on_window, but resolves
    across the full 27-policy registry (historical + Policy Library v2) via
    registry.make_policy_library_v2, instead of builder's
    candidates.make_candidate_policy (historical + external baselines only,
    no Policy Library v2 support). Kept local to this smoke script rather
    than changing the shared Selector Dataset v2 builder/candidates path,
    which active Wulver pilots depend on."""
    policy = make_policy_library_v2(policy_name, seed=seed)
    counts = {"admit": 0, "preempt": 0, "swap": 0, "migrate": 0}
    orig_select_action = policy.select_action

    def counting_select_action(state):
        action = orig_select_action(state)
        counts["admit"] += sum(len(v) for v in action.admit.values())
        counts["preempt"] += sum(len(v) for v in action.preempt.values())
        counts["swap"] += sum(len(v) for v in action.swap.values())
        counts["migrate"] += sum(len(v) for v in action.migrate.values())
        return action

    policy.select_action = counting_select_action
    metrics = run_policy(
        policy=policy, requests=requests, gpu_configs=gpu_configs, service_model=service_model,
        workload_tag=workload_tag, seed=seed, drain_steps=drain_steps,
    )
    return metrics_to_outcome_vector(policy_name, metrics, counts, gpu_count=len(gpu_configs))


def chronological_split_labels(n_windows: int, train_frac: float, val_frac: float) -> List[str]:
    if n_windows < 3:
        raise ValueError("At least 3 windows are required for TRAIN/VALIDATION/TEST")
    train_end = max(1, int(round(n_windows * train_frac)))
    val_end = max(train_end + 1, int(round(n_windows * (train_frac + val_frac))))
    if val_end >= n_windows:
        val_end = n_windows - 1
    labels = []
    for idx in range(n_windows):
        if idx < train_end:
            labels.append("TRAIN")
        elif idx < val_end:
            labels.append("VALIDATION")
        else:
            labels.append("TEST")
    return labels


def load_requests(args: argparse.Namespace):
    if args.input_format == "burstgpt_csv":
        path = resolve_burstgpt_trace_path(args.trace_path)
        if path is None:
            raise FileNotFoundError(
                "BurstGPT CSV not found. Provide an existing --trace-path, "
                f"stage {DEFAULT_BURSTGPT_TRACE_REL} under the repo, or set "
                f"{ENV_BURSTGPT_CSV} / {ENV_DATA_ROOT} (see resolve_burstgpt_trace_path)."
            )
        # Record the resolved absolute/relative path for provenance reports.
        args.trace_path = str(path)
        df = load_burstgpt_raw(path)
        cfg = BurstGPTConversionConfig(max_requests=args.max_requests, time_scale=args.time_scale)
        requests, report = convert_burstgpt_to_requests(df, cfg, seed=args.seed)
        return requests, {
            "input_format": args.input_format,
            "resolved_trace_path": str(path),
            "conversion_report": conversion_report_to_dict(report),
        }
    path = ROOT / args.trace_path
    if args.input_format == "extended_jsonl":
        requests, metadata = load_extended_jsonl(path)
        if args.max_requests is not None:
            requests = requests[: args.max_requests]
        return requests, {
            "input_format": args.input_format,
            "metadata_rows_read": len(metadata),
        }
    raise ValueError(f"Unsupported input format: {args.input_format}")


def select_rows_for_policies(
    windows_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    selections: Sequence[str],
) -> pd.DataFrame:
    selected = []
    for window_idx, policy in zip(windows_df["window_idx"].tolist(), selections):
        match = matrix_df[(matrix_df["window_idx"] == window_idx) & (matrix_df["policy_name"] == policy)]
        if match.empty:
            raise KeyError(f"No policy-matrix row for window={window_idx}, policy={policy}")
        selected.append(match.iloc[0])
    return pd.DataFrame(selected)


def summarize_selection(policy_rows: pd.DataFrame, label: str) -> Dict[str, float | int | str]:
    return {
        "label": label,
        "n_windows": int(len(policy_rows)),
        "mean_anwg": _mean(policy_rows["metric_arrival_normalized_weighted_goodput"]),
        "mean_completion_fraction": _mean(policy_rows["metric_completion_fraction"]),
        "mean_completed_request_quality": _mean(policy_rows["metric_weighted_goodput"]),
        "mean_ttft": _mean(policy_rows["metric_mean_ttft"]),
        "p95_latency_mean": _mean(policy_rows["metric_p95_latency"]),
        "mean_tpot": _mean(policy_rows["metric_mean_tpot"]),
        "mean_request_throughput": _mean(policy_rows["metric_request_throughput"]),
    }


def _mean(values) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None
    return round(float(vals.mean()), 6)


def evaluate_split(
    split_name: str,
    windows_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    anwg_wide: pd.DataFrame,
    selector,
    feature_cols: List[str],
    best_fixed_policy: str,
    policies: Sequence[str],
) -> Dict:
    split_windows = windows_df[windows_df["split"] == split_name].copy()
    if split_windows.empty:
        return {"split": split_name, "n_windows": 0}

    x_split = split_windows[["window_idx", *feature_cols]].copy()
    selector_selection = selector.predict(x_split)
    oracle_selection = [anwg_wide.loc[idx].idxmax() for idx in split_windows["window_idx"]]

    entries = {
        "selector_reward_regression": summarize_selection(
            select_rows_for_policies(split_windows, matrix_df, selector_selection), "selector_reward_regression"
        ),
        f"best_fixed__{best_fixed_policy}": summarize_selection(
            select_rows_for_policies(split_windows, matrix_df, [best_fixed_policy] * len(split_windows)),
            f"best_fixed__{best_fixed_policy}",
        ),
        "oracle_per_window": summarize_selection(
            select_rows_for_policies(split_windows, matrix_df, oracle_selection), "oracle_per_window"
        ),
    }
    for policy in policies:
        entries[f"fixed__{policy}"] = summarize_selection(
            select_rows_for_policies(split_windows, matrix_df, [policy] * len(split_windows)), f"fixed__{policy}"
        )

    selector_mean = entries["selector_reward_regression"]["mean_anwg"]
    fixed_mean = entries[f"best_fixed__{best_fixed_policy}"]["mean_anwg"]
    oracle_mean = entries["oracle_per_window"]["mean_anwg"]
    return {
        "split": split_name,
        "n_windows": int(len(split_windows)),
        "entries": entries,
        "selector_minus_best_fixed_anwg": (
            round(selector_mean - fixed_mean, 6)
            if selector_mean is not None and fixed_mean is not None else None
        ),
        "selector_regret_to_oracle_anwg": (
            round(oracle_mean - selector_mean, 6)
            if selector_mean is not None and oracle_mean is not None else None
        ),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", default="data/raw/burstgpt/BurstGPT_1.csv")
    parser.add_argument("--input-format", choices=["burstgpt_csv", "extended_jsonl"], default="burstgpt_csv")
    parser.add_argument("--output-dir", default="results/local_e2e_smoke/latest")
    parser.add_argument("--max-requests", type=int, default=120)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--min-partial-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--drain-steps", type=int, default=5000)
    parser.add_argument("--n-estimators", type=int, default=50)
    args = parser.parse_args()

    invalid = sorted(set(args.policies) - set(POLICY_LIBRARY_V2_NAMES))
    if invalid:
        raise SystemExit(f"Unknown or non-deployable policies: {invalid}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    requests, load_report = load_requests(args)
    requests = sorted(requests, key=lambda r: r.arrival_time)
    if len(requests) < args.window_size * 3:
        raise SystemExit(
            f"Need at least {args.window_size * 3} requests for 3 chronological splits; got {len(requests)}"
        )
    save_extended_jsonl(requests, out_dir / "canonical_workload.jsonl", source=args.input_format)

    windows = make_windows(
        requests,
        trace_id=Path(args.trace_path).stem,
        window_size=args.window_size,
        min_partial=args.min_partial_window,
        keep_partial=False,
    )
    split_labels = chronological_split_labels(len(windows), args.train_frac, args.val_frac)

    gpu_configs = [GPUConfig(0, max_active_sequences=8, max_batch_tokens=512, max_kv_tokens=4096)]
    service_model = ServiceModel(
        enable_prefill_modeling=True,
        decode_first=True,
        enable_decode_prefill_contention=True,
        step_token_budget=512,
        max_prefill_chunk_tokens=512,
    )

    window_rows: List[Dict] = []
    policy_rows: List[Dict] = []
    reqs_list = list(requests)

    for window, split in zip(windows, split_labels):
        prefix = reqs_list[: window.start_request_index]
        features = extract_selector_v2_features(
            window_requests=window.requests,
            window_start_time=window.start_time,
            prefix_requests=prefix,
            gpu_configs=gpu_configs,
            topology_class="monolithic",
            step_token_budget=service_model.step_token_budget,
        )
        window_row = {
            "window_idx": window.window_id,
            "trace_id": window.trace_id,
            "split": split,
            "start_request_index": window.start_request_index,
            "end_request_index": window.end_request_index,
            "start_time": window.start_time,
            "end_time": window.end_time,
            "num_requests": window.num_requests,
        }
        window_row.update({f"feat_{name}": value for name, value in features.items()})
        window_rows.append(window_row)

        for policy in args.policies:
            outcome = run_policy_library_v2_candidate_on_window(
                policy,
                window.requests,
                gpu_configs,
                service_model,
                workload_tag=f"{window.trace_id}_w{window.window_id}",
                seed=args.seed,
                drain_steps=args.drain_steps,
            )
            row = {
                "window_idx": window.window_id,
                "trace_id": window.trace_id,
                "split": split,
                "policy_name": policy,
                "simulator_based": True,
            }
            row.update(outcome.to_row_dict(prefix="metric"))
            policy_rows.append(row)

    windows_df = pd.DataFrame(window_rows)
    matrix_df = pd.DataFrame(policy_rows)
    anwg_wide = matrix_df.pivot_table(
        index="window_idx",
        columns="policy_name",
        values="metric_arrival_normalized_weighted_goodput",
        aggfunc="first",
    ).reindex(columns=args.policies)

    label_rows = windows_df.copy()
    for policy in args.policies:
        label_rows[anwg_column(policy)] = label_rows["window_idx"].map(anwg_wide[policy])
    feature_cols = validate_feature_columns(sorted(col for col in label_rows.columns if col.startswith("feat_")))
    label_rows[feature_cols] = label_rows[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    train_rows = label_rows[label_rows["split"] == "TRAIN"]

    best_fixed_policy = anwg_wide.loc[train_rows["window_idx"]].mean(axis=0).idxmax()
    selector = PolicyRewardRegressorSelector(
        name="local_smoke_rf_reward_regression",
        allowed_policies=args.policies,
        feature_cols=feature_cols,
        estimator="random_forest",
        n_estimators=args.n_estimators,
        max_depth=4,
        random_state=args.seed,
    ).fit(train_rows)

    reports = {
        split: evaluate_split(
            split,
            windows_df,
            matrix_df,
            anwg_wide,
            selector,
            feature_cols,
            best_fixed_policy,
            args.policies,
        )
        for split in ("TRAIN", "VALIDATION", "TEST")
    }

    elapsed = time.perf_counter() - t0
    write_csv(window_rows, out_dir / "windows.csv")
    write_csv(policy_rows, out_dir / "policy_matrix.csv")

    result = {
        "pipeline": "local_e2e_smoke",
        "simulator_based": True,
        "real_serving_measurements": False,
        "git_commit": git_commit(),
        "seed": args.seed,
        "trace_path": args.trace_path,
        "load_report": load_report,
        "n_requests": len(requests),
        "n_windows": len(windows),
        "split_counts": windows_df["split"].value_counts().to_dict(),
        "policies": args.policies,
        "selector": selector.name,
        "best_fixed_policy_on_train": best_fixed_policy,
        "primary_metric": "arrival_normalized_weighted_goodput",
        "feature_columns": feature_cols,
        "reports_by_split": reports,
        "runtime_s": round(elapsed, 3),
        "outputs": {
            "canonical_workload": str(out_dir / "canonical_workload.jsonl"),
            "windows": str(out_dir / "windows.csv"),
            "policy_matrix": str(out_dir / "policy_matrix.csv"),
            "selector_eval": str(out_dir / "selector_eval.json"),
            "manifest": str(out_dir / "manifest.json"),
        },
    }
    (out_dir / "selector_eval.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "manifest.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "README.md").write_text(
        "# Local E2E Smoke Result\n\n"
        "Simulator-only pipeline smoke test. Do not interpret as real-serving validation.\n\n"
        f"- Requests: {len(requests)}\n"
        f"- Windows: {len(windows)}\n"
        f"- Policies: {', '.join(args.policies)}\n"
        f"- Runtime: {elapsed:.3f}s\n"
    )
    print(json.dumps({
        "output_dir": str(out_dir),
        "n_requests": len(requests),
        "n_windows": len(windows),
        "split_counts": result["split_counts"],
        "best_fixed_policy_on_train": best_fixed_policy,
        "runtime_s": round(elapsed, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
