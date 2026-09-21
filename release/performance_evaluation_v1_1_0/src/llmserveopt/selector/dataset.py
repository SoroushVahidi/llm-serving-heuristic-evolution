"""
Selector dataset builder.

For each window in a trace:
1. Extract features (causal mode by default).
2. Run each deployable candidate policy on the window (isolated simulation).
3. Assign label = best policy by weighted_goodput.
4. Write CSV and metadata.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


from ..core.metrics import RunMetrics
from ..core.types import GPUConfig, Request
from ..evaluation.run_policy import run_policy
from ..policies.registry import make_policy
from ..simulator.service_model import ServiceModel
from .candidates import SELECTOR_CANDIDATES
from .features import FeatureMode, FEATURE_NAMES, extract_features
from .labels import WindowLabel, label_windows
from .windows import RequestWindow, make_windows, DEFAULT_WINDOW_SIZE, MIN_PARTIAL_WINDOW


@dataclass
class DatasetConfig:
    """Configuration for one selector dataset build run."""
    trace_id: str = "trace"
    window_size: int = DEFAULT_WINDOW_SIZE
    min_partial_window: int = MIN_PARTIAL_WINDOW
    feature_mode: FeatureMode = FeatureMode.CAUSAL
    gpu_configs: Optional[List[GPUConfig]] = None
    service_model: Optional[ServiceModel] = None
    drain_steps: int = 5_000
    seed: int = 0
    verbose: bool = False


def run_policy_on_window(
    policy_name: str,
    window: RequestWindow,
    gpu_configs: List[GPUConfig],
    service_model: Optional[ServiceModel] = None,
    drain_steps: int = 5_000,
    seed: int = 0,
) -> RunMetrics:
    """Run a single policy on a single window (isolated simulation)."""
    policy = make_policy(policy_name, seed=seed)
    return run_policy(
        policy=policy,
        requests=window.requests,
        gpu_configs=gpu_configs,
        service_model=service_model,
        workload_tag=f"{window.trace_id}_w{window.window_id}",
        seed=seed,
        drain_steps=drain_steps,
    )


def build_selector_dataset(
    requests: Sequence[Request],
    config: DatasetConfig,
    policy_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Build selector dataset rows for a single trace.

    Parameters
    ----------
    requests : full ordered trace.
    config : DatasetConfig.
    policy_names : subset of SELECTOR_CANDIDATES to evaluate (None = all).

    Returns
    -------
    List of row dicts, one per window, ready to write as CSV.
    """
    if policy_names is None:
        policy_names = list(SELECTOR_CANDIDATES)
    else:
        invalid = set(policy_names) - set(SELECTOR_CANDIDATES)
        if invalid:
            raise ValueError(f"Non-candidate policy names: {invalid}")

    gpu_configs = config.gpu_configs or _default_gpu_configs()
    service_model = config.service_model

    windows = make_windows(
        requests=requests,
        trace_id=config.trace_id,
        window_size=config.window_size,
        min_partial=config.min_partial_window,
    )

    if config.verbose:
        print(f"  [{config.trace_id}] {len(windows)} windows from {len(requests)} requests")

    window_metrics_list: List[Dict[str, RunMetrics]] = []
    window_feature_list: List[Dict[str, float]] = []

    reqs_list = list(requests)

    for w in windows:
        if config.verbose:
            print(f"    window {w.window_id}: [{w.start_request_index}:{w.end_request_index}] "
                  f"t=[{w.start_time:.2f},{w.end_time:.2f}] n={w.num_requests}")

        # Feature extraction
        prefix = reqs_list[:w.start_request_index]
        feats = extract_features(
            window_requests=w.requests,
            window_start_time=w.start_time,
            mode=config.feature_mode,
            prefix_requests=prefix,
        )
        window_feature_list.append(feats)

        # Per-policy simulation
        policy_metrics: Dict[str, RunMetrics] = {}
        for pname in policy_names:
            m = run_policy_on_window(
                policy_name=pname,
                window=w,
                gpu_configs=gpu_configs,
                service_model=service_model,
                drain_steps=config.drain_steps,
                seed=config.seed,
            )
            policy_metrics[pname] = m
        window_metrics_list.append(policy_metrics)

    # Assign labels
    labels: List[WindowLabel] = label_windows(window_metrics_list)

    # Assemble rows
    rows = []
    for w, feats, lbl, pmetrics in zip(windows, window_feature_list, labels, window_metrics_list):
        row: Dict = {
            "trace_id": w.trace_id,
            "window_id": w.window_id,
            "start_request_index": w.start_request_index,
            "end_request_index": w.end_request_index,
            "start_time": w.start_time,
            "end_time": w.end_time,
            "num_requests": w.num_requests,
            "feature_mode": config.feature_mode.value,
            "window_size": config.window_size,
            "candidate_policy_count": len(policy_names),
            "seed": config.seed,
        }
        # Features
        for fname in FEATURE_NAMES:
            row[f"feat_{fname}"] = feats.get(fname, float("nan"))
        # Label
        row["best_policy"] = lbl.best_policy
        row["best_weighted_goodput"] = lbl.best_weighted_goodput
        row["second_best_policy"] = lbl.second_best_policy
        row["second_best_weighted_goodput"] = lbl.second_best_weighted_goodput
        row["policy_margin"] = lbl.policy_margin
        row["regret_to_best_fixed"] = lbl.regret_to_best_fixed
        row["oracle_weighted_goodput"] = lbl.oracle_weighted_goodput
        row["slo_violation_rate_best"] = lbl.slo_violation_rate_best
        row["p95_ttft_best"] = lbl.p95_ttft_best
        row["p95_latency_best"] = lbl.p95_latency_best
        # Per-policy reward vector and aux metrics (for admission/completion audits)
        for pname in SELECTOR_CANDIDATES:
            row[f"reward_{pname}"] = lbl.reward_vector.get(pname, float("nan"))
            pm = pmetrics.get(pname)
            if pm is not None:
                row[f"completion_{pname}"] = pm.completion_fraction
                row[f"slo_violation_{pname}"] = pm.slo_violation_rate
        rows.append(row)

    return rows


def save_dataset(
    rows: List[Dict],
    output_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """Write CSV and metadata JSON to output_path (no extension) or exact path."""
    import csv
    p = Path(output_path)
    if p.suffix == ".csv":
        csv_path = p
        meta_path = p.with_suffix(".metadata.json")
        reward_path = p.with_name(p.stem + "_reward_matrix.csv")
    else:
        p.mkdir(parents=True, exist_ok=True)
        csv_path = p / "selector_dataset.csv"
        meta_path = p / "selector_dataset_metadata.json"
        reward_path = p / "policy_reward_matrix.csv"

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("WARNING: no rows to write.")
        return

    # Main CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})

    # Reward matrix (windows × policies)
    reward_cols = [k for k in rows[0].keys() if k.startswith("reward_")]
    with open(reward_path, "w", newline="") as f:
        fields = ["trace_id", "window_id", "best_policy"] + reward_cols
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row[k]) for k in fields})

    # Metadata
    meta = {
        "num_windows": len(rows),
        "feature_names": FEATURE_NAMES,
        "candidate_policies": SELECTOR_CANDIDATES,
        "feature_mode": rows[0].get("feature_mode"),
        "window_size": rows[0].get("window_size"),
        "candidate_policy_count": rows[0].get("candidate_policy_count"),
        "oracle_excluded": True,
        "actual_output_tokens_used": False,
        "csv_path": str(csv_path),
        "reward_matrix_path": str(reward_path),
    }
    if metadata:
        meta.update(metadata)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(rows)} windows → {csv_path}")
    print(f"Reward matrix  → {reward_path}")
    print(f"Metadata       → {meta_path}")


def _default_gpu_configs() -> List[GPUConfig]:
    return [
        GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=512, max_kv_tokens=4096),
        GPUConfig(gpu_id=1, max_active_sequences=8, max_batch_tokens=512, max_kv_tokens=4096),
    ]


def _fmt(v) -> object:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return v
