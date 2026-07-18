"""Leakage-safe, topology-aware feature extraction for Selector Dataset v2."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ...core.types import GPUConfig, Request

FeatureValue = Optional[float]


IDENTIFIER_FIELDS = {
    "scenario_id",
    "scenario_family_id",
    "dataset_family",
    "source_trace",
    "temporal_block_id",
    "seed",
    "topology_class",
    "resource_configuration_id",
    "window_id",
    "split",
}


def selector_v2_feature_columns(rows: Sequence[Dict]) -> list[str]:
    """Return model-eligible feature columns from flat v2 rows.

    Only columns with the explicit ``feat_`` prefix are eligible. Identifier,
    provenance, split, policy name, and label/metric columns are therefore
    excluded by construction.
    """
    if not rows:
        return []
    return sorted(k for k in rows[0] if k.startswith("feat_") and k not in IDENTIFIER_FIELDS)


def extract_selector_v2_features(
    *,
    window_requests: Sequence[Request],
    window_start_time: float,
    prefix_requests: Sequence[Request] = (),
    gpu_configs: Sequence[GPUConfig] = (),
    topology_class: str = "monolithic",
    recent_violation_rate: FeatureValue = None,
    active_sequence_count: int = 0,
    aggregate_kv_utilization: FeatureValue = None,
    active_batch_size: FeatureValue = None,
    prefill_queue_length: FeatureValue = None,
    decode_queue_length: FeatureValue = None,
    bridge_queue_length: FeatureValue = None,
    prefill_side_utilization: FeatureValue = None,
    decode_side_utilization: FeatureValue = None,
    transfer_delay_s: FeatureValue = None,
    instance_count: FeatureValue = None,
    load_imbalance: FeatureValue = None,
    kv_imbalance: FeatureValue = None,
    incoming_migration_count: FeatureValue = None,
    migration_pressure: FeatureValue = None,
) -> Dict[str, FeatureValue]:
    """Extract only information observable at the selector decision time.

    The observable request set is the prefix plus any current-window request
    whose arrival time is already at or before ``window_start_time``. Later
    within-window requests are intentionally ignored.
    """
    observable = [
        r for r in list(prefix_requests) + list(window_requests)
        if r.arrival_time <= window_start_time
    ]
    recent = _recent_requests(observable, window_start_time)

    features: Dict[str, FeatureValue] = {}
    features.update(_arrival_load_features(observable, recent, window_start_time, active_sequence_count, gpu_configs))
    features.update(_token_stats("prompt", [r.prompt_tokens for r in recent]))
    features.update(_token_stats("pred_output", [r.predicted_output_tokens for r in recent]))
    features.update(_slo_features(recent, window_start_time, recent_violation_rate))
    features.update(_priority_features(recent))
    features.update(_resource_features(gpu_configs))

    features.update({
        "monolithic_aggregate_kv_utilization": aggregate_kv_utilization if topology_class == "monolithic" else None,
        "monolithic_active_batch_size": active_batch_size if topology_class == "monolithic" else None,
        "disagg_prefill_gpu_count": _role_count(gpu_configs, "prefill") if topology_class == "disaggregated_prefill_decode" else None,
        "disagg_decode_gpu_count": _role_count(gpu_configs, "decode") if topology_class == "disaggregated_prefill_decode" else None,
        "disagg_prefill_queue_length": prefill_queue_length if topology_class == "disaggregated_prefill_decode" else None,
        "disagg_decode_queue_length": decode_queue_length if topology_class == "disaggregated_prefill_decode" else None,
        "disagg_bridge_queue_length": bridge_queue_length if topology_class == "disaggregated_prefill_decode" else None,
        "disagg_prefill_side_utilization": prefill_side_utilization if topology_class == "disaggregated_prefill_decode" else None,
        "disagg_decode_side_utilization": decode_side_utilization if topology_class == "disaggregated_prefill_decode" else None,
        "disagg_transfer_delay_s": transfer_delay_s if topology_class == "disaggregated_prefill_decode" else None,
        "multi_instance_count": instance_count if topology_class == "multi_instance_migratory" else None,
        "multi_instance_load_imbalance": load_imbalance if topology_class == "multi_instance_migratory" else None,
        "multi_instance_kv_imbalance": kv_imbalance if topology_class == "multi_instance_migratory" else None,
        "multi_instance_incoming_migration_count": incoming_migration_count if topology_class == "multi_instance_migratory" else None,
        "multi_instance_migration_pressure": migration_pressure if topology_class == "multi_instance_migratory" else None,
    })
    return features


def _recent_requests(requests: Sequence[Request], now: float, max_count: int = 200) -> list[Request]:
    reqs = sorted((r for r in requests if r.arrival_time <= now), key=lambda r: r.arrival_time)
    return reqs[-max_count:]


def _arrival_load_features(
    observable: Sequence[Request],
    recent: Sequence[Request],
    now: float,
    active_sequence_count: int,
    gpu_configs: Sequence[GPUConfig],
) -> Dict[str, FeatureValue]:
    arrivals = np.array([r.arrival_time for r in recent if r.arrival_time <= now], dtype=float)
    prefix_arrivals = np.array([r.arrival_time for r in observable if r.arrival_time <= now], dtype=float)

    recent_rate = _rate(arrivals)
    prefix_rate = _rate(prefix_arrivals)
    inter_arrival_cv = _interarrival_cv(arrivals)
    queue_length = float(sum(1 for r in recent if r.arrival_time <= now))
    if len(arrivals) >= 4:
        midpoint = len(arrivals) // 2
        older = _rate(arrivals[:midpoint])
        newer = _rate(arrivals[midpoint:])
        queue_growth = newer - older
    else:
        queue_growth = 0.0

    total_seq_capacity = sum(g.max_active_sequences for g in gpu_configs)
    saturation = None
    if total_seq_capacity > 0 and recent_rate is not None:
        saturation = recent_rate / total_seq_capacity

    return {
        "arrival_rate_recent": recent_rate,
        "arrival_rate_prefix": prefix_rate,
        "inter_arrival_cv": inter_arrival_cv,
        "burstiness_cv": inter_arrival_cv,
        "queue_length": queue_length,
        "recent_queue_growth_rate": float(queue_growth),
        "active_sequence_count": float(active_sequence_count),
        "saturation_load_estimate": saturation,
    }


def _token_stats(prefix: str, values: Sequence[int]) -> Dict[str, FeatureValue]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
            f"{prefix}_variance": None,
            f"{prefix}_cv": None,
        }
    arr = np.array(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_variance": float(np.var(arr)),
        f"{prefix}_cv": _cv(arr),
    }


def _slo_features(
    requests: Sequence[Request],
    now: float,
    recent_violation_rate: FeatureValue,
) -> Dict[str, FeatureValue]:
    if not requests:
        return {
            "tight_slo_fraction": None,
            "mean_slack": None,
            "p10_slack": None,
            "minimum_slack": None,
            "recent_slo_violation_rate": recent_violation_rate,
        }
    slacks = np.array([r.slo_deadline - now for r in requests], dtype=float)
    tight = np.array([1.0 if r.class_id in {"tight", "interactive", "critical"} else 0.0 for r in requests])
    return {
        "tight_slo_fraction": float(np.mean(tight)),
        "mean_slack": float(np.mean(slacks)),
        "p10_slack": float(np.percentile(slacks, 10)),
        "minimum_slack": float(np.min(slacks)),
        "recent_slo_violation_rate": recent_violation_rate,
    }


def _priority_features(requests: Sequence[Request]) -> Dict[str, FeatureValue]:
    if not requests:
        return {
            "priority_mean": None,
            "priority_p90": None,
            "priority_high_fraction": None,
            "priority_class_count": None,
        }
    priorities = np.array([r.priority for r in requests], dtype=float)
    return {
        "priority_mean": float(np.mean(priorities)),
        "priority_p90": float(np.percentile(priorities, 90)),
        "priority_high_fraction": float(np.mean(priorities >= 3.0)),
        "priority_class_count": float(len({r.class_id for r in requests})),
    }


def _resource_features(gpu_configs: Sequence[GPUConfig]) -> Dict[str, FeatureValue]:
    gpu_count = len(gpu_configs)
    total_kv = sum(g.max_kv_tokens for g in gpu_configs)
    total_seq = sum(g.max_active_sequences for g in gpu_configs)
    token_budget = sum(g.max_batch_tokens for g in gpu_configs)
    block_size = 16.0
    return {
        "resource_gpu_count": float(gpu_count),
        "resource_kv_capacity": float(total_kv),
        "resource_block_size": block_size,
        "resource_sequence_capacity": float(total_seq),
        "resource_token_budget": float(token_budget),
    }


def _role_count(gpu_configs: Sequence[GPUConfig], role: str) -> float:
    return float(sum(1 for g in gpu_configs if g.role == role))


def _rate(arrivals: np.ndarray) -> FeatureValue:
    if len(arrivals) < 2:
        return None
    span = float(arrivals[-1] - arrivals[0])
    if span <= 0:
        return None
    return float((len(arrivals) - 1) / span)


def _interarrival_cv(arrivals: np.ndarray) -> FeatureValue:
    if len(arrivals) < 3:
        return None
    gaps = np.diff(arrivals)
    return _cv(gaps)


def _cv(arr: np.ndarray) -> FeatureValue:
    mean = float(np.mean(arr)) if len(arr) else 0.0
    if mean == 0.0:
        return 0.0
    return float(np.std(arr) / mean)
