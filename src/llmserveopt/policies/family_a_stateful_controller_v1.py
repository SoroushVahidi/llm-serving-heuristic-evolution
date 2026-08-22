"""Stateful Family-A ESTF/WFS regime controller.

The controller is intentionally small: a frozen observable-state scorer,
minimum dwell, hysteresis, and delegation to unchanged ESTF/WFS parents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.action import Action
from ..core.types import ObservableGPUState, ObservableRequest, ObservableState
from .base import BasePolicy
from .estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from .policy_library_v2_helpers import queue_class_counts
from .scoring import DEFAULT_ALPHA, DEFAULT_BETA, deadline_slack, predicted_service_proxy
from .weighted_fair_share import WeightedFairSharePolicy


ESTF_MODE = "ESTF_MODE"
WFS_MODE = "WFS_MODE"

STATEFUL_CONTROLLER_FEATURES: Tuple[str, ...] = (
    "step",
    "queue_length",
    "active_count",
    "completed_count",
    "n_gpus",
    "max_class_deficit_ratio",
    "longest_waiting_age",
    "n_distinct_classes_in_queue",
    "queue_age_p10",
    "queue_age_p50",
    "queue_age_p90",
    "queue_age_mean",
    "predicted_output_tokens_p10",
    "predicted_output_tokens_p50",
    "predicted_output_tokens_p90",
    "predicted_output_tokens_mean",
    "prompt_tokens_p10",
    "prompt_tokens_p50",
    "prompt_tokens_p90",
    "prompt_tokens_mean",
    "est_service_time_p10",
    "est_service_time_p50",
    "est_service_time_p90",
    "est_service_time_mean",
    "laxity_p10",
    "laxity_p50",
    "laxity_p90",
    "laxity_mean",
    "fraction_laxity_negative",
    "fraction_laxity_near_deadline",
    "mean_kv_utilization",
    "max_kv_utilization",
    "free_kv_capacity",
    "prefilling_count",
    "decoding_count",
)

FORBIDDEN_RUNTIME_INPUT_NAMES: Tuple[str, ...] = (
    "favlong",
    "favshort",
    "family_label",
    "generator",
    "seed",
    "split",
    "scenario_id",
    "future",
    "counterfactual",
    "outcome",
)


def snapshot_gpu_counters(
    state: ObservableState,
) -> Tuple[Tuple[int, Tuple[int, ...], int], ...]:
    """Capture only the observable counters parent policies mutate."""
    return tuple(
        (gpu.gpu_id, tuple(gpu.active_request_ids), int(gpu.current_kv_tokens))
        for gpu in state.gpu_states
    )


def restore_gpu_counters(
    state: ObservableState,
    snapshot: Sequence[Tuple[int, Sequence[int], int]],
) -> None:
    by_gpu_id = {gpu.gpu_id: gpu for gpu in state.gpu_states}
    for gpu_id, active_ids, current_kv_tokens in snapshot:
        gpu = by_gpu_id[gpu_id]
        gpu.active_request_ids = list(active_ids)
        gpu.current_kv_tokens = int(current_kv_tokens)


def canonical_action(action: Action) -> Tuple[Tuple[str, int, Tuple[int, ...]], ...]:
    """Canonicalize admission-style parent actions for disagreement gating."""
    rows: List[Tuple[str, int, Tuple[int, ...]]] = []
    for verb in ("admit", "preempt", "swap", "hold_decode"):
        mapping = getattr(action, verb)
        for gpu_id, req_ids in sorted(mapping.items()):
            rows.append((verb, int(gpu_id), tuple(sorted(int(rid) for rid in req_ids))))
    for gpu_id, pairs in sorted(action.migrate.items()):
        rows.append(
            (
                "migrate",
                int(gpu_id),
                tuple(sorted((int(rid), int(dest)) for rid, dest in pairs)),
            )
        )
    if action.prefill_chunk_override:
        rows.append(
            (
                "prefill_chunk_override",
                -1,
                tuple(sorted((int(g), int(v)) for g, v in action.prefill_chunk_override.items())),
            )
        )
    return tuple(rows)


def actions_disagree(left: Action, right: Action) -> bool:
    return canonical_action(left) != canonical_action(right)


def _safe_quantiles(values: Sequence[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_mean": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_mean": float(np.mean(arr)),
    }


def _gpu_kv_util(gpu: ObservableGPUState) -> float:
    return gpu.current_kv_tokens / max(1, gpu.max_kv_tokens)


def extract_family_a_stateful_features(
    state: ObservableState,
    *,
    step_size: float = 0.001,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> Dict[str, float]:
    """Extract the frozen V1 causal feature vector from observable state."""
    waiting = list(state.waiting_queue) + list(state.migrating_queue)
    gpu_states = list(state.gpu_states)

    queue_ages = [max(0.0, state.time - req.arrival_time) for req in waiting]
    predicted_outputs = [float(req.predicted_output_tokens) for req in waiting]
    prompt_tokens = [float(req.prompt_tokens) for req in waiting]
    est_services = [predicted_service_proxy(req, alpha, beta) for req in waiting]
    laxities = [
        deadline_slack(req, state.time, service_proxy=svc, alpha=alpha, beta=beta)
        for req, svc in zip(waiting, est_services)
    ]

    features: Dict[str, float] = {
        "step": float(state.step),
        "queue_length": float(len(waiting)),
        "active_count": float(sum(len(gpu.active_request_ids) for gpu in gpu_states)),
        "completed_count": float(state.completed_count),
        "n_gpus": float(len(gpu_states)),
    }
    features.update(_safe_quantiles(queue_ages, "queue_age"))
    features.update(_safe_quantiles(predicted_outputs, "predicted_output_tokens"))
    features.update(_safe_quantiles(prompt_tokens, "prompt_tokens"))
    features.update(_safe_quantiles(est_services, "est_service_time"))
    features.update(_safe_quantiles(laxities, "laxity"))

    if laxities:
        near_deadline = 10.0 * step_size
        features["fraction_laxity_negative"] = float(sum(v < 0.0 for v in laxities) / len(laxities))
        features["fraction_laxity_near_deadline"] = float(
            sum(v <= near_deadline for v in laxities) / len(laxities)
        )
    else:
        features["fraction_laxity_negative"] = 0.0
        features["fraction_laxity_near_deadline"] = 0.0

    class_counts = queue_class_counts(waiting)
    total_class_count = sum(class_counts.values())
    if total_class_count:
        ideal_share = 1.0 / max(1, len(class_counts))
        observed_shares = [count / total_class_count for count in class_counts.values()]
        features["max_class_deficit_ratio"] = float(
            max(max(0.0, ideal_share - share) / ideal_share for share in observed_shares)
        )
    else:
        features["max_class_deficit_ratio"] = 0.0
    features["longest_waiting_age"] = float(max(queue_ages, default=0.0))
    features["n_distinct_classes_in_queue"] = float(len(class_counts))

    kv_utils = [_gpu_kv_util(gpu) for gpu in gpu_states]
    features["mean_kv_utilization"] = float(np.mean(kv_utils)) if kv_utils else 0.0
    features["max_kv_utilization"] = float(max(kv_utils, default=0.0))
    features["free_kv_capacity"] = float(sum(gpu.max_kv_tokens - gpu.current_kv_tokens for gpu in gpu_states))
    features["prefilling_count"] = float(sum(gpu.prefilling_count for gpu in gpu_states))
    features["decoding_count"] = float(sum(gpu.decoding_count for gpu in gpu_states))

    return {name: float(features.get(name, 0.0)) for name in STATEFUL_CONTROLLER_FEATURES}


@dataclass(frozen=True)
class FrozenTreeModeModel:
    """Pure-Python snapshot of a fitted sklearn binary decision tree."""

    feature_names: Tuple[str, ...]
    children_left: Tuple[int, ...]
    children_right: Tuple[int, ...]
    feature: Tuple[int, ...]
    threshold: Tuple[float, ...]
    value: Tuple[Tuple[float, float], ...]

    def predict_estf_probability(self, features: Mapping[str, float]) -> float:
        node = 0
        while self.children_left[node] != self.children_right[node]:
            feature_idx = self.feature[node]
            name = self.feature_names[feature_idx]
            raw_value = float(features.get(name, 0.0))
            if not math.isfinite(raw_value):
                raw_value = 0.0
            if raw_value <= self.threshold[node]:
                node = self.children_left[node]
            else:
                node = self.children_right[node]
        counts = self.value[node]
        total = float(counts[0] + counts[1])
        return float(counts[1] / total) if total > 0.0 else 0.5

    @classmethod
    def from_sklearn(cls, tree_model: object, feature_names: Sequence[str]) -> "FrozenTreeModeModel":
        tree = tree_model.tree_
        values = []
        classes = list(getattr(tree_model, "classes_", [0, 1]))
        for raw in tree.value:
            counts_by_class = raw[0]
            count0 = float(counts_by_class[classes.index(0)]) if 0 in classes else 0.0
            count1 = float(counts_by_class[classes.index(1)]) if 1 in classes else 0.0
            values.append((count0, count1))
        return cls(
            feature_names=tuple(feature_names),
            children_left=tuple(int(v) for v in tree.children_left),
            children_right=tuple(int(v) for v in tree.children_right),
            feature=tuple(int(v) for v in tree.feature),
            threshold=tuple(float(v) for v in tree.threshold),
            value=tuple(values),
        )

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "children_left": list(self.children_left),
            "children_right": list(self.children_right),
            "feature": list(self.feature),
            "threshold": list(self.threshold),
            "value": [list(v) for v in self.value],
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "FrozenTreeModeModel":
        return cls(
            feature_names=tuple(str(v) for v in payload["feature_names"]),
            children_left=tuple(int(v) for v in payload["children_left"]),
            children_right=tuple(int(v) for v in payload["children_right"]),
            feature=tuple(int(v) for v in payload["feature"]),
            threshold=tuple(float(v) for v in payload["threshold"]),
            value=tuple((float(v[0]), float(v[1])) for v in payload["value"]),
        )


@dataclass
class ControllerDecision:
    step: int
    mode_before: str
    mode_after: str
    candidate_region: bool
    estf_probability: Optional[float]
    switched: bool
    reason: str


@dataclass
class FamilyAStatefulControllerV1(BasePolicy):
    """Persistent ESTF/WFS regime controller for Family-A TRAIN/VAL only."""

    mode_model: object
    step_size: float = 0.001
    min_dwell_steps: int = 20
    estf_enter_threshold: float = 0.65
    wfs_enter_threshold: float = 0.35
    initial_mode: str = WFS_MODE
    require_parent_disagreement: bool = True
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    estf_policy: EstimatedServiceTimeFirstPolicy = field(default_factory=EstimatedServiceTimeFirstPolicy)
    wfs_policy: WeightedFairSharePolicy = field(default_factory=WeightedFairSharePolicy)

    name: str = "family_a_stateful_controller_v1"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mode = self.initial_mode
        self._steps_in_mode = 0
        self._decisions: List[ControllerDecision] = []
        self._switch_steps: List[int] = []
        self._mode_counts = {ESTF_MODE: 0, WFS_MODE: 0}
        self._candidate_count = 0
        self._abstention_count = 0
        self.estf_policy.reset()
        self.wfs_policy.reset()

    @property
    def mode(self) -> str:
        return self._mode

    def select_action(self, state: ObservableState) -> Action:
        mode_before = self._mode
        candidate = False
        estf_probability: Optional[float] = None
        switched = False
        reason = "outside_candidate_region"

        if state.waiting_queue or state.migrating_queue:
            candidate = self._candidate_region(state)
            if candidate:
                self._candidate_count += 1
                features = extract_family_a_stateful_features(
                    state,
                    step_size=self.step_size,
                    alpha=self.alpha,
                    beta=self.beta,
                )
                estf_probability = self._predict_estf_probability(features)
                if self._steps_in_mode >= self.min_dwell_steps:
                    if self._mode == WFS_MODE and estf_probability >= self.estf_enter_threshold:
                        self._set_mode(ESTF_MODE, state.step)
                        switched = True
                        reason = "switch_wfs_to_estf"
                    elif self._mode == ESTF_MODE and estf_probability <= self.wfs_enter_threshold:
                        self._set_mode(WFS_MODE, state.step)
                        switched = True
                        reason = "switch_estf_to_wfs"
                    else:
                        self._abstention_count += 1
                        reason = "hysteresis_remain"
                else:
                    self._abstention_count += 1
                    reason = "dwell_remain"
            else:
                self._abstention_count += 1
        else:
            self._abstention_count += 1
            reason = "empty_queue"

        self._mode_counts[self._mode] += 1
        self._decisions.append(
            ControllerDecision(
                step=int(state.step),
                mode_before=mode_before,
                mode_after=self._mode,
                candidate_region=bool(candidate),
                estf_probability=estf_probability,
                switched=switched,
                reason=reason,
            )
        )

        pre_selected = snapshot_gpu_counters(state)
        selected_policy = self.estf_policy if self._mode == ESTF_MODE else self.wfs_policy
        action = selected_policy.select_action(state)
        self._steps_in_mode += 1
        # Defensive: selected parent action is the only mutation that should remain.
        if not isinstance(action, Action):
            restore_gpu_counters(state, pre_selected)
            return Action()
        return action

    def diagnostics(self) -> Dict[str, object]:
        total_decisions = len(self._decisions)
        switch_directions: Dict[str, int] = {}
        for decision in self._decisions:
            if decision.switched:
                key = f"{decision.mode_before}->{decision.mode_after}"
                switch_directions[key] = switch_directions.get(key, 0) + 1
        dwell_segments = self._dwell_segments()
        return {
            "mode": self._mode,
            "total_decisions": total_decisions,
            "candidate_count": self._candidate_count,
            "abstention_count": self._abstention_count,
            "switch_count": len(self._switch_steps),
            "switch_directions": switch_directions,
            "estf_occupancy_fraction": (
                self._mode_counts[ESTF_MODE] / total_decisions if total_decisions else 0.0
            ),
            "wfs_occupancy_fraction": (
                self._mode_counts[WFS_MODE] / total_decisions if total_decisions else 0.0
            ),
            "mode_counts": dict(self._mode_counts),
            "dwell_segments": dwell_segments,
            "short_dwell_segment_fraction": (
                sum(seg < self.min_dwell_steps for seg in dwell_segments) / len(dwell_segments)
                if dwell_segments else 0.0
            ),
        }

    def decision_log(self) -> List[Dict[str, object]]:
        return [decision.__dict__.copy() for decision in self._decisions]

    def _candidate_region(self, state: ObservableState) -> bool:
        if not self.require_parent_disagreement:
            return True
        snapshot = snapshot_gpu_counters(state)
        estf_action = self.estf_policy.select_action(state)
        restore_gpu_counters(state, snapshot)
        wfs_action = self.wfs_policy.select_action(state)
        restore_gpu_counters(state, snapshot)
        return actions_disagree(estf_action, wfs_action)

    def _predict_estf_probability(self, features: Mapping[str, float]) -> float:
        predictor = getattr(self.mode_model, "predict_estf_probability", None)
        if predictor is None:
            return 0.5
        try:
            probability = float(predictor(features))
        except Exception:
            return 0.5
        if not math.isfinite(probability):
            return 0.5
        return min(1.0, max(0.0, probability))

    def _set_mode(self, mode: str, step: int) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._steps_in_mode = 0
        self._switch_steps.append(int(step))

    def _dwell_segments(self) -> List[int]:
        if not self._decisions:
            return []
        segments: List[int] = []
        current_mode = self._decisions[0].mode_after
        current_len = 0
        for decision in self._decisions:
            if decision.mode_after != current_mode and current_len:
                segments.append(current_len)
                current_mode = decision.mode_after
                current_len = 1
            else:
                current_len += 1
        if current_len:
            segments.append(current_len)
        return segments


@dataclass
class FamilyAStatelessTreeControllerV1(FamilyAStatefulControllerV1):
    """Diagnostic stateless comparator using the same candidate gate and scorer."""

    name: str = "family_a_stateless_tree_controller_v1"

    def select_action(self, state: ObservableState) -> Action:
        mode_before = self._mode
        candidate = False
        estf_probability: Optional[float] = None
        reason = "outside_candidate_region"

        if state.waiting_queue or state.migrating_queue:
            candidate = self._candidate_region(state)
            if candidate:
                self._candidate_count += 1
                features = extract_family_a_stateful_features(
                    state,
                    step_size=self.step_size,
                    alpha=self.alpha,
                    beta=self.beta,
                )
                estf_probability = self._predict_estf_probability(features)
                self._mode = ESTF_MODE if estf_probability >= 0.5 else WFS_MODE
                reason = "stateless_score"
            else:
                self._abstention_count += 1
        else:
            self._abstention_count += 1
            reason = "empty_queue"

        self._mode_counts[self._mode] += 1
        self._decisions.append(
            ControllerDecision(
                step=int(state.step),
                mode_before=mode_before,
                mode_after=self._mode,
                candidate_region=bool(candidate),
                estf_probability=estf_probability,
                switched=(mode_before != self._mode),
                reason=reason,
            )
        )
        selected_policy = self.estf_policy if self._mode == ESTF_MODE else self.wfs_policy
        return selected_policy.select_action(state)


def validate_feature_names(feature_names: Iterable[str]) -> None:
    names = tuple(feature_names)
    missing = [name for name in STATEFUL_CONTROLLER_FEATURES if name not in names]
    if missing:
        raise ValueError(f"missing frozen Family-A controller features: {missing}")
    lower_names = [name.lower() for name in names]
    forbidden_hits = [
        forbidden
        for forbidden in FORBIDDEN_RUNTIME_INPUT_NAMES
        if any(forbidden in name for name in lower_names)
    ]
    if forbidden_hits:
        raise ValueError(f"forbidden runtime input names present: {forbidden_hits}")
