"""KV-aware composition falsification v1 policies.

Contains:
- Top-1 contextual parent selector (sklearn LogisticRegression/DecisionTree)
- Hard conditional parent selector (symbolic if/else over observable features)
- KVAdaptiveReserveChildPolicy: the within-scenario composition target --
  delegates every select_action() call, unmodified, to one of the two frozen
  parent policy instances, chosen per-step from a single online-observable
  trigger (count of currently-waiting urgent-classified requests).

See docs/design/KV_COMPOSITION_FALSIFICATION_V1.md sections 2-4.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..core.action import Action
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..policies.kv_constrained_online import KVConstrainedOnlinePolicy
from ..policies.least_laxity_first import LeastLaxityFirstPolicy

from .kv_composition_features import (
    FEATURE_NAMES,
    assert_no_hidden_leakage,
    feature_vector,
    n_urgent_waiting,
    scenario_observable_features,
    step_features,
)

PARENT_KV = "kv_constrained_online"
PARENT_LLF = "least_laxity_first"

# Preregistered tiny candidate grid (design doc SS2/SS4) -- fit on TRAIN,
# confirmed on VAL, frozen before TEST/OOD.
TAU_URGENT_GRID = (1, 2, 3)


# ===================================================================
# Top-1 contextual parent selector
# ===================================================================

class _ConstantClassifier:
    def __init__(self, label: int) -> None:
        self.label = int(label)

    def fit(self, X, y) -> None:  # noqa: ANN001
        pass

    def predict(self, X) -> np.ndarray:  # noqa: ANN001
        return np.full(shape=(len(X),), fill_value=self.label, dtype=int)


class FittedKVSelector:
    """sklearn-wrapped selector predicting kv_constrained_online or
    least_laxity_first from scenario-level observable features."""

    def __init__(self, model_type: str, classes_: Tuple[str, str], raw: Any) -> None:
        self.model_type = model_type
        self.classes_ = classes_
        self.raw = raw

    def predict_parent(self, features: Mapping[str, float]) -> str:
        assert_no_hidden_leakage(features)
        x = feature_vector(features).reshape(1, -1)
        idx = int(self.raw.predict(x)[0])
        return self.classes_[idx]


def _parent_label(kv_score: float, llf_score: float, eps: float = 1e-12) -> int:
    """1 if kv_constrained_online wins, 0 otherwise."""
    return 1 if kv_score > llf_score + eps else 0


def fit_kv_top1_selector(
    feature_rows: Sequence[Mapping[str, float]],
    kv_scores: Sequence[float],
    llf_scores: Sequence[float],
    *,
    model_type: str = "logreg",
    seed: int = 20261201,
) -> FittedKVSelector:
    X = np.vstack([feature_vector(f) for f in feature_rows])
    y = np.array(
        [_parent_label(k, l) for k, l in zip(kv_scores, llf_scores)], dtype=int
    )
    classes = (PARENT_LLF, PARENT_KV)  # index 0 -> llf, index 1 -> kv

    if len(set(int(v) for v in y)) < 2:
        clf: Any = _ConstantClassifier(int(y[0]))
        clf.fit(X, y)
    elif model_type == "tree":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=max(2, len(y) // 10), random_state=seed
        )
        clf.fit(X, y)
    else:
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X, y)

    return FittedKVSelector(model_type=model_type, classes_=classes, raw=clf)


def select_kv_model_on_val(
    feature_train: Sequence[Mapping[str, float]],
    kv_train: Sequence[float],
    llf_train: Sequence[float],
    feature_val: Sequence[Mapping[str, float]],
    kv_val: Sequence[float],
    llf_val: Sequence[float],
) -> Tuple[FittedKVSelector, Dict[str, Any]]:
    """Fit candidate top-1 selectors on TRAIN; pick model type by VAL accuracy."""
    candidates = []
    for mt in ("logreg", "tree"):
        sel = fit_kv_top1_selector(feature_train, kv_train, llf_train, model_type=mt)
        correct = 0
        for f, k, l in zip(feature_val, kv_val, llf_val):
            pred = sel.predict_parent(f)
            truth = PARENT_KV if k > l + 1e-12 else PARENT_LLF
            correct += int(pred == truth)
        acc = correct / max(1, len(feature_val))
        candidates.append((acc, mt, sel))
    candidates.sort(key=lambda t: (-t[0], t[1]))
    best_acc, best_mt, best_sel = candidates[0]
    meta = {
        "selector_model_type": best_mt,
        "selector_val_accuracy": best_acc,
        "candidates": [{"model_type": mt, "val_accuracy": acc} for acc, mt, _ in candidates],
    }
    return best_sel, meta


# ===================================================================
# Hard conditional parent selector
# ===================================================================

def hard_conditional_rule(features: Mapping[str, float]) -> str:
    """Symbolic if/else over observable scenario features.

    Mechanism theory (design doc SS1D point 3): kv_constrained_online's
    urgent-first strict sort tier helps exactly when there is real
    urgent-vs-bulk contention -- approximated at the scenario level by
    fraction_urgent_waiting (fraction of the initially-loaded trace that is
    urgent by KVConstrainedOnlinePolicy's own laxity threshold).
    """
    assert_no_hidden_leakage(features)
    frac_urgent = float(features.get("fraction_urgent_waiting", 0.0))
    n_queued = float(features.get("n_queued_requests", 0.0))
    if frac_urgent >= 0.15 and n_queued >= 10:
        return PARENT_KV
    return PARENT_LLF


# ===================================================================
# KVAdaptiveReserveChildPolicy -- the within-scenario composition target
# ===================================================================

class KVAdaptiveReserveChildPolicy(BasePolicy):
    """Delegates every select_action() call to an unmodified instance of
    kv_constrained_online or least_laxity_first, chosen fresh at every step
    from tau_urgent-gated count of currently-waiting urgent requests.

    mode = "reserve" if n_urgent_waiting(state) >= tau_urgent else "llf"

    No new admission logic is introduced; the parents' own select_action()
    implementations are called verbatim (this is the "same fundamental
    actions/mechanisms already represented by the parents" design
    constraint). Fully instrumented for the design doc's non-degeneracy
    gate (G1): mode_log, transition_count, and per-step admitted-request-ids
    are all recorded.
    """

    name = "kv_adaptive_reserve_child"

    def __init__(self, tau_urgent: int = 2) -> None:
        self.tau_urgent = int(tau_urgent)
        self._kv = KVConstrainedOnlinePolicy()
        self._llf = LeastLaxityFirstPolicy()
        self.mode_log: List[str] = []
        self.transition_count: int = 0
        self.admitted_by_step: Dict[int, List[int]] = {}
        self.kv_util_at_transition: List[float] = []
        self._last_mode: Optional[str] = None

    def reset(self) -> None:
        self.mode_log.clear()
        self.transition_count = 0
        self.admitted_by_step.clear()
        self.kv_util_at_transition.clear()
        self._last_mode = None
        self._kv = KVConstrainedOnlinePolicy()
        self._llf = LeastLaxityFirstPolicy()

    @property
    def n_llf_steps(self) -> int:
        return sum(1 for m in self.mode_log if m == "llf")

    @property
    def n_reserve_steps(self) -> int:
        return sum(1 for m in self.mode_log if m == "reserve")

    def select_action(self, state: ObservableState) -> Action:
        n_urgent = n_urgent_waiting(state)
        mode = "reserve" if n_urgent >= self.tau_urgent else "llf"
        self.mode_log.append(mode)

        if self._last_mode is not None and mode != self._last_mode:
            self.transition_count += 1
            kv_util = 0.0
            if state.gpu_states:
                g = state.gpu_states[0]
                kv_util = float(g.current_kv_tokens) / max(g.max_kv_tokens, 1)
            self.kv_util_at_transition.append(kv_util)
        self._last_mode = mode

        action = self._kv.select_action(state) if mode == "reserve" else self._llf.select_action(state)
        admitted_ids: List[int] = []
        for ids in action.admit.values():
            admitted_ids.extend(ids)
        self.admitted_by_step[state.step] = admitted_ids
        return action


# ===================================================================
# KVAdaptiveReserveHysteresisChildPolicy -- the safety-refined composition target
# ===================================================================

class KVAdaptiveReserveHysteresisChildPolicy(BasePolicy):
    """Safety-refined within-scenario composition target implementing
    transition hysteresis based entirely on online-observable states.

    - LLF -> reserve: require urgent trigger AND current KV occupancy <= 0.90
    - reserve -> LLF: require urgent trigger to clear AND current KV occupancy <= 0.82
    """

    name = "kv_adaptive_reserve_hysteresis_child"

    ENTER_THRESHOLD = 0.63
    RELEASE_THRESHOLD = 0.82

    def __init__(self, tau_urgent: int = 2) -> None:
        self.tau_urgent = int(tau_urgent)
        self._kv = KVConstrainedOnlinePolicy()
        self._llf = LeastLaxityFirstPolicy()
        self.mode_log: List[str] = []
        self.transition_count: int = 0
        self.admitted_by_step: Dict[int, List[int]] = {}
        self.kv_util_at_transition: List[float] = []
        self.n_urgent_at_transition: List[int] = []
        self._last_mode: Optional[str] = None

    def reset(self) -> None:
        self.mode_log.clear()
        self.transition_count = 0
        self.admitted_by_step.clear()
        self.kv_util_at_transition.clear()
        self.n_urgent_at_transition.clear()
        self._last_mode = None
        self._kv = KVConstrainedOnlinePolicy()
        self._llf = LeastLaxityFirstPolicy()

    @property
    def n_llf_steps(self) -> int:
        return sum(1 for m in self.mode_log if m == "llf")

    @property
    def n_reserve_steps(self) -> int:
        return sum(1 for m in self.mode_log if m == "reserve")

    def select_action(self, state: ObservableState) -> Action:
        n_urgent = n_urgent_waiting(state)
        kv_util = 0.0
        if state.gpu_states:
            g = state.gpu_states[0]
            kv_util = float(g.current_kv_tokens) / max(g.max_kv_tokens, 1)

        trigger_reserve = n_urgent >= self.tau_urgent

        if self._last_mode is None:
            mode = "reserve" if trigger_reserve else "llf"
        elif self._last_mode == "llf":
            if trigger_reserve and kv_util <= self.ENTER_THRESHOLD:
                mode = "reserve"
            else:
                mode = "llf"
        else:  # self._last_mode == "reserve"
            if not trigger_reserve and kv_util <= self.RELEASE_THRESHOLD:
                mode = "llf"
            else:
                mode = "reserve"

        self.mode_log.append(mode)

        if self._last_mode is not None and mode != self._last_mode:
            self.transition_count += 1
            self.kv_util_at_transition.append(kv_util)
            self.n_urgent_at_transition.append(n_urgent)
        self._last_mode = mode

        action = self._kv.select_action(state) if mode == "reserve" else self._llf.select_action(state)
        admitted_ids: List[int] = []
        for ids in action.admit.values():
            admitted_ids.extend(ids)
        self.admitted_by_step[state.step] = admitted_ids
        return action
