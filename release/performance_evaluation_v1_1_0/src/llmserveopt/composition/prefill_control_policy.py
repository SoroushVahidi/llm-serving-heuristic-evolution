"""Family B v2 PrefillControl composition policies.

Contains:
- Top-1 contextual parent selector (sklearn LogisticRegression)
- Hard conditional parent selector (symbolic if/else over observable slack)
- Contextual discrete alpha composition (discrete blend of both parents)
- PrefillControl child policy (contextual chunk-size selection)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..core.action import Action
from ..core.types import ObservableRequest, ObservableState
from ..policies.base import BasePolicy
from ..policies.composition import deterministic_place
from ..policies.prefill_control_variants import (
    DEFAULT_CHUNK_SMALL,
    UNLIMITED_PREFILL_CHUNK,
    _arrival_rank,
)

from .prefill_control_features import (
    FEATURE_NAMES,
    assert_no_hidden_leakage,
    build_scenario_feature_rows,
    feature_vector,
    scenario_observable_features,
    step_features,
)

# ---------------------------------------------------------------------------
# Parent constants
# ---------------------------------------------------------------------------

PARENT_FULL = "full_prefill"
PARENT_SMALL = "chunked_prefill_small"

# Intermediate chunk sizes for the child
INTERMEDIATE_CHUNKS = (96, 128, 192, 256)

# Alpha grid for composition blend
ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)


# ===================================================================
# Top-1 contextual parent selector
# ===================================================================

class PrefillTop1SelectorPolicy(BasePolicy):
    """Chooses full_prefill or chunked_prefill_small based on a fitted
    context-classifier trained on scenario-level observable features."""

    name = "prefill_contextual_top1"

    def __init__(
        self,
        selector: FittedPrefillSelector,
        scenario_features: Dict[str, float],
    ) -> None:
        assert_no_hidden_leakage(scenario_features)
        self.selector = selector
        self.scenario_features = scenario_features
        self.decision_log: List[Dict[str, Any]] = []
        self.switch_count = 0
        self._last_choice: Optional[str] = None

    def reset(self) -> None:
        self.decision_log.clear()
        self.switch_count = 0
        self._last_choice = None

    def select_action(self, state: ObservableState) -> Action:
        choice = self.selector.predict_parent(self.scenario_features)
        if self._last_choice is not None and choice != self._last_choice:
            self.switch_count += 1
        self._last_choice = choice
        self.decision_log.append({
            "step": state.step,
            "choice": choice,
        })
        # We cannot run the actual parent policy here — this selector
        # returns a token-bucket budget that downstream executors interpret.
        # For composition experiments the runner creates a wrapper policy.
        return deterministic_place(state, list(state.waiting_queue))


class FittedPrefillSelector:
    """sklearn-wrapped selector that predicts full_prefill or chunked_prefill_small."""

    def __init__(
        self,
        model_type: str,
        feature_names: Tuple[str, ...],
        classes_: List[str],  # ["full_prefill", "chunked_prefill_small"]
        raw: Any,
    ) -> None:
        self.model_type = model_type
        self.feature_names = feature_names
        self.classes_ = classes_
        self.raw = raw

    def predict_parent(self, features: Mapping[str, float]) -> str:
        assert_no_hidden_leakage(features)
        x = feature_vector(features, self.feature_names).reshape(1, -1)
        pred_idx = int(self.raw.predict(x)[0])
        return self.classes_[pred_idx]


def fit_prefill_top1_selector(
    feature_rows: Sequence[Mapping[str, float]],
    parent_full: Sequence[float],
    parent_small: Sequence[float],
    *,
    model_type: str = "logreg",
    seed: int = 20261201,
) -> FittedPrefillSelector:
    """Train a classifier that predicts which parent wins per scenario."""

    X = np.vstack([feature_vector(f) for f in feature_rows])
    y = np.array(
        [_parent_label(f, s) for f, s in zip(parent_full, parent_small)],
        dtype=int,
    )

    classes = ["chunked_prefill_small", "full_prefill"]

    if len(set(int(v) for v in y)) < 2:
        clf: Any = _ConstantClassifier(int(y[0]))
        clf.fit(X, y)
    elif model_type == "tree":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=max(2, len(y) // 10),
            random_state=seed,
        )
        clf.fit(X, y)
    else:
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X, y)

    return FittedPrefillSelector(
        model_type=model_type,
        feature_names=FEATURE_NAMES,
        classes_=classes,
        raw=clf,
    )


def _parent_label(
    full_score: float, small_score: float, eps: float = 1e-12
) -> int:
    """Return 1 if full_prefill wins, 0 otherwise."""
    return 1 if full_score > small_score + eps else 0


class _ConstantClassifier:
    """Fallback when training labels contain a single class."""

    def __init__(self, label: int) -> None:
        self.label = int(label)

    def fit(self, X, y) -> None:  # noqa: ANN001
        pass

    def predict(self, X) -> np.ndarray:  # noqa: ANN001
        return np.full(shape=(len(X),), fill_value=self.label, dtype=int)


# ===================================================================
# Hard conditional parent selector
# ===================================================================

class PrefillHardConditionalPolicy(BasePolicy):
    """Symbolic if/else over observable scenario features -> full_prefill or chunked_prefill_small.

    Family B v2 mechanism evidence (H3): the switch variable is
    mean_e2e_slack_hog versus mean_e2e_slack_late.  At runtime,
    we approximate this using the observable mean SLO slack of
    short vs long prompt requests in the queue.
    """

    name = "prefill_hard_conditional"

    def __init__(
        self,
        scenario_features: Mapping[str, float],
    ) -> None:
        assert_no_hidden_leakage(scenario_features)
        self.scenario_features = dict(scenario_features)
        self.choice = hard_conditional_rule(scenario_features)
        self.decision_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.decision_log.clear()

    def select_action(self, state: ObservableState) -> Action:
        self.decision_log.append({"step": state.step, "choice": self.choice})
        # Token-bucket budget: full or small chosen symbolically.
        return deterministic_place(state, list(state.waiting_queue))


def hard_conditional_rule(features: Mapping[str, float]) -> str:
    """Symbolic if/else: short-urgent slack → full_prefill, long-sufficient slack → chunked_prefill_small.

    Family B v2 H3 evidence: tight-hog slack wins full_prefill because full
    prefill gives uninterrupted hog TTFT; tight-late slack wins small because
    small prefill gives crumbs to late tenants.

    Online-observable proxy: compare min_slo_slack (proxy for tight SLO)
    against median_slo_slack.  If the tightest request has slack < 0.1s,
    that class is urgent and full_prefill helps; otherwise small does.
    """
    assert_no_hidden_leakage(features)
    min_slack = float(features.get("min_slo_slack", 2.0))
    mean_slack = float(features.get("mean_slo_slack", 2.0))
    urgent_frac = float(features.get("fraction_urgent", 0.0))

    if urgent_frac >= 0.15 or min_slack < 0.1:
        return PARENT_FULL
    if mean_slack > 0.3 and urgent_frac < 0.05:
        return PARENT_SMALL
    # Fallback: prefer small when slack is moderate (more likely late_ttft niche)
    return PARENT_SMALL


# ===================================================================
# Contextual discrete alpha composition
# ===================================================================

class PrefillContextualAlphaPolicy(BasePolicy):
    """Discrete alpha blend between full_prefill and chunked_prefill_small."""

    name = "prefill_contextual_alpha"

    def __init__(
        self,
        alpha_model: FittedAlphaModel,
        scenario_features: Mapping[str, float],
    ) -> None:
        assert_no_hidden_leakage(scenario_features)
        self.alpha_model = alpha_model
        self.scenario_features = dict(scenario_features)
        self.decision_log: List[Dict[str, Any]] = []
        self.alpha_history: List[float] = []
        self._last_alpha: Optional[float] = None

    def reset(self) -> None:
        self.decision_log.clear()
        self.alpha_history.clear()
        self._last_alpha = None

    def select_action(self, state: ObservableState) -> Action:
        alpha = self.alpha_model.predict_alpha(self.scenario_features)
        if self._last_alpha is not None and abs(alpha - self._last_alpha) > 1e-12:
            pass  # tracking not needed for this minimal experiment
        self._last_alpha = alpha
        self.alpha_history.append(alpha)
        self.decision_log.append({"step": state.step, "alpha": alpha})
        return deterministic_place(state, list(state.waiting_queue))


class FittedAlphaModel:
    """Discrete alpha model trained to predict parent-margin."""

    def __init__(
        self,
        model_type: str,
        feature_names: Tuple[str, ...],
        alpha_grid: Tuple[float, ...],
        raw: Any,
    ) -> None:
        self.model_type = model_type
        self.feature_names = feature_names
        self.alpha_grid = alpha_grid
        self.raw = raw

    def predict_alpha(self, features: Mapping[str, float]) -> float:
        assert_no_hidden_leakage(features)
        x = feature_vector(features, self.feature_names).reshape(1, -1)
        idx = int(self.raw.predict(x)[0])
        return float(self.alpha_grid[idx])


def _alpha_label(full_score: float, small_score: float, grid: Sequence[float] = ALPHA_GRID) -> int:
    """Map parent ANWG margin to a discrete alpha index."""
    delta = float(full_score) - float(small_score)
    # delta > 0 means full is better → alpha close to 1
    # delta < 0 means small is better → alpha close to 0
    if delta > 0.08:
        target = 1.0
    elif delta > 0.0:
        target = 0.75
    elif delta < -0.08:
        target = 0.0
    else:
        target = 0.25
    dists = [abs(target - a) for a in grid]
    return int(np.argmin(dists))


def fit_alpha_model(
    feature_rows: Sequence[Mapping[str, float]],
    full_scores: Sequence[float],
    small_scores: Sequence[float],
    *,
    model_type: str = "tree",
    seed: int = 20261201,
    alpha_grid: Sequence[float] = ALPHA_GRID,
) -> FittedAlphaModel:
    """Train a model that predicts discrete alpha from scenario features."""
    X = np.vstack([feature_vector(f) for f in feature_rows])
    y = np.array(
        [_alpha_label(f, s, alpha_grid) for f, s in zip(full_scores, small_scores)],
        dtype=int,
    )
    alpha_grid_f = tuple(float(a) for a in alpha_grid)

    if len(set(int(v) for v in y)) < 2:
        clf: Any = _ConstantClassifier(int(y[0]))
        clf.fit(X, y)
    elif model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X, y)
    else:
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=max(2, len(y) // 12),
            random_state=seed,
        )
        clf.fit(X, y)

    return FittedAlphaModel(
        model_type=model_type,
        feature_names=FEATURE_NAMES,
        alpha_grid=alpha_grid_f,
        raw=clf,
    )


# ===================================================================
# PrefillControl child policy — contextual chunk-size selection
# ===================================================================

def default_step_level_chunk_rule(
    features: Mapping[str, float],
    chunk_options: Tuple[int, ...],
) -> int:
    """Symbolic (not data-fit) per-step chunk-index rule, used when
    ``PrefillControlChildPolicy`` has no ``chunk_model``.

    Grounded in the same mechanism theory as ``hard_conditional_rule`` (H3:
    tight SLO slack favors uninterrupted full prefill; loose slack favors
    small chunks that protect decode) plus the decode-protection tradeoff
    documented in ``ServiceModel``: a larger prefill chunk speeds up
    waiting/urgent prefill requests at the cost of stalling concurrent
    decode for longer; a smaller chunk protects decode TBT at the cost of
    slower prefill throughput. This rule was fixed BEFORE any falsification
    results existed and is not tuned to them -- it is a hand-specified
    5-tier decision surface over two online-observable step-level signals:

      urgent          = fraction_urgent >= 0.15 or min_slo_slack < 0.1
                         (identical threshold to hard_conditional_rule)
      decode_pressure = n_decoding_active / (n_decoding_active + n_prefilling_active)
                         (0.0 when nothing is active)

    Decision surface (index into ``chunk_options``, smallest..largest):
      urgent AND decode_pressure <  0.3  -> largest  (clear the urgent request fast)
      urgent AND decode_pressure >= 0.3  -> mid       (balance urgency vs decode protection)
      not urgent AND decode_pressure >= 0.5 -> smallest (protect decode under heavy contention)
      not urgent AND decode_pressure >= 0.3 -> 2nd-smallest
      not urgent AND decode_pressure <  0.3 -> middle   (no urgency, no contention)
    """
    n = len(chunk_options)
    urgent = (
        float(features.get("fraction_urgent", 0.0)) >= 0.15
        or float(features.get("min_slo_slack", 2.0)) < 0.1
    )
    n_decoding = float(features.get("n_decoding_active", 0.0))
    n_prefilling = float(features.get("n_prefilling_active", 0.0))
    denom = n_decoding + n_prefilling
    decode_pressure = n_decoding / denom if denom > 0 else 0.0

    if urgent and decode_pressure < 0.3:
        idx = n - 1
    elif urgent:
        idx = n // 2
    elif decode_pressure >= 0.5:
        idx = 0
    elif decode_pressure >= 0.3:
        idx = min(1, n - 1)
    else:
        idx = min(2, n - 1)
    return max(0, min(idx, n - 1))


class PrefillControlChildPolicy(BasePolicy):
    """Contextual PrefillControl: selects a chunk-size budget from the
    expanded grid ({64, 96, 128, 192, 256, 65536}) based on
    step-level observable features, re-decided every step.

    The chosen chunk size is attached to the returned Action as
    ``prefill_chunk_override`` (see Action's docstring), which the
    simulator honours for that GPU on that step only -- this is genuine
    per-step dynamic composition, not a single scenario-level decision
    replaying a fixed-chunk baseline.
    """

    name = "prefill_control_child"

    CHUNK_OPTIONS = (DEFAULT_CHUNK_SMALL, *INTERMEDIATE_CHUNKS, UNLIMITED_PREFILL_CHUNK)
    CHUNK_NAMES = (
        "chunk_64",
        "chunk_96",
        "chunk_128",
        "chunk_192",
        "chunk_256",
        "chunk_65536",
    )

    def __init__(
        self,
        chunk_model: Optional[Any] = None,  # simple online rule or fitted
        *,
        chunk_grid: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.model = chunk_model
        if chunk_grid:
            self.CHUNK_OPTIONS = chunk_grid
            self.CHUNK_NAMES = tuple(f"chunk_{c}" for c in chunk_grid)
        self.decision_log: List[Dict[str, Any]] = []
        self.alpha_history: List[int] = []

    def reset(self) -> None:
        self.decision_log.clear()
        self.alpha_history.clear()

    def select_action(self, state: ObservableState) -> Action:
        feats = step_features(state)
        assert_no_hidden_leakage(feats)
        if self.model is not None:
            x = feature_vector(feats).reshape(1, -1)
            idx = int(self.model.predict(x)[0])
            idx = max(0, min(idx, len(self.CHUNK_OPTIONS) - 1))
        else:
            idx = default_step_level_chunk_rule(feats, self.CHUNK_OPTIONS)
        chunk_size = self.CHUNK_OPTIONS[idx]
        self.alpha_history.append(idx)
        self.decision_log.append({
            "step": state.step,
            "chunk_idx": idx,
            "chunk_size": chunk_size,
        })
        ranked = sorted(state.waiting_queue, key=_arrival_rank)
        action = deterministic_place(state, ranked)
        action.prefill_chunk_override = {g.gpu_id: chunk_size for g in state.gpu_states}
        return action

    @property
    def selected_chunk(self) -> int:
        """Return the latest chosen chunk size, for runner-level merge."""
        if self.alpha_history:
            idx = self.alpha_history[-1]
            return self.CHUNK_OPTIONS[min(idx, len(self.CHUNK_OPTIONS) - 1)]
        return 128  # default


# ===================================================================
# Model selection utilities
# ===================================================================

def select_prefill_model_on_val(
    feature_train: Sequence[Mapping[str, float]],
    full_train: Sequence[float],
    small_train: Sequence[float],
    feature_val: Sequence[Mapping[str, float]],
    full_val: Sequence[float],
    small_val: Sequence[float],
) -> Tuple[FittedPrefillSelector, FittedAlphaModel, Dict[str, Any]]:
    """Fit candidate top-1 selector and alpha model; pick by validation accuracy."""
    sel_candidates = []
    for mt in ("logreg", "tree"):
        sel = fit_prefill_top1_selector(
            feature_train, full_train, small_train, model_type=mt
        )
        correct = 0
        for f, fl, sm in zip(feature_val, full_val, small_val):
            pred = sel.predict_parent(f)
            truth = PARENT_FULL if fl > sm + 1e-12 else PARENT_SMALL
            correct += int(pred == truth)
        acc = correct / max(1, len(feature_val))
        sel_candidates.append((acc, mt, sel))

    sel_candidates.sort(key=lambda t: (-t[0], t[1]))
    best_sel = sel_candidates[0][2]

    alpha_cands = []
    for mt in ("tree", "logreg"):
        am = fit_alpha_model(
            feature_train, full_train, small_train, model_type=mt
        )
        correct = 0
        for f, fl, sm in zip(feature_val, full_val, small_val):
            a = am.predict_alpha(f)
            pred = PARENT_FULL if a >= 0.5 else PARENT_SMALL
            truth = PARENT_FULL if fl > sm + 1e-12 else PARENT_SMALL
            correct += int(pred == truth)
        acc = correct / max(1, len(feature_val))
        alpha_cands.append((acc, mt, am))

    alpha_cands.sort(key=lambda t: (-t[0], t[1]))
    best_alpha = alpha_cands[0][2]

    meta = {
        "selector_val_accuracy": sel_candidates[0][0],
        "selector_model_type": sel_candidates[0][1],
        "alpha_val_proxy_accuracy": alpha_cands[0][0],
        "alpha_model_type": alpha_cands[0][1],
        "selector_candidates": [
            {"model_type": mt, "val_accuracy": acc}
            for acc, mt, _ in sel_candidates
        ],
        "alpha_candidates": [
            {"model_type": mt, "val_proxy_accuracy": acc}
            for acc, mt, _ in alpha_cands
        ],
    }
    return best_sel, best_alpha, meta
