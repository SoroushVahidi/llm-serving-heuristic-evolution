"""ESTF/WFS composition policies (ranking-only; shared placement path)."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

from ..core.action import Action
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..policies.composition import (
    RankExpertSpec,
    StaticRankEnsemblePolicy,
    causal_context_features,
    rank_with_named_expert,
    weighted_borda_aggregate,
)
from ..policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from ..policies.policy_library_v2_helpers import deterministic_place
from ..policies.weighted_fair_share import WeightedFairSharePolicy
from .estf_wfs_features import FEATURE_NAMES, assert_no_hidden_leakage, scenario_observable_features
from .estf_wfs_models import FittedAlphaModel, FittedSelector, hard_conditional_rule

ESTF = "estimated_service_time_first"
WFS = "weighted_fair_share"


def make_static_estf_wfs_blend(alpha_estf: float) -> StaticRankEnsemblePolicy:
    """score = alpha*rank_ESTF + (1-alpha)*rank_WFS (normalized Borda ranks)."""
    if not 0.0 <= alpha_estf <= 1.0:
        raise ValueError(f"alpha_estf must be in [0,1], got {alpha_estf}")
    policy = StaticRankEnsemblePolicy(
        [
            RankExpertSpec(ESTF, float(alpha_estf) if alpha_estf > 0 else 0.0),
            RankExpertSpec(WFS, float(1.0 - alpha_estf) if alpha_estf < 1 else 0.0),
        ],
        # Disable zero-weight experts via weight; _normalize_weights drops zeros.
        fallback_policy=EstimatedServiceTimeFirstPolicy()
        if alpha_estf >= 0.5
        else WeightedFairSharePolicy(),
    )
    # If alpha is exactly 0 or 1, only one expert has positive weight.
    if alpha_estf <= 0.0:
        policy = StaticRankEnsemblePolicy(
            [RankExpertSpec(WFS, 1.0)],
            fallback_policy=WeightedFairSharePolicy(),
        )
    elif alpha_estf >= 1.0:
        policy = StaticRankEnsemblePolicy(
            [RankExpertSpec(ESTF, 1.0)],
            fallback_policy=EstimatedServiceTimeFirstPolicy(),
        )
    policy.name = f"estf_wfs_static_alpha_{alpha_estf:.2f}"
    return policy


class EstfWfsTop1SelectorPolicy(BasePolicy):
    """Each step: choose ESTF or WFS ranking from a fitted contextual classifier.

    Scenario-level features are frozen at construction (from the loaded trace).
    Online causal features are logged but the decision uses the scenario vector
    so the learned map matches the train labels (scenario-level ANWG winners).
    """

    name = "estf_wfs_contextual_top1"

    def __init__(
        self,
        selector: FittedSelector,
        scenario_features: Mapping[str, float],
    ) -> None:
        assert_no_hidden_leakage(scenario_features)
        self.selector = selector
        self.scenario_features = dict(scenario_features)
        self.estf = EstimatedServiceTimeFirstPolicy()
        self.wfs = WeightedFairSharePolicy()
        self.decision_log: list[dict] = []
        self.switch_count = 0
        self._last_choice: Optional[str] = None

    def reset(self) -> None:
        self.estf.reset()
        self.wfs.reset()
        self.decision_log.clear()
        self.switch_count = 0
        self._last_choice = None

    def select_action(self, state: ObservableState) -> Action:
        choice = self.selector.predict_parent(self.scenario_features)
        if self._last_choice is not None and choice != self._last_choice:
            self.switch_count += 1
        self._last_choice = choice
        online = causal_context_features(state)
        self.decision_log.append(
            {
                "step": state.step,
                "choice": choice,
                "queue_length": online.get("queue_length", 0.0),
            }
        )
        if choice == "estf":
            return self.estf.select_action(state)
        return self.wfs.select_action(state)


class EstfWfsContextualAlphaPolicy(BasePolicy):
    """Each step: discrete alpha(x) blend of normalized ESTF/WFS ranks."""

    name = "estf_wfs_contextual_alpha"

    def __init__(
        self,
        alpha_model: FittedAlphaModel,
        scenario_features: Mapping[str, float],
    ) -> None:
        assert_no_hidden_leakage(scenario_features)
        self.alpha_model = alpha_model
        self.scenario_features = dict(scenario_features)
        self.decision_log: list[dict] = []
        self.alpha_history: list[float] = []
        self._last_alpha: Optional[float] = None
        self.switch_count = 0

    def reset(self) -> None:
        self.decision_log.clear()
        self.alpha_history.clear()
        self._last_alpha = None
        self.switch_count = 0

    def select_action(self, state: ObservableState) -> Action:
        alpha = self.alpha_model.predict_alpha(self.scenario_features)
        if self._last_alpha is not None and abs(alpha - self._last_alpha) > 1e-12:
            self.switch_count += 1
        self._last_alpha = alpha
        self.alpha_history.append(alpha)
        online = causal_context_features(state)
        self.decision_log.append(
            {
                "step": state.step,
                "alpha": alpha,
                "queue_length": online.get("queue_length", 0.0),
            }
        )
        if alpha <= 0.0:
            return WeightedFairSharePolicy().select_action(state)
        if alpha >= 1.0:
            return EstimatedServiceTimeFirstPolicy().select_action(state)

        weights = {ESTF: alpha, WFS: 1.0 - alpha}
        outputs = {name: rank_with_named_expert(name, state) for name in weights}
        aggregate, support, _contrib = weighted_borda_aggregate(outputs, weights)
        by_id = {r.request_id: r for r in state.waiting_queue}
        ranked_ids = [
            rid
            for rid, _val in sorted(
                aggregate.items(),
                key=lambda item: (
                    -item[1],
                    -support.get(item[0], 0),
                    by_id[item[0]].arrival_time,
                    item[0],
                ),
            )
            if rid in by_id
        ]
        ranked = [by_id[rid] for rid in ranked_ids]
        return deterministic_place(state, ranked)


class EstfWfsHardConditionalPolicy(BasePolicy):
    """Symbolic if/else over observable scenario features → ESTF or WFS."""

    name = "estf_wfs_hard_conditional"

    def __init__(self, scenario_features: Mapping[str, float]) -> None:
        assert_no_hidden_leakage(scenario_features)
        self.scenario_features = dict(scenario_features)
        self.estf = EstimatedServiceTimeFirstPolicy()
        self.wfs = WeightedFairSharePolicy()
        self.choice = hard_conditional_rule(scenario_features)
        self.decision_log: list[dict] = []

    def reset(self) -> None:
        self.estf.reset()
        self.wfs.reset()
        self.decision_log.clear()

    def select_action(self, state: ObservableState) -> Action:
        self.decision_log.append({"step": state.step, "choice": self.choice})
        if self.choice == "estf":
            return self.estf.select_action(state)
        return self.wfs.select_action(state)


def alpha_collapse_stats(alphas: list[float], *, edge: float = 0.05) -> Dict[str, float]:
    if not alphas:
        return {
            "n": 0.0,
            "frac_near_0": 0.0,
            "frac_near_1": 0.0,
            "frac_intermediate": 0.0,
            "mean_alpha": float("nan"),
        }
    near0 = sum(1 for a in alphas if a <= edge)
    near1 = sum(1 for a in alphas if a >= 1.0 - edge)
    n = len(alphas)
    return {
        "n": float(n),
        "frac_near_0": near0 / n,
        "frac_near_1": near1 / n,
        "frac_intermediate": (n - near0 - near1) / n,
        "mean_alpha": sum(alphas) / n,
    }
