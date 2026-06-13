"""
Policy registry: map string names to policy instances.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import BasePolicy
from .edf import EDFPolicy
from .fifo import FIFOPolicy
from .greedy_token_fill import GreedyTokenFillPolicy
from .least_loaded import LeastLoadedPolicy
from .multi_bin_batching import MultiBinBatchingPolicy
from .orca_style import OrcaStylePolicy
from .random_feasible import RandomFeasiblePolicy
from .sarathi_style import SarathiStylePolicy
from .shortest_output_first import ShortestOutputFirstPolicy
from .shortest_prompt_first import ShortestPromptFirstPolicy
from .slo_slack_score import SloSlackScorePolicy
from .splitfuse_style import SplitFuseStylePolicy
from .vllm_style_token_budget import VLLMStyleTokenBudgetPolicy
from .weighted_shortest_processing import WeightedShortestProcessingPolicy


_REGISTRY: Dict[str, type] = {
    # Phase 1 baselines
    "fifo":                    FIFOPolicy,
    "edf":                     EDFPolicy,
    "shortest_output_first":   ShortestOutputFirstPolicy,
    "shortest_prompt_first":   ShortestPromptFirstPolicy,
    "greedy_token_fill":       GreedyTokenFillPolicy,
    "least_loaded":            LeastLoadedPolicy,
    "multi_bin_batching":      MultiBinBatchingPolicy,
    "random_feasible":         RandomFeasiblePolicy,
    # Phase 1.5 serving-style baselines
    "orca_style":              OrcaStylePolicy,
    "vllm_style_token_budget": VLLMStyleTokenBudgetPolicy,
    "sarathi_style":           SarathiStylePolicy,
    "splitfuse_style":         SplitFuseStylePolicy,
    "slo_slack_score":         SloSlackScorePolicy,
    "weighted_shortest_processing": WeightedShortestProcessingPolicy,
}

BASELINE_NAMES: List[str] = list(_REGISTRY.keys())

# Convenience subsets for experiment configs
PHASE1_BASELINES: List[str] = [
    "fifo", "edf", "shortest_output_first", "shortest_prompt_first",
    "greedy_token_fill", "least_loaded", "multi_bin_batching", "random_feasible",
]

SERVING_STYLE_BASELINES: List[str] = [
    "orca_style", "vllm_style_token_budget", "sarathi_style",
    "splitfuse_style", "slo_slack_score", "weighted_shortest_processing",
]


def make_policy(name: str, seed: int = 0, **kwargs) -> BasePolicy:
    """Instantiate a policy by name.

    Parameters
    ----------
    name : str
        Registry key (e.g. "fifo", "orca_style").
    seed : int
        Passed to stochastic policies (random_feasible).
    **kwargs
        Additional constructor arguments forwarded to the policy constructor.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown policy '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[name]
    if name == "random_feasible":
        return cls(seed=seed, **kwargs)
    return cls(**kwargs)


def all_baseline_policies(seed: int = 0) -> List[BasePolicy]:
    """Return one instance of every registered baseline policy."""
    return [make_policy(name, seed=seed) for name in BASELINE_NAMES]


def phase1_policies(seed: int = 0) -> List[BasePolicy]:
    """Return one instance of each Phase 1 baseline policy."""
    return [make_policy(name, seed=seed) for name in PHASE1_BASELINES]


def serving_style_policies(seed: int = 0) -> List[BasePolicy]:
    """Return one instance of each Phase 1.5 serving-style policy."""
    return [make_policy(name, seed=seed) for name in SERVING_STYLE_BASELINES]
