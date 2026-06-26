"""
Selector package: maps online-observable workload features to a deployable scheduling policy.

The selector is a supervised portfolio policy — it does not create new heuristics;
it chooses among the 16 registered deployable policies at window granularity.
"""
from .candidates import SELECTOR_CANDIDATES, SELECTOR_CANDIDATE_COUNT, SELECTOR_CANDIDATE_POLICIES
from .windows import RequestWindow, make_windows, DEFAULT_WINDOW_SIZE, MIN_PARTIAL_WINDOW
from .features import (
    extract_features,
    FEATURE_NAMES,
    FeatureMode,
    parse_feature_mode,
    feature_mode_is_deployable,
)
from .roles import (
    classify_selectors,
    is_deployable_headline_selector,
    is_oracle_assisted_selector,
    selector_role,
)
from .labels import label_window, label_windows, TieBreaker
from .dataset import build_selector_dataset

__all__ = [
    "SELECTOR_CANDIDATES",
    "SELECTOR_CANDIDATE_POLICIES",
    "SELECTOR_CANDIDATE_COUNT",
    "RequestWindow",
    "make_windows",
    "DEFAULT_WINDOW_SIZE",
    "MIN_PARTIAL_WINDOW",
    "extract_features",
    "FEATURE_NAMES",
    "FeatureMode",
    "parse_feature_mode",
    "feature_mode_is_deployable",
    "classify_selectors",
    "is_deployable_headline_selector",
    "is_oracle_assisted_selector",
    "selector_role",
    "label_window",
    "label_windows",
    "TieBreaker",
    "build_selector_dataset",
]
