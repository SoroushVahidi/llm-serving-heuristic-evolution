"""
Selector package: maps online-observable workload features to a deployable scheduling policy.

The selector is a supervised portfolio policy — it does not create new heuristics;
it chooses among the 16 registered deployable policies at window granularity.
"""
from .candidates import SELECTOR_CANDIDATES, SELECTOR_CANDIDATE_COUNT
from .windows import RequestWindow, make_windows, DEFAULT_WINDOW_SIZE, MIN_PARTIAL_WINDOW
from .features import extract_features, FEATURE_NAMES, FeatureMode
from .labels import label_window, label_windows, TieBreaker
from .dataset import build_selector_dataset

__all__ = [
    "SELECTOR_CANDIDATES",
    "SELECTOR_CANDIDATE_COUNT",
    "RequestWindow",
    "make_windows",
    "DEFAULT_WINDOW_SIZE",
    "MIN_PARTIAL_WINDOW",
    "extract_features",
    "FEATURE_NAMES",
    "FeatureMode",
    "label_window",
    "label_windows",
    "TieBreaker",
    "build_selector_dataset",
]
