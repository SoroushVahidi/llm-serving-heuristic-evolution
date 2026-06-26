"""Selector role classification for evaluation reporting."""
from __future__ import annotations

from typing import Dict, List

ORACLE_ASSISTED_SELECTOR_PREFIX = "safe_fallback_wsp_margin"

ALWAYS_FIXED_SELECTORS: List[str] = [
    "always_scorpio",
    "always_wsp",
]

DEPLOYABLE_LEARNED_SELECTORS: List[str] = [
    "rule_based",
    "rf_anwg",
    "rf_anwg_regret",
    "dt_anwg",
    "dt_anwg_regret",
    "knn_anwg",
    "regression_anwg",
]

# Literature-inspired / external-style baselines (subset of deployable policies).
EXTERNAL_STYLE_BASELINES: List[str] = [
    "orca_style",
    "vllm_style_token_budget",
    "sarathi_style",
    "splitfuse_style",
    "multi_bin_batching",
    "estimated_service_time_first",
    "scorpio_style_slo_guard",
]

PRIMARY_RANK_METRIC = "mean_arrival_normalized_wg"

SELECTOR_ROLE_DEPLOYABLE_LEARNED = "deployable_learned"
SELECTOR_ROLE_ALWAYS_FIXED = "always_fixed"
SELECTOR_ROLE_ORACLE_ASSISTED = "oracle_assisted"
SELECTOR_ROLE_UNKNOWN = "unknown"


def is_oracle_assisted_selector(name: str) -> bool:
    return name.startswith(ORACLE_ASSISTED_SELECTOR_PREFIX)


def selector_role(name: str) -> str:
    if name in ALWAYS_FIXED_SELECTORS:
        return SELECTOR_ROLE_ALWAYS_FIXED
    if is_oracle_assisted_selector(name):
        return SELECTOR_ROLE_ORACLE_ASSISTED
    if name in DEPLOYABLE_LEARNED_SELECTORS:
        return SELECTOR_ROLE_DEPLOYABLE_LEARNED
    return SELECTOR_ROLE_UNKNOWN


def is_deployable_headline_selector(name: str) -> bool:
    """Selectors eligible for deployable headline summaries."""
    return selector_role(name) == SELECTOR_ROLE_DEPLOYABLE_LEARNED


def is_external_style_baseline(name: str) -> bool:
    return name in EXTERNAL_STYLE_BASELINES


def classify_selectors(names: List[str]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {
        SELECTOR_ROLE_DEPLOYABLE_LEARNED: [],
        SELECTOR_ROLE_ALWAYS_FIXED: [],
        SELECTOR_ROLE_ORACLE_ASSISTED: [],
        SELECTOR_ROLE_UNKNOWN: [],
    }
    for name in names:
        buckets[selector_role(name)].append(name)
    return buckets
