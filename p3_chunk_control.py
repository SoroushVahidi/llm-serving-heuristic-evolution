#!/usr/bin/env python3
"""Family B v2 PrefillControl composition — child / composition / chunk-control logic.

This module is the Family B v2 composition engine. It imports from four
composition-submodules and re-exports only the interfaces needed by the
runner (p7_runner.py) and analysis (p5_analysis_chunk_comp.py).

Usage: import p3_chunk_control  (after sys.path includes src/)

Key invariants:
- Uses ``templates_prefill_decode_v2`` (Family B v2) **only**.
- No generator labels (scenario_id, seed, hog_count, late_pressure,
  slo_emphasis, intended_winner, class_id with ".hog" suffix, etc.)
  leak into features or policy decisions.
- Family B v2 classes are ``tenant_prefill`` / ``tenant_late`` — no
  ``class_id.endswith(".hog")`` assumptions.
- Composition endpoint behaviour is deterministic: when the child policy
  is forced to pick the parent-endpoint chunk sizes, it exactly reproduces
  the parent scores on every scenario.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package-level imports resolve when this file is imported as a module.
_ROOT = Path(__file__).resolve().parent
if (Path(_ROOT / "src")).is_dir():
    sys.path.insert(0, str(_ROOT / "src"))
elif (Path(_ROOT / ".." / "src")).resolve().is_dir():
    sys.path.insert(0, str(_ROOT / ".." / "src"))

# === Family B v2 scenario template ===

from llmserveopt.policy_separation.templates_prefill_decode_v2 import (
    CLASS_HOG as CLASS_PREFILL,
    CLASS_LATE,
    ALLOWED_CLASS_IDS,
    case_prefill_decode_ttft_contention,
    assert_policy_visible_fields_clean_v2,
)

# === v2 policy variants (parents) ===

from llmserveopt.policies.prefill_control_variants import (
    DEFAULT_CHUNK_SMALL,
    UNLIMITED_PREFILL_CHUNK,
    GreedyArrivalPrefillControlPolicy,
    make_prefill_decode_variants_v2,
)

# === Composition modules ===

from llmserveopt.composition.prefill_control_features import (
    FEATURE_NAMES,
    FORBIDDEN_FEATURE_KEYS,
    assert_no_hidden_leakage,
    scenario_observable_features,
    step_features,
    feature_vector,
    build_scenario_feature_rows,
)

from llmserveopt.composition.prefill_control_policy import (
    PARENT_FULL,
    PARENT_SMALL,
    INTERMEDIATE_CHUNKS,
    ALPHA_GRID,
    PrefillTop1SelectorPolicy,
    PrefillHardConditionalPolicy,
    PrefillContextualAlphaPolicy,
    PrefillControlChildPolicy,
    FittedPrefillSelector,
    FittedAlphaModel,
    fit_prefill_top1_selector,
    fit_alpha_model,
    hard_conditional_rule,
    select_prefill_model_on_val,
)

from llmserveopt.composition.prefill_control_metrics import (
    PRIMARY,
    PRACTICAL_EPS,
    parent_envelope,
    envelope_gain,
    bootstrap_ci,
    paired_bootstrap_ci,
    paired_deltas,
    best_fixed_parent_score,
    best_fixed_intermediate_score,
    pairwise_comparison,
    oracle_scores,
    oracle_regret,
)

from llmserveopt.composition.prefill_control_splits import (
    SplitAssignment,
    assign_family_b_v2_splits,
    assert_no_split_leakage,
    PILOT_SEEDS,
    TRAIN_SEEDS,
    TEST_SEED,
    VAL_SEED,
)

# ===================================================================
# Composition grid definition
# ===================================================================

# Full composition grid: every chunk option, including both parents.
CHILD_CHUNK_OPTIONS: tuple[int, ...] = (DEFAULT_CHUNK_SMALL, 96, 128, 192, 256, UNLIMITED_PREFILL_CHUNK)
CHILD_CHUNK_NAMES: tuple[str, ...] = (
    "chunk_64", "chunk_96", "chunk_128", "chunk_192", "chunk_256", "chunk_65536"
)

# Composition parameter control grid (for the contextual prefill control child).
# These are the step-level chunk-size budgets the child can select.
CHILD_CONTROL_GRID: tuple[int, ...] = CHILD_CHUNK_OPTIONS

# Fixed intermediate parents for baseline comparison.
FIXED_INTERMEDIATE_PARENTS: tuple[dict, ...] = (
    {"name": "chunk_96", "max_prefill_chunk_tokens": 96, "decode_first": False},
    {"name": "chunk_128", "max_prefill_chunk_tokens": 128, "decode_first": False},
    {"name": "chunk_192", "max_prefill_chunk_tokens": 192, "decode_first": False},
)

# Composition baselines computed at analysis time.
BASINELIST: tuple[str, ...] = (
    "full_prefill",
    "chunked_prefill_small",
    "best_fixed_parent",
    "parent_oracle",
    "contextual_top1",
    "hard_conditional",
    "contextual_alpha",
    "best_fixed_intermediate",
)

# ===================================================================
# Composition operator: merges a parent config with a chunk control
# ===================================================================

def composition_config(
    chunk_size: int,
) -> dict:
    """Return a composition config dict for a given chunk size.

    This is the composition operator: given a chunk-size budget, return
    the service-model kwargs that the runner merges over the scenario base.
    """
    return {
        "max_prefill_chunk_tokens": int(chunk_size),
        "decode_first": False,
    }


def composition_policy_for_chunk(
    chunk_size: int,
    policy: GreedyArrivalPrefillControlPolicy | None = None,
) -> GreedyArrivalPrefillControlPolicy:
    """Return a policy instance for a fixed chunk configuration.

    All parents and fixed intermediates use GreedyArrivalPrefillControlPolicy;
    the mechanism difference lives in the service-model kwargs.
    """
    if policy is None:
        policy = GreedyArrivalPrefillControlPolicy()
    policy.name = f"chunk_{chunk_size}" if chunk_size != UNLIMITED_PREFILL_CHUNK else "chunk_65536"
    return policy


# ===================================================================
# Composition endpoint identity: reproducing parent behaviour
# ===================================================================

def make_parent_config(name: str) -> dict:
    """Return the correct service-model kwargs for a parent by name.

    Ensures the composition operator exactly reproduces each parent
    at its corresponding endpoint.
    """
    if name == PARENT_FULL:
        return {"max_prefill_chunk_tokens": UNLIMITED_PREFILL_CHUNK, "decode_first": False}
    elif name == PARENT_SMALL:
        return {"max_prefill_chunk_tokens": DEFAULT_CHUNK_SMALL, "decode_first": False}
    else:
        raise KeyError(f"Unknown parent: {name!r}")


# ===================================================================
# Selector training interface
# ===================================================================

def train_selector(
    feature_rows_train: list[dict],
    parent_full_scores: list[float],
    parent_small_scores: list[float],
    feature_rows_val: list[dict],
    parent_full_scores_val: list[float],
    parent_small_scores_val: list[float],
    *,
    family: str = "family_b_prefill_decode_v2",
) -> tuple[FittedPrefillSelector, dict]:
    """Train the contextual top-1 selector between the two parents.

    Uses only allowed Family B v2 online-observable feature vectors.
    Returns (best_selector, meta).

    Meta contains:
    - model_type: "logreg" or "tree"
    - feature_names: list[str]
    - classes_: ["chunked_prefill_small", "full_prefill"]
    - selector_val_accuracy: float
    - alpha_val_proxy_accuracy: float
    - alpha_model_type: str
    """
    best_sel, best_alpha, meta = select_prefill_model_on_val(
        feature_rows_train, parent_full_scores, parent_small_scores,
        feature_rows_val, parent_full_scores_val, parent_small_scores_val,
    )

    meta["model_type"] = meta.get("selector_model_type", "logreg")
    meta["feature_names"] = list(best_sel.feature_names)
    meta["classes_"] = best_sel.classes_
    meta["family"] = family
    return best_sel, meta


# ===================================================================
# Feature-schema guard: assert observable-only at scenario and step level
# ===================================================================

def assert_scenario_features_obs_only(
    scenario,
    *,
    feature_rows: list[dict],
    scenario_idx: int,
) -> dict:
    """Extract scenario features and verify no forbidden leakage.

    Returns the feature dict for the scenario.
    """
    feats = scenario_observable_features(list(scenario.requests))
    for k in FORBIDDEN_FEATURE_KEYS:
        assert k not in feats, f"forbidden key {k!r} leaked into features of scenario {scenario.scenario_id}"
    feature_rows.append(feats)
    return feats
