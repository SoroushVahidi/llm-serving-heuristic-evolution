"""Tests for the feature-based RuleBasedSelector (Phase 2B.5)."""
import pytest

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.models import RuleBasedSelector


def feats(**kwargs) -> dict:
    """Build a feature dict with safe defaults, overriding with kwargs."""
    defaults = {
        "fraction_tight_slo": 0.0,
        "min_slack": 50.0,
        "recent_slo_violation_rate": 0.0,
        "kv_utilization": 0.2,
        "mean_prompt_tokens": 128.0,
        "p95_prompt_tokens": 256.0,
        "mean_pred_output_tokens": 100.0,
        "pred_output_cv": 0.8,
        "burstiness_cv": 0.3,
    }
    defaults.update(kwargs)
    return defaults


# -------------------------------------------------------------------------
# Rule 1: tight-SLO / urgency → least_laxity_first
# -------------------------------------------------------------------------

def test_tight_slo_fraction_triggers_llf():
    sel = RuleBasedSelector()
    f = feats(fraction_tight_slo=0.6)
    assert sel.predict_one(f) == "least_laxity_first"


def test_low_min_slack_triggers_llf():
    sel = RuleBasedSelector()
    f = feats(fraction_tight_slo=0.0, min_slack=0.5)
    assert sel.predict_one(f) == "least_laxity_first"


# -------------------------------------------------------------------------
# Rule 2: high recent violations → admission_control
# -------------------------------------------------------------------------

def test_high_violation_rate_triggers_admission_control():
    sel = RuleBasedSelector()
    f = feats(fraction_tight_slo=0.0, min_slack=50.0, recent_slo_violation_rate=0.5)
    assert sel.predict_one(f) == "admission_control"


def test_moderate_violation_below_threshold_no_admission_control():
    sel = RuleBasedSelector()
    f = feats(recent_slo_violation_rate=0.2)
    # Below threshold — should not trigger admission_control
    result = sel.predict_one(f)
    assert result != "admission_control"


# -------------------------------------------------------------------------
# Rule 3: high KV utilization → vllm_style_token_budget
# -------------------------------------------------------------------------

def test_high_kv_utilization_triggers_vllm():
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.85,
    )
    assert sel.predict_one(f) == "vllm_style_token_budget"


# -------------------------------------------------------------------------
# Rule 4: prefill-heavy → sarathi_style
# -------------------------------------------------------------------------

def test_large_mean_prompt_triggers_sarathi():
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.2,
        mean_prompt_tokens=1024.0,
    )
    assert sel.predict_one(f) == "sarathi_style"


def test_large_p95_prompt_triggers_sarathi():
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.2,
        mean_prompt_tokens=200.0, p95_prompt_tokens=1500.0,
    )
    assert sel.predict_one(f) == "sarathi_style"


# -------------------------------------------------------------------------
# Rule 5: short uniform outputs → estimated_service_time_first
# -------------------------------------------------------------------------

def test_short_uniform_outputs_triggers_estf():
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.2,
        mean_prompt_tokens=128.0, p95_prompt_tokens=256.0,
        mean_pred_output_tokens=32.0, pred_output_cv=0.3,
    )
    assert sel.predict_one(f) == "estimated_service_time_first"


def test_short_but_high_cv_does_not_trigger_estf():
    """Short mean output but high variance → not ESTF regime."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.2,
        mean_prompt_tokens=128.0, p95_prompt_tokens=256.0,
        mean_pred_output_tokens=32.0, pred_output_cv=1.5,  # high variance
    )
    result = sel.predict_one(f)
    assert result != "estimated_service_time_first"


# -------------------------------------------------------------------------
# Rule 6: bursty arrivals → slo_slack_score
# -------------------------------------------------------------------------

def test_bursty_arrivals_triggers_slo_slack():
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.2,
        mean_prompt_tokens=128.0, p95_prompt_tokens=256.0,
        mean_pred_output_tokens=100.0, pred_output_cv=0.8,
        burstiness_cv=2.0,
    )
    assert sel.predict_one(f) == "slo_slack_score"


# -------------------------------------------------------------------------
# Rule 7: safe default → edf
# -------------------------------------------------------------------------

def test_moderate_workload_returns_edf():
    """No special conditions → fall through to safe default edf."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.1, min_slack=20.0,
        recent_slo_violation_rate=0.05, kv_utilization=0.3,
        mean_prompt_tokens=100.0, p95_prompt_tokens=200.0,
        mean_pred_output_tokens=80.0, pred_output_cv=0.9,
        burstiness_cv=0.5,
    )
    assert sel.predict_one(f) == "edf"


# -------------------------------------------------------------------------
# All predictions are valid SELECTOR_CANDIDATES
# -------------------------------------------------------------------------

def test_all_predictions_are_valid_candidates():
    sel = RuleBasedSelector()
    test_cases = [
        feats(fraction_tight_slo=0.6),
        feats(recent_slo_violation_rate=0.5),
        feats(kv_utilization=0.9),
        feats(mean_prompt_tokens=2000.0),
        feats(mean_pred_output_tokens=20.0, pred_output_cv=0.2),
        feats(burstiness_cv=3.0),
        feats(),
    ]
    for f in test_cases:
        pred = sel.predict_one(f)
        assert pred in SELECTOR_CANDIDATES, f"Invalid: {pred}"


# -------------------------------------------------------------------------
# Determinism
# -------------------------------------------------------------------------

def test_deterministic_same_features():
    """Same features always produce same prediction."""
    sel = RuleBasedSelector()
    f = feats(fraction_tight_slo=0.5)
    assert sel.predict_one(f) == sel.predict_one(f)


def test_predict_list_matches_predict_one():
    """predict(list) == [predict_one(f) for f in list]."""
    sel = RuleBasedSelector()
    fs = [
        feats(fraction_tight_slo=0.6),
        feats(kv_utilization=0.9),
        feats(),
    ]
    assert sel.predict(fs) == [sel.predict_one(f) for f in fs]


# -------------------------------------------------------------------------
# Supports feat_* prefixed keys (dataset row format)
# -------------------------------------------------------------------------

def test_feat_prefix_keys_work():
    """predict_one accepts 'feat_*' prefixed keys (dataset row format)."""
    sel = RuleBasedSelector()
    row = {"feat_fraction_tight_slo": 0.6, "feat_min_slack": 50.0}
    result = sel.predict_one(row)
    assert result == "least_laxity_first"


def test_bare_and_prefixed_keys_equivalent():
    """Bare and feat_*-prefixed keys produce same result."""
    sel = RuleBasedSelector()
    bare = feats(kv_utilization=0.85,
                 fraction_tight_slo=0.0, min_slack=50.0,
                 recent_slo_violation_rate=0.0)
    prefixed = {f"feat_{k}": v for k, v in bare.items()}
    assert sel.predict_one(bare) == sel.predict_one(prefixed)


# -------------------------------------------------------------------------
# Does not choose FIFO (meaningful improvement over placeholder)
# -------------------------------------------------------------------------

def test_never_chooses_fifo():
    """The new rule-based selector should not dispatch to fifo in any built-in rule."""
    sel = RuleBasedSelector()
    test_cases = [
        feats(fraction_tight_slo=0.6),
        feats(recent_slo_violation_rate=0.5),
        feats(kv_utilization=0.9),
        feats(mean_prompt_tokens=2000.0),
        feats(mean_pred_output_tokens=20.0, pred_output_cv=0.2),
        feats(burstiness_cv=3.0),
        feats(),
    ]
    for f in test_cases:
        pred = sel.predict_one(f)
        assert pred != "fifo", f"RuleBasedSelector should not return 'fifo'; got it for {f}"
