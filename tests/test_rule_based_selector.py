"""Tests for the feature-based RuleBasedSelector (Phase 2B.5 + Phase 2B.8 + Phase 2B.11)."""

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
# Rule 4 (Phase 2B.8): tight-SLO / urgency → slo_slack_score (not LLF)
# Phase 2B.7 found LLF catastrophic under overload; slo_slack_score is the
# composite urgency+throughput score that generalises better.
# -------------------------------------------------------------------------

def test_tight_slo_fraction_triggers_slo_slack():
    sel = RuleBasedSelector()
    f = feats(fraction_tight_slo=0.6)
    assert sel.predict_one(f) == "slo_slack_score"


def test_low_min_slack_triggers_slo_slack():
    sel = RuleBasedSelector()
    f = feats(fraction_tight_slo=0.0, min_slack=0.5)
    assert sel.predict_one(f) == "slo_slack_score"


def test_tight_slo_does_not_trigger_llf():
    """Phase 2B.8: tight SLO must NOT choose least_laxity_first (catastrophic under overload)."""
    sel = RuleBasedSelector()
    for fts, ms in [(0.6, 50.0), (0.0, 0.5), (0.5, 0.4)]:
        f = feats(fraction_tight_slo=fts, min_slack=ms)
        assert sel.predict_one(f) != "least_laxity_first", (
            f"tight SLO (fts={fts}, min_slack={ms}) must not choose LLF; "
            f"got {sel.predict_one(f)}"
        )


# -------------------------------------------------------------------------
# Rule 2 (Phase 2B.8): high prediction noise → admission_control
# When pred_output_cv > 1.0, laxity estimates are unreliable; AC is more robust.
# evidence: high_prediction_noise fail_002, LLF WG=0.584 vs AC WG=0.988
# -------------------------------------------------------------------------

def test_high_pred_output_cv_triggers_admission_control():
    """Phase 2B.8: pred_output_cv > 1.0 routes to admission_control."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.2,
        mean_pred_output_tokens=96.0, pred_output_cv=1.5,
    )
    assert sel.predict_one(f) == "admission_control"


def test_high_noise_with_tight_slo_triggers_admission_control():
    """Phase 2B.8: high noise + tight SLO → admission_control (not LLF/slo_slack).

    Under high prediction noise, tight SLO cannot be served reliably with LLF
    (service estimates are wrong). admission_control with urgency sort is empirically
    better (fail_002: AC WG=0.988 vs LLF WG=0.584).
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.5, min_slack=0.5,
        kv_utilization=0.0, pred_output_cv=1.5,
    )
    result = sel.predict_one(f)
    assert result == "admission_control", (
        f"High noise + tight SLO should choose admission_control, not {result}"
    )
    assert result != "least_laxity_first"


def test_low_noise_does_not_trigger_admission_control_via_noise_rule():
    """Low pred_output_cv should not trigger the noise→admission_control rule."""
    sel = RuleBasedSelector()
    # Low CV, no tight SLO, no violations — should reach default
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.2,
        mean_pred_output_tokens=96.0, pred_output_cv=0.8,
    )
    result = sel.predict_one(f)
    # Should NOT be admission_control (noise rule didn't fire, violation rule didn't fire)
    assert result != "admission_control"


# -------------------------------------------------------------------------
# Rule 3: high recent violations → scorpio_style_slo_guard (Phase 2B.11)
# Changed from admission_control: SCORPIO's admission budget + TTFT guard is
# strictly more expressive for violation-prone regimes.
# -------------------------------------------------------------------------

def test_high_violation_rate_triggers_scorpio():
    """Phase 2B.11: high recent violations (standalone) route to scorpio_style_slo_guard."""
    sel = RuleBasedSelector()
    f = feats(fraction_tight_slo=0.0, min_slack=50.0, recent_slo_violation_rate=0.5)
    assert sel.predict_one(f) == "scorpio_style_slo_guard"


def test_moderate_violation_below_threshold_no_admission_control():
    sel = RuleBasedSelector()
    f = feats(recent_slo_violation_rate=0.2)
    # Below threshold — should not trigger admission_control
    result = sel.predict_one(f)
    assert result != "admission_control"


# -------------------------------------------------------------------------
# Rule 1 (Phase 2B.8): decode-heavy / KV-pressure proxy
# → weighted_shortest_processing (elevated from old Rule 3; now fires first)
# -------------------------------------------------------------------------

def test_high_kv_utilization_triggers_wsp():
    """Phase 2B.8: high kv_utilization → weighted_shortest_processing (not vllm_style)."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.85,
    )
    assert sel.predict_one(f) == "weighted_shortest_processing"


def test_large_mean_output_triggers_wsp():
    """Phase 2B.8: large mean predicted output → weighted_shortest_processing (KV proxy)."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.0,
        mean_pred_output_tokens=384.0,
    )
    assert sel.predict_one(f) == "weighted_shortest_processing"


def test_high_kv_with_tight_slo_still_triggers_wsp():
    """Phase 2B.8: KV pressure + tight SLO must pick WSP, not slo_slack_score/LLF.

    Under KV saturation, urgency-based scheduling (LLF, slo_slack) is catastrophic
    because it promotes large-output requests that fill KV longest.
    fail_003: LLF WG=0.101, WSP WG=0.477 under kv_pressure_decode_heavy.
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.5, min_slack=0.8,
        kv_utilization=0.85,
    )
    result = sel.predict_one(f)
    assert result == "weighted_shortest_processing", (
        f"KV pressure + tight SLO must choose WSP, not {result}"
    )
    assert result != "least_laxity_first"
    assert result != "slo_slack_score"


def test_large_output_with_tight_slo_still_triggers_wsp():
    """Phase 2B.8: decode-heavy proxy + tight SLO still picks WSP (KV guard first)."""
    sel = RuleBasedSelector()
    f = feats(
        mean_pred_output_tokens=384.0,
        fraction_tight_slo=0.5, min_slack=0.8,
        kv_utilization=0.0,
    )
    assert sel.predict_one(f) == "weighted_shortest_processing"


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
    """predict_one accepts 'feat_*' prefixed keys (dataset row format).

    Phase 2B.8: tight SLO now routes to slo_slack_score, not LLF.
    """
    sel = RuleBasedSelector()
    row = {"feat_fraction_tight_slo": 0.6, "feat_min_slack": 50.0}
    result = sel.predict_one(row)
    assert result == "slo_slack_score"


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


# -------------------------------------------------------------------------
# Phase 2B.8: oracle exclusion and non-candidate guard
# -------------------------------------------------------------------------

def test_never_returns_oracle_srtf():
    """RuleBasedSelector must never return oracle_srtf (non-deployable oracle)."""
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    assert "oracle_srtf" not in SELECTOR_CANDIDATES, "oracle_srtf must not be in candidates"
    sel = RuleBasedSelector()
    all_cases = [
        feats(fraction_tight_slo=0.6),
        feats(min_slack=0.5),
        feats(kv_utilization=0.9),
        feats(mean_pred_output_tokens=384.0),
        feats(pred_output_cv=1.5),
        feats(recent_slo_violation_rate=0.5),
        feats(mean_prompt_tokens=2000.0),
        feats(mean_pred_output_tokens=20.0, pred_output_cv=0.2),
        feats(burstiness_cv=3.0),
        feats(),
    ]
    for f in all_cases:
        assert sel.predict_one(f) != "oracle_srtf"


def test_all_policy_choices_are_candidates():
    """Every policy in _POLICY_CHOICES must be in SELECTOR_CANDIDATES."""
    sel = RuleBasedSelector()
    for p in sel._POLICY_CHOICES:
        assert p in SELECTOR_CANDIDATES, f"_POLICY_CHOICES contains non-candidate: {p}"


# -------------------------------------------------------------------------
# Phase 2B.8: failure-case scenario regression tests
# Reproduce the exact feature signatures that caused fail_001–fail_003.
# -------------------------------------------------------------------------

def test_fail001_overloaded_mixed_slo_regression():
    """Regression: overloaded_mixed_slo features must NOT choose least_laxity_first.

    fail_001: fraction_tight_slo=0.5, min_slack=0.4 → old Rule 1 fired LLF (WG=0.474).
    Phase 2B.8 fix: tight SLO without KV pressure → slo_slack_score (WG=0.905 winner).
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.5,
        min_slack=0.4,
        kv_utilization=0.0,          # offline: unavailable
        pred_output_cv=0.87,         # overloaded_mixed_slo with 15% noise
        mean_pred_output_tokens=96.0,
    )
    result = sel.predict_one(f)
    assert result != "least_laxity_first", f"fail_001 regression: must not choose LLF, got {result}"
    assert result == "slo_slack_score", f"fail_001 regression: expected slo_slack_score, got {result}"


def test_fail002_high_prediction_noise_regression():
    """Regression: high_prediction_noise features must NOT choose least_laxity_first.

    fail_002: min_slack < 1.0 + pred_output_cv >> 1.0 → old Rule 1 fired LLF (WG=0.584).
    Phase 2B.8 fix: high pred_output_cv → admission_control (WG=0.988 winner).
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.4,
        min_slack=0.5,
        kv_utilization=0.0,
        pred_output_cv=1.5,          # high_prediction_noise: output_sigma=1.0 + 70% noise
        mean_pred_output_tokens=96.0,
    )
    result = sel.predict_one(f)
    assert result != "least_laxity_first", f"fail_002 regression: must not choose LLF, got {result}"
    assert result == "admission_control", f"fail_002 regression: expected admission_control, got {result}"


def test_fail003_kv_pressure_decode_heavy_regression():
    """Regression: kv_pressure_decode_heavy features must NOT choose least_laxity_first.

    fail_003: min_slack=0.8 < 1.0 → old Rule 1 fired LLF (WG=0.101).
    Phase 2B.8 fix: large mean_pred_output (KV proxy) → weighted_shortest_processing (WG=0.477).
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.3,
        min_slack=0.8,
        kv_utilization=0.0,          # offline: unavailable
        pred_output_cv=0.78,         # kv_pressure: output_sigma=0.6 + 25% noise
        mean_pred_output_tokens=384.0,  # decode-heavy: output_mean=384
    )
    result = sel.predict_one(f)
    assert result != "least_laxity_first", f"fail_003 regression: must not choose LLF, got {result}"
    assert result == "weighted_shortest_processing", (
        f"fail_003 regression: expected weighted_shortest_processing, got {result}"
    )


# -------------------------------------------------------------------------
# Phase 2B.8: other rules still work correctly (prefill, burstiness, default)
# -------------------------------------------------------------------------

def test_prefill_heavy_rule_unchanged():
    """Rule 5 (prefill-heavy) still routes to sarathi_style when no higher rule fires."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.2,
        mean_pred_output_tokens=80.0, pred_output_cv=0.8,
        mean_prompt_tokens=1024.0,
    )
    assert sel.predict_one(f) == "sarathi_style"


def test_burstiness_rule_unchanged():
    """Rule 7 (bursty arrivals) still routes to slo_slack_score when no higher rule fires."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0, min_slack=50.0,
        recent_slo_violation_rate=0.0, kv_utilization=0.2,
        mean_pred_output_tokens=100.0, pred_output_cv=0.8,
        burstiness_cv=2.0,
    )
    assert sel.predict_one(f) == "slo_slack_score"


def test_default_rule_unchanged():
    """Rule 8 (default) still routes to edf when no other rule fires."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.1, min_slack=20.0,
        recent_slo_violation_rate=0.05, kv_utilization=0.3,
        mean_pred_output_tokens=80.0, pred_output_cv=0.9,
        burstiness_cv=0.5,
    )
    assert sel.predict_one(f) == "edf"


# -------------------------------------------------------------------------
# Phase 2B.11: SCORPIO routing rules
# -------------------------------------------------------------------------

def test_scorpio_in_policy_choices():
    """scorpio_style_slo_guard must be in RuleBasedSelector._POLICY_CHOICES (Phase 2B.11)."""
    assert "scorpio_style_slo_guard" in RuleBasedSelector._POLICY_CHOICES


def test_overload_tight_slo_with_violations_triggers_scorpio():
    """Phase 2B.11 Rule 0: overloaded tight SLO + recent violations → scorpio_style_slo_guard.

    When tight deadlines AND observed SLO violations coincide, SCORPIO's admission
    budget throttling + TTFT/laxity guard beats slo_slack_score.
    Evidence: dev_overloaded_mixed_slo and heldout_bursty_mixed_slo families.
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.5,
        min_slack=0.4,
        recent_slo_violation_rate=0.35,  # > 0.2 threshold
        kv_utilization=0.0,
        pred_output_cv=0.87,
        mean_pred_output_tokens=96.0,
    )
    assert sel.predict_one(f) == "scorpio_style_slo_guard"


def test_low_min_slack_with_violations_triggers_scorpio():
    """Phase 2B.11 Rule 0: min_slack < 1.0 + recent violations → scorpio_style_slo_guard."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0,
        min_slack=0.5,
        recent_slo_violation_rate=0.25,  # > 0.2 threshold
        kv_utilization=0.0,
        pred_output_cv=0.8,
        mean_pred_output_tokens=96.0,
    )
    assert sel.predict_one(f) == "scorpio_style_slo_guard"


def test_tight_slo_without_violations_does_not_trigger_scorpio_rule0():
    """Phase 2B.11: tight SLO alone (no violations) must NOT fire Rule 0 → SCORPIO.

    Without active violations, the system may be handling tight SLOs fine.
    Rule 4 (slo_slack_score) applies here — keeps fail_001 regression intact.
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.5,
        min_slack=0.4,
        recent_slo_violation_rate=0.0,   # no violations
        kv_utilization=0.0,
        pred_output_cv=0.87,
        mean_pred_output_tokens=96.0,
    )
    # Rule 0 should NOT fire; Rule 4 (tight SLO → slo_slack_score) should
    assert sel.predict_one(f) == "slo_slack_score"


def test_violations_below_threshold_no_scorpio_rule0():
    """Phase 2B.11: violation rate at 0.2 (not > 0.2) does not fire Rule 0."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.5,
        min_slack=0.4,
        recent_slo_violation_rate=0.2,   # at threshold, not above
        kv_utilization=0.0,
        pred_output_cv=0.87,
        mean_pred_output_tokens=96.0,
    )
    assert sel.predict_one(f) == "slo_slack_score"


def test_very_high_pred_output_cv_triggers_scorpio():
    """Phase 2B.11 Rule 2a: pred_output_cv > 2.0 → scorpio_style_slo_guard.

    At extreme prediction noise (CV > 2.0), SCORPIO's composite SLO guard beats
    admission_control.  Evidence: heldout_very_high_noise_s4 (90% noise, fail_004).
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.4,
        min_slack=0.5,
        recent_slo_violation_rate=0.0,
        kv_utilization=0.0,
        pred_output_cv=2.5,              # > 2.0 threshold
        mean_pred_output_tokens=96.0,
    )
    assert sel.predict_one(f) == "scorpio_style_slo_guard"


def test_moderate_high_noise_still_routes_to_admission_control():
    """Phase 2B.11: pred_output_cv in (1.0, 2.0] still routes to admission_control.

    Moderate-high noise (Rule 2b, unchanged from Phase 2B.8).
    dev_high_prediction_noise (70% noise): admission_control WG=0.988.
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0,
        min_slack=50.0,
        recent_slo_violation_rate=0.0,
        kv_utilization=0.0,
        pred_output_cv=1.5,              # > 1.0 but ≤ 2.0
        mean_pred_output_tokens=96.0,
    )
    assert sel.predict_one(f) == "admission_control"


def test_scorpio_rule0_fires_before_kv_pressure_rule1():
    """Phase 2B.11: Rule 0 (SCORPIO overload) has higher priority than Rule 1 (KV → WSP).

    When tight SLO + SLO violations + KV pressure all coincide, Rule 0 fires first
    and returns scorpio_style_slo_guard.  SCORPIO handles KV pressure internally via
    its _guard_active() + long-decode token filter (kv_fill_ratio threshold check).
    This is correct per the Phase 2B.11 rule ordering: overload guard precedes KV proxy.

    Contrast: KV pressure WITHOUT tight SLO or violations → Rule 1 (WSP) still fires.
    """
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.5,
        min_slack=0.4,
        recent_slo_violation_rate=0.4,   # triggers Rule 0
        kv_utilization=0.85,             # would trigger Rule 1 if Rule 0 absent
        pred_output_cv=0.8,
        mean_pred_output_tokens=250.0,
    )
    # Rule 0 fires first: tight SLO + violations → SCORPIO
    assert sel.predict_one(f) == "scorpio_style_slo_guard"


def test_kv_pressure_alone_still_triggers_wsp():
    """Phase 2B.11: KV pressure without tight SLO / violations still routes to WSP (Rule 1)."""
    sel = RuleBasedSelector()
    f = feats(
        fraction_tight_slo=0.0,
        min_slack=50.0,
        recent_slo_violation_rate=0.0,   # no violations → Rule 0 doesn't fire
        kv_utilization=0.85,             # Rule 1 fires
        pred_output_cv=0.8,
        mean_pred_output_tokens=250.0,
    )
    assert sel.predict_one(f) == "weighted_shortest_processing"


def test_fail004_very_high_noise_regression():
    """Regression: heldout_very_high_noise_s4 feature profile → scorpio_style_slo_guard.

    fail_004: heldout_very_high_noise_s4 (90% noise, seed 4):
    Phase 2B.9 — rule selector chose admission_control (WG=0.970), best fixed edf (0.993).
    Phase 2B.10 — SCORPIO-style WG=1.000 (best).
    Phase 2B.11 fix: extreme CV (> 2.0) now routes to scorpio_style_slo_guard.
    """
    sel = RuleBasedSelector()
    # Feature profile for heldout_very_high_noise with 90% prediction noise:
    # output_sigma=1.0 lognormal + 90% multiplicative noise → very high pred_output_cv.
    f = feats(
        fraction_tight_slo=0.4,
        min_slack=0.5,
        recent_slo_violation_rate=0.0,
        kv_utilization=0.0,
        pred_output_cv=2.8,              # representative value for 90% noise + sigma=1.0
        mean_pred_output_tokens=96.0,
    )
    result = sel.predict_one(f)
    assert result == "scorpio_style_slo_guard", (
        f"fail_004 regression: extreme noise (cv=2.8) should route to scorpio_style_slo_guard, "
        f"not {result}"
    )
    assert result != "admission_control"


def test_selector_never_returns_non_candidate():
    """All RuleBasedSelector choices must be in SELECTOR_CANDIDATES (including SCORPIO)."""
    sel = RuleBasedSelector()
    test_cases = [
        feats(fraction_tight_slo=0.6),
        feats(recent_slo_violation_rate=0.5),
        feats(kv_utilization=0.9),
        feats(mean_prompt_tokens=2000.0),
        feats(mean_pred_output_tokens=20.0, pred_output_cv=0.2),
        feats(burstiness_cv=3.0),
        feats(),
        # Phase 2B.11 SCORPIO cases
        feats(fraction_tight_slo=0.5, min_slack=0.4, recent_slo_violation_rate=0.35),
        feats(pred_output_cv=2.5),
        feats(fraction_tight_slo=0.0, min_slack=50.0, recent_slo_violation_rate=0.5),
    ]
    for f in test_cases:
        pred = sel.predict_one(f)
        assert pred in SELECTOR_CANDIDATES, (
            f"selector returned non-candidate policy '{pred}' for features {f}"
        )
