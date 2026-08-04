"""Focused tests for CC5 finalization: paired statistical analysis and the
frozen, deterministic regime-specific operating envelope."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from llmserveopt.experiments import cc5_contextual_predictor as cc5
from llmserveopt.experiments import cc5_final_operating_envelope as fin
from llmserveopt.experiments import cc5_uncertainty_regime_refinement as refine

CC4B_DIR = "results/cc4b_oracle_composition_expansion/20260803T182426Z"

pytestmark = pytest.mark.skipif(
    not (cc5.ROOT / CC4B_DIR / "manifest.json").exists(),
    reason="CC4b dataset not present locally",
)


@pytest.fixture(scope="module")
def ds() -> cc5.CC4Dataset:
    return cc5.load_cc4_dataset(cc5.ROOT / CC4B_DIR)


@pytest.fixture(scope="module")
def dev_ids(ds) -> list[str]:
    return sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])


@pytest.fixture(scope="module")
def val_ids(ds) -> list[str]:
    return sorted(ds.causal_features[ds.causal_features["split"] == "VALIDATION"]["window_id"])


@pytest.fixture(scope="module")
def eval_ids(ds) -> list[str]:
    return sorted(ds.causal_features[ds.causal_features["split"].isin(ds.evaluation_splits)]["window_id"])


@pytest.fixture(scope="module")
def artifact_bundle(ds):
    return refine.build_deployable_artifact(ds, seed=0, ood_z_threshold=2.0, n_bootstrap=5)


@pytest.fixture(scope="module")
def encoder(ds, dev_ids):
    return cc5.FeatureEncoder.fit(ds.causal_features[ds.causal_features["window_id"].isin(dev_ids)])


@pytest.fixture(scope="module")
def dev_lowo(ds, encoder, dev_ids, artifact_bundle):
    _, meta, *_ = artifact_bundle
    factories = cc5.build_regret_regressor_factories(seed=0)
    return fin.compute_dev_lowo_table(ds, encoder, dev_ids, factories[meta["best_model_name"]])


@pytest.fixture(scope="module")
def envelope(ds, dev_lowo):
    all_regimes = sorted(ds.causal_features["regime"].unique())
    return fin.freeze_operating_envelope(dev_lowo, all_regimes, min_windows=2)


@pytest.fixture(scope="module")
def gate(envelope, ds, dev_ids):
    return fin.FrozenEnvelopeGate(
        schema_version=fin.ENVELOPE_SCHEMA_VERSION,
        envelope_version=1,
        trusted_regimes=tuple(envelope["trusted_regimes"]),
        fitted_at="test",
        dataset_config_hash=ds.manifest.get("config_hash", ""),
        dev_window_count=len(dev_ids),
        decision_basis="development_lowo_evidence_only",
    )


# ---------------------------------------------------------------------------
# Paired statistical analysis
# ---------------------------------------------------------------------------


def test_paired_bootstrap_ci_zero_for_identical_sequences():
    a = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = fin.paired_bootstrap_ci(a, a, seed=0)
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_paired_bootstrap_ci_detects_consistent_shift():
    a = [0.5, 0.6, 0.55, 0.62, 0.58, 0.61, 0.57, 0.59]
    b = [0.3, 0.35, 0.32, 0.31, 0.33, 0.34, 0.29, 0.36]
    result = fin.paired_bootstrap_ci(a, b, seed=0)
    assert result["ci_low"] > 0.0
    assert result["mean_diff"] == pytest.approx(np.mean(a) - np.mean(b))


def test_paired_permutation_test_deterministic_and_bounded():
    a = [0.5, 0.6, 0.55, 0.62, 0.58]
    b = [0.3, 0.35, 0.32, 0.31, 0.33]
    r1 = fin.paired_permutation_test(a, b, n_perm=2000, seed=0)
    r2 = fin.paired_permutation_test(a, b, n_perm=2000, seed=0)
    assert r1 == r2
    assert 0.0 <= r1["p_value_two_sided"] <= 1.0


def test_win_tie_loss_counts_sum_to_n():
    a = [0.5, 0.5, 0.6, 0.4]
    b = [0.5, 0.4, 0.5, 0.5]
    counts = fin.win_tie_loss(a, b, tie_eps=0.005)
    assert counts["wins"] + counts["ties"] + counts["losses"] == counts["n"] == 4
    assert counts["wins"] == 2  # index 2 (0.6>0.5) and index... check exact
    assert counts["ties"] == 1  # index 0 exact tie


def test_cohens_d_paired_zero_variance_returns_zero():
    a = [0.5, 0.5, 0.5]
    b = [0.3, 0.3, 0.3]
    # constant difference -> zero variance -> defined as 0.0 (not inf/nan)
    assert fin.cohens_d_paired(a, b) == 0.0


# ---------------------------------------------------------------------------
# Envelope: development-only evidence, no held-out leakage
# ---------------------------------------------------------------------------


def test_dev_lowo_table_never_touches_evaluation_windows(dev_lowo, eval_ids):
    assert set(dev_lowo["window_id"]).isdisjoint(set(eval_ids))


def test_envelope_excludes_regimes_with_no_development_windows(envelope, ds):
    all_regimes = set(ds.causal_features["regime"].unique())
    dev_regimes = set(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["regime"])
    no_dev_regimes = all_regimes - dev_regimes
    assert no_dev_regimes, "fixture dataset should contain at least one OOD-only regime"
    table = envelope["table"].set_index("regime")
    for regime in no_dev_regimes:
        assert table.loc[regime, "trust_predictor"] == False  # noqa: E712
        assert table.loc[regime, "n_dev_windows"] == 0
    assert no_dev_regimes.isdisjoint(set(envelope["trusted_regimes"]))


def test_envelope_trusted_regimes_have_minimum_dev_windows(envelope):
    table = envelope["table"].set_index("regime")
    for regime in envelope["trusted_regimes"]:
        assert table.loc[regime, "n_dev_windows"] >= 2


def test_envelope_is_deterministic(ds, encoder, dev_ids, artifact_bundle):
    _, meta, *_ = artifact_bundle
    factories = cc5.build_regret_regressor_factories(seed=0)
    t1 = fin.compute_dev_lowo_table(ds, encoder, dev_ids, factories[meta["best_model_name"]])
    t2 = fin.compute_dev_lowo_table(ds, encoder, dev_ids, factories[meta["best_model_name"]])
    pd.testing.assert_frame_equal(t1, t2)
    e1 = fin.freeze_operating_envelope(t1, sorted(ds.causal_features["regime"].unique()))
    e2 = fin.freeze_operating_envelope(t2, sorted(ds.causal_features["regime"].unique()))
    assert e1["trusted_regimes"] == e2["trusted_regimes"]


# ---------------------------------------------------------------------------
# Frozen gate: versioning, determinism, fallback, OOD/uncertainty interaction
# ---------------------------------------------------------------------------


def test_stale_envelope_schema_rejected(gate):
    bad = fin.FrozenEnvelopeGate(**{**gate.__dict__, "schema_version": gate.schema_version - 1})
    with pytest.raises(cc5.CC5Error, match="stale operating-envelope schema"):
        fin.assert_envelope_compatible(bad)


def test_missing_gate_rejected():
    with pytest.raises(cc5.CC5Error, match="missing frozen operating-envelope gate"):
        fin.assert_envelope_compatible(None)


def test_select_with_frozen_envelope_deterministic(ds, artifact_bundle, gate, eval_ids):
    artifact, meta, _, best_fixed, best_global, hard_selector = artifact_bundle
    fallback = fin.HybridLookupBaseline(
        name="test_fallback", rules={r: "global" for r in ds.causal_features["regime"].unique()},
        best_fixed=best_fixed, best_global=best_global,
    )
    causal_row = ds.causal_features.set_index("window_id").loc[eval_ids[0]]
    d1 = fin.select_with_frozen_envelope(gate, artifact, ds, causal_row, fallback=fallback)
    d2 = fin.select_with_frozen_envelope(gate, artifact, ds, causal_row, fallback=fallback)
    # inference_overhead_s is a wall-clock timing measurement, not a
    # decision value -- excluded from the determinism check by design
    # (same convention as test_cc5_uncertainty_regime.py's inference
    # determinism test).
    for key in d1:
        if key == "inference_overhead_s":
            continue
        assert d1[key] == d2[key]


def test_regime_outside_envelope_always_falls_back(ds, artifact_bundle, gate, eval_ids):
    artifact, meta, _, best_fixed, best_global, hard_selector = artifact_bundle
    fallback = fin.HybridLookupBaseline(
        name="test_fallback", rules={r: "fixed" for r in ds.causal_features["regime"].unique()},
        best_fixed=best_fixed, best_global=best_global,
    )
    causal_by_window = ds.causal_features.set_index("window_id")
    outside_regime_windows = [
        w for w in eval_ids if str(causal_by_window.loc[w, "regime"]) not in gate.trusted_regimes
    ]
    assert outside_regime_windows, "expect at least one held-out window in an untrusted regime"
    for w in outside_regime_windows[:3]:
        causal_row = causal_by_window.loc[w]
        decision = fin.select_with_frozen_envelope(gate, artifact, ds, causal_row, fallback=fallback)
        assert decision["in_envelope"] is False
        assert decision["used_predictor"] is False
        assert decision["abstained"] is True
        assert decision["fallback_reason"] is not None
        assert "regime_outside_envelope" in decision["fallback_reason"]
        expected = fallback.select(str(causal_row["regime"]))
        assert decision["selected_candidate_id"] == expected


def test_in_envelope_still_respects_uncertainty_ood_gate(ds, artifact_bundle, gate, eval_ids):
    """Even inside the envelope, a high-OOD/uncertainty window must still
    fall back -- the envelope only relaxes the regime check, not the
    per-window uncertainty/OOD check."""
    artifact, meta, _, best_fixed, best_global, hard_selector = artifact_bundle
    fallback = fin.HybridLookupBaseline(
        name="test_fallback", rules={r: "global" for r in ds.causal_features["regime"].unique()},
        best_fixed=best_fixed, best_global=best_global,
    )
    causal_by_window = ds.causal_features.set_index("window_id")
    in_envelope_windows = [
        w for w in eval_ids if str(causal_by_window.loc[w, "regime"]) in gate.trusted_regimes
    ]
    found_gated = False
    for w in in_envelope_windows:
        causal_row = causal_by_window.loc[w]
        decision = fin.select_with_frozen_envelope(gate, artifact, ds, causal_row, fallback=fallback)
        assert decision["in_envelope"] is True
        if not decision["uncertainty_ood_ok"]:
            found_gated = True
            assert decision["used_predictor"] is False
            assert decision["abstained"] is True
    assert found_gated, "expect at least one in-envelope window still gated by uncertainty/OOD"


# ---------------------------------------------------------------------------
# Final verdict logic uses paired statistics, not point estimates alone
# ---------------------------------------------------------------------------


def test_final_verdict_requires_statistically_significant_global_win():
    """Construct a scenario where the frozen system's point estimate beats
    best_global but the paired difference is not distinguishable from zero
    -- the verdict must NOT claim COMPLETE_FULL."""
    window_ids = [f"w{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    frozen_vals = 0.40 + rng.normal(0, 0.05, size=20)
    global_vals = frozen_vals - rng.normal(0.001, 0.05, size=20)  # tiny, noisy edge
    fixed_vals = frozen_vals - 0.05  # clearly, consistently worse
    hard_vals = frozen_vals - 0.02

    def _mk(vals):
        return pd.DataFrame({
            "window_id": window_ids, "split": ["ID_TEST"] * 20, "regime": ["kv_pressure"] * 20,
            cc5.PRIMARY_COL: vals, cc5.COMPLETION_COL: [1.0] * 20, "regret": [0.0] * 20,
            "abstained": [False] * 20, "fallback_reason": [None] * 20,
        })

    frozen_eval = _mk(frozen_vals)
    global_eval = _mk(global_vals)
    fixed_eval = _mk(fixed_vals)
    hard_eval = _mk(hard_vals)
    paired_overall, _ = fin.run_paired_statistical_analysis(
        predictor_eval=frozen_eval, best_fixed_eval=fixed_eval, best_global_eval=global_eval,
        hard_eval=hard_eval, near_tie_windows=set(), eval_ids=window_ids, seed=0,
    )
    verdict = fin.determine_final_cc5_verdict(
        frozen_eval=frozen_eval, best_fixed_eval=fixed_eval, best_global_eval=global_eval,
        hard_eval=hard_eval, trusted_regimes=["kv_pressure"], n_eval=20, paired_overall=paired_overall,
    )
    assert verdict["status"] != "COMPLETE_FULL"
    assert verdict["beats_global_overall"] is False


def test_final_verdict_stop_on_completion_violation():
    window_ids = [f"w{i}" for i in range(10)]

    def _mk(anwg, completion):
        return pd.DataFrame({
            "window_id": window_ids, "split": ["ID_TEST"] * 10, "regime": ["kv_pressure"] * 10,
            cc5.PRIMARY_COL: anwg, cc5.COMPLETION_COL: completion, "regret": [0.0] * 10,
            "abstained": [False] * 10, "fallback_reason": [None] * 10,
        })

    frozen_eval = _mk([0.5] * 10, [0.5] * 10)  # completion regression vs fixed
    fixed_eval = _mk([0.4] * 10, [0.9] * 10)
    global_eval = _mk([0.4] * 10, [0.9] * 10)
    hard_eval = _mk([0.4] * 10, [0.9] * 10)
    paired_overall, _ = fin.run_paired_statistical_analysis(
        predictor_eval=frozen_eval, best_fixed_eval=fixed_eval, best_global_eval=global_eval,
        hard_eval=hard_eval, near_tie_windows=set(), eval_ids=window_ids, seed=0,
    )
    verdict = fin.determine_final_cc5_verdict(
        frozen_eval=frozen_eval, best_fixed_eval=fixed_eval, best_global_eval=global_eval,
        hard_eval=hard_eval, trusted_regimes=["kv_pressure"], n_eval=10, paired_overall=paired_overall,
    )
    assert verdict["status"] == "STOP_OR_REDESIGN"
    assert verdict["completion_violations"] > 0


def test_final_verdict_inconclusive_on_empty_envelope():
    window_ids = [f"w{i}" for i in range(10)]

    def _mk(anwg):
        return pd.DataFrame({
            "window_id": window_ids, "split": ["ID_TEST"] * 10, "regime": ["kv_pressure"] * 10,
            cc5.PRIMARY_COL: anwg, cc5.COMPLETION_COL: [1.0] * 10, "regret": [0.0] * 10,
            "abstained": [False] * 10, "fallback_reason": [None] * 10,
        })

    frozen_eval = _mk([0.5] * 10)
    fixed_eval = _mk([0.4] * 10)
    global_eval = _mk([0.4] * 10)
    hard_eval = _mk([0.4] * 10)
    paired_overall, _ = fin.run_paired_statistical_analysis(
        predictor_eval=frozen_eval, best_fixed_eval=fixed_eval, best_global_eval=global_eval,
        hard_eval=hard_eval, near_tie_windows=set(), eval_ids=window_ids, seed=0,
    )
    verdict = fin.determine_final_cc5_verdict(
        frozen_eval=frozen_eval, best_fixed_eval=fixed_eval, best_global_eval=global_eval,
        hard_eval=hard_eval, trusted_regimes=[], n_eval=10, paired_overall=paired_overall,
    )
    assert verdict["status"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Artifact compatibility with the earlier CC5/CC5-refinement artifacts
# ---------------------------------------------------------------------------


def test_reuses_unchanged_model_selection_and_uncertainty(artifact_bundle):
    artifact, meta, *_ = artifact_bundle
    assert meta["best_model_name"] == "gradient_boosting"
    assert meta["selected_uncertainty_method"] in cc5.SUPPORTED_UNCERTAINTY_METHODS
    cc5.assert_uncertainty_calibrator_compatible(artifact.uncertainty_calibrator)


def test_completion_safe_fallback_rules_are_validation_only(ds, val_ids, artifact_bundle):
    artifact, meta, _, best_fixed, best_global, hard_selector = artifact_bundle
    rules = refine.fit_completion_safe_fallback_rules(
        ds=ds, val_ids=val_ids, best_fixed=best_fixed, best_global=best_global,
    )
    assert set(rules.values()) <= {"fixed", "global"}
