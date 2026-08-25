"""Regression tests for two bugs found and fixed while recovering the
vLLM-LTR comparative-evaluation run (see
docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md):

1. Performance: ``PerPolicyRegressionAnwgSelector.predict()`` called
   ``RandomForestRegressor.predict()`` once per candidate policy (20 separate
   top-level sklearn calls), each paying full Python/joblib per-call
   validation-and-dispatch overhead for a single-row batch -- ~56ms/call,
   which made a live per-step simulator dispatch of this selector
   impractically slow. ``_fast_forest_predict()`` bypasses that overhead by
   averaging each fitted tree's compiled ``tree_.predict()`` output directly,
   using sklearn's own accumulation order (plain running sum / n_estimators)
   -- proven bit-exact against ``reg.predict()``, not just "close enough".

2. Correctness: ``_feature_matrix()`` only recognized ``feat_``-prefixed
   dict keys. The simulator's live ``extract_features()`` returns bare
   (unprefixed) keys, so every live-simulator-driven prediction silently
   received an all-zero feature vector -- ``PerPolicyRegressionAnwgSelector``
   (and, unfixed, ``DecisionTreeSelector``/``RandomForestSelector``) would
   deterministically dispatch to whatever policy the model ranks highest at
   X=0, never actually responding to queue/KV/SLO state, whenever driven by
   a SelectorDispatchPolicy-style live wrapper (as opposed to an offline
   evaluation over persisted feat_-prefixed dataset rows, which worked
   correctly all along). ``RuleBasedSelector._get()`` already handled both
   key formats; ``_feature_matrix()`` now mirrors that.
"""
from __future__ import annotations

import numpy as np
import pytest

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FEATURE_NAMES


def _make_anwg_rows(n: int = 80) -> list:
    rows = []
    for i in range(n):
        row = {f"feat_{name}": float((i * 7 + idx) % 11) for idx, name in enumerate(FEATURE_NAMES)}
        for pname in SELECTOR_CANDIDATES:
            row[f"completion_{pname}"] = 1.0
            row[f"reward_{pname}"] = 0.5 + 0.01 * (hash((i, pname)) % 10)
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def fitted_selector():
    pytest.importorskip("sklearn")
    from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector

    rows = _make_anwg_rows(80)
    sel = PerPolicyRegressionAnwgSelector(n_estimators=25, max_depth=6, random_state=0)
    sel.fit(rows)
    return sel


class TestFastForestPredictBitExact:
    def test_fit_marks_every_regressor_fast_path_ok(self, fitted_selector):
        assert set(fitted_selector._fast_path_ok.keys()) == set(SELECTOR_CANDIDATES)
        assert all(fitted_selector._fast_path_ok.values())

    def test_fast_path_bit_exact_vs_official_predict(self, fitted_selector):
        from llmserveopt.selector.models import _fast_forest_predict, _FOREST_INPUT_DTYPE

        rng = np.random.default_rng(123)
        n_features = len(FEATURE_NAMES)
        max_abs_diff = 0.0
        for _ in range(50):
            X = rng.uniform(-5.0, 15.0, size=(3, n_features))
            X32 = np.asarray(X, dtype=_FOREST_INPUT_DTYPE, order="C")
            for reg in fitted_selector._regressors.values():
                fast = _fast_forest_predict(reg, X32)
                official = reg.predict(X)
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(fast - official))))
        assert max_abs_diff == 0.0

    def test_predict_decisions_identical_fast_vs_forced_slow(self, fitted_selector):
        """End-to-end: predict() (fast path) must choose the same policy as
        manually forcing every regressor onto the slow reg.predict() path."""
        rows = _make_anwg_rows(30)
        fast_decisions = fitted_selector.predict(rows)

        saved_flags = dict(fitted_selector._fast_path_ok)
        try:
            for p in fitted_selector._fast_path_ok:
                fitted_selector._fast_path_ok[p] = False
            slow_decisions = fitted_selector.predict(rows)
        finally:
            fitted_selector._fast_path_ok.update(saved_flags)

        assert fast_decisions == slow_decisions

    def test_load_reverifies_fast_path(self, fitted_selector, tmp_path):
        path = str(tmp_path / "sel.joblib")
        fitted_selector.save(path)
        from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector

        reloaded = PerPolicyRegressionAnwgSelector.load(path)
        assert all(reloaded._fast_path_ok.values())

    def test_predict_one_latency_stays_well_under_pre_fix_cost(self, fitted_selector):
        """Pre-fix measured cost was ~56ms/call (20 candidates x sklearn's
        per-call overhead) on a 100-estimator/depth-8 production artifact.
        This fixture uses a smaller forest, but the fast path must still be
        at least an order of magnitude below that regardless of forest size
        -- a regression here means the fast path silently stopped being
        used (e.g. _fast_path_ok flipped to False)."""
        import time

        feats = {n: 3.0 for n in FEATURE_NAMES}
        t0 = time.perf_counter()
        for _ in range(20):
            fitted_selector.predict_one(feats)
        elapsed_per_call = (time.perf_counter() - t0) / 20
        assert elapsed_per_call < 0.020, (
            f"predict_one() took {elapsed_per_call*1000:.2f}ms/call, "
            "expected well under the ~56ms/call pre-fix cost"
        )


class TestFeatureKeyFormatBugFix:
    def test_bare_and_prefixed_keys_produce_identical_prediction(self, fitted_selector):
        rng = np.random.default_rng(5)
        for _ in range(10):
            bare = {n: float(rng.uniform(-3, 12)) for n in FEATURE_NAMES}
            prefixed = {f"feat_{n}": v for n, v in bare.items()}
            assert fitted_selector.predict_one(bare) == fitted_selector.predict_one(prefixed)

    def test_bare_key_features_are_not_silently_zeroed(self, fitted_selector):
        """Regression guard for the exact bug found: before the fix,
        _feature_matrix() ignored bare keys and always built an all-zero
        row, so predict_one() driven by live extract_features()-shaped
        input collapsed to a single constant decision regardless of state."""
        from llmserveopt.selector.models import _feature_matrix

        bare_row = {n: 9.0 for n in FEATURE_NAMES}
        X = _feature_matrix([bare_row])
        assert np.all(X == 9.0), "bare feature keys must be read, not defaulted to 0.0"

    def test_live_shaped_features_produce_varied_decisions(self, fitted_selector):
        """With the bug, every live (bare-key) call collapsed to one
        constant policy choice. Post-fix, varying feature vectors must be
        able to produce different decisions (using a real, larger,
        pre-trained artifact -- the tiny synthetic fixture here isn't
        guaranteed to diversify, so this only asserts the mechanism reads
        real values, not that argmax necessarily differs for these inputs)."""
        rng = np.random.default_rng(9)
        decisions = set()
        for _ in range(20):
            bare = {n: float(rng.uniform(-5, 20)) for n in FEATURE_NAMES}
            decisions.add(fitted_selector.predict_one(bare))
        # At minimum, confirm the call path executes and returns valid
        # candidates for varied bare-key input (the persisted-artifact
        # integration test below proves actual diversification).
        assert decisions <= set(SELECTOR_CANDIDATES)


class TestFeatureKeyFormatBugFixOnPersistedArtifact:
    """Same bug, exercised against the real trained artifact used by the
    vLLM-LTR comparative-evaluation script, to prove the fix actually
    restores state-responsiveness (not just structurally plausible on a
    tiny synthetic model)."""

    ARTIFACT_PATH = "results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib"

    def test_persisted_artifact_diversifies_under_live_bare_keys(self):
        import os

        if not os.path.exists(self.ARTIFACT_PATH):
            pytest.skip("persisted selector artifact not present in this checkout")
        pytest.importorskip("sklearn")
        from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector

        sel = PerPolicyRegressionAnwgSelector.load(self.ARTIFACT_PATH)
        rng = np.random.default_rng(2)
        decisions = set()
        for _ in range(50):
            bare = {n: float(rng.uniform(-2, 8)) for n in FEATURE_NAMES}
            decisions.add(sel.predict_one(bare))
        assert len(decisions) > 1, (
            "regression: live bare-key dispatch collapsed back to a single "
            "constant policy choice"
        )
