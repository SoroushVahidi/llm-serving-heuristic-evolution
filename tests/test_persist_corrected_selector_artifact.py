"""Tests for scripts/persist_corrected_selector_artifact.py.

Covers the fix for an import-time coupling bug: the script used to load
its sibling (run_phase2b15_corrected_objective_selector_retraining.py) via
exec_module at module scope, so merely importing this module executed the
sibling script immediately. It is now lazy (see _load_b15 / __getattr__),
loaded only on first actual use.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT_PATH = ROOT / "scripts" / "persist_corrected_selector_artifact.py"
B15_MODULE_NAME = "phase2b15_mod"


def _load_module():
    spec = importlib.util.spec_from_file_location("persist_corrected_selector_artifact", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _clean_b15_from_sys_modules():
    sys.modules.pop(B15_MODULE_NAME, None)
    yield
    sys.modules.pop(B15_MODULE_NAME, None)


def test_import_does_not_trigger_sibling_load():
    """Importing the module alone must not exec_module the Phase 2B.15 sibling."""
    mod = _load_module()
    assert mod._b15_module is None
    assert B15_MODULE_NAME not in sys.modules


def test_load_b15_is_lazy_and_cached():
    mod = _load_module()
    assert mod._b15_module is None

    b15 = mod._load_b15()
    assert mod._b15_module is b15
    assert hasattr(b15, "relabel_rows")
    assert hasattr(b15, "df_to_rows")
    assert hasattr(b15, "split_rows")
    assert hasattr(b15, "_anwg")

    # Second call returns the cached module, not a fresh exec_module.
    b15_again = mod._load_b15()
    assert b15_again is b15


def test_module_getattr_b15_lazy_external_access():
    """`mod.b15` (external attribute access, e.g. from a REPL or another
    test) must work via the module-level __getattr__ and trigger the same
    lazy load as _load_b15()."""
    mod = _load_module()
    assert mod._b15_module is None
    b15_via_getattr = mod.b15
    assert mod._b15_module is b15_via_getattr
    assert hasattr(b15_via_getattr, "relabel_rows")


def test_module_getattr_unknown_name_raises():
    mod = _load_module()
    with pytest.raises(AttributeError):
        mod.not_a_real_attribute


def test_evaluate_on_uses_lazily_loaded_b15():
    """The original runtime path (evaluate_on, used by main()) must still work."""
    pytest.importorskip("sklearn")
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    from llmserveopt.selector.features import FEATURE_NAMES

    mod = _load_module()
    assert mod._b15_module is None

    rows = []
    for i in range(20):
        row = {f"feat_{name}": float((i * 3 + idx) % 7) for idx, name in enumerate(FEATURE_NAMES)}
        for p in SELECTOR_CANDIDATES:
            row[f"completion_{p}"] = 1.0
            row[f"reward_{p}"] = 0.5 + 0.01 * ((i + hash(p)) % 10)
        rows.append(row)

    from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector
    selector = PerPolicyRegressionAnwgSelector(n_estimators=5, max_depth=3, random_state=0)
    selector.fit(rows)

    result = mod.evaluate_on(rows, selector, "unit-test rows")
    assert mod._b15_module is not None, "evaluate_on must trigger the lazy b15 load"
    assert result["n_windows"] == len(rows)
    assert "selector_mean_anwg" in result


def test_main_runtime_path_still_works(tmp_path, monkeypatch):
    """Full main() path against real (read-only) inputs, redirected to a
    scratch output dir so results/ is never touched by this test."""
    pytest.importorskip("sklearn")
    mod = _load_module()

    if not mod.TRAIN_CSV.exists() or not mod.FRESH_CSV.exists():
        pytest.skip("phase2b13/phase2b16 result CSVs not present in this checkout")

    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "corrected_selector_artifact_regression_anwg")
    mod.main()

    out_dir = tmp_path / "corrected_selector_artifact_regression_anwg"
    assert (out_dir / "regression_anwg_selector.joblib").exists()
    assert (out_dir / "manifest.json").exists()
    assert mod._b15_module is not None
