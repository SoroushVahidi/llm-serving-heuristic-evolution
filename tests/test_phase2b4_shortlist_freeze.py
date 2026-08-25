"""
Tests for Phase 2B.4 shortlist freeze logic.

Verifies:
1. TEST_REGIMES are accessible and have 'test' split label.
2. TEST_REGIMES are distinct from TRAIN_REGIMES and VALIDATION_REGIMES.
3. include_oracle flag is False by default on MultiRegimeConfig.
4. MultiRegimeConfig with include_oracle=True passes the flag through.
5. Phase 2B.4 config files parse without error.
"""
from pathlib import Path


def test_test_regimes_defined():
    from llmserveopt.llm_generation.multi_regime_evaluation import TEST_REGIMES
    assert len(TEST_REGIMES) >= 3, "Need at least 3 test regimes for held-out evaluation"


def test_test_regimes_have_test_split():
    from llmserveopt.llm_generation.multi_regime_evaluation import TEST_REGIMES
    for regime in TEST_REGIMES:
        assert regime.split == "test", (
            f"TEST_REGIMES entry '{regime.name}' has split='{regime.split}', expected 'test'"
        )


def test_test_regimes_distinct_from_train():
    from llmserveopt.llm_generation.multi_regime_evaluation import (
        TRAIN_REGIMES, TEST_REGIMES
    )
    train_names = {r.name for r in TRAIN_REGIMES}
    test_names = {r.name for r in TEST_REGIMES}
    overlap = train_names & test_names
    assert not overlap, f"Test regimes overlap with train: {overlap}"


def test_test_regimes_distinct_from_validation():
    from llmserveopt.llm_generation.multi_regime_evaluation import (
        VALIDATION_REGIMES, TEST_REGIMES
    )
    val_names = {r.name for r in VALIDATION_REGIMES}
    test_names = {r.name for r in TEST_REGIMES}
    overlap = val_names & test_names
    assert not overlap, f"Test regimes overlap with validation: {overlap}"


def test_multi_regime_config_no_oracle_by_default():
    from llmserveopt.llm_generation.multi_regime_evaluation import MultiRegimeConfig
    cfg = MultiRegimeConfig()
    assert cfg.include_oracle is False


def test_multi_regime_config_oracle_flag():
    from llmserveopt.llm_generation.multi_regime_evaluation import MultiRegimeConfig
    cfg = MultiRegimeConfig(include_oracle=True)
    assert cfg.include_oracle is True


def test_default_regimes_does_not_include_test():
    from llmserveopt.llm_generation.multi_regime_evaluation import DEFAULT_REGIMES
    for r in DEFAULT_REGIMES:
        assert r.split != "test", (
            f"DEFAULT_REGIMES should not include test regime '{r.name}'"
        )


def test_phase2b4_train_val_config_parses():
    path = Path(__file__).parent.parent / "configs" / "llm_generation" / \
        "phase2b4_shortlist_train_validation.yaml"
    assert path.exists(), f"Config not found: {path}"
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert "evaluation" in cfg


def test_phase2b4_heldout_test_config_parses():
    path = Path(__file__).parent.parent / "configs" / "llm_generation" / \
        "phase2b4_final_heldout_test.yaml"
    assert path.exists(), f"Config not found: {path}"
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert "evaluation" in cfg
    assert cfg["evaluation"].get("include_oracle") is True


def test_evaluate_multi_regime_split_flag_help():
    """verify --split flag is available in evaluate_multi_regime.py"""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_multi_regime.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--split" in result.stdout
    assert "test" in result.stdout
    assert "--include-oracle" in result.stdout


def test_all_baselines_flag_available():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_multi_regime.py", "--help"],
        capture_output=True, text=True
    )
    assert "--all-baselines" in result.stdout
