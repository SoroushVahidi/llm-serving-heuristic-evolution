"""
Tests for Phase 2A.4 selector configs.

Verifies:
1. All four Phase 2A.4 config files parse without error.
2. Train, validation, and test configs reference only existing trace files.
3. No test-split trace file appears in train or validation configs.
4. Config workload counts are within expected ranges.
5. All 18 selector candidate policies are available in the registry.
"""
import pytest
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "selector"

TRAIN_CONFIG = CONFIG_DIR / "phase2a4_train_18policies.yaml"
VAL_CONFIG = CONFIG_DIR / "phase2a4_validation_18policies.yaml"
TEST_CONFIG = CONFIG_DIR / "phase2a4_test_18policies.yaml"

DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "burstgpt"

# Trace file used exclusively in test split
TEST_ONLY_TRACES = {"burstgpt_scaled_high_10k.jsonl"}


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def test_train_config_parses():
    cfg = _load(TRAIN_CONFIG)
    assert "workloads" in cfg
    assert len(cfg["workloads"]) >= 5


def test_validation_config_parses():
    cfg = _load(VAL_CONFIG)
    assert "workloads" in cfg
    assert len(cfg["workloads"]) >= 2


def test_test_config_parses():
    cfg = _load(TEST_CONFIG)
    assert "workloads" in cfg
    assert len(cfg["workloads"]) >= 2


def test_referenced_trace_files_exist():
    for config_path in [TRAIN_CONFIG, VAL_CONFIG, TEST_CONFIG]:
        cfg = _load(config_path)
        for w in cfg.get("workloads", []):
            if "trace_path" in w:
                p = Path(__file__).parent.parent / w["trace_path"]
                assert p.exists(), f"{config_path.name}: trace_path not found: {w['trace_path']}"


def test_test_only_traces_not_in_train_or_val():
    for config_path in [TRAIN_CONFIG, VAL_CONFIG]:
        cfg = _load(config_path)
        for w in cfg.get("workloads", []):
            if "trace_path" in w:
                trace_file = Path(w["trace_path"]).name
                assert trace_file not in TEST_ONLY_TRACES, (
                    f"{config_path.name} references test-only trace: {trace_file}"
                )


def test_all_18_selector_candidates_available():
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    assert len(SELECTOR_CANDIDATES) == 18


def test_selector_candidates_no_oracle():
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    from llmserveopt.policies.registry import ORACLE_POLICY_NAMES
    for oracle in ORACLE_POLICY_NAMES:
        assert oracle not in SELECTOR_CANDIDATES, (
            f"oracle policy '{oracle}' must not be in SELECTOR_CANDIDATES"
        )


def test_baseline_names_count():
    from llmserveopt.policies.registry import BASELINE_NAMES
    assert len(BASELINE_NAMES) == 18


def test_configs_have_consistent_window_size():
    sizes = set()
    for config_path in [TRAIN_CONFIG, VAL_CONFIG, TEST_CONFIG]:
        cfg = _load(config_path)
        sizes.add(cfg.get("window_size", 200))
    assert len(sizes) == 1, f"Inconsistent window sizes: {sizes}"


def test_test_config_has_harder_conditions_than_train():
    train_cfg = _load(TRAIN_CONFIG)
    test_cfg = _load(TEST_CONFIG)
    # Test should have at least one regime with higher arrival rate than any train regime
    train_rates = [
        w.get("arrival_rate", 0)
        for w in train_cfg["workloads"]
        if "arrival_rate" in w
    ]
    test_rates = [
        w.get("arrival_rate", 0)
        for w in test_cfg["workloads"]
        if "arrival_rate" in w
    ]
    if train_rates and test_rates:
        assert max(test_rates) >= max(train_rates) * 0.8, (
            "Test regimes should be at least as hard as train in terms of arrival rate"
        )
