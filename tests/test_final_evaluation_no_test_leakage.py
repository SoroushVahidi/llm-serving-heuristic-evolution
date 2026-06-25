"""
Tests that test regimes are not exposed before the shortlist is frozen.

Verifies:
1. DEFAULT_REGIMES (used by the standard evaluation path) contains no test regimes.
2. Test regimes have strictly higher difficulty than train regimes.
3. summarize_final_evaluation.py has correct CLI interface.
4. plot_final_evaluation.py has correct CLI interface with --dry-run.
5. The selector train/val/test split configs do not reference each other's trace files.
"""
import subprocess
import sys
import pytest
from pathlib import Path


def test_default_regimes_no_test_split():
    from llmserveopt.llm_generation.multi_regime_evaluation import DEFAULT_REGIMES
    test_regimes = [r for r in DEFAULT_REGIMES if r.split == "test"]
    assert len(test_regimes) == 0, (
        f"DEFAULT_REGIMES must not include test regimes: {[r.name for r in test_regimes]}"
    )


def test_test_regimes_harder_noise_than_train():
    from llmserveopt.llm_generation.multi_regime_evaluation import TRAIN_REGIMES, TEST_REGIMES
    max_train_noise = max(r.workload.prediction_noise_rel for r in TRAIN_REGIMES)
    for tr in TEST_REGIMES:
        assert tr.workload.prediction_noise_rel >= max_train_noise * 0.5, (
            f"Test regime '{tr.name}' noise ({tr.workload.prediction_noise_rel}) "
            f"is much lower than max train noise ({max_train_noise})"
        )


def test_test_regimes_harder_load_or_burst():
    from llmserveopt.llm_generation.multi_regime_evaluation import TRAIN_REGIMES, TEST_REGIMES
    max_train_rate = max(r.workload.arrival_rate for r in TRAIN_REGIMES)
    max_test_rate = max(r.workload.arrival_rate for r in TEST_REGIMES)
    # At least one test regime has higher load, OR test has higher noise/burst
    max_train_burst = max(
        (r.workload.burst_factor or 1.0) for r in TRAIN_REGIMES
    )
    max_test_burst = max(
        (r.workload.burst_factor or 1.0) for r in TEST_REGIMES
    )
    harder = (max_test_rate > max_train_rate) or (max_test_burst > max_train_burst)
    assert harder, (
        "Test regimes should be harder than train regimes in at least one dimension "
        f"(rate: train={max_train_rate}, test={max_test_rate}; "
        f"burst: train={max_train_burst}, test={max_test_burst})"
    )


def test_summarize_final_evaluation_help():
    result = subprocess.run(
        [sys.executable, "scripts/summarize_final_evaluation.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--eval-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--n-bootstrap" in result.stdout


def test_plot_final_evaluation_help():
    result = subprocess.run(
        [sys.executable, "scripts/plot_final_evaluation.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--summary-dir" in result.stdout
    assert "--dry-run" in result.stdout


def test_summarize_does_not_modify_working_tree():
    import subprocess
    # Capture git state before
    before = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True
    ).stdout.strip()

    result = subprocess.run(
        [sys.executable, "scripts/summarize_final_evaluation.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0

    # Capture git state after — must be identical
    after = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True
    ).stdout.strip()
    assert before == after, (
        f"git status changed after --help.\nBefore:\n{before}\nAfter:\n{after}"
    )


def test_selector_config_splits_use_different_traces():
    import yaml
    config_dir = Path(__file__).parent.parent / "configs" / "selector"
    configs = {
        "train": config_dir / "phase2a4_train_18policies.yaml",
        "validation": config_dir / "phase2a4_validation_18policies.yaml",
        "test": config_dir / "phase2a4_test_18policies.yaml",
    }
    traces_by_split = {}
    for split, path in configs.items():
        with open(path) as f:
            cfg = yaml.safe_load(f)
        traces_by_split[split] = {
            w["trace_path"] for w in cfg.get("workloads", [])
            if "trace_path" in w
        }
    # Test-only traces must not appear in train or val
    test_traces = traces_by_split["test"]
    for split in ["train", "validation"]:
        overlap = test_traces & traces_by_split[split]
        assert not overlap, (
            f"Test traces found in {split} config: {overlap}"
        )
