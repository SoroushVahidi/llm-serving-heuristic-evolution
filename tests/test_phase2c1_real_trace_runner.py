"""
Tests for Phase 2C.1 real-trace ingestion validation support.
"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from llmserveopt.core.types import GPUConfig
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.augmentation import AugmentationConfig
from llmserveopt.workloads.trace_io_extended import load_extended_jsonl, save_extended_jsonl
from scripts.data.convert_azure_llm_trace import load_azure_csv, convert_azure_to_requests

RUNNER_PATH = ROOT / "scripts" / "run_phase2c1_real_trace_ingestion_validation.py"
CONFIG_PATH = ROOT / "configs" / "phase2c1_real_trace_ingestion_validation.yaml"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_phase2c1", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _tiny_azure_csv(tmpdir: Path) -> Path:
    csv_path = tmpdir / "azure_tiny.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
        )
        writer.writeheader()
        writer.writerow({
            "TIMESTAMP": "2024-01-01 00:00:00.0000000",
            "ContextTokens": "128",
            "GeneratedTokens": "16",
        })
        writer.writerow({
            "TIMESTAMP": "2024-01-01 00:00:01.5000000",
            "ContextTokens": "256",
            "GeneratedTokens": "32",
        })
        writer.writerow({
            "TIMESTAMP": "2024-01-01 00:00:03.0000000",
            "ContextTokens": "512",
            "GeneratedTokens": "64",
        })
    return csv_path


def test_azure_csv_conversion_parsing_tiny_temp_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _tiny_azure_csv(Path(tmpdir))
        arrivals, context_tokens, generated_tokens, report = load_azure_csv(
            csv_path,
            time_scale=0.5,
        )
        assert len(arrivals) == 3
        assert float(arrivals[0]) == 0.0
        assert round(float(arrivals[-1]), 3) == 1.5
        assert report["rows_read"] == 3
        assert report["rows_retained"] == 3
        assert list(context_tokens) == [128, 256, 512]
        assert list(generated_tokens) == [16, 32, 64]


def test_converted_azure_jsonl_is_simulator_compatible():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = _tiny_azure_csv(tmp)
        arrivals, context_tokens, generated_tokens, _report = load_azure_csv(
            csv_path,
            time_scale=0.25,
        )
        requests = convert_azure_to_requests(
            arrivals,
            context_tokens,
            generated_tokens,
            AugmentationConfig(),
            seed=17,
        )
        out_path = tmp / "azure_tiny.jsonl"
        save_extended_jsonl(requests, out_path, source="azure_test")

        loaded_requests, loaded_meta = load_extended_jsonl(out_path)
        assert len(loaded_requests) == 3
        assert loaded_meta[0]["source"] == "azure_test"

        metrics = run_policy(
            policy=FIFOPolicy(),
            requests=loaded_requests,
            gpu_configs=[
                GPUConfig(
                    gpu_id=0,
                    max_active_sequences=4,
                    max_batch_tokens=4096,
                    max_kv_tokens=32768,
                )
            ],
            service_model=ServiceModel(
                step_size=0.001,
                enable_prefill_modeling=True,
                prefill_cost_per_token=1.0,
                max_prefill_chunk_tokens=512,
                step_token_budget=4096,
                decode_first=False,
            ),
            workload_tag="azure_tiny",
            seed=17,
            drain_steps=20000,
        )
        assert metrics.num_completed == 3
        assert metrics.num_dropped == 0
        assert metrics.mean_latency >= 0.0


def test_phase2c1_config_loads_and_validates():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["experiment"] == "phase2c1_real_trace_ingestion_validation"
    assert len(cfg["workloads"]) == 6
    mod = _load_runner_module()
    issues, plan = mod.validate_phase2c1_config(cfg)
    assert issues == []
    assert plan["experiment"] == "phase2c1_real_trace_ingestion_validation"
    assert len(plan["workloads"]) == 6
    assert len(plan["azure_2023"]) == 2


def test_phase2c1_runner_dry_run():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dry-run" in result.stdout.lower()
    assert "no files written" in result.stdout.lower()


def test_phase2c1_runner_refuses_full_run_without_flag():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--allow-full-run" in result.stderr


def test_phase2c1_config_uses_causal_feature_mode():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["feature_mode"] == "causal"


def test_phase2c1_config_rejects_offline_lookahead_feature_mode():
    mod = _load_runner_module()
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    bad_cfg = dict(cfg)
    bad_cfg["feature_mode"] = "offline_window_lookahead"
    issues, _plan = mod.validate_phase2c1_config(bad_cfg)
    assert any("feature_mode must be 'causal'" in issue for issue in issues)


def test_selector_roles_classify_oracle_assisted():
    from llmserveopt.selector.roles import (
        classify_selectors,
        is_oracle_assisted_selector,
        is_deployable_headline_selector,
        selector_role,
    )

    assert is_oracle_assisted_selector("safe_fallback_wsp_margin0.005")
    assert selector_role("safe_fallback_wsp_margin0.005") == "oracle_assisted"
    assert not is_deployable_headline_selector("safe_fallback_wsp_margin0.005")
    assert is_deployable_headline_selector("regression_anwg")

    roles = classify_selectors([
        "regression_anwg",
        "always_scorpio",
        "safe_fallback_wsp_margin0.001",
    ])
    assert roles["deployable_learned"] == ["regression_anwg"]
    assert roles["always_fixed"] == ["always_scorpio"]
    assert roles["oracle_assisted"] == ["safe_fallback_wsp_margin0.001"]


def test_phase2c1_smoke_writes_deployable_selector_summary():
    mod = _load_runner_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "smoke_out"
        workloads = mod.build_smoke_workloads(out_dir)
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        metadata = mod.run_phase2c1_validation(
            cfg,
            workloads=workloads,
            out_dir=out_dir,
            allow_fallback_selectors=True,
            window_size=4,
            min_partial_window=2,
        )
        assert (out_dir / "deployable_selector_summary.csv").exists()
        assert metadata["primary_rank_metric"] == "mean_arrival_normalized_wg"
        assert metadata["deployable_headline_selectors"]
        assert not any(
            s.startswith("safe_fallback_wsp_margin")
            for s in metadata["deployable_headline_selectors"]
        )
        with open(out_dir / "deployable_selector_summary.csv") as f:
            rows = list(csv.DictReader(f))
        assert rows
        assert all(row.get("deployable_headline") == "True" for row in rows)
        assert all("safe_fallback" not in row["selector"] for row in rows)


def test_phase2c1_azure_materialization_requires_explicit_download_flag():
    mod = _load_runner_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg = {
            "azure": {
                "2023": {
                    "code": {
                        "url": "https://example.invalid/code.csv",
                        "raw_path": str(tmp / "missing_code.csv"),
                        "processed_path": str(tmp / "missing_code.jsonl"),
                        "time_scale": 0.05,
                        "source_tag": "azure_2023_code",
                    },
                    "conv": {
                        "url": "https://example.invalid/conv.csv",
                        "raw_path": str(tmp / "missing_conv.csv"),
                        "processed_path": str(tmp / "missing_conv.jsonl"),
                        "time_scale": 0.10,
                        "source_tag": "azure_2023_conv",
                    },
                }
            }
        }
        try:
            mod.ensure_azure_2023_inputs(
                cfg,
                allow_download=False,
                dry_run=False,
            )
        except RuntimeError as exc:
            assert "--allow-azure-download" in str(exc)
        else:
            raise AssertionError("expected RuntimeError when Azure download flag is missing")
