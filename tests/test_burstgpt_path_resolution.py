"""Focused portability tests for BurstGPT path resolution.

These tests must not depend on /mmfs1 or any account-specific layout.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_local_e2e_smoke.py"
    spec = importlib.util.spec_from_file_location("run_local_e2e_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke():
    return _load_smoke_module()


def _touch_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Timestamp,Request Token,Response Token\n1.0,8,4\n")
    return path


def test_no_hardcoded_mmfs1_account_path_in_resolver_source(smoke):
    src = Path(smoke.__file__).read_text()
    assert "/mmfs1/" not in src
    assert "sv96" not in src


def test_explicit_existing_path_wins(smoke, tmp_path, monkeypatch):
    monkeypatch.delenv(smoke.ENV_BURSTGPT_CSV, raising=False)
    monkeypatch.delenv(smoke.ENV_DATA_ROOT, raising=False)
    explicit = _touch_csv(tmp_path / "explicit.csv")
    # Even if portable/env exist, explicit existing path wins.
    monkeypatch.setenv(smoke.ENV_BURSTGPT_CSV, str(tmp_path / "env.csv"))
    _touch_csv(tmp_path / "env.csv")
    assert smoke.resolve_burstgpt_trace_path(explicit) == explicit


def test_nondefault_missing_explicit_does_not_fallback(smoke, tmp_path, monkeypatch):
    monkeypatch.delenv(smoke.ENV_BURSTGPT_CSV, raising=False)
    monkeypatch.delenv(smoke.ENV_DATA_ROOT, raising=False)
    # Create a data-root candidate that would otherwise be used.
    root = tmp_path / "data_root"
    cluster = root / smoke.CLUSTER_BURSTGPT_REL
    _touch_csv(cluster)
    monkeypatch.setenv(smoke.ENV_DATA_ROOT, str(root))
    missing = tmp_path / "missing_custom.csv"
    assert smoke.resolve_burstgpt_trace_path(missing) is None


def test_env_csv_used_when_portable_default_missing(smoke, tmp_path, monkeypatch):
    monkeypatch.delenv(smoke.ENV_DATA_ROOT, raising=False)
    monkeypatch.setattr(smoke, "ROOT", tmp_path / "empty_repo")
    (tmp_path / "empty_repo").mkdir()
    env_csv = _touch_csv(tmp_path / "from_env.csv")
    monkeypatch.setenv(smoke.ENV_BURSTGPT_CSV, str(env_csv))
    assert smoke.resolve_burstgpt_trace_path(smoke.DEFAULT_BURSTGPT_TRACE_REL) == env_csv


def test_env_csv_missing_fails_closed(smoke, tmp_path, monkeypatch):
    monkeypatch.setenv(smoke.ENV_BURSTGPT_CSV, str(tmp_path / "absent.csv"))
    monkeypatch.delenv(smoke.ENV_DATA_ROOT, raising=False)
    monkeypatch.setattr(smoke, "ROOT", tmp_path / "empty_repo")
    (tmp_path / "empty_repo").mkdir()
    # Bad env must not fall through to other sources.
    data_root = tmp_path / "llmserveopt-data"
    _touch_csv(data_root / smoke.CLUSTER_BURSTGPT_REL)
    monkeypatch.setenv(smoke.ENV_DATA_ROOT, str(data_root))
    assert smoke.resolve_burstgpt_trace_path(smoke.DEFAULT_BURSTGPT_TRACE_REL) is None


def test_portable_default_used_when_present(smoke, tmp_path, monkeypatch):
    monkeypatch.delenv(smoke.ENV_BURSTGPT_CSV, raising=False)
    monkeypatch.delenv(smoke.ENV_DATA_ROOT, raising=False)
    portable = _touch_csv(tmp_path / "repo" / "data" / "raw" / "burstgpt" / "BurstGPT_1.csv")
    monkeypatch.setattr(smoke, "ROOT", tmp_path / "repo")
    assert smoke.resolve_burstgpt_trace_path(smoke.DEFAULT_BURSTGPT_TRACE_REL) == portable


def test_data_root_cluster_fallback(smoke, tmp_path, monkeypatch):
    monkeypatch.delenv(smoke.ENV_BURSTGPT_CSV, raising=False)
    data_root = tmp_path / "llmserveopt-data"
    cluster = _touch_csv(data_root / smoke.CLUSTER_BURSTGPT_REL)
    monkeypatch.setenv(smoke.ENV_DATA_ROOT, str(data_root))
    # Point ROOT at an empty fake repo so portable default is absent.
    monkeypatch.setattr(smoke, "ROOT", tmp_path / "empty_repo")
    (tmp_path / "empty_repo").mkdir()
    assert smoke.resolve_burstgpt_trace_path(smoke.DEFAULT_BURSTGPT_TRACE_REL) == cluster


def test_missing_everything_returns_none(smoke, tmp_path, monkeypatch):
    monkeypatch.delenv(smoke.ENV_BURSTGPT_CSV, raising=False)
    monkeypatch.delenv(smoke.ENV_DATA_ROOT, raising=False)
    monkeypatch.setattr(smoke, "ROOT", tmp_path / "empty_repo")
    (tmp_path / "empty_repo").mkdir()
    assert smoke.resolve_burstgpt_trace_path(smoke.DEFAULT_BURSTGPT_TRACE_REL) is None
