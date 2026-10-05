"""Focused tests for scripts/check_project_handoff_consistency.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_project_handoff_consistency.py"

_spec = importlib.util.spec_from_file_location("check_project_handoff_consistency", SCRIPT)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)  # type: ignore[union-attr]


def test_required_current_docs_exist():
    assert _module.check_required_docs_exist() == []


def test_entry_points_link_to_current_study_index():
    assert _module.check_entry_point_links() == []


def test_historical_entry_points_carry_banner():
    assert _module.check_historical_banners() == []


def test_current_docs_free_of_stale_framing():
    assert _module.check_current_docs_not_stale() == []


def test_at_most_one_resume_document():
    assert _module.check_single_resume_doc() == []


def test_banner_check_detects_missing_banner(tmp_path, monkeypatch):
    doc = tmp_path / "STATUS.md"
    doc.write_text("# Status\n\nThe current next action is X.\n", encoding="utf-8")
    monkeypatch.setattr(_module, "ROOT", tmp_path)
    monkeypatch.setattr(_module, "HISTORICAL_ENTRY_POINTS", [doc])
    assert _module.check_historical_banners() != []


def test_main_passes_on_current_repository_state():
    assert _module.main() == 0
