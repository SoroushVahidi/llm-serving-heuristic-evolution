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


def test_entry_points_link_to_canonical_resume_doc():
    assert _module.check_canonical_links() == []


def test_no_forbidden_stale_claims_present():
    assert _module.check_forbidden_claims() == []


def test_required_current_status_claims_present():
    assert _module.check_required_claims() == []


def test_exactly_one_canonical_resume_document():
    assert _module.check_single_canonical_resume_doc() == []


def test_main_passes_on_current_repository_state():
    assert _module.main() == 0
