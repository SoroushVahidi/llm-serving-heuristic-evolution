from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_contextual_composition_status_checker_passes():
    result = subprocess.run(
        [sys.executable, "scripts/check_contextual_composition_status.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "contextual composition status check passed" in result.stdout


def test_cc1_spec_required_sections_are_present():
    text = (ROOT / "docs/experiments/cc1_composition_opportunity_spec.md").read_text()
    required = [
        "## Scientific Question",
        "## Hypotheses",
        "## Minimal Representative Policy Subset",
        "## Exact Composition Semantics",
        "## Normalization Method",
        "## Simulator Execution Path",
        "## Workloads And Splits",
        "## Baselines",
        "## Composition-Opportunity-Gap Formula",
        "## Query 4 File-By-File Implementation Plan",
    ]
    for section in required:
        assert section in text


def test_roadmap_links_cc1_spec_and_has_one_next_phase():
    text = (ROOT / "docs/contextual_composition_roadmap.md").read_text()
    assert "experiments/cc1_composition_opportunity_spec.md" in text
    rows = [line for line in text.splitlines() if line.startswith("| CC")]
    statuses = [row.strip("|").split("|")[2].strip() for row in rows]
    assert statuses.count("NEXT") == 1
