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


def test_contextual_composition_resume_readiness_checker_passes():
    result = subprocess.run(
        [sys.executable, "scripts/check_contextual_composition_status.py", "--resume-readiness"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "contextual composition resume-readiness check passed" in result.stdout


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


def test_roadmap_links_cc1b_report_and_has_cc5_next():
    text = (ROOT / "docs/contextual_composition_roadmap.md").read_text()
    assert "experiments/cc1_composition_opportunity_spec.md" in text
    assert "audits/contextual_composition_query4_cc1_results_20260731.md" in text
    assert "audits/contextual_composition_query5_discriminativeness_review_20260731.md" in text
    assert "audits/contextual_composition_pause_checkpoint_20260731.md" in text
    assert "RESUME_CONTEXTUAL_COMPOSITION.md" in text
    assert "architecture/contextual_composition_primitives.md" in text
    assert "CC1b `PROCEED`" in text
    rows = [line for line in text.splitlines() if line.startswith("| CC")]
    phases = {row.strip("|").split("|")[0].strip(): row.strip("|").split("|")[2].strip() for row in rows}
    assert phases["CC1"] == "COMPLETE"
    assert phases["CC2"] == "COMPLETE"
    assert phases["CC3"] == "COMPLETE"
    assert phases["CC4"] == "COMPLETE"
    assert phases["CC5"] == "NEXT"
    assert list(phases.values()).count("NEXT") == 1


def test_pause_checkpoint_records_cc1b_evidence_and_cc2_scope():
    text = (ROOT / "docs/audits/contextual_composition_pause_checkpoint_20260731.md").read_text()
    required = [
        "Current phase: `CC2`",
        "Status: `NEXT`",
        "best fixed ANWG: `0.198977`",
        "oracle fixed ANWG: `0.203773`",
        "best global mixture ANWG: `0.198977`",
        "oracle mixture ANWG: `0.220547`",
        "non-near-tie opportunity gap: `0.0167735`",
        "completion impact: `0.0`",
        "verdict: `PROCEED`",
        "Do not extend the DSL yet.",
    ]
    for needle in required:
        assert needle in text


def test_resume_doc_names_branch_expected_sha_field_and_exact_task():
    text = (ROOT / "docs/RESUME_CONTEXTUAL_COMPOSITION.md").read_text()
    assert "Authoritative branch: `contextual-compositional-heuristics-20260731`" in text
    assert "Query 6 checkpoint SHA: `f6b4be9dc15fc4f13286f23b5aae39f48fbd01fb`" in text
    assert "Current phase: `CC5 - Contextual composition predictor`" in text
    assert "python scripts/check_contextual_composition_status.py --resume-readiness" in text
    assert (
        "python -m pytest tests/test_contextual_composition_status_checker.py "
        "tests/test_cc1_composition_opportunity.py tests/test_policy_composition.py "
        "tests/test_score_and_reciprocal_rank_composition.py tests/test_primitive_interface.py "
        "tests/test_primitive_reconstructed_policies.py tests/test_contextual_composition_cc3_dsl.py "
        "tests/test_cc4_oracle_composition_dataset.py tests/test_cc5_contextual_predictor.py -q"
    ) in text
    assert "must be **retried**, not begun fresh" in text
    assert "oracle_labels.parquet" in text
    assert "GitHub issue #5" in text


def test_canonical_docs_do_not_make_cc1_current():
    canonical_paths = [
        ROOT / "docs/contextual_composition_roadmap.md",
        ROOT / "docs/START_HERE_CONTEXTUAL_COMPOSITION.md",
        ROOT / "docs/CONTEXTUAL_COMPOSITION_BRANCH.md",
        ROOT / "docs/contextual_composition_decisions.md",
        ROOT / "docs/RESUME_CONTEXTUAL_COMPOSITION.md",
    ]
    forbidden = [
        "Current phase: `CC1",
        "current_phase: CC1",
        "CC1 is the only `NEXT` phase",
        "CC1 remains the single `NEXT` phase",
    ]
    for path in canonical_paths:
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{path.relative_to(ROOT)} contains {needle!r}"


def test_canonical_docs_do_not_make_cc2_in_progress():
    canonical_paths = [
        ROOT / "docs/contextual_composition_roadmap.md",
        ROOT / "docs/START_HERE_CONTEXTUAL_COMPOSITION.md",
        ROOT / "docs/CONTEXTUAL_COMPOSITION_BRANCH.md",
        ROOT / "docs/contextual_composition_decisions.md",
        ROOT / "docs/RESUME_CONTEXTUAL_COMPOSITION.md",
    ]
    forbidden = [
        "Current status: `IN PROGRESS`",
        "current_status: IN PROGRESS",
        "CC2 has started",
        "CC2 is `IN PROGRESS`",
    ]
    for path in canonical_paths:
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{path.relative_to(ROOT)} contains {needle!r}"
