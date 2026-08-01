#!/usr/bin/env python3
"""Check the contextual-composition roadmap/navigation contract."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ROADMAP = ROOT / "docs" / "contextual_composition_roadmap.md"
DECISIONS = ROOT / "docs" / "contextual_composition_decisions.md"
START_HERE = ROOT / "docs" / "START_HERE_CONTEXTUAL_COMPOSITION.md"
BRANCH_MARKER = ROOT / "docs" / "CONTEXTUAL_COMPOSITION_BRANCH.md"
AUDIT = ROOT / "docs" / "audits" / "local_branch_compositional_path_audit_20260731.md"
QUERY2_REPORT = ROOT / "docs" / "audits" / "contextual_composition_query2_roadmap_report_20260731.md"
CC1_SPEC = ROOT / "docs" / "experiments" / "cc1_composition_opportunity_spec.md"
QUERY5_REPORT = ROOT / "docs" / "audits" / "contextual_composition_query5_discriminativeness_review_20260731.md"
PAUSE_CHECKPOINT = ROOT / "docs" / "audits" / "contextual_composition_pause_checkpoint_20260731.md"
RESUME_DOC = ROOT / "docs" / "RESUME_CONTEXTUAL_COMPOSITION.md"
QUERY6_REPORT = ROOT / "docs" / "audits" / "contextual_composition_query6_pause_report_20260731.md"

ALLOWED_STATUS = {
    "COMPLETE",
    "IN PROGRESS",
    "NEXT",
    "BLOCKED",
    "PLANNED",
    "PAUSED",
    "INVALIDATED",
}

REQUIRED_MARKER = {
    "canonical_branch": "contextual-compositional-heuristics-20260731",
    "current_phase": "CC2",
    "current_status": "NEXT",
    "next_action": "Project paused after CC1b; Query 7 should verify resume readiness without implementing CC2",
    "roadmap_version": 1,
}

CANONICAL_FILES = [
    ROADMAP,
    START_HERE,
    BRANCH_MARKER,
    DECISIONS,
    RESUME_DOC,
]


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def read(path: Path) -> str:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def extract_marker(text: str) -> dict[str, object]:
    match = re.search(r"```yaml\n(.*?)\n```", text, flags=re.DOTALL)
    if not match:
        fail("roadmap missing fenced YAML marker block")
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        fail("roadmap marker is not a YAML mapping")
    return data


def check_marker(marker: dict[str, object]) -> None:
    for key, expected in REQUIRED_MARKER.items():
        actual = marker.get(key)
        if actual != expected:
            fail(f"marker {key!r} expected {expected!r}, found {actual!r}")


def check_status_table(text: str) -> None:
    rows = [line for line in text.splitlines() if line.startswith("| CC")]
    phases = {}
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) < 7:
            fail(f"malformed roadmap table row: {row}")
        phase, _purpose, status = cells[:3]
        phases[phase] = status
        if status not in ALLOWED_STATUS:
            fail(f"phase {phase} has invalid status {status!r}")

    expected = {
        "CC0": "COMPLETE",
        "CC1": "COMPLETE",
        "CC2": "NEXT",
        "CC3": "BLOCKED",
        "CC4": "BLOCKED",
        "CC5": "BLOCKED",
        "CC6": "PLANNED",
        "CC7": "PLANNED",
        "CC8": "PLANNED",
    }
    for phase, status in expected.items():
        if phases.get(phase) != status:
            fail(f"phase {phase} expected {status}, found {phases.get(phase)!r}")

    next_count = sum(1 for status in phases.values() if status == "NEXT")
    if next_count != 1:
        fail(f"expected exactly one NEXT phase, found {next_count}")


def check_required_strings(text: str, path: Path, required: list[str]) -> None:
    for needle in required:
        if needle not in text:
            fail(f"{path.relative_to(ROOT)} missing required text: {needle}")


def check_cc1_spec(text: str) -> None:
    required_sections = [
        "## Scientific Question",
        "## Hypotheses",
        "## Minimal Representative Policy Subset",
        "## Exact Composition Semantics",
        "## Normalization Method",
        "## Simulator Execution Path",
        "## Workloads And Splits",
        "## Primary Metric: Arrival-Normalized Weighted Goodput",
        "## Completion-Fraction Constraints",
        "## Near-Tie Handling",
        "## Baselines",
        "## Composition-Opportunity-Gap Formula",
        "## Success And Stop Thresholds",
        "## Required Output Files",
        "## Reproducibility Requirements",
        "## Expected Runtime And Resource Limits",
        "## Query 4 File-By-File Implementation Plan",
        "## Rejected Approaches",
    ]
    check_required_strings(text, CC1_SPEC, required_sections)
    check_required_strings(
        text,
        CC1_SPEC,
        [
            "https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1",
            "StaticRankEnsemblePolicy(method=\"borda\")",
            "arrival_normalized_weighted_goodput",
            "Reward vectors may be used only to report fixed-policy and hard-selector baselines",
        ],
    )


def check_pause_contract(
    roadmap: str,
    start_here: str,
    branch_marker: str,
    resume_doc: str,
    pause_checkpoint: str,
    query6_report: str,
) -> None:
    check_required_strings(
        start_here,
        START_HERE,
        [
            "docs/audits/contextual_composition_pause_checkpoint_20260731.md",
            "docs/RESUME_CONTEXTUAL_COMPOSITION.md",
            "Query 7 should perform final repository polish",
        ],
    )
    check_required_strings(
        roadmap,
        ROADMAP,
        [
            "current_phase: CC2",
            "current_status: NEXT",
            "audits/contextual_composition_pause_checkpoint_20260731.md",
            "RESUME_CONTEXTUAL_COMPOSITION.md",
            "Project paused after CC1b",
        ],
    )
    check_required_strings(
        branch_marker,
        BRANCH_MARKER,
        [
            "Pause Checkpoint",
            "Resume Guide",
            "Query 7 should perform final repository polish",
        ],
    )
    check_required_strings(
        resume_doc,
        RESUME_DOC,
        [
            "Authoritative branch: `contextual-compositional-heuristics-20260731`",
            "Expected checkpoint SHA:",
            "Current phase: `CC2 - Canonical primitive interface`",
            "Define the canonical primitive interface for ranking, admission, placement, batching, and resource guards",
            "GitHub issue #2",
        ],
    )
    check_required_strings(
        pause_checkpoint,
        PAUSE_CHECKPOINT,
        [
            "Current phase: `CC2`",
            "Status: `NEXT`",
            "best fixed ANWG: `0.198977`",
            "oracle fixed ANWG: `0.203773`",
            "best global mixture ANWG: `0.198977`",
            "oracle mixture ANWG: `0.220547`",
            "non-near-tie opportunity gap: `0.0167735`",
            "completion impact: `0.0`",
            "verdict: `PROCEED`",
            "GitHub issue #2",
        ],
    )
    check_required_strings(
        query6_report,
        QUERY6_REPORT,
        [
            "# Contextual Composition Query 6 Pause Report - 2026-07-31",
            "Issue #1 was updated",
            "Issue #2 was updated",
            "Query 7 should perform final repository polish",
        ],
    )


def check_no_cc1_current() -> None:
    forbidden_patterns = [
        r"Current phase:\s*`CC1\b",
        r"current_phase:\s*CC1\b",
        r"CC1\s+is\s+the\s+only\s+`?NEXT`?\s+phase",
        r"CC1\s+remains\s+the\s+single\s+`?NEXT`?\s+phase",
    ]
    for path in CANONICAL_FILES:
        text = read(path)
        for pattern in forbidden_patterns:
            if re.search(pattern, text):
                fail(f"{path.relative_to(ROOT)} still describes CC1 as current")


def main() -> int:
    roadmap = read(ROADMAP)
    decisions = read(DECISIONS)
    start_here = read(START_HERE)
    branch_marker = read(BRANCH_MARKER)
    audit = read(AUDIT)
    query2_report = read(QUERY2_REPORT)
    cc1_spec = read(CC1_SPEC)
    query5_report = read(QUERY5_REPORT)
    pause_checkpoint = read(PAUSE_CHECKPOINT)
    resume_doc = read(RESUME_DOC)
    query6_report = read(QUERY6_REPORT)

    check_marker(extract_marker(roadmap))
    check_status_table(roadmap)

    check_required_strings(
        roadmap,
        ROADMAP,
        [
            "# Contextual Compositional Heuristics Roadmap",
            "## Research Invariants",
            "## Roadmap Update Protocol",
            "arrival-normalized weighted goodput",
            "experiments/cc1_composition_opportunity_spec.md",
            "audits/contextual_composition_query4_cc1_results_20260731.md",
            "audits/contextual_composition_query5_discriminativeness_review_20260731.md",
            "audits/contextual_composition_pause_checkpoint_20260731.md",
            "RESUME_CONTEXTUAL_COMPOSITION.md",
            "audits/contextual_composition_query2_roadmap_report_20260731.md",
            "https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1",
            "https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/2",
        ],
    )
    check_required_strings(
        decisions,
        DECISIONS,
        [f"CCD-{idx:03d}" for idx in range(1, 12)],
    )
    check_required_strings(
        start_here,
        START_HERE,
        [
            "Authoritative branch: `contextual-compositional-heuristics-20260731`",
            "Current phase: `CC2 - Canonical primitive interface`",
            "docs/experiments/cc1_composition_opportunity_spec.md",
            "docs/audits/contextual_composition_query4_cc1_results_20260731.md",
            "docs/audits/contextual_composition_query5_discriminativeness_review_20260731.md",
            "docs/audits/contextual_composition_pause_checkpoint_20260731.md",
            "docs/RESUME_CONTEXTUAL_COMPOSITION.md",
            "docs/audits/contextual_composition_query2_roadmap_report_20260731.md",
            "python scripts/check_contextual_composition_status.py",
        ],
    )
    check_required_strings(
        branch_marker,
        BRANCH_MARKER,
        [
            "contextual-compositional-heuristics-20260731",
            "contextual_composition_roadmap.md",
            "experiments/cc1_composition_opportunity_spec.md",
            "Query 7 should perform final repository polish",
        ],
    )
    check_required_strings(
        audit,
        AUDIT,
        ["## Synchronization Addendum - 2026-07-31"],
    )
    check_required_strings(
        query2_report,
        QUERY2_REPORT,
        ["# Contextual Composition Query 2 Roadmap Report - 2026-07-31"],
    )
    check_cc1_spec(cc1_spec)
    check_required_strings(
        query5_report,
        QUERY5_REPORT,
        [
            "# Contextual Composition Query 5 Discriminativeness Review - 2026-07-31",
            "CC1b verdict: `PROCEED`",
            "non-near-tie composition-opportunity gap: `0.0167735`",
            "Query 6 should begin CC2",
        ],
    )
    check_pause_contract(
        roadmap=roadmap,
        start_here=start_here,
        branch_marker=branch_marker,
        resume_doc=resume_doc,
        pause_checkpoint=pause_checkpoint,
        query6_report=query6_report,
    )
    check_no_cc1_current()

    print("contextual composition status check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
