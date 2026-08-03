#!/usr/bin/env python3
"""Check the contextual-composition roadmap/navigation contract."""

from __future__ import annotations

import re
import subprocess
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
QUERY7_REPORT = ROOT / "docs" / "audits" / "contextual_composition_query7_final_pause_readiness_20260731.md"
CC2_ARCHITECTURE_DOC = ROOT / "docs" / "architecture" / "contextual_composition_primitives.md"
CC2_REPORT = ROOT / "docs" / "audits" / "contextual_composition_cc2_primitive_interface_report_20260802.md"
CC3_ARCHITECTURE_DOC = ROOT / "docs" / "architecture" / "contextual_composition_dsl.md"
CC3_REPORT = ROOT / "docs" / "audits" / "contextual_composition_cc3_dsl_verifier_report_20260803.md"
CC4_REPORT = ROOT / "docs" / "audits" / "contextual_composition_cc4_oracle_dataset_report_20260803.md"
CC5_REPORT = ROOT / "docs" / "audits" / "contextual_composition_cc5_predictor_report_20260803.md"

EXPECTED_BRANCH = "contextual-compositional-heuristics-20260731"
EXPECTED_UPSTREAM = f"origin/{EXPECTED_BRANCH}"

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
    "canonical_branch": EXPECTED_BRANCH,
    "current_phase": "CC5",
    "current_status": "IN PROGRESS",
    "next_action": "the CC4b/CC5 retry completed with verdict REGIME_SPECIFIC_ONLY (predictor beats best fixed policy and is competitive with the hard selector on 76 held-out windows, but does not clearly beat best_global_composition); CC6 remains not queued; next step is addressing the uncertainty-method gap (no LOWO-CV-selected model has supported ensemble uncertainty across two retries) then a per-regime regret breakdown, per the CC4b/CC5 retry report section 10",
    "roadmap_version": 7,
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


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "unknown git error"
        fail(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


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
        "CC2": "COMPLETE",
        "CC3": "COMPLETE",
        "CC4": "COMPLETE",
        "CC5": "IN PROGRESS",
        "CC6": "BLOCKED",
        "CC7": "BLOCKED",
        "CC8": "BLOCKED",
    }
    for phase, status in expected.items():
        if phases.get(phase) != status:
            fail(f"phase {phase} expected {status}, found {phases.get(phase)!r}")

    # "Active" phase = the single phase currently queued (NEXT) or actively
    # being worked (IN PROGRESS) -- exactly one of either value must be
    # present, matching the real state of whatever query/job is running.
    active_count = sum(1 for status in phases.values() if status in ("NEXT", "IN PROGRESS"))
    if active_count != 1:
        fail(f"expected exactly one active (NEXT or IN PROGRESS) phase, found {active_count}")

    # CC6 must never be marked active while CC5 (its dependency) is
    # unresolved -- CC5 not yet COMPLETE implies CC6 must stay BLOCKED.
    if phases.get("CC5") != "COMPLETE" and phases.get("CC6") in ("NEXT", "IN PROGRESS", "COMPLETE"):
        fail(f"CC6 is marked {phases.get('CC6')!r} while CC5 is unresolved ({phases.get('CC5')!r})")


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
    query7_report: str,
) -> None:
    """Checks named for the CC1b-to-CC2 pause era; retained (with updated
    expectations) for the CC2-to-CC3 transition because they verify the same
    invariant -- current navigation docs describe the current NEXT phase --
    plus, unchanged, that the frozen historical pause-era reports still
    record what they always recorded."""
    check_required_strings(
        start_here,
        START_HERE,
        [
            "docs/audits/contextual_composition_pause_checkpoint_20260731.md",
            "docs/RESUME_CONTEXTUAL_COMPOSITION.md",
            "Exact next task",
        ],
    )
    check_required_strings(
        roadmap,
        ROADMAP,
        [
            "current_phase: CC5",
            "current_status: IN PROGRESS",
            "audits/contextual_composition_pause_checkpoint_20260731.md",
            "RESUME_CONTEXTUAL_COMPOSITION.md",
            "audits/contextual_composition_query7_final_pause_readiness_20260731.md",
            "architecture/contextual_composition_primitives.md",
        ],
    )
    check_required_strings(
        branch_marker,
        BRANCH_MARKER,
        [
            "Pause Checkpoint",
            "Resume Guide",
            "Do not start a third CC4b build or CC5 retry",
        ],
    )
    check_required_strings(
        resume_doc,
        RESUME_DOC,
        [
            "Authoritative branch: `contextual-compositional-heuristics-20260731`",
            "Query 6 checkpoint SHA: `f6b4be9dc15fc4f13286f23b5aae39f48fbd01fb`",
            "Current phase: `CC5 - Contextual composition predictor`",
            "read this one for current status",
            "GitHub issue #5",
            "python scripts/check_contextual_composition_status.py --resume-readiness",
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
    check_required_strings(
        query7_report,
        QUERY7_REPORT,
        [
            "# Contextual Composition Query 7 Final Pause Readiness - 2026-07-31",
            "No CC2 primitives",
            "python scripts/check_contextual_composition_status.py --resume-readiness",
            "Define the canonical primitive interface for ranking, admission, placement, batching, and resource guards",
            "ready to resume at CC2 without another audit",
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


def check_no_cc2_in_progress() -> None:
    # CC2-anchored patterns only -- a bare "IN PROGRESS" is no longer an
    # error signal on its own now that CC5 legitimately carries that status
    # (see check_status_table's active_count invariant for the current
    # generic "exactly one active phase" guarantee instead).
    forbidden_patterns = [
        r"CC2\s+\|\s+Canonical primitive interface\s+\|\s+IN PROGRESS",
        r"CC2\s+is\s+`?IN PROGRESS`?",
        r"CC2\s+has\s+started",
    ]
    for path in CANONICAL_FILES:
        text = read(path)
        for pattern in forbidden_patterns:
            if re.search(pattern, text):
                fail(f"{path.relative_to(ROOT)} says CC2 is already in progress")


def check_no_cc6_active() -> None:
    """CC6 must never be marked active (NEXT/IN PROGRESS/COMPLETE) while
    CC5's own decision gate is unresolved -- complements check_status_table's
    table-row check by also catching stray prose claims outside the table."""
    forbidden_patterns = [
        r"Current phase:\s*`CC6\b",
        r"current_phase:\s*CC6\b",
        r"CC6\s+is\s+`?NEXT`?",
        r"CC6\s+is\s+`?IN PROGRESS`?",
        r"CC6\s+is\s+`?COMPLETE`?",
        r"CC6\s+has\s+started",
    ]
    for path in CANONICAL_FILES:
        text = read(path)
        for pattern in forbidden_patterns:
            if re.search(pattern, text):
                fail(f"{path.relative_to(ROOT)} marks CC6 active while CC5 is unresolved")


def check_start_here_and_resume_name_same_current_task(start_here: str, resume_doc: str) -> None:
    needle = "CC5 - Contextual composition predictor"
    if needle not in start_here:
        fail(f"{START_HERE.relative_to(ROOT)} does not name the current task ({needle!r})")
    if needle not in resume_doc:
        fail(f"{RESUME_DOC.relative_to(ROOT)} does not name the current task ({needle!r})")


def check_future_work_labeled(roadmap: str) -> None:
    check_required_strings(
        roadmap,
        ROADMAP,
        [
            "## Future Research Directions -- Not Yet Implemented",
            "None of them exist in this repository today",
        ],
    )


def check_active_issue_referenced() -> None:
    for path in CANONICAL_FILES:
        text = read(path)
        if "issues/5" not in text and "#5" not in text:
            fail(f"{path.relative_to(ROOT)} does not reference the active issue (#5)")


def check_cc4b_retry_linked() -> None:
    for path in CANONICAL_FILES:
        text = read(path)
        if "cc4b_cc5_retry" not in text and "cc4b_oracle_composition_expansion" not in text:
            fail(f"{path.relative_to(ROOT)} does not link the current CC4b/CC5 retry work")


def check_no_stale_final_cc5_verdict_claim() -> None:
    """Guards against a canonical doc reverting to describing the first,
    INCONCLUSIVE CC5 attempt as if it were the final verdict."""
    forbidden = "CC5 remains the roadmap's `NEXT` phase to return to"
    for path in CANONICAL_FILES:
        text = read(path)
        if forbidden in text:
            fail(f"{path.relative_to(ROOT)} still describes the first CC5 result as final (stale phrase found)")


def check_resume_readiness_extra() -> None:
    branch = run_git(["branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        fail(f"expected branch {EXPECTED_BRANCH}, found {branch!r}")

    upstream = run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream != EXPECTED_UPSTREAM:
        fail(f"expected upstream {EXPECTED_UPSTREAM}, found {upstream!r}")

    status = run_git(["status", "--porcelain"])
    if status:
        fail("working tree is not clean")

    ahead_behind = run_git(["rev-list", "--left-right", "--count", "@{u}...HEAD"])
    if ahead_behind != "0\t0":
        fail(f"expected 0 ahead / 0 behind, found {ahead_behind!r}")

    roadmap = read(ROADMAP)
    resume_doc = read(RESUME_DOC)
    pause_checkpoint = read(PAUSE_CHECKPOINT)
    start_here = read(START_HERE)
    decisions = read(DECISIONS)
    branch_marker = read(BRANCH_MARKER)

    check_required_strings(
        resume_doc,
        RESUME_DOC,
        [
            EXPECTED_BRANCH,
            "Query 6 checkpoint SHA: `f6b4be9dc15fc4f13286f23b5aae39f48fbd01fb`",
            "python scripts/check_contextual_composition_status.py --resume-readiness",
            "python -m pytest tests/test_contextual_composition_status_checker.py tests/test_cc1_composition_opportunity.py tests/test_policy_composition.py tests/test_score_and_reciprocal_rank_composition.py tests/test_primitive_interface.py tests/test_primitive_reconstructed_policies.py tests/test_contextual_composition_cc3_dsl.py tests/test_cc4_oracle_composition_dataset.py tests/test_cc5_contextual_predictor.py -q",
            "results/cc1b_composition_discriminative/query5_cc1b_full_20260731/",
            "GitHub issue #5",
        ],
    )
    check_required_strings(
        pause_checkpoint,
        PAUSE_CHECKPOINT,
        [
            "Checkpoint prepared from HEAD: `db4dcaa40abe1312ea71c40c440445172cd1c509`",
            "Query 6 checkpoint SHA: `f6b4be9dc15fc4f13286f23b5aae39f48fbd01fb`",
            "results/cc1b_composition_discriminative/query5_cc1b_full_20260731/",
            "GitHub issue #2",
        ],
    )
    check_required_strings(
        roadmap,
        ROADMAP,
        [
            "current_phase: CC5",
            "current_status: IN PROGRESS",
            "Retry in progress (started 2026-08-03)",
            "https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/2",
            "https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/3",
        ],
    )
    for path, text in [
        (START_HERE, start_here),
        (DECISIONS, decisions),
        (BRANCH_MARKER, branch_marker),
    ]:
        check_required_strings(
            text,
            path,
            [
                "CC2",
                "CC3",
                "CC4",
                "CC5",
            ],
        )
    check_no_cc2_in_progress()
    check_no_cc6_active()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    resume_readiness = False
    if argv == ["--resume-readiness"]:
        resume_readiness = True
    elif argv:
        fail("usage: check_contextual_composition_status.py [--resume-readiness]")

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
    query7_report = read(QUERY7_REPORT)

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
            "audits/contextual_composition_query7_final_pause_readiness_20260731.md",
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
            "Current phase: `CC5 - Contextual composition predictor`",
            "docs/experiments/cc1_composition_opportunity_spec.md",
            "docs/audits/contextual_composition_query4_cc1_results_20260731.md",
            "docs/audits/contextual_composition_query5_discriminativeness_review_20260731.md",
            "docs/audits/contextual_composition_pause_checkpoint_20260731.md",
            "docs/audits/contextual_composition_query7_final_pause_readiness_20260731.md",
            "docs/architecture/contextual_composition_primitives.md",
            "docs/audits/contextual_composition_cc2_primitive_interface_report_20260802.md",
            "docs/architecture/contextual_composition_dsl.md",
            "docs/audits/contextual_composition_cc3_dsl_verifier_report_20260803.md",
            "docs/audits/contextual_composition_cc4_oracle_dataset_report_20260803.md",
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
            "architecture/contextual_composition_primitives.md",
            "Do not start a third CC4b build or CC5 retry",
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
        query7_report=query7_report,
    )
    check_no_cc1_current()
    check_no_cc2_in_progress()
    check_no_cc6_active()
    check_start_here_and_resume_name_same_current_task(start_here, resume_doc)
    check_future_work_labeled(roadmap)
    check_active_issue_referenced()
    check_cc4b_retry_linked()
    check_no_stale_final_cc5_verdict_claim()

    if resume_readiness:
        check_resume_readiness_extra()
        print("contextual composition resume-readiness check passed")
    else:
        print("contextual composition status check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
