#!/usr/bin/env python3
"""Check that this repository's current documentation is internally consistent.

Deliberately lightweight: this is a handful of string/existence checks, not
a general documentation framework. It complements, and does not replace,
``scripts/check_contextual_composition_status.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_RESUME = ROOT / "docs" / "current" / "RESUME_HERE.md"
PROJECT_MAP = ROOT / "docs" / "PROJECT_MAP.md"
WORK_STATUS = ROOT / "docs" / "current" / "WORK_STATUS.md"
NEXT_ACTIONS = ROOT / "docs" / "current" / "NEXT_ACTIONS.md"
BASELINE_STATUS = ROOT / "docs" / "BASELINE_STATUS.md"
PHASE_G_AUDIT = ROOT / "docs" / "audits" / "apt_serve_phase_g_analysis_20260809.md"

REQUIRED_CURRENT_DOCS = [
    CANONICAL_RESUME,
    PROJECT_MAP,
    ROOT / "docs" / "current" / "PROJECT_MAP.md",
    WORK_STATUS,
    NEXT_ACTIONS,
    ROOT / "docs" / "current" / "SCIENTIFIC_DECISIONS.md",
    BASELINE_STATUS,
    PHASE_G_AUDIT,
]

# Files that must point a reader at the canonical resume doc, rather than
# competing with it. Path is relative to ROOT; link_text is any substring
# that proves the pointer exists (kept loose on purpose -- relative link
# depth differs by file location).
ENTRY_POINTS_REQUIRING_CANONICAL_LINK = [
    ROOT / "README.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "current" / "README.md",
]

REQUIRED_CLAIMS = [
    (CANONICAL_RESUME, "COMPLETE_REGIME_SPECIFIC"),
    (CANONICAL_RESUME, "Posthoc analysis: complete with wrapper `exit_code=0`"),
    (CANONICAL_RESUME, "post-Phase-G module-envelope interpretation"),
    (CANONICAL_RESUME, "1182306"),
    (CANONICAL_RESUME, "USEFUL_DIAGNOSTIC_ONLY"),
    (CANONICAL_RESUME, "Family A v2"),
    (CANONICAL_RESUME, "unweighted SLO-success"),
    (PROJECT_MAP, "Documentation Authority"),
    (PROJECT_MAP, "Return from Apt-Serve-specific collection to broader library-envelope"),
    (PROJECT_MAP, "1182306"),
    (PROJECT_MAP, "Family A v2"),
    (WORK_STATUS, "Apt-Serve Phase G analysis"),
    (WORK_STATUS, "REDESIGN_REQUIRED"),
    (NEXT_ACTIONS, "post-Phase-G module-envelope interpretation"),
    (NEXT_ACTIONS, "Family A v2"),
    (BASELINE_STATUS, "Positive marginal portfolio contribution; no global superiority claim"),
    (PHASE_G_AUDIT, "Not Yet Established"),
]

FORBIDDEN_CLAIMS = [
    (CANONICAL_RESUME, "CC5 IN PROGRESS", "CC5 is finalized COMPLETE_REGIME_SPECIFIC"),
    (CANONICAL_RESUME, "CC6 has started", "CC6 is not started"),
    (CANONICAL_RESUME, "CC6 is COMPLETE", "CC6 is not started"),
    (CANONICAL_RESUME, "Design and execute a targeted missing-mechanism pilot for Family A", "Family A pilot Job 1182306 already executed"),
    (CANONICAL_RESUME, "ANALYSIS PENDING", "Family A v1 scientific analysis is complete"),
    (NEXT_ACTIONS, "Draft the design and configuration for **Family A", "Family A generator+pilot already exist"),
    (NEXT_ACTIONS, "Analyze Family A Fairness and Starvation Pilot", "Family A v1 analysis is complete; next is v2"),
    (BASELINE_STATUS, "Phase G reached only 9.5%", "Phase G collection is complete"),
    (BASELINE_STATUS, "sweep has not yet been relaunched", "Phase G collection was resumed and completed"),
    (BASELINE_STATUS, "Relaunch the Phase G sweep", "the next task is post-Phase-G interpretation"),
]

LIVE_STATUS_DOCS = [
    ROOT / "README.md",
    PROJECT_MAP,
    CANONICAL_RESUME,
    WORK_STATUS,
    NEXT_ACTIONS,
    BASELINE_STATUS,
]

STALE_PHASE_G_TOKENS = [
    "PHASE_G_SS15_FIXED_RESUME_PENDING",
    "Phase G UNSTARTED",
    "Phase G `UNSTARTED`",
    "do not start Phase G",
    "Do not start Phase G",
    "resume the Phase G sweep",
    "Resume the Phase G sweep",
    "sweep has not yet been relaunched",
    "Phase G reached only 9.5%",
    "Phase F work is uncommitted",
    "e413ba1dcbe8b79f0ebc0f7511e846481548b6bb",
    "891881281b650f549b0bbebaa49df8182e535ba8",
]


def check_required_docs_exist() -> list[str]:
    errors = []
    for path in REQUIRED_CURRENT_DOCS:
        if not path.is_file():
            errors.append(f"missing required current document: {path.relative_to(ROOT)}")
    return errors


def check_canonical_links() -> list[str]:
    errors = []
    for path in ENTRY_POINTS_REQUIRING_CANONICAL_LINK:
        if not path.is_file():
            errors.append(f"missing entry-point file that should link to RESUME_HERE.md: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        if "RESUME_HERE.md" not in text:
            errors.append(
                f"{path.relative_to(ROOT)} does not link to the canonical entry point "
                f"(docs/current/RESUME_HERE.md) -- it should point readers there, not "
                f"present itself as a competing entry point"
            )
    return errors


def check_forbidden_claims() -> list[str]:
    errors = []
    for path, forbidden, reason in FORBIDDEN_CLAIMS:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            errors.append(
                f"{path.relative_to(ROOT)} contains a stale/contradictory claim: "
                f"{forbidden!r} ({reason})"
            )
    return errors


def check_no_stale_phase_g_tokens() -> list[str]:
    errors = []
    for path in LIVE_STATUS_DOCS:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in STALE_PHASE_G_TOKENS:
            if token in text:
                errors.append(f"{path.relative_to(ROOT)} contains stale Phase G token: {token!r}")
    return errors


def check_resume_and_next_action_agree() -> list[str]:
    phrase = "post-Phase-G module-envelope interpretation"
    errors = []
    for path in [CANONICAL_RESUME, NEXT_ACTIONS, WORK_STATUS]:
        if path.is_file() and phrase not in path.read_text(encoding="utf-8"):
            errors.append(f"{path.relative_to(ROOT)} does not name the current next action: {phrase}")
    return errors


def check_required_claims() -> list[str]:
    errors = []
    for path, required in REQUIRED_CLAIMS:
        if not path.is_file():
            errors.append(f"cannot check required claim -- missing file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        if required not in text:
            errors.append(
                f"{path.relative_to(ROOT)} is missing an expected current-status string: "
                f"{required!r}"
            )
    return errors


def check_single_canonical_resume_doc() -> list[str]:
    """Guard against a future query adding a second competing resume file."""
    errors = []
    candidates = [
        p
        for p in (ROOT / "docs").rglob("RESUME_HERE*.md")
        if "worktrees" not in p.parts and ".claude" not in p.parts
    ]
    if len(candidates) != 1:
        errors.append(
            "expected exactly one canonical RESUME_HERE*.md under docs/, found: "
            + ", ".join(str(p.relative_to(ROOT)) for p in candidates)
        )
    return errors


def main() -> int:
    errors: list[str] = []
    errors += check_required_docs_exist()
    errors += check_canonical_links()
    errors += check_forbidden_claims()
    errors += check_no_stale_phase_g_tokens()
    errors += check_required_claims()
    errors += check_resume_and_next_action_agree()
    errors += check_single_canonical_resume_doc()

    if errors:
        print("project handoff consistency check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("project handoff consistency check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
