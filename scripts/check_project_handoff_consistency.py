#!/usr/bin/env python3
"""Check that this repository's pause/resume documentation is internally
consistent: the canonical entry point exists and is linked from every
competing "start here" document, and no live document contradicts the
current CC5/CC6/baseline status.

Deliberately lightweight: this is a handful of string/existence checks, not
a general documentation framework. It complements, and does not replace,
``scripts/check_contextual_composition_status.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_RESUME = ROOT / "docs" / "current" / "RESUME_HERE.md"

REQUIRED_CURRENT_DOCS = [
    CANONICAL_RESUME,
    ROOT / "docs" / "current" / "PROJECT_MAP.md",
    ROOT / "docs" / "current" / "WORK_STATUS.md",
    ROOT / "docs" / "current" / "NEXT_ACTIONS.md",
    ROOT / "docs" / "current" / "SCIENTIFIC_DECISIONS.md",
    ROOT / "docs" / "BASELINE_STATUS.md",
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

# (file, forbidden substring, reason) -- each is a stale/contradictory claim
# that must not appear in a live status document. Historical, explicitly
# dated audit docs under docs/audits/ are intentionally NOT checked here --
# they are point-in-time snapshots, not living status.
FORBIDDEN_CLAIMS = [
    (ROOT / "docs" / "current" / "RESUME_HERE.md", "CC5 IN PROGRESS", "CC5 is finalized COMPLETE_REGIME_SPECIFIC"),
    (ROOT / "docs" / "current" / "RESUME_HERE.md", "CC5 is `IN PROGRESS`", "CC5 is finalized COMPLETE_REGIME_SPECIFIC"),
    (ROOT / "docs" / "BASELINE_STATUS.md", "**Llumnix** | — | — | — | Not integrated", "Llumnix row must not read as unimplemented"),
    (ROOT / "docs" / "BASELINE_STATUS.md", "**Apt-Serve** | — | — | — | Not integrated | N/A | N/A | Not implemented | N/A | N/A | N/A | N/A | N/A | Not prioritized", "Apt-Serve row must not read as merely 'not prioritized'"),
    (ROOT / "docs" / "current" / "RESUME_HERE.md", "CC6 has started", "CC6 is queued/restricted, not started"),
    (ROOT / "docs" / "current" / "RESUME_HERE.md", "CC6 is COMPLETE", "CC6 has not begun"),
    (ROOT / "docs" / "current" / "RESUME_HERE.md", "CC6 is `IN PROGRESS`", "CC6 has not begun"),
]

# (file, required substring) -- current status facts that MUST be present,
# to catch silent regressions (e.g. a future edit removing the correction).
REQUIRED_CLAIMS = [
    (CANONICAL_RESUME, "COMPLETE_REGIME_SPECIFIC"),
    (CANONICAL_RESUME, "restricted"),
    (ROOT / "docs" / "BASELINE_STATUS.md", "llumnix_faithful"),
    (ROOT / "docs" / "BASELINE_STATUS.md", "REMOTE_STATE_UNVERIFIED"),
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
    errors += check_required_claims()
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
