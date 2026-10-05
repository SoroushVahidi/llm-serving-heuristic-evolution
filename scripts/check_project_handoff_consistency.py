#!/usr/bin/env python3
"""Check that the repository's documentation hierarchy is internally consistent.

Deliberately lightweight: a handful of string/existence checks, not a general
documentation framework. The hierarchy it enforces (see ``docs/README.md``):

1. ``README.md`` and ``REPRODUCIBILITY.md`` -- public entry points;
2. ``docs/current/README.md`` -- the current-study documentation index;
3. study-specific documents linked from that index;
4. historical material, whose former "current"/"next action" entry points
   carry a *Historical document* banner.

Until 2026-10-04 this script enforced the August 2026 project-handoff state
(``docs/current/RESUME_HERE.md`` as the canonical entry point and a
post-Phase-G next action). Those documents are now historical records.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CURRENT_INDEX = ROOT / "docs" / "current" / "README.md"
CURRENT_INDEX_LINK = "docs/current/README.md"

REQUIRED_CURRENT_DOCS = [
    ROOT / "README.md",
    ROOT / "REPRODUCIBILITY.md",
    ROOT / "docs" / "README.md",
    CURRENT_INDEX,
    ROOT / "docs" / "current" / "PERFORMANCE_EVALUATION_REPRODUCIBILITY.md",
    ROOT / "paper" / "performance_evaluation" / "README.md",
]

# Entry points that must route a reader to the current-study index. The link
# text differs by file location, so a path suffix is enough.
ENTRY_POINTS = {
    ROOT / "README.md": CURRENT_INDEX_LINK,
    ROOT / "REPRODUCIBILITY.md": CURRENT_INDEX_LINK,
    ROOT / "docs" / "README.md": "current/README.md",
}

# Former entry points that describe an earlier project state. Each must start
# with the historical banner (checked in its first lines) so that nobody reads
# it as current status.
HISTORICAL_BANNER = "**Historical document.**"
BANNER_WINDOW_LINES = 6
HISTORICAL_ENTRY_POINTS = [
    ROOT / "docs" / "PROJECT_MAP.md",
    ROOT / "docs" / "INDEX.md",
    ROOT / "docs" / "RESULTS_INDEX.md",
    ROOT / "docs" / "current" / "RESUME_HERE.md",
    ROOT / "docs" / "current" / "WORK_STATUS.md",
    ROOT / "docs" / "current" / "NEXT_ACTIONS.md",
    ROOT / "docs" / "current" / "ACTIVE_JOBS.md",
    ROOT / "docs" / "current" / "FGCS_CURRENT_STATUS.md",
    ROOT / "docs" / "current" / "PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md",
    ROOT / "docs" / "current" / "FINAL_MANUSCRIPT_FREEZE.md",
]

# Stale framings that must not reappear in current documents.
CURRENT_DOCS_FORBIDDEN = [
    ("Target journal", "editorial framing; describe the manuscript and reproducibility package instead"),
    ("SUBMISSION_READY", "submission-time status belongs to the dated roadmap"),
    ("single authoritative source of truth", "the submission roadmap is historical"),
    ("~2,500 tests", "stale test count"),
    ("redistributed in `data/`", "derived trace windows are not committed"),
]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check_required_docs_exist() -> list[str]:
    return [f"missing required current document: {_rel(p)}" for p in REQUIRED_CURRENT_DOCS if not p.is_file()]


def check_entry_point_links() -> list[str]:
    errors = []
    for path, target in ENTRY_POINTS.items():
        if not path.is_file():
            errors.append(f"missing entry point: {_rel(path)}")
        elif target not in path.read_text(encoding="utf-8"):
            errors.append(f"{_rel(path)} does not link to {target}")
    return errors


def check_historical_banners() -> list[str]:
    errors = []
    for path in HISTORICAL_ENTRY_POINTS:
        if not path.is_file():
            errors.append(f"missing historical entry point: {_rel(path)}")
            continue
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:BANNER_WINDOW_LINES])
        if HISTORICAL_BANNER not in head:
            errors.append(f"{_rel(path)} lacks the historical banner ({HISTORICAL_BANNER!r})")
        elif CURRENT_INDEX_LINK not in head:
            errors.append(f"{_rel(path)} banner does not point to {CURRENT_INDEX_LINK}")
    return errors


def check_current_docs_not_stale() -> list[str]:
    errors = []
    for path in REQUIRED_CURRENT_DOCS:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for phrase, reason in CURRENT_DOCS_FORBIDDEN:
            if phrase in text:
                errors.append(f"{_rel(path)} contains stale phrase {phrase!r} ({reason})")
    return errors


def check_single_resume_doc() -> list[str]:
    """Guard against a second competing resume file appearing under docs/."""
    candidates = [p for p in (ROOT / "docs").rglob("RESUME_HERE*.md") if "worktrees" not in p.parts]
    if len(candidates) > 1:
        return ["more than one RESUME_HERE*.md under docs/: " + ", ".join(_rel(p) for p in candidates)]
    return []


def main() -> int:
    errors: list[str] = []
    errors += check_required_docs_exist()
    errors += check_entry_point_links()
    errors += check_historical_banners()
    errors += check_current_docs_not_stale()
    errors += check_single_resume_doc()

    if errors:
        print("project documentation consistency check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("project handoff consistency check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
