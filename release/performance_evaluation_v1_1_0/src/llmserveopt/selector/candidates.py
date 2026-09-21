"""
Selector candidate policy set.

All online-deployable registered policies. Oracle policies are explicitly excluded.
This is the single source of truth — do not hardcode candidate names elsewhere.
"""
from __future__ import annotations

from typing import List

from ..policies.registry import BASELINE_NAMES, ORACLE_POLICY_NAMES

# Verified at import time: oracle names must not appear in candidates.
_excluded = set(ORACLE_POLICY_NAMES)
SELECTOR_CANDIDATES: List[str] = [n for n in BASELINE_NAMES if n not in _excluded]
SELECTOR_CANDIDATE_COUNT: int = len(SELECTOR_CANDIDATES)

# Sanity check executed at import, not just at test time.
for _name in ORACLE_POLICY_NAMES:
    assert _name not in SELECTOR_CANDIDATES, (
        f"Oracle policy '{_name}' leaked into SELECTOR_CANDIDATES — fix registry."
    )

assert SELECTOR_CANDIDATE_COUNT > 0, "SELECTOR_CANDIDATES must not be empty."

# Alias for backwards compatibility and explicit naming conventions.
SELECTOR_CANDIDATE_POLICIES: List[str] = SELECTOR_CANDIDATES
