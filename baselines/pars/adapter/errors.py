"""Explicit failure modes for the PARS adapter.

Every failure here is a deliberate, typed rejection rather than a silent
fallback to a hand-written heuristic -- mirrors
``baselines/vllm_ltr/adapter/errors.py``'s convention exactly.
"""
from __future__ import annotations


class PARSAdapterError(Exception):
    """Base class for all PARS adapter errors."""


class MissingDependencyError(PARSAdapterError):
    """Raised when ``torch``/``transformers`` are not installed."""


class MissingOfficialCloneError(PARSAdapterError):
    """Raised when the pinned SPEAR-UIC/PARS clone is not found at the
    expected local path. This adapter never vendors the official repo's
    source into this project's git history (see PROVENANCE.md's license
    section) -- it dynamically imports ``PairwiseRanker`` from the clone at
    runtime, so the clone must exist locally to use this adapter at all."""


class StaleCloneCommitError(PARSAdapterError):
    """Raised when the local clone's HEAD commit does not match this
    adapter's pinned commit (``adapter.provenance.PINNED_COMMIT``)."""


class MissingCheckpointError(PARSAdapterError):
    """Raised when the requested trained checkpoint path has no weights."""


class MissingScoreError(PARSAdapterError):
    """Raised by the simulator policy wrapper when a request in the current
    step has no precomputed PARS score. There is no fallback heuristic: a
    request without a score cannot be ranked by this policy."""
