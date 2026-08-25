"""Explicit failure modes for the vLLM-LTR adapter.

Every failure here is a deliberate, typed rejection rather than a silent
fallback to a hand-written heuristic -- see the task's explicit instruction
not to replace the official ranker with a regression model or hand-written
score.
"""
from __future__ import annotations


class VLLMLTRAdapterError(Exception):
    """Base class for all vLLM-LTR adapter errors."""


class MissingDependencyError(VLLMLTRAdapterError):
    """Raised when ``torch``/``transformers`` are not installed.

    The core simulator (``pyproject.toml``) has no hard dependency on either
    package -- this adapter treats them as optional, exactly like the
    existing ``datasets``/``selector`` extras.
    """


class MissingCheckpointError(VLLMLTRAdapterError):
    """Raised when the requested checkpoint path/repo id has no weights."""


class StaleArtifactError(VLLMLTRAdapterError):
    """Raised when a checkpoint directory has no provenance record at all,
    or its recorded pinned commit does not match this adapter's pinned
    commit (``adapter.provenance.PINNED_COMMIT``)."""


class VersionMismatchError(VLLMLTRAdapterError):
    """Raised when a checkpoint's recorded torch/transformers versions (in
    its ``vllm_ltr_provenance.json`` sidecar) do not match the versions
    actually installed at load time."""


class MissingScoreError(VLLMLTRAdapterError):
    """Raised by the simulator policy wrapper when a request in the current
    step has no precomputed LTR score. There is no fallback heuristic: a
    request without a score cannot be ranked by this policy."""
