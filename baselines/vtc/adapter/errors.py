"""Explicit failure modes for the VTC adapter.

Every failure here is a deliberate, typed rejection rather than a silent
fallback to a hand-written heuristic -- mirrors
``baselines/pars/adapter/errors.py``'s convention exactly.
"""
from __future__ import annotations


class VTCAdapterError(Exception):
    """Base class for all VTC adapter errors."""


class MissingOfficialCloneError(VTCAdapterError):
    """Raised when the pinned Ying1123/VTC-artifact clone is not found at
    the expected local path. This adapter never vendors the official
    repo's source into this project's git history (see PROVENANCE.md) --
    it dynamically imports ``VTCReqQueue``/``ReqQueue``/``Req``/``Batch``/
    ``SamplingParams`` from the clone at runtime, so the clone must exist
    locally to use this adapter at all."""


class StaleCloneCommitError(VTCAdapterError):
    """Raised when the local clone's HEAD commit does not match this
    adapter's pinned commit (``adapter.provenance.PINNED_COMMIT``)."""


class UnsupportedTopologyError(VTCAdapterError):
    """Raised when the simulator state has more than one GPU. The official
    VTC artifact is a single-server scheduler with no multi-GPU/shared-pool
    story of its own -- this adapter deliberately does not invent one, and
    refuses rather than silently guessing how VTC's single `served` counter
    space should be split or shared across independent GPUs."""


class UnsupportedCostFunctionError(VTCAdapterError):
    """Raised when a cost function other than "linear" is requested. The
    official "profile" cost function is a regression fit to the authors'
    own A10G + Llama-2-7B hardware and is not portable to this simulator's
    timing model (see PROVENANCE.md) -- there is no meaningful fallback."""


class MissingTenantIdError(VTCAdapterError):
    """Raised when a request in the current step has no tenant id under
    the configured tenant-mapping function. There is no fallback tenant:
    VTC's entire mechanism is per-tenant accounting, so an unattributable
    request cannot be scheduled by this policy."""


class UnregisteredTenantError(VTCAdapterError):
    """Raised when a request's tenant id was not included in the
    ``known_tenants`` set the policy was constructed with. The official
    ``VTCReqQueue.__init__`` only populates its ``fairw`` (per-tenant
    weight) dict for the ``adapter_dirs`` list given at construction time
    -- an adapter_dir/tenant encountered later that was not in that list
    causes the official, unmodified code to raise ``KeyError`` deep inside
    ``generate_new_batch``/``update_counter``. This adapter requires the
    full tenant set to be known upfront (matching the official system's
    own pre-registered-LoRA-adapter design) and fails fast with a clear
    error here instead of surfacing that KeyError."""
