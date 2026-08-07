"""apt_serve_faithful: Interface scaffolding, configuration schemas, typed contracts,
and JSON-based versioned IPC schemas for Apt-Serve's upcoming implementation.

This is a Phase A pure scaffolding and contract definition. No allocation
logic or subprocess runner is implemented in this phase.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol, Set, Tuple, Union, Any

from .base import BasePolicy
from ..core.action import Action
from ..core.types import GPUConfig, ObservableRequest, ObservableState


# ======================================================================
# 1. TYPED CACHE INTERFACES (Step 5)
# ======================================================================

class CacheTier(str, Enum):
    KV = "kv"
    HIDDEN = "hidden"
    NONE = "none"


class CacheRepresentation(str, Enum):
    KV_BLOCKED = "kv_blocked"
    COMPRESSED_HIDDEN = "compressed_hidden"


@dataclass(frozen=True)
class CacheAssignment:
    request_id: int
    target_tier: CacheTier
    required_units: int
    current_tier: CacheTier
    reason: str
    scheduler_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CacheTransitionKind(str, Enum):
    KV_TO_HIDDEN = "kv_to_hidden"
    HIDDEN_TO_KV = "hidden_to_kv"
    EVICT_FULL = "evict_full"


@dataclass(frozen=True)
class CacheTransitionRequest:
    request_id: int
    transition_kind: CacheTransitionKind
    source_tier: CacheTier
    destination_tier: CacheTier


@dataclass(frozen=True)
class CacheTransitionResult:
    request_id: int
    source_tier: CacheTier
    destination_tier: CacheTier
    transition_kind: CacheTransitionKind
    expected_delay: float
    recomputation_required: bool
    success: bool
    error_message: Optional[str] = None


@dataclass(frozen=True)
class CacheCapacitySnapshot:
    tier: CacheTier
    total_capacity_blocks: int
    used_blocks: int
    free_blocks: int


@dataclass(frozen=True)
class HybridCacheSnapshot:
    step: int
    timestamp: float
    kv_snapshot: CacheCapacitySnapshot
    hidden_snapshot: CacheCapacitySnapshot
    resident_request_ids: List[int] = field(default_factory=list) # sorted for determinism

    def __post_init__(self) -> None:
        if self.resident_request_ids != sorted(self.resident_request_ids):
            raise ValueError("resident_request_ids must be sorted deterministically")


@dataclass(frozen=True)
class AptServeRequestView:
    request_id: int
    waiting_duration: float
    running_duration: float
    ttft_slo: float
    tbt_slo: float
    current_cache_tier: CacheTier
    kv_blocks_needed: int
    hidden_blocks_needed: int
    recomputation_cost_model: str
    priority: float
    slo_violation_state: bool


@dataclass(frozen=True)
class AptServeSchedulerDecision:
    selected_request_ids: List[int]
    cache_assignments: Dict[int, CacheTier]
    evictions: List[int]
    deprioritized_requests: List[int]
    value_scores: Dict[int, float]
    schema_version: int = 1


# ======================================================================
# 2. ADAPTER ERROR HIERARCHY & CONTRACTS (Step 6)
# ======================================================================

class AptServeAdapterError(Exception):
    """Base exception for all Apt-Serve adapter errors."""
    pass


class AptServeSourceCheckoutMissing(AptServeAdapterError):
    """Raised when the official Apt-Serve checkout cannot be found."""
    pass


class AptServeWrongCommit(AptServeAdapterError):
    """Raised when the checkout commit does not match the pinned commit."""
    pass


class AptServeSourceHashMismatch(AptServeAdapterError):
    """Raised when the source code file hashes do not match known provenance."""
    pass


class AptServeEnvironmentMissing(AptServeAdapterError):
    """Raised when the pinned Python 3.11 environment is not available."""
    pass


class AptServeProtocolMismatch(AptServeAdapterError):
    """Raised when the subprocess IPC schema version does not match."""
    pass


class AptServeSubprocessTimeout(AptServeAdapterError):
    """Raised when the external scheduler subprocess times out."""
    pass


class AptServeMalformedResponse(AptServeAdapterError):
    """Raised when the subprocess stdout contains invalid JSON."""
    pass


class AptServeInvalidSchedulerDecision(AptServeAdapterError):
    """Raised when the returned decision is invalid or mathematically impossible."""
    pass


class AptServeCapacityViolation(AptServeAdapterError):
    """Raised when the scheduler tries to allocate beyond physical bounds."""
    pass


class AptServeUnsupportedConfiguration(AptServeAdapterError):
    """Raised when the given configuration is not supported by the adapter."""
    pass


@dataclass(frozen=True)
class AptServeAdapterConfig:
    checkout_path: str
    conda_env_name: str = "apt-serve"
    subprocess_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class AptServeEnvironmentSpec:
    required_python_version: str = "3.11"
    required_torch_version: str = "2.3.0"
    required_vllm_version: str = "0.5.0.post1"


@dataclass(frozen=True)
class AptServeSourceProvenance:
    official_repo_url: str = "https://github.com/eddiegaoo/Apt-Serve"
    pinned_commit: str = "c953217988274a761da35cf06c01033b18dadf68"
    schema_version: int = 1


class AptServeSchedulerClient(Protocol):
    """Protocol defining the interface for the upcoming subprocess scheduler client."""
    def initialize(self, config: AptServeAdapterConfig) -> None:
        """Verify checkout, environment, and launch the subprocess."""
        ...

    def schedule_step(self, state_input: AptServeSchedulerInput) -> AptServeSchedulerDecision:
        """Serialize state, run subprocess, and parse returned decision."""
        ...

    def terminate(self) -> None:
        """Terminate the subprocess cleanly."""
        ...


# ======================================================================
# 3. VERSIONED IPC SCHEMAS (Step 7)
# ======================================================================

@dataclass(frozen=True)
class AptServeSchedulerInput:
    schema_version: int
    request_id: int
    simulator_step: int
    timestamp: float
    gpus: List[Dict[str, Any]]
    waiting_requests: List[Dict[str, Any]]
    running_requests: List[Dict[str, Any]]
    cache_snapshot: Dict[str, Any]

    def serialize_json(self) -> bytes:
        """Return a deterministic JSON byte string (sorted keys)."""
        d = asdict(self)
        return json.dumps(d, sort_keys=True).encode("utf-8")

    @classmethod
    def deserialize_json(cls, data: bytes) -> AptServeSchedulerInput:
        payload = json.loads(data.decode("utf-8"))
        if payload.get("schema_version") != 1:
            raise AptServeProtocolMismatch(f"Expected schema_version=1, got {payload.get('schema_version')}")
        return cls(
            schema_version=payload["schema_version"],
            request_id=payload["request_id"],
            simulator_step=payload["simulator_step"],
            timestamp=payload["timestamp"],
            gpus=payload["gpus"],
            waiting_requests=payload["waiting_requests"],
            running_requests=payload["running_requests"],
            cache_snapshot=payload["cache_snapshot"]
        )


@dataclass(frozen=True)
class AptServeSchedulerOutput:
    schema_version: int
    request_id: int
    selected_request_ids: List[int]
    cache_assignments: Dict[str, str] # mapped request_id as str -> CacheTier name as str
    evictions: List[int]
    deprioritized_requests: List[int]
    value_scores: Dict[str, float] # mapped request_id as str -> float

    def serialize_json(self) -> bytes:
        d = asdict(self)
        return json.dumps(d, sort_keys=True).encode("utf-8")

    @classmethod
    def deserialize_json(cls, data: bytes) -> AptServeSchedulerOutput:
        payload = json.loads(data.decode("utf-8"))
        if payload.get("schema_version") != 1:
            raise AptServeProtocolMismatch(f"Expected schema_version=1, got {payload.get('schema_version')}")
        return cls(
            schema_version=payload["schema_version"],
            request_id=payload["request_id"],
            selected_request_ids=payload["selected_request_ids"],
            cache_assignments=payload["cache_assignments"],
            evictions=payload["evictions"],
            deprioritized_requests=payload["deprioritized_requests"],
            value_scores=payload["value_scores"]
        )


# ======================================================================
# 4. PLACEHOLDER SCHEDULER POLICY (Step 5)
# ======================================================================

class AptServeSchedulerPolicy(BasePolicy):
    """Placeholder policy scaffolding for Apt-Serve's upcoming subprocess runner.

    Raises NotImplementedError on execution in Phase A.
    """
    name = "apt_serve_faithful"

    def __init__(self, adapter_config: Optional[AptServeAdapterConfig] = None) -> None:
        self.adapter_config = adapter_config
        self.provenance = AptServeSourceProvenance()

    def select_action(self, state: ObservableState) -> Action:
        raise NotImplementedError(
            "Apt-Serve baseline execution is NOT IMPLEMENTED in Phase A scaffolding "
            "(see docs/design/apt_serve_simulator_architecture_20260806.md). "
            "Only configuration schemas, validation rules, and IPC boundaries exist."
        )
