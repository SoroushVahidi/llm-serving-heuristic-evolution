"""apt_serve_faithful: Interface scaffolding, configuration schemas, typed contracts,
and JSON-based versioned IPC schemas for Apt-Serve's upcoming implementation.

This is a Phase C implementation of the official scheduler subprocess adapter.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
import hashlib
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
    python_executable: Optional[str] = None
    execution_mode: str = "official" # "official", "test", or "recorded_trace"


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


APT_SERVE_EXPECTED_HASHES = {
    "additional_designs/aptserve_block.py": "771d3590abfef2e6fc3a71a37bce231c276bade4188c0eadd12bf48d642980c5",
    "additional_designs/aptserve_sequence.py": "e50a546a267c832256eaa554e74cecd8e3e50ef8cd737e4e4ba8647c9943ac52",
    "additional_designs/core/aptserve_block_manager.py": "8ec4fed8417227f2bdd40c695b9c4bbe3d7164272f771200e72aeef9ad552943",
    "additional_designs/core/aptserve_interfaces.py": "703009b951c1bf61e34c6a1bc92123334ea6568b6b668d2f966192df76e036d1",
    "additional_designs/core/aptserve_scheduler.py": "b381415aafeb46d8cdad3598a00983dfad7a1c9ff992b0a3a7708596b979b02e"
}


class AptServeSubprocessClient:
    """Subprocess client running Apt-Serve scheduler in isolated python 3.11 environment."""
    def __init__(self, config: AptServeAdapterConfig) -> None:
        self.config = config
        self.provenance = AptServeSourceProvenance()
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> AptServeSubprocessClient:
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.terminate()

    def initialize(self) -> None:
        if self.config.execution_mode == "official":
            # 1. Verify checkout path exists
            if not self.config.checkout_path or not os.path.exists(self.config.checkout_path):
                raise AptServeSourceCheckoutMissing(
                    f"Official Apt-Serve checkout directory missing: {self.config.checkout_path}"
                )

            # 2. Run git verification
            git_dir = os.path.join(self.config.checkout_path, ".git")
            if not os.path.exists(git_dir):
                raise AptServeSourceCheckoutMissing(
                    f"Apt-Serve checkout path is not a git repository: {self.config.checkout_path}"
                )

            try:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.config.checkout_path,
                    text=True,
                    stderr=subprocess.DEVNULL
                ).strip()
            except subprocess.CalledProcessError as e:
                raise AptServeSourceCheckoutMissing(f"Failed to check git commit on {self.config.checkout_path}: {e}")

            if commit != self.provenance.pinned_commit:
                raise AptServeWrongCommit(
                    f"Official Apt-Serve commit mismatch: expected {self.provenance.pinned_commit}, got {commit}"
                )

            # 3. Verify file hashes
            for rel_path, expected_sha in APT_SERVE_EXPECTED_HASHES.items():
                abs_path = os.path.join(self.config.checkout_path, rel_path)
                if not os.path.exists(abs_path):
                    raise AptServeSourceHashMismatch(f"Expected source file missing: {rel_path}")
                h = hashlib.sha256()
                with open(abs_path, "rb") as f:
                    h.update(f.read())
                actual_sha = h.hexdigest()
                if actual_sha != expected_sha:
                    raise AptServeSourceHashMismatch(
                        f"Source file hash mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"
                    )

            # 4. Verify external python environment
            py_exe = self.config.python_executable or "python3"
            env_check_code = """
import sys
if sys.version_info[:2] != (3, 11):
    sys.exit(11)
try:
    import torch
    import vllm
    import xformers
    import vllm_flash_attn
except ImportError as e:
    print(e)
    sys.exit(12)
"""
            try:
                res = subprocess.run(
                    [py_exe, "-c", env_check_code],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if res.returncode == 11:
                    raise AptServeEnvironmentMissing(f"Python 3.11 required, but {py_exe} version is different.")
                elif res.returncode == 12:
                    raise AptServeEnvironmentMissing(f"Missing required imports in 3.11 environment: {res.stdout.strip()}")
                elif res.returncode != 0:
                    raise AptServeEnvironmentMissing(f"Conda/Python environment check failed with exit code {res.returncode}: {res.stderr.strip()}")
            except FileNotFoundError:
                raise AptServeEnvironmentMissing(f"External Python executable '{py_exe}' not found.")

        # Launch the subprocess worker in corresponding mode
        policy_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(policy_dir)))

        if self.config.execution_mode == "official":
            worker_path = os.path.join(project_root, "scripts", "apt_serve", "apt_serve_scheduler_worker.py")
            py_exe = self.config.python_executable or "python3"
            cmd = [py_exe, worker_path, "--checkout", self.config.checkout_path]
        else:
            worker_path = os.path.join(project_root, "scripts", "apt_serve", "fake_scheduler_worker.py")
            py_exe = sys.executable
            cmd = [py_exe, worker_path]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
        except Exception as e:
            raise AptServeAdapterError(f"Failed to start subprocess worker: {e}")

    def schedule_step(self, state_input: AptServeSchedulerInput) -> AptServeSchedulerDecision:
        if not self.proc or self.proc.poll() is not None:
            raise AptServeAdapterError("Subprocess worker is not running.")

        # Serialize request
        payload = state_input.serialize_json().decode("utf-8") + "\n"
        if len(payload) > 10 * 1024 * 1024:
            raise AptServeUnsupportedConfiguration("Request payload size exceeds maximum limit of 10MB.")

        # Non-blocking communication with timeout
        try:
            stdout_data, stderr_data = self.proc.communicate(
                input=payload,
                timeout=self.config.subprocess_timeout_seconds
            )
        except subprocess.TimeoutExpired:
            self.terminate()
            raise AptServeSubprocessTimeout("External scheduler subprocess timed out.")
        except Exception as e:
            raise AptServeAdapterError(f"Subprocess communication failed: {e}")

        # Post-execution cleanup for one-shot worker lifecycle
        ret_code = self.proc.poll()
        if ret_code is not None and ret_code != 0:
            raise AptServeAdapterError(f"Subprocess worker exited with non-zero code {ret_code}. Stderr: {stderr_data.strip()}")

        if not stdout_data.strip():
            raise AptServeMalformedResponse("Subprocess worker returned empty response.")

        try:
            output = AptServeSchedulerOutput.deserialize_json(stdout_data.encode("utf-8"))
        except AptServeProtocolMismatch:
            raise
        except Exception as e:
            raise AptServeMalformedResponse(f"Failed to parse subprocess JSON output: {e}. Output was: {stdout_data}")

        # Validate response consistency
        # Validate that every selected ID was present in input
        input_ids = {r["request_id"] for r in state_input.waiting_requests} | {r["request_id"] for r in state_input.running_requests}
        for sid in output.selected_request_ids:
            if sid not in input_ids:
                raise AptServeInvalidSchedulerDecision(f"Selected request ID {sid} was not present in input requests.")

        return AptServeSchedulerDecision(
            selected_request_ids=output.selected_request_ids,
            cache_assignments={int(k): CacheTier(v) for k, v in output.cache_assignments.items()},
            evictions=output.evictions,
            deprioritized_requests=output.deprioritized_requests,
            value_scores={int(k): v for k, v in output.value_scores.items()}
        )

    def terminate(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None


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
