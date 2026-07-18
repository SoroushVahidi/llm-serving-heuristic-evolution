"""
Topology-aware registry for the five faithful/paper-reimplementation
"external baseline" policies (`vllm_faithful`, `sarathi_faithful`,
`distserve_faithful`, `tetriinfer_paper_reimplementation`,
`llumnix_faithful`).

Deliberately SEPARATE from `registry.py`'s `BASELINE_NAMES`/
`SELECTOR_CANDIDATE_NAMES` (the historical, deployable-policy registry used
by every existing experiment sweep and the trained selector) -- nothing
here is ever imported by `registry.py`, and none of these five names ever
appear in `BASELINE_NAMES`/`SELECTOR_CANDIDATE_NAMES`. This module exists
so the five external baselines can be discovered, instantiated, and
evaluated as a group WITHOUT silently changing the historical
20-deployable-policy/20-selector-candidate counts documented in
docs/research_status.md and relied upon by every existing config/sweep.

See docs/external_baseline_integration.md for the full integration matrix,
resource-normalization protocols, and selector-eligibility analysis this
metadata supports.

Why a topology_class field is unavoidable
------------------------------------------
Unlike the historical registry (every entry assumes one homogeneous,
role=None GPU pool), these five baselines assume THREE structurally
different deployments:
  - MONOLITHIC: any number of role=None GPUs, sharing one global admission
    queue (vllm_faithful, sarathi_faithful).
  - DISAGGREGATED_PREFILL_DECODE: role="prefill"/role="decode" GPUs
    (distserve_faithful, tetriinfer_paper_reimplementation).
  - MULTI_INSTANCE_MIGRATORY: N independent role=None GPUs with no shared
    admission queue, connected only by an explicit live-migration
    primitive (llumnix_faithful).
A baseline cannot be meaningfully run, let alone fairly compared to
another, without knowing which of these three topologies it assumes --
hence this metadata is required infrastructure, not decoration.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

from .base import BasePolicy
from .distserve_faithful import DistServeFaithfulPolicy
from .llumnix_faithful import LlumnixFaithfulPolicy
from .sarathi_faithful import SarathiFaithfulPolicy
from .tetriinfer_paper_reimplementation import TetriInferPaperReimplementationPolicy
from .vllm_faithful import VLLMFaithfulPolicy


class FidelityClass(str, Enum):
    #: Verified against a specific, pinned, author-maintained source-code
    #: commit (vllm_faithful, sarathi_faithful, distserve_faithful,
    #: llumnix_faithful).
    FAITHFUL = "faithful"
    #: Verified only against a paper's prose description -- no official
    #: code/artifact exists to pin (tetriinfer_paper_reimplementation).
    PAPER_REIMPLEMENTATION = "paper_reimplementation"


class TopologyClass(str, Enum):
    MONOLITHIC = "monolithic"
    DISAGGREGATED_PREFILL_DECODE = "disaggregated_prefill_decode"
    MULTI_INSTANCE_MIGRATORY = "multi_instance_migratory"


class PreemptionMode(str, Enum):
    RECOMPUTE = "recompute"
    SWAP = "swap"
    ADMISSION_AVOIDANCE = "admission_avoidance"  # e.g. reserve-static/dynamic; no eviction at all


@dataclass(frozen=True)
class ExternalBaselineSpec:
    name: str
    fidelity_class: FidelityClass
    topology_class: TopologyClass
    pinned_source: str
    reference_doc: str
    factory: Callable[..., BasePolicy]

    # --- Topology requirements ---
    min_gpu_count: int
    required_roles: Tuple[Optional[str], ...]   # e.g. (None,) or ("prefill", "decode")
    # For DISAGGREGATED_PREFILL_DECODE: (min_prefill_gpus, min_decode_gpus).
    # None for topologies without a role split.
    min_role_counts: Optional[Tuple[int, int]] = None

    # --- Mechanism requirements (see docs/external_baseline_integration.md §1) ---
    requires_kv_block_model: bool = True
    requires_disaggregation: bool = False
    requires_cross_instance_migration: bool = False
    preemption_mode: PreemptionMode = PreemptionMode.RECOMPUTE
    requires_chunked_prefill_scheduling: bool = False
    requires_length_prediction: bool = False

    # --- Selector integration status (see docs/external_baseline_integration.md §10) ---
    selector_eligible: bool = False
    historical: bool = False

    notes: str = ""


def _vllm_faithful_factory(**kwargs) -> VLLMFaithfulPolicy:
    return VLLMFaithfulPolicy(**kwargs)


def _sarathi_faithful_factory(**kwargs) -> SarathiFaithfulPolicy:
    return SarathiFaithfulPolicy(**kwargs)


def _distserve_faithful_factory(**kwargs) -> DistServeFaithfulPolicy:
    return DistServeFaithfulPolicy(**kwargs)


def _tetriinfer_factory(**kwargs) -> TetriInferPaperReimplementationPolicy:
    return TetriInferPaperReimplementationPolicy(**kwargs)


def _llumnix_faithful_factory(**kwargs) -> LlumnixFaithfulPolicy:
    return LlumnixFaithfulPolicy(**kwargs)


EXTERNAL_BASELINE_REGISTRY: dict = {
    "vllm_faithful": ExternalBaselineSpec(
        name="vllm_faithful",
        fidelity_class=FidelityClass.FAITHFUL,
        topology_class=TopologyClass.MONOLITHIC,
        pinned_source="vLLM commit 67d96c29fba9b72cb4c4edbc26211c208a00ebdd (tag v0.1.0)",
        reference_doc="docs/vllm_faithful_scheduler_reference.md",
        factory=_vllm_faithful_factory,
        min_gpu_count=1,
        required_roles=(None,),
        requires_kv_block_model=True,
        requires_disaggregation=False,
        requires_cross_instance_migration=False,
        preemption_mode=PreemptionMode.RECOMPUTE,
        requires_chunked_prefill_scheduling=False,
        requires_length_prediction=False,
        selector_eligible=False,
        historical=False,
        notes=(
            "Supports N role=None GPUs as ONE shared-queue pool (the "
            "policy's own multi-GPU extension, not part of the pinned "
            "single-engine reference -- see its own reference doc). Not "
            "the same resource-sharing model as llumnix_faithful's N "
            "INDEPENDENT instances at the same GPU count -- see "
            "docs/external_baseline_integration.md §1."
        ),
    ),
    "sarathi_faithful": ExternalBaselineSpec(
        name="sarathi_faithful",
        fidelity_class=FidelityClass.FAITHFUL,
        topology_class=TopologyClass.MONOLITHIC,
        pinned_source="microsoft/sarathi-serve branch osdi-sarathi-serve commit ceaa0660ea2487976101a8167aad5c8046e85b27",
        reference_doc="docs/sarathi_faithful_scheduler_reference.md",
        factory=_sarathi_faithful_factory,
        min_gpu_count=1,
        required_roles=(None,),
        requires_kv_block_model=True,
        requires_disaggregation=False,
        requires_cross_instance_migration=False,
        preemption_mode=PreemptionMode.RECOMPUTE,
        requires_chunked_prefill_scheduling=True,
        requires_length_prediction=False,
        selector_eligible=False,
        historical=False,
        notes="Same shared-queue multi-GPU model as vllm_faithful (reuses its KVBlockSpaceManager/preemption pattern).",
    ),
    "distserve_faithful": ExternalBaselineSpec(
        name="distserve_faithful",
        fidelity_class=FidelityClass.FAITHFUL,
        topology_class=TopologyClass.DISAGGREGATED_PREFILL_DECODE,
        pinned_source="LLMServe/DistServe branch camera-ready-simulator commit 0ec355c8743d3fbd2d02f3cd62b5be6eae368f92",
        reference_doc="docs/distserve_faithful_scheduler_reference.md",
        factory=_distserve_faithful_factory,
        min_gpu_count=2,
        required_roles=("prefill", "decode"),
        min_role_counts=(1, 1),  # EXACTLY 1+1 enforced (ValueError otherwise) -- see its own docstring
        requires_kv_block_model=True,
        requires_disaggregation=True,
        requires_cross_instance_migration=False,
        preemption_mode=PreemptionMode.SWAP,
        requires_chunked_prefill_scheduling=False,
        requires_length_prediction=False,
        selector_eligible=False,
        historical=False,
        notes=(
            "Hard single-prefill-worker/single-decode-worker requirement "
            "(raises ValueError otherwise) -- cannot scale to more decode "
            "workers, unlike tetriinfer_paper_reimplementation."
        ),
    ),
    "tetriinfer_paper_reimplementation": ExternalBaselineSpec(
        name="tetriinfer_paper_reimplementation",
        fidelity_class=FidelityClass.PAPER_REIMPLEMENTATION,
        topology_class=TopologyClass.DISAGGREGATED_PREFILL_DECODE,
        pinned_source="arXiv:2401.11181 v1 (no official code/artifact exists -- see its own reference doc §0)",
        reference_doc="docs/tetriinfer_reference.md",
        factory=_tetriinfer_factory,
        min_gpu_count=2,
        required_roles=("prefill", "decode"),
        min_role_counts=(1, 1),  # minimum viable; supports >=1 of each, unlike distserve_faithful
        requires_kv_block_model=True,
        requires_disaggregation=True,
        requires_cross_instance_migration=False,
        preemption_mode=PreemptionMode.ADMISSION_AVOIDANCE,
        requires_chunked_prefill_scheduling=True,
        requires_length_prediction=True,
        selector_eligible=False,
        historical=False,
        notes=(
            "Supports MULTIPLE decode-role GPUs (power-of-two routing) -- "
            "the only disaggregated baseline here that can scale decode "
            "workers. Comparable to distserve_faithful ONLY when "
            "constrained down to its 1+1 minimum topology; its own "
            "natural multi-decode-worker regime has no distserve_faithful "
            "analogue at all."
        ),
    ),
    "llumnix_faithful": ExternalBaselineSpec(
        name="llumnix_faithful",
        fidelity_class=FidelityClass.FAITHFUL,
        topology_class=TopologyClass.MULTI_INSTANCE_MIGRATORY,
        pinned_source="alibaba/llm-scheduling-artifact commit a90824307249573f9c7548645c22994c65f83a08 (OSDI 2024 artifact)",
        reference_doc="docs/llumnix_faithful_scheduler_reference.md",
        factory=_llumnix_faithful_factory,
        min_gpu_count=1,  # enforced minimum; migration is a structural no-op with only 1 instance
        required_roles=(None,),
        requires_kv_block_model=True,  # via composed vllm_faithful local scheduler
        requires_disaggregation=False,
        requires_cross_instance_migration=True,
        preemption_mode=PreemptionMode.RECOMPUTE,  # via composed vllm_faithful local scheduler
        requires_chunked_prefill_scheduling=False,
        requires_length_prediction=False,
        selector_eligible=False,
        historical=False,
        notes=(
            "min_gpu_count=1 is the code-enforced minimum, NOT a "
            "meaningful evaluation configuration -- migration (this "
            "baseline's entire point) cannot occur with fewer than 2 "
            "independent instances; recommend >=2 for any real "
            "evaluation. Each instance is an INDEPENDENT admission queue "
            "(no shared visibility) -- structurally different from "
            "vllm_faithful/sarathi_faithful's shared-queue multi-GPU "
            "model even at the same total GPU count."
        ),
    ),
}

EXTERNAL_BASELINE_NAMES = list(EXTERNAL_BASELINE_REGISTRY.keys())


def get_external_baseline_spec(name: str) -> ExternalBaselineSpec:
    if name not in EXTERNAL_BASELINE_REGISTRY:
        raise KeyError(
            f"Unknown external baseline '{name}'. Available: {EXTERNAL_BASELINE_NAMES}"
        )
    return EXTERNAL_BASELINE_REGISTRY[name]


def make_external_baseline(name: str, **kwargs) -> BasePolicy:
    """Instantiate an external baseline policy by name, forwarding kwargs
    to its constructor. Mirrors registry.py's make_policy() but operates
    over EXTERNAL_BASELINE_REGISTRY, never BASELINE_NAMES."""
    spec = get_external_baseline_spec(name)
    return spec.factory(**kwargs)


def external_baselines_by_topology(topology_class: TopologyClass) -> list:
    """Names of every external baseline assuming the given topology class
    -- the set of baselines that CAN be run on identical physical topology
    without violating any baseline's own structural requirements (see
    docs/external_baseline_integration.md §1 for whether they SHOULD be
    compared even when they can be co-located)."""
    return [
        name for name, spec in EXTERNAL_BASELINE_REGISTRY.items()
        if spec.topology_class == topology_class
    ]
