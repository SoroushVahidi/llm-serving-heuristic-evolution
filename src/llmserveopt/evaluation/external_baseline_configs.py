"""
Topology-compatible evaluation configs for the six external baselines
(see src/llmserveopt/policies/external_baselines_registry.py and
docs/external_baseline_integration.md).

Every builder here returns a `(gpu_configs, service_model)` pair ready to
pass to `evaluation.run_policy.run_policy` (or the harness in
`external_baseline_harness.py`). Nothing here is imported by any existing
config/sweep -- these are new, additive, and only used by the external
baseline evaluation path.

Resource accounting (see docs/external_baseline_integration.md §4 for the
full protocol discussion)
--------------------------------------------------------------------------
Every builder takes an explicit `total_kv_tokens` budget (aggregate KV
capacity across ALL GPUs it allocates) rather than a hardcoded per-GPU
value, so callers can hold the AGGREGATE resource pool constant across
different topologies (Protocol B) or simply match GPU COUNT (Protocol A)
by choosing `total_kv_tokens` accordingly. Disaggregated builders also
take an explicit `prefill_kv_fraction` (default 0.5, disclosed, NOT
paper-sourced -- no baseline's own pinned reference specifies a
prefill:decode capacity RATIO, since real deployments size this from
profiling data this project does not have) governing how the aggregate
splits between prefill and decode roles.

None of these defaults are claimed to be realistic hardware sizings; they
exist purely to make cross-baseline comparison protocols concrete and
reproducible. See docs/external_baseline_integration.md for the full
disclosure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..core.types import GPUConfig
from ..simulator.service_model import ServiceModel

# Disclosed, non-paper-sourced defaults (see module docstring).
DEFAULT_TOTAL_KV_TOKENS = 20_000
DEFAULT_MAX_ACTIVE_SEQUENCES_PER_GPU = 64
DEFAULT_MAX_BATCH_TOKENS_PER_GPU = 1_000_000  # effectively unbounded; KV capacity is the binding constraint
DEFAULT_PREFILL_KV_FRACTION = 0.5
DEFAULT_TRANSFER_DELAY = 0.005
DEFAULT_MIGRATION_DELAY = 0.005


@dataclass(frozen=True)
class TopologyDescription:
    """Machine-readable record of the physical resource allocation a
    config represents -- attached to harness results so every reported
    number can be traced back to an explicit GPU count/role split (see
    docs/external_baseline_integration.md §4's "document resource
    accounting" requirement)."""
    topology_class: str
    total_gpus: int
    num_prefill_gpus: int = 0
    num_decode_gpus: int = 0
    num_instances: int = 0
    total_kv_tokens: int = 0
    prefill_kv_tokens: int = 0
    decode_kv_tokens: int = 0


def monolithic_config(
    n_gpus: int = 1,
    total_kv_tokens: int = DEFAULT_TOTAL_KV_TOKENS,
    max_active_sequences: int = DEFAULT_MAX_ACTIVE_SEQUENCES_PER_GPU,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS_PER_GPU,
) -> Tuple[List[GPUConfig], ServiceModel, TopologyDescription]:
    """Monolithic topology (vllm_faithful, sarathi_faithful): N role=None
    GPUs sharing one global admission queue. `total_kv_tokens` is split
    evenly across `n_gpus`."""
    if n_gpus < 1:
        raise ValueError(f"n_gpus must be >= 1, got {n_gpus}")
    per_gpu_kv = total_kv_tokens // n_gpus
    gpu_configs = [
        GPUConfig(gpu_id=i, max_active_sequences=max_active_sequences,
                  max_batch_tokens=max_batch_tokens, max_kv_tokens=per_gpu_kv)
        for i in range(n_gpus)
    ]
    service_model = ServiceModel()
    topo = TopologyDescription(
        topology_class="monolithic", total_gpus=n_gpus,
        total_kv_tokens=per_gpu_kv * n_gpus,
    )
    return gpu_configs, service_model, topo


def disaggregated_config(
    n_prefill: int = 1,
    n_decode: int = 1,
    total_kv_tokens: int = DEFAULT_TOTAL_KV_TOKENS,
    prefill_kv_fraction: float = DEFAULT_PREFILL_KV_FRACTION,
    max_active_sequences: int = DEFAULT_MAX_ACTIVE_SEQUENCES_PER_GPU,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS_PER_GPU,
    transfer_delay: float = DEFAULT_TRANSFER_DELAY,
    max_prefill_chunk_tokens: int = 512,
    step_token_budget: int = 100_000,
) -> Tuple[List[GPUConfig], ServiceModel, TopologyDescription]:
    """Disaggregated prefill/decode topology (distserve_faithful requires
    exactly n_prefill=n_decode=1; tetriinfer_paper_reimplementation
    supports n_prefill>=1, n_decode>=1). `total_kv_tokens` splits between
    the prefill and decode roles per `prefill_kv_fraction`, then evenly
    across each role's own GPU count."""
    if n_prefill < 1 or n_decode < 1:
        raise ValueError(f"n_prefill and n_decode must both be >= 1, got {n_prefill}, {n_decode}")
    if not (0.0 < prefill_kv_fraction < 1.0):
        raise ValueError(f"prefill_kv_fraction must be in (0, 1), got {prefill_kv_fraction}")

    prefill_total = int(total_kv_tokens * prefill_kv_fraction)
    decode_total = total_kv_tokens - prefill_total
    per_prefill_kv = prefill_total // n_prefill
    per_decode_kv = decode_total // n_decode

    gpu_configs = [
        GPUConfig(gpu_id=i, max_active_sequences=max_active_sequences,
                  max_batch_tokens=max_batch_tokens, max_kv_tokens=per_prefill_kv, role="prefill")
        for i in range(n_prefill)
    ]
    gpu_configs += [
        GPUConfig(gpu_id=100 + i, max_active_sequences=max_active_sequences,
                  max_batch_tokens=max_batch_tokens, max_kv_tokens=per_decode_kv, role="decode")
        for i in range(n_decode)
    ]
    service_model = ServiceModel(
        enable_prefill_modeling=True, enable_disaggregation=True, decode_first=True,
        step_token_budget=step_token_budget, max_prefill_chunk_tokens=max_prefill_chunk_tokens,
        prefill_cost_per_token=1.0, migration_transfer_delay=transfer_delay,
    )
    topo = TopologyDescription(
        topology_class="disaggregated_prefill_decode",
        total_gpus=n_prefill + n_decode, num_prefill_gpus=n_prefill, num_decode_gpus=n_decode,
        total_kv_tokens=per_prefill_kv * n_prefill + per_decode_kv * n_decode,
        prefill_kv_tokens=per_prefill_kv * n_prefill, decode_kv_tokens=per_decode_kv * n_decode,
    )
    return gpu_configs, service_model, topo


def multi_instance_migratory_config(
    n_instances: int = 2,
    total_kv_tokens: int = DEFAULT_TOTAL_KV_TOKENS,
    max_active_sequences: int = DEFAULT_MAX_ACTIVE_SEQUENCES_PER_GPU,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS_PER_GPU,
    migration_delay: float = DEFAULT_MIGRATION_DELAY,
) -> Tuple[List[GPUConfig], ServiceModel, TopologyDescription]:
    """Multi-instance migratory topology (llumnix_faithful): N
    INDEPENDENT role=None GPUs (no shared admission queue -- see
    docs/external_baseline_integration.md §1 for why this is NOT the same
    resource-sharing model as monolithic_config's shared-queue pool, even
    at identical n_gpus/total_kv_tokens). n_instances=1 is accepted (the
    code-enforced minimum) but migration is then structurally a no-op --
    see llumnix_faithful's own registry notes."""
    if n_instances < 1:
        raise ValueError(f"n_instances must be >= 1, got {n_instances}")
    per_instance_kv = total_kv_tokens // n_instances
    gpu_configs = [
        GPUConfig(gpu_id=i, max_active_sequences=max_active_sequences,
                  max_batch_tokens=max_batch_tokens, max_kv_tokens=per_instance_kv)
        for i in range(n_instances)
    ]
    service_model = ServiceModel(llumnix_migration_delay=migration_delay)
    topo = TopologyDescription(
        topology_class="multi_instance_migratory", total_gpus=n_instances,
        num_instances=n_instances, total_kv_tokens=per_instance_kv * n_instances,
    )
    return gpu_configs, service_model, topo


# ---------------------------------------------------------------------------
# Per-baseline recommended configs (Protocol C: architecture-native --
# see docs/external_baseline_integration.md §4)
# ---------------------------------------------------------------------------

def native_config_for(name: str, total_kv_tokens: int = DEFAULT_TOTAL_KV_TOKENS, **kwargs):
    """Each baseline's own architecture-native topology at a given
    aggregate KV budget -- Protocol C: report resource consumption
    explicitly rather than forcing identical topology shapes."""
    if name in ("vllm_faithful", "sarathi_faithful", "vllm_chunked_prefill_faithful", "slai_faithful"):
        return monolithic_config(n_gpus=kwargs.pop("n_gpus", 1), total_kv_tokens=total_kv_tokens, **kwargs)
    if name == "distserve_faithful":
        return disaggregated_config(n_prefill=1, n_decode=1, total_kv_tokens=total_kv_tokens, **kwargs)
    if name == "tetriinfer_paper_reimplementation":
        return disaggregated_config(
            n_prefill=kwargs.pop("n_prefill", 1), n_decode=kwargs.pop("n_decode", 2),
            total_kv_tokens=total_kv_tokens, **kwargs,
        )
    if name == "llumnix_faithful":
        return multi_instance_migratory_config(
            n_instances=kwargs.pop("n_instances", 3), total_kv_tokens=total_kv_tokens, **kwargs,
        )
    raise KeyError(f"No native config builder for external baseline '{name}'")


def matched_gpu_count_configs(
    n_gpus: int, total_kv_tokens: int = DEFAULT_TOTAL_KV_TOKENS,
) -> dict:
    """Protocol A (equal total GPU count): every baseline gets exactly
    `n_gpus` total GPUs and the SAME aggregate KV budget, split according
    to each baseline's own required role structure. Disaggregated
    baselines need n_gpus>=2 (>=1 prefill + >=1 decode); distserve_faithful
    additionally requires EXACTLY n_gpus=2 (its own hard 1+1 constraint).
    Returns {baseline_name: (gpu_configs, service_model, topology)}."""
    configs = {}
    configs["vllm_faithful"] = monolithic_config(n_gpus=n_gpus, total_kv_tokens=total_kv_tokens)
    configs["sarathi_faithful"] = monolithic_config(n_gpus=n_gpus, total_kv_tokens=total_kv_tokens)
    configs["slai_faithful"] = monolithic_config(n_gpus=n_gpus, total_kv_tokens=total_kv_tokens)
    if n_gpus == 2:
        configs["distserve_faithful"] = disaggregated_config(n_prefill=1, n_decode=1, total_kv_tokens=total_kv_tokens)
    if n_gpus >= 2:
        configs["tetriinfer_paper_reimplementation"] = disaggregated_config(
            n_prefill=1, n_decode=n_gpus - 1, total_kv_tokens=total_kv_tokens,
        )
    configs["llumnix_faithful"] = multi_instance_migratory_config(n_instances=n_gpus, total_kv_tokens=total_kv_tokens)
    return configs
