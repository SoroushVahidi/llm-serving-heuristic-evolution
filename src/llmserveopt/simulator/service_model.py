"""
Service-time model for the simulator.

Phase 1 default (enable_prefill_modeling=False)
-----------------------------------------------
* Prefill is instantaneous (zero cost).
* Each decode step produces 1 output token per active request.
* All existing Phase 1 tests use this mode.

Phase 1.5 (enable_prefill_modeling=True)
-----------------------------------------
* Prefill must complete before decoding can start.
* Each step processes up to max_prefill_chunk_tokens per request in prefill.
* Total per-GPU token budget is step_token_budget:
    decode slots   = n_active_decoding  (1 per request)
    prefill slots  = min(max_prefill_chunk_tokens, prefill_remaining) per request
  If decode_first=True the decode budget is guaranteed first; prefill only
  gets the remainder (Sarathi-style stall-free principle).

TODO (Phase 2+)
---------------
* Memory-bandwidth-limited decode slow-down at large batch sizes.
* Realistic GPU FP16 FLOPS model for prefill.
* Heterogeneous GPU throughput multipliers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceModel:
    step_size: float = 0.001          # wall-clock seconds per decode step

    # --- Phase 1.5 prefill parameters (ignored when enable_prefill_modeling=False) ---
    enable_prefill_modeling: bool = False
    prefill_cost_per_token: float = 1.0        # budget tokens consumed per prompt token
    max_prefill_chunk_tokens: int = 512        # max prefill tokens processed per step
    step_token_budget: int = 4096             # total token budget per GPU per step
    decode_first: bool = False                 # guarantee decode budget before prefill
    allow_chunked_prefill: bool = True         # allow multi-step chunked prefill

    # --- Legacy (Phase 1 compat, not actively used in Phase 1.5) ---
    prefill_tokens_per_step: int = 512         # kept for doc purposes

    def compute_prefill_tokens(self, prompt_tokens: int) -> int:
        """Number of prompt tokens that must be processed before decode starts.

        When enable_prefill_modeling=False: always 0 (instant prefill).
        When True: prompt_tokens × prefill_cost_per_token, rounded up.
        """
        if not self.enable_prefill_modeling:
            return 0
        return max(0, math.ceil(prompt_tokens * self.prefill_cost_per_token))

    def prefill_steps(self, prompt_tokens: int) -> int:
        """Minimum steps to complete prefill for a request (for planning only)."""
        total = self.compute_prefill_tokens(prompt_tokens)
        if total == 0:
            return 0
        return math.ceil(total / max(1, self.max_prefill_chunk_tokens))

    def decode_time(self, output_tokens: int) -> float:
        """Wall-clock seconds to decode `output_tokens` (single request)."""
        return output_tokens * self.step_size
