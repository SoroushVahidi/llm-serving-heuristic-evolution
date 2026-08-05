"""Simulator policy wrapper driving the official, unmodified VTCReqQueue.

Mirrors ``baselines/pars/adapter/simulator_policy.py`` /
``baselines/vllm_ltr/adapter/simulator_policy.py``'s structure: this file
is a translation shim only. Every fairness-scheduling DECISION (which
tenant/request is admitted next, how much "served" cost each request
accrues) is made by calling the real ``VTCReqQueue.append`` /
``generate_new_batch`` / ``update_counter`` methods, dynamically imported
unmodified from the pinned official clone (see ``official_loader.py`` and
``../PROVENANCE.md``) -- nothing here reimplements VTC's own selection or
accounting logic.

Deliberately **not** registered in ``src/llmserveopt/policies/registry.py``
or any selector-candidate set -- evaluation-only, per this task's explicit
scope boundary.

Known, disclosed deviations from a literal full-system run (all explained
in detail in ``../PROVENANCE.md``):

1. **Single monolithic GPU only.** The official artifact is a single-server
   scheduler with no multi-GPU/shared-pool story of its own; this adapter
   refuses (``UnsupportedTopologyError``) rather than inventing one for
   ``len(state.gpu_states) != 1``.
2. **LoRA-adapter memory bookkeeping disabled.** ``lora_ranks`` is passed
   as an all-zero dict for every tenant, matching the paper's own
   ``--no-lora`` "vanilla" evaluation mode (this project's simulator has
   no LoRA-adapter memory model at all).
3. **Real capacity fed to the official memory gate.** ``generate_new_batch``
   is called with ``max_total_tokens``/``batch_max_tokens``/
   ``running_max_req_size`` taken directly from this project's
   ``ObservableGPUState`` (not an artificially unlimited budget) -- the
   official ``_can_add_new_req`` memory-safety check runs for real, on
   real numbers, verbatim.
4. **``cost_func="linear"`` only.** The official "profile" cost function is
   a regression fit to the authors' own A10G + Llama-2-7B hardware and is
   not portable here (see ``provenance.SUPPORTED_COST_FUNC``).
5. **Decode-cost reconstructed from per-step deltas.** The official system
   calls ``update_counter`` once per decode iteration from inside its own
   router loop; this simulator only invokes policies once per admission
   step. This adapter calls the real, unmodified ``update_counter`` method
   once per token of ``tokens_decoded_per_request`` delta observed since
   the previous step -- semantically identical to being called every
   iteration, since ``update_counter``'s linear branch adds a flat
   per-call increment. A request that disappears from
   ``active_requests_info`` between two calls (i.e. completed) is
   finalized up to its ``predicted_output_tokens`` the next time
   ``select_action`` runs, closing the gap that would otherwise leave
   every completed request's last decode tick uncharged. **One bounded
   exception remains, disclosed rather than engineered around:** requests
   that complete during the simulation's absolute FINAL step have no
   subsequent ``select_action`` call at all (the run loop terminates
   immediately after `_advance_decode`), so their last decode tick's
   charge (``output_price`` each) is never observed by any per-step
   policy -- structurally unobservable through the ``BasePolicy``
   interface, not a bug in the delta reconstruction itself. This affects
   only however many requests complete on the run's single final
   simulated step -- typically small, but for a workload where many
   requests share the same arrival time AND the same predicted output
   length (lockstep decoding), it can be the entire concurrently-admitted
   batch, not just ``max_active_sequences``. It never accumulates beyond
   that one final step, and staggering arrival times or output lengths
   avoids it entirely (see the fidelity test suite,
   ``tests/test_vtc_baseline_adapter.py``, for a workload that hits this
   edge case vs. one that avoids it). See
   ``docs/audits/vtc_official_artifact_audit_20260805.md`` §8 for the
   measured magnitude.
6. **Synthetic placeholder token ids.** The simulator has no request text,
   only token counts. Official ``Req`` objects are built with
   ``[0] * n`` placeholder id lists of the correct length -- verified by
   direct inspection that no code path in ``vtc_req_queue.py``/
   ``req_queue.py`` ever dereferences token id *content*, only ``len()``
   (via ``Req.__init__``'s ``self.input_len = len(prompt_ids)``).
7. **All known tenants must be registered upfront.** ``VTCReqQueue.__init__``
   only populates its ``fairw`` dict for the ``adapter_dirs`` list given at
   construction; a tenant id encountered later that wasn't pre-registered
   raises ``UnregisteredTenantError`` here (see errors.py) rather than a
   raw ``KeyError`` from inside the official code.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.base import BasePolicy

from . import provenance
from .errors import (
    MissingTenantIdError,
    UnregisteredTenantError,
    UnsupportedCostFunctionError,
    UnsupportedTopologyError,
)
from .official_loader import load_vtc_official_classes

#: Never a selector candidate / never in the historical registries.
SELECTOR_ELIGIBLE = False


def default_tenant_of(req: ObservableRequest) -> Optional[str]:
    """Default tenant-mapping function: reuses ``class_id`` (an existing,
    otherwise-generic string field on every request) as the tenant
    identifier. See ``../../../docs/audits/vtc_official_artifact_audit_20260805.md``
    §7 for why this project's canonical workloads carry no dedicated
    tenant field and why repurposing ``class_id`` for VTC-specific
    fairness-extension workloads (rather than adding a new schema field)
    was chosen."""
    return req.class_id


class VTCFairnessPolicy(BasePolicy):
    """Admits requests in the order the official, unmodified VTCReqQueue
    decides, given a per-request tenant mapping. See module docstring for
    the full list of disclosed deviations from a literal full-system run.
    """

    name = "vtc_fairness_reference"

    def __init__(
        self,
        known_tenants: Sequence[str],
        tenant_of: Optional[Callable[[ObservableRequest], Optional[str]]] = None,
        fair_weights: Optional[Dict[str, float]] = None,
        input_price: Optional[float] = None,
        output_price: Optional[float] = None,
        cost_func: str = provenance.SUPPORTED_COST_FUNC,
        clone_path: Optional[str] = None,
    ):
        if cost_func != provenance.SUPPORTED_COST_FUNC:
            raise UnsupportedCostFunctionError(
                f"cost_func={cost_func!r} is not supported; only "
                f"{provenance.SUPPORTED_COST_FUNC!r} is portable to this "
                "simulator (see PROVENANCE.md)."
            )

        official = load_vtc_official_classes(clone_path)
        self._Req = official.Req
        self._Batch = official.Batch
        self._SamplingParams = official.SamplingParams
        self._VTCReqQueue = official.VTCReqQueue

        self._known_tenants: List[str] = list(known_tenants)
        if not self._known_tenants:
            raise UnregisteredTenantError("known_tenants must be non-empty.")
        self._tenant_of = tenant_of or default_tenant_of
        weights = fair_weights or {}
        self._fair_weight_by_tenant: Dict[str, float] = {
            t: weights.get(t, 1.0) for t in self._known_tenants
        }
        self._input_price = provenance.DEFAULT_INPUT_PRICE if input_price is None else input_price
        self._output_price = provenance.DEFAULT_OUTPUT_PRICE if output_price is None else output_price
        self._cost_func = cost_func
        self._lora_ranks = {t: 0 for t in self._known_tenants}

        self._vtc = None
        self._known_request_ids: set = set()
        self._last_decoded: Dict[int, int] = {}
        self._req_tenant: Dict[int, str] = {}
        self._req_predicted_output: Dict[int, int] = {}
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._vtc = self._VTCReqQueue(
            max_total_tokens=1,          # placeholder; re-set per-call from real GPU capacity
            batch_max_tokens=1,
            running_max_req_size=1,
            adapter_dirs=self._known_tenants,
            fair_weights=[self._fair_weight_by_tenant[t] for t in self._known_tenants],
            cost_func=self._cost_func,
            input_price=self._input_price,
            output_price=self._output_price,
        )
        self._known_request_ids = set()
        self._last_decoded = {}
        self._req_tenant = {}
        self._req_predicted_output = {}

    # ------------------------------------------------------------------
    def _make_req(self, request_id: int, tenant: str, prompt_tokens: int, max_new_tokens: int):
        sp = self._SamplingParams(max_new_tokens=max(1, max_new_tokens))
        return self._Req(tenant, request_id, [0] * max(1, prompt_tokens), sp)

    def _register_new_arrivals(self, waiting_queue: Sequence[ObservableRequest]) -> None:
        for req in waiting_queue:
            if req.request_id in self._known_request_ids:
                continue
            tenant = self._tenant_of(req)
            if tenant is None:
                raise MissingTenantIdError(
                    f"Request {req.request_id} has no tenant id under the "
                    "configured tenant-mapping function."
                )
            if tenant not in self._fair_weight_by_tenant:
                raise UnregisteredTenantError(
                    f"Request {req.request_id} maps to tenant {tenant!r}, "
                    f"which is not in known_tenants={self._known_tenants!r}."
                )
            self._req_tenant[req.request_id] = tenant
            self._req_predicted_output[req.request_id] = req.predicted_output_tokens
            official_req = self._make_req(
                req.request_id, tenant, req.prompt_tokens, req.predicted_output_tokens
            )
            self._vtc.append(official_req)
            self._known_request_ids.add(req.request_id)

    def _charge_one(self, rid: int, tenant: str, n_tokens: int) -> None:
        if n_tokens <= 0:
            return
        stub = self._make_req(rid, tenant, 1, 1)
        batch = self._Batch(f"decode-delta-{rid}", [stub])
        for _ in range(n_tokens):
            self._vtc.update_counter(batch)

    def _charge_decode_deltas(self, gpu: ObservableGPUState) -> None:
        current_active_ids = {info.request_id for info in gpu.active_requests_info}

        # Finalize any request that was active last call but has since
        # disappeared (completed) without an intervening select_action
        # call to observe its final decode tick's delta -- otherwise the
        # last output token of every completed request would silently go
        # uncharged (see simulator_policy.py module docstring, deviation
        # 5, and docs/audits/vtc_official_artifact_audit_20260805.md).
        # Best-effort: assumes it decoded exactly its predicted_output_tokens,
        # which this project's policies are already blind-to-actual-length
        # and plan around; a genuine actual-vs-predicted mismatch would
        # leave a residual under/over-charge of the same kind every
        # length-prediction-based policy in this project already accepts.
        for rid in list(self._last_decoded.keys() - current_active_ids):
            tenant = self._req_tenant.get(rid)
            target = self._req_predicted_output.get(rid, 0)
            remaining = target - self._last_decoded.get(rid, 0)
            if tenant is not None:
                self._charge_one(rid, tenant, remaining)
            del self._last_decoded[rid]

        for info in gpu.active_requests_info:
            rid = info.request_id
            decoded_now = gpu.tokens_decoded_per_request.get(rid, 0)
            decoded_before = self._last_decoded.get(rid, 0)
            delta = decoded_now - decoded_before
            tenant = self._req_tenant.get(rid)
            if tenant is not None:
                self._charge_one(rid, tenant, delta)
            self._last_decoded[rid] = decoded_now

    def _build_current_batch(self, gpu: ObservableGPUState):
        if not gpu.active_requests_info:
            return None
        reqs = []
        for info in gpu.active_requests_info:
            tenant = self._req_tenant.get(info.request_id)
            if tenant is None:
                # Active request this policy never saw arrive (shouldn't
                # happen -- every admitted request passed through
                # _register_new_arrivals first) -- skip defensively rather
                # than crash the whole run mid-sweep.
                continue
            stub = self._make_req(info.request_id, tenant, info.prompt_tokens, info.predicted_output_tokens)
            decoded = gpu.tokens_decoded_per_request.get(info.request_id, 0)
            stub.output_ids = [0] * decoded
            reqs.append(stub)
        return self._Batch("current", reqs) if reqs else None

    # ------------------------------------------------------------------
    def select_action(self, state: ObservableState) -> Action:
        if len(state.gpu_states) != 1:
            raise UnsupportedTopologyError(
                f"VTCFairnessPolicy supports exactly 1 GPU (monolithic "
                f"single-server, matching the official artifact's own "
                f"topology); got {len(state.gpu_states)}."
            )
        gpu = state.gpu_states[0]

        self._register_new_arrivals(state.waiting_queue)
        self._charge_decode_deltas(gpu)

        self._vtc.max_total_tokens = gpu.max_kv_tokens
        self._vtc.batch_max_tokens = gpu.max_batch_tokens
        self._vtc.running_max_req_size = gpu.max_active_sequences

        current_batch = self._build_current_batch(gpu)
        new_batch = self._vtc.generate_new_batch(current_batch, self._lora_ranks)

        admit_ids: List[int] = []
        if new_batch is not None:
            admit_ids = [r.request_id for r in new_batch.reqs]

        return Action(admit={gpu.gpu_id: admit_ids})

    # ------------------------------------------------------------------
    def served_snapshot(self) -> Dict[str, float]:
        """Read-only snapshot of each tenant's current VTC `served` cost
        counter -- for fairness metrics/tests, never consulted by
        scheduling logic itself."""
        return dict(self._vtc.served)
