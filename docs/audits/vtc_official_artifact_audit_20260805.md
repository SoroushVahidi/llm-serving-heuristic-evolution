# VTC Official Artifact Audit — 2026-08-05

Audits the official artifact for VTC ("Virtual Token Counter"), the
multi-tenant fairness scheduler from Sheng et al., *"Fairness in Serving
Large Language Models,"* OSDI 2024. Produced per this task's explicit
instructions to begin the VTC external-baseline phase. Full provenance,
license, requirements, and line-by-line algorithm audit live in
`baselines/vtc/PROVENANCE.md` — this document summarizes the decision
trail, records what was verified and how, and captures findings from the
initial smoke integration (`docs/audits/vtc_initial_integration_20260805.md`
has the integration-specific record; this doc is the artifact/mechanism
audit).

## 0. Repository state at start (per this task's step 1)

- Branch: `contextual-compositional-heuristics-20260731`
- Starting SHA: `1e8df68998e9cc52109efa2c052f30ed13ecc0e6`
- Upstream: `origin/contextual-compositional-heuristics-20260731`, 0 ahead / 0 behind, clean working tree
- `python scripts/check_contextual_composition_status.py` — passed
- `python scripts/check_contextual_composition_status.py --resume-readiness` — passed
- PARS-Serve-2026: confirmed `EVALUATION_ONLY` (`docs/BASELINE_STATUS.md`)
- vLLM-LTR: confirmed `EVALUATION_ONLY` (`docs/BASELINE_STATUS.md`)

## 1. Why VTC, now

`docs/BASELINE_STATUS.md`, `docs/audits/vllm_ltr_baseline_audit_20260804.md`,
`docs/audits/branch_and_pars_readiness_audit_20260804.md`, and
`docs/audits/pars_first_comparative_evaluation_20260804.md` each contain a
line to the effect of "do not begin VTC ... in this task." Read in
context, every one of those is scoping language for the *specific prior
task* that wrote it (e.g. "finish the PARS evaluation before touching
anything else new"), not a standing project-wide prohibition on VTC. This
task's own instructions explicitly say to begin it now — that supersedes
the earlier scoping notes for those tasks, and `docs/BASELINE_STATUS.md`'s
VTC row (previously "Not started ... explicitly not to be started") is
updated by this pass to reflect that the phase has now begun.

## 2. Official artifact identification (task step 2)

| Field | Value |
|---|---|
| Paper | Sheng, Cao, Li, Zhu, Li, Zhuo, Gonzalez, Stoica, *"Fairness in Serving Large Language Models"* |
| Venue | OSDI 2024 (18th USENIX Symposium on Operating Systems Design and Implementation) |
| arXiv | 2401.00588 |
| Official artifact repo | `https://github.com/Ying1123/VTC-artifact` (author-owned, verified via `gh api repos/Ying1123/VTC-artifact`) |
| Artifact-evaluation instructions | `fair_bench/README.md` in the repo itself (no separate AE-committee page found; this is the paper's own AE submission material) |
| Pinned commit | `192c2e2014c69c8c6c699d7113c3822e4db632e6` ("Add changes during revision (#10)", 2024-06-07) — HEAD of `main`; no tags/releases exist on the repo |
| License | Apache-2.0 (LICENSE file present, verified by hash) |
| Requirements | CUDA 11.8, PyTorch ≤2.1.2, `triton==2.1.0`, custom CUDA `bgmv` kernels |
| Supported models | Llama-7B/13B/30B/70B (S-LoRA settings S1/S2/S4/S5/S6) |
| Workloads | 8 synthetic suites (`fair_bench/exp_suite.py`) + 1 masked real trace from chat.lmsys.org |
| Official commands | `launch_server.py` + `run_exp.py --suite <name>` + `plot/plot_*.py` — see `baselines/vtc/official_reference/` |
| Official metrics | Per-client cumulative service over time, response-time distributions, throughput |
| Hardware | 1× A10G (24GB) + 1× A100 (80GB), per the AE instructions |
| Pretrained assets | None — VTC is a pure algorithmic scheduler, no learned component |

No third-party fork was used or needed — the author's own repository is
public, real, and directly identifiable from the paper.

## 3. Mechanism audit (task step 3)

Full detail (with source excerpts) is in `baselines/vtc/PROVENANCE.md`'s
"Algorithm audit" section and `baselines/vtc/official_reference/
vtc_req_queue_excerpt.md`. Summary:

- **Service-accounting state:** `self.served: Dict[tenant, float]`
  (cumulative virtual cost), `self.user_req_list: Dict[tenant, deque]`
  (per-tenant FIFO), `self.fairw: Dict[tenant, float]` (static weights).
- **Scheduling rule:** repeatedly admit from
  `min(active_served, key=active_served.get)` until nothing more fits —
  greedy, work-conserving, min-served-first. Ranking and admission are the
  same step; there is no separate "priority label" phase.
- **Fairness objective:** bound the max service-received difference
  between two continuously-backlogged clients (paper's Theorem: 2×-tight),
  not throughput- or SLO-maximizing.
- **Normalization:** cost ÷ per-tenant `fairw` weight.
- **Tie-breaking:** an *implementation artifact*, not a documented rule —
  Python `min()`'s first-encountered-minimum behavior over `served`-dict
  insertion order (order each tenant's queue first became non-empty).
  Reproduced automatically by this integration because the real object is
  called, never re-derived.
- **Continuous-batching interaction:** `generate_new_batch` (admission,
  once/step) and `update_counter` (decode-cost accrual, once/decode
  iteration) both mutate `served` directly.
- **Prompt/decode accounting:** prompt tokens charged in full at
  admission; decode tokens charged incrementally, one unit per iteration,
  at a configurable price ratio (official default 1:2).
- **Prefix caching:** not modeled anywhere in this artifact — a scope
  limitation of the *official code itself*, not something this
  integration added or removed.
- **What VTC changes vs. base FCFS `ReqQueue`:** admission order only;
  decode-execution mechanics (KV paging, kernel dispatch) are untouched.
- **Runtime state this simulator lacks natively:** (1) no tenant/client
  identity field anywhere in `src/llmserveopt/core/types.py` (confirmed by
  `grep -rn "tenant"` → zero matches); (2) no native per-decode-iteration
  hook — `BasePolicy.select_action` runs once per admission step, not once
  per decode tick; (3) a structurally different KV-feasibility formula
  (simulator: simple additive; official: worst-case sorted-cumulative,
  entangled with LoRA-adapter bookkeeping this project has no analogue
  for).
- **Fairness-theorem assumptions**, all visible directly in the code:
  work-conserving scheduling; the "counter lift" rule in `append()`
  (a returning tenant's counter is floored at the min of currently-active
  peers' counters, preventing unbounded return-time advantage); one
  shared, monotonic cost function across tenants.
- **Core files:** `slora/server/router/vtc_req_queue.py` (`VTCReqQueue`)
  and `slora/server/router/req_queue.py` (`ReqQueue`, base class).

## 4–5. Integration classification and upstream fidelity (task steps 4–5)

**Classification: official policy reused with simulator adapter.**

The preference order is (1) run official code unchanged, (2) wrap it, (3)
reproduce only as a last resort. (1) and a literal-subprocess (2) are both
blocked by a **hardware-generation incompatibility**: this machine's GPU
is an RTX 5060 Ti (Blackwell, `sm_120`); the official artifact requires
CUDA 11.8, whose `nvcc` predates Blackwell and cannot target it at all —
not a version to pin around, a compiler generation gap. `triton==2.1.0`
likewise predates Blackwell codegen. Building the artifact's custom
`slora._kernels` CUDA extension is not possible here. Full detail,
including why this is categorically different from (and worse than) the
version-skew issues the real-vLLM baseline pilot resolved, is in
PROVENANCE.md's "Hardware blocker" section.

What *is* achievable, and what was verified directly: VTC's fairness
**algorithm** (`vtc_req_queue.py`/`req_queue.py`) is pure Python + NumPy
with zero GPU/CUDA/Triton dependency. It was dynamically imported
unmodified from the pinned clone (via `importlib.util.spec_from_file_location`,
with a minimal synthetic package hierarchy so its relative imports
resolve — no source copied or edited) and executed successfully in this
project's plain CPython 3.12 environment. See
`baselines/vtc/adapter/official_loader.py`.

**Upstream fidelity:** the clone lives at
`~/.cache/external_baselines/VTC`, outside this repo's git tree, pinned to
`192c2e2014c69c8c6c699d7113c3822e4db632e6`, verified by
`baselines/vtc/adapter/official_loader.verify_official_clone()` (checks
both the commit hash and that every file this adapter reads still exists).
Not vendored into this repo's git history, even though Apache-2.0 would
permit it, for consistency with `baselines/pars/` and `baselines/vllm_ltr/`.

**Disclosed deviations** (also documented in
`baselines/vtc/adapter/simulator_policy.py`'s module docstring):

1. Single monolithic GPU only — the official artifact has no multi-GPU
   story of its own; the adapter refuses (`UnsupportedTopologyError`)
   rather than inventing one.
2. `lora_ranks` passed as all-zero, matching the paper's own `--no-lora`
   evaluation mode (this simulator has no LoRA-adapter memory model).
3. The official memory-safety gate (`_can_add_new_req`) is fed this
   project's REAL per-GPU capacity numbers and runs for real, verbatim —
   not bypassed with an artificial unlimited budget.
4. `cost_func="linear"` only — the official `"profile"` cost function is a
   regression fit to the authors' own A10G + Llama-2-7B hardware and is
   not portable.
5. Decode-cost is reconstructed from `tokens_decoded_per_request` deltas
   between admission-step calls (the official system calls
   `update_counter` once per decode iteration from inside its own router
   loop; this simulator only calls policies once per admission step).
   Verified exact under headroom (see
   `tests/test_vtc_baseline_adapter.py::TestSimulatorPolicyFairnessBehavior::
   test_heavy_hitter_does_not_starve_light_tenant`); one bounded,
   documented exception remains for requests completing on a run's
   absolute final simulated step (§8 below).
6. `Req` objects carry synthetic `[0] * n` placeholder token-id lists —
   verified by direct inspection that no code path ever dereferences
   token-id *content*, only `len()`.
7. All tenant ids must be known upfront (`known_tenants=`) — the official
   `VTCReqQueue.__init__` only populates its `fairw` dict for the
   `adapter_dirs` list given at construction; an unregistered tenant
   raises a clear `UnregisteredTenantError` here rather than a raw
   `KeyError` from inside the official code (confirmed directly:
   `TestOfficialAlgorithmDirect::test_unregistered_adapter_raises_keyerror_in_official_code`).

Not vendored. Marked `EVALUATION_ONLY`. Not registered as a
composition-library / selector candidate (`baselines/vtc/adapter/
provenance.SELECTOR_CANDIDATE = False`), per this task's explicit scope.

## 6. Minimum faithful integration delivered (task step 6)

```
baselines/vtc/
  PROVENANCE.md
  __init__.py
  adapter/
    __init__.py
    provenance.py          # importable constants mirroring PROVENANCE.md
    errors.py               # typed, explicit failure modes
    official_loader.py      # pinned-clone verification + dynamic import
    simulator_policy.py     # VTCFairnessPolicy(BasePolicy) -- the adapter
  official_reference/
    vtc_req_queue_excerpt.md  # non-executable citation, for readability
  fairness_workloads.py    # 6 fairness-extension workload families (§7)
  smoke_results/
    vtc_smoke_20260805.json
```

Implements exactly the items the task asked for: provenance manifest,
official-checkout loader/adapter, VTC state/accounting adapter (the real
`VTCReqQueue` instance held by `VTCFairnessPolicy`), simulator-compatible
wrapper, workload/tenant mapping (`class_id` → tenant id, see §7),
deterministic tie-breaking (inherited automatically), explicit
unsupported-mode checks (`UnsupportedTopologyError`,
`UnsupportedCostFunctionError`, `UnregisteredTenantError`,
`MissingTenantIdError`), and no use of unavailable future information
(the policy only ever reads `ObservableRequest.predicted_output_tokens`,
never `actual_output_tokens`, matching `BasePolicy`'s own blinding rule).

## 7. Tenant/workload semantics (task step 7)

Confirmed by direct inspection: `src/llmserveopt/core/types.py` has no
tenant/client field anywhere, and the accepted canonical suite
(`benchmarks/canonical_suite/`) was generated without one — every
canonical family is tenant-agnostic. VTC's entire reason for existing
(equalizing service across competing clients) cannot be exercised by the
canonical suite at all.

**Decision:** reuse `Request.class_id` (an existing, otherwise-generic
string field, normally used for SLO-class labels like "tight"/"medium"/
"loose") as the tenant identifier for VTC-specific workloads, rather than
adding a new schema field to `Request`/`ObservableRequest`. This avoids
touching `src/llmserveopt/core/types.py` (core CC5/CC6-adjacent
infrastructure, out of this task's scope) and avoids touching the
canonical suite at all.

Six labeled fairness-extension families were added in
`baselines/vtc/fairness_workloads.py` (none of them touch or regenerate
`benchmarks/canonical_suite/`):

| Family | What it probes |
|---|---|
| `balanced_tenants` | Sanity check: symmetric demand → near-equal service |
| `one_heavy_hitter` | VTC's headline claim: one dominant tenant should not starve the rest |
| `heterogeneous_token_sizes` | Token-WEIGHTED fairness vs. request-COUNT fairness |
| `bursty_tenant` | A burst from one tenant vs. a steady-rate tenant |
| `returning_inactive_tenant` | The official "counter lift" mechanism directly |
| `priority_fairness_conflict` | VTC (fairness-blind to `priority`/`slo_deadline`) vs. an SLO-aware policy |

## 8. Fidelity tests (task step 8)

`tests/test_vtc_baseline_adapter.py` — 25 tests, all passing (verified
`python -m pytest tests/test_vtc_baseline_adapter.py -q`). Split into:

- `TestOfficialLoaderVerification` — missing-clone and stale-commit
  rejection; confirms the real official classes load.
- `TestOfficialAlgorithmDirect` (8 tests) — exercises the raw,
  unmodified `VTCReqQueue` directly (no simulator involved): min-served
  selection order, insertion-order tie-breaking, the counter-lift rule,
  aborted-request zero-charge, the exact linear cost formula, fair-weight
  scaling, and that an unregistered adapter really does raise a bare
  `KeyError` in the official code (motivating `UnregisteredTenantError`
  in the adapter layer).
- `TestSimulatorPolicyAdapterConfig` — every explicit unsupported-mode
  check (multi-GPU, non-linear cost function, empty/unregistered/missing
  tenant).
- `TestSimulatorPolicyFairnessBehavior` — end-to-end simulator runs:
  symmetric-workload service equality, heavy-hitter cost accounting
  (within a small, explicitly bounded tolerance — see below),
  deterministic replay (`==` across two independent runs of the same
  trace), and a returning-tenant scenario that completes cleanly.

**One honestly-disclosed, bounded fidelity gap, found while writing these
tests:** decode-cost reconstruction (deviation 5 above) cannot observe the
very last decode tick of whichever requests happen to complete on a run's
absolute final simulated step, because `BasePolicy.select_action` is never
called again after that step. Measured magnitude: for a 20-request
uniform-arrival, uniform-length workload (worst case — every request
decodes in lockstep and can all complete simultaneously), the undercount
was exactly `output_price` (2 tokens' worth of cost) per synchronized
completion; staggering arrival times reduces this to 0–2 tokens total
across an entire 22-request run. This is structurally unobservable through
the `BasePolicy` interface (not a bug in the delta-reconstruction
approach) and never accumulates beyond the final step. Locked in by
`test_heavy_hitter_does_not_starve_light_tenant`'s `abs=8` tolerance and
explained in `simulator_policy.py`'s deviation-5 docstring.

## 9–10. Smoke evaluation and benchmark-scope decision (task steps 9–10)

Full record: `docs/audits/vtc_initial_integration_20260805.md`. Headline
finding, reported honestly rather than smoothed over: at the smoke-scale
GPU capacity used (`max_active_sequences=8, max_batch_tokens=1024,
max_kv_tokens=4096`), FIFO / `shortest_prompt_first` /
`scorpio_style_slo_guard` produced **identical** per-tenant outcomes to
VTC in 5 of 6 fairness-extension families — not because VTC has no effect,
but because demand never generated sustained admission backlog at that
capacity (nothing was made to wait long enough for scheduling ORDER to
matter). In the one family that did diverge
(`heterogeneous_token_sizes`), the dominant effect was a **methodological
confound**, not a fairness signal: `VTCReqQueue`'s own official admission
gate reserves KV memory for a request's full *predicted* decode length
before admitting it, while every native policy in this simulator
(FIFO/`shortest_prompt_first`/`scorpio_style_slo_guard`) uses a
much simpler check that reserves nothing for future decode growth. Under
tight capacity with long-output tenants, this makes VTC dramatically more
conservative for a reason unrelated to its fairness mechanism.

**Decision on benchmark scope:** VTC is **not** ready for a fair,
apples-to-apples comparison against this project's native policies on
canonical-suite-style regimes at matched nominal capacity numbers, because
the admission-conservativeness confound above would contaminate any
throughput/ANWG comparison. It is appropriately scoped to the dedicated
`baselines/vtc/fairness_workloads.py` extension only (not WildChat
control, not the canonical suite), and even there, a full sweep should
wait for either (a) capacity/contention tuning that produces genuine
backlog without triggering the reservation confound, or (b) giving the
native comparison policies a matched worst-case reservation-style
admission check (a deeper change, out of scope for this integration pass).
This is precisely the kind of finding a smoke test is supposed to surface
before a full sweep is greenlit — see `docs/audits/
vtc_initial_integration_20260805.md` for the exact next action.
