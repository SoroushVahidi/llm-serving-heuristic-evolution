# Baselines (Canonical)

Synthesizes `docs/baselines.md`, `docs/external_baseline_integration.md`, and
`docs/selector_v2_faithful_baseline_scope_audit.md`. See those for full
detail; this doc is the quick-reference summary and exact-count source.

## A. Historical / internal policy portfolio -- 20 policies

Registered in `src/llmserveopt/policies/registry.py::BASELINE_NAMES`. Mix of
classical schedulers (FIFO, EDF), packing heuristics (first/best-fit),
composite scores (WSPT, SLO-slack), and "style" baselines loosely inspired by
published systems (each self-disclaims "NOT an official reproduction" --
see §F). Full table with provenance: `docs/baselines.md`.

## B. Selector v2 trainable action space -- exactly 8 policies (Option B)

Decided in `docs/selector_v2_faithful_baseline_scope_audit.md` after 1,511
window-evaluations found the faithful external baselines genuinely
dominated under ANWG (see §C and [SELECTOR_V2.md](SELECTOR_V2.md)):

```
fifo
edf
scorpio_style_slo_guard
admission_control
weighted_shortest_processing
estimated_service_time_first
best_fit
multi_bin_batching
```

This is a strict subset of the 20 in §A. **Faithful external baselines are
evaluation-time references only -- they are never part of the trainable
action space.** `orca_style`, `slo_slack_score`, and `shortest_output_first`
are also excluded from this 8-policy set (they remain part of an earlier,
wider "diagnostic pool" of 14 -- `MONOLITHIC_DIAGNOSTIC_POLICY_POOL` /
`monolithic_candidate_policies()` in
`src/llmserveopt/selector/dataset_v2/candidates.py` -- kept for historical
reproducibility and broader diagnostic exploration, not the current
trainable action space).

The canonical Option B constant is
`src/llmserveopt/selector/dataset_v2/candidates.py::SELECTOR_V2_OPTION_B_POLICIES`
(exactly the 8 policies above, import-time-asserted against both the
historical registry and the external baseline registry).
`calibrated_targeted_pilot.py::CANDIDATE_POLICIES` imports this constant
directly rather than duplicating the list.

## C-E. Faithful external baselines -- 6 total, all `selector_eligible=False`

3 monolithic + 2 disaggregated + 1 migratory. All pinned to an exact
upstream commit; none reproduce full production behavior (see each row's
"Important limitation").

### C. Faithful monolithic (3)

| Policy | Topology | Pin | Validation status | Safe claim | Important limitation |
|---|---|---|---|---|---|
| `vllm_faithful` | monolithic, 1 GPU | vLLM v0.1.0, commit `67d96c29fb` | Confirmed genuinely dominated under ANWG (Option B analysis); execution-health clean (0 errors / 910 windows) | "Faithful reimplementation of vLLM v0.1.0's FCFS admission + KV-block scheduler" | All-or-nothing admission (2,560-token cap) -- cannot admit longer prompts at all |
| `vllm_chunked_prefill_faithful` | monolithic, 1 GPU | vLLM v0.4.2, commit `c7f2cf2b7f` | Benchmark-pack-validated for `xlong_context_burst16`; 2 other benchmark-pack scenarios noted as mismatched for a shared-simulator-infrastructure reason, not this baseline's own fidelity | "Faithful reimplementation of vLLM v0.4.2's chunked-prefill `SchedulingBudget` scheduler" | Real default `max_num_batched_tokens=512` (chunked, can eventually admit any prompt length) |
| `sarathi_faithful` | monolithic, 1 GPU | `microsoft/sarathi-serve`, branch `osdi-sarathi-serve`, commit `ceaa0660ea` | Real-hardware-validated (Wulver A100, N=5 repeated trials vs. real vLLM) | "Faithful reimplementation of Sarathi-Serve's stall-free chunked-prefill scheduler" | Real default `max_num_seqs=128` (vs. 256 for the two vLLM variants) -- confirmed inert in all windows generated so far, since `GPUConfig.max_active_sequences` (4-64) is always the tighter binding constraint |

### D. Disaggregated (2)

| Policy | Topology | Pin | Validation status | Safe claim | Important limitation |
|---|---|---|---|---|---|
| `distserve_faithful` | disaggregated, exactly 1 prefill + 1 decode GPU (hard-enforced) | `LLMServe/DistServe`, branch `camera-ready-simulator`, commit `0ec355c874` | Execution-health clean | "Faithful reimplementation of DistServe's online prefill/decode-stage scheduling (offline parallelism/placement planning excluded)" | Online scheduling only |
| `tetriinfer_paper_reimplementation` | disaggregated, >=1 prefill + >=1 decode (multi-decode, power-of-two routing) | arXiv:2401.11181v1 -- **no official code/artifact exists** (verified live), hence `PAPER_REIMPLEMENTATION`, not `_faithful` | Execution-health clean | "Paper-description reimplementation of TetriInfer; not a pinned-commit reproduction" | Length-prediction stand-in is simplified vs. TetriInfer's real fine-tuned OPT-125M classifier |

### E. Migratory (1)

| Policy | Topology | Pin | Validation status | Safe claim | Important limitation |
|---|---|---|---|---|---|
| `llumnix_faithful` | multi-instance migratory, N independent instances, no shared queue | `alibaba/llm-scheduling-artifact` (OSDI 2024 artifact repo, not the continuously-evolving `AlibabaPAI/llumnix`), commit `a90824307` | Execution-health clean | "Faithful reimplementation of Llumnix's dispatch/migration-pair-selection/LCFS-candidate-selection/destination-admission logic" | `min_gpu_count=1` is code-enforced but explicitly not a meaningful evaluation configuration -- needs >=2 to exercise migration |

## F. Style / proxy baselines (subset of A, 7 policies)

`orca_style`, `vllm_style_token_budget`, `sarathi_style`, `splitfuse_style`,
`multi_bin_batching`, `estimated_service_time_first`, `scorpio_style_slo_guard`
(`src/llmserveopt/selector/roles.py::EXTERNAL_STYLE_BASELINES`). Each is an
**original implementation** loosely inspired by a published system's key
idea -- not pinned to a commit, not verified against source. Distinct from
§C-E, which are pinned, commit-verified reimplementations.

## G. Reference-only / non-deployable (1)

`oracle_srtf` -- hindsight SRTF, uses `actual_output_tokens`. Excluded from
`BASELINE_NAMES` and `SELECTOR_CANDIDATE_NAMES` by both code and a runtime
assertion. Never claim it as a deployable policy.

## Exact-count summary

```
Historical/internal portfolio         = 20
Selector v2 trainable action space    = 8   (Option B, strict subset of the 20)
Faithful external baselines, total    = 6
  monolithic                          = 3
  disaggregated                       = 2
  migratory                           = 1
Reference-only (non-deployable)       = 1   (oracle_srtf)
```

## Why Option B (short version)

Across 1,511 window-evaluations (910 SLO-calibrated + 151 restricted +
450 targeted-favorable), the 3 faithful monolithic baselines had **zero**
strong/moderate ANWG wins -- a mechanistically-explained, faithfully
reproduced FCFS-under-overload effect (serving strictly in arrival order
degrades aggregate deadline attainment under load relative to
priority/deadline-aware reordering -- textbook queueing theory, not a
simulator bug; execution health was 910/910 clean, ruling out integration
issues). They remain scientifically valuable as external evaluation
references precisely because they faithfully replicate real systems --
their value doesn't depend on winning this synthetic selection game. Full
seven-angle analysis: `docs/selector_v2_faithful_baseline_scope_audit.md`.
