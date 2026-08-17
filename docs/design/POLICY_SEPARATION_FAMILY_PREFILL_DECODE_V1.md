# Policy Separation Family B v1 — Prefill–Decode Interference / Chunk-Control

**Date:** 2026-08-16  
**Status:** PREREGISTERED + CALIBRATED (design fixed before full pilot; smoke
gate documented below)  
**Predecessors:**
- Family A v2 (fairness vs size), Job 1182377 — verdict `USEFUL_BUT_NEEDS_REFINEMENT`
  (`docs/design/POLICY_SEPARATION_FAMILY_A_V2.md`)
- ESTF↔WFS minimal composition falsification — verdict `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
  (`docs/design/ESTF_WFS_COMPOSITION_FALSIFICATION_V1.md`)

## 1. Scientific goal

Construct a controlled workload family where the **prefill/decode execution
mechanism** — how aggressively/fully a scheduler processes prefill work, and
whether it protects later decode-tenant progress — produces genuine,
interpretable bidirectional policy separation. The family is the structural-
mechanism successor to Family A's ranking-level (ESTF↔WFS) separation.

Question:

> "When should a scheduler process a prefill aggressively/full, and when should
> it chunk or defer prefill work to protect active decode progress?"

Eventual composition/synthesis hypothesis (**NOT** tested in this pilot):

> "A state-dependent PrefillControl policy may outperform fixed full-prefill and
> fixed chunked-prefill policies by adapting chunk behavior to decode pressure
> and SLO slack."

## 2. Why this family

1. Family A v2 proved ranking-level complementarity (ESTF↔WFS), but minimal
   composition did not beat top-1 selection (envelope gain 0) →
   `SELECTION_SUFFICIENT_FOR_THIS_PAIR`.
2. Ranking complementarity alone is insufficient evidence that composition helps.
3. Next justified direction: **structural** mechanisms that differ in execution
   behavior.
4. Prefill/decode interference + chunk control is the strongest next family
   from literature (vLLM chunked prefill, Sarathi stall-free prefill, SplitFuse).

## 3. Execution model (reused, not re-implemented)

Phase 1.5 (`ServiceModel.enable_prefill_modeling=True`; see
`docs/decode_prefill_contention_execution_model.md`):

- `enable_decode_prefill_contention=True, decode_first=False` → shared per-step
  budget, FCFS-by-**arrival** among active decode+prefill work.
- `enable_decode_prefill_contention=True, decode_first=True` → decode-protected
  (Sarathi-style): active decodes get budget first; prefill uses the remainder.

**Arrival-order implication (critical):** a later-arriving prefill cannot stall
an earlier-arriving decode under shared FCFS. Discriminative workloads therefore
use a **prefill convoy first**, with decode-tenant arrivals overlapping the
convoy. Entering the `decode_stalled_steps` microbench shape on greedy natural
traces is structurally hard when the early prefill uses an unlimited chunk
(no crumbs for a late tenant to reach the decoding phase); primary separation
signals are ANWG, per-class TTFT, prefill-stall diagnostics, and decode-tenant
SLO attainment. The GPU microbench still verifies decode-deferral semantics.

Policies choose admission only (`Action.admit`); chunk/decode-first live in
`ServiceModel`. Fixed variants A/B/C share greedy arrival-ordered admission so
the **only** difference is execution mechanism.

## 4. Policy / mechanism set

| ID | Name | Admission | Execution |
|---|---|---|---|
| A | `full_prefill` | greedy arrival-order | `max_prefill_chunk_tokens=65536`, `decode_first=False` |
| B1 | `chunked_prefill_small` | greedy arrival-order | chunk=**64**, shared |
| B2 | `chunked_prefill_large` | greedy arrival-order | chunk=**256**, shared |
| C | `decode_priority_chunked` | greedy arrival-order | chunk=**64**, `decode_first=True` |
| D | `adaptive_prefill_control` | defer long prefills under decode pressure + slack | chunk=64, shared |

- A/B/C isolate execution. D is **diagnostic only** (not the synthesized child).
- Calibrated at `step_token_budget=512`. Rejected: chunk∈{128,512} with this
  budget (512 collapses toward full; 128 insufficient TTFT contrast vs 64).

## 5. Workload dimensions

Single GPU. Two observable tenant classes:

| Class | Role | Prompt | Output | Deadline axis |
|---|---|---|---|---|
| `tenant_prefill` | convoy | `prefill_size_class` | short | TTFT / prefill slack |
| `tenant_decode` | overlapping late arrivals | short (~128) | controlled long | decode-tenant e2e margin |

| Factor | Levels |
|---|---|
| `prefill_size_class` | `short`, `medium`, `long`, `mixed` |
| `decode_occupancy` | `low`, `medium`, `high` (n_decode + overlap timing) |
| `slo_regime` | `ttft_tight`, `tbt_tight`, `balanced` |
| `offered_load` | `moderate`, `high` (n_prefill) |
| `seed` | 2 seeds (`20260818`, `20260819`) |

Grid: 4×3×3×2×2 = **144 scenarios × 5 policies = 720 evaluations**.

## 6. Field provenance

| Field | Kind |
|---|---|
| `prompt_tokens` (both classes) | real-trace-anchored BurstGPT when staged; class windows are controlled interventions |
| `actual_output_tokens` / predicted (prefill class) | BurstGPT-anchored/staged in short-output window |
| `actual_output_tokens` / predicted (decode class) | **controlled** lognormal around occupancy median; BurstGPT **shape-anchored** (`prefer_real=False`) — required for deadline geometry |
| `arrival_time` | synthetic (prefill convoy + overlapping decode) |
| `slo_deadline` | synthetic intervention |
| `priority` | fixed 1.0 |
| `class_id` | synthetic tenant label — **observable** |
| factor labels / seed / scenario_id / intended winner | **hidden** |

Production requires staged BurstGPT (`LLM_SERVEOPT_BURSTGPT_CSV`,
`--datasets-root`, or cluster `DATASETS_ROOT`).

## 7. Metrics

**Primary:** canonical `arrival_normalized_weighted_goodput` (ANWG).

**Secondary:** mean/p95/p99 TTFT; mean/p95 TPOT; mean queue wait; completion
fraction; throughput; post-hoc TTFT/TBT attainment vs hidden bounds.

**Mechanism diagnostics:** decode-stalled steps; cumulative decode tokens
deferred; steps with prefill while decode deferred; prefill-stalled steps;
budget saturation; mean decoding/prefilling occupancy; fraction of prefill
tokens scheduled while decodes active; mean prefill delay.

**Documented omissions:** per-request chunk count / inter-chunk gap / iterations
to first decode are not separately instrumented beyond TTFT and contention
aggregates.

## 8. Preregistered hypotheses (falsifiable)

| ID | Hypothesis |
|---|---|
| H1 | Full/large prefill is competitive/superior under low decode overlap and TTFT-tight conditions |
| H2 | Chunked/decode-priority reduces decode-tenant SLO harm under high overlap + decode-tight deadlines |
| H3 | Long-prefill + high-overlap cells produce the strongest separation |
| H4 | Fixed chunking is not universally optimal |
| H5 | Full prefill is not universally optimal |
| H6 | Winner identity changes across regimes (≥2 policies each win ≥1 cell at ε=0.01 ANWG) |
| H7 | Near-tie rate at ε=0.01 is low enough for learning signal (≤ ~0.45) |
| H8 | At least one structural pair shows bidirectional separation at ε=0.01 ANWG |
| H9 | Results are reasonably seed-stable (winner-set agreement ≥ ~0.7) |
| H10 | Diagnostics explain transitions (e.g. lower decode-tenant TTFT / fewer prefill stalls when chunked wins; lower prefill TTFT when full wins) |

Classify later: `CONFIRM` / `CONTRADICT` / `AMBIGUOUS` / `DESIGN_CONFOUND`.

## 9. Adversarial / edge cases

| Case | Factor sketch | Preregistered expectation (falsifiable) |
|---|---|---|
| 1. Low-load counterexample | occ=low, prefill=long, ttft_tight | full/large wins on ANWG via faster prefill TTFT |
| 2. Prefill convoy vs late decodes | occ=high, prefill=long, tbt_tight | chunked/decode-priority wins via lower decode-tenant TTFT / more decode SLO success |
| 3. Prefill burst transition | convoy + overlapping decode (all cells) | disruption visible in TTFT + prefill-stall diagnostics |
| 4. Mixed prefill lengths | `mixed` class | fixed single chunk not universally best |

## 10. Calibration findings (smoke, 2026-08-16)

**Rejected settings:**
- Decode-background-then-late-prefill arrivals: **no** shared-FCFS interference
  when prefills arrive later → universal near-ties / ANWG≈1.
- `prefer_real=True` for decode outputs: BurstGPT short responses collapsed
  occupancy control → no pressure.
- `chunk_large=512` (= step budget): collapses toward full.
- Extreme decode margins (0.02): all policies fail almost all decode SLOs →
  near-ties at low ANWG.

**Chosen settings:**
- Arrival shape `prefill_convoy_then_overlapping_decode`.
- `step_token_budget=512`, chunks `{64, 256, unlimited}`.
- `tbt_tight` decode_margin≈0.15; `ttft_tight` prefill slack≈0.08.
- Controlled decode-output medians by occupancy.

**Smoke gate evidence** (`experiments/policy_separation_prefill_decode_smoke_v1_20260817T020443Z/`, 8×5=40 evals, BurstGPT staged, 2.0s):

- `long × high × tbt_tight`: chunked_small / decode_priority / adaptive ANWG **0.918** vs full **0.864** (spread 0.055)
- `medium × low × ttft_tight`: full / chunked_large **0.381** vs small/decode-priority **0.333** (spread 0.048)
- `medium × high × ttft_tight`: full / chunked_large **0.764** vs others **0.745** (spread 0.018)
- 5/8 smoke cells near-tie (loose-deadline / short-prefill); remaining cells show bidirectional structural contrast
- 40/40 success; canonical ANWG finite; BurstGPT `burstgpt_staged`

**Launch-gate verdict: GO.**

## 11. GO/NO-GO launch gate

Launch full pilot only if smoke shows: ≥2 policies win somewhere; no universal
dominant; both low-overlap/full and high-overlap/chunked regimes (or other
credible bidirectional boundary); no metric bug; anti-leakage tests pass;
BurstGPT path verified; runtime manageable.

## 12. Non-goals

MAP-Elites/QD, selector retraining, symbolic distillation, GP evolution,
LLM-guided synthesis, Fireworks/Cloudrift APIs, broad real-vLLM validation.

## 13. Artifacts

- `src/llmserveopt/policy_separation/templates_prefill_decode.py`
- `src/llmserveopt/policies/prefill_control_variants.py`
- `configs/policy_separation_prefill_decode_pilot_v1.yaml`
- `configs/policy_separation_prefill_decode_smoke_v1.yaml`
- `scripts/run_policy_separation_prefill_decode_pilot_v1.py`
- `scripts/slurm/run_policy_separation_prefill_decode_pilot_v1.sbatch`
- `tests/test_policy_separation_prefill_decode_v1.py`
