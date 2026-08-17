# Policy Separation Family B v2 — Prefill/Decode TTFT-Contention Refinement

**Date:** 2026-08-17  
**Status:** PREREGISTERED — not yet executed  
**Predecessor audit:** [`docs/audits/policy_separation_prefill_decode_pilot_v1_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v1_20260817.md)  
**Frozen v1 run:** `experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z/` (do not modify)  
**v1 verdict (authoritative, unchanged):** `USEFUL_BUT_NEEDS_REFINEMENT`; `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`

This document preregisters Family B v2 **before** the full pilot. It is not a
composition experiment, not a child-synthesis design, and not a
reinterpretation of frozen v1 CSVs.

## 1. Scientific goal

Answer one question:

> Does prefill/decode control contain at least two stable, mechanistically
> distinct, online-observable policy niches that could justify a *future*
> composition/synthesis experiment?

This is a structural-parent refinement. Possible terminal verdicts:

| Verdict | Meaning |
|---|---|
| `FAMILY_B_COMPOSITION_READY` | Preregistered composition-readiness gate passes. Recommend the smallest subsequent two-parent PrefillControl composition falsification — **do not run it in this task**. |
| `USEFUL_BUT_NEEDS_REFINEMENT` | Real contrast exists, but the gate fails (stability, ties, mechanism, or held-out). Stop composition. |
| `FAMILY_B_REFINEMENT_NO_GO` | Cannot demonstrate bidirectional practical wins, even in smoke. Stop Family B composition work. |
| `DESIGN_CONFOUND` | Integrity / leakage / KV / BurstGPT / schema failure. Do not interpret ANWG. |

The purpose is **not** to obtain a positive composition result.

## 2. v1 failure diagnosis (expected FCFS consequence)

Frozen v1 facts (not re-analyzed here): `decode_priority_chunked` ≡
`chunked_prefill_small` ANWG on 144/144 cells; `decode_stalled_steps` ≡ 0
everywhere; `adaptive_prefill_control` ≡ small; `chunked_prefill_large` ≈
`full_prefill`.

### 2.1 Why `decode_priority_chunked == chunked_prefill_small`

Both use `max_prefill_chunk_tokens=64`. The only intended difference is
`decode_first`.

Phase 1.5 execution (`GPUState._advance_shared_contention` vs
`_advance_decode_protected`):

- Shared (`decode_first=False`): one step budget, FCFS by
  `(arrival_time, request_id)`. An earlier still-prefilling request can
  exhaust the budget before a later already-decoding request.
- Decode-protected (`decode_first=True`): every *already-decoding* request is
  served first; prefill gets the remainder.

On **clean admission traces** every request starts in prefill. A later
request Y can enter decode only by consuming leftover the earlier hog Z did
not claim that step. Z, still prefilling, claims up to its own chunk from
leftover *before* Y sees it (Z has higher arrival priority). Therefore the
concurrently-served decoding cohort self-limits to exactly that leftover —
identically under both `decode_first` values (fixed-point equilibrium;
documented in `src/llmserveopt/selector/dataset_v2/contention_fixtures.py`
and `docs/decode_prefill_contention_execution_model.md`).

This is **not** an implementation bug and **not** a metric bug. It is an
expected FCFS-by-arrival-time consequence of the implemented semantics.
v2 does **not** alter semantics to manufacture decode-priority separation.

`decode_first` *does* diverge when mid-flight state is injected (already-
decoding late requests whose count exceeds leftover after the hog's chunk).
That state is unreachable from greedy natural traces. Diagnosis tests:
`tests/test_family_b_v1_mechanism_diagnosis.py`.

### 2.2 Why `decode_stalled_steps == 0`

The counter increments only when a request that is **already decoding**
receives 0 tokens this step. v1 late tenants arrived with ~128-token
prompts, so they were blocked in **prefill** (`prefill_stalled_steps`), not
decode. Instrumentation is correct; the workload never created the counted
state.

### 2.3 What actually separated full vs small in v1

Full prefill: earlier hog takes the whole 512-token budget → later tenants
wait in prefill → hog-class TTFT better, late-class TTFT worse.

Small chunk: 64-token crumbs → later tenants that overlap mid-convoy get
FCFS crumbs ahead of later hogs → late-class TTFT better, hog-class TTFT
worse.

v1 ANWG moved through **e2e SLO via TTFT delay**, not TBT/decode-stall
(TBT was saturated; `decode_stalled_steps` dead). The reverse small-chunk
niche was real but small and seed-fragile. Per-class TTFT/SLO were missing
from the frozen CSV.

## 3. Policy set (v2)

| Name | Admission | Execution | Role |
|---|---|---|---|
| `full_prefill` | greedy arrival-order | chunk=65536, `decode_first=False` | Anchor A |
| `chunked_prefill_small` | greedy arrival-order | chunk=64, `decode_first=False` | Anchor B |

**Dropped (twins / un-activatable):**

- `chunked_prefill_large` (256): near-identical to full on v1 (132/144 exact).
- `decode_priority_chunked`: identical to small on clean traces; activating
  it requires injected mid-flight decode state or a semantics change.
- `adaptive_prefill_control`: collapsed to small; not a distinct parent and
  not a synthesized child.

No third policy is added. A third decode-priority mechanism would only be
justified if the workload could activate it from online-admitted traces
without changing FCFS semantics. It cannot. H7 is therefore
`NOT_APPLICABLE`.

Factory: `make_prefill_decode_variants_v2()` (does not replace the v1
factory).

## 4. Workload / intervention design

Compact factorial aimed at the TTFT-class tradeoff. **Not** the v1 144-cell
grid.

| Factor | Levels | Causal reason |
|---|---|---|
| `slo_emphasis` | `hog_ttft`, `late_ttft` | Which class has a tight e2e margin. Online-observable via per-request `slo_deadline` vs prompt length. |
| `late_pressure` | low=12, high=40 late tenants | Overlapping short-prompt demand during the convoy. |
| `hog_count` | low=12, high=24 | Convoy severity / uninterrupted-prefill value. |
| seed | 4 values, see §9 | Reproducibility / held-out split. |

Fixed geometry:

- Hog convoy first, prompt window [4096, 16384] median 8192, dt=0.003 s.
- Late tenants start at 25% of convoy span, prompt window [64, 256] median 128.
- `step_token_budget=512`, contention enabled.
- `class_id`: `tenant_prefill` (hog) / `tenant_late` (overlapping short).

**Synthetic intervention (labeled):** both classes use short outputs
(median 80, window [48, 128], BurstGPT-shape-anchored, `prefer_real=False`).
Necessary to isolate TTFT-contention SLO. v1 long decode outputs diluted
TTFT into e2e and made TBT saturate. This is **not** a BurstGPT occupancy
reconstruction.

SLO construction (e2e deadline = arrival + output×step + class slack;
prefill time is *not* in the deadline):

- `hog_ttft`: hog slack 0.05 s, late slack 2.0 s
- `late_ttft`: hog slack 2.0 s, late slack 0.08 s

BurstGPT provenance: hog/late **prompts** staged from the matching window
when the pool is large enough, else anchored. Production refuses silent
synthetic fallback.

Generator labels (`slo_emphasis`, `hog_count`, `late_pressure`, seed,
scenario_id) are **not** policy inputs.

## 5. Observability / metric schema

**Primary (canonical):** `arrival_normalized_weighted_goodput`

**Secondary outcomes:** unweighted e2e SLO success, completion fraction,
request/token throughput, global TTFT/TPOT.

**Per-class secondary:** mean/p95 TTFT, mean/p95 TPOT (TBT proxy), e2e SLO
success, implied TTFT-budget attainment
(`ttft ≤ slo_deadline − arrival − predicted_output × step_size`).

**Mechanism diagnostics:** `prefill_stalled_steps`,
`cumulative_prefill_requests_stalled`, `decode_stalled_steps` (expected ~0),
`cumulative_decode_tokens_deferred`, `steps_with_prefill_while_decode_deferred`,
budget saturation, mean num prefilling/decoding, fraction prefill tokens
while decodes active, theoretical chunk counts, mean inter-chunk extra wait
(`prefill_delay − n_chunks × step_size`), queue composition on saturated
steps (mean num_prefilling / num_decoding).

`decode_stalled_steps` remaining 0 is **not** a failed activation of the v2
mechanism. The v2 mechanism is class-TTFT via prefill crumbs vs hog
uninterrupted prefill.

## 6. Leakage denylist

Policies may use only online observables (`ObservableRequest` /
`ObservableState`): arrival time, prompt tokens, predicted output, SLO
deadline, priority, class_id, GPU prefilling/decoding/KV counts.

Explicitly forbidden as policy inputs: scenario_id, seed, intended winner,
treatment/factor labels (`slo_emphasis`, `hog_ttft`, `late_ttft`,
`hog_count`, `late_pressure`), generator version.

v2 policies are greedy arrival-order admission and do not read class_id;
the denylist protects a future selector.

## 7. Smoke gate (must pass before the full pilot)

Config: `configs/policy_separation_prefill_decode_smoke_v2.yaml`

Grid: `slo_emphasis` × `late_pressure` × `hog_count=low` × seeds `{7, 11}`
= 8 scenarios × 2 policies = 16 evals.

**GO** only if all of:

- S1. 100% task success, no NaN/Inf primary.
- S2. ≥1 cell where `full_prefill` beats `chunked_prefill_small` by > 0.01 ANWG.
- S3. ≥1 cell where small beats full by > 0.01 ANWG.
- S4. Policy set is exactly the two anchors.
- S5. BurstGPT prompts are `burstgpt_staged` or `burstgpt_anchored`; outputs
  carry `synthetic_short_output_for_ttft_isolation`.

If any fail: **`FAMILY_B_REFINEMENT_NO_GO`**. Do not launch the full pilot.

## 8. Full factorial (only if smoke GO)

Config: `configs/policy_separation_prefill_decode_pilot_v2.yaml`

`hog_count` ∈ {12, 24} × `late_pressure` ∈ {12, 40} × `slo_emphasis` ∈
{hog_ttft, late_ttft} × seeds `{20260820, 20260821, 20260822, 20260823}`
= **32 scenarios × 2 policies = 64 evals**.

Held-out grouped split (H8): seed `20260823` is held out; the other three
seeds are the train/confirm set. No peeking at full results before this
document is committed.

## 9. Preregistered hypotheses (score after the full run only)

ε = 0.01 on canonical ANWG. Unique win (2-policy family) = margin > ε.

| ID | Hypothesis | CONFIRM if |
|---|---|---|
| H1 | `full_prefill` has a reproducible niche | unique ε=0.01 wins ≥ 8/32 **and** ≥ 3 of 4 seeds have ≥1 practical full win |
| H2 | small-chunk has a reproducible niche | same for `chunked_prefill_small` |
| H3 | Direction of full-vs-small follows an **online-observable** state variable, not hidden scenario identity | On non-near-tie cells, predictor `sign(mean_e2e_slack_hog − mean_e2e_slack_late)` matches `sign(ANWG_full − ANWG_small)` with accuracy ≥ 0.80. Slacks are computed from request `slo_deadline − arrival` (observable). |
| H4 | Reverse (small≻full) niche survives multiple seeds | small≻full in ≥ 2 distinct seeds **and** seed sign-agreement of full−small ≥ 0.75 |
| H5 | Tie/near-tie prevalence materially below v1 | exact-tie rate ≤ 0.25 **and** near-tie ε=0.01 ≤ 0.35 (v1: 0.931 / 0.958) |
| H6 | Mechanism diagnostics explain the sign | ≥ 80% of full-win cells have hog mean TTFT(full) < hog mean TTFT(small); ≥ 80% of small-win cells have late mean TTFT(small) < late mean TTFT(full) |
| H7 | If a third decode-priority mechanism is retained, it is behaviorally distinct from ordinary small chunking | **`NOT_APPLICABLE`**: third policy not retained (see §3). Diagnosis tests document why. |
| H8 | Niches survive held-out seed 20260823 | held-out seed has ≥ 1 practical win each direction **and** slo_emphasis sign on held-out matches the majority train-seed sign in ≥ 75% of the 8 factor cells |
| H9 | No policy universally dominates | each policy has ≥ 4 practical losses |
| H10 | Composition-readiness gate | all of G1–G10 in §10 |

## 10. Composition-readiness gate (strict; defined before seeing full results)

Declare `FAMILY_B_COMPOSITION_READY` **only if every** item holds:

| ID | Requirement |
|---|---|
| G1 | Bidirectional practical wins: full≻small ≥ 8 **and** small≻full ≥ 8 at ε=0.01 |
| G2 | Each policy uniquely wins ≥ 8 cells at ε=0.01 |
| G3 | Seed winner-set agreement ≥ 0.75 across 4 seeds (8 non-seed cells) |
| G4 | Near-tie ε=0.01 ≤ 0.35 |
| G5 | Mean \|Δ\| (full−small) ≥ 0.02 |
| G6 | H6 CONFIRM |
| G7 | H3 CONFIRM |
| G8 | H8 CONFIRM |
| G9 | Exact ANWG match rate between the two policies < 0.10 |
| G10 | Policy set has no twins: exactly `{full_prefill, chunked_prefill_small}` |

Winning “a few cells” is not enough. If G1 holds but any of G2–G10 fail →
`USEFUL_BUT_NEEDS_REFINEMENT`. If G1 fails after a GO smoke → still
`FAMILY_B_REFINEMENT_NO_GO` for composition (the full grid did not
reproduce bidirectional practical volume). Integrity failure →
`DESIGN_CONFOUND`.

If and only if this gate passes, the **next** scientific action is a
minimal two-parent PrefillControl composition falsification (contextual
selection vs a simple state-dependent mix of the two anchors). That
experiment is **not** part of this task.

## 11. Compute discipline

1. Unit tests + diagnosis microbenchmarks.
2. Discriminative smoke (16 evals).
3. Commit design/implementation/tests (and smoke provenance) **before**
   the full 64-eval pilot.
4. Full pilot only if smoke GO.
5. Analyze only a completed run. No Fireworks/Cloudrift, GP, MAP-Elites,
   LLM policies, selector training, or adaptive-child synthesis.

## 12. Artifacts

- Templates: `src/llmserveopt/policy_separation/templates_prefill_decode_v2.py`
- Variants: `make_prefill_decode_variants_v2`
- Runner: `scripts/run_policy_separation_prefill_decode_pilot_v2.py`
- Analyzer: `scripts/analyze_policy_separation_prefill_decode_pilot_v2.py`
- Tests: `tests/test_family_b_v1_mechanism_diagnosis.py`,
  `tests/test_policy_separation_prefill_decode_v2.py`
