# Family B v2 Prefill/Decode TTFT-Contention Pilot — Scientific Audit

**Date:** 2026-08-17  
**Run:** [`experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/`](../../experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/)  
**Smoke:** [`experiments/policy_separation_prefill_decode_smoke_v2_20260817T024136Z/`](../../experiments/policy_separation_prefill_decode_smoke_v2_20260817T024136Z/)  
**Design (preregistered):** [`docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md`](../design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md)  
**Predecessor (frozen, not rewritten):** [`docs/audits/policy_separation_prefill_decode_pilot_v1_20260817.md`](policy_separation_prefill_decode_pilot_v1_20260817.md)  
**Analyzer:** `scripts/analyze_policy_separation_prefill_decode_pilot_v2.py`  
**Primary metric:** canonical `arrival_normalized_weighted_goodput`  
**Family B v2 verdict:** `FAMILY_B_COMPOSITION_READY`

This audit scores the completed 64-eval v2 pilot against hypotheses and the
composition-readiness gate that were committed **before** launch (`ecc0422`).
It does not launch PrefillControl composition, rewrite frozen v1 CSVs, or
change simulator FCFS semantics.

## A. Run integrity

| Check | Result |
|---|---|
| Evaluations | 32 × 2 = **64**; `n_failed=0` |
| Unique scenario IDs | **32** |
| Duplicate `(scenario_id, policy)` | **0** |
| Completion fraction | **1.0** every row |
| NaN/Inf in primary | **0** |
| Primary column | `arrival_normalized_weighted_goodput` |
| Policy set | exactly `{full_prefill, chunked_prefill_small}` |
| ANWG range | `[0.333, 1.000]` |
| BurstGPT prompts | **32/32** `burstgpt_staged` (hog and late) |
| Outputs | `burstgpt_anchored` + labeled `synthetic_short_output_for_ttft_isolation` |
| KV/resource | no infeasibility; all requests complete |
| Elapsed | 0.17 s (local, 8 workers) |

Smoke predecessor (16/16): `SMOKE_GO` — 4 full practical wins, 4 small practical wins.

## B. Git state

| Field | Value |
|---|---|
| Branch | `contextual-compositional-heuristics-20260731` |
| Launch HEAD | `ecc0422286886c83d263e87655ed1123e62d2565` |
| Launch commit | `feat: Family B v2 TTFT-contention refinement (smoke GO)` |
| Frozen v1 HEAD (untouched) | `138cabf249b4a14e9f5fd1fa8e14f8dc641da0da` audit; raw CSV from `ff78897` |

## C. Exact v1 failure diagnosis

Not an implementation bug, not a metric bug, and not a reason to change FCFS
semantics.

1. **`decode_priority_chunked ≡ chunked_prefill_small` on clean traces.** Both
   use chunk=64. Shared FCFS vs decode-protected only diverge when a request is
   *already decoding* and leftover after an earlier hog's chunk is `< n_decoding`.
   On admission traces every request starts in prefill. A later request can
   enter decode only via leftover the hog did not claim; that leftover then
   self-limits the decoding cohort (fixed-point equilibrium). Tests:
   `tests/test_family_b_v1_mechanism_diagnosis.py`.
2. **`decode_stalled_steps ≡ 0`.** The counter counts already-decoding requests
   that received 0 tokens. v1 late tenants were blocked in **prefill**.
3. **`chunked_prefill_large ≈ full`** and **`adaptive ≡ small`** were twins, so
   they were dropped rather than carried into v2.
4. The real v1 mechanism was class-TTFT via crumbs vs uninterrupted hog
   prefill. v2 isolates that mechanism.

Injected mid-flight state *does* make `decode_first` diverge (500 already-
decoding late requests + hog chunk=64). v2 does not manufacture that state.

## D. v2 policy definitions

| Name | Admission | Execution |
|---|---|---|
| `full_prefill` | greedy arrival-order | `max_prefill_chunk_tokens=65536`, `decode_first=False` |
| `chunked_prefill_small` | greedy arrival-order | chunk=**64**, `decode_first=False` |

No third policy. H7 = `NOT_APPLICABLE`.

## E. Workload / intervention design

Compact factorial (not the v1 144-cell grid):

`hog_count` ∈ {12, 24} × `late_pressure` ∈ {12, 40} × `slo_emphasis` ∈
{hog_ttft, late_ttft} × 4 seeds = 32 cells.

Geometry: long-prompt hog convoy first (BurstGPT window [4096, 16384]),
short-prompt late tenants overlapping at 25% of convoy span ([64, 256]).
`step_token_budget=512`. e2e deadline = arrival + output×step + class slack
(prefill time omitted so convoy blocking bites).

## F. Real vs derived vs synthetic data

| Field | Provenance |
|---|---|
| Hog / late prompts | **real-trace-anchored** BurstGPT staged |
| Outputs | **synthetic intervention**: short window median 80, BurstGPT-shape-anchored (`prefer_real=False`). Necessary so e2e SLO is TTFT-dominated; v1 long decode outputs diluted TTFT and saturated TBT. |
| Arrivals / SLO slacks / convoy shape | constructed (labeled) |
| Occupancy from long decode | **not** reconstructed |

## G. Observable vs forbidden features

Policies see only `ObservableRequest` / `ObservableState`. They are greedy
arrival-order and do not read class_id. Denylist: scenario_id, seed, intended
winner, `slo_emphasis` / `hog_ttft` / `late_ttft` / `hog_count` / `late_pressure`.

Online-observable decision variable used in H3: per-class mean
`slo_deadline − arrival` (tight hog slack vs tight late slack). Prompt length
is also observable.

## H. Preregistered hypotheses and thresholds

See design §9–10. ε=0.01. Unique win = margin > ε. Held-out seed `20260823`.
Gate G1–G10 committed in `ecc0422` before this run.

## I. H1–H10 outcomes

| ID | Result | Evidence |
|---|---|---|
| H1 | **CONFIRM** | `full_prefill` unique ε=0.01 wins **16/32**; all 4 seeds |
| H2 | **CONFIRM** | small-chunk unique wins **15/32**; all 4 seeds |
| H3 | **CONFIRM** | slack-mix predictor accuracy **1.00** on 31 non-near-tie cells |
| H4 | **CONFIRM** | reverse niche in all 4 seeds; sign agree **0.875** |
| H5 | **CONFIRM** | exact-tie **3.1%** (1/32); near-tie **3.1%** (v1: 93.1% / 95.8%) |
| H6 | **CONFIRM** | 16/16 full-win cells: hog TTFT(full) < hog TTFT(small); 15/15 small-win cells: late TTFT(small) < late TTFT(full) |
| H7 | **NOT_APPLICABLE** | third policy not retained |
| H8 | **CONFIRM** | held-out seed: 4 full and 4 small practical wins; sign match **1.00** |
| H9 | **CONFIRM** | each policy has ≥ 4 practical losses (16 and 15) |
| H10 | **CONFIRM** | G1–G10 all true |

## J. Unique winners and tie rates

| Quantity | v2 | v1 (frozen) |
|---|---:|---:|
| Unique ε=0.01 winners | full 16, small 15 | only full 6 |
| Exact-tie rate | **3.1%** | 93.1% |
| Near-tie ε=0.01 | **3.1%** | 95.8% |
| Mean \|Δ\| | **0.131** | 0.013 |
| Seed winner-set agree | **0.875** | 0.611 |

The single exact tie is `pd2.hog12.late12.slolate_ttft.s20260820` (both ANWG=1.0):
the easiest late-tight cell saturates for both policies. It does not create a
hidden third niche.

## K. Complete pairwise practical-win matrix

Only one pair exists.

| Pair | i ≻ j | j ≻ i | near-tie | bidirectional | mean \|Δ\| |
|---|---:|---:|---:|---|---:|
| full ↔ small | **16** | **15** | 1 | **yes** | 0.131 |

By `slo_emphasis`:

- `hog_ttft`: full wins **16/16** cells
- `late_ttft`: small wins **15/16** cells (1 exact tie)

`hog_count` and `late_pressure` change margin size, not sign.

## L. Seed stability

Winner-set agreement **0.875** (7/8 non-seed factor cells identical across 4
seeds). Sign agreement **0.875**. The disagreement is the one saturating
late-tight low/low cell on seed 20260820; other seeds still prefer small there.

## M. Per-class TTFT / TBT / SLO

TBT/TPOT is saturated at 0.001 s (one step) for every policy/cell.
`global_tbt_attainment=1.0`. TBT is **not** the separator.

When full wins (`hog_ttft`):

| | full | small |
|---|---:|---:|
| hog mean TTFT | **0.072** | 0.120 |
| late mean TTFT | 0.083 | 0.064 |
| hog e2e SLO | **0.391** | 0.000 |
| late e2e SLO | 1.000 | 1.000 |

When small wins (`late_ttft`):

| | full | small |
|---|---:|---:|
| hog mean TTFT | 0.074 | 0.122 |
| late mean TTFT | 0.085 | **0.065** |
| hog e2e SLO | 1.000 | 1.000 |
| late e2e SLO | 0.476 | **0.691** |

ANWG tracks the tight class's e2e SLO via TTFT.

## N. Contention / mechanism diagnostics

| Diagnostic | Observation |
|---|---|
| `decode_stalled_steps` | **0** everywhere (expected; mechanism is prefill-side) |
| `prefill_stalled_steps` | higher for full (~183–188) than small (~143–148) |
| Theoretical chunks | full **1.0**; small ~40 |
| Queue on saturated steps | both classes prefilling concurrently (crumbs vs hog monopoly) |

Direction matches the causal story: uninterrupted hog prefill lowers hog TTFT
and raises hog SLO; small chunks lower late TTFT and raise late SLO.

## O. ID / held-out / grouped-split behavior

Held-out seed `20260823` (8 cells):

| Cell | Δ (full−small) |
|---|---:|
| hog12 late12 hog_ttft | +0.250 |
| hog12 late12 late_ttft | −0.292 |
| hog12 late40 hog_ttft | +0.115 |
| hog12 late40 late_ttft | −0.038 |
| hog24 late12 hog_ttft | +0.167 |
| hog24 late12 late_ttft | −0.083 |
| hog24 late40 hog_ttft | +0.094 |
| hog24 late40 late_ttft | −0.047 |

Both niches present; every non-near-tie sign matches the train-seed majority.

## P. Strongest complementary parent pair

`full_prefill` and `chunked_prefill_small`. There is no other pair. The
online-observable switch is **which tenant class currently has tight e2e
slack** (equivalently: long-prompt requests with small remaining slack vs
short-prompt requests with small remaining slack).

## Q. Composition-readiness gate

| ID | Result |
|---|---|
| G1 bidirectional ≥ 8 each way | 16 / 15 |
| G2 unique wins ≥ 8 each | 16 / 15 |
| G3 seed agree ≥ 0.75 | 0.875 |
| G4 near-tie ≤ 0.35 | 0.031 |
| G5 mean \|Δ\| ≥ 0.02 | 0.131 |
| G6 H6 | CONFIRM |
| G7 H3 | CONFIRM |
| G8 H8 | CONFIRM |
| G9 exact-match rate < 0.10 | 0.031 |
| G10 two anchors, no twins | yes |

**Verdict: `FAMILY_B_COMPOSITION_READY`.**

This is not a composition result. It says the parent set is a credible
structural family for a *later* two-parent PrefillControl falsification.

## R. Limitations

- Short outputs are a labeled synthetic intervention; niches may shrink if
  decode work again dominates e2e time.
- `decode_first` remains un-activatable on clean traces; v2 does not claim a
  Sarathi-style third parent.
- Factorial is compact and mechanism-targeted; it is not a broad production
  mix. Generalization beyond hog-convoy / mid-overlap late tenants is untested.
- Simulator FCFS-by-arrival is still the execution model; real engines with
  different running-queue orders could activate decode-priority.
- H3 uses scenario-level mean slack (aggregates of online fields), not a
  trained selector.

## S. Tests

- `tests/test_family_b_v1_mechanism_diagnosis.py` (FCFS equilibrium / stall counter)
- `tests/test_policy_separation_prefill_decode_v2.py` (uniqueness, leakage,
  BurstGPT, KV, micro bidirectional wins)
- `tests/test_analyze_policy_separation_prefill_decode_v2.py`
- Frozen v1 tests still pass; v1 factory unchanged

## T. Commit / push status

Launch provenance commit: `ecc0422`. This audit and derived `analysis/`
summaries are a **separate** follow-up commit. Frozen v1
`per_policy_results.csv` was not modified.

## U. Exact next scientific action

Run the **smallest two-parent PrefillControl composition falsification**:

- Parents: `full_prefill` vs `chunked_prefill_small` only.
- Context: online-observable slack × prompt-length mix (not scenario_id).
- Compare contextual top-1 selection vs a simple state-dependent mix on a
  held-out split of this geometry (and/or a slightly shifted BurstGPT draw).
- Success = envelope expansion beyond selection. Failure =
  `SELECTION_SUFFICIENT_FOR_THIS_PAIR` (as with ESTF↔WFS).

**Do not** start adaptive-child GP, MAP-Elites, symbolic distillation,
Fireworks/Cloudrift, or selector training on the five-policy v1 family.
This audit does **not** itself run that composition experiment.
