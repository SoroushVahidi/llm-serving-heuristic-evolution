# Family B v1 Prefill/Decode Chunk-Control Pilot — Scientific Audit

**Date:** 2026-08-16  
**Run:** [`experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z/`](../../experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z/)  
**Design:** [`docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md`](../design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md)  
**Launch provenance:** [`docs/audits/policy_separation_prefill_decode_pilot_v1_launch_20260816.md`](policy_separation_prefill_decode_pilot_v1_launch_20260816.md)  
**Analyzer:** `scripts/analyze_policy_separation_prefill_decode_pilot_v1.py`  
**Primary metric:** canonical `arrival_normalized_weighted_goodput`  
**Family B verdict:** `USEFUL_BUT_NEEDS_REFINEMENT`  
**Composition decision:** `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`

This audit scores the frozen 720-eval pilot. It does not launch a new run, rewrite `per_policy_results.csv`, or implement PrefillControl synthesis.

## 1. Provenance / integrity

| Check | Result |
|---|---|
| Branch / HEAD at this audit | `contextual-compositional-heuristics-20260731` (analysis commit; run HEAD was `ff78897`) |
| Git HEAD at launch | `ff78897a08a2dd4d04dd0317f79df9a0ba1485ba` |
| Evaluations | 144 × 5 = **720**; `n_failed=0` |
| Unique scenario IDs | **144** |
| Policies per scenario | **5** |
| Duplicate `(scenario_id, policy)` | **0** |
| NaN/Inf in primary or mechanism metrics | **0** |
| Primary column | `arrival_normalized_weighted_goodput`; no ambiguous `anwg` |
| BurstGPT | **144/144** consistent: prefill prompt/output and decode prompt `burstgpt_staged`; decode output `burstgpt_anchored` (controlled occupancy, as preregistered) |
| Canonical ANWG | primary in every row |
| Raw CSV | frozen; analysis writes only under `analysis/` |
| Grid | prefill∈{short,medium,long,mixed} × occ∈{low,medium,high} × SLO∈{ttft_tight,tbt_tight,balanced} × load∈{moderate,high} × 2 seeds |

Smoke predecessor: `experiments/policy_separation_prefill_decode_smoke_v1_20260817T020443Z/` (40/40). Coarse launch peek of 58/144 cells with spread > 0.01 is reproduced exactly by the structural max−min spread.

## 2. Headline quantitative findings

| Quantity | Value |
|---|---:|
| Exact-tie rate (5 policies / structural 4) | **93.1%** (134/144) |
| Near-tie ε=0.01 | **95.8%** (138/144) |
| Mean / median best−second margin | **0.00085 / 0.000** |
| Fraction of cells with unique-winner margin > 0.01 | **4.2%** (6/144) |
| Structural max−min spread > 0.01 | **40.3%** (58/144) |
| Unique exact winners | `full_prefill` **10**; all others **0** |
| Unique winners at ε=0.01 | `full_prefill` **6** |
| Winner entropy (unique winners) | **0 bits** |
| `full_prefill`≻`chunked_prefill_small` / reverse @ε=0.01 | **47 / 11** |
| Pairwise near-ties full↔small @ε=0.01 | **86/144 (59.7%)** |
| `chunked_prefill_small` ≡ `decode_priority_chunked` | **144/144** exact ANWG |
| `adaptive_prefill_control` ≡ `chunked_prefill_small` | **144/144** exact ANWG |
| `full_prefill` ≡ `chunked_prefill_large` | **132/144** exact; **137/144** within 0.01 |
| Adaptive envelope expansion @ε=0.01 | **0** |
| Seed winner-set agreement | **61.1%** (44/72 cells) |

Exact structural winner-set occupancy:

| Winner set | n |
|---|---:|
| all four structural policies tie | 78 |
| `full_prefill` ≡ `chunked_prefill_large` (beat small/decode-priority) | 44 |
| `chunked_prefill_small` ≡ `decode_priority_chunked` (beat full/large) | 12 |
| `full_prefill` unique | 10 |

The family therefore has **one pairwise structural contrast** (aggressive prefill vs small-chunk) and **two near-twin collapses** (`full`≈`large`, `small`≡`decode_priority`≡`adaptive`).

## 3. Pre-registered hypotheses (H1–H10)

Scored against the wording in the design doc. Unique-winner criteria are applied where the hypothesis states them explicitly (H6, H7). Pairwise ANWG is used where the claim is about optimality rather than unique-winner diversity (H4, H5).

| ID | Hypothesis | Result | Evidence |
|---|---|---|---|
| H1 | Full/large competitive/superior under low decode overlap and TTFT-tight | **CONFIRM** | 16-cell subset: unique ε=0.01 wins full/large **3** vs small/decode-priority **0**; mean ANWG 0.298 vs 0.287 |
| H2 | Chunked/decode-priority reduces decode-tenant SLO harm under high overlap + decode-tight | **AMBIGUOUS** | 16-cell high×TBT-tight mean ANWG 0.967 vs full 0.962 (Δ=0.0057≤0.01); TBT attainment **1.0** and TPOT **0.001** for every policy; unique wins **0**. Long×high×TBT-tight pairwise Δ≈0.023 favors small, but decode-stall diagnostics are dead |
| H3 | Long-prefill + high-overlap cells produce the strongest separation | **CONTRADICT** | long×high mean unique-winner margin **0.00069** and frac>0.01 **0**; short×low **0.00116 / 0.083**; overall **0.00085 / 0.042** |
| H4 | Fixed chunking is not universally optimal | **CONFIRM** | `full_prefill` uniquely wins 6 cells at ε=0.01; chunking is not in every winner set |
| H5 | Full prefill is not universally optimal | **CONFIRM** | **11** cells where small-chunk beats full by >0.01 (and 12 exclusive small≡decode-priority winner-set cells). Unique non-full ε=0.01 wins remain 0 because small≡decode-priority |
| H6 | Winner identity changes (≥2 policies each uniquely win ≥1 cell at ε=0.01) | **CONTRADICT** | only `full_prefill` uniquely wins at ε=0.01 (6 cells) |
| H7 | Near-tie rate at ε=0.01 ≤ ~0.45 | **CONTRADICT** | **0.958** |
| H8 | At least one structural pair is bidirectional at ε=0.01 | **CONFIRM** | full↔small **47/11**; full↔decode-priority **47/11** (identical, because small≡decode-priority); large↔small **45/11**; full↔large **6/1** (weak) |
| H9 | Seed-stable winner-set agreement ≥ ~0.7 | **AMBIGUOUS** | structural winner-set agree **0.611**; full↔small sign agree **0.694** (near-miss of 0.70) |
| H10 | Diagnostics explain transitions | **AMBIGUOUS** | when full uniquely wins, mean TTFT 0.059 vs small 0.074; when small beats full pairwise, prefill-stalled steps fall (229 vs 267) but aggregate TTFT is *higher* for small. `decode_stalled_steps` are **identically 0** for all structural policies (documented arrival-FCFS limitation) |

## 4. Pairwise separation

ε=0.01 unless noted. `i ≻ j` means ANWG_i − ANWG_j > ε.

| Pair | i ≻ j | j ≻ i | near-tie | bidirectional | mean \|Δ\| |
|---|---:|---:|---:|---|---:|
| full ↔ small | 47 | 11 | 86 | **yes** | 0.0133 |
| full ↔ decode-priority | 47 | 11 | 86 | **yes** (duplicate of full↔small) | 0.0133 |
| large ↔ decode-priority | 45 | 11 | 88 | yes | 0.0125 |
| small ↔ large | 11 | 45 | 88 | yes | 0.0125 |
| full ↔ large | 6 | 1 | 137 | weak | 0.0010 |
| small ↔ decode-priority | 0 | 0 | 144 | **no** | **0** |

The only scientifically distinct parent contrast is **full (≈ large) vs small-chunk (≡ decode-priority ≡ adaptive)**.

### Regimes driving full ≻ small (47 cells)

- SLO: balanced **30**, ttft_tight **17**, tbt_tight **0**
- occupancy: low 18 / medium 16 / high 13
- prefill: long 14, medium 13, short 11, mixed 9
- load: high 26 / moderate 21
- seed: 20260818 **27** vs 20260819 **20**

### Regimes driving small ≻ full (11 cells)

- SLO: tbt_tight **6**, ttft_tight **5**, balanced **0**
- occupancy: medium 6 / high 3 / low 2
- prefill: mixed **6**, long **5**, medium **0**, short **0**
- load: moderate 7 / high 4
- seed: 20260818 **8** vs 20260819 **3** (unstable)

Small-chunk's niche is real but **small, concentrated in long/mixed prefills, and seed-skewed**. Full-prefill's advantage is broader, especially under balanced SLOs.

## 5. Factor / mechanism surfaces

Causal interpretation to test: *low decode / TTFT-tight → full or large can win; high decode / TBT-tight → small chunking and/or decode-priority can win.*

**Partially supported, not as a unique-winner law.**

- `slo=ttft_tight × occ=low` (n=16): full mean ANWG **0.298** vs small **0.287**; 3 unique ε=0.01 full wins. Direction matches.
- `slo=tbt_tight × occ=high` (n=16): small mean ANWG **0.967** vs full **0.962**; unique wins **0**; TBT saturated. Directional mean only.
- `slo=tbt_tight × occ=medium`: small **0.969** vs full **0.959** (same pattern).
- `slo=balanced`: full higher at every occupancy (unique-winner margins still tiny).
- `prefill=long × slo=tbt_tight`: small **0.915** vs full **0.896**.
- `prefill=mixed × slo=ttft_tight`: small **0.533** vs full **0.522** (chunked can win even under TTFT-tight when prefills are mixed).
- Decode-priority never separates from small-chunk on any surface.

H3's expectation that long×high is the *strongest unique-winner* surface is false; that cell is a 4-way or 2-way tie on unique-winner margin. Pairwise, long/mixed is where small can beat full, which is a different statement.

## 6. Low-load counterexample

Preregistered case 1: `occ=low, prefill=long, slo=ttft_tight` (n=4).

| Policy | Mean ANWG | Mean TTFT | TTFT attainment | Prefill stalls |
|---|---:|---:|---:|---:|
| full / large | 0.264 | 0.120 / 0.124 | 0.366 / 0.344 | 359 / 353 |
| small / decode-priority / adaptive | 0.258 | 0.152 | 0.258 | 306 |

Task-7 slice (`occ=low, load=moderate, slo=ttft_tight`, n=8): full/large mean ANWG **0.320** vs small **0.314**; unique-winner margin **0**.

Aggressive prefill is **directionally better** (faster TTFT, higher TTFT attainment, slightly higher ANWG). It is **not** a unique ε=0.01 winner there, because full≡large and Δ vs small is ~0.006. Chunking is not universally better; the counterexample exists on mechanism metrics and mean ANWG, but is below the practical unique-winner threshold.

## 7. Prefill-convoy / active-decode edge case

Preregistered case 2: `prefill=long, occ=high, slo=tbt_tight` (n=4).

| Policy | Mean ANWG | Mean TTFT | Prefill stalls | Decode stalled steps | TBT attainment |
|---|---:|---:|---:|---:|---:|
| small / decode-priority / adaptive | **0.869** | 0.103 | 352 | **0** | 1.0 |
| full | 0.846 | 0.095 | 377 | **0** | 1.0 |
| large | 0.844 | 0.097 | 375 | **0** | 1.0 |

Pairwise Δ ≈ **0.023** (practical). Unique-winner margin is **0** because small≡decode-priority. Mixed×high×TBT-tight is a four-way ANWG=1.0 tie.

Full prefill does **not** increase decode stall or TBT/TPOT (both already 0 / 1.0 / 0.001). Chunking reduces prefill-stalled steps and raises canonical ANWG. The improvement **does** translate into ANWG, but **not** through the hoped-for decode-stall/TBT channel. That channel is not isolated in this grid (arrival-ordered shared FCFS, as the design already warned).

## 8. Secondary / mechanism metrics

Across the full 144 cells, mean ANWG:

| Policy | Mean ANWG | Median |
|---|---:|---:|
| `full_prefill` | 0.8060 | 0.921 |
| `chunked_prefill_large` | 0.8052 | 0.921 |
| `chunked_prefill_small` | 0.7981 | 0.900 |
| `decode_priority_chunked` | 0.7981 | 0.900 |
| `adaptive_prefill_control` | 0.7981 | 0.900 |

TBT attainment is essentially saturated; mean TPOT is 0.001 for all policies. Decode-stalled steps and cumulative decode tokens deferred are **0**. When full wins uniquely, it does so by **lower aggregate TTFT / prefill delay**. When small beats full pairwise, it does so despite **higher** aggregate TTFT, via fewer prefill stalls and higher e2e SLO success (ANWG = unweighted SLO success in these rows). Per-class TTFT/SLO is not in the frozen CSV, so decode-tenant vs prefill-tenant attribution remains inferential.

## 9. Adaptive prefill control (diagnostic only)

`adaptive_prefill_control` is **not** a synthesized child.

- Unique wins: **0**
- Envelope expansion over the four fixed structural policies @ε=0.01: **0**
- Exact ANWG identity with `chunked_prefill_small` and `decode_priority_chunked`: **144/144**
- Mean gap vs best structural: **−0.0106**

It collapses to small-chunk behavior everywhere and never expands the envelope. It does **not** provide evidence that state-dependent PrefillControl works. It also does not falsify that idea: the diagnostic admission rule simply did not leave the small-chunk manifold on this grid.

## 10. Seed stability

72 non-seed factor cells × 2 seeds.

| Quantity | Value |
|---|---:|
| Winner-set agreement | 44/72 = **0.611** |
| Top-policy agreement | **0.611** |
| Unstable cells (winner-set, top policy, or important-pair sign) | **31** |
| full↔small sign agreement @ε=0.01 | **0.694** |
| small↔decode-priority sign agreement | **1.0** (identical) |

The 11-cell small≻full niche is seed-imbalanced (8 vs 3). The most important structural boundary is **not** acceptably seed-stable by the preregistered ~0.7 winner-set bar.

## 11. Family B scientific verdict

**`USEFUL_BUT_NEEDS_REFINEMENT`**

Not `STRUCTURAL_SEPARATION_VALIDATED`: only one policy has a unique ε=0.01 win region; 5-policy near-tie rate is 96%; seed agreement is 61%; decode-priority is not a distinct mechanism on this grid.

Not `REDESIGN_REQUIRED` in the Family A v1 sense: the intended **full vs small-chunk** pairwise contrast is real and bidirectional (47/11), BurstGPT+canonical ANWG integrity is clean, H1/H4/H5/H8 hold, and chunking is not a universal winner. The failure mode is **near-twin confounding plus weak unique-winner headroom**, not a dead family.

What must be refined before composition:

1. Drop or replace `decode_priority_chunked` until a workload actually isolates `decode_first` (currently identical to chunk=64 shared).
2. Drop or re-calibrate `chunked_prefill_large` (256 ≈ unlimited 65536 on 132/144 cells).
3. Instrument **per-class** TTFT/SLO (prefill vs decode tenants). Aggregate TBT/decode-stall are saturated/zero.
4. Increase unique-winner headroom or accept pairwise (not unique-winner) labels; 93% exact ties cannot train a selector.
5. Add seeds; the 11-cell reverse niche is seed-fragile.

## 12. Strongest candidate parent pair

Ranked structural pairs (diagnostic excluded). Rank 1 is the only non-redundant bidirectional pair:

**`full_prefill` vs `chunked_prefill_small`**

| Item | Value |
|---|---|
| Bidirectional @ε=0.01 | **47 / 11** |
| Pairwise near-ties @ε=0.01 | 86/144 (59.7%) |
| Mean \|Δ\| | 0.0133 |
| Practical-margin cells >0.01 | 58 |
| Seed sign agreement | 0.694 |
| Structural? | Yes (chunk cap 65536 vs 64; shared FCFS; identical greedy admission) |
| Online features that loosely separate niches | SLO regime (balanced → full; tbt_tight → small), prefill size (long/mixed for the reverse niche), decode occupancy (weaker) |
| Redundant twins | `chunked_prefill_large` ≈ full; `decode_priority_chunked` ≡ small |

`full_prefill` vs `decode_priority_chunked` has identical counts and is **not** a distinct pair. Do not treat it as a second composition parent.

This pair is recorded as the strongest *candidate*. It is **not** authorized as a composition experiment in this audit.

## 13. PrefillControl composition go/no-go

| Question | Answer |
|---|---|
| A. Real structural decision boundary? | **Pairwise yes, unique-winner no.** full↔small is bidirectional; H6 unique-winner diversity fails |
| B. Predictable from legitimate online state? | **Only weakly.** SLO regime and long/mixed prefill correlate with the reverse niche; occupancy is not a clean switch; seed disagreement is high |
| C. Is a fixed chunk size insufficient? | **Yes in pairwise terms** (64 vs 256/unlimited), **no as three distinct chunk policies** (256≈unlimited) |
| D. Does decode_priority add a distinct niche? | **No.** ANWG identical to small-chunk on 144/144 cells |
| E. Does adaptive_prefill_control suggest state-dependent control may work? | **No.** Zero unique wins, zero envelope expansion, exact collapse to small-chunk |
| F. Enough evidence for `minimal adaptive PrefillControl child vs fixed parents vs contextual selector`? | **No** |

**`PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`**

A composition experiment on this frozen grid would mostly compare a parent against a near-identical twin, with 60% pairwise near-ties and a 11-cell reverse niche that is seed-unstable. That is a worse setting than ESTF↔WFS, which already returned `SELECTION_SUFFICIENT_FOR_THIS_PAIR`.

## 14. Limitations

- No per-class TTFT/SLO in the frozen CSV.
- `decode_stalled_steps` identically 0; arrival-FCFS makes later prefills unable to stall earlier decodes.
- Adaptive diagnostic uses the same chunk=64 as small-chunk, so collapse is unsurprising.
- Pilot scale, two seeds, single GPU, synthetic arrivals with BurstGPT-anchored sizes.
- Hypothesis H8 lists several bidirectional pairs, but three of them are the same contrast copied across twins.

## 15. Exact next scientific action

**Family B v1 refinement, not PrefillControl synthesis.** Keep the frozen run as pairwise evidence that full vs small-chunk can separate. Do not start adaptive-child synthesis, symbolic distillation, MAP-Elites, GP, or Fireworks/Cloudrift LLM loops from this family.

Optional later refinement (not this commit): a Family B v2 mechanism set with two non-redundant parents, per-class metrics, and a decode-first isolation check that actually moves ANWG.
