# Family C Reconstruction v1 and Unified Matrix Completion — v2

Date: 2026-08-17

## 0. Scope

Executes the `FAMILY_C_RECONSTRUCTION_BOUNDED` fallback preregistered in
[`FAMILY_C_RECONSTRUCTION_V1.md`](../design/FAMILY_C_RECONSTRUCTION_V1.md),
following
[`family_c_step2_reconstruction_audit_20260817.md`](family_c_step2_reconstruction_audit_20260817.md).
Builds `CURRENT_RECONSTRUCTED_FAMILY_C_V1` (a new, explicitly-versioned
evaluation layer, not historical replay) and rebuilds the Step-2 unified
utility matrix as **v2**, now fully dense (176×6 = 1,056 cells). **Data
generation and audit only.** No selector trained, no pairwise-regret model,
no mechanism attribution, no composition/synthesis. Historical KV v2
evidence, MF-PSD v1's frozen Family-C rows, and
`experiments/unified_utility_matrix_v1/` are all untouched.

## A. Reconstruction Design and Launch

- Preregistration: [`FAMILY_C_RECONSTRUCTION_V1.md`](../design/FAMILY_C_RECONSTRUCTION_V1.md),
  committed and pushed before launch (`0716fbe`).
- Harness: `src/llmserveopt/policy_separation/family_c_reconstruction_v1.py`
  (generate-once → serialize → replay-from-disk → evaluate, reusing
  `unified_utility_matrix._build_policy` for policy construction). CLI:
  `scripts/build_family_c_reconstruction_v1.py` (resume-safe).
- Tests: `tests/test_family_c_reconstruction_v1.py`, 14 tests, all passing
  — exact scenario count/ID match against MF-PSD v1's Family-C set,
  deterministic generation (two independent calls produce identical
  requests), byte-stable serialization, exact round-trip replay, a
  dedicated bytecode-level guard proving the loader never references any
  BurstGPT/generator symbol, all 6 policies confirmed to share the exact
  same in-memory request tuple per scenario, anti-leakage, and no mutation
  of MF-PSD v1 / historical KV v1+v2 / `unified_utility_matrix_v1`.
- Regression: 220/220 relevant tests pass (203 pre-existing + 20
  unified-matrix-v1 + 14 new + the 3 known env-var-artifact tests
  confirmed passing standalone).
- Smoke: 3 scenarios × 6 policies (18 cells) — all succeeded, replay
  verified exact.
- Launch SHA: `0716fbe`.

## B. Full-Scale Run

tmux session `fc_recon_v1_build`, command
`python scripts/build_family_c_reconstruction_v1.py --out-dir experiments/family_c_reconstruction_v1 --workers 8`.
Ran to natural completion in **2.6s** (well inside the health-check
window). Generated 72 scenarios in one call (IDs verified against MF-PSD
v1's Family-C set — exact match), serialized to
`family_c_reconstruction_v1_scenarios.jsonl`, verified 72/72 exact
round-trip replay, then evaluated **432/432 cells successfully, 0
failures**.

## C. Run Integrity

- 432 rows, 0 duplicates, 0 non-finite `primary_utility_anwg`, 0
  out-of-`[0,1]` values, every one of the 72 scenarios has exactly 6
  policy rows.
- Provenance manifest records: `build_git_head_sha=0716fbe`,
  `config_sha256` for `configs/kv_pressure_pilot_v2.yaml`,
  `burstgpt_dataset_sha256=a4d068a7...` (same file used throughout this
  project's current-environment Family-C work — the "candidate A" from
  the forensic audit), `scenarios_jsonl_sha256`, exact command line.
- `git status --short` on `experiments/kv_pressure_pilot_v1_20260817T162650Z/`,
  `experiments/kv_pressure_pilot_v2_20260817T165053Z/`,
  `experiments/mf_psd_v1/`, and `experiments/unified_utility_matrix_v1/`:
  **empty** — zero diff, before and after this build.

## D. Historical-vs-Reconstructed Crosswalk (diagnostic only)

`experiments/family_c_reconstruction_v1/family_c_historical_vs_reconstructed_crosswalk_v1.csv`
(144 rows: 72 scenarios × 2 native policies). Compares MF-PSD v1's frozen
historical `kv_constrained_online`/`least_laxity_first` values against this
layer's freshly re-evaluated values for the same `(scenario_id, policy)`.

| Metric | Value |
|---|---|
| Exact match (`\|Δ\|<1e-9`) | 45/144 |
| Real difference (`\|Δ\|≥1e-6`) | 99/144 |
| Max `\|Δ ANWG\|` | 0.2500 |
| Mean `\|Δ\|` over the 99 mismatches | 0.0928 |
| Practical-winner flips (ε=0.01) | 17/72 |
| Historical winners (win-or-tie) | `kv_constrained_online`: 67, `least_laxity_first`: 27 |
| Reconstructed winners (win-or-tie) | `kv_constrained_online`: 72, `least_laxity_first`: 20 |

**This independently reproduces
[`kv_v2_reproducibility_forensic_20260817.md`](kv_v2_reproducibility_forensic_20260817.md)'s
own numbers exactly** (same 45/99 split, same max Δ=0.25, same 17/72 flip
count) — expected, since this reconstruction resolves the identical
BurstGPT file (`a4d068a7...`) the forensic audit already characterized as
"candidate A." **These differences are not interpreted as errors in this
new layer** — per the reconstruction audit's own conclusion, they simply
confirm the reconstruction is a genuinely different (not merely re-labeled)
sample from the same underlying generator, exactly as expected for a
layer that is explicitly not historical replay.

## E. Historical vs. Reconstructed Layer Separation — Confirmed

- `experiments/mf_psd_v1/` (all 5 files): unchanged, checksums match.
- `experiments/kv_pressure_pilot_v1_20260817T162650Z/`,
  `experiments/kv_pressure_pilot_v2_20260817T165053Z/`: unchanged
  (`git status --short` empty).
- `experiments/unified_utility_matrix_v1/`: unchanged — Family A/B's cells
  are carried into v2 (§F) by direct copy, not recomputation.
- No historical native Family-C value appears anywhere in
  `experiments/family_c_reconstruction_v1/` or in the rebuilt v2 matrix's
  Family-C rows — every one of the 432 Family-C cells in v2 comes from
  `CURRENT_RECONSTRUCTED_FAMILY_C_V1`, uniformly, across all 6 anchors.

## F. Rebuilt Unified Matrix v2

`scripts/build_unified_utility_matrix_v2.py` (new, committed) assembles:

- **Family A** (432 cells): MF-PSD v1's 144 native rows (`SOURCE_NATIVE`)
  + `unified_utility_matrix_v1`'s 288 valid cross-family rows
  (`STEP2_CROSS_FAMILY_EVALUATION`) — unchanged from v1, copied through.
- **Family B** (192 cells): MF-PSD v1's 64 native rows + 128 valid
  cross-family rows — unchanged from v1, copied through.
- **Family C** (432 cells): **all** from
  `experiments/family_c_reconstruction_v1/` (`cell_source =
  FAMILY_C_RECONSTRUCTION_V1`) — no MF-PSD v1 Family-C row is included.

Output: `experiments/unified_utility_matrix_v2/unified_utility_matrix_long_v2.csv`
(1,056 rows) and `unified_utility_matrix_wide_v2.csv` (176 scenarios × 6
anchor columns). **1,056/1,056 cells populated, 0 missing, 0 failed
status.** `cell_source` breakdown: 208 `SOURCE_NATIVE` (Family A/B only),
416 `STEP2_CROSS_FAMILY_EVALUATION` (Family A/B only), 432
`FAMILY_C_RECONSTRUCTION_V1` (Family C, all 6 anchors).

## G. Winner / Tie / Oracle Analysis (full 176×6 dense matrix, ε=0.01)

**Winner counts** (a scenario can have >1 winner if tied within ε):

| Anchor | Wins | % of 176 |
|---|---|---|
| `estimated_service_time_first` | 92 | 52.3% |
| `kv_constrained_online` | 82 | 46.6% |
| `weighted_fair_share` | 79 | 44.9% |
| `full_prefill` | 49 | 27.8% |
| `chunked_prefill_small` | 48 | 27.3% |
| `least_laxity_first` | 36 | 20.5% |

Unique-winner scenarios: 95/176 (54.0%). Tie scenarios: 81/176 (46.0%).
Every one of the 6 anchors wins on a substantial number of scenarios — no
universal or near-universal dominant policy.

**Mean ANWG:** `weighted_fair_share` 0.7829, `estimated_service_time_first`
0.7797, `kv_constrained_online` 0.7679, `full_prefill` 0.5882,
`least_laxity_first` 0.5850, `chunked_prefill_small` 0.5824.

**Oracle vs. best global fixed:** best fixed = `weighted_fair_share`
(0.7829); oracle (best-of-6 per scenario) = 0.8166; **gain = 0.0336**.

**Per-family oracle gain** (all three positive — genuine signal
everywhere, not concentrated in one family):

| Family | n | Best fixed | Oracle | Gain |
|---|---|---|---|---|
| A | 72 | `weighted_fair_share` (0.7406) | 0.7623 | 0.0217 |
| B | 32 | `estimated_service_time_first` (0.7328) | 0.7822 | 0.0494 |
| C | 72 | `kv_constrained_online` (0.8658) | 0.8861 | 0.0203 |

**Pairwise practical-win matrix** (row beats col, ε=0.01, n=176):

```
        estf    wfs   full  chunk    llf    kvc
estf:      -     50    100    116    104     58
 wfs:     42      -    100    116    118     64
full:     18     23      -     16     75     12
chunk:    33     38     15      -     90     27
 llf:     23     11     42     58      -      5
 kvc:     50     50    108    124    118      -
```

## H. Family-C Diversity / Degeneracy (with the new reconstructed data)

Unlike Family B (§I), **Family C retains substantial cross-family policy
diversity**, even though its two native anchors (`kv_constrained_online`,
`least_laxity_first`) are now freshly re-evaluated:

- `full_prefill == chunked_prefill_small` exact on **72/72** Family-C
  scenarios — the same pre-registered ServiceModel-degeneracy finding
  (design doc §1 Finding 1) holds identically here, as expected (Family C
  never sets `enable_prefill_modeling`, same as Family A).
- All 4 non-native anchors (`estf`/`wfs`/`full_prefill`/
  `chunked_prefill_small`) exactly equal on only **16/72** scenarios — a
  **partial**, not total, degeneracy. Contrast with Family B, where this
  figure was 32/32 (§I). Family C's admission-order-sensitive policies
  (`estf`, `wfs`) genuinely differentiate on the majority (56/72, 77.8%)
  of its scenarios.
- Family C's own winner distribution (n=72):
  `kv_constrained_online` 49, `estimated_service_time_first` 36,
  `weighted_fair_share` 23, `full_prefill`/`chunked_prefill_small` 32 each
  (tied to each other, per the ServiceModel degeneracy), `least_laxity_first`
  19. Unique-winner rate: **28/72 (38.9%)** — real, substantial
  contextual-selection signal, not degenerate.
- `kv_constrained_online` vs. `least_laxity_first` (the two native
  anchors) are within ε of each other on only 20/72 scenarios (27.8%) —
  most of the time they are meaningfully separated, consistent with the
  original KV v2 pilot's own qualitative finding of bidirectional wins
  (though the exact win counts differ from history, §D, as expected for a
  fresh reconstruction).

**Family A+C combined unique-winner rate (144 scenarios, excluding Family
B): 55.6%** — essentially identical to the full 176-scenario rate
(54.0%), confirming Family B's inclusion does not meaningfully dilute the
matrix's overall signal; it simply contributes less of its own.

## I. Family-B Diversity Caveat (carried forward, unchanged by this task)

Re-confirmed, unchanged from
[`unified_policy_utility_matrix_v1_20260817.md`](unified_policy_utility_matrix_v1_20260817.md)
§G2: on all 32 Family-B scenarios, `estf`/`wfs`/`least_laxity_first`/
`kv_constrained_online` remain byte-identical to each other and to
`full_prefill` (only `chunked_prefill_small` differs). This task did not
change or re-touch Family B's cells (§F — copied through unmodified from
`unified_utility_matrix_v1`), so this finding stands exactly as before.
**Family B alone would justify `READY_LOW_DIVERSITY`**, but it is 32/176
(18%) of the full matrix, and its inclusion barely moves the aggregate
unique-winner rate (54.0% vs. 55.6% without it) — the matrix as a whole is
not diversity-limited by Family B's specific degeneracy.

## J. Matrix Completeness

**176/176 scenarios, 6/6 anchors each — 1,056/1,056 cells, 0 missing, 0
failed.** This is a genuine change from
`unified_policy_utility_matrix_v1_20260817.md`'s 768/1,056 (72.7%).

## K. Step-2 Verdict

**`UNIFIED_UTILITY_MATRIX_READY`**

Justification against every stated criterion:

- Complete 176×6 matrix — ✓ (§J).
- Scientifically valid common evaluation within each scenario — ✓: every
  scenario's 6 cells come from one consistent source per family (Family
  A/B: native + Step-2-v1 cross-family, both independently verified
  byte-exact-reconstructible in the prior task; Family C: all 6 from one
  single frozen generate-once reconstruction, §A/§E — never a
  historical/reconstructed hybrid).
- Deterministic replay — ✓ (§A: exact round-trip verified; generation
  independently confirmed reproducible twice).
- Provenance complete — ✓ (§C, plus the two prior Step-2 audits' own
  provenance).
- No unexplained missing/failing cells — ✓ (0/1,056).
- Anti-leakage satisfied — ✓ (tested at every layer: MF-PSD v1,
  `unified_utility_matrix_v1`, this task).
- Contextual-selection opportunity is real and broad-based, not
  concentrated in one family or dominated by one policy — ✓ (§G: 54.0%
  unique-winner rate, positive oracle gain 0.02–0.05 in every family, no
  anchor wins fewer than 36/176 scenarios or more than 92/176).

**Not `READY_LOW_DIVERSITY`**: that label would misrepresent the matrix as
a whole — aggregate diversity is substantial and is barely affected by
excluding the one genuinely low-diversity family (§H last paragraph).
Family B's specific degeneracy (§I) is real and must inform Step 3's
design (e.g., a leave-one-family-out evaluation on Family B specifically
should be expected to show near-zero headroom beyond the
`chunked_prefill_small`-vs-rest split), but it does not license labeling
the *entire* matrix low-diversity.

## L. Files Created / Modified

- `docs/design/FAMILY_C_RECONSTRUCTION_V1.md`
- `src/llmserveopt/policy_separation/family_c_reconstruction_v1.py`
- `scripts/build_family_c_reconstruction_v1.py`
- `scripts/build_unified_utility_matrix_v2.py`
- `tests/test_family_c_reconstruction_v1.py`
- `experiments/family_c_reconstruction_v1/` (scenarios JSONL, long-form
  results, crosswalk CSV, build manifest)
- `experiments/unified_utility_matrix_v2/` (long-form + wide-form matrix)
- This document.
- `docs/current/{RESUME_HERE,NEXT_ACTIONS}.md` (reconciled).

**Not modified:** `experiments/mf_psd_v1/`, both frozen KV pressure run
directories, `experiments/unified_utility_matrix_v1/`,
`docs/audits/kv_v2_reproducibility_forensic_20260817.md`,
`docs/audits/family_c_step2_reconstruction_audit_20260817.md`,
`docs/audits/unified_policy_utility_matrix_v1_20260817.md`,
`docs/audits/multi_family_policy_separation_dataset_v1_20260817.md`.

## M. Exact Next Scientific Action

**With explicit authorization, and only after this: design the
preregistered multi-family contextual-selector experiment (Step 3)** —
per the reassessment roadmap
([`reassessment_composition_hypothesis_20260817.md`](reassessment_composition_hypothesis_20260817.md)
§O step 3), using `experiments/unified_utility_matrix_v2/` as the frozen
input. That design should explicitly account for Family B's low
cross-family diversity (§I) when specifying evaluation splits (e.g. a
leave-Family-B-out fold is expected to be uninformative about anything
beyond the `chunked_prefill_small` contrast). **Not started here.** No
selector training, hyperparameter tuning, pairwise-regret learning, or
mechanism attribution was performed in this task.
