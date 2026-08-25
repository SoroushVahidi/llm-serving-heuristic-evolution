# Joint-240 Terminal Criticality — Utility Robustness v1

**Date:** 2026-08-25  
**Parent experiment:** `experiments/decision_criticality_terminal_anwg_joint240_v1/`  
**Artifacts:** `experiments/decision_criticality_terminal_anwg_joint240_v1/utility_robustness_v1/`  
**Mode:** analysis-only; no branch rerun; manuscript not edited.

## Verdict

**`CONTINUOUS_UTILITY_DATA_UNAVAILABLE`**

Also applicable framing: reviewer concern is **structurally well-posed** on this corpus
(ANWG effects are pure deadline-credit with universal completion), but **smooth-utility
Δ cannot be computed** from existing branch artifacts.

Exact required statement:

> continuous-utility robustness cannot be computed from existing branch artifacts without rerunning branches with per-request terminal traces enabled.

---

## A. Preflight

- Host: `al-khwarizmi`
- Branch: `contextual-compositional-heuristics-20260731` @ `2987b718…`
- Dirty tree preserved; no reset/clean
- Terminal-criticality: **DONE**
- Guarded selector: **DONE**
- No active scientific processes / no tmux

---

## B. Input validation — per-request data

Inspected `branches.csv` / `branches.jsonl` (3541 rows, 59 scalar columns).

### Present (aggregates only)

- `cf_anwg`, `reference_anwg`, `delta_anwg`, `abs_delta_anwg`
- `cf_num_completed`, `reference_num_completed`, `completion_count_delta`
- `cf_num_dropped`, `cf_sim_duration`, `reference_sim_duration`, `sim_duration_delta`
- scenario pressure / acquisition metadata

### Missing (required for continuous utilities)

- per-request weights \(w_i\)
- per-request completion times \(C_i\) (REF and CF)
- per-request deadlines \(D_i\)
- per-request on-time / lateness / tardiness
- per-request dropped/unfinished flags beyond aggregates
- any per-request terminal outcome table

**No nested per-request payloads** exist in JSONL.

Therefore metrics WCG (as a non-trivial Δ), weighted tardiness, and soft-deadline credit
**cannot** be recomputed offline.

### Aggregate structural finding (does not replace continuous Δ)

| Fact | Value |
|---|---|
| Every REF and CF branch completes **all** scenario requests | True |
| `completion_count_delta` | **always 0** |
| `cf_num_dropped` on nonzero-ANWG branches | **always 0** |
| Among ANWG-zero branches, duration still diverges | 508 / 3335 (15.2%) |

**Implication:** on joint-240 forks, \(\Delta\mathrm{ANWG}\) is **entirely** deadline on-time credit among fully completed request sets. Weighted completion goodput without deadlines would be

\[
\mathrm{WCG} \equiv 1
\]

for every branch, hence \(\Delta\mathrm{WCG} \equiv 0\). That is informative about
ANWG’s step function, but **tardiness / soft-deadline** robustness still needs
\((C_i, D_i, w_i)\) traces.

---

## C. Metric definitions (preregistered; not computed)

Documented for a future re-run with traces enabled. No scales were tuned.

1. **ANWG** (existing): arrival-normalized weighted goodput with hard deadline credit.
2. **WCG**: \(\sum_i w_i 1[\mathrm{completed}_i] / \sum_i w_i\) (deadline-blind).
3. **Weighted mean tardiness**: \(\sum_i w_i \max(0, C_i-D_i) / \sum_i w_i\) (lower better;
   report \(\Delta_{\mathrm{improvement}} = \mathrm{penalty}_{\mathrm{ref}} - \mathrm{penalty}_{\mathrm{cf}}\)).
4. **Soft deadline** (only if principled scale exists): \(e^{-T_i/\mathrm{scale}_i}\) for
   completed requests; **skip** if scale is arbitrary.

**Not computed here** due to missing traces.

---

## D. ANWG recomputation check

Not applicable: no per-request data to recompute ANWG from. Existing stored
`delta_anwg` values remain the authoritative ANWG effects from the frozen run
(3541 branches; REF-replay 240/240).

---

## E–G. Sparsity / concentration / disagreement by continuous metric

**Not available** — blocked on data.

No continuous-metric tables, bootstrap CIs, or AUROC/AUPRC by smooth utility are produced.

---

## H. Top-5 scenario concentration CI sanity-check

Reported anomaly:

- Point (top **5 scenarios**): **0.293**
- Old CI95: **[0.297, 0.562]** (excludes the point)

### Diagnosis: **BUG** (not benign percentile-bootstrap quirk)

In `analyze()`, scenario-level bootstrap did:

1. draw scenarios **with replacement**;
2. concatenate their branch rows;
3. `groupby(scenario_id).sum()` → **collapses duplicate draws**;
4. compute top-5 share on the collapsed unique set.

That biases concentration **upward** and can place the unique-scenario point estimate
**below** the CI lower bound.

State-level concentration bootstrap (concatenated states) was **OK** (multiplicity retained).

### Correction (multiplicity retained)

| Statistic | Point | Corrected CI95 | Old (buggy) CI |
|---|---:|---|---|
| Top 5 scenarios | **0.293** | **[0.192, 0.399]** | [0.297, 0.562] |
| Top 5% scenarios | 0.488 | [0.384, 0.590] | — |
| Top 10% scenarios | 0.685 | [0.590, 0.763] | prior ~[0.593, 0.760] (already multiplicity-based in later patch) |

Point estimate **is inside** the corrected CI.  
Analysis script fixed; `summary.json` / `bootstrap.json` updated with correction history.
Prior criticality analysis markdown should cite **corrected** scenario CIs.

**Labeling check:** point 0.293 is top **5 scenarios**, not top 5% (top 5% ≈ 0.488).

---

## I. Interpretation of the reviewer concern

| Question | Answer |
|---|---|
| 1. Does sparse criticality survive under continuous utility? | **Unknown** — data unavailable. |
| 2. Does concentration survive under continuous utility? | **Unknown** — data unavailable. |
| 3. Is ANWG’s step function responsible for most exact zeros? | **Plausible and supported structurally**: with universal completion, ANWG zeros mean identical weighted on-time credit; smooth tardiness could turn many of these into tiny nonzeros. Cannot quantify without traces. |
| 4. Does the criticism materially weaken the claim today? | It **qualifies** the claim: sparsity of **ANWG** on joint-240 is solid; sparsity of **deadline-smooth value** is **unverified**. Do not over-claim metric-robustness. |
| 5. Manuscript wording? | See §J. |

### Outcome label

**`CONTINUOUS_UTILITY_DATA_UNAVAILABLE`**

Secondary note for authors: if a future re-run finds WCG Δ≡0 (expected here) but tardiness effects dense/unconcentrated, prefer  
`ANWG_SPARSITY_PARTLY_STEP_FUNCTION_ARTIFACT` or `CONCENTRATION_ROBUST_BUT_ZERO_RATE_NOT_ROBUST`.

---

## J. Recommended manuscript wording (do not edit yet)

Conservative language:

- “On joint-240 Alive counterfactual forks, one-step **ANWG** effects are sparse and concentrated.”
- “In this workload all requests complete under both branches, so ANWG contrasts are **deadline-credit** contrasts, not completion-count contrasts.”
- “Whether sparsity persists under continuous tardiness / soft-deadline utilities requires per-request terminal traces not retained in the current branch dump; we treat ANWG sparsity as **metric-specific** pending that check.”
- Keep the same-distribution criticality result as supporting mechanism for why unguarded switching is fragile, without claiming metric-universal Q-value sparsity.
- Cite corrected top-5 scenario mass **0.293, CI95 [0.192, 0.399]** (not the buggy CI).

---

## K. What a minimal future re-run would need

Enable per-request terminal dumps on each CF/REF continuation:

- `request_id`, `weight`, `deadline`, `completion_time` or `+inf` if unfinished, `completed`, `dropped`

Then compute WCG / weighted tardiness / optional soft credit with **pre-registered** scales only.

---

## Artifacts

- `utility_robustness_v1/data_availability.json`
- `utility_robustness_v1/top5_scenario_ci_sanity.json`
- `utility_robustness_v1/summary.json`
- Updated parent `bootstrap.json` / `summary.json` (scenario CI correction only)
- Fixed `scripts/run_decision_criticality_terminal_anwg_joint240_v1.py` bootstrap
- Manuscript: **not edited**
