# Joint-240 Terminal Criticality — SBS Continuation Robustness v1

**Status:** PREREGISTERED before SBS-continuation evaluation  
**Date:** 2026-08-25  
**Experiment:** `experiments/joint240_terminal_criticality_sbs_continuation_v1/`  
**Parent (unmodified):** `experiments/decision_criticality_terminal_anwg_joint240_v1/`

## 0. Gate context

Primary joint-240 Alive-continuation criticality already exists and is **not**
re-run. This experiment answers only the secondary robustness question:

> Are measured one-step terminal effects qualitatively robust when the
> post-intervention continuation is SBS (`kv_constrained_online`) instead of
> cloned OOF A_live?

## 1. Acquisition (frozen; outcome-blind)

Use **exact** parent acquisition keys from
`experiments/decision_criticality_terminal_anwg_joint240_v1/branches.csv`:

`(scenario_id, step, acquisition_type, alt_policy_id)`

- Do **not** resample states.
- Do **not** change caps.
- Do **not** drop/add states based on |Δ| outcomes.
- Theoretical n = parent n = 3541 (assert exact match of keys evaluated).

## 2. Estimand (SBS continuation)

At each acquired state \(s_t\) reached by replaying the **same OOF Alive
reference trajectory** as the parent:

**REFERENCE-SBS**
- force the original Alive reference action \(a_{\mathrm{ref}}\) at \(t\);
- continue thereafter with fixed SBS policy `kv_constrained_online`.

**INTERVENTION-SBS**
- force the parent-recorded alternative action \(a_{\mathrm{alt}}\) at \(t\)
  (same `alt_policy_id` / admit-set rule as parent);
- continue thereafter with the **same** SBS policy.

\[
C_t^{(\mathrm{SBS})}
=
\mathrm{ANWG}^{(\mathrm{SBS})}_{\mathrm{intervention}}
-
\mathrm{ANWG}^{(\mathrm{SBS})}_{\mathrm{reference}}
\]

Both arms share SBS continuation. **Not** a Q-value.

Join parent \(C_t^{(\mathrm{A\_live})}\) from frozen branches for paired
robustness comparisons.

## 3. Why SBS continuation is semantically valid

After forking at \(t\), the simulator `continue_run` accepts any `BasePolicy`.
A fixed P6 policy (`kv_constrained_online`) is a valid closed-loop continuation
from an arbitrary mid-trajectory state: it only needs the current
`ObservableState`. No Alive FSM / Stage-1 state is required for SBS.

## 4. Integrity

- Reconstruct OOF Alive Stage-1 exactly as parent (frozen folds).
- Replay Alive until each target step; assert step match.
- Live fingerprint unchanged across forks.
- Optional: one REF-Alive replay check per scenario (parent already 240/240).
- Parent `branches.csv` / `summary.json` hashes recorded; never overwritten.

## 5. Analyses (frozen)

1. Prevalence of nonzero \(C_t^{(\mathrm{SBS})}\) (+ CI, scenario-clustered).
2. Top-1/5/10% |Δ| mass under SBS continuation.
3. Paired vs parent Alive continuation:
   - zero/nonzero agreement;
   - sign agreement among dual-nonzero;
   - Spearman of |C|;
   - Jaccard of top-1% and top-5% critical-state sets;
   - disagreement→nonzero AUROC/AUPRC under SBS.
4. Classification:
   - `ROBUST_SPARSE_CONCENTRATION` if SBS also sparse (<15% nonzero) and
     top-10% mass ≥ 0.50 and Spearman(|C|) ≥ 0.3;
   - `CONTINUATION_SENSITIVE` if sparsity/concentration or top-set overlap
     collapses (Jaccard top-5% < 0.2) or Spearman < 0.1;
   - else `MIXED_CONTINUATION`.

## 6. Seeds / compute

- Model/fold seed: parent `20260825`
- Bootstrap seed: `20260827`, \(B=10{,}000\)
- Prefer **local** execution (parent ran locally in ~24 min; Wulver queue empty
  and no remote copy of this repo path required).

## 7. Non-goals

- No new primary Alive criticality run.
- No acquisition-budget expansion.
- No manuscript edit.
- No LTR claim.
