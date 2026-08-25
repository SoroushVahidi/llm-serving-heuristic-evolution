# Joint-240 Terminal Criticality v1 — Gate / Reconciliation Design

**Status:** PREREGISTERED as a **hard gate** document (2026-08-25)  
**Does not authorize a new primary Alive-continuation acquisition run.**

## 0. Purpose of this document

This file is the named design target requested for joint-240 terminal
criticality. After mandatory local + Wulver discovery, the primary scientific
question is **already answered** by a completed frozen experiment. Expanding
acquisition caps would duplicate that work and risk outcome-adaptive redesign.

## 1. Hard gate (binding)

| Question | Answer |
|---|---|
| Does completed work already measure one-step terminal ANWG criticality on joint-240 under cloned A_live continuation? | **YES** |
| Authoritative experiment | `experiments/decision_criticality_terminal_anwg_joint240_v1/` |
| Design | `docs/design/DECISION_CRITICALITY_TERMINAL_ANWG_JOINT240_V1.md` |
| In manuscript yet? | **NO** |
| Launch new `experiments/joint240_terminal_criticality_v1/` primary with larger caps (15+10 / scenario)? | **FORBIDDEN** |
| Still missing? | SBS continuation-policy robustness on the **same** frozen acquisition keys |

## 2. Authoritative primary (Alive continuation) — frozen, not re-run

Source of truth (do not overwrite):

- Scenario source: frozen joint-240 (n=240), same folds as adaptive exploitability.
- Caps: 10 disagreement + 5 agreement/control per scenario (theoretical ≤3600).
- Realized: **3541** branch states; **206** nonzero (5.82%).
- Estimand: \(C_t^{(A\_live)}\) = terminal ANWG(intervention) − terminal ANWG(reference),
  both continued with cloned OOF A_live. **Continuation-policy conditional; not a Q-value.**
- Verdict already computed: `JOINT240_TERMINAL_CRITICALITY_REPLICATED`.
- H10 on joint-240: **NOT AVAILABLE**.
- Analysis: `docs/current/decision_criticality_terminal_anwg_joint240_v1_analysis_20260825.md`.

Larger fixed budgets (e.g. 15+10 → ~4800–6000) are **not** executed in this
cycle: existing \(n_{\mathrm{nz}}=206\) already ≫ the A/B/C study’s 27, and
increasing caps after seeing sparsity would be outcome-contingent.

## 3. Authorized follow-on only: SBS continuation robustness

Use existing preregistration (unchanged):

- Design: `docs/design/JOINT240_TERMINAL_CRITICALITY_SBS_CONTINUATION_V1.md`
- Experiment dir: `experiments/joint240_terminal_criticality_sbs_continuation_v1/`
- Acquisition: **exact** parent keys from `branches.csv` (no resample, no cap change).
- Estimand: \(C_t^{(SBS)}\) with `kv_constrained_online` on both REF and INT arms.
- Bootstrap: seed `20260827`, \(B=10{,}000\), scenario-clustered.

### Recovery note (2026-08-25)

A full local SBS run was **aborted** at progress ~190/240 with **no**
`branches_sbs_continuation.csv` written (runner materializes CSV only at end).
Recovery = **single relaunch** of that same experiment into the same directory.
This is not a duplicate primary; it is completion of the authorized robustness arm.

## 4. Post-hoc A_hgb join

Analysis-only, after primary Alive results frozen. May use existing
`posthoc_a_hgb_criticality_join.json` under the SBS experiment directory.
Descriptive; no causal claim; no redesign.

## 5. Non-goals

- No manuscript edit (`paper/llm2026/main.tex`).
- No parent overwrite (A/B/C criticality, joint-240 Alive criticality, adaptive, A_hgb).
- No commit / push.
- No claim of impossibility of adaptive scheduling.
- No second environment duplicate (prefer local; parent Alive run was local ~24 min).

## 6. Interpretation labels (unchanged from parents)

Primary Alive (already decided):

- `REPLICATES_SPARSE_CONCENTRATION` ↔ existing `JOINT240_TERMINAL_CRITICALITY_REPLICATED`
- Proxy: AUROC 0.680 → not strongly informative as a precise detector
  (`PROXY_UNINFORMATIVE` / below 0.70 usefulness bar in parent design)

SBS continuation (pending completion):

- `ROBUST_SPARSE_CONCENTRATION` / `MIXED_CONTINUATION` / `CONTINUATION_SENSITIVE`
  per `JOINT240_TERMINAL_CRITICALITY_SBS_CONTINUATION_V1.md`.
