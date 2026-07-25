# Selector Status

> **Pause addendum 2026-07-25.** The repaired balanced real-trace pilot
> (`PARTIALLY_READY`) improves evidence about *load discrimination* under
> stratified real windows, but it does **not** make the selector “solved”
> and does **not** authorize a full fingerprint sweep or unrestricted
> retraining. Natural/busy near-ties remain high; treat scaled windows as
> stress evidence. Primary bottleneck remains simulator/objective
> discrimination plus natural-load signal. Outcome-signature diagnostics
> are not action traces.

---


Current selector evidence as of 2026-07-22.

## Completed Selector Milestones

- Leakage-safe Selector Dataset v2 pipeline was generated and audited.
- Selector v2 OOD diagnosis identified distribution shift and WSP-vs-SCORPIO
  routing failures.
- Selector v3 added broader domains and richer causal-state features.
- The 27-policy V2 selector/regret benchmark trained full-information
  suitability models over the expanded policy library.

## Main Results

| Experiment | Status | Key result |
| --- | --- | --- |
| Selector v2 Overnight Scale | COMPLETE | 1600 leakage-safe windows, 775 meaningful windows, audit PASS. RF per-policy regressor beat WSP on ID (`0.559481` vs `0.527113`) but lost on OOD (`0.247707` vs `0.256383`). |
| Selector v2 OOD Investigation | COMPLETE | `SELECTOR_STATUS = IMPROVE_DATA_OR_FEATURES`; OOD shift was detectable, uncertainty fallback did not fix robustness. |
| Selector v3 Multi-Domain Causal-State | COMPLETE | `SELECTOR_STATUS = DATA_LIMITED`; richer dynamic causal features helped boundary learnability, but selectors still did not consistently beat fixed WSP on held-out ID/OOD/final splits. |
| 27-policy V2 Selector/Regret Benchmark | COMPLETE | `SELECTOR_V2_27_STATUS = STRONG` in the experiment report, with useful suitability/ranking signals. Important caveat: learned top-1 selection still did **not** meaningfully capture the V1-to-V2 oracle-envelope gain on held-out OOD and remains substantially below the 27-policy oracle. |

## Current Interpretation

The selector is useful but not solved.

What is supported:

- full-information reward vectors are valuable supervision;
- regret-aware/listwise and suitability-style outputs are more appropriate than
  plain RMSE or exact top-1 classification alone;
- the selector can beat some fixed policies on some splits;
- the suitability vector is useful diagnostic input for later donor/module
  discovery.

What is not supported:

- a claim that the selector captures most of the V2 oracle gain;
- a claim that top-1 selection is robust under all temporal/cross-source OOD;
- a claim that suitability quality is strong enough for unrestricted synthesis.

## Current Bottleneck For Selector Work

The immediate bottleneck is upstream of model choice. The simulator/objective
often collapses diverse workload regimes to near-identical policy rewards,
especially in SwissAI and TraceLab. Until KV/cache, prefix reuse,
prefill/decode contention, overload, and SLO feasibility are better coupled to
simulated pressure and ANWG, new selector models will be trained on weak or
misleading labels in many regimes.

## Current Stop/Go Position

- Freeze broad generic selector-model sweeps.
- Do not retrain the 27-policy selector as the next major step.
- Resume selector work only after bounded simulator calibration and controlled
  re-evaluation produce trustworthy policy separation.
- Continue to use grouped leakage-safe splits, near-tie-aware regret metrics,
  and development-only model selection.
