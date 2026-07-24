# Roadmap Gap Analysis

This document records the current gap diagnosis after Selector v2/v3, Policy
Library V2, real-OOD library validation, native composition, module
intervention, SwissAI, TraceLab, SLO/deadline augmentation, and the simulator
discriminative-power audit. It is reconciled for
`wulver-selector-v2-and-composition-integrated`, which also carries Phase 2C
selector-improvement and split-leakage-fix lineage from
`phase2c-final-selector-improvement`.

This document distinguishes **implemented infrastructure** (code exists and
is tested) from **experimentally validated results** (a completed evaluation
supports a specific claim) from **future planned research** (not started). Do
not read an infrastructure item as a validated result.

## Evidence So Far

- The 27-policy V2 library is scientifically useful: real-OOD oracle ANWG
  improved by `0.008904` over the V1 20-policy envelope, about `3.54%`
  relative, with CI `[0.008191, 0.009646]`.
- The 27-policy selector benchmark completed and produced useful suitability
  signals, but the learned top-1 selector did not meaningfully capture the
  V1-to-V2 oracle-envelope gain on held-out OOD.
- On the Phase 2C real-trace eval split (325 windows, all-non-oracle pool,
  8-policy Option B action space), the prior Phase 2C.3 `native_non_oracle_dt`
  selector remains strict-best at ANWG 0.8063 vs. best fixed SCORPIO 0.7963
  and oracle/envelope 0.8298 (29.8% gap closed); the causal advanced-selector
  formulations in `selector/advanced.py` did not beat it under strict
  validation-based model selection --
  `docs/audits/phase2c_final_selector_improvement_audit.md`,
  `SELECTOR_STATUS = IMPROVABLE`.
- A real-trace split-grouping leakage bug was found and independently fixed on
  both lineages; the fixes are reconciled into one implementation -- see
  [LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md).
- Naive rank mixtures, dense composition, and the native component-wise pilot
  did not beat discrete selection or expand the frontier.
  Job `1120123` returned `NATIVE_COMPOSITION_PILOT_DECISION = NO_GO`, but only
  the qualitative decision string was independently verifiable from a
  non-Wulver checkout on 2026-07-24 (Level B evidence); raw numeric artifacts
  remain cluster-only pending read-only recovery.
- Single-module structural interventions produced sparse positive transfer and
  occasional envelope expansion, but module-credit learning generalization is
  still weak. Local overnight/post-processing reports show
  `top1_beats_both_parents_fraction = 0.0` and
  `expands_envelope_fraction = 0.0` at every top-k, so structural synthesis is
  empirically `NOT_READY` even though the harness is interface-ready.
- Weighted reciprocal-rank and normalized score aggregation are implemented
  and unit-tested (`capabilities.py`, `score_aggregation.py`, instrumentation),
  but the local smoke is correctness-only and these operators lack large-scale
  performance validation. No Wolverine oracle-mixture sweep has been launched.
- SwissAI and TraceLab add raw feature-space novelty but their 27-policy sweeps
  saturated ANWG and produced zero strict V2 marginal oracle gain.
- Synthetic SLO/deadline augmentation partially improves class balance and
  exposes EDF/SCORPIO/admission-control regimes, but it is synthetic
  training/regime-probing evidence, not natural real-OOD evidence.
- The simulator discriminative audit found:
  - TraceLab mean ANWG `0.998822` and effective winner classes `1.12`;
  - SwissAI mean ANWG `0.991726` and effective winner classes `1.00`;
  - V2 real-OOD mean ANWG `0.169498` and effective winner classes `10.12`;
  - `KV_CACHE_COUPLING_VERDICT = WEAK_DIRECT_COUPLING`;
  - `PREFILL_DECODE_COUPLING_VERDICT = PARTIAL_AND_WEAK_UNDER_CURRENT_WORKLOADS`;
  - `COMBINER_TRAINING_SIGNAL = WEAK`;
  - `COMBINER_EVALUATION_READINESS = NEEDS_SIMULATOR_FIX`.

## Ranked Bottlenecks

| Rank | Bottleneck | Current likelihood | Evidence |
| --- | --- | --- | --- |
| 1 | Reward saturation / objective ceiling | Very high | SwissAI and TraceLab collapse to near-1 ANWG and near-zero best-second margins despite raw workload novelty. |
| 2 | Weak feature-to-simulator coupling | Very high | Prefix/cache/session/context features are present in artifacts but do not strongly alter KV occupancy, prefill work, queueing, or ANWG. |
| 3 | Insufficient modeled resource pressure | High | Large context and reuse do not automatically become modeled capacity, admission, or queue pressure. |
| 4 | Neutral/missing SLO treatment | High | Deadline-aware policies differentiate under synthetic tight/heterogeneous SLO pressure but not under neutral SwissAI/TraceLab compatibility labels. |
| 5 | Winner/class imbalance | Medium-high | FIFO/simple policies dominate saturated datasets; SLO augmentation helps but remains synthetic. |
| 6 | Policy behavioral collapse | Medium | The 27 nominal policies collapse to about 21 reward-distinguishable classes across audited outputs, with fewer effective winner classes in saturated datasets. |
| 7 | Workload/domain diversity | Medium-low | More raw data alone is insufficient; SwissAI and TraceLab are diverse but non-discriminative under current simulator semantics. Phase 2C's Azure-conv-like failure regime still has zero original train/val examples. |
| 8 | Selector/model capacity or sample size | Low for next action | Model choice is secondary until rewards and pressure produce stable learnable separation. Phase 2C advanced formulations (`selector/advanced.py`) were not the decisive lever. |
| — | Selector v2 split-leakage | resolved (reconciled) | See [LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md); regenerate the stale pre-fix calibrated pilot before trusting VALIDATION/ID_TEST claims. |

## Current Research Posture

Do not spend the next major effort on:

- broad selector model sweeps;
- more generic dataset ingestion;
- dense weighted policy averaging;
- unrestricted structural synthesis;
- broad module-intervention expansion;
- launching the Wolverine/Wulver composition sweep from this checkout.

**Composition decision status: BLOCKED** (2026-07-24 audit). The next
composition-specific operational action is read-only recovery of job
`1120123` numeric artifacts on Wulver, not a new experiment. Reciprocal-rank
and score composition remain correctness-validated only.

The next high-value scientific step is simulator calibration and
discriminative-power validation. The project should make KV/cache reuse,
prefill/decode contention, capacity pressure, and SLO feasibility affect
simulator state transitions and ANWG in scientifically defensible ways, then
rerun bounded subsets before training selectors or combiners. Only after
simulator signal quality improves and recovered pilot numerics are
re-audited should restricted evidence-guided combination/synthesis resume.
