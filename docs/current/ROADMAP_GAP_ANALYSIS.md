# Roadmap Gap Analysis

This document records the current gap diagnosis after Selector v2/v3, Policy
Library V2, real-OOD library validation, native composition, module
intervention, SwissAI, TraceLab, SLO/deadline augmentation, and the simulator
discriminative-power audit.

## Evidence So Far

- The 27-policy V2 library is scientifically useful: real-OOD oracle ANWG
  improved by `0.008904` over the V1 20-policy envelope, about `3.54%`
  relative, with CI `[0.008191, 0.009646]`.
- The 27-policy selector benchmark completed and produced useful suitability
  signals, but the learned top-1 selector did not meaningfully capture the
  V1-to-V2 oracle-envelope gain on held-out OOD.
- Naive rank mixtures, dense composition, and the native component-wise pilot
  did not beat discrete selection or expand the frontier.
- Single-module structural interventions produced sparse positive transfer and
  occasional envelope expansion, but module-credit learning generalization is
  still weak.
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
| 7 | Workload/domain diversity | Medium-low | More raw data alone is insufficient; SwissAI and TraceLab are diverse but non-discriminative under current simulator semantics. |
| 8 | Selector/model capacity or sample size | Low for next action | Model choice is secondary until rewards and pressure produce stable learnable separation. |

## Current Research Posture

Do not spend the next major effort on:

- broad selector model sweeps;
- more generic dataset ingestion;
- dense weighted policy averaging;
- unrestricted structural synthesis;
- broad module-intervention expansion.

The next high-value step is simulator calibration and discriminative-power
validation. The project should make KV/cache reuse, prefill/decode contention,
capacity pressure, and SLO feasibility affect simulator state transitions and
ANWG in scientifically defensible ways, then rerun bounded subsets before
training selectors or combiners.
