# Composition and Synthesis Architecture

> **Pause addendum 2026-07-25.** The repaired load-discrimination pilot (`PARTIALLY_READY`) does **not** justify reopening composition or synthesis work. Native composition pilot remains `NO_GO` with verified-readable artifacts; structural synthesis remains empirically `NOT_READY`. Prioritize simulator/load discrimination on natural evidence first.


This document is the concise source-of-truth overview for the path:

```text
Policy Library -> Discrete Selection -> Contextual Composition -> Structural Synthesis
```

It summarizes what is implemented, what is prototype-only, and what is
currently blocked by simulator/objective discriminative power.

## Implemented Components

### Policy Library

`src/llmserveopt/policies/registry.py` exposes two deployable libraries:

- `BASELINE_NAMES`: historical 20-policy library.
- `POLICY_LIBRARY_V2_NAMES`: historical library plus 7 new deployable approximation policies.

Policy Library v2 is implemented and unit-tested. Real-OOD evidence shows the
V2 library materially improves the oracle envelope, but SwissAI and TraceLab
also show that raw workload novelty can collapse to near-tie policy rewards
under the current simulator/objective.

### Discrete Selection

The existing selector path chooses one complete deployable policy per window/context. Prior Selector v2/v3 evidence shows:

- adaptive selection can beat WSP in-distribution on the 1600-window Selector v2 dataset;
- robust held-out/OOD performance still tends to favor fixed WSP;
- richer causal features make WSP-vs-SCORPIO behavior more learnable, but have not yet produced a selector that consistently beats WSP across held-out domains.

### Rank Composition

`src/llmserveopt/policies/composition.py` implements typed composition scaffolding:

- `RankExpertSpec`
- `RankExpertOutput`
- `StaticRankEnsemblePolicy`
- `ContextualRankEnsemblePolicy`
- normalized rank/percentile aggregation
- deterministic tie handling
- sparse top-k expert support
- fallback policy handling
- contribution, entropy, fallback, and switching logs

Raw score averaging is not the default and is not considered generally valid because policies expose heterogeneous quantities. Normalized rank aggregation is the safest common representation.

Two operator families were added on top of this scaffolding (2026-07-24), see `docs/current/COMPOSITION_IMPLEMENTATION_STATUS.md` addendum for detail:

- `StaticRankEnsemblePolicy(method="reciprocal_rank")` / `composition.weighted_reciprocal_rank_aggregate` — weighted reciprocal-rank fusion as an alternative to Borda-style normalized-rank aggregation.
- `score_aggregation.py` — normalized weighted score aggregation (`none`/`min_max`/`zscore`/`robust_mad`) for the subset of policies (`capabilities.SCORE_CAPABLE_EXPERTS`) that expose a genuine single comparable scalar, distinct from rank-only experts.

### Component-Wise Composition

`ComponentWiseCompositionPolicy` prototypes a conservative combination of:

- SCORPIO-style admission filtering;
- WSP-style priority;
- KV, prefill, and aging safeguards when expressible in the current action space.

This is a prototype, not proof that the component combination is scientifically superior.

### Experiment Harness

`src/llmserveopt/selector/composition_experiment.py` provides:

- upstream final-report readiness checks;
- policy-vector CSV loading;
- split-group leakage validation;
- development-only best-fixed selection;
- guards that prevent held-out split use during treatment selection.

`tools/composition_smoke_experiment.py`, `tools/native_composition_pilot.py`, and sbatch wrappers provide smoke and pilot execution paths.

### Structural Synthesis

`src/llmserveopt/policies/genome.py` defines `SchedulerGenomeV1`, a typed genome wrapper around the verified heuristic DSL. It supports:

- canonical JSON serialization;
- deterministic SHA256 hashing;
- reproducible parsing;
- semantic validation;
- causal feature whitelist enforcement;
- conversion to verified heuristic policies.

`src/llmserveopt/policies/structural_synthesis.py` implements:

- representative parent-policy mappings;
- module swap;
- conditional regime composition;
- typed subtree crossover;
- bounded constant mutation;
- whitelisted feature/operator mutation;
- development-only frontier-value scoring;
- structured prompt-template rendering for future LLM-guided synthesis without calling an LLM.

`src/llmserveopt/selector/parent_selection.py` implements deterministic parent-pair scoring and a composition gate.

## Validation Status

- Composition harness: 37/37 tests previously passed under SLURM job `1119434`.
- Structural synthesis harness: 14/14 tests previously passed under SLURM job `1120181`.
- Native composition pilot: job `1120123` completed with `NATIVE_COMPOSITION_PILOT_DECISION = NO_GO`. **Precision caveat (2026-07-24 audit):** this job's raw artifacts (`native_composition_pilot_20260721T194929Z/*`) are cluster-only and were not present in the checkout used for that audit -- only the qualitative decision string was verifiable locally, not the underlying ANWG/CI/regret numbers. Treat the NO_GO conclusion as real-harness Level B evidence, not an independently re-derived Level C/D falsification, until those artifacts are recovered into `results/wulver_imports/`.

The native pilot found that static/contextual rank mixtures and component-wise composition did not clear the decisive held-out meaningful-window bar against the discrete selector. That argues against launching a full naive composition sweep immediately, though the current overall recommendation is BLOCKED (pending read-only artifact recovery), not a fully-settled STOP -- see `docs/current/WOLVERINE_ORACLE_MIXTURE_HANDOFF.md`.

## Current Scientific Status

The full composition experiment should not be launched from the current
evidence.

Completed evidence now indicates:

- naive/native rank composition did not beat discrete selection or expand the
  frontier;
- single-module interventions contain sparse positive transfer, but module
  credit is weakly learnable;
- pairwise module combinations did not expand the 27-policy envelope;
- the selector suitability vector is useful but not yet reliable enough for
  unrestricted donor/module selection;
- `COMBINER_TRAINING_SIGNAL = WEAK`;
- `COMBINER_EVALUATION_READINESS = NEEDS_SIMULATOR_FIX`.

## What Is Not Implemented

The current simulator cannot faithfully execute:

- prefix-cache reuse policies;
- cache-loading-aware policies;
- disaggregated prefill/decode routing;
- request splitting;
- heterogeneous GPU routing;
- exact chunk-size prefill actions;
- durable multi-tenant credit state.

Those capabilities require simulator/action-space extensions before they can be used in policy composition or structural synthesis.

## Current Recommendation

**Overall decision (2026-07-24 audit): BLOCKED**, not a fully-settled STOP.
Do not launch the full Wolverine/Wulver composition sweep from this checkout
yet. Upstream frontier/library workflows are `COMPLETE` on Wulver, but job
`1120123`'s raw numeric artifacts remain cluster-only and were not present in
the non-Wulver checkout used for the audit (see the precision caveat above and
`docs/current/WOLVERINE_ORACLE_MIXTURE_HANDOFF.md`). The immediate operational
next action for composition evidence is read-only Wulver artifact recovery, not
a new experiment. Weighted reciprocal-rank and normalized score aggregation are
implemented and unit-tested, but they are correctness-only so far and lack
large-scale performance validation.

Independently, module-level/structural credit assignment is also `NOT_READY` by
direct empirical measurement (not just by the harness-readiness sense in
`STRUCTURAL_SYNTHESIS_READINESS.md`): the overnight module-credit search's best
model had `top1_beats_both_parents_fraction = 0.0` and
`expands_envelope_fraction = 0.0` at every top-k (see
`results/module_credit_overnight/module_credit_overnight_20260722T000121/final_report.md`
and
`results/module_credit_report/real_wulver_20260721T224322Z/real_module_credit_report.md`,
generated by `scripts/run_module_credit_overnight.py` and
`scripts/run_real_module_credit_evaluation.py`). That is evidence -- not yet
conclusive -- that blind structural recombination is not obviously better
positioned than output-level composition.

Broader scientific posture (post-SwissAI/TraceLab/simulator audit): do not
launch broad composition, unrestricted structural synthesis, or large-scale
module-credit expansion as the next scientific step. First run bounded
simulator calibration and discriminative-power validation so KV/cache reuse,
prefill/decode contention, overload, and SLO pressure produce scientifically
meaningful policy-reward differences. After that, recover the native-pilot
numeric artifacts, rerun controlled subsets, retrain suitability models, and
only then resume restricted evidence-guided combination/synthesis.
