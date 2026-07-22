# Research Roadmap

Current roadmap as of 2026-07-22.

## Stage 1 - Repository Consolidation

Status: current polish pass.

Goals:

- keep validated source code and documentation coherent;
- preserve negative scientific evidence;
- make the current bottleneck obvious to the next researcher or agent;
- avoid changing simulator semantics during documentation cleanup.

## Stage 2 - Simulator Calibration and Discriminative-Power Validation

Status: highest priority.

Scientific question: can the simulator and ANWG objective translate important
workload differences into meaningful resource pressure and policy-reward
separation?

Required subgoals:

- strengthen and validate KV/cache coupling;
- model prefix reuse effects on actual prefill/service cost where justified;
- strengthen KV occupancy and resource-pressure semantics;
- validate prefill/decode contention;
- validate capacity and overload pressure;
- calibrate SLO feasibility effects;
- audit ANWG/objective ceiling behavior and decide whether auxiliary metrics are
  needed in saturated regimes.

Go/no-go criterion: bounded diagnostic windows should show policy separation
for scientifically defensible reasons when KV/cache, phase, overload, or SLO
pressure is intentionally present.

Compute target: Wulver CPU/Slurm for bounded simulator sweeps; no GPU unless
real-backend calibration is explicitly part of the task.

## Stage 3 - Small Controlled Re-Evaluation

Status: pending Stage 2.

After simulator fixes, rerun bounded subsets of:

- V2 real-OOD;
- SwissAI;
- TraceLab;
- SLO/deadline augmentation.

Go/no-go criterion: policy separation should improve in the regimes the
simulator claims to model, without fabricating natural labels that are not
observed.

## Stage 4 - Suitability Model Retraining

Status: blocked until Stage 3 passes.

Use:

- full 27-policy reward vectors;
- regret-aware/listwise objectives;
- near-tie handling;
- grouped OOD robustness;
- uncertainty and abstention;
- synthetic SLO augmentation only as labeled training/regime-probing support.

Go/no-go criterion: learned selectors should capture a meaningful fraction of
the V2 oracle gain on held-out OOD and improve ranking/suitability quality, not
only ID top-1 accuracy.

## Stage 5 - Targeted Module-Credit Refresh

Status: blocked until reliable suitability/uncertainty exists.

Use reliable selector uncertainty and suitability vectors to choose:

- states;
- donor policies;
- base policies;
- module types;
- frontier/pressure regimes.

Go/no-go criterion: the module-credit model must beat simple donor-whole-policy
and structural-nearest baselines on held-out decision quality.

## Stage 6 - Restricted State-Conditioned Combination/Synthesis

Status: long-term main contribution, not ready for broad launch.

Allowed only when evidence supports:

- donor selection;
- module transfer;
- typed module compatibility;
- restricted structural combinations;
- simulator pressure that can fairly evaluate the child policy.

Do not use unrestricted structural synthesis while `COMBINER_TRAINING_SIGNAL =
WEAK` and `COMBINER_EVALUATION_READINESS = NEEDS_SIMULATOR_FIX`.

## Stage 7 - External-Baseline Comparison

Status: pending a validated adaptive/synthesized method.

Compare final methods fairly against external scheduling baselines only after
the internal evaluation system is calibrated enough to distinguish meaningful
policy behavior.

## Stage 8 - Real Backend/API Validation

Status: later validation.

Use Azure/Gemini/Cohere or other available backends where scientifically useful
to validate simulator assumptions and selected policy behaviors. Do not use
paid APIs or real serving workloads as a substitute for fixing the simulator's
internal discriminative-power issue.

## Current Stop/Go Summary

- Full composition experiment: **STOP** until simulator calibration and
  suitability signals improve.
- Broad structural synthesis: **STOP**.
- Generic dataset ingestion: **STOP** as a primary next action.
- Selector retraining: **WAIT** until controlled re-evaluation produces
  trustworthy reward separation.
- Simulator calibration/discriminative validation: **GO**.
