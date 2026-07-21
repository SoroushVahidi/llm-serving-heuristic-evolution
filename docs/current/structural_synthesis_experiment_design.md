# Structural Synthesis Experiment Design

## Purpose

Use completed development evidence to generate new symbolic scheduler children from high-value parent policies. This is a successor to naive weighted composition, not a replacement for leakage-safe evaluation.

## Inputs

- Policy Frontier Cartography final report and development artifacts.
- Policy Library v2 final report and 27-policy complementarity artifacts.
- Native Composition Pilot report.
- Existing causal feature whitelist and verified heuristic DSL.

Do not use final OOD labels for parent selection, child design, constant tuning, or gate thresholds.

## Pipeline

1. Select parent candidates from development evidence.
2. Apply the composition gate:
   - `SELECT_SINGLE` when top-1 is confident or modules are incompatible.
   - `ATTEMPT_STRUCTURAL_COMPOSITION` when complementarity, uncertainty, or marginal frontier value justifies synthesis.
3. Generate a small child set using:
   - module swap;
   - conditional regime composition;
   - typed subtree crossover;
   - bounded constant mutation;
   - causal feature/operator mutation.
4. Verify every child:
   - parse;
   - type check;
   - causal whitelist check;
   - heuristic verifier;
   - tiny simulator smoke.
5. Score candidate children on development-only frontier value.
6. Freeze a small child set.
7. Evaluate frozen children on held-out ID/OOD splits using existing leakage-safe protocol.

## Initial Parent Sets

- WSP + SCORPIO-style
- WSP + aging_priority
- SCORPIO-style + kv_constrained_online
- SCORPIO-style + adaptive_chunked_prefill

## Initial Child Operators

- SCORPIO admission + WSP priority.
- If SLO/KV pressure is high, use SCORPIO-like priority; otherwise use WSP.
- WSP priority with aging fairness module.
- SCORPIO-like priority with KV-constrained admission.
- Small constant perturbations around development-selected thresholds.

## Metrics

- ANWG.
- Marginal frontier value on development data.
- Unique win count.
- Meaningful unique win count.
- Regret-profile novelty.
- Complexity penalty.
- Feasibility violations.
- SLO violation.
- Completion fraction.
- Held-out oracle regret.

## Stop Rules

Do not launch large child-generation or evaluation sweeps if:

- native component composition is still below discrete selector;
- Policy Library v2 shows no meaningful frontier gaps;
- generated children fail verification frequently;
- frontier gains are confined to development data.

## Success Criteria

A structural child is worth scaled evaluation only if it:

- passes verifier and smoke tests;
- has positive development marginal frontier value;
- has meaningful unique wins in a pre-registered frontier niche;
- is not simply a renamed parent;
- does not require unsupported simulator semantics.
