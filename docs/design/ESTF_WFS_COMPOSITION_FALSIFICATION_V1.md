# ESTF↔WFS Minimal Composition Falsification (v1)

**Date:** 2026-08-16  
**Status:** COMPLETE + ANALYZED — verdict `SELECTION_SUFFICIENT_FOR_THIS_PAIR`  
**Parents:** `estimated_service_time_first` (ESTF), `weighted_fair_share` (WFS)  
**Corpus:** Family A v2 Job 1182377  
**Primary metric:** canonical `arrival_normalized_weighted_goodput`

## Scientific question

Does a simple **ranking-only** composition of ESTF and WFS improve held-out
canonical ANWG and parent-envelope gain beyond a contextual top-1 selector that
chooses ESTF or WFS?

## Composition semantics

For waiting set \(Q\):

1. Obtain ESTF and WFS orderings via `rank_with_named_expert`.
2. Convert each ordering to normalized ranks in \([0,1]\) (best = 1).
3. Aggregate: \(\mathrm{score}(r)=\alpha\cdot\mathrm{rank}_{ESTF}(r)+(1-\alpha)\cdot\mathrm{rank}_{WFS}(r)\).
4. Sort by score desc, then support, arrival, request_id.
5. Project through existing `deterministic_place` (admission/placement unchanged).

No raw-score averaging. No changes to batching/KV/preemption modules.

## Treatments

| ID | Method |
|---|---|
| A | Contextual top-1 selector (scenario-level features → ESTF or WFS) |
| B | Static blends \(\alpha\in\{0.25,0.50,0.75\}\) |
| C | Contextual discrete \(\alpha(x)\in\{0,0.25,0.5,0.75,1\}\) |
| D | Hard conditional if/else (optional, simple) |

## Features (observable only)

Scenario summaries from request fields only: queue size, prompt/output stats,
estimated service stats, priority skew, class imbalance, SLO slack, arrivals.
**Forbidden:** favored size, util, skew treatment, noise label, seed, scenario_id.

## Splits

- TRAIN: seed 20260816, exclude OOD  
- VAL: seed 20260817, util∈{1.1,1.3}, exclude OOD  
- TEST: seed 20260817, util=1.5, exclude OOD  
- OOD: favored=long ∧ skew=10 (both seeds)

## Decision rule

On TEST:

- `COMPOSITION_GO` if contextual composition beats top-1 by ≥0.01 ANWG **and**
  envelope-gain CI excludes ≤0 **and** alpha is not collapsed to {0,1}.
- `SELECTION_SUFFICIENT_FOR_THIS_PAIR` if top-1 matches/beats composition or
  alpha collapses to hard selection.
- `INCONCLUSIVE` otherwise.

## Non-goals

MAP-Elites, symbolic distillation, LLM APIs, multi-policy dense mixtures.
