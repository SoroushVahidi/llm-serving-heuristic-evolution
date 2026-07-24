# Structural Synthesis Readiness

This readiness layer prepares the next direction after naive rank/weight composition: state-dependent parent/module selection, structural symbolic child synthesis, verification, and frontier-gain evaluation.

It does not call an LLM and does not launch a large simulation sweep.

## Current Caveat

The implementation scaffold remains useful, but broad structural synthesis is
not scientifically ready as the next action. Completed module-credit and
simulator-discriminative audits show:

- module-credit learning is weak/generalization-limited;
- pairwise module combinations did not expand the 27-policy envelope;
- SwissAI and TraceLab saturated ANWG and produced weak policy separation;
- `COMBINER_TRAINING_SIGNAL = WEAK`;
- `COMBINER_EVALUATION_READINESS = NEEDS_SIMULATOR_FIX`.

Therefore, treat `READY_WITH_SMALL_EXTENSIONS` below as **engineering
readiness of the harness**, not permission to launch unrestricted structural
synthesis before simulator calibration and controlled re-evaluation.

## Existing DSL Audit

The existing heuristic DSL can already encode:

- request ranking and priority expressions;
- admission conditions;
- simple conditional regime switching;
- aging/fairness terms based on `req.waiting_time` and `req.priority_weight`;
- aggregate KV-pressure guards through `sys.kv_utilization`;
- aggregate prefill/token-budget pressure through `sys.token_budget_utilization`;
- bounded constants and safe arithmetic;
- a causal feature whitelist through `heuristics/dsl_schema.py` and `heuristics/verifier.py`.

It cannot exactly encode:

- stateful token/credit-budget admission such as SCORPIO's budget refill;
- true chunk-size or partial-prefill actions;
- per-GPU post-placement KV routing decisions;
- cache/prefix reuse;
- cache loading;
- request splitting;
- disaggregated prefill/decode routing;
- heterogeneous hardware affinity;
- learned stateful module weights without an outer policy wrapper.

## Implemented Interfaces

- `src/llmserveopt/policies/genome.py`
  - `SchedulerGenomeV1`
  - typed optional modules: `admission_rule`, `priority_rule`, `prefill_rule`, `kv_guard`, `fairness_rule`, `regime_conditions`
  - canonical JSON serialization
  - deterministic SHA256 hash
  - reproducible parsing
  - semantic validation
  - causal feature whitelist enforcement
  - conversion into the verified heuristic DSL

- `src/llmserveopt/selector/parent_selection.py`
  - configurable parent-pair scoring
  - deterministic top-parent selection
  - composition gate returning `SELECT_SINGLE` or `ATTEMPT_STRUCTURAL_COMPOSITION`

- `src/llmserveopt/policies/structural_synthesis.py`
  - best-effort parent genome mappings
  - module swap
  - conditional regime composition
  - typed subtree crossover
  - bounded constant mutation
  - whitelisted feature/operator mutation
  - frontier-value scoring interface
  - LLM prompt/template rendering without calling external APIs

## Parent Genome Mappings

| Policy | Mapping | Notes |
| --- | --- | --- |
| `weighted_shortest_processing` | EXACT | Expressed as negative estimated service divided by priority, with existing safe expression primitives. |
| `edf` | EXACT | Expressed as minimum slack / earliest deadline ranking. |
| `aging_priority` | APPROXIMATE | Captures waiting-time age bonus but not every coefficient detail. |
| `scorpio_style_slo_guard` | APPROXIMATE | Captures slack admission, urgency/priority/age ranking, and aggregate KV guard; cannot encode stateful budget refill or per-GPU decode filters exactly. |
| `kv_constrained_online` | APPROXIMATE | Captures aggregate KV guard and urgency per KV-cost priority; not exact per-GPU post-placement KV reserve. |
| `adaptive_chunked_prefill` | APPROXIMATE | Captures prefill/token-budget pressure; no true chunk-size action. |

Unsupported families remain unsupported until simulator/action semantics expand.

## Frontier Value

The readiness interface defines:

```text
MarginalFrontierValue(child) = mean(max(envelope(P), child) - envelope(P)) - complexity_penalty
```

It also reports:

- unique win count;
- meaningful unique win count;
- mean gain on wins;
- complexity penalty.

This must be computed on development data only before any held-out evaluation.

## LLM-Guided Synthesis

`render_llm_synthesis_prompt` creates a structured future request containing:

- target workload niche;
- parent genomes;
- parent strengths;
- pairwise advantage evidence;
- frontier gap;
- allowed primitives;
- forbidden features;
- output contract.

No LLM call is made by this harness.

## Readiness Status

STRUCTURAL_SYNTHESIS_READINESS = READY_WITH_SMALL_EXTENSIONS

GENOME_SCHEMA = IMPLEMENTED

EXISTING_POLICIES_MAPPED = weighted_shortest_processing, edf, aging_priority, scorpio_style_slo_guard, kv_constrained_online, adaptive_chunked_prefill

EXACT_MAPPINGS = weighted_shortest_processing, edf

APPROXIMATE_MAPPINGS = aging_priority, scorpio_style_slo_guard, kv_constrained_online, adaptive_chunked_prefill

UNSUPPORTED_MAPPINGS = cache_prefix_reuse_aware, cache_loading_aware, disaggregated_prefill_decode_routing, request_splitting, heterogeneous_gpu_routing

PARENT_SELECTION_INTERFACE = IMPLEMENTED

COMPOSITION_GATE = IMPLEMENTED

MODULE_SWAP = IMPLEMENTED

CONDITIONAL_COMPOSITION = IMPLEMENTED

TYPED_SUBTREE_CROSSOVER = IMPLEMENTED

MUTATION_OPERATORS = bounded_constant_mutation, causal_feature_or_operator_mutation

FRONTIER_VALUE_SCORING = IMPLEMENTED

SMOKE_TEST_STATUS = PASS

TESTS_PASSED = 14/14 via SLURM job 1120181

LLM_PROMPT_TEMPLATE_READY = YES

MAIN_REPRESENTATION_LIMITATION = current DSL compiles to score/admission heuristics only; it cannot encode stateful credit budgets, true chunked prefill actions, per-GPU placement/KV routing, cache reuse, request splitting, or heterogeneous/disaggregated routing exactly

MAIN_SCIENTIFIC_RISK = structurally valid children may only reproduce approximations of high-value parent behavior and may not improve held-out/frontier performance once verified against real policy vectors

RECOMMENDED_NEXT_ACTION = first perform simulator calibration and discriminative-power validation; after bounded reruns produce reliable policy separation, use development-only frontier gaps and suitability uncertainty to generate a small restricted verified child set before any large evolutionary run
