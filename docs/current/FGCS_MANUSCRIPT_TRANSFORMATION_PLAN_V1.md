# FGCS Manuscript Transformation Plan V1

Audit date: 2026-09-19

Phase-D outcomes accessed: NO

Canonical baseline: `paper/llm2026/main.tex`

## Proposed FGCS Thesis Options

1. Production-derived LLM-serving traces can be structurally scheduler-null under native resource abundance, and scheduler action opportunity emerges only after concrete KV or active-sequence pressure crosses workload demand.
2. Adaptive scheduling should be evaluated as a chain of support, causality, and predictability: `P(D)` says whether different executable actions exist, `P(B|D)` says whether they help locally, and learning is only meaningful after both are nontrivial.
3. Modern LLM-serving adaptation is workload-regime dependent: pressure can create canonical scheduler disagreement, but only default-relative causal headroom can tell whether that disagreement is actionable.

## Title Candidates

- When Does LLM-Serving Scheduler Adaptation Matter?
- Action Opportunity and Causal Headroom in Production-Derived LLM Serving
- From Resource Pressure to Scheduler Headroom in LLM Serving
- Measuring When Adaptive LLM-Serving Schedulers Have Actionable Choice
- Default-Relative Causal Headroom for Resource-Constrained LLM Serving

## Proposed Research Questions

RQ1. Under faithful production-derived replay, how often does SBS-vs-P6 scheduler choice create genuinely different executable canonical actions?

Evidence: Phase A native replay, zero canonical disagreement across the same 60 faithful windows.

RQ2. How does one-axis resource pressure govern the transition from scheduler agreement to canonical disagreement?

Evidence: Phase B V2 pressure map, including arrival-null behavior and KV/active-sequence onset.

RQ3. When canonical disagreement exists, how often is the SBS action locally causally suboptimal and how large is one-step SBS-relative oracle headroom?

Evidence: Phase D V1 preregistered but not executed.

RQ4. Is causal headroom broad and stable enough to justify later predictor/selector work?

Evidence: Not answered by Phase D alone; Phase D can decide whether a later predictability phase is warranted.

## Retain Largely Unchanged

- Definitions of SBS/VBS/headroom/gap closure, with updated notation for `P(D)` and `P(B|D)`.
- Description of the P6 policies as the historical portfolio, while clarifying it is not an exhaustive 2026 scheduler set.
- The caution that offline envelope opportunity does not imply deployable adaptive gain.
- Reproducibility/data availability statements, updated for Phase A/B/D artifacts.

## Retain But Shorten

- Controlled A/B/C complementarity.
- Joint-240 VBS headroom.
- Same-distribution HGB selector result.
- Terminal counterfactual criticality on joint-240.
- Constructive exploitation beyond selection.
- Local vLLM semantic mismatch and token-budget tradeoff.

These become context for why the project pivoted from "can we exploit synthetic portfolio headroom?" to "does real production-derived traffic even offer executable scheduler alternatives, and when?"

## Rewrite

- Abstract: replace learned-selector/exploitability-gap framing with production-derived regime characterization and preregistered causal-headroom framing.
- Introduction: start from modern production LLM-serving literature, not from generic portfolio selection.
- Contributions: center the chain `resource pressure -> canonical disagreement P(D) -> causal headroom P(B|D)`.
- Related work: split into production workload characterization, adaptive/resource-aware serving systems, KV/cache/load balancing, and causal/action-level scheduler evaluation.
- Limitations: explicitly state simulation/replay boundaries, P6 scope, and no deployment effect.

## Replace

- Replace "public-trace saturation sanity check" with Phase A faithful production replay as a primary result.
- Replace synthetic joint-240 centrality with Phase B V2 production-derived one-axis pressure map.
- Replace any selector-generalization optimism with the failed fresh confirmation as cautionary background.
- Replace broad "adaptive scheduling opportunity" language with denominator-specific `P(D)` and `P(B|D)` language.

## Add

- Phase A design/results: exact 60 faithful windows, zero native canonical disagreement, resource telemetry showing native abundance.
- Phase B V2 design/results: 1,080 one-axis conditions, validity classes, 11,328 valid disagreement states, onset/binding map.
- Phase D V1 preregistration before results: one-step SBS-relative estimand, selected causal universe, exhaustive plan, confidence protocol.
- Literature differentiation matrix and claim boundary.
- Practitioner tuple: `(P(D), P(B|D), mean_oracle_headroom)` with binding/pressure context after Phase D.

## Remove Or Move To Appendix

- Full controlled-family selector history unless needed for motivation.
- Typed synthesis details.
- Support expansion and guarded composition details.
- Extended joint-240 tables that do not help the production-derived FGCS thesis.
- Any language that implies a learned selector is validated for fresh production distributions.

## Role Of Failed V2 Confirmation

Correct role: methodological caution.

It should not be a major contribution. It supports the discipline of separating action support, causal headroom, and predictability, and justifies why Phase D does not train a selector.

## Required Modern Baseline Position

No modern system must be implemented before Phase D V1 if claims remain scoped to SBS-vs-P6 default-relative headroom.

Must discuss: LMetric, ServeGen, OpenTela, A Year in LLM Serving, Libra, Llumnix, MorphServe, Strata, Mooncake, FastServe, Sarathi-Serve, DistServe, SOLA.

Optional future baseline: an LMetric-style KV/load policy in a separately preregistered support phase if the paper later wants a broader modern scheduler-portfolio claim.

## Phase D Decision

PHASE_D_V1_STATUS = EXECUTE_AS_PREREGISTERED

Reason: the literature narrows but does not invalidate the causal-headroom question. No audited work measures canonical SBS-vs-alternative action prevalence and one-step default-relative causal advantage under production-derived constrained replay. Adding modern policies would change the `D` population and belongs to a future, separately preregistered extension, not a silent edit to V1.
