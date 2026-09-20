# Latest Literature And FGCS Manuscript Differentiation Audit V1

Audit date: 2026-09-19

Repository: `SoroushVahidi/llm-serving-heuristic-evolution`

Branch: `contextual-compositional-heuristics-20260731`

Starting HEAD: `540aeedaff1fc53f79cb034899859e162dd9c242`

Phase-D preregistration commit: `540aeedaff1fc53f79cb034899859e162dd9c242`

Phase-D outcomes accessed: NO

## Preflight

Local and remote branch heads matched at audit start:

- local `HEAD`: `540aeedaff1fc53f79cb034899859e162dd9c242`
- `origin/contextual-compositional-heuristics-20260731`: `540aeedaff1fc53f79cb034899859e162dd9c242`

The repository had pre-existing dirty/untracked files outside this audit. This audit does not modify Phase A, Phase B, or Phase D preregistration results.

Phase-B verification:

- `PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv`: 1,080 data rows.
- Valid Phase-B canonical disagreement states reproduced from existing artifacts: 11,328.
- Validity distribution: 926 `VALID_UNCONSTRAINED`, 38 `VALID_PARTIALLY_CONSTRAINED`, 96 `VALID_STRONGLY_CONSTRAINED`, 20 `INVALID_HORIZON_TRUNCATED`.
- Phase-D eligible state manifest rows: 11,328 states.
- Phase-D eligible branch manifest rows: 12,169 non-SBS branches.

## Manuscript Lineage

The canonical existing manuscript is:

`paper/llm2026/main.tex`

Supporting package files:

- `paper/llm2026/main.pdf`
- `paper/llm2026/submission_package/source/main.tex`
- `docs/current/llm2026_release_manifest_20260825.md`
- `docs/current/llm2026_submission_checklist_20260825.md`

Release manifest date: 2026-08-25.

Submission evidence:

- Commit `efc24cc` on 2026-09-04 corrected repository wording to state that the manuscript was submitted to LLM 2026 on 2026-08-25.
- The final compliance branch `llm2026-final-compliance-20260825` records the LLM submission-compliance lineage.

Withdrawal evidence:

- No local repository commit, status file, or manuscript artifact containing explicit withdrawal evidence was found.
- The audit records withdrawal as user-provided project context, not locally verified provenance.

No more advanced FGCS-specific manuscript source was found. Therefore:

CANONICAL_FGCS_BASELINE_MANUSCRIPT = `paper/llm2026/main.tex`

## Current Manuscript Story

The current LLM manuscript argues that LLM-serving scheduler portfolios exhibit an exploitability gap: offline portfolio complementarity and VBS headroom exist in controlled/joint synthetic workloads, but tested online selectors and constructive mechanisms fail to recover much of that headroom. It uses SBS, VBS, a six-policy P6 portfolio, scenario-level selectors, live routing, terminal one-step counterfactuals, public trace replay, and a local vLLM validation probe.

That story remains valuable but is no longer the best FGCS story. Phase A/B shifted the strongest evidence toward production-derived action opportunity:

- Native faithful production replay is structurally scheduler-null under abundant resources.
- Arrival scaling through 8x remains action-null.
- KV and active-sequence pressure induce genuine canonical SBS-vs-P6 disagreement.
- Phase D is preregistered to test whether those disagreements have causal headroom.

## Literature Search Coverage

Primary period: 2024-01-01 through 2026-09-19.

Venues and sources searched: OSDI, NSDI, SOSP-adjacent systems proceedings, ASPLOS, MLSys, FAST, EuroSys/ATC where relevant, HPCA/ISCA/MICRO where serving/resource work was relevant, SIGCOMM-oriented routing/load-balancing sources, NeurIPS/ICML/ICLR systems-adjacent work, FGCS/high-quality journals, arXiv, USENIX proceedings, MLSys proceedings, ACL Anthology, project artifact audits, and citation/reference trails from closest works.

Candidate papers screened: 41.

Deeply audited works recorded in the evidence matrix: 18.

The exact title "Load balancing for large language model inference serving: A control and learning oriented survey" was searched but not found in indexed public sources. The closest surveyed materials found were `A Survey of LLM Inference Systems`, `Taming the Titans`, `Dynamic Model Routing and Cascading for Efficient LLM Inference`, and the September 2026 production-trace/load-balancing paper `A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing`.

## Closest Work Findings

ServeGen establishes production workload characterization and realistic workload generation at cloud scale. It substantially narrows any claim that "industry realism" itself is new. It does not measure executable scheduler disagreement or one-step default-relative causal headroom.

LMetric is the closest scheduler paper. It studies KV-aware routing/load balancing, real chatbot and coding-agent workloads, production deployment, and mathematical failure conditions. It overlaps with resource-pressure/failure-regime thinking. It does not ask how often a strong default and an alternative scheduler produce different canonical executable actions, nor does it force exactly one alternate scheduling action and revert to default continuation.

A Year in LLM Serving is close on production traces, cache locality, and load-balancing tension. It strengthens the need to cite workload evolution, prefix caching, and routing-induced cache duplication. It does not characterize default-relative scheduler action opportunity or causal headroom.

OpenTela establishes an operational production trace/control-plane contribution. It makes production trace release less unique, but does not overlap with the causal/action-level question.

Libra, Llumnix, MorphServe, Strata, Mooncake, FastServe, Sarathi-Serve, DistServe, and SOLA show that adaptive/resource-aware LLM serving is already a crowded and strong area. They mostly propose systems and evaluate whole-policy performance under selected regimes. They do not center "when adaptation is unnecessary" as a prevalence/causal characterization.

## Novelty Tests

Candidate novelty A: canonical action-opportunity prevalence.

Result: no audited prior work systematically measures how often a strong default scheduler and alternatives produce distinct executable canonical actions rather than different scores or whole-policy performance.

Candidate novelty B: default-relative one-step causal headroom.

Result: no audited prior LLM-serving work estimates `Q_default(s,a) - Q_default(s,default)` by forcing exactly one scheduling action and reverting to a fixed default continuation.

Candidate novelty C: pressure to disagreement transition.

Result: LMetric, MorphServe, Strata, Mooncake, and A Year in LLM Serving study pressure/cache/load effects, but none maps resource binding to canonical scheduler disagreement prevalence on fixed production-derived traces.

Candidate novelty D: disagreement to beneficial-headroom transition.

Result: no direct prior found. This is Phase-D's central role.

Candidate novelty E: when adaptation is unnecessary.

Result: closest works discuss failure conditions or selected regime performance, but do not make null action-support/headroom regimes a central research object.

Candidate novelty F: transferability of local exploitability.

Result: model-routing literature studies transfer/generalization, but the failed fresh selector confirmation should be supporting caution, not a central novelty claim.

NOVELTY_DIFFERENTIATION = MODERATE

## LMetric Mandatory Audit

An OSDI reviewer familiar with LMetric would not accept novelty claims about KV-aware scheduling, load-aware routing, simple score design, production deployment, chatbot/coding-agent evaluation, or failure-condition analysis.

What remains genuinely distinct is the measurement question. LMetric asks how to route/schedule better with a simple KV/load score and when that score can fail. This project asks whether, in a fixed production-derived workload regime, the default and alternative schedulers even expose different canonical actions, and whether those differences have one-step default-relative causal value. LMetric's failure conditions overlap with "when adaptation may matter," but not with `P(D)`, `P(B|D)`, canonical action collapse, or single-intervention causal headroom.

## ServeGen Mandatory Audit

ServeGen owns much of the production workload realism/generation space. The revised manuscript must cite it prominently and avoid claiming generic industry-realistic workload characterization as novelty.

The additive chain is:

production-derived workload -> resource binding -> canonical scheduling opportunity -> causal local headroom -> later predictability/generalization.

ServeGen covers the first link and benchmark implications. It does not cover the remaining action/causal decomposition.

## P6 Portfolio Audit

P6 remains scientifically sufficient for the preregistered SBS-vs-P6 characterization because the claim is explicitly default-relative and portfolio-scoped.

P6 is not sufficient for a broad claim about all modern 2026 scheduler classes. LMetric, Strata, MorphServe, Libra, FastServe, and SOLA are modern classes that must be discussed. A bounded LMetric-style extension is optional for a later preregistered phase if the manuscript wants a broader modern-portfolio opportunity claim.

P6_PORTFOLIO_STATUS = SUFFICIENT

## Phase-D Decision

PHASE_D_V1_STATUS = EXECUTE_AS_PREREGISTERED

Rationale: the latest literature narrows the contribution but does not undermine the Phase-D causal-headroom question. Adding a new policy now would change the disagreement population and require a new support phase. Since V1 is scoped to SBS-vs-P6 and no Phase-D outcomes have been accessed, execution as preregistered is appropriate.

## Required Baselines

MUST RUN before Phase D: none.

MUST DISCUSS: LMetric, ServeGen, A Year in LLM Serving, OpenTela, Libra, Llumnix, MorphServe, Strata, Mooncake, FastServe, Sarathi-Serve, DistServe, SOLA, BurstGPT, Azure/DynamoLLM.

OPTIONAL later: LMetric-style multiplicative KV/load policy, FastServe-style preemptive policy, SOLA-style state-aware policy. Each would require a separately preregistered support/opportunity extension.

## FGCS Readiness Checkpoint

Dimension scores:

- Scientific novelty: 17/20
- Industry realism: 15/20
- Technical depth: 14/20
- Experimental rigor: 16/20
- Practitioner value: 7/10
- Reproducibility/community value: 8/10

FGCS_CONTRIBUTION_READINESS_SCORE = 77/100

FGCS_CONTRIBUTION_STRENGTH_CONFIDENCE = 70%

Hard gates:

- Real-world evidence: PARTIAL, strengthened by Phase A/B production-derived traces but still simulation/replay.
- Systems-regime characterization: PASS after Phase B.
- Causal headroom: PENDING until Phase D.
- Literature novelty: PASS with narrowed claims.
- Practitioner value: PARTIAL until Phase D reports headroom.
- Reproducibility: PASS for artifacts; public packaging still needs final polish.

Shortest path to >90%:

1. Execute Phase D V1 as preregistered and report causal headroom without selector training.
2. Rewrite manuscript around Phase A/B/D, not the LLM 2026 selector story.
3. Add current literature citations and remove unsafe firstness/generalization language.
4. Decide whether a bounded modern-policy extension is needed only after Phase D and reviewer-risk assessment, not before V1 execution.

## Artifact Index

- `docs/current/FGCS_MANUSCRIPT_CLAIM_MAP_V1.md`
- `docs/current/FGCS_MANUSCRIPT_TRANSFORMATION_PLAN_V1.md`
- `experiments/literature_differentiation_v1/LATEST_LITERATURE_EVIDENCE_MATRIX_V1.csv`
- `experiments/literature_differentiation_v1/LATEST_LITERATURE_EVIDENCE_MATRIX_V1.json`
- `experiments/literature_differentiation_v1/CLOSEST_WORK_COMPARISON_V1.csv`
- `experiments/literature_differentiation_v1/REQUIRED_BASELINES_V1.json`
- `experiments/literature_differentiation_v1/NOVELTY_GATE_V1.json`

## Exact Next Task

Execute the frozen Phase-D V1 causal-headroom campaign exactly as preregistered at commit `540aeedaff1fc53f79cb034899859e162dd9c242`, after first running the preregistered causal-correctness tests and without adding selector training or new policy classes.
