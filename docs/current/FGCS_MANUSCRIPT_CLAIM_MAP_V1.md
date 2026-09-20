# FGCS Manuscript Claim Map V1

Audit date: 2026-09-19

Phase-D outcomes accessed: NO

Canonical baseline manuscript: `paper/llm2026/main.tex`

Baseline manuscript title: "The Exploitability Gap in LLM-Serving Scheduler Portfolios"

Baseline hashes:

- `paper/llm2026/main.tex`: `39bfa54b51b2c27a5c90ee141edc00959168799035cd4521cffd85125df2f857`
- `paper/llm2026/main.pdf`: `a48c824bcea566ed080a1e2658dea7a2c0acaa15e193cf99b52e57df0a5efdf1`

## Lineage

Repository evidence identifies the LLM 2026 manuscript package under `paper/llm2026/`.

- Original submission source: `paper/llm2026/main.tex`
- Submission package source: `paper/llm2026/submission_package/source/main.tex`
- Release manifest: `docs/current/llm2026_release_manifest_20260825.md`
- Submission checklist: `docs/current/llm2026_submission_checklist_20260825.md`
- Submission status correction commit: `efc24cc` on 2026-09-04, recording that the manuscript was submitted to LLM 2026 on 2026-08-25.
- Final compliance branch: `llm2026-final-compliance-20260825`

No local repository evidence of withdrawal was found by targeted searches for withdrawal-related terms in current files and commit messages. The audit therefore records withdrawal as user-provided project context rather than a locally verified repository artifact.

No later FGCS-specific manuscript source was found. The most advanced current manuscript source remains `paper/llm2026/main.tex`, but its story is not yet the right FGCS story.

## Claim Classification Summary

| Classification | Count | Interpretation |
| --- | ---: | --- |
| SUPPORTED_AND_CURRENT | 7 | Can remain with updated citations and scope language. |
| SUPPORTED_BUT_SECONDARY | 8 | Useful as historical or supporting evidence, but not the FGCS center. |
| SUPERSEDED_BY_NEW_DIRECTION | 9 | Should be replaced or reframed by Phase A/B/D action-opportunity results. |
| CONTRADICTED_BY_LATER_EVIDENCE | 3 | Must be removed or explicitly corrected. |
| NOT_YET_ESTABLISHED | 5 | Requires Phase D or later work. |
| NEEDS_LATEST_LITERATURE_RECHECK | 6 | Requires citation/wording updates before FGCS submission. |

## Claim Map

| Manuscript item | Existing claim | Existing evidence | Closest latest work | Current validity | Required change |
| --- | --- | --- | --- | --- | --- |
| Abstract: portfolio complementarity | P6 has workload-dependent complementarity and VBS headroom. | Controlled A/B/C and joint-240. | SOLA, FastServe, Llumnix, Libra. | SUPPORTED_BUT_SECONDARY | Retain as background, shorten, and do not make it the main FGCS claim. |
| Abstract: online adapters fail to recover VBS headroom | Lightweight selectors/routers underperform SBS; HGB is near-SBS but gains are tiny/CI includes zero. | Same-distribution joint-240 selector study. | Dynamic model routing survey, SOLA, LMetric. | SUPPORTED_BUT_SECONDARY | Keep as caution, but no longer central. |
| Abstract: terminal one-step counterfactuals | Terminal criticality sparse/concentrated. | Joint-240 terminal fork and TRAIN/VAL controlled study. | General counterfactual decision literature, no close LLM-serving equivalent found. | SUPPORTED_BUT_SECONDARY | Reframe as precursor to Phase-D default-relative causal-headroom design. |
| Abstract: real-vLLM validation | Simulator-engine mismatch; token-budget tradeoff exists. | Local vLLM probe. | vLLM, Sarathi-Serve, DistServe, Mooncake. | SUPPORTED_BUT_SECONDARY | Move to limitations/appendix unless FGCS includes a real-system validation section. |
| Introduction: adaptive scheduling promise | Scheduler choice can matter, but exploitability may be hard. | Synthetic and controlled workloads. | LMetric, Llumnix, Libra, MorphServe. | SUPPORTED_AND_CURRENT | Retain, but cite current adaptive systems and avoid implying they are unstudied. |
| Contribution 1: six-policy complementarity | P6 VBS headroom exists on joint-240. | Joint-240 metrics. | Modern scheduler systems. | SUPPORTED_BUT_SECONDARY | Retain as origin of portfolio, not main contribution. |
| Contribution 2: adaptive selectors | Stronger nonlinear selector recovers only about 2.5% of SBS-to-VBS headroom. | HGB same-distribution experiment. | Dynamic routing/cascading survey. | SUPPORTED_BUT_SECONDARY | Present as negative caution, not evidence against all adaptive schedulers. |
| Contribution 3: terminal counterfactual criticality | Criticality is sparse and continuation-policy conditional. | Joint-240 and TRAIN/VAL forks. | No direct LLM-serving one-step default-relative analog found. | SUPPORTED_AND_CURRENT | Keep as methodological bridge but replace terminal/live-router framing with SBS-relative Phase-D framing after execution. |
| Contribution 4: constructive exploitation beyond selection | Support expansion, guarded composition, and typed synthesis fail under criteria. | Family-A and synthesis screens. | Autopoiesis, H-MAS, hyper-heuristic literature. | SUPPORTED_BUT_SECONDARY | Move to appendix/history unless tied to action-support null results. |
| Public-trace saturation | Public replay did not expose scheduler differentiation. | Old processed public replay; Phase A later stronger: exact faithful production windows have zero native canonical disagreement. | ServeGen, BurstGPT, OpenTela, A Year in LLM Serving. | SUPERSEDED_BY_NEW_DIRECTION | Replace with Phase A native faithful production trace zero-support result. |
| Workload realism claim | Public traces are a sanity check, not main discriminating benchmark. | LLM 2026 paper mostly synthetic/joint workloads. | ServeGen, OpenTela, A Year in LLM Serving. | CONTRADICTED_BY_LATER_EVIDENCE | FGCS must make production-derived traces central, not secondary. |
| Joint-240 centrality | Joint-240 is the main discriminating workload. | Synthetic joint-240. | ServeGen and production trace papers. | SUPERSEDED_BY_NEW_DIRECTION | Demote; Phase A/B/D production-derived regimes become central. |
| Same-distribution selector result | HGB nearly matches SBS but does not recover VBS headroom. | Joint-240 cross-fitting. | Dynamic routing/cascading survey. | SUPPORTED_BUT_SECONDARY | Keep as caution; do not claim generalization. |
| Learned override generalization | Any implication that a learned selector is deployable/generalizes. | V1 positive uncertain; V2 development positive; fresh one-shot failed. | Modern routing literature. | CONTRADICTED_BY_LATER_EVIDENCE | Remove or explicitly state not established. |
| Natural OOD action support | Natural OOD support is sparse/null. | Later action-support null results. | None direct. | SUPPORTED_AND_CURRENT | Use to motivate separating opportunity from predictability. |
| Phase A production native replay | Native production-derived replay has zero canonical SBS-vs-P6 disagreement. | Phase A: 0/99,992 code; 0/461,985 conversation; 0/440,461 BurstGPT. | ServeGen, BurstGPT, OpenTela. | SUPPORTED_AND_CURRENT | Add as FGCS RQ1 result. |
| Phase B pressure transition | Disagreement emerges only under tight KV/active caps, not arrival scaling through 8x. | Phase B V2: 1,080 conditions, 11,328 valid disagreement states. | LMetric, MorphServe, Strata, Mooncake. | SUPPORTED_AND_CURRENT | Add as FGCS RQ2 result. |
| Phase D causal headroom | Beneficial one-step SBS-relative headroom exists or not. | Preregistered only; no outcomes. | No direct prior found. | NOT_YET_ESTABLISHED | Await Phase D; no result language yet. |
| P(D) denominator | Prevalence uses all SBS decision states. | Phase A/B support scanners. | No direct prior found. | SUPPORTED_AND_CURRENT | Make denominator explicit and central. |
| P(B|D) denominator | Causal population is only canonical disagreement states. | Phase-D preregistration. | No direct prior found. | NOT_YET_ESTABLISHED | Keep as design until execution. |
| Practitioner regime map | Distinguish no choice, choice/no benefit, choice/benefit. | Phase A/B plus Phase-D design. | LMetric failure conditions, A Year in LLM Serving. | NOT_YET_ESTABLISHED | Add after Phase D with outcome-safe boundaries. |
| Claim of "first" | Current wording "to our knowledge" around joint portfolio/headroom/criticality. | Literature before 2026 incomplete. | LMetric, ServeGen, OpenTela, A Year in LLM Serving, Libra. | NEEDS_LATEST_LITERATURE_RECHECK | Replace with narrow "we characterize..." wording rather than firstness. |
| Baseline sufficiency | P6 plus VTC/vLLM-style proxy is enough to rule out modern alternatives. | External Pext on joint-240 only. | LMetric, Strata, MorphServe, Libra. | NEEDS_LATEST_LITERATURE_RECHECK | Scope P6 as explicit portfolio; do not claim exhaustive modern scheduler coverage. |
| Production impact | Real production deployment improvement. | None. | LMetric, Mooncake, Strata, OpenTela have deployment evidence. | NOT_YET_ESTABLISHED | Do not claim. |
| Real-system improvement | Real vLLM improvement. | Local mismatch/tradeoff probe only. | vLLM ecosystem papers. | NOT_YET_ESTABLISHED | Do not claim. |

## Required Claim Policy For FGCS

The revised manuscript must not imply that learned override selection has established generalization. It may report the failed fresh confirmation as methodological caution and motivation for measuring opportunity/headroom before training predictors.

The revised manuscript may claim production-derived replay, resource-constrained simulation under modeled serving semantics, canonical SBS-vs-P6 action opportunity, and preregistered one-step default-relative causal headroom after Phase D executes.

It must not claim production deployment effects, measured production latency improvement, or real-vLLM improvement.
