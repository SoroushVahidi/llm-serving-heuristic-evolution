# LLM 2026 Paper Consolidation Plan

Date: 2026-08-24

Scope: publication-oriented consolidation of completed local evidence only. This
document does not propose or launch new experiments, does not use TEST/FINAL, and
does not reinterpret frozen thresholds after results.

## Thesis

Policy complementarity in LLM-serving schedulers is real under controlled stress
workloads, but the completed evidence shows that contextual selection, routing,
support expansion, guarded ESTF/WFS composition, and portfolio-guided structural
crossover do not robustly convert that complementarity into transferable
six-policy envelope gain.

## Result Hierarchy

### Established Positive Results

| Result | Evidence | Split/status | Safe use |
| --- | --- | --- | --- |
| Six-policy complementarity exists on controlled stress workloads | Unified 176x6 utility matrix; A/B dense unique winners 67/104; best-fixed WFS mean 0.7382 vs oracle 0.7684 | Stress benchmark; A/B dense paper-safe with caveat that C nonnative coverage was partial in this artifact | Main result supporting portfolio motivation |
| Public traces are near-degenerate for scheduler differentiation | 60 windows; all policies ANWG 1.0; six-policy envelope gain 0; active p99 5/512; max KV util 0.003802 | Public trace replay; descriptive, not a new policy claim | Main or appendix motivation for stress benchmark |
| Online regime signals are detectable | Stage telemetry 127,319 rows; A precision 1.0/recall 0.558; B precision 1.0/recall 0.434; C precision 1.0/recall 0.681; C AUROC 0.993 | Train/diagnostic telemetry; not end-to-end policy success | Main result separating detectability from utility gain |
| Family-A ESTF/WFS oracle headroom exists | Pair-envelope advantage over WFS 3.2585 and over ESTF 24.5511 in TRAIN/D1 evidence; 86.37% material decisions | TRAIN/D1 oracle-labeled diagnostic; no held-out policy claim | Supports why selector hypothesis was scientifically meaningful |
| Family-A pi0 was learnable offline | DEV mean regret 1.115 vs WFS/majority 4.087 and ESTF 12.087; balanced sign accuracy 0.855 | Frozen DEV offline; no retraining after result | Evidence that simple offline separability existed |
| Wulver target-free sweep expanded TRAIN-side support | 200 cells, 400 scenarios, 24,314 unique fingerprints, 156 novel TRAIN-side domains; V1/V2/task-subspace outside-support about 65-67% | TRAIN-side target-free support audit | Strong negative-control support expansion result |
| Typed GP infrastructure exactly represents the six parents | Parent reproduction gates passed before screen; canonical genomes, structural hashes, fingerprints, equal-budget harness | Implementation/test evidence; not a scheduler result | Methods appendix or reproducibility contribution |
| Random grammar search found TRAIN envelope holes | Best random candidate mean MG 0.011295, 6 unique wins, 39/60 positive-MG candidates | TRAIN only, selected best of 60; not freeze-ready | Appendix/preliminary synthesis evidence with selection-bias warning |

### Falsified or Demoted Results

| Hypothesis | Evidence | Verdict |
| --- | --- | --- |
| Pooled universal contextual selector can exploit the six-policy portfolio | Pooled selector mean regret 0.0463 vs best-fixed 0.0233 and majority 0.0127; LOFO held-out A 6.2x worse than best fixed | MULTIFAMILY_SELECTOR_NO_GO |
| Shared-feature selector rescues cross-family selection | Shared 2-feature A/B selector regret 0.0611 vs best-fixed 0.0122 | NO_GO |
| Hierarchical regime router converts detectable regimes into practical end-to-end gain | Live delta vs WFS 0.00616, CI lower 0.00055, below 0.01 practical bar; gap closure 0.1427 | HIERARCHICAL_ROUTER_NO_GO |
| Family-A DAgger/support expansion rescues statewise selector | D1 offline no-go; bridge pilot 0/104 DEV rows closer; Wulver DEV support all 8 gates failed | SELECTOR_HYPOTHESIS_FALSIFIED |
| Large target-free Wulver support expansion reaches DEV support | 24,314 unique states and 156 novel TRAIN-side domains, but DEV mean NN improvement 0.3147%, p90 0%, rows closer 1/104 | WULVER_DEV_SUPPORT_NO_GO |
| Guarded ESTF/WFS composite gives viable direct rule | Best guarded rule improves WFS by only 3.11%, releases ESTF 0.6%, violates mechanism quadrant ordering | MECHANISM_COMPOSITE_STATIC_NO_GO |
| Portfolio structural crossover is better than random GP/mutation-only under equal budget | Random best MG 0.011295; mutation-only best 0.002551; crossover best 0.0 | SYNTHESIS_NO_GO |
| Best random grammar TRAIN child is ready to freeze | Mean MG 0.011295 and 6 wins, but worst group regression 0.125 and family concentration 0.8915 | RANDOM_GP_CANDIDATE_NOT_FREEZE_READY |

## Scientific Lineage

| Stage | Hypothesis | Method and split | Primary result | Verdict and lesson |
| --- | --- | --- | --- | --- |
| Six-policy complementarity / MF-PSD | Existing schedulers expose complementary mechanisms under stress | Mechanism-focused policy stress design and unified 176x6 utility matrix | A/B dense unique-winner scenarios 67/104; oracle gain over best fixed 0.0303 | Complementarity exists and justifies portfolio study |
| Unified utility matrix | A common reward matrix can support portfolio analysis | Six anchors: full_prefill, chunked_prefill_small, ESTF, WFS, LLF, KV | Family-specific winners; degeneracy in some family-policy combinations | Portfolio is analyzable, but mechanisms are not uniformly active |
| Flat multi-family selector | A learned selector can exploit cross-family complementarity | Pooled and LOFO selector over 176 scenarios | Pooled worse than best fixed; LOFO held-out A severe failure | Cross-family contextual selection is brittle |
| Shared-feature selector | Removing family-specific artifacts improves generalization | Shared 2-feature A/B selector | Regret 0.0611 vs best-fixed 0.0122 | Shared schema did not rescue selection |
| Mechanism-target experiments | More mechanism-specific targets can expose usable structure | Family A/B/C stress and mechanism diagnostics | Real mechanisms observed, but no robust selector rescue | Moved focus from pooled selector to routing/mechanisms |
| Hierarchical routing | Online regime detection plus specialized policies can beat a fixed policy | Stage-1 telemetry classifier plus Stage-2 router; TEST/live eval | Stage-1 strong, live gain 0.00616 below practical bar | Detectability is not sufficient for utility gain |
| Online regime-signal analysis | Regime pressure is causally observable online | 127,319 telemetry rows | High precision signals, distinct A/B/C activation | Supports diagnostics, not enough for routing success |
| Family-A selector/support-expansion | ESTF/WFS statewise selector can be made robust by support expansion | V2 oracle rows, pi0, D1, support diagnostics | Offline pi0 strong, closed-loop mixed, D1 no-go | Statewise selection weakened |
| DAgger / support expansion | Adding TRAIN-only selector states fixes support gaps | D1 oracle rows; no DEV/FINAL labels for design | Offline gate failed; no automatic oracle path | Data addition alone not enough |
| Wulver target-free search | Broad target-free sweep expands reachable disagreement support | 64-task Wulver sweep; 200 cells; 400 scenarios | 24,314 unique fingerprints; 156 novel TRAIN-side domains | Moderate TRAIN-side expansion achieved |
| Frozen DEV support evaluation | Target-free expansion moves support toward frozen DEV states | Fixed support gate against 104 DEV rows | All 8 gates failed; 1/104 rows closer | Support-rescue hypothesis falsified |
| Selector closure | The full evidence chain closes Family-A statewise selector | Closure audit across pi0, D1, Wulver, DEV support | Ranked failure modes; selected portfolio synthesis pivot | SELECTOR_HYPOTHESIS_FALSIFIED |
| Guarded composite static feasibility | WFS-safe ESTF release guards can beat WFS without old index collapse | 48 TRAIN/D1 static rules | Best rule 3.11% WFS regret reduction; quadrant semantics failed | Guarded ESTF/WFS synthesis no-go |
| Portfolio synthesis design | New contribution should be a genuinely new portfolio policy | Design-only audit of six-policy holes and redundancy | Chose debt scheduler direction before later typed-GP branch | Useful design record, not a result |
| Typed-GP implementation | Exact parent mechanisms can be encoded as genetic material | Parent genomes, DSL, fingerprint, equal-budget harness | Six parent reproduction gates passed | Infrastructure contribution, not performance success |
| Equal-budget GP screen | Structural crossover discovers envelope-expanding policies better than random/mutation-only | TRAIN-only 60 candidates/treatment, 24 scenarios | Random best 0.011295; mutation best 0.002551; crossover best 0.0 | Portfolio crossover hypothesis no-go |
| Random candidate audit | Best random child is valid enough to freeze | TRAIN-only audit of proposal 40 | Nonredundant and compact, but group regression/concentration fail | Not freeze-ready; no held-out validation |

## Core Claims and Limitations

1. Controlled stress workloads reveal scheduler complementarity that public traces
   can hide.
   - Evidence: public trace saturation and MF-PSD unique winners.
   - Limitation: stress workloads are constructed; claims should not imply natural
     production prevalence.

2. Complementarity does not imply transferable contextual selection.
   - Evidence: flat selector, shared-feature selector, hierarchical router, and
     Family-A support-rescue failures.
   - Limitation: does not prove all possible selectors fail; it falsifies the
     tested formulations under frozen contracts.

3. Target-free support expansion can be scientifically successful on TRAIN-side
   coverage while failing to move held-out support.
   - Evidence: Wulver sweep generated 24,314 unique states and 156 novel
     TRAIN-side domains, but DEV support movement was negligible.
   - Limitation: DEV was used once as a fixed support target, not for performance
     validation.

4. Structural recombination of exact parent mechanisms did not yield new envelope
   capability in the first fair typed-GP screen.
   - Evidence: equal-budget GP results; crossover best mean MG 0.0.
   - Limitation: first small screen only; not a proof against all GP or QD methods.

5. The strongest manuscript is a diagnostic empirical study, not a new-scheduler
   performance paper.
   - Evidence: repeated frozen no-go results and no freeze-ready candidate.
   - Limitation: paper should avoid claiming a final algorithmic improvement.

## Most Important Quantitative Evidence

| Number | Source artifact | Split/status | Paper-safe interpretation |
| --- | --- | --- | --- |
| Public trace envelope gain 0.0; all policies ANWG 1.0 on 60 windows | docs/current/public_trace_replay_v1_analysis_20260820.md | Public descriptive | Public traces used here are saturated for scheduling comparison |
| Active p99 5/512; max KV util 0.003802 | public trace replay | Public descriptive | Saturation is explained by low load/KV pressure |
| A/B dense unique winners 67/104; WFS and ESTF each win 56 within epsilon | docs/audits/unified_policy_utility_matrix_v1_20260817.md | Controlled stress | Stress benchmark exposes nontrivial complementarity |
| Best fixed WFS 0.7382 vs oracle 0.7684; oracle gain 0.0303 | unified matrix | A/B dense controlled stress | A measurable portfolio envelope exists |
| Pooled selector regret 0.0463 vs best-fixed 0.0233 and majority 0.0127 | docs/audits/multifamily_contextual_selector_v1_20260817.md | Selector eval | Pooled contextual selection failed |
| LOFO held-out A regret 0.4786 vs best-fixed 0.0767 | multifamily selector audit | LOFO diagnostic | Cross-family generalization can collapse |
| Stage-1 router macro-F1 0.9887 but live delta 0.00616 and gap closure 0.1427 | hierarchical router audits | TEST/live no-go | Good regime detection did not translate into practical router gain |
| Family-A pi0 DEV mean regret 1.115 vs WFS 4.087 and ESTF 12.087 | family_a_scientific_evidence_audit_20260823.md | Frozen DEV offline | Offline statewise selector was plausible before closed-loop failure |
| Wulver sweep 24,314 unique states, 156 novel domains | family_a_wulver_medium_sweep_v1_analysis_20260824.md | TRAIN-side target-free | Large support expansion occurred |
| Wulver DEV mean NN improvement 0.3147%, p90 0%, 1/104 closer | family_a_wulver_dev_support_eval_v1_analysis_20260824.md | Frozen DEV support eval | Support expansion did not reach held-out states |
| Best guarded rule WFS regret reduction 3.11%; quadrant ordering failed | family_a_mechanism_composite_rule_static_feasibility_v1_analysis_20260824.md | TRAIN/D1 static | Direct guarded ESTF/WFS rule failed |
| GP screen A/B/C best mean MG: 0.011295 / 0.002551 / 0.0 | portfolio_guided_typed_gp_screen_v1_train_analysis_20260824.md | TRAIN-only screen | Random grammar found TRAIN holes; crossover did not |
| Random child worst group regression 0.125 and family concentration 0.8915 | random_grammar_best_candidate_audit_v1_20260824.md | TRAIN-only audit | Candidate should not be frozen for held-out validation |

## Contribution Map

### Contribution 1: A controlled evidence pipeline for LLM-serving scheduler complementarity

- Claim: public traces can be too saturated to reveal scheduling mechanism
  differences, while controlled stress workloads expose a measurable portfolio
  envelope.
- Supporting experiments: public trace replay, MF-PSD, unified utility matrix.
- Reviewer objection: stress workloads may be artificial.
- Existing answer: public-trace saturation is explicitly measured; stress workloads
  are used to reveal mechanisms, not to claim production prevalence.
- Do not claim: the stress distribution is representative of all deployments.

### Contribution 2: A systematic falsification of contextual scheduler selection

- Claim: multiple plausible selection/routing formulations fail under frozen
  contracts despite observable complementarity.
- Supporting experiments: flat selector, shared-feature selector, hierarchical
  router, Family-A pi0/D1/Wulver/DEV-support chain.
- Reviewer objection: a better model could work.
- Existing answer: the paper tests progressively stronger hypotheses, including
  support expansion and online regime routing; it does not claim universal
  impossibility.
- Do not claim: no contextual selector can ever work.

### Contribution 3: Evidence that support expansion and mechanistic composition are not enough

- Claim: large target-free TRAIN-side disagreement-state expansion and hand-designed
  guarded composition both failed their predeclared gates.
- Supporting experiments: Wulver sweep, Wulver DEV support evaluation, guarded
  composite static feasibility.
- Reviewer objection: DEV support is not policy performance.
- Existing answer: DEV support was a preregistered gate for whether oracle
  acquisition was scientifically justified.
- Do not claim: the Wulver states were useless for every future purpose.

### Contribution 4: Exact parent-encoding infrastructure plus a negative structural synthesis result

- Claim: the six-policy portfolio was encoded exactly enough for fair typed
  symbolic synthesis, and structural crossover still failed in the first
  equal-budget TRAIN screen.
- Supporting experiments: parent reproduction implementation, smoke/readiness,
  equal-budget GP screen, random candidate audit.
- Reviewer objection: screen is small and TRAIN-only.
- Existing answer: present as preliminary diagnostic evidence, not final
  generalization.
- Do not claim: GP scheduling is broadly ineffective.

## Novelty Positioning

- Against dynamic algorithm selection and algorithm portfolios: this work asks
  whether observable policy complementarity can be exploited in closed-loop
  LLM-serving scheduling. The key finding is the gap between oracle headroom,
  detectable regimes, and realized router/selector utility.
- Against GP-based scheduling hyper-heuristics, grammar-guided GP, QDGP, and
  MAP-Elites: this work does not claim a new winning GP algorithm. Its distinctive
  contribution is controlled, exact-parent, equal-budget falsification of structural
  recombination as a mechanism for exploiting an existing scheduler portfolio.
- Against Autopoiesis-like or LLM-guided synthesis systems: the paper should avoid
  claims about broad automated discovery. It can instead show that exact structural
  access to known mechanisms was insufficient in this setting.
- Against LLM-serving schedulers such as vLLM, Sarathi, Llumnix, PARS, and
  Apt-Serve: this paper is not a production replacement scheduler paper. It studies
  when scheduler mechanisms produce complementarity and why selecting or
  recombining them is difficult.

## Recommended Paper Type

Recommended type: diagnostic empirical study with a negative-results spine.

This can be framed as a full paper if the venue values systematic empirical
methodology and falsification. It should not be framed as a constructive scheduler
paper because no held-out validated new scheduler exists.

## Paper Structure

1. Introduction
   - Claims: complementarity is tempting but not automatically exploitable.
   - Figures: hypothesis/falsification pipeline.
   - Sources: this plan; current audits.

2. Problem Formulation
   - Claims: six-policy envelope, marginal gain, selector/router/synthesis tasks.
   - Tables: notation and evaluation splits.
   - Sources: portfolio design, GP design, utility matrix.

3. Policy Portfolio and Workload Construction
   - Claims: six parent mechanisms and stress families.
   - Tables: policy x mechanism matrix.
   - Sources: portfolio_policy_synthesis_design_v1, unified utility matrix.

4. Evidence of Policy Complementarity
   - Claims: public traces saturate; stress workloads reveal unique winners.
   - Figures: public trace saturation, utility heatmap, oracle vs best fixed.
   - Sources: public_trace_replay, unified matrix.

5. Why Contextual Selection Fails
   - Claims: pooled, shared-feature, and Family-A statewise selection fail to
     generalize or close the loop.
   - Figures: selector regret and support-gap progression.
   - Sources: multifamily selector, Family-A closure.

6. Why Routing and Support Expansion Fail
   - Claims: online signals are detectable; routing gain too small; Wulver expands
     TRAIN support without moving DEV support.
   - Figures: regime-signal metrics, Wulver support before/after.
   - Sources: online regime signal, router audits, Wulver analyses.

7. Why Static and Structural Composition Fail
   - Claims: guarded ESTF/WFS composition and exact typed crossover do not produce
     robust envelope expansion.
   - Tables: guarded rule no-go, GP A/B/C comparison.
   - Sources: composite static feasibility, GP screen, random candidate audit.

8. Synthesis Implications and Design Lessons
   - Claims: mechanism complementarity requires more than statewise selection or
     shallow recombination; frozen gates prevent post-hoc success claims.
   - Figures: failure-mode map.
   - Sources: closure and pivot audit.

9. Related Work
   - Claims: position versus algorithm portfolios, GP hyper-heuristics, LLM-serving
     schedulers, and automated synthesis.

10. Limitations
   - Claims: controlled stress distribution, small GP screen, no final new policy,
     simulator dependence, support-evaluation scope.

11. Conclusion
   - Claims: negative result is scientifically useful; complementarity is necessary
     but insufficient.

## Figures and Tables

| Artifact | Source data | Question answered | Split/status |
| --- | --- | --- | --- |
| Figure: hypothesis pipeline with verdicts | All current/audit verdict docs | How did the project move from complementarity to selector and synthesis no-go? | Mixed; label each node |
| Figure: public trace saturation | public_trace_replay_v1 outputs | Why are stress workloads needed? | Public descriptive |
| Figure: six-policy utility heatmap | unified utility matrix | Where do policies win and overlap? | Controlled stress |
| Figure: best fixed vs oracle headroom by family | unified matrix and portfolio redundancy | How much exploitable complementarity exists? | Controlled stress |
| Figure: router detection vs utility gap | online signal and router audits | Why does accurate regime detection not imply router success? | Diagnostic/test/live |
| Figure: Wulver TRAIN support expansion vs DEV support movement | Wulver sweep and DEV support eval | Why did support expansion not rescue selection? | TRAIN-side plus frozen DEV support |
| Figure: GP treatment comparison | GP train screen | Did structural crossover beat random/mutation? | TRAIN-only |
| Table: six-policy mechanism matrix | portfolio design | What mechanisms are in the portfolio? | Descriptive |
| Table: hypothesis/verdict ledger | current docs and audits | What was tested and closed? | Mixed |
| Table: claim-safety audit | this document | What can be safely claimed? | Publication control |

## Claim Safety Audit

| Claim | Supported? | Evidence | Split | Safe wording | Unsafe wording |
| --- | --- | --- | --- | --- | --- |
| Policy complementarity exists | Yes | Unified utility matrix, unique winners, envelope gain | Controlled stress | "The tested stress workloads show measurable complementarity." | "LLM serving policies are broadly complementary in production." |
| Public traces are saturated | Yes for tested traces | Public trace replay | Public descriptive | "Our public-trace windows are near-degenerate under this simulator." | "Public traces cannot evaluate schedulers." |
| Contextual selection fails | Yes for tested formulations | Flat/shared/Family-A selector audits | Mixed | "The tested contextual selectors failed frozen gates." | "All contextual selection is impossible." |
| Routing fails | Yes for tested router | Hierarchical router live no-go | TEST/live | "High-quality regime detection produced too little end-to-end gain." | "Online regime routing never helps." |
| Support expansion fails | Yes for Family-A support rescue | Wulver sweep and DEV support eval | TRAIN plus frozen DEV support | "Large target-free support expansion did not move the frozen DEV support gate." | "Support expansion is useless." |
| Guarded composite fails | Yes for ESTF/WFS guarded rules | Static feasibility no-go | TRAIN/D1 | "The tested guarded ESTF/WFS rules failed." | "Mechanistic rule design cannot work." |
| Structural crossover fails | Yes for first typed GP screen | Equal-budget GP screen | TRAIN only | "In the first equal-budget TRAIN screen, crossover found no envelope gain." | "GP crossover is generally ineffective." |
| Random grammar finds envelope holes | Yes but exploratory | GP screen and random audit | TRAIN only | "Random grammar search found TRAIN-only envelope holes, but the best child was not freeze-ready." | "Random GP discovered a new validated scheduler." |
| Generalization is poor | Partly | Selector/LOFO/Wulver DEV support | Mixed | "Several tested generalization paths failed." | "The scheduler problem is inherently non-generalizable." |
| Existing public traces are saturated | Yes for current replay | public_trace_replay | Public descriptive | "The replayed public windows did not differentiate these policies." | "Real deployments are saturated." |

## What To Omit or Move To Appendix

- Main paper: evidence chain, key verdicts, public trace saturation, utility matrix,
  selector/router/support/synthesis summaries.
- Appendix: full Wulver engineering integrity details, exact shard accounting,
  full Family-A support metrics, random-candidate branch activation and ablation
  details, full parent reproduction tables, complete GP candidate ledger.
- Omit unless needed for reproducibility: stale Apt-Serve lineage, exhaustive
  command logs, transient environment issues, every failed micro-variant within
  static guard search.

## Remaining Work Before Writing

Required before manuscript drafting:

1. Reconcile canonical docs so `RESUME_HERE`, `WORK_STATUS`, and `NEXT_ACTIONS`
   no longer point toward closed selector/synthesis work.
2. Build paper-ready tables from existing artifacts only, with split labels on
   every number.
3. Generate figures from existing result artifacts only.
4. Freeze the claim-safety language in the paper outline.

Nice to have before submission, not required to start writing:

1. A reproducibility appendix for the typed parent-genome reproduction harness.
2. A compact figure of Wulver support movement showing "large TRAIN expansion,
   negligible DEV movement."
3. A related-work comparison table focused on what the paper does not claim.

Post-LLM-2026:

1. A genuinely new native scheduler family, if pursued, should be separate from
   this paper's core no-go story.
2. Larger evolutionary/QD searches should wait until there is a new hypothesis
   not already falsified by the current typed-crossover screen.
3. Real-serving validation should be reserved for a frozen policy candidate with
   held-out evidence.

## Explicit Exclusions

- Do not present the random grammar candidate as a validated scheduler.
- Do not reopen ESTF/WFS contextual selection or DAgger.
- Do not include DEV-driven redesign recommendations.
- Do not claim TEST/FINAL evidence where the artifact is TRAIN-only or diagnostic.
- Do not use the Wulver support result as evidence of policy performance.

## Exact Next Task

Start manuscript production by creating a paper skeleton and figure/table manifest
from this consolidation plan, then reconcile canonical status documents so the local
roadmap reflects the publication-focused stage.
