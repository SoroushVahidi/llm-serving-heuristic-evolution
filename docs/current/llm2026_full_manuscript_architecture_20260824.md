# LLM 2026 Full Manuscript Architecture

Date: 2026-08-24

Scope: publication-architecture audit only. No experiments, selector/router
training, GP search, vLLM/GPU work, Wulver/Vulver jobs, TEST/FINAL use, DEV
redesign, threshold changes, git push, or destructive git operation were
performed for this report.

## 1. Repository State Audited

- Branch: `contextual-compositional-heuristics-20260731`
- HEAD: `2987b7181efa2bc550d8a894c537eca8f6393eb6`
- Upstream: `origin/contextual-compositional-heuristics-20260731`
- Ahead/behind from local refs: `0 2` from `git rev-list --left-right --count @{u}...HEAD`; `git status` reports `[ahead 2]` because the local branch contains two commits not on the configured upstream.
- Worktrees: single worktree at `/home/soroush/llm-serving-heuristic-evolution`.
- Git locks / merge / rebase / cherry-pick / bisect markers: none found.
- Dirty tracked files before this report: `docs/current/NEXT_ACTIONS.md`, `docs/current/RESUME_HERE.md`, `docs/current/WORK_STATUS.md`.
- Major untracked publication/science artifacts include current Family-A audits, Wulver support artifacts, portfolio synthesis/typed-GP artifacts, real-vLLM artifacts, joint generalization artifacts, and `paper/llm2026/`.
- Active tmux: none. `screen` is not installed.
- Active plausible repo processes: editor/Claude daemon processes in the repo, existing Wulver SSH master sockets, and an unrelated local `njit_auditor` uvicorn process. No running scientific experiment, GPU/vLLM job, selector/router training, GP run, or API job was observed.

## 2. Venue Constraints

Target interpreted as **The 2026 International Conference on Large Language
Models (LLM 2026)**, American CSE, December 16-18, 2026, Las Vegas.

Sources checked:

- Official page: `https://www.american-cse.org/LLM2026/paper_categories`
- Official publisher page: `https://www.american-cse.org/LLM2026/publisher`
- Official presentation modes page: `https://www.american-cse.org/LLM2026/presentation_modes`
- Rendered content is in a React bundle:
  `https://www.american-cse.org/LLM2026/scripts/app.6a62fd44614f8fdcf269.js`

Verified from official site bundle:

| Constraint | Verified value | Manuscript consequence |
|---|---|---|
| Paper category | `FULL/REGULAR RESEARCH PAPERS`; site says full/regular and late-breaking are used interchangeably | Target the full/regular research category. |
| Page limit | Full/regular: 12-15 pages in one-column Springer Nature format OR 6-8 pages in two-column format | Use an 8-page two-column architecture unless final Springer one-column prep becomes mandatory. |
| Page-count inclusions | The site states the page counts include all figures, tables, and references | Be conservative: target 8 pages including references. |
| Review-submission format | Any reasonable formatting/typesetting is acceptable for draft papers submitted for evaluation | Current generic scaffold is acceptable for drafting, but not enough for final formatting. |
| Final publication format | Accepted papers receive final typesetting instructions; final papers expected to use Springer Nature standard format, US letter size | Later migrate to the official Springer Nature author template. |
| Publisher | Accepted papers of Research Tracks published by Springer Nature in LLM 2026 proceedings | Use Springer-compatible bibliography/style later. |
| Submission file | MS doc or PDF, one file per paper | Produce one PDF for submission. |
| First-page metadata | Title; each author name, affiliation, city, country, email; contact author | Review appears not anonymized from available text; do not remove authors unless a later call contradicts this. |
| Abstract | About 100 words | Current abstract draft is too long; final abstract should compress to about 100-130 words. |
| Keywords | Maximum 5 topical keywords | Reserve 4-5 keywords. |
| Appendix / supplement | No separate appendix/supplement allowance found; page limit explicitly includes references | Treat appendices as outside the main submission unless the portal later allows supplementary material. |

Open verification items:

- Exact downloadable class/template for final Springer Nature proceedings.
- Whether optional supplementary material is accepted in the submission portal.
- Any figure-font/minimum-resolution rules beyond Springer defaults.

## 3. Reconstructed Paper Thesis

Recommended thesis:

> LLM-serving scheduler portfolios exhibit real workload-dependent complementarity, including under jointly varying multi-mechanism workloads. However, the experiments reveal an exploitability gap: oracle headroom, detectable regimes, and structural access to parent mechanisms do not by themselves yield robust deployable adaptive scheduling gain under frozen gates. Real-vLLM validation further shows that simulator mechanisms must be checked against native serving-engine semantics, because direct mechanism transfer can fail even when the real engine exposes its own reproducible scheduling-control tradeoff.

This is stronger and safer than a chronology of failed experiments. It makes
three claims and keeps each scoped:

| Thesis component | Strongest support | Weakest support / risk | Safe wording | Unsafe wording |
|---|---|---|---|---|
| Complementarity exists | Unified 176x6 matrix; joint 240-scenario result | Workloads are synthetic/mechanism-focused | Complementarity persists in controlled and jointly varying synthetic workloads. | Complementarity holds in arbitrary production traffic. |
| Oracle headroom is positive | Joint best fixed 0.314072, oracle 0.333106, headroom 0.019034, CI [0.015988, 0.022433] | Headroom is modest in absolute ANWG | Oracle headroom is measurable and statistically stable in the joint population. | Adaptive scheduling can necessarily capture this headroom. |
| Exploitation is hard | Pooled/shared selectors, hierarchical router, Wulver support, guarded composition, typed GP crossover all no-go | Methods are not exhaustive | Tested exploitation mechanisms fail frozen transfer/practicality gates. | All adaptive schedulers are impossible. |
| Real-system semantics matter | vLLM direct analogue no-go, semantic mismatch diagnosis, native token-budget strong result | One model, one GPU, one vLLM version | Simulator-to-engine mapping must be validated; native vLLM exposes a different control dimension. | Simulator results are invalid generally. |

## 4. Contribution Hierarchy

| Candidate contribution | Classification | Rationale |
|---|---|---|
| Six-policy complementarity | Core contribution, main text | Establishes the phenomenon that motivates the paper. Use unified 176x6 plus controlled A/B/C results. |
| Controlled A/B/C mechanism isolation | Supporting contribution, main text compact + appendix details | Needed to explain mechanisms and why the portfolio was constructed. Do not let A/B/C dominate the narrative. |
| Joint generalization result | Core contribution, main text | Directly answers the breadth objection; should be the strongest positive empirical result. |
| Oracle-headroom characterization | Core contribution, main text | Defines the gap between best fixed and oracle envelope. |
| Exploitability gap | Core contribution, main text | The central intellectual contribution: measurable complementarity is not the same as deployable gain. |
| Contextual-selector failure | Main-text evidence, appendix details | Use as the first exploitation no-go; include only the key metrics/gates in main text. |
| Hierarchical-routing failure | Main-text evidence | Strong because router quality was high but utility gain failed; good reviewer-facing evidence. |
| Wulver support expansion | Supporting contribution, compact main-text row + appendix | Important constructive falsification; too detailed for main text. |
| Guarded composition | Supporting contribution, compact main-text row + appendix | Shows direct ESTF/WFS guarded synthesis was tested, but details are secondary. |
| Exact-parent typed GP | Supporting contribution, appendix-heavy | Infrastructure validates fair structural-synthesis comparison; include only crossover no-go in main text. |
| Constructive falsification sequence | Core/supporting bridge, main text | Bundle support expansion, guarded composition, typed GP into one compact section/table. |
| Real-vLLM semantic mismatch | Core contribution, main text | Distinct systems-methodology value. |
| Native vLLM budget effect | Core contribution, main text | Positive real-system result; prevents the validation section from being purely negative. |
| Public-trace saturation | Supporting motivation, prose only | Explains why stress workloads are necessary; do not overclaim production representativeness. |
| Random-grammar best candidate audit | Appendix-only | Useful for transparency; not a main claim because candidate was not freeze-ready. |
| Decision-criticality/timescale work | Omit or future work unless already analyzed in an appendix note | Not part of frozen main evidence for this manuscript architecture. |

## 5. Recommended Paper Structure

The current ten-section scaffold is too fragmented for an 8-page full/regular
two-column submission if references count. Use a seven-section paper.

| Sec. | Title | Purpose | Experiments/results included | Figures/tables | Target pages | Min | Max | Cut first |
|---:|---|---|---|---|---:|---:|---:|---|
| 1 | Introduction | State the oracle-vs-deployable paradox and contributions | No detailed results beyond 2-3 headline numbers | none or tiny inline contribution list | 0.75 | 0.55 | 0.95 | Long motivation paragraphs |
| 2 | Portfolio, Workloads, and Metrics | Define six policies, ANWG, envelope, workload suites, frozen-gate discipline | Public-trace saturation; six policies; A/B/C + joint suite descriptions | Table 1, optional mini workload matrix | 1.05 | 0.80 | 1.30 | Detailed simulator internals |
| 3 | Complementarity Beyond Handcrafted Families | Positive result section | 176x6 matrix; controlled family headroom; joint 240-scenario result | Figure 1 joint complementarity; short table/prose for old A/B/C | 1.45 | 1.15 | 1.75 | Old A/B/C per-family detail |
| 4 | The Exploitability Gap | Show selection/routing failures after complementarity is established | Pooled selector, shared feature, mechanism target, hierarchical router | Table 2 no-go ledger; optional small router inset | 1.25 | 1.00 | 1.55 | Mechanism-target details |
| 5 | Constructive Falsification Beyond Selection | Show stronger attempts also fail | Family-A/Wulver support, guarded composition, typed GP crossover, random candidate caveat | Table 2 rows or Figure 2 compact pipeline | 0.90 | 0.65 | 1.15 | Random candidate audit |
| 6 | Real-Serving Semantic Validation | Ground simulator mechanism claims against native vLLM | Direct analogue no-go; semantic mismatch; native token-budget strong result | Figure 3 real-vLLM semantic/trace panel; appendix semantic table | 1.05 | 0.80 | 1.30 | Full per-regime latency table |
| 7 | Implications, Limitations, Related Work, Conclusion | Prevent overclaiming and position the result | Claim safety, limitations, related-work contrasts, conclusion | none | 0.90 | 0.65 | 1.15 | Detailed related-work prose |
| R | References | Venue requires page count includes references | Verified citations only | bibliography | 0.65 | 0.50 | 1.00 | Move marginal citations to appendix/supplement if allowed |

Target total: 8.00 two-column pages including references.

Sections to merge relative to the scaffold:

- Merge `Problem Setting`, `Scheduler Portfolio`, and `Workloads` into one
  methodology section.
- Merge `Implications`, `Related Work`, `Limitations`, and `Conclusion` into a
  compact final section if the page limit remains strict.
- Keep real-vLLM as its own section because it is a distinct systems-methodology
  contribution.

## 6. Exact Page Budget

Assumption: full/regular paper, two-column, maximum 8 pages including figures,
tables, and references.

### Target Allocation

| Component | Pages |
|---|---:|
| Title/authors/abstract/keywords | 0.35 |
| Introduction | 0.75 |
| Portfolio, workloads, metrics | 1.05 |
| Complementarity and oracle headroom | 1.45 |
| Exploitability gap | 1.25 |
| Constructive falsification | 0.90 |
| Real-serving semantic validation | 1.05 |
| Implications/limitations/related work/conclusion | 0.55 |
| References | 0.65 |
| **Total** | **8.00** |

Figure/table footprint inside the above:

- Main figures: about 1.75 pages total.
- Main tables: about 1.10 pages total.
- Text budget after visuals: about 4.50 pages plus references.

### Compression Allocation

If the draft is 1-2 pages over:

| Compression action | Expected saving |
|---|---:|
| Drop standalone evidence-pipeline figure; use Table 2 only | 0.40-0.60 pages |
| Move simulator-vLLM semantic table to appendix; keep only three-stage figure | 0.25-0.40 pages |
| Collapse `Constructive Falsification` to one paragraph plus ledger rows | 0.30-0.45 pages |
| Compress related work to one dense paragraph and cite only essential systems/portfolio/GP categories | 0.25-0.40 pages |
| Move random-grammar audit entirely to appendix | 0.15-0.25 pages |
| Replace old A/B/C table with prose plus one sentence of headroom numbers | 0.20-0.30 pages |

## 7. Main Text Manifest

| Result | Why main text | Quantitative content | Form | Space |
|---|---|---|---|---:|
| Public-trace saturation | Motivates synthetic/mechanism stress workloads | 60/60 windows tie at ANWG 1.0; envelope gain 0; active p99 5/512; max KV 0.003802 | Prose only | 0.08 page |
| Six-policy controlled complementarity | Establishes base portfolio | 176x6 = 1056 cells; 54.0% unique-winner rate; family headrooms A 0.021742, B 0.049433, C 0.020261 | Prose + small table row | 0.25 page |
| Joint multi-mechanism generalization | Main breadth result | 240 scenarios; 92.9% >=2 elevated pressures; unique-winner fraction 59.6%; best fixed 0.314072 vs oracle 0.333106; headroom 0.019034 CI [0.015988, 0.022433] | Figure + prose | 0.75 page |
| Multi-mechanism gain | Counters handcrafted-family objection | 93.1% of oracle gain from >=2 pressures; 63.3% from >=3; top-10% gain share 40.5% | Figure panel/prose | 0.25 page |
| Pooled/shared selector no-go | First exploitability failure | `MULTIFAMILY_SELECTOR_NO_GO`; pooled regret 0.0463 vs best-fixed 0.0233 and majority 0.0127; LOFO-A 0.4786 vs fixed 0.0767 | Ledger table + prose | 0.25 page |
| Hierarchical router no-go | Shows detectability not sufficient | macro-F1 0.9887; live delta 0.00616 < 0.01; oracle-gap closure 0.143 | Ledger table + prose/inset | 0.30 page |
| Wulver support expansion no-go | Strong constructive falsification | 24,314 unique Wulver fingerprints; 156 novel TRAIN-side domains; DEV mean NN improvement 0.315%; p90 0; 1/104 closer; all 8 gates failed | Ledger row + prose | 0.20 page |
| Guarded composition no-go | Rules out simple ESTF/WFS guard story | Best guarded rule WFS regret reduction 3.11%; quadrant ordering failed | Ledger row | 0.12 page |
| Typed GP screen no-go | Tests structural recombination claim | 4320 candidate-scenario evals; random best 0.011295; mutation-only best 0.002551; crossover best 0.0; `SYNTHESIS_NO_GO` | Ledger row + prose | 0.18 page |
| Real-vLLM direct + semantic diagnosis + native budget result | Main systems validation | Direct `PREFILL_REAL_VALIDATION_NO_GO`; `SIMULATOR_VLLM_SEMANTICS_MISMATCH`; native `NATIVE_VLLM_BUDGET_EFFECT_STRONG`; trace and latency numbers | Figure + prose | 0.80 page |

## 8. Appendix Manifest

| Result | Why appendix is sufficient | Role | Proposed appendix subsection |
|---|---|---|---|
| Full A/B/C generator specs and simulator assumptions | Too much detail for main text | Reproducibility | A. Workload Construction |
| Unified matrix build/audit lineage | Main text only needs the final 176x6 facts | Reproducibility/provenance | B. Unified Matrix Integrity |
| Full public-trace replay outputs | Saturation is motivation, not central evidence | Robustness | C. Public Trace Replay |
| Shared-feature and mechanism-target diagnostics | Supports selector no-go but too detailed | Claim support | D. Selector Failure Details |
| Online regime-signal telemetry | Supports router section | Robustness | E. Router and Regime Signals |
| Family-A pi0/D1/Wulver chain | Complex but important for constructive falsification | Claim support | F. Family-A Support Expansion |
| Wulver shard integrity/accounting | Engineering provenance not main science | Reproducibility | G. Wulver Integrity |
| Guarded composite search space and quadrant tables | Main text needs verdict only | Reproducibility | H. Guarded Composition |
| Typed GP grammar/parent reproduction/smoke | Main text needs fair-screen assurance only | Reproducibility | I. Typed-GP Infrastructure |
| Random-grammar candidate audit/activation/ablation | Candidate was not freeze-ready | Transparency | J. Random Candidate Audit |
| Full real-vLLM request manifests/server configs/traces | Main text needs compact semantic result | Reproducibility | K. Real-vLLM Details |
| Venue/provenance ledgers | Internal drafting controls | Omit from formal appendix unless supplementary allowed | Supplementary artifacts |

## 9. Figure Plan

Final main-text figure set should be three figures, not five.

### Figure 1: Complementarity Across Joint Workloads

- Question: Does complementarity persist beyond disjoint handcrafted families?
- Sources:
  - `experiments/joint_multimechanism_generalization_v1/figures/winner_distribution.png`
  - `experiments/joint_multimechanism_generalization_v1/figures/oracle_gain_histogram.png`
  - `experiments/joint_multimechanism_generalization_v1/figures/winner_map_prefill_kv.png`
  - `winner_summary.json`, `oracle_summary.json`, `mixed_mechanism_summary.json`
- Panels:
  1. winner count by policy;
  2. oracle-gain histogram with mean/headroom;
  3. compact winner map over two mechanism-pressure axes or a gain-by-pressure panel.
- Highlight: 59.6% epsilon unique-winner fraction; headroom 0.019034 CI
  [0.015988, 0.022433]; 93.1% gain from >=2 pressures.
- Space: 0.65 page.
- Replaces: separate A/B/C and joint tables.

### Figure 2: Exploitability Gap Under Frozen Gates

- Question: Why is oracle headroom not deployable gain here?
- Sources:
  - `llm2026_claim_evidence_ledger_20260824.md`
  - selector/router/support/composition/GP decision artifacts.
- Visual: compact pipeline or matrix with stages: selector -> shared features ->
  router -> support expansion -> guarded composition -> typed crossover. Show
  key pass/fail gate numbers, not every metric.
- Highlight: router macro-F1 0.9887 but delta 0.00616; Wulver 24,314 unique
  states but 1/104 DEV rows closer; GP crossover best MG 0.0.
- Space: 0.55 page.
- Replaces: standalone router figure and long chronological prose.

### Figure 3: Simulator-to-vLLM Semantic Validation

- Question: What happened when simulator mechanisms were checked against native
  vLLM?
- Sources:
  - `real_vllm_prefill_decode_validation_v1_20260824.md`
  - `real_vllm_prefill_decode_fidelity_diagnosis_v1_20260824.md`
  - `native_vllm_chunk_budget_semantics_probe_v1_20260824.md`
  - `native_vllm_chunk_budget_semantics_probe_v1/mechanism_summary.json`
- Panels:
  1. simulator Family-B abstraction versus native vLLM scheduling semantics;
  2. T512 vs T4096 step traces: scheduled steps, mixed steps, partial-prefill items, prompt tokens/prefill step;
  3. low-late latency tradeoff: T4096 late TTFT -30.6 ms, hog E2E +16.3 ms.
- Space: 0.65 page.
- Replaces: full simulator-vLLM semantic table in main text.

Figures to remove/combine:

- Standalone evidence-pipeline overview: fold into Figure 2.
- Standalone router detectability figure: fold into Figure 2 or prose.
- Separate Wulver support figure: appendix only unless Figure 2 has room.

## 10. Table Plan

### Main Table 1: Six-Policy Mechanism Matrix

- Placement: Section 2.
- Columns: policy, primary mechanism, key runtime observables, strongest stress
  regime, known weakness.
- Source: `experiments/portfolio_policy_synthesis_design_v1/policy_mechanism_matrix.json`
  and policy implementations.
- Space: 0.45 page.
- Claim supported: the portfolio spans distinct serving mechanisms.

### Main Table 2: Frozen Hypothesis/Gate/Outcome Ledger

- Placement: Sections 4-5.
- Columns: exploitation attempt, hypothesis, frozen gate/criterion, strongest
  result, verdict, safe interpretation.
- Rows: pooled selector, shared feature, mechanism target, hierarchical router,
  Wulver support expansion, guarded composition, typed structural crossover,
  random candidate audit.
- Space: 0.65 page.
- Claim supported: the exploitability gap is systematic, not a single failed
  model run.

### Appendix Table A: Workload Suite Matrix

- Columns: suite, size, split/status, varied mechanisms, metric, safe claim.
- Main text can describe this in prose plus Table 1; full table is appendix if
  tight.

### Appendix Table B: Simulator-vLLM Semantic Map

- Columns: primitive, simulator Family-B implementation, vLLM 0.27.1 analogue,
  fidelity, consequence.
- Main text Figure 3 should carry the takeaway.

Tables to omit:

- Full GP candidate ledger in main text.
- Full Wulver support-gate table in main text.
- Full per-regime vLLM latency table in main text.

## 11. Reviewer-Risk Audit

| Objection | Severity | Existing evidence | Where to address | More experiment required? |
|---|---|---|---|---|
| Workloads are synthetic. | High | Joint 240-scenario distribution broadens A/B/C; public trace saturation explains why stress workloads are needed. | Intro, Section 2, Limitations | No, if wording stays scoped. |
| Results may not generalize across real models/hardware. | High | Real-vLLM is framed as semantic validation on one concrete stack, not broad performance validation. | Real-vLLM section, Limitations | No for current claims. |
| 0.5B model and one GPU are too small. | Medium-high | Native vLLM trace shows actual scheduler separation and latency tradeoff; no high-KV claim is made. | Real-vLLM limitations | No unless claiming scale generality. |
| Simulator fidelity is questionable. | High | Direct no-go plus semantic diagnosis explicitly audits the mismatch. | Real-vLLM section | No; this is part of the contribution. |
| Oracle headroom is modest. | Medium | Joint headroom CI is stable; per-scenario distribution has p90 0.058040 and positive gain in 60.4%. | Complementarity section | No. |
| Negative adaptive results may be uninteresting. | Medium-high | The paper frames a sequence of stronger frozen tests and explains why regime detectability and support expansion were plausible. | Exploitability sections | No. |
| Tested methods are not exhaustive. | High | Claim is bounded to tested selectors/routers/composition/synthesis; no impossibility theorem. | Intro, Limitations | No. |
| No positive systems contribution. | Medium-high | Positive contributions are complementarity map, joint workload evidence, and native-vLLM budget tradeoff. | Contributions, Real-vLLM | No. |
| Paper could read like an experiment log. | High | Architecture is thesis-driven: complementarity -> exploitability gap -> semantic validation. | Whole structure | No. |
| Policy choices may not be representative. | Medium | Mechanism matrix covers prefill, chunking, service-time, fairness, laxity, KV. | Section 2, Related Work | No, but cite related schedulers carefully. |
| Real-vLLM evidence is too narrow. | Medium | Use as methodological validation/limitation, not as universal result. | Section 6, Limitations | No. |
| Claims an impossibility result without proof. | High | Claim-safety ledger explicitly prohibits this wording. | Abstract, Intro, Limitations | No. |

## 12. Related-Work Architecture

Do not write a full related-work section until citations are verified. Use this
comparison structure:

1. LLM inference/serving engines and continuous batching:
   vLLM, paged attention, serving-engine scheduling abstractions.
2. Prefill/decode interference and chunked prefill:
   systems that expose prompt/decode contention, chunking, and batching controls.
3. KV-cache management and admission/scheduling:
   KV capacity, block allocation, preemption, and cache-aware policies.
4. SLO-aware/fair LLM scheduling:
   urgency, laxity, priority, tenant fairness, weighted goodput.
5. Adaptive scheduling and algorithm portfolios:
   per-instance algorithm selection, portfolio oracles, selection/gating limits.
6. Hyper-heuristics, GP, grammar-guided program synthesis, MAP-Elites/QD:
   position typed structural synthesis as a tested exploitation path, not the
   main positive result.
7. Self-evolving/autopoietic serving systems:
   relevant only if cited as a contrast to the paper's frozen-gate empirical
   methodology.

Essential positioning:

- Unlike papers that propose a new scheduler, this paper measures when a
  scheduler portfolio has oracle headroom and then tests whether increasingly
  expressive exploitation mechanisms can convert it into practical gain.
- Unlike simulator-only work, it includes a real-vLLM semantic validation that
  shows direct abstraction transfer can fail.
- Unlike generic GP/hyper-heuristic work, the structural search is a falsified
  hypothesis under exact parent reproduction and equal-budget accounting.

## 13. Title and Abstract Architecture

Recommended title:

1. **The Exploitability Gap in LLM-Serving Scheduler Portfolios**

Runner-up titles:

2. **When Scheduler Complementarity Is Not Enough for Adaptive LLM Serving**
3. **From Oracle Complementarity to Adaptive Scheduling: A Falsification Study in LLM Serving**

One-sentence thesis:

> Scheduler complementarity is measurable in LLM-serving workloads, but the tested ways of exploiting it reveal a gap between oracle headroom and robust deployable adaptive gain.

Abstract content bullets, sized for the venue's about-100-word constraint:

1. Problem: LLM-serving workloads stress different scheduling mechanisms, making portfolios attractive.
2. Positive finding: six-policy complementarity persists in controlled and joint workloads; joint headroom is 0.019034 ANWG with CI [0.015988, 0.022433].
3. Exploitability finding: selectors, routing, support expansion, guarded composition, and structural crossover fail frozen gates.
4. Real-system implication: direct Family-B simulator mapping fails under native vLLM semantics, but native token budget shows a reproducible tradeoff.
5. Conclusion: measure complementarity, but validate exploitability and serving-engine semantics before claiming adaptive gains.

## 14. Recommended Manuscript Architecture

| Sec. | Section | Purpose | Main evidence | Figures/Tables | Target pages |
|---:|---|---|---|---|---:|
| 1 | Introduction | Frame exploitability gap and contributions | Thesis + headline numbers | none | 0.75 |
| 2 | Portfolio, Workloads, and Metrics | Define portfolio, ANWG, envelope, gates | Six policies, public trace saturation, workloads | Table 1 | 1.05 |
| 3 | Complementarity Beyond Handcrafted Families | Establish positive complementarity/headroom | 176x6, old A/B/C, joint 240 | Figure 1 | 1.45 |
| 4 | The Exploitability Gap | Show selector/router failure | Pooled/shared selector, mechanism target, hierarchical router | Table 2, optional inset | 1.25 |
| 5 | Constructive Falsification Beyond Selection | Show stronger exploitation attempts fail | Wulver, guarded composition, typed GP | Table 2 rows / compact visual | 0.90 |
| 6 | Real-Serving Semantic Validation | Validate simulator-engine semantics | vLLM direct no-go, mismatch diagnosis, native budget strong | Figure 3 | 1.05 |
| 7 | Implications, Limitations, Related Work, Conclusion | Scope claims and position against prior work | Claim-safety table, related categories | none | 0.90 |
| R | References | Required citations | Verified references only | bibliography | 0.65 |

Total target page count: 8.00 pages including references.

Estimated references pages: 0.65-1.00. If references exceed 0.8 pages, compress
Sections 5 and 7 first.

Estimated main-text figure/table footprint: 2.85 pages total. This is tight but
acceptable only if figures are compact and tables replace prose.

Appendix strategy: keep all integrity ledgers, full gates, configs, traces, GP
ledgers, and Wulver details outside the main eight-page narrative. If the venue
does not permit supplementary appendices, preserve these as repository artifacts
and cite paths only where appropriate.

No additional experiment is required before writing. The next task is manuscript
production, not science.

Exact first writing task:

> Convert `paper/llm2026/main.tex` from the current generic one-column scaffold
> into the verified LLM 2026 submission-format scaffold, then draft Sections 1
> and 2 against the seven-section architecture and about-100-word abstract
> constraint.

Exact second writing task:

> Produce Figure 1 and draft Section 3, "Complementarity Beyond Handcrafted
> Families," using the joint generalization artifacts and old A/B/C comparison.

