# LLM 2026 Full Manuscript Audit

Date: 2026-08-24

Scope: final manuscript-completion audit for `paper/llm2026/main.tex`. No new
scientific experiment, simulation, selector, router, GP, vLLM, GPU,
Wulver/Vulver, API, TEST, FINAL, DEV redesign, or threshold change was run.

## A. Central Thesis

LLM-serving scheduler portfolios exhibit real workload-dependent
complementarity and measurable VBS/oracle headroom, including in a jointly
varying multi-mechanism workload. However, in the evaluated workloads and
frozen gates, tested contextual selection, hierarchical routing, support
expansion, guarded composition, and typed structural synthesis capture limited
robust deployable gain. Real-vLLM validation further shows that
simulator-derived scheduler mechanisms require native engine semantic checks.

Safe wording:

- "under the evaluated workloads"
- "the tested methods"
- "VBS/oracle headroom"
- "does not imply deployable adaptive gain"
- "native vLLM 0.27.1 in our tested configuration"

Unsafe wording avoided:

- adaptive scheduling is impossible
- GP cannot synthesize schedulers
- the six policies are state of the art
- simulators are unreliable
- the oracle/selector-gap concept is new

## B. Contribution List

1. Portfolio complementarity: a mechanism-diverse six-policy portfolio shows
   workload-dependent winners and positive VBS/oracle headroom.
2. Exploitability analysis: classical SBS/VBS methodology is instantiated in
   LLM-serving scheduler portfolios, showing that regime detectability is not
   sufficient for closed-loop scheduling utility.
3. Constructive falsification: support expansion, guarded semantic composition,
   and exact-parent typed synthesis fail frozen robustness/transfer gates under
   the tested formulations.
4. Real-serving semantic validation: a direct simulator-to-vLLM Family-B
   analogue fails due to semantic mismatch, while native vLLM token-budget
   control shows a reproducible tradeoff.

## C. Strongest Evidence For Each Contribution

| Contribution | Strongest evidence | Source |
|---|---|---|
| Complementarity | Joint 240 scenarios, 1,440 successful cells, 59.6% epsilon-0.01 unique-winner fraction, VBS headroom 0.019034 ANWG CI [0.015988, 0.022433] | `experiments/joint_multimechanism_generalization_v1/*` |
| Exploitability gap | Router macro-F1 0.9887 but live gain +0.00616 ANWG < +0.01 gate and closure 0.143 | `experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json`; `docs/current/WORK_STATUS.md` |
| Constructive falsification | Wulver 1/104 DEV rows closer; guarded composition 3.11% and mechanism-ordering failure; typed crossover best MG 0.0 | Section 5 source artifacts |
| Real-serving validation | Direct run 40/40 and 300/300 but no reversal; diagnosis `SIMULATOR_VLLM_SEMANTICS_MISMATCH`; native budget verdict `NATIVE_VLLM_BUDGET_EFFECT_STRONG` | real-vLLM artifacts |

## D. Novelty Positioning

The manuscript now acknowledges that SBS/VBS, closed-gap, and oracle-versus
realized selector analysis are classical algorithm-selection concepts. The
paper's novelty is not the general methodology; it is the LLM-serving
specialization, the heterogeneous scheduler mechanisms, the frozen sequence of
exploitation tests, and the simulator/native-engine semantic validation.

Typed GP and grammar-guided synthesis are also not claimed as novel. The
manuscript positions the GP screen as a controlled LLM-serving instantiation
with exact parent reproduction and a portfolio-envelope marginal-gain objective.

## E. Closest Prior Work

- LLM serving and scheduling: Orca, vLLM/PagedAttention, Sarathi-Serve,
  DistServe, Llumnix, Mooncake, VTC, SOLA, FastServe, QLM,
  learning-to-rank scheduling, Online KV scheduling, WAIT/Nested WAIT.
- Algorithm selection and portfolios: Rice, Gomes and Selman, SATzilla,
  AutoFolio, Hydra, and closed-gap/SUNNY-style portfolio evaluation.
- Hyper-heuristics and GP: Koza, Montana STGP, Whigham grammar GP, and
  dispatching-rule hyper-heuristics.
- Program evolution/adaptive serving: FunSearch and Autopoiesis.

The strongest closest serving-policy synthesis precedent is Autopoiesis; the
manuscript explicitly distinguishes this paper's offline, portfolio-envelope,
exact-parent typed screen from runtime evolutionary serving-policy design.

## F. Claim-Safety Findings

Unsafe-wording scan covered `first`, `novel`, `introduce`, `unprecedented`,
`state-of-the-art`, `proves`, `impossible`, `cannot`, `general`, and
`production`. Remaining occurrences are bounded or explanatory:

- `cannot` appears only for offline/oracle information unavailable to online
  schedulers and for "does not imply stronger methods cannot close the gap."
- `production` appears only in limitations and to avoid production-general
  claims.
- `novel` appears only in "novel TRAIN-side domains," a support-artifact term.
- `first` appears in ordinary prose, not priority claims.

No manuscript sentence asserts impossibility or universal generalization.

## G. Number Audit Result

Every scientific number in the manuscript is present in
`docs/current/llm2026_number_source_of_truth_20260824.md`.

Highest-risk numbers checked:

- Joint headroom: 0.019034 ANWG, CI [0.015988, 0.022433].
- Joint unique-winner fraction: 59.6%.
- Router macro-F1/live gain/closure: 0.9887 / +0.00616 / 0.143.
- Support expansion: 24,314 fingerprints, 156 domains, 1/104 DEV rows closer.
- Guarded composition: 3.11% WFS-regret reduction.
- Typed GP: random 0.011295, mutation 0.002551, crossover 0.0.
- Real-vLLM direct: 40/40 runs, 300/300 requests, max waiting/running 7/4,
  max KV 2.84%, 0 preemptions.
- Native vLLM budget: T512/T4096 trace counts and TTFT/E2E deltas with CIs.

Section 7 adds no new scientific result numbers.

## H. Citation Audit Result

Actually cited keys compile with no undefined citations. Weak secondary sources
were not cited. Metadata was checked against primary or canonical sources where
available:

- USENIX pages for DistServe, Llumnix, VTC, Mooncake FAST 2025, and FastServe
  NSDI 2026.
- NeurIPS proceedings for Efficient LLM Scheduling by Learning to Rank.
- MLSys proceedings listing for SOLA.
- IBM/ACM-facing publication page for QLM.
- arXiv for WAIT/Nested WAIT, Online KV scheduling, and Autopoiesis where no
  stronger publication metadata was used locally.

Known conservative choices:

- QLM is cited as SoCC 2024 from the IBM/ACM-facing record; page numbers were
  not added.
- WAIT/Nested WAIT and Autopoiesis remain arXiv citations.
- Section 7 mentions SUNNY-style closed-gap analyses without adding a dedicated
  SUNNY citation to save space; algorithm-selection lineage is still covered by
  Rice, Gomes/Selman, SATzilla, AutoFolio, and Hydra.

## I. Page-Budget Result

Compile command:

```bash
cd paper/llm2026 && tectonic --keep-logs main.tex
```

Result:

- Compile status: success.
- Final page count: 15 one-column LNCS pages including references.
- Section 7 begins on page 11.
- References begin on page 13.
- No undefined citations, duplicate labels, missing figures, or fatal errors.
- Remaining warnings are non-fatal overfull/underfull boxes, primarily caused
  by compact tables, abstract/keyword text, and bibliography line breaks.

The draft is at the hard working target and should not absorb more prose
without compression.

## J. Remaining Weaknesses

- The paper is page-tight at exactly 15 pages; author edits need compression
  discipline.
- The public-trace result is a saturated replay under one frozen configuration,
  not broad trace evidence.
- The real-vLLM validation is local, 0.5B, one GPU, one vLLM version, and one
  mechanism family.
- The six policies are mechanism-diverse baselines, not exhaustive modern
  systems.
- Several no-go chains rely on frozen gates and controlled workloads; the paper
  must preserve bounded language.
- The related work is compact by necessity and may need author judgment if the
  venue expects broader citation coverage.

## K. Blocking Issue Before Submission

No scientific blocker was found. The manuscript is content-complete for author
review and submission preparation.

Non-scientific pre-submission checks still needed:

- Human author review for clarity and tone.
- Venue-specific anonymization/metadata check.
- Final figure readability check in the compiled PDF.
- Optional line-box cleanup if page budget permits.
- Final bibliography polish for any entries the authors want to strengthen.

## Skeptical-Reviewer Read

1. Research question clear? Yes: portfolio opportunity versus realized adaptive
   gain in LLM-serving schedulers.
2. Central contribution obvious by page 1? Yes, after SBS/VBS framing and four
   contribution bullets.
3. Coherent study rather than experiment log? Mostly yes; Sections 4-5 remain
   dense but are organized by exploitation capability rather than chronology.
4. Negative results informative? Yes, because they are tied to frozen gates,
   positive complementarity, and increasingly strong exploitation attempts.
5. VBS lineage acknowledged? Yes.
6. Modern LLM-serving systems represented fairly? Yes, as related mechanisms,
   not reproduced baselines.
7. Six-policy portfolio sufficiently motivated? Adequate, with explicit
   repository-specific wording.
8. Conclusions supported? Yes, under bounded workload/method language.
9. Real-vLLM validation prominent enough? Yes, Section 6 is a major section and
   has its own figure.
10. Obvious reviewer-objection risk? Main risks are synthetic workload scope,
    page-tight related work, and one local vLLM configuration.
11. Missing experiment submission-blocking? No. Additional experiments would
    broaden scope but are not required for the current claims.
