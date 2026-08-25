# Algorithm Stress-Test Library — Literature Research (2026-08-05)

Primary-source research backing every stress-test catalog entry in
`configs/stress_tests/algorithm_stress_test_catalog.yaml`. Each claim
below is classified into exactly one of the four required evidence
classes. Sources are cited by title/authors/venue with a URL where a
canonical arXiv/DOI/GitHub link was found via WebSearch; page/section
pointers are given where practical (many of these are classic
theory-only results without a fixed public PDF page count, in which case
the theorem/result name is cited instead).

## Evidence-class definitions (recap, applied strictly below)

- **PROVEN_WORST_CASE** — a formal theorem, bound, or explicit adversarial
  construction in a peer-reviewed/preprint source.
- **DOCUMENTED_LIMITATION** — stated as a known limitation by the
  algorithm's own authors, or by a reliable follow-up study, without
  necessarily being a formal theorem.
- **PAPER_MOTIVATING_STRESS_CASE** — used by a paper (often not the
  algorithm's own) as narrative/experimental motivation for a NEW method,
  implicitly characterizing the baseline's weakness.
- **HYPOTHESIZED_ADVERSARIAL_REGIME** — reasoned from the algorithm's own
  stated assumptions; no literature source found that establishes it
  directly. Used only for this project's own internal-only heuristics
  that have no dedicated paper (ESTF, WSP-as-implemented, SCORPIO-style
  guard's specific parameterization, the regression selector's specific
  training regime).

---

## 1. FIFO / FCFS

**Target case.** No dedicated optimality paper is needed — FIFO's target
regime (homogeneous service times, low contention) is definitional, not a
literature claim requiring external support.

**Counter case — head-of-line blocking.**
- **Source:** general queueing-theory result, well-established; formal
  treatment in a multiserver job-queueing model: "In the multiserver job
  queueing model with FIFO discipline, HOL blocking occurs when the job
  at the front of the queue cannot be served ... which prevents all
  subsequent jobs from entering service, even if enough servers are
  available." (ScienceDirect, "The Multiserver Job Queuing Model with big
  and small jobs," 2025, https://www.sciencedirect.com/science/article/pii/S0166531625000112)
- **Directly LLM-serving-specific reinforcement:** Fu et al., "Efficient
  LLM Scheduling by Learning to Rank," NeurIPS 2024
  (https://arxiv.org/pdf/2408.15792) states its own motivating problem
  explicitly: "Most LLM serving systems use FCFS ... due to the
  unpredictable output length of requests, which leads to
  Head-Of-Line (HOL) blocking and reduced performance." Also: "Clairvoyant:
  Predictive SJF Scheduling to Mitigate Head-of-Line Blocking in Serial
  LLM Backends" (https://arxiv.org/pdf/2606.07248) is built specifically
  around this failure mode of FCFS LLM serving.
- **Evidence class: PAPER_MOTIVATING_STRESS_CASE** (the general queueing
  HOL-blocking mechanism is well-established theory, but the SPECIFIC
  claim "this is a real, serving-relevant failure of FCFS" is grounded by
  citing papers that build alternative methods explicitly to fix it —
  the strongest class this specific claim supports without a dedicated
  formal FIFO-in-LLM-serving worst-case theorem).
- **Simulator feasibility:** yes — one long-prompt/long-output request
  followed by many short ones, arriving in that order, is directly
  constructible with `Request`.
- **Real-system validation required:** no — this is a scheduling-order
  effect fully captured by this simulator's queuing-delay model.

---

## 2. EDF (Earliest Deadline First)

**Target case.** Classical real-time scheduling theory (Liu & Layland,
1973, the foundational EDF optimality result for uniprocessor systems
with utilization ≤ 1) — EDF is provably optimal (any feasible schedule
exists iff EDF finds one) under that regime. Not re-derived here as a new
search result; it is the textbook baseline every real-time scheduling
course states, and is not in dispute.

**Counter case — the domino effect under transient overload.**
- **Source:** classic real-time systems result, documented consistently
  across course materials derived from the primary literature (Buttazzo,
  *Hard Real-Time Computing Systems*, and the original transient-overload
  analyses it surveys): "In real-time operating systems under EDF,
  whenever a task in transient overload condition misses its deadline,
  each of the other tasks starts missing their deadlines one after the
  other in sequence — the domino effect ... it jeopardizes the behaviour
  of the whole system." Consistent with Giuseppe Lipari's EDF lecture
  notes (http://retis.sssup.it/~lipari/courses/rtos/lucidi/edf.pdf) and
  standard treatments (e.g. microcontrollerslab.com's EDF explainer,
  cross-checked for consistency, not relied on alone).
- **Evidence class: PROVEN_WORST_CASE** (this is a formally characterized
  cascading-failure construction under overload — the standard
  adversarial-arrival argument: introduce one deadline-violating task and
  EDF continues admitting by-deadline order regardless, pushing every
  other task's completion later, missing deadlines in cascade).
- **Simulator feasibility:** yes — an arrival burst whose combined demand
  exceeds GPU throughput capacity for a sustained window, with tight,
  clustered deadlines, directly exercises this.
- **Real-system validation required:** no.

---

## 3. LLF (Least Laxity First)

**Target case.** LLF is the natural refinement of EDF for
service-time-heterogeneous workloads — using `laxity = deadline - now -
est_remaining_service` instead of raw deadline is the textbook motivation
(see any real-time scheduling survey covering LLF vs. EDF; not
independently re-derived here).

**Counter case — laxity instability under prediction error.**
- **Source:** LLF's laxity computation depends on `est_remaining_service`,
  which for this project's implementation
  (`policies/least_laxity_first.py`) is an online-observable α/β proxy,
  not ground truth. This is the SAME class of problem formally analyzed
  by Wierman & Nuyens, "Scheduling Despite Inexact Job-Size Information"
  (SIGMETRICS 2008; discussed with concrete bounds in Wierman's broader
  "Uniform Bounds for Scheduling with Job Size Estimates,"
  https://arxiv.org/pdf/2110.00633) — inexact-size scheduling policies
  suffer bounded but real graceful-degradation as estimate error grows.
- **Evidence class: PROVEN_WORST_CASE** for the general
  inexact-service-time degradation phenomenon (formal bounds exist in the
  cited works for the closely related SRPT/SJF family — see §4); **downgraded
  to HYPOTHESIZED_ADVERSARIAL_REGIME specifically for LLF's laxity
  formula**, since no source found analyzes laxity-based (as opposed to
  size-based) scheduling directly — the mechanism is structurally
  analogous but not literally the same objective function analyzed by the
  cited papers.
- **Simulator feasibility:** yes — construct requests with systematically
  biased `predicted_output_tokens` (understated relative to what the
  workload's α/β cost model implies) clustered near their deadlines.
- **Real-system validation required:** no.

---

## 4. SOF (Shortest Output First) / ESTF (Estimated Service Time First) / WSP (Weighted Shortest Processing)

**Target case — SJF/SRPT optimality.** Classical: SRPT/SJF minimizes mean
flow time under TRUE service times (well-established, textbook result,
e.g. Schrage 1968's original SRPT-optimality proof — not re-derived
here). Smith's rule (WSPT) is separately, formally proven optimal for
minimizing weighted completion time on a single machine:
- **Source:** "Smith (1956) introduced the shortest processing time (SPT)
  rule and proved that, when all jobs have identical release times, SPT
  is optimal for minimizing the total completion time. This principle was
  later extended to the weighted setting through the weighted shortest
  processing time (WSPT) rule ... proof [via] an exchange argument."
  (Pinedo scheduling course materials, cross-referenced against
  "Analysis of Smith's Rule in Stochastic Machine Scheduling,"
  https://marcuetz.personalweb.utwente.nl/Preprints/WSEPT-Final.pdf, which
  extends the deterministic result.)
- **Evidence class: PROVEN_WORST_CASE-adjacent target case** (this is a
  proven OPTIMALITY result under TRUE processing times — cited here as
  the formal basis for these policies' TARGET regime, not their failure
  mode).

**Counter case — starvation under continuous short-job arrival.**
- **Source:** "SRPT has the potential for process starvation: long
  processes may be held off indefinitely if short processes are
  continually added ... It has often been cited that SRPT may lead to
  starvation of large jobs, [demonstrated via] adversarial arrival
  sequences." Formally analyzed for UNFAIRNESS (not just starvation) in
  Bansal & Harchol-Balter, "Analysis of SRPT Scheduling: Investigating
  Unfairness," SIGMETRICS 2001
  (https://www.cs.cmu.edu/~harchol/Papers/Sigmetrics01.pdf).
- **Evidence class: PROVEN_WORST_CASE** (explicit adversarial
  construction: a continuous stream of short-job arrivals provably
  starves a long job indefinitely under strict SRPT/SJF; Bansal &
  Harchol-Balter additionally provide a formal unfairness bound, not just
  an anecdote).

**Counter case — prediction-error degradation (ESTF/SOF/WSP specifically,
since they rank by PREDICTED, not true, service time).**
- **Source:** Lykouris & Vassilvitskii, "Competitive Caching with Machine
  Learned Advice," (the α-consistency/β-robustness framework, foundational
  for the whole "algorithms with predictions" literature, formalized
  clearly in Mitzenmacher, "Scheduling with Predictions and the Price of
  Misprediction," ITCS 2020, https://arxiv.org/pdf/1902.00732) and
  Wierman & Nuyens (SIGMETRICS 2008; bounds restated in
  https://arxiv.org/pdf/2110.00633): "SRPT-B is 1-consistent and has
  3.5-graceful degradation; PSJF-E is 1.5-consistent and has 1.5-graceful
  degradation" as job-size-estimate error grows.
- **Evidence class: PROVEN_WORST_CASE** (formal, quantified competitive-ratio
  degradation bounds as a direct function of prediction error — the
  strongest available evidence class, directly transferable to
  ESTF/SOF/WSP's own predicted-length ranking mechanism).
- **LLM-specific reinforcement (why prediction error is realistic, not
  just theoretical):** chain-of-thought / reasoning-heavy prompts are
  documented to have fundamentally hard-to-predict output length ahead of
  time — "a precise global plan does not emerge early in chain-of-thought
  reasoning, even for task-aware models" (research on CoT length
  predictability, cross-referenced across multiple 2026 preprints
  including "Predicting LLM Output Length via Entropy-Guided
  Representations," https://arxiv.org/pdf/2602.11812, and "How Far Ahead
  Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought
  Reasoning," https://arxiv.org/pdf/2602.02103). Evidence class:
  **PAPER_MOTIVATING_STRESS_CASE** for the specific claim "reasoning
  prompts systematically break length predictors."
- **Simulator feasibility:** yes for all — starvation (continuous short
  arrivals + one long job) and prediction-error degradation (a workload
  where `predicted_output_tokens` is systematically biased relative to
  `actual_output_tokens`, subject to the constraint that policies must
  never read `actual_output_tokens` — the BIAS is baked into the
  WORKLOAD GENERATOR's ground truth, not leaked to the policy).
- **Real-system validation required:** no for starvation (pure ordering
  effect); recommended-but-not-required for the magnitude of
  reasoning-prompt length unpredictability (this project has no live LLM
  reasoning-prompt generation pipeline to draw real α/β bias distributions
  from — the bias magnitude used in the workload generator is a
  reasoned, disclosed parameter choice, not measured from a real system).

---

## 5. SCORPIO-style SLO guard

**Target case.** SCORPIO's own paper: Chen et al. (verify author list
directly if citing in a manuscript), "SCORPIO: Serving the Right Requests
at the Right Time for Heterogeneous SLOs in LLM Inference,"
arXiv:2505.23022 (https://arxiv.org/abs/2505.23022), OpenReview
(https://openreview.net/forum?id=LSrpJkynU2), official code
`MisterBrookT/Scorpio`. Confirms the mechanism this project's
`scorpio_style_slo_guard.py` approximates: "TTFT Guard, which employs
least-deadline-first reordering and rejects unattainable requests, and a
TPOT Guard, which utilizes VBS-based admission control and a
credit-based batching mechanism." This project's own file's docstring
(4-part composite: TTFT/deadline guard, TPOT/decode-pressure guard,
composite urgency, admission throttling) is structurally consistent with
the real paper's description, cross-verified here — good evidence the
"SCORPIO-inspired" label is honestly earned, not just aspirational.
Reported target-regime gains: "improves system goodput by up to 14.4X and
SLO adherence by up to 46.5% compared to state-of-the-art baselines."
Evidence class: **DOCUMENTED_LIMITATION-adjacent target case**, directly
sourced from the official paper.

**Counter case — false rejection under noisy/pessimistic estimates.**
- **Source:** no dedicated formal analysis of SCORPIO's own
  admission-threshold false-rejection rate under estimator noise was
  located in this pass (the official paper's own evaluation focuses on
  aggregate goodput/SLO-adherence gains, not a stress analysis of its
  own admission-control component in isolation).
- **Evidence class: HYPOTHESIZED_ADVERSARIAL_REGIME** — reasoned directly
  from the mechanism's own stated assumption (admission decisions are
  made from `laxity`/decode-pressure PROXIES, not ground truth): any
  admission-control policy driven by a noisy or systematically pessimistic
  estimator will reject some feasible requests near its threshold. This
  project's OWN implementation is being stress-tested here, not the
  original SCORPIO system, so this classification is appropriate and
  should not be overstated as literature-backed.
- **Simulator feasibility:** yes — a workload with request laxity
  clustered just above/below the admission threshold, combined with
  deliberately noisy `predicted_output_tokens` (systematic bias, injected
  at the workload-generator level per the same non-leakage constraint as
  §4).
- **Real-system validation required:** recommended for any claim about
  the OFFICIAL SCORPIO system's false-rejection behavior specifically
  (out of scope here — this project's guard is explicitly "NOT an
  official SCORPIO reproduction," per its own file docstring).

---

## 6. Regression ANWG selector

**Target case.** In-distribution generalization is the basic premise of
any supervised learning system — not a claim requiring an external
citation; it is this project's own trained artifact
(`selector/models.py`, `name="regression_anwg"`), evaluated against its
own training-window feature distribution (`docs/current/SELECTOR_V2.md`,
`research_status.md`).

**Counter case — out-of-distribution feature combinations / regime
shift.**
- **Source:** general supervised-learning distribution-shift theory is
  extremely well-established (covariate shift, concept drift — textbook
  material, not a single paper to cite); no LLM-serving-scheduling-specific
  formal treatment of THIS project's own selector's OOD failure modes
  exists in the literature, since this is an internal, project-specific
  model.
- **Evidence class: HYPOTHESIZED_ADVERSARIAL_REGIME** for every specific
  claim in this category (OOD features, rapid regime transitions, feature
  aliasing, misleading correlations, near-boundary policy-utility ties) —
  these are reasoned directly from supervised-learning first principles
  and this project's own selector architecture, not established by a
  dedicated external source.
- **Simulator feasibility:** yes — workloads engineered to fall outside
  the selector's documented training-window feature ranges (see
  `selector/dataset_v2/` for the actual training feature ranges to stay
  OUTSIDE of) are directly constructible.
- **Real-system validation required:** no (this is entirely a property of
  the trained model + this simulator's feature computation, both fully
  available offline).

---

## 7. vLLM-LTR / PARS-Serve-2026 (length/rank predictors)

**Target case.** Both papers' own reported results (already documented
in this project's `baselines/vllm_ltr/`, `baselines/pars/`) — vLLM-LTR:
"reduces latency by 2.8x in chatbot serving and increases throughput by
6.5x in synthetic data generation" (Fu et al., NeurIPS 2024,
https://arxiv.org/pdf/2408.15792). PARS: see
`baselines/pars/PROVENANCE.md` for the full citation (Tao et al., ISC
2026, arXiv:2510.03243).

**Counter case — domain shift / reasoning-prompt unpredictability.**
- Same underlying evidence as §4's prediction-error discussion applies
  directly: both are LEARNED length/rank predictors, subject to the same
  general α-consistency/β-robustness degradation under distribution shift
  (Lykouris & Vassilvitskii framework, Mitzenmacher ITCS 2020) and the
  same CoT-length-unpredictability evidence
  (https://arxiv.org/pdf/2602.11812, https://arxiv.org/pdf/2602.02103).
- **Additionally, this project's OWN prior comparative evaluation**
  (`docs/audits/pars_first_comparative_evaluation_20260804.md`) already
  found PARS "never ranks above 5th of 10 policies in any family, records
  zero unique wins across all 8 families" on this project's own canonical
  suite — this is neither a formal proof nor an official-paper admission,
  but a directly-relevant, already-collected internal empirical result.
  Evidence class: **PAPER_MOTIVATING_STRESS_CASE** is too strong a label
  for this project's OWN prior finding (not a paper); classified here as
  a fifth, project-local category worth flagging distinctly in the
  catalog as `INTERNAL_EMPIRICAL_FINDING` rather than forcing it into one
  of the four literature-evidence classes, since it did not come from
  external literature at all.
- **Evidence class (domain-shift claim specifically): PROVEN_WORST_CASE**
  (via the same formal consistency/robustness bounds as §4) for the
  general mechanism; **PAPER_MOTIVATING_STRESS_CASE** for the
  reasoning-prompt-specific instantiation of it.
- **Simulator feasibility:** yes, WITH A CAVEAT — both policies are
  offline-scored (precomputed score maps, since this simulator has no
  live text/predictor pipeline mid-run — see both baselines' adapter
  docstrings). A domain-shift stress workload must therefore be
  constructed as a DIFFERENT offline scoring pass (different prompt
  corpus/distribution fed to the same predictor), not as an online
  runtime perturbation. This is a real, structural simulator constraint,
  not a simplification chosen for convenience.
- **Real-system validation required:** recommended for any claim about
  the MAGNITUDE of real-world domain shift (this project has offline
  scores only for its own existing corpora; a genuinely adversarial
  out-of-domain prompt set would need new offline scoring runs against
  the real checkpoints — feasible but not attempted in this pass).

---

## Summary table

| Algorithm family | Target evidence | Counter evidence (strongest class found) | Key source(s) |
|---|---|---|---|
| FIFO/FCFS | definitional | PAPER_MOTIVATING_STRESS_CASE | Fu et al. NeurIPS 2024; multiserver HOL-blocking analysis 2025 |
| EDF | classical (Liu & Layland 1973) | PROVEN_WORST_CASE (domino effect) | Lipari EDF lecture notes; standard real-time scheduling theory |
| LLF | classical (EDF refinement) | HYPOTHESIZED_ADVERSARIAL_REGIME (laxity-specific); PROVEN_WORST_CASE (general inexact-size class) | Wierman & Nuyens 2008/2110.00633 |
| SOF/ESTF/WSP | PROVEN (Smith 1956, Schrage 1968) | PROVEN_WORST_CASE (starvation + prediction-error bounds) | Bansal & Harchol-Balter SIGMETRICS 2001; Mitzenmacher ITCS 2020; Wierman & Nuyens |
| SCORPIO-style guard | official paper (arXiv:2505.23022) | HYPOTHESIZED_ADVERSARIAL_REGIME | (no dedicated source for false-rejection stress found) |
| Regression ANWG selector | this project's own training regime | HYPOTHESIZED_ADVERSARIAL_REGIME | (internal, no external source applicable) |
| vLLM-LTR / PARS | official papers | PROVEN_WORST_CASE (general) / PAPER_MOTIVATING_STRESS_CASE (reasoning-specific) / INTERNAL_EMPIRICAL_FINDING | Lykouris & Vassilvitskii; Mitzenmacher; CoT-length-prediction 2026 preprints; this project's own PARS evaluation |

VTC is deliberately excluded from this document (already fully researched
and cited in `baselines/vtc/PROVENANCE.md` and the VTC audit docs — out
of this task's scope per the exclusion list).
