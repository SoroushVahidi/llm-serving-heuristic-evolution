# LLM 2026 Related Work Plan

Date: 2026-08-24

Purpose: compact routing notes for Section 7. Do not treat this as a completed
literature review; it records which clusters the final manuscript must cover
without expanding the current Sections 5-6 draft.

Final-pass status: the four clusters below were incorporated compactly into
Section 7 on 2026-08-24. The paper cites primary-source or canonical references
for the main clusters and preserves stronger literature expansion for author
review rather than adding a long survey.

## A. LLM Serving And Scheduler Mechanisms

- Orca introduced iteration-level scheduling for transformer serving; our work uses a broader scheduler-portfolio lens rather than proposing a single batching system.
- vLLM/PagedAttention targets memory-efficient serving and exposes native scheduler controls; our real-system section studies how those controls differ from simulator abstractions.
- Sarathi-Serve uses chunked prefill to manage throughput-latency tradeoffs; our Family-B simulator is motivated by chunking but is not a faithful implementation of Sarathi's full stall-free algorithm.
- DistServe, Llumnix, and Mooncake explore disaggregation and cluster-level serving architecture; our paper studies portfolio exploitability rather than architecture design.
- VTC studies fairness in LLM serving; our WFS baseline is a repository-specific weighted service-deficit heuristic inspired by classical fairness and LLM token fairness.
- SOLA, FastServe, QLM, learning-to-rank schedulers, PARS, online KV scheduling, WAIT, and Nested WAIT are relevant modern scheduling baselines/contrasts for Section 7, but this paper does not claim to reproduce them.

## B. Algorithm Selection / Portfolios

- Rice formalized algorithm selection; we adopt this framing rather than claiming novelty for oracle-versus-selector analysis.
- Gomes and Selman, SATzilla, AutoFolio, SUNNY, and Hydra develop solver portfolios, virtual-best comparisons, and closed-gap perspectives; our contribution is specializing this methodology to LLM-serving scheduler portfolios under closed-loop serving dynamics.
- The manuscript should use SBS/VBS terminology explicitly and avoid claiming a new algorithm-selection concept.

## C. Hyper-Heuristics / GP

- Classical selection hyper-heuristics study state-dependent choice among dispatching rules; our selector/router experiments are LLM-serving instances with frozen utility gates.
- Koza, Montana, Whigham, and GP scheduling hyper-heuristics provide the methodological background for typed symbolic search; our typed-GP screen is not a first GP scheduler claim.
- QDGP/MAP-Elites-style scheduling work is a useful contrast for diversity-oriented synthesis, but the current paper tested a smaller equal-budget typed-GP screen.

## D. Program Evolution / Adaptive Serving

- FunSearch demonstrates program search with language models in mathematical domains; our screen used constrained offline grammar search and no LLM-guided mutation.
- Autopoiesis evolves LLM-serving policy programs under runtime dynamics; our experiment instead tests exact-parent typed synthesis and portfolio-envelope marginal gain, and it produced a no-go result for structural crossover.
- Section 7 should position Autopoiesis as an important constructive precedent and avoid any claim that this is the first evolutionary LLM scheduler.
