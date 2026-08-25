# Higher-Level Structural Reassessment of the Composition Hypothesis

Date: 2026-08-17

This document synthesizes all experimental composition evidence collected to date (Family A, Family B, KV Family, CC lineages, and Module Credit evaluations) to answer a fundamental scientific question: Should "combine parent heuristics to create a better new heuristic" remain a central hypothesis of this project?

## A. Unified Composition-Evidence Table

| Family / Pair | Mechanism | Cross-Scenario Complementarity | Within-Scenario Complementarity | Selector Quality | Child Type | Non-Degenerate? | Beats Both? | Envelope Gain | Held-Out CI | OOD Result | Safety Result | Final Verdict | Primary Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Family A** (ESTF vs WFS) | Fairness/Starvation (Ranking/Sorting) | Strong (bidirectional niches) | None identified | High (matched oracle exactly on TEST) | Contextual Rank (weighted scoring) | Yes (spent time in both modes) | 0/32 | 0.0 | CI crossed 0 | Consistent (no gain) | PASS | `SELECTION_SUFFICIENT_FOR_THIS_PAIR` | Child collapses behavior to selector; no envelope expansion. |
| **Family B** (full_prefill vs chunked_small) | Prefill/Decode TTFT Contention | Strong (bidirectional niches) | None identified | High (matched oracle exactly on TEST/OOD) | Adaptive threshold (queue depth / prefill size) | Yes | 0/32 | 0.0 | CI crossed 0 | Consistent (no gain) | PASS | `SELECTION_SUFFICIENT_FOR_THIS_PAIR` | Dynamic threshold offers no gain over scenario-level choice. |
| **KV Family** (kv_constrained vs least_laxity) | Memory/KV Reserve (Admission Control) | Strong (bidirectional niches) | **Yes** (timing of urgent arrival matters) | Medium (collapsed to constant classifier in one run) | Dynamic Switch (urgent queue depth trigger) | Yes (frequent active transitions) | **5/12 TEST, 3/24 OOD** | **+0.019 (TEST)** | Positive | Consistent (+ gain) | **FAIL** (Peak KV exceeded max parents) | `KV_COMPOSITION_INCONCLUSIVE` | History-of-composition safety overshoots. |
| **KV Refined** (Hysteresis Child) | Memory/KV Reserve (Hysteresis guarded) | Strong | N/A | N/A | Dynamic Switch + State Hysteresis | Yes | 1/12 TEST, 3/24 OOD | -0.0321 (TEST loss) | Negative | Consistent (loss) | **FAIL** (Overshoots remain) | `KV_COMPOSITION_SAFETY_REFINEMENT_FAILED` | Safe-transition restrictions destroyed the performance advantage. |
| **CC Lineages** | Multiple primitive combinations | Unknown | None identified | N/A | Symbolic/DSL | Varies | N/A | N/A | N/A | N/A | PASS | `COMPLETE_REGIME_SPECIFIC` | Lacked structured policy separation boundaries. |
| **Module Credit** | Generic modular attribution | Unknown | None identified | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `NOT_READY` | Weak generalization/signal in generic environments. |

## B. Repeated Failure Pattern

We assessed the hypotheses based on the above unified evidence:

*   **H-A: Most useful diversity between scheduling heuristics is scenario-level rather than within-trajectory.**
    *   **SUPPORTED.** In Families A and B, a scenario-level selector achieved zero regret (perfectly matching the parent oracle).
*   **H-B: When within-trajectory composition does create additional performance, it relies on state histories that violate or weaken parent safety invariants.**
    *   **SUPPORTED.** The KV child only achieved its +0.019 ANWG envelope gain by filling the conservative parent's 18% reserve buffer. When hysteresis rules were strictly applied to prevent unsafe capacity violations, the performance advantage was destroyed.
*   **H-C: Simple selectors are a stronger inductive bias than direct policy combination for the mechanisms studied so far.**
    *   **SUPPORTED.** Top-1 contextual selectors trivially captured the available envelope for Families A and B, whereas complex within-scenario switching policies required significant tuning, risked safety violations (KV), and frequently failed to exceed the selector.
*   **H-D: Module-level composition may still be scientifically viable, but only when modules are independently composable by construction rather than switching between whole parent semantics.**
    *   **PARTIALLY_SUPPORTED.** The failures occurred specifically when switching between *entire, monolithic parent semantics* mid-trajectory, which resulted in state-history incompatibilities. Composing independent, compatible modules is structurally different but lacks deep evidence here.
*   **H-E: The current evidence weakens only the tested composition operators, not the broader synthesis hypothesis.**
    *   **NOT_SUPPORTED.** The evidence suggests that the *opportunity* for within-scenario synthesis itself is exceedingly narrow. Synthesis algorithms depend on composable atoms, but we have demonstrated that heuristic atoms possess hidden state dependencies that make naive combination actively harmful.

## C-F. Distinguishing the Four Claims

*   **A. Policy Selection ("choose the best existing policy for a context"):**
    *   **POSITIVELY SUPPORTED.** We have strong evidence of bidirectional niches across three distinct mechanism families and have demonstrated that contextual selection can achieve near-zero regret.
*   **B. Online Policy Switching ("switch between existing policies during a trajectory"):**
    *   **NOT SUPPORTED.** Actively falsified in Family A and B (no gain over selection). In the KV family, it produced gains only by violating safety invariants.
*   **C. Module Composition ("combine compatible modules from different schedulers"):**
    *   **INCONCLUSIVE / NOT READY.** Previous modular attribution efforts yielded weak signals. Needs foundational redesign before being viable.
*   **D. Genuine New-Policy Synthesis ("construct a standalone heuristic not equivalent to selecting/switching"):**
    *   **NOT SUPPORTED.** Premature. Synthesis requires a validated space of composable modules, which does not currently exist.

## G. Verdict on Original Composition Hypothesis

**COMPOSITION_DEMOTED**

The main project focus must shift to policy separation + contextual selection across multi-family libraries. Within-scenario composition and synthesis will become exploratory future work, only to be pursued when mechanism attribution predicts a genuinely safe and composable structural opportunity.

## H. Revised Novelty Framing

The original novelty story was:
`policy library -> contextual utility/regret -> composition -> new symbolic children -> envelope expansion`

The scientifically stronger, empirically justified revised novelty story is:
**`policy-separating workloads -> complementary policy library -> contextual selection (multi-family) -> mechanism attribution -> bounded envelope`**

We shift from "we can automatically breed new heuristics" to "we can build a multi-family heuristic library, rigorously separate their performance boundaries via targeted workloads, and use contextual selection to achieve zero-regret scheduling across diverse deployment contexts."

## I. Revised Role of the Policy-Separation Dataset

The Policy-Separation Dataset (PSD) is no longer a stepping stone to symbolic synthesis. Its primary purposes are now empirically justified:
1.  **Train a multi-family contextual selector:** To prove that selection works across different structural mechanism families (Ranking, Prefill/Decode Contention, Memory/Admission).
2.  **Learn pairwise regret/utility:** To build robust contextual performance models.
3.  **Map mechanism boundaries:** To deeply understand *why* heuristics fail in specific contexts.

## J. Whether Another Dataset Family is Needed

**No.** The current dataset coverage is sufficient for the next modeling step. We now possess three completely distinct structural families that have successfully passed rigorous policy-separation audits with confirmed bidirectional niches and low tie rates:
1.  **Family A:** Fairness/Starvation (Ranking)
2.  **Family B:** Prefill/Decode Contention (Chunking)
3.  **Family C (KV):** Memory/KV Pressure (Admission Control)

These three families provide sufficient policy diversity, winner entropy, and mechanism separation to train and evaluate a robust multi-family selector.

## K. Selector-Retraining Readiness

**READY.** The previous block on selector retraining is lifted. Family A v2 + Family B v2 + KV v2 provide high-quality, verified, low-tie-rate scenario boundaries.

*Proposed Concept:* Train a unified selector over the combined dataset.
*Evaluation:* Evaluate generalization across mechanism families and hold out entire families (e.g., train on A/B, test on KV) to test whether the selector learns deep state representations or simply overfits to scenario metadata.

## L. Pairwise-Regret Readiness

**READY.** Alongside multi-family selection, pairwise-regret models can now be evaluated against direct per-policy utility prediction using the diverse PSD.

## M. Module-Attribution Readiness

**PROMISING_BUT_NOT_READY.** Mechanism-specific attribution is promising now that we have separated the families. However, it should be deferred until after the multi-family contextual selector is proven.

## N. Recommended Single Next Scientific Question

**Can a contextual selector trained over the combined policy-separation dataset generalize across mechanism families and held-out families?**

## O. Revised 5-8 Step Roadmap

1.  **Data Unification:** Unify the three `_COMPOSITION_READY` datasets (Family A v2, Family B v2, Family C/KV v2) into a single, canonical Multi-Family Policy Separation Dataset (MF-PSD).
    *   *GO:* Unified schema, combined splits.
2.  **Unified Baseline Evaluation:** Evaluate all anchors (`estf`, `wfs`, `full_prefill`, `chunked_prefill_small`, `least_laxity`, `kv_constrained`) across all scenarios in the MF-PSD to build the full 6-policy utility matrix.
    *   *GO:* Completed utility matrix with no missing cells.
3.  **Multi-Family Contextual Selection:** Train a top-1 contextual selector over the MF-PSD. Compare against per-policy utility and pairwise-regret models.
    *   *GO:* Selector achieves tight regret against the 6-parent oracle on TEST/OOD.
4.  **Hold-Out Family Generalization:** Evaluate selector performance when an entire mechanism family is held out during training.
    *   *GO:* Selector outperforms the best fixed parent on the held-out family.
5.  **Mechanism Attribution (Post-Selection):** Use the selector's learned representations to attribute wins to specific mechanisms (ranking vs admission vs chunking).
    *   *STOP:* Module synthesis is strictly deferred until this attribution yields a reliable, actionable signal.

## P. Explicit Deferred Items

*   All forms of policy synthesis (GP, MAP-Elites, LLM-guided).
*   Any new within-scenario composition policies (online switching).
*   Symbolic distillation of the selector.
*   Real-system vLLM transfer (deferred until the MF-selector is validated).
