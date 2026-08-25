# Cross-Family Transfer Well-Posedness — Higher-Level Reassessment v1

Date: 2026-08-17

## 0. Scope

**REASSESSMENT ONLY.** No models trained, no third target redesign, no new
workload families, no composition/synthesis/mechanism-attribution work, no
frozen artifact modified. This task synthesizes three consecutive
independently-scoped NO_GO results into one project-level question: *is
cross-family policy-selection transfer well-posed for Family A/B/C as
currently designed, or should the project adopt a hierarchical /
family-specific formulation instead?*

## A. Initial Git State

Branch `contextual-compositional-heuristics-20260731`, clean, HEAD
`4aaf011` ("feat: mechanism-choice target redesign feasibility
investigation -- `MECHANISM_TARGET_NO_GO`"), already pushed to origin.

## B. Unified Failure-Evidence Table

Read directly from the three authoritative audits (not from living status
docs, per task instruction), cross-checked against their underlying JSON
diagnostics:

| Formulation | Input representation | Target | Source of failure | Failure category | Fixable without changing the scientific question? |
|---|---|---|---|---|---|
| Flat 6-policy selector ([`multifamily_contextual_selector_v1_20260817.md`](multifamily_contextual_selector_v1_20260817.md)) | 33-col, family-prefixed, structurally-missing MF-PSD schema | Exact top-1 policy (6-way) | 100% family-predictable from missingness alone; pooled (Regime B) underperforms majority baseline; LOFO wins 1/3, Family-A-held-out 6.2× worse than best-fixed | **Feature geometry** (structural: missingness itself) | Partially — motivated the next task |
| Shared-feature redesign ([`shared_cross_family_feature_schema_feasibility_v1_20260817.md`](shared_cross_family_feature_schema_feasibility_v1_20260817.md)) | SHARED_CORE_V1, 17-col, zero missingness, semantically shared | Exact top-1 policy (6-way, unchanged) | Still 100% family-predictable — not from structure, but genuine disjoint feature-space regions (range-overlap ≈0 on 15/17 features); cross-family NN pairs *less* utility-consistent than random pairs; independently, 2/6 policies bit-identical outside native family, 4/6 collapse on Family B | **Feature geometry** (now confirmed genuine, not artifact) **+ Policy/target semantics** | No — both root causes are properties of the frozen evidence, not the schema |
| Mechanism-choice target redesign ([`mechanism_choice_target_feasibility_v1_20260817.md`](mechanism_choice_target_feasibility_v1_20260817.md)) | SHARED_CORE_V1 (unchanged) | 3-way mechanism-gain argmax (`ranking`/`chunk`/`kv`) | `kv` contrast confounded (largest on Family A, which has no KV pressure; within-family dose-response absent on A, ρ=−0.13, present on C, ρ=+0.54); majority class corrupted; two-stage oracle-approximation ceiling ties a single fixed policy | **Target semantics** (contrast validity, not merely 6-way collapse) | No — the confound stems from `least_laxity_first` being a generally weak policy outside its native family, a property of the frozen policy library, not the contrast formula |

**Three different root causes across three tasks, not one recurring
problem restated three ways**: (1) a fixable schema-structure artifact
(missingness) that was fixed and turned out not to be the deciding factor;
(2) genuine feature-space disjointness, a property of how the three
pilots' sweep ranges were independently designed; (3) a genuine
policy-competence confound, a property of the frozen policy library. (2)
and (3) are both properties of *already-frozen, already-audited evidence*
— neither is fixable by another representation or target choice without
either new data (broader/overlapping sweeps) or a new policy (a
KV-agnostic-but-otherwise-competent reference policy), both out of this
task's scope and not attempted here.

## C. H1–H5 Well-Posedness Verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** — no smooth universal `shared context → best policy` mapping exists across A/B/C | **SUPPORTED** | All three §B rows independently converge on this; most direct evidence is §D's NN-utility-consistency result (mean Spearman −0.038 on nearest cross-family neighbors, worse than the +0.197 random-pair baseline — "smooth" would require nearby contexts to have similar preferences, and they demonstrably don't) |
| **H2** — the three families are better viewed as different tasks/domains than samples from one continuous distribution | **SUPPORTED** | Feature-space range-overlap ≈0 on 15/17 SHARED_CORE_V1 features (prior audit §I); `max_active_sequences`/`max_kv_tokens` are literal per-family hardware constants (not sampled — a design choice, §D below); `token_footprint_per_kv` differs by >13× between Family A (0.58, no pressure) and Family C (7.64, by-design pressure) — these are qualitatively different operating regimes, not draws from one continuum |
| **H3** — useful generalization is mostly WITHIN family, not ACROSS family | **SUPPORTED** | Step-3 Regime A (within-family): 0 regret on Family A/B holdouts, competitive on C. Regime C (LOFO): only 1/3 directions win, Family-A-held-out collapses to 6.2× worse than best-fixed. Family-specific oracle gaps (§D3) are all small (0.020–0.049) — within-family, a single fixed policy already captures nearly all achievable value; the *hard* part specifically is crossing family boundaries |
| **H4** — family identity corresponds to genuinely different system mechanisms, not just a nuisance label | **SUPPORTED** | `chunked_prefill_small`/`full_prefill` are bit-identical (not just similar) outside Family B — chunking is not a weaker version of the same mechanism elsewhere, it is *absent* (B's `service_model_kwargs.enable_decode_prefill_contention` is the only place it is ever turned on). KV-admission-control's operative resource (KV capacity pressure) is structurally different by GPU-config design: `max_kv_tokens` = 200,000 / 8,000,000 / 6,000 for A/B/C respectively — not three samples of one config distribution, three deliberately different fixed choices |
| **H5** — a hierarchical `family/mechanism regime → family-specific selector` formulation is better posed than one universal selector | **PARTIALLY_SUPPORTED** | Conceptually well-motivated by H1–H4 and by real (if modest) cross-family structure (§D1–D2: moderate policy-ranking-similarity ρ=0.53–0.59 for A↔B/A↔C; `estimated_service_time_first` wins the oracle in *all three* families). But two concrete gates are **unverified, not merely assumed**: (a) no evidence yet that regime is inferable from genuinely online-observable, pre-decision state rather than full-scenario retrospective aggregates (§E); (b) `chunk` regime specifically has no SHARED_CORE_V1 feature that represents its defining mechanism (prefill/decode contention) at all — a structural gap for regime detection, not just an accuracy question (§E) |

## D. Quantified Task Separation

All computed from already-frozen artifacts (`unified_utility_matrix_v2`,
no policy re-run), saved to
`experiments/cross_family_transfer_wellposedness_reassessment_v1/task_separation_diagnostics.json`
via `scripts/analyze_cross_family_task_separation_v1.py`.

**D1. Shared-feature distribution distance** — already quantified in the
shared-feature audit (§I): range-overlap ≈0.00 on 15/17 features; only
`n_distinct_request_classes` overlaps fully (every family always uses 2
request classes). Not recomputed here; referenced as frozen evidence.

**D2. Policy-utility ranking similarity between families** (Spearman
correlation of the 6 policies' mean-ANWG vectors):

| Pair | ρ | p |
|---|---|---|
| A vs. B | 0.531 | 0.278 |
| A vs. C | 0.588 | 0.219 |
| B vs. C | 0.266 | 0.611 |

Moderate positive point estimates for A↔B and A↔C, none reaching
significance (n=6 policies, low power by construction). **Algebraic note:**
this is numerically identical to the per-policy mean-regret-profile
correlation the task also asked for (§3 of the prompt) — verified directly,
not assumed: `mean_regret[policy] = mean(oracle) − mean(anwg[policy])`
within a family, a constant-shift transform of the ANWG vector, which
Spearman correlation is invariant to. Reported once to avoid a redundant
duplicate table.

**D3. Oracle-winning policy sets per family:**

| Family | Policies that ever win the 6-policy oracle (win count) |
|---|---|
| A | `estimated_service_time_first` (37), `weighted_fair_share` (23), `kv_constrained_online` (12) |
| B | `estimated_service_time_first` (16), `chunked_prefill_small` (16) |
| C | `chunked_prefill_small` (32), `kv_constrained_online` (19), `estimated_service_time_first` (17), `weighted_fair_share` (4) |

`full_prefill` and `least_laxity_first` **never** win the oracle in any of
the 176 scenarios — consistent with §B's degeneracy findings (they are
either identical to a better policy or generally dominated). `estimated_service_time_first`
wins in **all three families**; `weighted_fair_share` and
`kv_constrained_online` each win in 2/3; `chunked_prefill_small` wins in
2/3. **The oracle-winning set is not empty-overlap across families** — this
is a genuine, if modest, positive signal that some cross-family structure
exists, distinct from (and weaker than) what would be needed for a
per-scenario contextual selector.

**D4. Family-specific oracle gains** (best-fixed-per-family vs. that
family's own 6-policy oracle):

| Family | Best-fixed policy | Best-fixed mean ANWG | Oracle mean ANWG | Gap |
|---|---|---|---|---|
| A | `weighted_fair_share` | 0.7406 | 0.7623 | **0.0217** |
| B | `estimated_service_time_first` | 0.7328 | 0.7822 | **0.0494** |
| C | `kv_constrained_online` | 0.8658 | 0.8861 | **0.0203** |
| Global pooled | `weighted_fair_share` | 0.7829 | 0.8166 | 0.0336 |

Every family-specific gap is small (2.0–4.9 percentage points) — within any
one family, simply picking the single best fixed policy already captures
95–98% of the theoretical 6-policy-oracle value. This is the quantitative
version of H3: there is little headroom left for a within-family selector
to capture that a good fixed default doesn't already get, and *zero* of
that headroom, in aggregate, survives crossing family boundaries (the
global pooled gap, 0.0336, is not meaningfully larger than the largest
single-family gap, 0.0494 — pooling families does not reveal additional
value a cross-family selector could uniquely unlock).

**D5. Nearest-neighbor utility consistency** — already quantified in the
shared-feature audit (§J): mean Spearman correlation of the full 6-policy
ANWG vector between a scenario and its nearest cross-family neighbor is
**−0.038**, vs. **+0.197** for random cross-family pairs. Not recomputed
here; referenced as frozen evidence.

**D6. Within-family vs. cross-family selector difficulty** — already
quantified in the Step-3 audit: Regime A (within-family) achieves 0 mean
regret on Family A and B test holdouts (competitive, not superior, on C);
Regime C (leave-one-family-out) wins only 1/3 directions, with a severe
6.2×-worse-than-best-fixed collapse when Family A is held out. Not
recomputed here; referenced as frozen evidence.

**Synthesis: the families behave like (B) separate domains/tasks, not (A)
one continuous task manifold** — but with a modest, real, non-zero amount
of shared coarse structure (D2, D3) that a purely universal per-scenario
selector could not exploit (D5, D6) but a coarser mechanism/regime-level
framing plausibly could.

## E. Reassessing the Role of Family Identity

Two possibilities named in the task: (A) bad leakage — an artificial
generator label with no operational meaning; (B) legitimate routing
context — family identity corresponds to observable system conditions.

**Evidence points to (B), with real caveats.** Each family's defining
condition has a plausible operationally-observable analogue:

| Family | Defining condition | Plausible online-observable proxy | Present in SHARED_CORE_V1? |
|---|---|---|---|
| A (fairness/starvation) | Multiple tenants with skewed priority contending under load | `priority_cv`, `n_distinct_request_classes` | **Yes** |
| B (prefill/decode contention) | Long-prompt, short-output requests causing decode stalls under uninterrupted prefill | *(no direct mechanism-state feature; only an indirect request-size-mix proxy)* | **Partial** — `mean_prompt_tokens`/`mean_predicted_output_tokens` show a distinctive B signature (large A-vs-B effect sizes, e.g. `mean_predicted_output_tokens` d=18.19), but nothing in SHARED_CORE_V1 observes prefill/decode contention *state* directly (excluded on purpose in the prior audit, §D of that document — it is not definable outside B's simulation config) |
| C (KV pressure) | Total predicted token demand approaching or exceeding KV capacity | `token_footprint_per_kv`, `concurrency_pressure` | **Yes** |

This is **not** simply reintroducing `mechanism_family` as a feature — each
proxy is a genuine derived statistic of the same online-observable
`Request`/`GPUConfig` fields already validated in the shared-feature audit,
computed the same way regardless of family, with no missingness and no
lookup into which pilot generated the scenario. A deployable system could
plausibly monitor `priority_cv` and `token_footprint_per_kv` in real time
without ever knowing which of three historical pilots resembles its
current traffic.

**The caveat that keeps H5/E at PARTIALLY_SUPPORTED, not SUPPORTED:** (1)
Family B has no such direct proxy — only an indirect, unvalidated
request-size-mix correlate; a regime router would likely be weakest exactly
on detecting the one mechanism (`chunk`) that the mechanism-choice-target
audit already showed has zero cross-family activation and the cleanest
on/off semantics of the three. (2) Every feature used above is a
whole-scenario aggregate (MF-PSD's own §P limitation, inherited by
SHARED_CORE_V1) — no test has been run on partial-trajectory,
truly-causal-order-respecting online state, only full-scenario retrospect.
Both are concrete, checkable gates for a future experiment (§10), not
resolved here.

## F. Universal-Selector Assessment

**Not supported by any evidence gathered across three tasks.** §B/§C/§D
converge: family-predictable feature space (even when clean), incoherent
global policy identities (2/6 always-degenerate, 4/6 collapse on B),
confounded mechanism contrasts, and near-zero net oracle advantage from
pooling relative to per-family fixed baselines (D4). Continuing to pursue a
single flat or lightly-reformulated universal selector on this evidence
would not be a good use of the next experiment slot.

## G. Hierarchical-Router Assessment

**Scientifically coherent as a *falsification candidate*, not yet
validated.** Distinct from the failed mechanism-choice target in a
specific, defensible way: Stage 1 would predict *operational regime*
(grounded in real proxies, §E) rather than a *derived utility contrast*
(§B row 3's confound came specifically from taking a difference of two
policies' *outcomes*, which inherits their competence gaps; a regime
classifier trained on request/GPU-state features, not on any policy's
ANWG, cannot inherit that particular confound). Stage 2 would directly
reuse the one unambiguously strong piece of evidence from all three tasks
combined — within-family selection works (D6, Regime A). Two concrete,
unresolved risks: the Family-B/`chunk` proxy gap (§E) and the
whole-trajectory-aggregate-vs.-online-partial-state gap (§E) — both must be
addressed as explicit preregistered gates (§M), not assumed away.

## H. Mixture-of-Experts Assessment

| Approach | Scientific meaning | Deployability | Interpretability | Leakage risk | Transfer required | Current evidence |
|---|---|---|---|---|---|---|
| (A) Universal flat selector | Weak (§F) | High (one model) | Low (opaque boundary) | High (family leaks via missingness/geometry) | Full cross-family | **Against** (3× NO_GO) |
| (B) Hard hierarchical router + family selector | Strong if §E's proxies hold up online (§G) | Medium (two models, clear boundary) | High (each stage has one job) | Low if router uses only online proxies, not family ID | Router only, not the policy selector | **Partially for** (D2/D3/E) |
| (C) Soft mixture-of-experts gating | Weakest of the four here — a smooth gate presumes exactly the smooth manifold H1 rejects | Medium-low (harder to audit than a hard boundary) | Low (blended decisions are hard to explain post-hoc) | Medium (gate could still learn to approximate family ID smoothly) | Full cross-family (worse than (B): needs the gate itself to generalize) | **Against** — H1's rejection of a smooth mapping applies most directly to a smooth gate; not recommended merely because it is the more fashionable choice |
| (D) Independent per-family selectors, no cross-family model at all | Modest but solid (§D6) | High within each deployed regime | Highest (no cross-family claim to defend) | None | None | **Directly supported** (D4/D6), but forgoes the D2/D3 cross-family signal entirely |

**(B) is the most evidence-consistent non-trivial next step; (D) is the
safest fallback if (B)'s unresolved gates (§E, §M) fail; (C) is not
recommended.**

## I. Family-Specific-Selector Assessment

Directly supported, lowest-risk option. Each family's own within-family
gap to oracle is small (D4: 2.0–4.9pp) and Step-3 already showed
near-perfect within-family selection on A/B. The honest limitation: this
option makes **no claim** about handling a scenario that doesn't resemble
any of the three known regimes — it requires knowing (or a Stage-1 router
deciding, §G) which family-specific selector applies before dispatching,
which is exactly the open question (B) vs (D) hinges on.

## J. Revised Novelty Framing

Proposed framing (task's own wording): *"heterogeneous LLM-serving regimes
require regime-aware policy selection, with policy-separation datasets used
to identify regime boundaries and train specialized selectors."*

**Scientifically defensible, evidence-supported, and stronger than the
failed universal-transfer framing** — it does not claim what three
consecutive tasks failed to demonstrate (universal per-scenario transfer),
and it is directly grounded in what those same three tasks *did*
demonstrate (H1–H4, D2–D4, D6). **Novelty is modest, not large**: "regimes
need regime-aware selection" is a well-established idea in the broader
scheduling literature; this project's specific contribution would be the
*empirical falsification trail* — three independently-designed, honestly
negative attempts at a universal formulation, each diagnosing a different
concrete cause, converging on a specific, checkable hierarchical
formulation with named open risks (§E) rather than an assumed one. That
falsification trail, not the hierarchical idea alone, is this project's
actual novel contribution so far. Do not oversell "regime-aware selection"
itself as new.

## K. Is Another Dataset Family Needed?

**Not yet.** A/B/C already give three genuinely distinct, mechanistically-
grounded regimes with strong within-family evidence (D4, D6) — sufficient
for a first hierarchical-routing falsification (§G's Stage 1 vs. Stage 2
split). A fourth family would only be justified *after* that falsification,
specifically to test whether the router generalizes to an unseen regime
type (a meaningfully different question from anything tested here) — adding
one now would not resolve either of §E's open gates (the Family-B proxy
gap, the online-vs-retrospective-aggregate gap), which are about the
*existing* three families' feature representation, not about needing more
regimes.

## L. Recommended Next Experimental Question

**(A) Hierarchical regime-router + family-specific selector.**

Chosen over (B) (family-specific-only, no cross-family claim) because
§D2/§D3/§E provide enough real, if modest, cross-family structure to make
a routing experiment worth the risk, and because (B) is always available
as the fallback if (A)'s gates fail (§H, §M) — running (A) first is
strictly more informative. Chosen over (C) (add a 4th regime first) because
existing evidence gaps are about representation, not regime count (§K).
Chosen over (D) (stop entirely) because within-family selection is a real,
usable positive result already in hand (D4/D6) that a hierarchical
formulation would preserve and a full stop would discard for no evidenced
reason. Not chosen based on existing infrastructure — chosen because it is
the one option that directly targets the two concrete, unresolved
questions this reassessment surfaced (§E) rather than repeating a
previously-falsified formulation shape.

## M. Conceptual GO/STOP Gates for That Future Experiment

(Not run here — preregistration only, for whoever designs that experiment
next.)

1. **Regime classifier must use only online-observable, pre-decision
   features** — no `mechanism_family`, no scenario ID, no outcome/utility
   column (same discipline as SHARED_CORE_V1's own denylist).
2. **Router must be validated on genuinely online/partial-trajectory state,
   not only whole-scenario retrospective aggregates** — §E's second open
   gap; a router that only works in retrospect is not deployable.
3. **The `chunk`/Family-B proxy gap must be resolved or explicitly
   accepted as a known blind spot** — either a genuine online proxy for
   prefill/decode contention state is found, or the router's expected
   failure mode on `chunk`-regime traffic must be characterized and bounded
   before deployment framing.
4. **Routing accuracy must be high enough to avoid catastrophic
   misrouting** — a concrete bar, e.g. worst-case per-regime misroute rate
   bounded such that expected regret stays below the within-family
   selector's own regret budget; not simply "better than chance."
5. **Family-specific selectors must retain their known within-family
   gains** post-integration (Regime A's near-0 regret must not regress
   when wrapped in a router).
6. **End-to-end hierarchy must beat the best single global fixed policy**
   (0.7829 mean ANWG, §D4) — the low bar any reformulation must clear.
7. **End-to-end hierarchy should approach the family-aware oracle** — i.e.
   the sum of each family's own best-fixed value weighted by regime
   frequency (a stronger, family-aware target above the global-fixed bar),
   not the unconditional 6-policy oracle (§D4 already shows the
   family-aware best-fixed baseline is close to each family's own oracle,
   2.0–4.9pp — the router's job is mostly not to lose that, not to close
   an already-small remaining gap).
8. **Misrouting errors must be interpretable** — e.g. attributable to a
   specific observable feature being out of the router's training
   distribution, not an opaque failure.
9. **No hidden family-label leakage** anywhere in the pipeline (same
   discipline this project has enforced since MF-PSD v1).

## N. Project-Level Verdict

**`CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY`**

**"READY" here means ready to be designed and preregistered as the next
falsification experiment — not validated to work.** Three consecutive,
independently-scoped universal-transfer attempts each failed for a
different, well-diagnosed reason (§B), which itself is evidence that the
problem is with the *universal* framing, not with any one representation
or target choice (§C: H1/H2/H3/H4 all `SUPPORTED`). At the same time,
within-family evidence is genuinely strong (D4/D6) and cross-family
structure is not entirely absent (D2/D3, §E's plausible regime proxies) —
enough to justify one more preregistered attempt at a *qualitatively
different* formulation (hierarchical routing) rather than either giving up
on selector work entirely (`SELECTOR_DIRECTION_NO_GO` — too strong, would
discard the real within-family result) or retreating all the way to
`FAMILY_SPECIFIC_SELECTION_ONLY` without testing whether the modest
cross-family signal (§D2/§D3) is usable at the coarser regime-routing
level first.

## O. Scientific Interpretation

1. **Why did three different-looking tasks all fail, and why is that
   itself informative?** Because each addressed a different plausible
   cause of the first failure (structural leakage → geometric disjointness
   → target/contrast confound) and each *found real evidence for its own
   hypothesis rather than merely inheriting the previous failure* — that
   convergent, independently-derived pattern is stronger evidence for H1
   than any single result would be.
2. **Is the project's central hypothesis (contextual multi-family
   selection) falsified?** The *universal, single-selector* version is
   falsified by convergent evidence. The *hierarchical, regime-aware*
   version is not falsified — it has not yet been tested, and this
   reassessment finds a genuine (if partial) evidentiary basis for it that
   the failed universal attempts do not undermine (within-family gains are
   real and were never in question; only cross-scenario, cross-family
   transfer at the finest grain was).
3. **What is the actual risk in the recommended next step?** Not that
   hierarchical routing is untested in general (it's a well-understood
   pattern) — the specific risk is local to this project's evidence: the
   Family-B/`chunk` regime has no clean online proxy in the current
   feature representation (§E), so a router built today would likely be
   weakest exactly where a wrong decision is cheapest to get wrong
   (Family B's own oracle gap, 0.049, is the largest of the three — a
   misrouted Family-B scenario costs more than a misrouted A or C one).
4. **Does this change how the prior two NO_GO verdicts should be read?**
   No — per task instruction, they are not rewritten. This reassessment
   adds a synthesis layer on top; `MULTIFAMILY_SELECTOR_NO_GO`,
   `SHARED_FEATURE_SCHEMA_NO_GO`, and `MECHANISM_TARGET_NO_GO` remain
   exactly as documented in their own audits.

## P. Files Changed

**New (additive only):**
- `scripts/analyze_cross_family_task_separation_v1.py`
- `experiments/cross_family_transfer_wellposedness_reassessment_v1/task_separation_diagnostics.json`
- `docs/audits/cross_family_transfer_wellposedness_reassessment_20260817.md` (this document)

**Confirmed unmodified:** `experiments/mf_psd_v1/`,
`experiments/unified_utility_matrix_v2/`,
`experiments/shared_cross_family_features_v1/`,
`experiments/mechanism_choice_target_feasibility_v1/`,
`experiments/multifamily_contextual_selector_v1/`, and every one of the
three prior audit documents (byte-unmodified — this task only adds a new
document, never edits an existing audit).

## Q. Tests / Checkers

No new unit tests were added — this task performs no new computation
subject to correctness risk beyond straightforward pandas aggregation
already exercised, informally, by hand-verification against the frozen
`unified_utility_matrix_wide_v2.csv` during this audit (e.g. the D2/D3/D4
numbers were cross-checked against the exact same file's already-tested
loader used by `tests/test_unified_utility_matrix_v1.py`). `python3
scripts/check_project_handoff_consistency.py` passes.

## R. Commit / Push State

Committed on `contextual-compositional-heuristics-20260731` and pushed to
`origin`. No force push. See the corresponding commit for the exact SHA.

## S. Exact Single Next Scientific Action

**Design (not launch) a preregistered hierarchical regime-router +
family-specific-selector experiment**, with the §M gates frozen in its
design document before any TRAIN/TEST data is touched — specifically
including an explicit, upfront plan for how the Family-B/`chunk` proxy gap
(§E) and the online-vs-retrospective-aggregate gap (§E) will be tested,
not assumed resolved. Not started here.
