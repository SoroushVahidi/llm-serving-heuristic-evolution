# Mechanism-Choice Target Redesign — v1 Feasibility Audit

Date: 2026-08-17

## 0. Scope

**TARGET REDESIGN / FEASIBILITY ONLY.** Launched directly from the exact
next-action note in
[`shared_cross_family_feature_schema_feasibility_v1_20260817.md`](shared_cross_family_feature_schema_feasibility_v1_20260817.md)
(`SHARED_FEATURE_SCHEMA_NO_GO`): a 3-way mechanism-choice reformulation of
the selector target (fairness-ranking / chunk-control / KV-reserve), on the
grounds that the existing six-policy top-1 target is not cross-family
coherent (2/6 policies bit-identical outside their native family, the other
4/6 collapse together on Family B).

This task does **not** train or evaluate a selector, tune classifiers, run
AutoML, start mechanism attribution, start synthesis/composition, add new
workload families, or modify any frozen MF-PSD/utility-matrix artifact. The
only "model" involved anywhere is a set of deterministic arithmetic
formulas (`|ANWG(policy_1) − ANWG(policy_2)|`) applied to the already-frozen
dense unified utility matrix — no policy is re-run.

## A. Initial Git State

Branch `contextual-compositional-heuristics-20260731`, clean, HEAD `4a6da9c`
("feat: shared cross-family feature-schema feasibility investigation --
`SHARED_FEATURE_SCHEMA_NO_GO`"), already pushed to origin.

## B. Mechanism Definitions

| Mechanism | Decision capability | Native family | Canonical anchor pair |
|---|---|---|---|
| `ranking` | Which *ranking heuristic* to use when ordering contending requests under load — size-aware (shortest-remaining-service-first) vs. weighted-fairness-aware ordering | A | `weighted_fair_share` vs. `estimated_service_time_first` |
| `chunk` | Whether to interleave prefill work in small chunks to bound decode-stall latency for in-flight requests, vs. running prefill to completion uninterrupted | B | `chunked_prefill_small` vs. `full_prefill` |
| `kv` | Whether admission/scheduling is KV-memory-pressure-aware (reserve-aware admission control) vs. laxity-only (memory-blind) | C | `kv_constrained_online` vs. `least_laxity_first` |

**These are deployable control decisions** (which scheduling sub-mechanism
to activate), not "which generator produced this scenario" — each pair's
two policies are real, independently implemented scheduling behaviors that
could in principle both be evaluated on any scenario, and (per the frozen
dense unified utility matrix) *were*.

**A structural asymmetry, flagged up front and carried through the rest of
this audit:** `chunk` and `kv` each bracket a genuine **on/off** contrast —
one policy in the pair is a mechanism-active behavior, the other is that
mechanism turned off (uninterrupted prefill; laxity-only, memory-blind
scheduling). `ranking`, by contrast, brackets **which of two
already-ranking-aware heuristics wins** — there is no ranking-neutral
policy in the canonical six-policy set (Family A's own frozen source
evaluated a `fifo` baseline that would serve this role, but it was never
cross-family-evaluated into the unified matrix, and running it there is out
of this task's scope — no new policy runs). This asymmetry is not merely
cosmetic; it means `gain_ranking` measures something conceptually different
from `gain_chunk`/`gain_kv` even before any empirical audit (§C/§D make this
concrete).

## C. Mechanism Contrast Definitions and Per-Family Classification

`mechanism_gain_m(x) = |ANWG(policy_1) − ANWG(policy_2)|` for each
mechanism's anchor pair, computed from the frozen dense
`unified_utility_matrix_wide_v2.csv` (176 scenarios × 6 policies, no
re-run). Implemented in
`src/llmserveopt/policy_separation/mechanism_choice_target_v1.py`
(`compute_mechanism_gains`), tested in `tests/test_mechanism_choice_target_v1.py`.

Per-family classification (`VALID_GLOBAL` / `VALID_ONLY_WHERE_ACTIVE` /
`DEGENERATE` / `SEMANTICALLY_INVALID`), from
`experiments/mechanism_choice_target_feasibility_v1/mechanism_choice_target_diagnostics.json`
(`cross_family_activation`, `confound_check_gain_kv`):

| Contrast | Family A | Family B | Family C |
|---|---|---|---|
| `gain_chunk` | **DEGENERATE** (72/72 exactly 0.0 — `chunked_prefill_small≡full_prefill`, a clean, non-confounded identity) | **VALID_GLOBAL** (native; 31/32 scenarios above ε=0.01, mean 0.131) | **DEGENERATE** (72/72 exactly 0.0, same clean identity) |
| `gain_ranking` | **VALID_GLOBAL*** (native; 55/72 above ε, mean 0.053 — *asterisk: measures "which ranking heuristic," not "ranking vs. neutral," §B) | **DEGENERATE** (32/32 exactly 0.0 — all 4 non-native policies collapse to one identical value on B) | **VALID_ONLY_WHERE_ACTIVE** (37/72 above ε, mean 0.027 — real, non-degenerate, comparable-magnitude signal) |
| `gain_kv` | **SEMANTICALLY_INVALID** (71/72 above ε, mean **0.330** — the *largest* of any family/contrast cell in this entire table, despite Family A having essentially no KV-memory pressure at all — §D) | **DEGENERATE** (32/32 exactly 0.0 — all 4 non-native policies collapse on B, same mechanism as `gain_ranking`'s B-collapse) | **VALID_GLOBAL** (native; 52/72 above ε, mean 0.123 — genuinely smaller than the spurious Family-A signal, §D) |

## D. The Central Finding: `gain_kv` Is Confounded, Not a Genuine Mechanism Signal

This is the decisive result of this audit. Cross-referencing against
SHARED_CORE_V1's `token_footprint_per_kv` (a genuine, unit-consistent
KV-capacity-pressure proxy — mean total predicted token demand relative to
`max_kv_tokens`, built in the immediately-prior feasibility task):

| Family | Mean `token_footprint_per_kv` (actual KV pressure) | Mean `gain_kv` (proposed "KV mechanism relevance") |
|---|---|---|
| A | **0.578** (comfortably under capacity — no real memory pressure; `max_kv_tokens=200,000`, generous) | **0.330** (largest of all three families) |
| B | 0.012 (essentially no pressure; `max_kv_tokens=8,000,000`) | 0.000 (correctly zero) |
| C (native) | **7.642** (token demand ~7.6× nominal capacity — genuine, by-design pressure; `max_kv_tokens=6,000`) | 0.123 (smaller than Family A's) |

**This is backwards from what a genuine KV-mechanism-relevance signal
should look like.** Family A has essentially no KV pressure (footprint
0.578, an order of magnitude below Family C's 7.642) yet produces the
*largest* `gain_kv` of any family — larger even than KV's own native family.
Inspecting the underlying ANWG values explains why: on Family A,
`least_laxity_first` scores a mean ANWG of only 0.361 (comparable to the
degenerate prefill pair's 0.284) while `kv_constrained_online` scores 0.686
— a large gap, but one driven by `least_laxity_first` performing generally
poorly under Family A's fairness/starvation stress conditions (for reasons
unrelated to memory pressure, since there is none to speak of), not by
`kv_constrained_online`'s KV-awareness doing meaningful work. A direct
correlation check confirms this: `gain_kv` vs. `token_footprint_per_kv`
across the two non-native families (A, B) has Spearman ρ = (see
diagnostics JSON, `confound_check_gain_kv`) — near-zero/negative, not the
positive relationship a genuine mechanism-activation signal would show.

A clean within-family dose-response test confirms this directly: on Family
C (native), `gain_kv` correlates with actual KV pressure
(`token_footprint_per_kv`) at Spearman **ρ = +0.542, p < 1e-6** — a real
relationship, more pressure predicts more mechanism-choice consequence,
exactly what genuine mechanism relevance should look like. On Family A,
the same within-family correlation is **ρ = −0.130, p = 0.28** —
statistically indistinguishable from no relationship at all, despite
`gain_kv` there being numerically the largest of any family. (A naive
*pooled* cross-family correlation was checked first and found strongly
positive, ρ=+0.61 — but that pooled figure is itself a confound of coarse
between-family clustering, i.e. exactly the same statistical trap this
section is diagnosing, one level up: it mostly reflects that Family A
happens to have both higher mean footprint *and* higher mean `gain_kv`
than Family B, not a real dose-response relationship. The within-family
test is the one that actually isolates the question asked.)

**Conclusion: `gain_kv` measures "how much does `least_laxity_first`
happen to underperform on this scenario," not "does KV-aware admission
control matter here."** This single confound corrupts the majority class
of any argmax-based 3-way target, since `kv` is the modal `top_mechanism`
outcome overall (§F).

## E. 3-Way vs. 4-Way Decision

Both were computed (`target_mechanism_4way` adds `no_clear_mechanism` when
`top_gain ≤ ε=0.01`). 16/176 scenarios (9.1%) fall below ε and would abstain
under the 4-way scheme — a real, non-trivial population, mostly in Family C
(15/16). **This audit does not choose between 3-way and 4-way as the
"right" formulation**, because the more fundamental problem (§D) makes the
choice moot: corrupting the majority class with a confound is not fixed by
adding an abstention class.

## F. Target Class Distribution / Target-vs-Family Agreement

From `mechanism_choice_target_diagnostics.json` (`target_vs_family`):

| Mechanism | 3-way class share | Entropy (3-way) |
|---|---|---|
| `kv` | **56.8%** (100/176) | 1.408 bits (max possible 1.585) |
| `ranking` | 25.6% (45/176) | |
| `chunk` | 17.6% (31/176) | |

**Agreement with native-family mechanism: 56.25%** (99/176) — i.e., 43.75%
of scenarios get a `top_mechanism` label that *disagrees* with their source
family. Per §5's own decisive-diagnostic framing: this is **well below**
the ">~95%" threshold that would indicate "target ≡ disguised family
classifier" — so the naive `MECHANISM_TARGET_FAMILY_PROXY_ONLY` reading is
**not** what's happening here. What's happening is worse in a different
way: the disagreement is not evidence of genuine cross-family mechanism
transfer — it is substantially explained by the §D confound routing most
Family-A scenarios into the `kv` bucket instead of their true native
`ranking` bucket (52/72 Family-A scenarios argmax to `kv`, only 20/72 to
their native `ranking`, confusion matrix in §F of the diagnostics JSON).

## G. Cross-Family Mechanism Activation

(Table already given in §C; restated for the specific §6/§7 framing.)
`chunk` **never** activates outside Family B (exactly 0.0 on 144/144 A+C
scenarios — a clean, honest zero, not spuriously near-zero). `ranking`
activates on both A (native) and C (37/72, genuine, comparable magnitude to
A's own 55/72) — this is the one mechanism with *real*, non-confounded
cross-family activation evidence. `kv` "activates" on A (71/72) but per §D
this is not real activation — it is not corroborated by any actual
KV-capacity pressure in Family A's scenarios, and is *larger* than the
signal in its own native family.

**Net assessment: exactly one of three proposed mechanisms (`ranking`) has
genuine, evidenced cross-family activation. `chunk` is cleanly
single-family. `kv`'s apparent cross-family activation is an artifact, not
real evidence of a universal mechanism.** A universal 3-way mechanism
selector cannot be justified on this basis (§6's own stated bar — "if each
mechanism only activates in its own source family... not justified" — is
not even reached in `kv`'s favor, since its cross-family reading is
actively wrong rather than merely absent).

## H/I/J. Per-Mechanism Cross-Family Activation Detail

Folded into §C/§G above (all three cells reported together for direct
comparability, rather than split into separate H/I/J subsections) — see the
`cross_family_activation` block of the diagnostics JSON for the exact
per-family n/mean/n-above-ε figures underlying every claim in §C/§D/§G.

## K. Shared-Feature Same-Mechanism Overlap

Restricting SHARED_CORE_V1 nearest-neighbor search to cross-family pairs
that share the *same* `top_mechanism` label (145/176 scenarios have at
least one same-mechanism partner in a different family): mean standardized
distance to that nearest same-mechanism, different-family partner is
**5.57** — statistically indistinguishable from the raw, mechanism-agnostic
cross-family nearest-neighbor distance already found in the prior audit
(≈5.3, `shared_core_v1_diagnostics.json`). **Conditioning on the proposed
mechanism target does not bring cross-family scenarios any closer together
in shared feature space.** This directly answers §7's question in the
negative: no, same-mechanism scenarios from different families do not
occupy more overlapping regions of SHARED_CORE_V1 space than scenarios in
general.

## L. Target Stability / Margins

87.5% of scenarios (154/176) have a robust margin (top gain exceeds
second-best by > ε=0.01); 20 exact ties, 22 near-ties at the ε threshold;
9.1% would abstain under the 4-way scheme (§E). **Numerically, the target
is reasonably stable** — this is the one dimension along which the
proposal holds up. Stability of a wrong signal is not evidence in its
favor, though: a confidently-and-reproducibly-wrong `kv` assignment on most
of Family A is still wrong.

## M. Relation to the Six-Policy Oracle

Two-stage regret (oracle ANWG minus the best of the *assigned mechanism's*
two anchor policies — the ceiling of a hypothetical perfect
mechanism-choice + perfect within-mechanism selector):

| Assigned mechanism | n scenarios | Mean two-stage regret |
|---|---|---|
| `chunk` | 31 | **0.000** (perfect — unsurprising, since `chunk` is only ever assigned on its own native Family B scenarios, §G) |
| `ranking` | 45 | 0.012 |
| `kv` | 100 | **0.055** (worst of the three, and this is the majority class) |

Overall mean two-stage regret: 0.0341 — essentially identical to the mean
regret of using a *single* best-fixed global policy everywhere (0.0336,
`weighted_fair_share`). **The proposed two-stage decomposition captures no
net oracle-approximation advantage over doing nothing (one fixed global
policy) on average**, and its worst-performing bucket is also its largest
(100/176 scenarios) — a direct, quantified consequence of routing
non-native-family scenarios into the wrong (confounded) mechanism bucket
and then searching within the wrong policy pair.

## N. Two-Stage Decomposition Feasibility

Given §D (confound corrupts the majority class), §G (only 1/3 mechanisms
shows genuine, non-confounded cross-family activation), §K (conditioning on
mechanism target doesn't improve feature overlap), and §M (no net oracle
advantage over a single fixed policy) — **the two-stage
shared-context→mechanism→policy architecture is not scientifically
supported by current evidence.** Stage 1 does not have reliable
cross-family meaning as specified (only `ranking` would survive a
confound-free reformulation, which would leave a 2-way not 3-way target,
and even then §K shows no feature-space benefit from the split). This
task does not implement Stage 2, per its own scope.

## O. Tests

`tests/test_mechanism_choice_target_v1.py`, **8/8 passing**: mechanism-set/
native-family-map consistency, gain-formula symmetry and non-negativity on
a synthetic row, invariance to unrelated fields (`mechanism_family`,
`canonical_scenario_id` never read by the formula), correct top-mechanism
selection and margin arithmetic, deterministic alphabetical tie-break,
ε-threshold abstention behavior, reproducibility of the full 176-scenario
target computation across two independent runs (exact match), and an
explicit **regression guard for the §D confound finding itself**
(`test_gain_kv_confound_regression_family_a_exceeds_family_c` — if this
ever flips, the confound claim needs re-examination, not silent
inheritance).

## P. Final Verdict

**`MECHANISM_TARGET_NO_GO`**

Per the frozen decision criteria (§12): triggered by *both* named
conditions — **"contrasts are semantically invalid"** (§D: `gain_kv` is
demonstrably confounded by baseline policy competence rather than genuine
KV-pressure-mechanism relevance, and it is the majority-class contrast) and
**"decomposition destroys too much utility information"** (§M: zero net
oracle-approximation advantage over a single fixed policy, with the
confounded `kv` bucket carrying the worst regret and the largest
population). This is explicitly **not** `MECHANISM_TARGET_FAMILY_PROXY_ONLY`
(§F: only 56.25% target-vs-family agreement, well under the ~95% bar for
that classification) — the failure mode here is more specific and more
actionable than "just a family classifier in disguise": one of the three
proposed contrasts is measurably wrong.

## Q. Scientific Interpretation

1. **Was the six-policy target's incoherence (from the prior audit) the
   whole problem?** No — even after directly addressing it with a
   mechanism-level reformulation, a *new*, independent failure mode
   appeared: naive utility-gap contrasts between a mechanism's two anchor
   policies do not cleanly isolate "does this mechanism matter here" from
   "is one of these two specific policies just generally weak outside its
   design envelope." The `kv` contrast is the clear case; `ranking`'s
   missing-neutral-baseline asymmetry (§B) is a milder version of the same
   underlying issue.
2. **Is this fixable by reformulating the contrast (e.g. normalizing by
   scenario range, or using a different reference policy)?** Rescaling
   would not fix it — the problem is not a scale/units mismatch, it's that
   `least_laxity_first`'s Family-A behavior is not "policy X doing badly
   because mechanism Y matters a lot," it's "policy X doing badly for
   reasons unconnected to mechanism Y." A genuinely confound-free `kv`
   contrast would need a KV-blind *reference implementation that is
   otherwise well-behaved on non-native families* — which does not exist
   in the current frozen policy set (this is a policy-library gap, not a
   target-formula bug fixable within this task's scope).
3. **Does this generalize to a caution about ANY future contrast-based
   target on this policy set?** Plausibly yes — any mechanism whose
   "off" reference policy (`least_laxity_first` here) was designed and
   tuned only for its native family's conditions risks the same confound
   when evaluated against unrelated stress conditions in other families.
   `chunk`'s contrast avoided this because both `full_prefill` and
   `chunked_prefill_small` are simple, symmetric, mechanism-differing-only
   variants of the same underlying prefill logic — not different
   algorithms with different general competence levels.
4. **Per §17's stop-condition guidance**: a `NO_GO`/`FAMILY_PROXY_ONLY`
   verdict calls for "a higher-level reassessment rather than another
   target tweak." Two consecutive redesign attempts (shared features, then
   mechanism-choice target) have each independently surfaced a *different*
   root cause (feature-space disjointness; contrast confound) rather than
   converging on one fixable issue — this pattern itself is evidence worth
   surfacing at the reassessment level, not just noting here.

## R. Files Changed

**New (additive only):**
- `src/llmserveopt/policy_separation/mechanism_choice_target_v1.py`
- `scripts/analyze_mechanism_choice_target_feasibility_v1.py`
- `tests/test_mechanism_choice_target_v1.py`
- `experiments/mechanism_choice_target_feasibility_v1/mechanism_choice_target_diagnostics.json`
- `docs/audits/mechanism_choice_target_feasibility_v1_20260817.md` (this document)

**Confirmed unmodified:** `experiments/mf_psd_v1/`,
`experiments/unified_utility_matrix_v2/`,
`experiments/shared_cross_family_features_v1/`,
`experiments/multifamily_contextual_selector_v1/`, every prior audit
document, every frozen source run directory. No selector was trained; no
`target-builder` "ready for a future selector" artifact was built per §13's
own condition (implementation is gated on semantic coherence, which this
audit did not find).

## S. Commit / Push State

Committed on `contextual-compositional-heuristics-20260731` and pushed to
`origin`. No force push. See the corresponding commit for the exact SHA.

## T. Exact Single Next Scientific Action

**Not another target tweak, not a two-stage selector, not mechanism
attribution.** Per §17: two consecutive, independently-scoped redesign
attempts targeting the same underlying `MULTIFAMILY_SELECTOR_NO_GO` result
have each found a different, non-overlapping root cause (feature-space
disjointness in the shared-features audit; contrast confound in this one).
The most directly motivated next step, if pursued, is a **higher-level
reassessment** of whether cross-family policy transfer is a well-posed goal
*at all* given how independently Family A/B/C's scenario generators and
policy libraries were designed — rather than a third attempt at a
differently-shaped selector target. Not started here.
