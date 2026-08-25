# Family-A Constrained-Formulation Feasibility Audit

Date: 2026-08-21

Diagnostic-only offline audit of whether the Family-A ESTF/WFS decision is
better represented as **maximize completion benefit subject to an explicit
SLO-risk constraint** than as **completion + fixed-weight scalar terminal
value** (the formulation that already failed twice: a uniform in-flight
progress scalar in
`docs/current/family_a_terminal_value_v1_analysis_20260820.md`, and three
coefficient-free contested-scalar proxies in
`docs/current/family_a_contested_request_value_diagnosis_20260821.md`). No
new simulation was run, no controller/policy/simulator code was touched, no
TEST data was read, no RL/DAgger was run, no broad threshold sweep was
performed, and nothing was staged/committed/pushed. All numbers come from
one deterministic, read-only script
(`scripts/analyze_family_a_constrained_formulation_feasibility.py`) over
already-extracted TRAIN/VAL artifacts.

---

## 1. Executive Verdict

Classification: **`CONSTRAINED_FORMULATION_PARTIAL_SIGNAL`**

Next step: **`KEEP_CONSTRAINED_FORMULATION_DIAGNOSTIC_ONLY`**

This is a **substantially stronger** partial signal than the prior
contested-scalar-proxy attempt. Separated into two binary offline targets —
does ESTF's admission causally rescue completion, and does choosing ESTF put
the WFS-favored request at SLO risk — both are **strongly and stably
predictable from ONLINE_CAUSAL state alone** via grouped cross-validation:
logistic regression reaches balanced accuracy 0.740 (completion benefit) and
0.892 (SLO risk), both far above the 0.500 majority baseline and both with
low fold-to-fold variance (std 0.036–0.064). This is categorically better
separability than any of the three scalar proxies tried in the prior
diagnosis (which topped out at or below chance). A resulting
constrained decision rule (SLO-risk gate at the natural 0.5 probability
boundary, then a completion-benefit gate within the permitted region)
produces qualitatively the *right shape*: it suppresses ESTF sharply in
`favlong` (13.3% ESTF share) while keeping it meaningfully available in
`favshort` (41.9% ESTF share) — the desired asymmetry — without collapsing
onto a trivial always-WFS, priority-only, or regime-identity rule (71.4%
overlap with the priority≥5 rule, not full collapse, and materially higher
accuracy in the disagreement cases).

The reason this stops at `PARTIAL` rather than `STRONG` is **§10**: measured
against the only available whole-branch "long-run/native" ground truth
(raw completed-request-count `delta_native`), the constrained rule's
agreement is weak-to-poor, especially in `favlong` (balanced accuracy 0.075
vs. a naive always-ESTF baseline's 0.500 on that same metric). This cannot
be cleanly interpreted, because that same ground truth is independently
**documented as ESTF-biased almost everywhere** (`delta_native` sign favors
ESTF in 58/91 events overall and is `WFS`-favoring in **zero** of the 60
`favlong` events, per the prior diagnosis chain) — exactly the regime where
raw completion count is already established to be a poor surrogate for true
priority/SLO-weighted ANWG. A rule that correctly moves toward WFS in
`favlong` would *necessarily* score badly against this ground truth whether
or not it is actually right. No independent per-event weighted/ANWG ground
truth exists to resolve this without new simulation, which this task
prohibits — so the alignment question is left open, and the classification
reflects that gap honestly rather than resolving it either way.

---

## 2. Preflight

| Check | Value |
|---|---|
| branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `8e1223beb58fd4d296061b6b48e3ba493714108f` |
| upstream | `origin/contextual-compositional-heuristics-20260731` |
| ahead/behind | 0 / 0 |
| worktrees | 1 (main only) |
| lock files | none |
| active scientific jobs / tmux sessions | none (`no server running on /tmp/tmux-1000/default`, no matching `ps` entries) |
| RAM | 24Gi free / 59Gi available of 62Gi |
| disk | 638G available of 835G (20% used) |
| load average | 0.05 / 0.04 / 0.02 |

`git status` shows one pre-existing modified test file
(`tests/test_decision_criticality_timescale_trainval_v1.py`) and a long list
of pre-existing untracked files from prior sessions (docs/experiments/
scripts/src), all predating this task and left untouched. All local changes
preserved; nothing staged.

---

## 3. The Constrained Question (as formulated, not assumed)

Tested decomposition:

```
choose ESTF if:
    predicted P(SLO-risk to the WFS-contested request | choose ESTF) <= 0.5
    AND predicted P(completion-benefit from choosing ESTF) > 0.5
else choose WFS
```

against the alternative already shown to fail: a single scalar
`completion + fixed_weight * SLO/fairness` (the terminal-value redesign) or
`Σ weight_i * causal_i` (the contested-scalar proxies). The audit below
tests, without assuming the answer: (a) whether the two risks are
*separately* representable from state (§8), (b) whether gating on one and
optimizing the other in the permitted region behaves sensibly by regime
(§9–§11), and (c) whether any apparent improvement is trivial (§14).

---

## 4. Online Causal Risk-Signal Inventory

From `ObservableRequest` / `ObservableGPUState` / `ObservableState`
(`src/llmserveopt/core/types.py`) and the already-computed aggregate feature
set (`family_a_observability_continuation_v1.py::extract_causal_features`):

**A. Completion risk**

| Variable | Classification |
|---|---|
| `predicted_service_proxy` (`alpha*prompt_tokens + beta*predicted_output_tokens`) | `ONLINE_CAUSAL` |
| `prompt_tokens`, `predicted_output_tokens` | `ONLINE_CAUSAL` |
| `queue_length`, `active_count` (queue position / capacity pressure) | `ONLINE_CAUSAL` |
| `tokens_decoded_per_request` (current progress, per `ObservableGPUState`) | `ONLINE_CAUSAL` (present in the struct; unused by any current policy but already exposed) |
| `free_sequences` / `free_kv_tokens` (capacity headroom) | `ONLINE_CAUSAL` (`ObservableGPUState` properties) |
| `n_admit_estf`, `n_admit_wfs`, `admit_symmetric_diff_size` (this-step admission geometry) | `ONLINE_CAUSAL` |
| eventual `br_*_completed` | `FUTURE_OUTCOME` (label only) |

**B. SLO/timeliness risk**

| Variable | Classification |
|---|---|
| `slo_deadline`, `arrival_time` → `queue_age` | `ONLINE_CAUSAL` |
| `laxity = slo_deadline - state.time` (aggregate, quantile-summarized) | `ONLINE_CAUSAL`, **unit-consistent** (no service estimate mixed in) |
| `deadline_slack_if_admitted_now = slo_deadline - now - service_proxy` | `ONLINE_CAUSAL` but **degenerate for absolute gating** — see §5 |
| `priority` / `weight` | `ONLINE_CAUSAL` |
| `fraction_laxity_negative`, `fraction_laxity_near_deadline` | `ONLINE_CAUSAL` |
| eventual `br_*_slo_violated` | `FUTURE_OUTCOME` (label only) |

**C. Fairness debt / starvation risk**

| Variable | Classification |
|---|---|
| `max_class_deficit_ratio = max_c[ demand_c / max(1, active_by_class_c + 1) ]` | `ONLINE_CAUSAL` — computed purely from `state.waiting_queue` / `state.gpu_states[*].active_requests_info`; **structurally identical** to the live scoring term inside `WeightedFairSharePolicy._score` (`deficit = demand / max(1, served_share + 1)`) — WFS is *already* an online, causal, class-deficit-ratio-driven policy, not a black box |
| `longest_waiting_age`, `n_distinct_classes_in_queue` | `ONLINE_CAUSAL` |
| per-request `class_id` | `ONLINE_CAUSAL` (no per-request deficit *value* is attached; only the aggregate max-over-classes is extracted) |

**Metadata / non-deployable** (unchanged from the prior diagnosis):
`canonical_scenario_id`, `split`, `step`, `event_id`, `favlong`/`favshort`
(parsed) are `EXPERIMENT_METADATA`; every `br_*` outcome field is
`FUTURE_OUTCOME`. All features used in the grouped-CV models and the
constrained rule (§8–§9) are `ONLINE_CAUSAL`.

---

## 5. Why `feasible_if_admitted_now` Is Degenerate

`scoring.py::deadline_slack(req, now, service_proxy) = req.slo_deadline -
now - service_proxy`, where `service_proxy = predicted_service_proxy =
alpha*prompt_tokens + beta*predicted_output_tokens` (`alpha=0.5, beta=1.0`).
**The function's own docstring states it directly**: *"service_proxy is in
steps; convert to seconds via step_size if needed. Phase 1 leaves it
unit-less (policies compare slacks relatively)."*

Diagnosis: this is **not** a "too strict" threshold in a graded sense, and
**not** an evaluated-too-late problem. It is **structurally comparing two
incommensurate scales by design**: `slo_deadline` is a small real-valued
time budget (contested-row means 2.5–18.7 across strata) while
`service_proxy` is a raw token-count proxy (contested-row means 228–960),
one to two orders of magnitude larger. Subtracting them and testing `>= 0`
as an *absolute* admissibility bound is close to guaranteed to return
`False` whenever `predicted_output_tokens` exceeds a handful of tokens,
independent of true urgency. `feasible_if_admitted_now` is
**`INHERENTLY_DEGENERATE_FOR_ABSOLUTE_GATING_BY_DESIGN`** — the function is
built (and already used, in `ESTF`/`urgency_score`) for *relative* ranking
between candidates, not as an absolute per-request gate.

**Confirmation via the unit-consistent alternative**: `laxity_own =
slo_deadline - state.time` (no service-proxy term, reconstructed here as
`slo_deadline - (queue_age + arrival_time)`, matching the already-online
aggregate `laxity` feature) is **not** degenerate: only 1.1% of ESTF-only
and 3.3% of WFS-only contested rows have negative `laxity_own` (vs. 100%
negative `deadline_slack_if_admitted_now` for both sides). This confirms the
degeneracy is a units artifact of `deadline_slack`'s design, not a genuine
fact that every contested request is hopeless.

---

## 6–7. Frozen Offline Targets

Both are `FUTURE_OUTCOME`-derived **labels only** — never used as features
of any rule.

- **`completion_benefit_label`** (ESTF-only request, per event): `1` if it
  completes under `br_estf_estf` (its own native branch) **and** does not
  complete under `br_wfs_wfs` (the other policy's own native branch); else
  `0`. Prevalence: **44.0%** overall (48.3% `favlong`, 35.5% `favshort`).
- **`slo_risk_label`** (WFS-only request, per event): `1` if it is **not**
  completed-and-SLO-safe under `br_estf_estf` (choosing ESTF) — either it
  never completes within the bounded window, or it completes but violates
  its SLO; else `0`. Answers "does choosing ESTF over WFS put the
  WFS-favored request at risk." Prevalence: **39.6%** overall, but sharply
  regime-split: **58.3% in `favlong`** vs. **3.2% in `favshort`** — almost a
  clean regime separator on its own, consistent with the entire prior
  diagnostic chain's finding that `favlong` is where WFS's protection
  matters and `favshort` is where it barely does.

---

## 8. Grouped-CV Separability (the central question)

`GroupKFold(5)` by `canonical_scenario_id` (32 groups with events), feature
set = all `ONLINE_CAUSAL` aggregate + per-request causal fields from §4.

**A. Completion-benefit prediction** (n=91, class balance 51/40):

| Model | Balanced acc. (mean±std) | ROC-AUC | Macro F1 |
|---|---:|---:|---:|
| majority baseline | 0.500 ± 0.000 | — | 0.359 |
| logistic regression | **0.740 ± 0.064** | 0.773 ± 0.115 | 0.740 |
| shallow tree (depth 3) | 0.657 ± 0.112 | 0.652 ± 0.095 | 0.658 |
| RF (capacity diagnostic) | 0.719 ± 0.127 | **0.829 ± 0.089** | 0.717 |

**B. SLO-risk prediction** (n=91, class balance 55/36):

| Model | Balanced acc. (mean±std) | ROC-AUC | Macro F1 |
|---|---:|---:|---:|
| majority baseline | 0.500 ± 0.000 | — | 0.377 |
| logistic regression | **0.892 ± 0.036** | **0.941 ± 0.037** | 0.882 |
| shallow tree (depth 3) | 0.864 ± 0.070 | 0.857 ± 0.061 | 0.858 |
| RF (capacity diagnostic) | 0.890 ± 0.039 | 0.977 ± 0.022 | 0.881 |

**Both risks are separately, strongly, and stably predictable from
ONLINE_CAUSAL state** — SLO-risk especially so (ROC-AUC up to 0.977, fold
std ≤ 0.070 across all three fitted models). This is a qualitatively
different result from the prior contested-value diagnosis, where the best
constructed scalar proxy achieved ROC-AUC ≈ 0.10–0.55 and balanced accuracy
at-or-below chance in most strata. Separability is not the weak link in
this formulation.

---

## 9. Constrained-Rule Definition

```
permit ESTF only if out-of-fold predicted P(slo_risk_label=1) <= 0.5
    (grouped-CV logistic regression, no leakage)
within the permitted region, choose ESTF only if
    out-of-fold predicted P(completion_benefit_label=1) > 0.5
else choose WFS
```

**No continuous threshold sweep was performed.** The 0.5 probability
boundary is the only principled, non-invented boundary available: the
natural zero-slack boundary (`deadline_slack_if_admitted_now >= 0`) is
unusable as shown in §5 (0% feasible for every contested row — using it
would collapse the rule to always-WFS), and no other frozen Family-A safety
tolerance applies at this per-decision granularity. Per the task's
instruction, no boundary was invented beyond this.

---

## 10. Constrained-Rule Alignment vs. the Native/Long-Run Signal

Ground truth = sign of `delta_native_whole_branch_raw` (the same
whole-branch raw-completed-count native-continuation signal used throughout
the prior diagnostic chain; **caveat repeated from the prior report**: this
is a documented ESTF-biased surrogate, not true weighted/SLO ANWG).

| Stratum | n | ESTF pred. share | Sign agreement | Balanced acc. (3-class) | False-ESTF rate\* | False-WFS rate\*\* |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 91 | 23.1% | 14.3% | 0.402 | 0.0% | 65.7% |
| favlong | 60 | 13.3% | 11.7% | **0.075** | 0.0% | 76.9% |
| favshort | 31 | 41.9% | 19.4% | 0.485 | 0.0% | 33.3% |

\*of ESTF predictions, fraction where ground truth is WFS. \*\*of WFS
predictions, fraction where ground truth is ESTF.

**Reading this honestly**: agreement with the raw-count ground truth is
weak, and worst exactly in `favlong` (0.075, well below the trivial
always-ESTF baseline's 0.500 on this same metric — see §14). But
`ground-truth WFS share in favlong is 0/60` (raw completion count never
favors WFS there, per the prior diagnostic chain), so **any** rule that
meaningfully shifts toward WFS in `favlong` will score poorly against this
particular yardstick by construction, whether or not that shift is actually
correct in true weighted/SLO terms. `false_estf_rate_of_estf_predictions =
0.0%` in every stratum is notable: whenever the constrained rule *does*
permit ESTF, it is never wrong relative to this ground truth — the rule's
conservatism, not its precision, is what drives the low overall agreement
score. This is the central, unresolved ambiguity of this audit (§17).

---

## 11. Favlong / Favshort Breakdown

| Quantity | favlong | favshort |
|---|---:|---:|
| completion-benefit prevalence | 48.3% | 35.5% |
| SLO-risk prevalence | **58.3%** | **3.2%** |
| constrained-rule ESTF share | **13.3%** | **41.9%** |
| false-ESTF rate (vs. raw-count gt) | 0.0% | 0.0% |
| false-WFS rate (vs. raw-count gt) | 76.9% | 33.3% |

**The desired qualitative shape is present**: SLO-risk prevalence itself is
an almost clean regime separator (58.3% vs. 3.2%), and the constrained
rule's ESTF share tracks it correctly in the intended direction — sharply
suppressed in `favlong`, meaningfully preserved in `favshort`. This is
exactly the behavior the terminal-value redesign *failed* to produce (it
moved ESTF share **up**, not down, in `favlong` at every horizon). Whether
the *magnitude* of favlong suppression (13.3% vs. the raw-count ground
truth's own 78.3% ESTF rate there) is correct cannot be confirmed without
an independent full-scenario ANWG evaluation (§17).

---

## 12. Virtual-Queue / Dual-Variable Feasibility

Classification: **`NATURAL_EXISTING_DEBT_SIGNAL`**

`src/llmserveopt/policies/weighted_fair_share.py::_score` already computes,
online, per real decision: `deficit = demand[cls] / max(1,
served_share[cls] + 1)`, where `demand` comes from `state.waiting_queue` and
`served_share` from `state.gpu_states[*].active_requests_info` —
**WFS's own scoring rule already is a per-class deficit-ratio computation**,
purely from `ONLINE_CAUSAL` state, no future information. The
identical-shape aggregate feature `max_class_deficit_ratio` (max over
classes of the same ratio) is already extracted online in
`family_a_observability_continuation_v1.py` Group C and present in the
existing 91-event artifact. **No new derivation is required** to obtain a
per-class debt/deficit state variable — it already exists and is already
read by a real, running policy.

A future `Z_{t+1} = max(0, Z_t + violation_signal_t - target)` formulation
is semantically plausible: this ratio is already a bounded-below,
queue-state-driven quantity that rises when a class is under-served and
falls as it is admitted — structurally the right shape for a
virtual-queue/Lyapunov debt variable. **No target epsilon is proposed
here** (per task instruction).

Empirical correlation (Spearman, n=91) between the existing aggregate
`max_class_deficit_ratio` and the two offline labels:

| Target | ρ | p |
|---|---:|---:|
| `slo_risk_label` | −0.348 | **0.0007** |
| `completion_benefit_label` | −0.174 | 0.098 |

The correlation with `slo_risk_label` is strong and highly significant,
confirming the deficit signal carries real information about the risk axis
— but its **sign is the opposite of the naive expectation** (higher
aggregate deficit correlates with *lower* SLO risk from choosing ESTF, not
higher). This uses the *aggregate max-over-all-classes* ratio, not the
*specific contested class's own* deficit — a more precise, request-specific
version (feasible from the same online fields, not computed in this
artifact) would be needed before treating this sign as load-bearing; flagged
explicitly as a limitation (§17), not resolved here.

---

## 13. Does WFS Behave Like Constraint Protection?

Yes, on two independent lines of evidence:

1. **Full-scenario** (`family_a_rollout_value_limit_diagnosis_20260820.md`
   §11): in `favlong`, WFS achieves the *best* `priority_weighted_slo_goodput`
   (0.6029) despite the *worst* (highest) `max_latency` (25.81 vs. ESTF's
   23.71) — WFS sacrifices raw latency to protect priority/SLO outcomes.
2. **Contested-request** (`family_a_contested_request_value_diagnosis_20260821.md`
   §7): WFS-only contested requests' SLO-success-given-completed rate is
   78.3% under WFS's own native continuation vs. only 45.5% under ESTF's —
   a +32.8pp gap, never negative (0/60 `favlong` WFS-only requests worse off
   under WFS's own path).

**Quantified two-sidedness**: `completion_benefit_label` prevalence
(ESTF-side value) = 44.0%; `slo_risk_label` prevalence (WFS-side value at
stake) = 39.6% — both substantial and roughly comparable in magnitude. This
is the structural precondition for a constrained formulation to be
meaningful rather than one side trivially dominating: **the tradeoff is
genuinely two-sided**, not a case where one policy is simply better and a
constraint would be vacuous.

---

## 14. Triviality Checks

| Rule | ALL bal. acc. | favlong bal. acc. | favshort bal. acc. |
|---|---:|---:|---:|
| always-WFS | 0.333 | 0.000 | 0.333 |
| always-ESTF / majority | 0.333 | **0.500** | 0.333 |
| priority ≥ 5 → WFS | 0.063 | 0.000 | 0.333 |
| regime-label-equivalent | 0.063 | 0.000 | 0.333 |
| **constrained rule** | **0.402** | 0.075 | **0.485** |

The constrained rule beats every trivial rule on `ALL` and `favshort`, but
is *worse* than always-ESTF specifically in `favlong` (0.075 vs. 0.500) —
consistent with the §10 caveat (always-ESTF wins there only because the
`favlong` ground truth itself is 100% ESTF/TIE, never WFS). **Overlap
checks**: the constrained rule agrees with the priority≥5 rule on 71.4% of
events and with the regime-equivalent rule on 71.4% of events — substantial
but not total overlap; in the ~29% of events where it *disagrees* with
these trivial rules, it captures materially more of the ground-truth signal
(overall balanced accuracy 0.402 vs. 0.063 for either trivial rule alone),
indicating the rule is not simply reconstructing priority or regime
identity through the back door, even though priority and regime correlate
strongly with its predictions (as they must, given §4's finding that
`priority` itself nearly perfectly separates the contested-side populations
by regime).

---

## 15. Literature-Aware Framing

No novelty is claimed for any of the following; framing only, per task
instruction:

- **Constrained MDPs** — the general "maximize objective subject to a risk
  constraint" structure tested here is the standard CMDP framing; this
  audit only asks whether Family-A's specific failure (a single scalar
  blending completion and SLO/fairness) is better modeled this way, not
  whether CMDPs are novel.
- **Primal-dual / Lyapunov / virtual-queue scheduling** — the natural debt
  signal identified in §12 (`max_class_deficit_ratio`, already computed
  online by WFS) is structurally the kind of virtual-queue state such
  methods use; this audit does not design or claim a primal-dual
  controller.
- **Risk-sensitive scheduling / deadline schedulability** — the SLO-risk
  target (§6) is a request-level miss-probability label, in the spirit of
  risk-sensitive admission control; not claimed as new.
- **WFS / VTC-style fairness debt** — WFS's own `_score` deficit term (§4,
  §12) is already a fairness-debt-style computation; any future
  constrained controller compared against WFS is comparing against a
  policy that already implements a related (simpler, un-constrained-by-a-
  completion-objective) mechanism, and must be benchmarked against it, not
  claimed as superseding it without evidence.

The only potential project-specific empirical value, if a future controller
were built and validated: **the Family-A failure may be better modeled as a
completion objective under a dynamic SLO/fairness-risk constraint than as
scalar terminal-value optimization** — this audit finds evidence consistent
with, but not yet dispositive of, that claim.

---

## 16. Classification

**`CONSTRAINED_FORMULATION_PARTIAL_SIGNAL`**

- Completion benefit and SLO risk **are** separately predictable, well
  above trivial baselines, stably across folds (§8) — clears `STRONG`'s
  first bullet decisively, and clears it more convincingly than anything in
  the prior contested-value diagnosis.
- Both ESTF and WFS remain nontrivial under the resulting rule (13–42% ESTF
  share by regime; §11) and no hidden metadata is required (§4) — clears
  two more `STRONG` bullets.
- The rule does **not** collapse to always-WFS, priority-only, or
  regime-identity (§14: 71.4% overlap, not 100%, and materially higher
  accuracy in the disagreement cases) — clears `STRONG`'s collapse bullet.
- **What is not established**: "the constraint-style rule improves
  alignment materially" (`STRONG`'s remaining bullet). Against the only
  available ground truth, alignment is weak in `favlong` specifically — but
  that ground truth is independently documented as ESTF-biased in exactly
  that regime, so this is genuinely ambiguous, not a clean failure. This
  single unresolved gap is what keeps the classification at `PARTIAL`
  rather than `STRONG`, and is a categorically better position than the
  prior contested-scalar-proxy diagnosis (which failed on separability
  itself, not just on the alignment-ground-truth ambiguity).

---

## 17. Exact Next Step

**`KEEP_CONSTRAINED_FORMULATION_DIAGNOSTIC_ONLY`**

The separability result (§8) and the qualitatively-correct regime shape
(§11) are strong enough to be worth keeping as validated diagnostic
knowledge and as the leading candidate direction for a future Family-A
controller redesign — clearly more promising than the scalar formulations
already ruled out. But committing to
`DESIGN_PRIMAL_DUAL_OR_VIRTUAL_QUEUE_CONTROLLER` now would require
resolving the §10 alignment ambiguity first (specifically: an independent
per-event or per-scenario priority/SLO-weighted ANWG ground truth, not the
raw-completion-count proxy used throughout this diagnostic chain), which
this task's scope explicitly excludes (no new simulation). Per the task's
own instruction, this next step is named, not executed.

---

## 18. Limitations

- **Ground-truth mismatch (repeated, central)**: `delta_native_whole_branch_raw`
  is a raw completed-request count, documented elsewhere in this chain as a
  poor surrogate for true priority/SLO-weighted ANWG specifically in
  `favlong`. Section 10's alignment numbers should not be read as a clean
  pass/fail on the constrained formulation without this caveat.
- **Small samples for grouped CV**: n=91 events, 32 groups, 5-fold
  `GroupKFold` — §8's balanced-accuracy standard deviations (0.036–0.127)
  are the honest measure of this; results are stable but not enormous-`n`
  guarantees.
- **`max_class_deficit_ratio` used in §12 is aggregate (max over all
  classes in queue), not the specific contested WFS-only request's own
  class deficit** — the inverted correlation sign there should be treated
  as suggestive, not confirmed, until a request-specific version is
  computed.
- **The constrained rule's thresholds (0.5/0.5) were deliberately not
  swept**, per task instruction; a swept or jointly-optimized threshold
  pair might perform differently in either direction — not tested here.
- **Both offline labels (§6–§7) are derived from the same bounded
  (≤1500-step) branch rollouts used throughout this diagnostic chain**, not
  full-scenario outcomes; they inherit that artifact's own scope limits
  (documented in the prior two reports).
- TRAIN/VAL only; no TEST, public-trace, or real-serving validation.
- This audit tests one specific decomposition (SLO-risk gate then
  completion-benefit optimization); it does not rule out other constrained
  formulations (e.g., a fairness-debt gate instead of an SLO-risk gate, or
  a joint two-constraint version).

---

## 19. Reproducible Commands / Artifacts

Inputs read (all pre-existing, none modified):

- `experiments/family_a_contested_request_value_diagnosis/{contested_events,contested_requests}.csv`
- `experiments/family_a_contested_request_value_diagnosis/contested_events_with_diagnosis_scores.csv`
- `experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv`
- `docs/current/family_a_contested_request_value_diagnosis_20260821.md`
- `docs/current/family_a_terminal_value_v1_analysis_20260820.md`
- `docs/current/family_a_rollout_value_limit_diagnosis_20260820.md`
- `docs/current/family_a_receding_horizon_oracle_v1_analysis_20260820.md`
- `docs/current/family_a_observability_continuation_v1_repaired_analysis_20260820.md`
- `src/llmserveopt/policies/scoring.py`, `src/llmserveopt/policies/weighted_fair_share.py`, `src/llmserveopt/core/types.py`, `src/llmserveopt/analysis/family_a_observability_continuation_v1.py` (read for exact semantics, not modified)

New artifacts created by this audit (deterministic, read-only; no
simulation):

- `scripts/analyze_family_a_constrained_formulation_feasibility.py`
- `experiments/family_a_contested_request_value_diagnosis/constrained_formulation_feasibility_summary.json`
- `experiments/family_a_contested_request_value_diagnosis/constrained_formulation_event_table.csv`
- This report: `docs/current/family_a_constrained_formulation_feasibility_20260821.md`

Reproduce: `python3 scripts/analyze_family_a_constrained_formulation_feasibility.py`
(pure pandas/numpy/scipy/sklearn over existing CSVs; no simulator import, no
RNG beyond fixed `random_state=0` in the diagnostic tree/RF models, no new
scenario execution; wall clock < 10s).
