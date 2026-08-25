# Family-A Observability / Continuation-Dependence Diagnostic v1 - Repaired Analysis

Date: 2026-08-20

Analysis-only pass over the repaired canonical run at
`experiments/family_a_observability_continuation_v1/`. This report does not overwrite
`docs/current/family_a_observability_continuation_v1_analysis_20260820.md`, which remains the
historical record of the invalid pre-repair zero-event run. No experiment was rerun, no TEST data
was read, no simulator/policy/model definition was changed, and no file was staged, committed, or
pushed. This is the single final repaired-analysis report created by this task.

---

## 1. Executive verdict

The repaired run is structurally valid and scientifically interpretable, but it does **not** support
immediate construction of a purely local action selector. It shows:

- a strong native ESTF-over-WFS effect: `Delta_native` mean = `+1.033`, median = `+1.0`, positive in
  `58/91` events;
- a much weaker same-continuation local action effect: `Delta_same` mean = `+0.055`, median = `0`,
  exactly zero in `70/91` events;
- substantial local/native sign mismatch: `42/91` events differ, driven mostly by
  `38/91` events where `Delta_same = 0` but `Delta_native > 0`;
- continuation dependence larger than the isolated local action effect in `52/91` events, with no
  event where the local effect magnitude exceeds continuation dependence.

The result is therefore **`CONTINUATION_DOMINATED`**. Family A still has real native policy value
and observable structure, but the decisive advantage is mostly produced by which parent continues
after the first action, not by the first disputed admission alone. The exact next step is
**`DESIGN_STATEFUL_FAMILY_A_CONTROLLER`**, not a stateless local child.

---

## 2. Integrity summary

Independent checks over the repaired artifacts passed:

| Check | Observed |
|---|---:|
| scenario rows | 64 |
| events rows | 91 |
| summary event sum | 91 |
| scenarios with events | 32 |
| scenarios with zero events | 32 |
| failures | 0 |
| strengthened integrity `ok` | true |
| duplicate scenario IDs | 0 |
| duplicate event join keys `(canonical_scenario_id, step)` | 0 |
| null/NaN critical counterfactual fields | 0 |
| TEST rows in summary/events | 0 / 0 |
| summary split composition | 54 train / 10 val |
| event split composition | 69 train / 22 val |
| ESTF ID | `estimated_service_time_first` |
| WFS ID | `weighted_fair_share` |

All four branch columns are populated for all `91/91` events:
`br_estf_estf_completed`, `br_wfs_wfs_completed`, `br_wfs_estf_completed`,
`br_estf_wfs_completed`. The invalid pre-repair run is preserved separately at
`experiments/family_a_observability_continuation_v1_invalid_pre_snapshot_fix_20260820/`.

Runtime metadata: start `2026-08-20T19:40:06Z`, end `2026-08-20T23:05:09Z`,
wall clock `12303.052783727646` seconds.

---

## 3. Event semantics

A repaired Family-A event is a step in a TRAIN/VAL Family-A scenario where ESTF and WFS propose
different canonical admit sets when both policies are evaluated symmetrically from the identical
true pre-decision `ObservableState`. Each sampled event then runs four bounded counterfactual
branches:

- `br_ESTF_ESTF`: ESTF first action, ESTF continuation.
- `br_WFS_WFS`: WFS first action, WFS continuation.
- `br_WFS_ESTF`: WFS first action, ESTF continuation.
- `br_ESTF_WFS`: ESTF first action, WFS continuation.

The repaired event count (`91`) is not expected to equal the parent study's `3,545` Family-A
disagreements. The repair audit documents that the parent disagreement detector was asymmetric:
it compared the real chosen candidate on a clean state against the alternate candidate after the
real action's admission bookkeeping had already consumed capacity. The repaired diagnostic is
symmetric: ESTF and WFS are both evaluated from the same clean pre-decision baseline. These are
**RELATED_BUT_NOT_IDENTICAL** definitions. The smaller count is therefore a semantic change in the
disagreement predicate, not by itself a loss of Family-A signal.

---

## 4. Delta_same - local action effect

Definition, frozen by the design:

- `Delta_same_commonESTF = U(br_ESTF_ESTF) - U(br_WFS_ESTF)`
- `Delta_same_commonWFS = U(br_ESTF_WFS) - U(br_WFS_WFS)`
- `Delta_same = mean(Delta_same_commonESTF, Delta_same_commonWFS)`

All signs are ESTF minus WFS.

| Metric | common ESTF | common WFS | symmetrized Delta_same |
|---|---:|---:|---:|
| n | 91 | 91 | 91 |
| > 0 | 0 (0.0%) | 17 (18.7%) | 16 (17.6%) |
| < 0 | 2 (2.2%) | 4 (4.4%) | 5 (5.5%) |
| = 0 | 89 (97.8%) | 70 (76.9%) | 70 (76.9%) |
| mean | -0.022 | +0.132 | +0.055 |
| median | 0 | 0 | 0 |
| std | 0.147 | 0.499 | 0.252 |
| p25 / p75 | 0 / 0 | 0 / 0 | 0 / 0 |
| p90 / p95 | 0 / 0 | 1 / 1 | 0.5 / 0.5 |
| min / max | -1 / 0 | -2 / +1 | -1 / +0.5 |
| mean absolute value | 0.022 | 0.242 | 0.121 |

The only preregistered materiality criterion available here is nonzero same-continuation effect.
By that criterion, `21/91` events (`23.1%`) are materially nonzero. No additional practical
significance cutoff is defined, and none is invented here.

Central answer: Family A retains a genuine mixed local action effect under fixed continuation, but
it is sparse and weak. Both parents win some local events (`16` ESTF-favored, `5` WFS-favored), yet
three quarters of repaired events have no isolated first-action effect at all. The effect is also
mostly visible under common WFS continuation; common ESTF continuation is nearly always zero.

---

## 5. Delta_native - native policy effect

Definition: `Delta_native = U(br_ESTF_ESTF) - U(br_WFS_WFS)`.

| Metric | Delta_native |
|---|---:|
| n | 91 |
| > 0 | 58 (63.7%) |
| < 0 | 1 (1.1%) |
| = 0 | 32 (35.2%) |
| mean | +1.033 |
| median | +1.0 |
| std | 1.027 |
| p25 / p75 | 0 / 2 |
| p90 / p95 | 2 / 3 |
| min / max | -1 / +4 |
| mean absolute value | 1.055 |
| nonzero | 59 (64.8%) |

At the scenario level, native sign is mostly one-sided:

- all-zero native scenarios: 3
- positive-only among nonzero native events: 28
- negative-only among nonzero native events: 0
- mixed native nonzero sign: 1
- zero-net native scenarios: 4

This repaired native estimate is directionally consistent with the parent decision-criticality
Family-A signal: prior bounded-branch mean `+0.886` completions/branch, prior approximate 61/39
Family-A router chosen-policy split, and the stored TEST-side `+0.0302` ANWG value. The TEST-side
number is cited only as already completed corroboration; it was not read or used for tuning here.

---

## 6. Local vs native sign agreement

Sign buckets treat zero as its own class.

| Relation | Count | Fraction |
|---|---:|---:|
| `sign(Delta_same) == sign(Delta_native)` | 49 | 53.8% |
| signs differ | 42 | 46.2% |
| `Delta_same = 0`, `Delta_native != 0` | 38 | 41.8% |
| `Delta_native = 0`, `Delta_same != 0` | 0 | 0.0% |
| both zero | 32 | 35.2% |
| nonzero sign reversal | 4 | 4.4% |

The important asymmetry is that native value often appears when the same-continuation first-action
effect is zero. There are no events where the native effect is zero but the local effect is
nonzero. This is direct evidence that native continuation is carrying much of the Family-A value.

---

## 7. Continuation dependence

Definition, frozen by the design: `C = Delta_native - Delta_same`.

| Metric | C | abs(C) |
|---|---:|---:|
| n | 91 | 91 |
| mean | +0.978 | 0.989 |
| median | +1.0 | 1.0 |
| p75 | 1.75 | 1.75 |
| p90 | 2.5 | 2.5 |
| p95 | 2.5 | 2.5 |
| min / max | -0.5 / +4.0 | 0 / 4.0 |

Additional dependence checks:

- continuation effect larger than local action effect: `52/91` (`57.1%`)
- local action effect larger than continuation effect: `0/91` (`0.0%`)
- equal magnitude: `39/91` (`42.9%`)
- continuation reverses the preferred parent: `4/91` (`4.4%`)
- continuation amplifies a same-direction local effect: `17/91` (`18.7%`)

Descriptive sub-result: **CONTINUATION_DOMINATED**. Continuation is not merely a small adjustment to
a local first-action signal; it is usually the dominant component of the native advantage.

---

## 8. Magnitude and concentration

Absolute local-action effect distribution:

| Metric | abs(Delta_same) |
|---|---:|
| mean | 0.121 |
| median | 0 |
| p75 | 0 |
| p90 | 0.5 |
| p95 | 0.5 |
| max | 1.0 |

There is no preregistered tiny/moderate/large threshold, so this report does not bin effects by an
invented practical-significance cutoff. Descriptively, the local effect is small in most events:
`70/91` are exactly zero and the largest symmetrized local effect magnitude is `1.0` completion.

Concentration:

- `10` events account for 50% of total `abs(Delta_same)`.
- top 10% of events (`10` events) account for 50.0% of total `abs(Delta_same)`.
- `17` scenarios have nonzero total `abs(Delta_same)`.
- top 1 scenario accounts for 13.6% of total `abs(Delta_same)`.
- top 5 scenarios account for 45.5% of total `abs(Delta_same)`.
- top 10% of event-bearing scenarios (`4` scenarios) account for 40.9%.

The local signal is not confined to a single scenario, but it is sparse and moderately concentrated.

---

## 9. Scenario heterogeneity

Among the 32 scenarios with events:

- events per scenario: mean `2.844`, median `3`, min `1`, max `3`
- scenarios with 3 events: 29
- scenarios with 2 events: 1
- scenarios with 1 event: 2
- local all-zero scenarios: 15
- local positive-only among nonzero events: 13
- local negative-only among nonzero events: 4
- local mixed nonzero sign within a scenario: 0
- local zero-net scenarios: 15

Parameter-level summaries:

| Parameter group | scenarios | scenarios with events | events | local mean | local mean abs | native mean |
|---|---:|---:|---:|---:|---:|---:|
| utilization 1.1 | 22 | 9 | 27 | +0.056 | 0.093 | +1.185 |
| utilization 1.3 | 18 | 8 | 22 | +0.114 | 0.114 | +0.818 |
| utilization 1.5 | 24 | 15 | 42 | +0.024 | 0.143 | +1.048 |
| skew 1.0 | 20 | 0 | 0 | n/a | n/a | n/a |
| skew 5.0 | 24 | 20 | 60 | +0.050 | 0.083 | +0.833 |
| skew 10.0 | 20 | 12 | 31 | +0.065 | 0.194 | +1.419 |
| favlong | 32 | 20 | 60 | +0.050 | 0.133 | +1.400 |
| favshort | 32 | 12 | 31 | +0.065 | 0.097 | +0.323 |
| noise 0.0 | 34 | 18 | 51 | +0.010 | 0.108 | +0.922 |
| noise 0.3 | 30 | 14 | 40 | +0.113 | 0.138 | +1.175 |

Interpretation: repaired symmetric disagreements require nontrivial skew (`skew=1.0` has no
events). Native value is much stronger in `favlong` and high-skew cases. Local sign variation is
mostly across scenarios, not within a scenario; no event-bearing scenario contains both positive
and negative nonzero local events.

---

## 10. Observable feature set

Features actually present in the repaired event CSV, grouped as the design intended:

- **Workload snapshot**: `step`, `queue_length`, `active_count`, `completed_count`, `n_gpus`.
- **Request/service distributions**: `queue_age_p10`, `queue_age_p50`, `queue_age_p90`,
  `queue_age_mean`, `predicted_output_tokens_p10`, `predicted_output_tokens_p50`,
  `predicted_output_tokens_p90`, `predicted_output_tokens_mean`, `prompt_tokens_p10`,
  `prompt_tokens_p50`, `prompt_tokens_p90`, `prompt_tokens_mean`, `est_service_time_p10`,
  `est_service_time_p50`, `est_service_time_p90`, `est_service_time_mean`.
- **Fairness/starvation**: `max_class_deficit_ratio`, `longest_waiting_age`,
  `n_distinct_classes_in_queue`.
- **Urgency/slack**: `laxity_p10`, `laxity_p50`, `laxity_p90`, `laxity_mean`,
  `fraction_laxity_negative`, `fraction_laxity_near_deadline`.
- **KV/resource**: `mean_kv_utilization`, `max_kv_utilization`, `free_kv_capacity`,
  `prefilling_count`, `decoding_count`.
- **Pair-specific ranking geometry**: `n_admit_estf`, `n_admit_wfs`,
  `admit_symmetric_diff_size`, `is_shallow_disagreement`, `pair_rank_spearman_topk`,
  `pair_topk_n`.
- **Causal history**: `history_queue_len_slope`, `history_kv_util_slope`,
  `history_admitted_count_slope`, `history_window_truncated`.

Explicitly not present as model features:

- raw per-class queue/active count vectors;
- raw per-request ranked lists or request IDs;
- `actual_output_tokens`;
- post-disagreement future state;
- counterfactual fork future state;
- TEST rows;
- router Stage-2 score/probability;
- scenario parameters as explicit columns. Utilization/skew/favlong/noise are encoded only in
  `canonical_scenario_id` and were used for heterogeneity summaries, not grouped-CV feature models.

`router_chosen_policy_id` is present for context but was not used as a feature, per the design.

---

## 11. Grouped prediction of sign(Delta_same)

Grouped 5-fold CV used `canonical_scenario_id` as the group key. Labels are `-1`, `0`, `+1`.
Some folds contain only two of the three classes; ROC-AUC is reported only where valid under the
fold's class support.

| Model | Balanced accuracy | ROC-AUC | Macro F1 | Confusion matrix labels [-1,0,+1] |
|---|---:|---:|---:|---|
| majority baseline | 0.433 +/- 0.091 | 0.500 +/- 0.000 | 0.289 +/- 0.024 | `[[0,5,0],[0,70,0],[0,16,0]]` |
| shallow tree depth 3 | 0.457 +/- 0.114 | 0.532 +/- 0.166 | 0.351 +/- 0.064 | `[[0,5,0],[7,49,14],[0,8,8]]` |
| logistic regression | **0.531 +/- 0.177** | 0.691 +/- 0.118 | 0.398 +/- 0.107 | `[[1,4,0],[8,49,13],[0,6,10]]` |
| RF capacity diagnostic | 0.514 +/- 0.141 | **0.713 +/- 0.174** | **0.402 +/- 0.069** | `[[0,5,0],[3,63,4],[0,10,6]]` |

Best primary model for local sign: logistic regression, by balanced accuracy. The result clears the
majority baseline only modestly and with high fold dispersion. This is not a robust
`LOCAL_ACTION_OBSERVABLE` result.

---

## 12. Grouped prediction of sign(Delta_native)

Same grouped CV and feature set.

| Model | Balanced accuracy | ROC-AUC | Macro F1 | Confusion matrix labels [-1,0,+1] |
|---|---:|---:|---:|---|
| majority baseline | 0.467 +/- 0.075 | 0.500 +/- 0.000 | 0.259 +/- 0.009 | `[[0,0,1],[0,0,32],[0,0,58]]` |
| shallow tree depth 3 | **0.726 +/- 0.126** | **0.774 +/- 0.078** | **0.508 +/- 0.082** | `[[0,0,1],[0,27,5],[1,15,42]]` |
| logistic regression | 0.615 +/- 0.199 | 0.765 +/- 0.117 | 0.440 +/- 0.120 | `[[0,1,0],[4,20,8],[3,15,40]]` |
| RF capacity diagnostic | 0.672 +/- 0.145 | 0.773 +/- 0.107 | 0.470 +/- 0.093 | `[[0,1,0],[0,21,11],[0,13,45]]` |

Native-continuation sign is easier to predict than local same-continuation sign. Continuation makes
the observed advantage more structured in this feature set, but also less locally synthesizable as
a one-step rule.

---

## 13. Feature ablations

Logistic regression was used as the fixed ablation model, with the same grouped CV. Values are
fold mean +/- fold standard deviation.

### sign(Delta_same)

| Feature group | Balanced accuracy | ROC-AUC | Macro F1 |
|---|---:|---:|---:|
| A minimal previous state | 0.301 +/- 0.180 | 0.567 +/- 0.166 | 0.204 +/- 0.082 |
| C fairness/starvation | 0.281 +/- 0.216 | 0.488 +/- 0.159 | 0.130 +/- 0.081 |
| D service/request distribution | 0.474 +/- 0.209 | 0.634 +/- 0.211 | 0.362 +/- 0.131 |
| E urgency/slack | 0.408 +/- 0.130 | 0.581 +/- 0.139 | 0.283 +/- 0.132 |
| F KV/resource | 0.233 +/- 0.224 | 0.500 +/- 0.000 | 0.034 +/- 0.033 |
| G pair/rank geometry | 0.233 +/- 0.224 | 0.500 +/- 0.000 | 0.034 +/- 0.033 |
| H short causal history | 0.370 +/- 0.090 | 0.642 +/- 0.220 | 0.283 +/- 0.072 |
| I combined state | **0.531 +/- 0.177** | **0.691 +/- 0.118** | **0.398 +/- 0.107** |

For local sign, richer state improves beyond minimal state, with the strongest single group being
service/request distribution. Fairness-debt features alone are weak; KV/resource and rank geometry
are uninformative in this Family-A repaired sample, largely because the relevant columns are
constant or near-constant at these single-admission disagreements.

### sign(Delta_native)

| Feature group | Balanced accuracy | ROC-AUC | Macro F1 |
|---|---:|---:|---:|
| A minimal previous state | 0.527 +/- 0.142 | 0.658 +/- 0.177 | 0.387 +/- 0.133 |
| C fairness/starvation | 0.552 +/- 0.151 | 0.628 +/- 0.172 | 0.410 +/- 0.126 |
| D service/request distribution | 0.597 +/- 0.177 | 0.718 +/- 0.165 | 0.426 +/- 0.104 |
| E urgency/slack | 0.467 +/- 0.132 | 0.575 +/- 0.084 | 0.350 +/- 0.103 |
| F KV/resource | 0.067 +/- 0.149 | 0.500 +/- 0.000 | 0.029 +/- 0.065 |
| G pair/rank geometry | 0.067 +/- 0.149 | 0.500 +/- 0.000 | 0.029 +/- 0.065 |
| H short causal history | 0.219 +/- 0.147 | 0.519 +/- 0.147 | 0.170 +/- 0.126 |
| I combined state | **0.615 +/- 0.199** | **0.765 +/- 0.117** | **0.440 +/- 0.120** |

Native sign is again most helped by service/request distribution plus combined state. Fairness
features matter modestly but do not dominate. History alone is weak.

---

## 14. History contribution

Snapshot-only means all non-history causal features; snapshot+history adds the three history slopes
and truncation flag.

| Target | Snapshot-only bal / AUC / F1 | + history bal / AUC / F1 | Delta |
|---|---|---|---|
| sign(Delta_same) | 0.488 / 0.656 / 0.374 | 0.531 / 0.691 / 0.398 | +0.043 / +0.035 / +0.024 |
| sign(Delta_native) | 0.629 / 0.772 / 0.448 | 0.615 / 0.765 / 0.440 | -0.014 / -0.007 / -0.008 |

History helps local sign slightly, but not consistently enough to call the mechanism strongly
stateful from history alone. For native sign, adding history slightly hurts the grouped mean. The
evidence points more to continuation-policy dependence than to missing short-history slopes.

---

## 15. Interpretable mechanism signal

The shallow trees and logistic coefficients consistently put service/request distribution ahead of
fairness debt, KV state, or pair-rank geometry.

Most stable associations for ESTF-favored local sign (`Delta_same > 0`), from grouped logistic
coefficient stability and supported by the tree:

| Feature | Direction for ESTF-favored local sign | Stability | Scheduling interpretation |
|---|---|---:|---|
| `predicted_output_tokens_p50` | positive | 5/5 folds | larger median predicted output in queue makes ESTF first action more locally favorable |
| `predicted_output_tokens_p10` | negative | 5/5 folds | when even the short-output tail is larger, the local ESTF advantage weakens |
| `est_service_time_p90` | negative | 5/5 folds | heavy high-end service tail tends to erase or reverse isolated first-action value |
| `prompt_tokens_p90` | negative | 5/5 folds | large prompt tail weakens ESTF local sign |
| `laxity_p10` | negative | 5/5 folds | more urgent lower-tail slack aligns against local ESTF advantage |
| `laxity_p90` | positive | 5/5 folds | less urgent upper-tail slack aligns with ESTF local advantage |
| `history_queue_len_slope` | negative | 5/5 folds | growing/less-declining queues do not reliably help local ESTF action |
| `queue_length` | positive | 5/5 folds | queue pressure modestly aligns with ESTF local advantage |
| `max_class_deficit_ratio` | positive | 5/5 folds | fairness debt has a signal, but weaker than service/request distribution |

Shallow tree for local sign uses splits on `predicted_output_tokens_p50`,
`prompt_tokens_p10`, `predicted_output_tokens_p90`, `laxity_mean`, `queue_age_p90`, and `step`.
The native-sign tree is even more service-size dominated, splitting first on
`predicted_output_tokens_p10`, then `prompt_tokens_p10`, `predicted_output_tokens_mean`,
`predicted_output_tokens_p90`, and `predicted_output_tokens_p50`.

These associations do **not** prove causality. They are descriptive, grouped-CV mechanism signals
over 91 repaired events. The strongest interpretable signal is service-size and slack structure,
with fairness debt secondary and KV/rank-geometry essentially unhelpful in this data.

---

## 16. Literature-aware interpretation

Using the already completed literature context, this mechanism has novelty risk:

- FSP threatens generic fair-SRPT claims.
- T-SRPT threatens simple threshold switching.
- VTC remains a must-compare fairness baseline.
- vLLM-LTR is highly relevant because service-time/output-token features dominate the repaired
  mechanism signal.
- PARS remains a strong secondary ranking baseline.

The repaired evidence does not reduce to a pure queue-population threshold switch; minimal
snapshot state is much weaker than service/request distributions. It also does not look like pure
virtual-time/deficit scheduling, because fairness-debt features are not the dominant ablation.
The closest risk is **fairness-debt-adjusted service cost / aging-augmented SJF**. Any child
designed from this result must be compared directly against VTC-style fairness scheduling,
vLLM-LTR/service-time baselines, and PARS-like ranking baselines before claiming novelty.

---

## 17. Exact classification

**`CONTINUATION_DOMINATED`**

Rationale:

- Native advantage exists and is strong (`Delta_native` mean `+1.033`, nonzero in `64.8%`).
- Same-continuation local effect is sparse and mostly zero (`Delta_same = 0` in `76.9%`).
- Continuation dependence is large (`mean abs(C) = 0.989`) and exceeds local effect magnitude in
  `57.1%` of events.
- No event has local action magnitude larger than continuation dependence.
- Native sign is substantially easier to predict than local sign.

This is not `LOCAL_ACTION_OBSERVABLE`; local grouped prediction is modest and high-variance. It is
not `NO_ROBUST_LOCAL_SYNTHESIS_SIGNAL`, because native Family-A value remains strong and local
nonzero effects exist in both directions. `PARTIALLY_OBSERVABLE` is too weak a label for the main
finding: the dominant source of value is continuation, not merely incomplete observability.

---

## 18. Exact next step

**`DESIGN_STATEFUL_FAMILY_A_CONTROLLER`**

Do not execute it in this task. The design target should not be a one-shot local action child.
It should be a Family-A controller that decides when to enter/continue an ESTF-like or WFS-like
mode over a continuation interval, using service/request distribution, slack, queue pressure, and
secondary fairness-debt state. History slopes may be optional, not central, because their grouped
gain is small and inconsistent.

Portfolio-expansion readiness: sufficient to proceed toward designing a candidate child and then
held-out evaluation, but not sufficient to claim portfolio value. The future child is scientifically
valuable only if it has positive marginal contribution beyond the existing portfolio envelope:

`MG_child(x;P) = max(R_child(x), E_P(x)) - E_P(x)`.

That remains untested.

---

## 19. Novelty-chain status

| Chain link | Status | Basis |
|---|---|---|
| 1. policy disagreement | SUPPORTED | repaired symmetric detector found 91 events across 32 scenarios |
| 2. downstream consequence | SUPPORTED | native bounded-rollout effect is positive and strong |
| 3. local-action vs continuation separation | SUPPORTED | repaired four-branch diagnostic separates `Delta_same`, `Delta_native`, and `C` |
| 4. observable mechanism attribution | PARTIAL | native sign predictable; local sign only modestly predictable; associations are not causal proof |
| 5. standalone interpretable scheduler | NOT_YET_DONE | no child designed here |
| 6. positive marginal portfolio contribution | NOT_YET_DONE | no held-out portfolio-envelope test yet |
| 7. public-trace generalization | NOT_YET_DONE | not evaluated |
| 8. real-serving validation | NOT_YET_DONE | not evaluated |

---

## 20. Limitations

- Only 91 repaired events.
- Only 32/64 scenarios have repaired symmetric events.
- TRAIN/VAL only; no TEST data was read here.
- Grouped CV has high fold dispersion, and some folds lack all three sign classes.
- Repaired event semantics differ from the parent study's asymmetric disagreement semantics.
- Counterfactual branches are bounded to 1500 extra steps.
- Same-continuation effects depend heavily on which common continuation is used.
- The scenario family is finite and synthetic.
- Some relevant state may be missing, especially raw class vectors and raw ranked request lists.
- Local WFS-favored events are rare (`5/91`), limiting confidence in sign-flip regime claims.
- No public-trace or real-serving validation exists yet for any child derived from this result.

---

## 21. Artifact paths and reproducible analysis commands

Inputs read:

- `docs/design/FAMILY_A_OBSERVABILITY_CONTINUATION_DIAGNOSTIC_V1.md`
- `docs/current/decision_criticality_timescale_trainval_v1_analysis_20260820.md`
- `docs/current/family_a_observability_continuation_v1_analysis_20260820.md`
- `docs/current/family_a_observability_continuation_v1_repair_audit_20260820.md`
- `experiments/family_a_observability_continuation_v1/family_a_observability_continuation_integrity_report.json`
- `experiments/family_a_observability_continuation_v1/family_a_observability_continuation_v1_results.json`
- `experiments/family_a_observability_continuation_v1/family_a_scenario_summaries.csv`
- `experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv`

Read-only command patterns used:

```bash
jq . experiments/family_a_observability_continuation_v1/family_a_observability_continuation_integrity_report.json
jq . experiments/family_a_observability_continuation_v1/family_a_observability_continuation_v1_results.json

python3 - <<'PY'
import pandas as pd
base = 'experiments/family_a_observability_continuation_v1'
events = pd.read_csv(f'{base}/family_a_observability_continuation_events.csv')
summary = pd.read_csv(f'{base}/family_a_scenario_summaries.csv')
print(len(events), len(summary), summary.n_events.sum())
print(events[['delta_same','delta_native','continuation_dependence']].describe())
PY

python3 - <<'PY'
# Grouped CV analysis: GroupKFold by canonical_scenario_id; models:
# majority baseline, DecisionTreeClassifier(max_depth=3), LogisticRegression,
# RandomForestClassifier capacity diagnostic.
# Features exclude canonical_scenario_id, split, router_chosen_policy_id, labels,
# branch outcomes, and deltas.
PY
```

No command launched an experiment, touched TEST data, modified simulator/policy code, or staged/
committed/pushed anything.
