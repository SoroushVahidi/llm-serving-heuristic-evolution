# Faithful-baseline scope audit (Selector v2)

Follow-up to `docs/selector_v2_slo_calibrated_frontier_search.md`, whose
910-window SLO-calibrated search cleared 8 of 9 Dataset v2 quality gates
but found zero moderate/strong ANWG wins for any of the three faithful
baselines (`vllm_faithful`, `sarathi_faithful`,
`vllm_chunked_prefill_faithful`). This audit determines whether that is
a genuine result or an artifact, across seven angles, and makes an
explicit scope decision.

## 1. Reproduction / integrity check

Loaded `slo_calibrated_windows.csv` (910 rows) and
`slo_calibrated_discriminativeness.csv` (5,460 = 910 × 6 objectives
rows) directly — counts match the prior task's summary exactly. On the
primary objective (`arrival_normalized_weighted_goodput`):

- `vllm_faithful` is `second_best_policy` in exactly 513 rows —
  identical to its own 513 `best_policy` wins. `sarathi_faithful` is
  `second_best_policy` in exactly 513 rows too. **These are the same 513
  windows**: `vllm_faithful` and `sarathi_faithful` are tied at the top
  in every one (both use `decode_first=True`, so they're expected to be
  near-identical per the contention-frontier-search task's derivation),
  and `vllm_faithful` wins the tie-break. All 513 are `NEAR_TIE` or
  `ALL_COMPLETE_OR_EFFECTIVELY_TIED` — not one is in the 135
  `STRONGLY_DISCRIMINATIVE` set. In the 135 `STRONGLY_DISCRIMINATIVE`
  windows specifically, `vllm_faithful` is `second_best_policy` in 67 of
  them (49.6%) but the margin is real and large (e.g. window 3:
  0.262 vs. 0.048; window 23: 0.467 vs. 0.133) — not a rounding or tie-
  break artifact. **Zero of the 135 strong-discriminative windows have
  any faithful policy inside the tie set.** Confirmed: not a tie-
  breaking artifact.

## 2. Execution-health audit

Re-ran the exact 910 windows (bit-identical reconstruction: same seed,
same family-cycling order, same real-trace loading) with EVERY exception
explicitly captured rather than silently swallowed:

```
vllm_faithful:                 910/910 ok, 0 errors
sarathi_faithful:               910/910 ok, 0 errors
vllm_chunked_prefill_faithful:  910/910 ok, 0 errors
(all 8 historical policies):    910/910 ok, 0 errors
```

**Zero exceptions, zero missing rows, zero silent fallbacks, for every
policy on every window.** Rules out registration omission, candidate-
evaluation omission, and hidden-exception-as-zero-score.

## 3. Loss decomposition (metric-level, not heuristic labels)

Mean values across the 151 windows `vllm_faithful` participated in as a
non-winner (winner = that window's actual best policy):

| Metric | Winner mean | vllm_faithful mean | Δ |
|---|---|---|---|
| `completion_fraction` | 0.853 | **0.943** | vllm_faithful completes *more* |
| `rejection_fraction` | 0.147 | 0.055 | winner rejects *more* |
| `slo_attainment` (of completed) | **0.911** | 0.466 | vllm_faithful's completions miss SLO more than half the time |
| `mean_latency` | 0.366 | 0.597 | vllm_faithful slower |
| `mean_ttft` | 0.261 | 0.495 | vllm_faithful ~2x slower TTFT |
| `mean_tpot` | 0.0009 | 0.0009 | **identical** |
| `request_throughput` | 42.8 | **172.9** | vllm_faithful ~4x higher raw throughput |
| ANWG | **0.748** | 0.383 | the actual objective |

**The mechanism, precisely**: `vllm_faithful` (FCFS admission, faithful
to real vLLM v0.1.0) admits and eventually completes almost everyone
(94.3% completion, ~4x the raw throughput of the winners) — but under a
calibrated, tight SLO, serving strictly in arrival order means *every*
admitted request's latency degrades roughly together, so barely half of
its completions (46.6%) land inside their deadline. The winners —
whether via voluntary rejection (drop the already-hopeless up front) or
pure reordering (serve the more time-critical one first) — complete
*fewer* total requests but get 91.1% of those inside deadline.
`arrival_normalized_weighted_goodput = completion_fraction ×
conditional_attainment` (approximately): `0.943 × 0.466 ≈ 0.439` vs.
`0.853 × 0.911 ≈ 0.777`, matching the observed 0.383/0.748 closely. TPOT
being *identical* is the clean confirming signal: the two policies'
decode-step cost is the same (governed by `ServiceModel.step_size`
either way) — the entire gap is in *admission/scheduling order*, exactly
as the mechanism above predicts, not in any per-token execution
difference.

This is a textbook FCFS-under-overload result from queueing theory
(strict arrival-order service degrades aggregate deadline-attainment
under load relative to priority/deadline-aware service), faithfully
reproduced here because `vllm_faithful`/`sarathi_faithful`/
`vllm_chunked_prefill_faithful` are, by design, faithful re-implementations
of real systems that use exactly this FCFS continuous-batching strategy.

## 4. Pairwise specialization search (450 windows, real default admission params)

**Methodological finding, corrected in this task**: every prior task in
this thread constructed the three faithful policies with
`max_num_batched_tokens=ADMIT_CHUNK=100_000` — a deliberate override from
the *original* contention-validation-pilot task ("decouple policy-level
admission chunking from execution-level contention"), silently inherited
by every subsequent search including the 910-window one. This masked
`vllm_chunked_prefill_faithful`'s real distinguishing feature: real
default `max_num_batched_tokens=512` (chunked admission, can eventually
admit any prompt length) vs. `vllm_faithful`'s real default `2560`
(all-or-nothing — a >2560-token prompt can *never* be admitted at all).
This is exactly the mechanism `docs/runtime_validation_benchmark_pack.md`'s
`long_context/xlong_context_burst16` fixture already documents: real
`vllm_faithful` completes 0/16 long-context requests, chunked-prefill
completes all 16.

Ran a targeted, bounded search using each policy's REAL default
parameters (no override), 150 windows per target, generated to favor
each faithful policy's theoretical strength (not fit to hardware
labels):

| Target | Scenario shape | Win fraction | Strong/moderate wins |
|---|---|---|---|
| `vllm_chunked_prefill_faithful` | long-context burst (4-16 reqs, 3,000-12,000 tokens, exceeds 2,560 whole-prompt budget) | 71.3% (107/150) | **0** |
| `sarathi_faithful` | active-decode cohort + arriving prefill overlap | 100% (150/150) | **0** |
| `vllm_faithful` | low-heterogeneity, similar-size/urgency, FCFS-friendly | 100% (150/150) | **0** |

**Every single window in all 450, across all three targeted-favorable
shapes, classified `NEAR_TIE` — zero `STRONGLY_/MODERATELY_
DISCRIMINATIVE` windows for any target.** The high raw win fractions
(71-100%) are trivial wins in a regime where the rivals (weighted_
shortest_processing/edf/scorpio_style_slo_guard/admission_control) ALSO
degenerate toward FCFS-equivalent behavior (nothing to reorder in a
low-heterogeneity or already-admission-friendly window) — not a genuine
faithful-specific advantage. Even hand-picking favorable conditions and
fixing the ADMIT_CHUNK masking does not produce a real specialization
region.

## 5. Objective decomposition

Non-trivial (non-`ALL_EQUIVALENT`) win distributions across all 910
windows, by objective:

| Objective | vllm_faithful wins | vllm_faithful rank |
|---|---|---|
| **`weighted_goodput`** (legacy, completed-only denominator) | **388/776 (50.0%)** | **#1, most frequent winner** |
| **`request_throughput`** | **328/681 (48.2%)** | **#1, most frequent winner** |
| **`slo_success_throughput`** | **258/708 (36.4%)** | **#1, most frequent winner** |
| `p95_latency` | 173/679 (25.5%) | #2 |
| `slo_attainment` | 90/477 (18.9%) | #3 |
| `arrival_normalized_weighted_goodput` (primary) | 0 strong/moderate | last |

**`vllm_faithful` has real, substantial, non-suppressed specialization
on 3 of 5 non-primary objectives — it is the single most frequent winner
on legacy `weighted_goodput`, raw throughput, and `slo_success_
throughput`.** This is the *same* completed-volume-vs.-attainment trade-
off section 3 quantified: `weighted_goodput`'s denominator is completed
requests only, so a policy that completes far more requests (even with a
worse per-completion attainment rate) can still look good on it. This
is not a new problem — it is the exact metric flaw this project's own
history (Phase 2B.14/2B.15, per prior memory) already identified and
fixed by promoting arrival-normalized ANWG to the primary objective.
**ANWG is correctly, not spuriously, suppressing this specialization —
recommending a change back to a completed-only-denominator objective
would reintroduce an already-diagnosed and already-fixed flaw.** No
change to the primary objective is justified here.

## 6. Admission-control fairness audit

Verified by source inspection which policies have **voluntary,
laxity-based rejection** (skip a request that's already SLO-hopeless,
permanently, via a static laxity filter — `admission_control.py`,
`scorpio_style_slo_guard.py`) vs. **pure reordering** (sort the waiting
queue differently but admit greedily whenever resources allow, exactly
like the faithful policies' own semantics — `fifo`, `edf`, `weighted_
shortest_processing`, `estimated_service_time_first`, `best_fit`,
`multi_bin_batching`):

- Of the 151 ANWG-discriminative windows, the winner was a
  **rejection-capable** policy in 73 (48.3%) and a **pure-reordering**
  policy in 78 (51.7%).
- **Restricted comparison** (`admission_control`/`scorpio_style_slo_guard`
  excluded entirely, 9 remaining policies including all 3 faithful):
  91 of the 151 windows are STILL genuinely discriminative
  (73 strong + 18 moderate). `edf` alone wins 61/151 (40.4%, all
  discriminative); `weighted_shortest_processing` 23, `multi_bin_
  batching` 13. **`vllm_faithful`'s 48 restricted "wins" are ALL in the
  trivial `ALL_COMPLETE_OR_EFFECTIVELY_TIED` bucket — zero
  strong/moderate. `sarathi_faithful` and `vllm_chunked_prefill_
  faithful` win zero windows in the restricted comparison, at any
  margin.**

**Conclusion: admission-control/rejection capability explains roughly
half of the original 151-window result, not the dominant cause.** Even a
strictly apples-to-apples, scheduling-order-only comparison (identical
"must eventually serve everyone" semantics as the faithful policies)
reproduces the same qualitative outcome. This rules OUT "the comparison
is structurally unfair because of admission-control asymmetry" as the
primary explanation, while confirming it as a real, secondary, and
legitimate contributor (SCORPIO/admission_control are valid alternative
monolithic scheduling+admission strategies, not a different topology
class — see `docs/external_baseline_integration.md`'s Protocol C, which
already treats them as ordinary monolithic candidates).

## 7. Resource/normalization audit

- **GPU count**: 1 GPU in every window, every policy — no asymmetry.
- **KV capacity / token budget / service model**: identical
  `max_kv_tokens`, `step_token_budget`, `max_prefill_chunk_tokens` per
  window across all 11 policies by construction
  (`_service_model_for_policy` only varies `decode_first`, `True` for
  10 of 11 policies, `False` only for `vllm_chunked_prefill_faithful`).
- **`max_active_sequences` (`GPUConfig`, binding cap)**: 64 in every
  synthetic family (A-E), 4-16 in every real-trace-stress spec (F) —
  always shared identically across all 11 policies in a given window.
- **`max_num_seqs` (policy-level default, potential asymmetry found)**:
  `sarathi_faithful`'s real default is 128; `vllm_faithful`/`vllm_
  chunked_prefill_faithful`'s is 256. **Flagged asymmetry, but
  confirmed inert**: `max_num_seqs_effective = min(policy's max_num_seqs,
  gpu.max_active_sequences)`, and `max_active_sequences` (4-64) is
  always far below both 128 and 256 in every window generated across
  this entire thread — the asymmetry never actually binds.

**No live resource asymmetry found.** The one latent asymmetry
(`max_num_seqs` default) never manifested because the shared
`GPUConfig.max_active_sequences` cap was always the tighter, binding
constraint in every window this project has generated so far — worth
keeping in mind if a future search ever uses `max_active_sequences` >128.

## 8. Decision test

Weighing all seven angles:

- **A (genuine specialization)**: not supported as stated — zero
  strong/moderate faithful wins survive across 910 + 151 (restricted) +
  450 (targeted-favorable) = 1,511 total window-evaluations.
- **B (admission-policy semantics)**: real but partial (~48%), not
  sufficient alone to explain the result (restricted comparison
  reproduces it without any rejection-capable policy present).
- **C (candidate-policy integration)**: ruled out — 100% clean execution
  health, zero errors, zero missing rows.
- **D (scenario coverage)**: largely ruled out — corrected the ADMIT_CHUNK
  masking (a genuine, now-fixed methodological gap) and ran a targeted,
  favorable-condition search; still zero non-trivial faithful wins.
- **E (objective/SLO construction)**: real but already correctly
  resolved by this project's own prior history — `vllm_faithful` *does*
  have real specialization on completed-volume metrics, which ANWG
  correctly discounts because it is arrival-normalized and attainment-
  conditional by design, not a bug.
- **F (service-model/execution asymmetry)**: the `decode_first=True`
  near-identity between `vllm_faithful`/`sarathi_faithful` is real
  (established two tasks ago) but only explains why THEY resemble each
  other, not why both lose to historical policies — the actual driver
  (FCFS vs. deadline/priority-aware reordering under tight SLO) is an
  accurate, faithful reflection of how these systems really behave, not
  a simulator artifact.
- **G**: no other implementation/evaluation issue found.

### `SELECTOR_SCOPE_DECISION = OPTION B`

Faithful policies are genuinely dominated under this monolithic ANWG
selector problem — confirmed by clean execution health, a mechanistic
metric-level decomposition matching textbook FCFS-under-overload theory,
a restricted apples-to-apples comparison that reproduces the result
without any admission-control asymmetry, and a targeted, favorable-
condition pairwise search (with a genuine methodological fix applied)
that still finds nothing. They remain scientifically important as
external baselines precisely because they faithfully replicate real,
deployed serving systems — their real-world importance does not depend
on winning a synthetic ANWG selection game, and `docs/external_baseline_
integration.md`'s Protocol C already establishes exactly this
comparison pattern (train over the candidate's own topology class,
report faithful/paper-reimplementation baselines separately).

## Verdict

`TARGETED_PILOT_CANDIDATE_SET`: the 8 historical monolithic policies
(`fifo`, `edf`, `scorpio_style_slo_guard`, `admission_control`,
`weighted_shortest_processing`, `estimated_service_time_first`,
`best_fit`, `multi_bin_batching`) — the set the 910-window search already
showed has real, robust, oracle-headroom-backed ANWG specialization.

`EXTERNAL_EVALUATION_BASELINES`: `vllm_faithful`, `sarathi_faithful`,
`vllm_chunked_prefill_faithful` — reported at evaluation time against
the trained selector, not included in its training candidate set.

`READY_FOR_TARGETED_DATASET_V2_PILOT = yes` (scope: historical policies
as the candidate set, per Option B — this was the open question the
prior task's Verdict section left unresolved).

`READY_FOR_SELECTOR_TRAINING = no` (not attempted in this task, per
instructions — the next task should generate the 250-500-window pilot
with this explicit candidate set, then train only if that pilot's own
quality gates hold).
