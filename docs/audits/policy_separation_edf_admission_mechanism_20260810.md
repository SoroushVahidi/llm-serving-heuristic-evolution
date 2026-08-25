# EDF / Admission-Policy Mechanism Audit (Study C, boundary-refinement v1)

Code-level inspection of `edf.py`, `admission_control.py`, and
`scorpio_style_slo_guard.py` (plus `least_laxity_first.py` for contrast),
performed before running any new Study C cells, per the boundary-refinement
experiment's requirement to understand *why* `admission_control ~= edf`
while `scorpio_style_slo_guard >> edf` under unsalvageable overload (job
1170116 finding).

## Ranking rule

| Policy | Sort key (ascending unless noted) |
|---|---|
| `edf` | `slo_deadline` only |
| `least_laxity_first` | `laxity` (= `slo_deadline - now - est_service_s`), then `slo_deadline`, then `-priority`, then `request_id` |
| `admission_control` | `laxity`, then `-priority`, then `est_service_s`, then `slo_deadline`, then `request_id` |
| `scorpio_style_slo_guard` | a composite score (`urgency + priority_weight*priority + age_bonus*age - decode_penalty`) descending, then `laxity`, then `-priority`, then `arrival_time`, then `request_id` |

## Admission/rejection rule

- **`edf`**: none. Every waiting request is a candidate every step; greedy
  first-fit onto any GPU with capacity. No concept of "this job cannot make
  its deadline" exists anywhere in the policy.
- **`admission_control`**: has a laxity-threshold *filter*
  (`laxity >= -laxity_threshold`) that is supposed to skip requests already
  too late to save -- but its constructor default is
  `laxity_threshold: float = float("inf")`, which makes `min_laxity = -inf`,
  so the filter condition `laxity >= -inf` is **always true**. As configured
  in both the job-1170116 three-case config and this boundary-refinement
  config (`policies:` lists carry no per-policy kwargs), `AdmissionControlPolicy`
  performs **zero actual rejection filtering**. It is not a capacity/SLO
  guard in this experiment's configuration -- it is a pure re-ranking
  policy (laxity-first instead of deadline-first) with the same
  never-reject admission behavior as `edf`. This -- not measurement noise --
  is the structural reason job 1170116 found `admission_control ~= edf`.
- **`scorpio_style_slo_guard`**: has three independent, all-active
  admission-shaping mechanisms, none of which are inert by default:
  1. A laxity/TTFT-slack filter (`laxity_threshold=0.0`,
     `ttft_slack_threshold=0.0` by default -- unlike `admission_control`,
     these defaults are finite, so the filter is live) that drops a request
     from *this step's* candidate set once it can no longer make its
     deadline even if admitted immediately.
  2. A "guard active" state
     (`kv_pressure/decode_pressure/queue_pressure` over threshold, or mean
     candidate laxity negative) that, when triggered, further excludes long
     predicted-decode candidates unless they are still salvageable
     (`laxity < 0.5`), and caps admissions per step via a refilling credit
     budget (token-bucket throttle) rather than admitting every feasible
     candidate immediately.
  3. A composite urgency score that actively penalizes long predicted
     decode work under guard-active pressure, rather than only ordering by
     laxity/deadline.

## SLO observables used

All four policies use only online-observable proxies
(`slo_deadline`, `predicted_output_tokens`, `prompt_tokens`, GPU state) --
none read `actual_output_tokens`. `admission_control` and
`scorpio_style_slo_guard` both derive an estimated-service-time proxy via
the shared `scoring.predicted_service_proxy` (`alpha*prompt + beta*predicted_output`).

## Whether impossible jobs are detected

- `edf`: no.
- `admission_control`: structurally capable (the laxity filter is exactly a
  "this job is unsalvageable" test), but inert at this experiment's default
  `laxity_threshold=inf` -- see above.
- `scorpio_style_slo_guard`: yes, actively, via both the per-step laxity/TTFT
  filter and the guard-active long-decode exclusion, which is the direct
  mechanism behind its win over `edf` in every stressed (impossible-job)
  cell of job 1170116.

## Conclusion

`admission_control` is **not** currently a genuine SLO-feasibility guard in
this codebase's default configuration -- it is a laxity-ordered variant of
EDF/LLF with an admission filter that is present in the code but disabled
by its own default constructor argument. `scorpio_style_slo_guard` is a
genuine admission/throttling guard: filter, pressure-adaptive exclusion,
and credit-budget throttling all fire by default. This fully explains job
1170116's finding without needing to hypothesize any simulator defect, and
motivates Study C's small targeted overload_factor x fraction_impossible
matrix (`edf_admission_mechanism_summary.csv`) to quantify the size of this
gap rather than only its sign.
