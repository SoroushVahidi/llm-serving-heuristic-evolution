# Portfolio Policy Synthesis Design v1 Analysis

Date: 2026-08-24  
Classification: `PIVOT_TO_PORTFOLIO_POLICY_SYNTHESIS`  
Chosen direction: `PORTFOLIO_SYNTHESIS_DEBT_SCHEDULER`

## Context

Family-A statewise selection is closed: `SELECTOR_HYPOTHESIS_FALSIFIED`. The guarded ESTF/WFS composite line is also closed: `MECHANISM_COMPOSITE_STATIC_NO_GO`. This design therefore does not reopen selectors, DAgger, support expansion, guarded ESTF/WFS blending, scalar analytic indices, or oracle labeling. It asks what genuinely new native policy family should be screened next against the current six-policy envelope.

## Current Six-Policy Portfolio

| policy | primary mechanism | strongest regimes | main failure |
|---|---|---|---|
| `full_prefill` | uninterrupted prompt/prefill admission | hog-prefill TTFT/E2E cases | late tenant convoying |
| `chunked_prefill_small` | small prefill chunks | late/short TTFT contention | interchunk overhead for hog prompts |
| `estimated_service_time_first` | shortest estimated service first | completion/size-sensitive Family-A cells | fairness/SLO debt failures |
| `weighted_fair_share` | instantaneous class deficit × priority / service | robust Family-A fairness/SLO default | completion-related regret |
| `least_laxity_first` | lowest deadline laxity first | urgency pockets | KV-blind admission |
| `kv_constrained_online` | KV reserve with urgent bypass | scarce-KV/bulk pressure | reserve over-conservatism pockets |

## Portfolio Redundancy And Contribution

MF-PSD is pairwise by mechanism family, not a dense six-policy matrix. Within that contract:

| family | scenarios | best fixed | envelope mean | headroom over best fixed | epsilon-unique wins |
|---|---:|---|---:|---:|---|
| `FAMILY_A_FAIRNESS_STARVATION_V2` | 72 | `WFS` | 0.757050 | 0.016488 | {'ESTF': 26, 'WFS': 29} |
| `FAMILY_B_PREFILL_DECODE_V2` | 32 | `full_prefill` | 0.782185 | 0.049433 | {'full_prefill': 16, 'chunked_prefill_small': 15} |
| `FAMILY_C_KV_PRESSURE_V2` | 72 | `kv_constrained_online` | 0.870670 | 0.003186 | {'kv_constrained_online': 45, 'least_laxity_first': 5} |


Public-trace six-policy replay is saturated: 60 annotated windows, six-policy envelope mean ANWG 1.000, best fixed 1.000, envelope gain 0.000, and all policies tie exactly in 60/60 windows.

## Largest Envelope Holes

The strongest holes are: completion pressure under fairness/SLO conflict, prefill/decode contention under long-prompt hogs and late tenants, queue-age starvation not captured by the six anchors, mixed-regime transitions, and bounded KV reserve/urgency interactions. The lowest pair-envelope scenarios are mostly Family-A high-utilization favored-long/skew conflicts and Family-B late-TTFT hog cases.

## Abandoned But Unfalsified Ideas

Closed: ESTF/WFS selector, DAgger/support expansion, guarded ESTF/WFS rules, coefficient-free analytic indices, Family-A stateful parent selectors, PrefillControl children for the full/chunked pair, and KV parent switching/hysteresis as previously formulated.

Still promising if treated as native policies rather than parent selectors: accumulated service/SLO/KV debt scheduling, two-timescale budget control, and stateful burst-aware prefill shaping.

## Candidate Families

1. `PORTFOLIO_SYNTHESIS_DEBT_SCHEDULER`: accumulated service, age, SLO, completion, and KV debt; chosen.
2. `PORTFOLIO_SYNTHESIS_TWO_TIMESCALE_CONTROLLER`: slow quota/chunk/reserve updates plus fast ranking.
3. `PORTFOLIO_SYNTHESIS_LEXICOGRAPHIC_NATIVE_POLICY`: hard deadline/debt/completion/KV buckets.
4. `PORTFOLIO_SYNTHESIS_KV_COMPLETION_POLICY`: KV reserve plus completion-release packing.
5. `PORTFOLIO_SYNTHESIS_BURST_AWARE_PREFILL_SHAPER`: burst debt modulates prompt chunks and class service.

## Chosen Direction

`PORTFOLIO_SYNTHESIS_DEBT_SCHEDULER` is the best next family. It is native, stateful, and mechanistically distinct from WFS because it accumulates debt across steps rather than using only instantaneous demand/served deficit. It can address fairness/SLO, completion, queue-age, KV, and mixed-transition holes while staying low-dimensional and causal.

## Screening Experiment Contract

Experiment: `portfolio_synthesis_debt_scheduler_screen_v1`.

Run locally on a fixed TRAIN-only subset: 8-12 Family-A scenarios, 8 Family-B scenarios, 8 Family-C scenarios, and optional saturated public-trace sanity windows. Baselines are the six existing policies plus the six-policy envelope. No DEV, FINAL, TEST, Wulver, GPU, oracle labels, selector training, or broad parameter search.

Primary score: `MG_c(x;P6)=max(R_c(x),E6(x))-E6(x)`, where `E6` is the current six-policy envelope.

GO requires all of:

- mean marginal envelope contribution >= 0.005 ANWG
- positive mean marginal contribution in at least 2 mechanism/config regions
- >= 3 unique wins at eps=0.005 across at least 2 mechanism families
- no mechanism-group standalone regression > 0.030 ANWG versus that group best fixed parent
- max decision overlap with any parent <= 95%
- max reward correlation with any parent <= 0.985 unless MG and unique-win gates pass by >2x
- debt/KV/SLO diagnostics match the mechanism hypothesis
- gains not concentrated in one scenario or one family

NO-GO fires if mean MG < 0.001, no real unique wins, collapse to an existing parent, severe group regression, mechanism inversion, or all gains come from one isolated subset.

## Stop Rule

If the debt scheduler is NO-GO, allow at most one additional genuinely distinct family from this design, preferably the two-timescale controller if diagnostics show budget-lag or transition failure. If that also fails strict screening, stop portfolio-policy synthesis and return to broader scheduler search outside Family-A-derived mechanisms.

## Publication Contribution If Successful

A successful debt scheduler would contribute: a new native scheduling mechanism based on accumulated service/SLO/KV debt, measurable marginal contribution to a six-policy envelope rather than just beating WFS, and a methodological story showing how a falsified selector path can still yield mechanism-grounded policy synthesis.

## Confirmations

No selector reopening, oracle acquisition, DEV-driven redesign, FINAL, TEST, heavy simulation, Wulver jobs, GPUs, or git mutation were performed.
