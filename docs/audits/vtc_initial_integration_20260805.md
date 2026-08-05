# VTC Initial Integration — 2026-08-05

Companion to `docs/audits/vtc_official_artifact_audit_20260805.md` (the
artifact/mechanism audit) and `baselines/vtc/PROVENANCE.md` (the full
provenance record). This document is the integration-and-smoke-evaluation
record specifically.

## What was built

- `baselines/vtc/adapter/` — dynamically imports and drives the real,
  unmodified `VTCReqQueue`/`ReqQueue`/`Req`/`Batch`/`SamplingParams`
  classes from a pinned local clone (`~/.cache/external_baselines/VTC` @
  `192c2e2014c69c8c6c699d7113c3822e4db632e6`). `VTCFairnessPolicy`
  (`BasePolicy` subclass, `name="vtc_fairness_reference"`) is the
  simulator-facing wrapper.
- `baselines/vtc/fairness_workloads.py` — six labeled fairness-extension
  workload families (see the audit doc §7), since the accepted canonical
  suite has no tenant concept at all.
- `tests/test_vtc_baseline_adapter.py` — 25 tests, all passing.
- `scripts/run_vtc_smoke_evaluation.py` — the smoke comparison this
  document reports.

## Smoke evaluation methodology

Compared four policies — `fifo`, `vtc_fairness_reference`,
`shortest_prompt_first` (throughput-oriented), `scorpio_style_slo_guard`
(SLO/admission-oriented) — across all six fairness-extension families, on
a single GPU (`max_active_sequences=8, max_batch_tokens=1024,
max_kv_tokens=4096`).

Two rounds of metric design are worth recording, since the first round
produced a misleading non-finding:

1. **First attempt (fully drained, end-of-run totals):** every policy
   produced identical per-tenant total-service numbers in every family,
   because with `drain_steps` generous enough for every request to
   eventually complete, per-tenant total service converges to per-tenant
   total DEMAND regardless of scheduling order — a policy-invariant
   quantity. This looked like "no difference between FIFO and VTC," which
   would have been a wrong conclusion drawn from an uninformative metric,
   not a real finding.
2. **Corrected methodology:** per-tenant service is measured at a
   **mid-run checkpoint** (60% of the way through each workload's arrival
   window), matching the paper's own primary evaluation methodology
   (cumulative service received *over time*, not at infinite drain). This
   is the metric reported below.

Metrics reported per (family, policy): ANWG
(`arrival_normalized_weighted_goodput`), completion fraction, Jain's
fairness index over per-tenant normalized (service ÷ completions) values,
max-min fairness ratio, service disparity (max − min per-tenant tokens),
and starved-tenant count — all computed at the mid-run checkpoint. Full
per-tenant breakdown: `baselines/vtc/smoke_results/vtc_smoke_20260805.json`.

## Results

| Family | fifo Jain | vtc Jain | spf Jain | scorpio Jain | Divergence? |
|---|---|---|---|---|---|
| `balanced_tenants` | 0.993 | 0.993 | 0.993 | 0.993 | No |
| `one_heavy_hitter` | 0.791 | 0.791 | 0.791 | 0.791 | No |
| `heterogeneous_token_sizes` | 0.715 | **0.483** | 0.715 | 0.715 | **Yes — see below** |
| `bursty_tenant` | 1.000 | 1.000 | 1.000 | 1.000 | No |
| `returning_inactive_tenant` | 0.995 | 0.995 | 0.995 | 0.995 | No |
| `priority_fairness_conflict` | 0.999 | 0.999 | 0.999 | 0.999 | No |

**5 of 6 families: no divergence at all.** Not because VTC has no effect
in principle, but because at this smoke-scale capacity, demand never
generated sustained admission backlog under any of the four policies —
everything gets admitted close enough to its arrival time that scheduling
ORDER never actually has to break a tie between two waiting requests from
different tenants. A smoke test at this scale simply did not stress the
system hard enough to distinguish a fair scheduler from FIFO in these
families.

**`heterogeneous_token_sizes` (the one family that did diverge):**
VTC's completion/ANWG dropped to 0.036 (vs. 1.0 for the other three) and
2 of 4 tenants were fully starved by the mid-run checkpoint. Root cause,
confirmed by direct code inspection and a smaller isolated reproduction
(see the audit doc §9-10): `VTCReqQueue.generate_new_batch`'s official
memory-safety gate (`_can_add_new_req`, inherited from `ReqQueue`)
reserves KV budget for a request's full **predicted** decode length before
admitting it (a worst-case, sorted-cumulative reservation formula); every
native policy compared against here (`fifo`, `shortest_prompt_first`,
`scorpio_style_slo_guard`) uses `BasePolicy._feasible_on_gpu`, a much
simpler check that only accounts for currently-consumed tokens and
reserves nothing for future decode growth. This family's `long_outputs`
tenant (mean predicted length 500 tokens) triggers this gate hard under a
4096-token KV budget. **This is a genuine, disclosed methodological
confound, not a fairness or throughput failure of VTC's algorithm** — it
would occur even with FIFO ordering substituted for VTC's min-served
selection, purely from the reservation formula being stricter.

Widening or narrowing the GPU capacity does not resolve this cleanly: a
tighter capacity (tested down to `max_active_sequences=2,
max_batch_tokens=256, max_kv_tokens=512`) makes the confound dominate
`one_heavy_hitter` too (VTC's full-run completion fraction dropped to
0.10 vs. FIFO's 1.00, purely from the reservation gate, confirmed by
inspecting per-tenant service directly) rather than revealing a cleaner
fairness signal.

## Central conclusion

The initial integration is **mechanically correct and faithful** (fidelity
tests all pass; the real, unmodified official algorithm runs verbatim) but
this first smoke pass **did not cleanly demonstrate VTC's fairness
advantage**, for two compounding, disclosed reasons: (1) the smoke-scale
capacity/demand ratio rarely produced real admission backlog, and (2)
where it did, an admission-gate-conservativeness confound (unrelated to
VTC's fairness mechanism) dominated the outcome instead. This is exactly
what a smoke test is supposed to surface before a full sweep is
greenlit — reporting it honestly here rather than only showing the
families that happened to look good.

## Benchmark readiness

- **Canonical suite compatibility:** not compatible — no tenant concept
  exists in any canonical family (see audit doc §7). VTC is scoped to
  `baselines/vtc/fairness_workloads.py` only.
- **Fairness-extension workloads:** built, labeled, not touching the
  canonical suite (§ above).
- **Smoke evaluation:** complete, with an honestly-reported non-result in
  5/6 families and one confound-dominated divergence in the sixth.
- **Full evaluation ready:** **no.** Two prerequisites before a full sweep
  would be worth running: (a) re-tune capacity/contention specifically to
  produce genuine backlog without triggering the reservation confound
  (e.g. moderate capacity + moderate, not extreme, predicted-output-length
  spread), or (b) give the comparison policies a matched worst-case
  reservation-style admission check so the comparison isolates ordering
  behavior alone. Neither was attempted in this pass — out of scope for
  an initial integration, and explicitly not requested ("Do not launch a
  long full benchmark unless the smoke and fidelity gates pass").

## Exact next action

Before any full VTC sweep: design a capacity/contention regime (or a
matched-admission-check variant of the comparison policies) that produces
genuine scheduling-order-dependent backlog in `one_heavy_hitter` and
`bursty_tenant` specifically (the two families most directly targeting
VTC's headline claim) without triggering the admission-reservation
confound documented above. This is a methodology-design task, not a code
change to the adapter itself (the adapter's fidelity is already verified).
