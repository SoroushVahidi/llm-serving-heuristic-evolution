# External Baseline Decision Document

**Phase:** 2B.9  
**Date:** 2026-06-25  
**Branch:** `phase2b9-selector-robustness-and-suite-freeze`  
**Purpose:** Decisive, publication-oriented specification of which external baselines to implement, cite, or skip for the first paper submission.

> **Addendum (post-2026-07-22): SLAI/RAD identity finding.** This document
> predates a live-research finding made while implementing `slai_faithful`
> (see `docs/slai_faithful_scheduler_reference.md`): **SLAI and RAD are the
> two schedulers introduced by the same paper** — Bari, Hegde, de Veciana,
> "Optimal Scheduling Algorithms for LLM Inference: Theory and Practice"
> (ACM SIGMETRICS 2026 / arXiv:2508.01002) — with official Apache-2.0 code
> at `github.com/agrimUT/SLAI`. This document does not mention SLAI/RAD at
> all (they postdate 2026-06-25); nothing here is contradicted, but a
> future "must-add" pass should note that `slai_faithful` is now
> **implemented** (§ new row not added to the table below to avoid
> rewriting a dated decision record — see `docs/current/BASELINES.md`
> instead for the current, live external-baseline inventory). Separately:
> while researching this, the "PROSERVE (2024)" and "Jaillet et al. (2024)"
> citations in §B.3/B.5 below were found to have the wrong year — the real
> papers are arXiv:2512.12928 (Dec 2025) and arXiv:2502.07115 (Feb 2025)
> respectively. Not corrected here (out of scope for the SLAI work), but
> flagged so a future citation-accuracy pass catches it.

---

## A. Already Implemented Deployable Baselines

All 19 are registered in `BASELINE_NAMES` and are selector candidates.  
None use `actual_output_tokens` (no oracle information).

| # | Policy | Category | Safe manuscript wording | Fidelity | Selector candidate |
|---|--------|----------|------------------------|----------|--------------------|
| 1 | `fifo` | Classical | "FIFO baseline" | Exact | ✅ |
| 2 | `edf` | Classical | "Earliest Deadline First (SLO-aware)" | Exact | ✅ |
| 3 | `shortest_output_first` | SRPT-style | "Shortest Remaining Processing Time proxy (predicted output)" | Faithful | ✅ |
| 4 | `shortest_prompt_first` | Heuristic | "Shortest Job First on prompt length" | Style | ✅ |
| 5 | `greedy_token_fill` | Packing | "Greedy token-budget batch filling" | Style | ✅ |
| 6 | `least_loaded` | Load balancing | "Least-loaded scheduling" | Exact | ✅ |
| 7 | `multi_bin_batching` | Batching | "Multi-bin length-aware batching" | Internal | ✅ |
| 8 | `random_feasible` | Stochastic | "Random feasible baseline" | Exact | ✅ |
| 9 | `first_fit` | Packing | "First-Fit KV-budget packing" | Exact | ✅ |
| 10 | `best_fit` | Packing | "Best-Fit KV-budget packing" | Exact | ✅ |
| 11 | `orca_style` | Serving-style inspired | "Orca-inspired continuous batching with iteration-level scheduling" | Style/inspired | ✅ |
| 12 | `vllm_style_token_budget` | Serving-style inspired | "vLLM-inspired token-budget scheduler" | Style/inspired | ✅ |
| 13 | `sarathi_style` | Serving-style inspired | "Sarathi-Serve-inspired stall-free chunked-prefill scheduler" | Style/inspired | ✅ |
| 14 | `splitfuse_style` | Serving-style inspired | "SplitFuse-inspired mixed prefill–decode batching" | Style/inspired | ✅ |
| 15 | `slo_slack_score` | Composite | "Composite SLO-slack urgency+throughput scheduler" | Internal | ✅ |
| 16 | `weighted_shortest_processing` | WSPT | "Weighted Shortest Processing Time (WSPT)" | Exact | ✅ |
| 17 | `least_laxity_first` | Deadline/laxity | "Least Laxity First (LLF) deadline scheduler" | Exact | ✅ |
| 18 | `estimated_service_time_first` | SJF proxy | "Estimated Service Time First (PARS-style SJF proxy)" | Faithful/inspired | ✅ |
| 19 | `admission_control` | Admission control | "Urgency-sorted admission control with laxity threshold" | Internal | ✅ |

**Fidelity key:**
- **Exact**: direct algorithm implementation (no approximation)
- **Faithful**: captures the core scheduling criterion of the referenced method
- **Style/inspired**: naming convention inspired by real systems; implementation is a simulator-level heuristic, not the full system
- **Internal**: novel/original baseline not derived from a specific published method

---

## B. Must Add Before Final Publication

These baselines are needed to fairly position the work in the LLM scheduling literature. Each represents a distinct decision axis (LTR/priority-aware, SLO guard, KV-cache scheduling, fairness, priority batching) not covered by the current 19.

---

### B.1 Prompt-Aware LTR / PARS-style Scheduler — **EVALUATION-READY, OFFLINE-SCORED (completed 2026-08-04)**

**Registry:** not registered (deliberately still not selector-eligible —
see below). **Files:** `baselines/vllm_ltr/` (isolated, outside
`src/llmserveopt`). **Status:** the official implementation for this exact
gap was found and used instead of a hand-rolled ranker —
[hao-ai-lab/vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr) (pinned
commit `13bbf6ff3dab661791d41362551b089e5f77c91c`, Apache-2.0), paper Fu et
al. *"Efficient LLM Scheduling by Learning to Rank,"* **NeurIPS 2024 (main
conference)**, arXiv:2408.15792 as supplementary preprint id. The official
checkpoint (`LLM-ltr/OPT-Predictors`) was downloaded, hash-recorded, and
verified — exact architecture match and bit-exact independent-recomputation
agreement on real ShareGPT text; a complete offline scoring pipeline turns
real prompt text into cached, integrity-checked scores; the ranking/
tie-break rule is reproduced exactly. The official predictor still cannot
be invoked *live* inside the per-step simulator loop, because
`ObservableRequest` carries only integer `prompt_tokens` counts and was
deliberately not modified to carry raw text (explicit scope boundary, not
an unresolved gap). See `docs/audits/vllm_ltr_baseline_audit_20260804.md`
for the full verification record, real overhead measurements, and the one
disclosed-but-infeasible check (a live differential against the actually
served vLLM-fork engine). A comparison sweep has since been run (real
WildChat-1M text, 300 requests, 3 seeds, 10 policies, independently
re-verified) — `docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`.
Result: it tied FIFO/EDF/EST/SOF/WSP/`oracle_srtf` exactly in that regime
(the oracle itself ties FIFO, so no reordering policy had headroom to show
benefit there); ranking agreement with EST/SOF is moderate, not near-1.0,
so it is not behaviorally redundant with them. **Still not registered as a
selector candidate** — EVALUATION_ONLY; foundational-library eligibility
was not established by this run and needs a higher-contention workload
regime to actually test.

**Why it matters (original rationale, unchanged):** PARS (2023) introduces learning-to-rank (LTR) for LLM request scheduling, predicting completion order to reduce mean latency. The current `estimated_service_time_first` is a simplified SJF proxy (uses predicted output as service proxy) but lacks the ranking model. Without a true LTR baseline, it is unclear whether the selector's benefit comes from policy selection or could be matched by a good learned ranker.

**Decisions it makes:** Ranks requests by predicted service time using a learned ranker (e.g., RankNet) rather than a hand-coded proxy.

**Features needed:** Prompt length, predicted output length, arrival time, SLO class. All online-observable.

**Simulator-compatible:** Yes. The simulator accepts any priority-sort function. LTR priority score replaces the hand-coded proxy in `estimated_service_time_first`.

**Implementation difficulty:** Medium. Requires:
1. Collecting (features, completion-time) training pairs from the simulator.
2. A lightweight scikit-learn RankNet or LGBMRanker.
3. An online inference wrapper.
Estimated ~2–3 days of implementation + testing.

**Selector candidate:** Yes — if implemented as a deployable policy with online-observable features.

**Leakage risks:** The ranker must not use `actual_output_tokens` during inference. Training pairs may use actual completion time (post-hoc supervision), but the inference path must be leakage-free.

**Tests required:** Leakage test (no actual_output_tokens in inference), regression test on overloaded workloads, correctness test on small synthetic trace.

**Safe manuscript wording:** "Prompt-aware learning-to-rank scheduler (inspired by PARS [cite]); implementation uses predicted completion-time ranking without access to ground-truth output lengths."

---

### B.2 SCORPIO-style SLO Guard — **IMPLEMENTED (Phase 2B.10)**

**Registry name:** `scorpio_style_slo_guard`  
**File:** `src/llmserveopt/policies/scorpio_style_slo_guard.py`  
**Status:** Simulator-compatible SCORPIO-inspired guard with TTFT proxy, decode-pressure
penalty, and admission credit throttling. See `docs/audits/phase2b10_scorpio_slo_guard_summary.md`.

---

### B.3 WAIT / Nested-WAIT or Jaillet-style KV-Cache-Aware Scheduler

**Why it matters:** Under KV-cache pressure, naive FIFO or urgency-based scheduling causes cache thrashing. Jaillet et al. (2024) and WAIT/Nested-WAIT (2024) propose scheduling policies that account for KV-cache slot availability before admission. The current `kv_pressure_decode_heavy` failure case (WG=0.477 even for WSP) shows that a KV-cache-aware baseline may outperform all current policies.

**Decisions it makes:** Admits requests only when KV-cache can accommodate them; may defer admission to avoid cache eviction; groups requests by KV-slot footprint.

**Features needed:** `kv_utilization`, `free_sequence_ratio`, estimated sequence length (predicted output). All online-observable.

**Simulator-compatible:** Yes. The simulator has `max_kv_tokens` and tracks `kv_utilization`. This policy can read the simulator's KV state.

**Implementation difficulty:** Medium-High. KV eviction is not currently modeled per-request in the simulator (only aggregate utilization is tracked). May need per-sequence KV tracking.

**Selector candidate:** Yes — if KV features are online-observable.

**Leakage risks:** Must use predicted (not actual) output length for KV footprint estimates.

**Tests required:** Test that policy defers admission when KV is full; regression test on kv_pressure_decode_heavy.

**Safe manuscript wording:** "KV-cache-aware admission scheduler (inspired by WAIT [cite] and Jaillet et al. [cite]); defers admission when KV-cache utilization exceeds threshold."

---

### B.4 FairBatching-style Fairness-Aware Batch Formation

**Why it matters:** FairBatching (2023) addresses fairness across request classes by ensuring no class is starved during overload. The current 19 policies all optimize for aggregate WG; none explicitly enforce fairness. Under priority-mix workloads, a fairness baseline may show different WG/fairness trade-offs that the selector should account for.

**Decisions it makes:** Targets proportional representation of SLO classes in each batch; uses per-class token budgets or admission quotas.

**Features needed:** SLO class fractions, recent per-class completion rates. Online-observable.

**Simulator-compatible:** Yes. Batch formation can be modified to enforce per-class token budgets.

**Implementation difficulty:** Low-Medium. Core: per-class admission queues + token budget split. ~150 LOC + tests.

**Selector candidate:** Yes.

**Leakage risks:** Class fractions are online-observable. No leakage risk.

**Tests required:** Test that FairBatching equalizes per-class completion rates; regression test on mixed-SLO workloads.

**Safe manuscript wording:** "Fairness-aware batch formation (FairBatching-inspired [cite]); allocates per-SLO-class token budgets to prevent request starvation."

---

### B.5 PROSERVE SlideBatching-style Priority-Aware Scheduler

**Why it matters:** PROSERVE (2024) introduces priority-aware preemptive scheduling with sliding window batches. It directly targets the SLO-aware + priority-weighted scheduling problem studied in this paper. Without it, reviewers may argue that the selector's improvement could be replicated by a single strong priority scheduler.

**Decisions it makes:** Assigns priorities based on SLO urgency + remaining budget; preempts low-priority requests when high-priority ones arrive; uses sliding window batching to balance throughput.

**Features needed:** SLO class, remaining deadline (laxity), estimated service time. All online-observable.

**Simulator-compatible:** Yes. Our simulator supports priority-ordered queues and urgency-based admission.

**Implementation difficulty:** Medium. Conceptually similar to LLF + token-budget batching, but with preemption. The simulator does not currently support preemption between windows; may require a preemption flag. Estimated 2–4 days.

**Selector candidate:** Yes.

**Leakage risks:** None if using estimated (predicted) service times.

**Tests required:** Test preemption under high-priority arrivals; regression test on overloaded_mixed_slo.

**Safe manuscript wording:** "Priority-aware sliding-window scheduler (PROSERVE-inspired [cite]); preempts low-priority requests under high-priority demand with urgency-based sliding batches."

---

## C. Strong Optional

These baselines would strengthen the paper but are not blockers for a first submission.

| Baseline | Rationale | Difficulty | Include as selector candidate? |
|----------|-----------|------------|-------------------------------|
| SlidingServe-style sliding-window chunking | Tests if chunked prefill beats full chunking under load | Medium | Yes |
| Apt-Serve-style hybrid-cache scheduler | Tests KV reuse + scheduling co-optimization | High | No (complex feature set) |
| AccelGen-style dynamic chunking | Tests adaptive chunk-size selection | Medium | Yes |
| SOLA-style state-aware SLO scheduler | Tests state-dependent SLO guard | Medium | Yes |

---

## D. Cite Only / Out of Scope for First Paper

These systems are cited as context but are not implemented.

| System | Reason for exclusion |
|--------|---------------------|
| Full vLLM integration | Requires running real GPU inference; out of simulator scope |
| Full Sarathi-Serve integration | Same; our `sarathi_style` is a simulator proxy |
| Full DeepSpeed-FastGen stack | Multi-node; out of scope |
| DistServe | Disaggregated prefill/decode; requires different simulator architecture |
| Mooncake / Conductor | Cluster-level KV transfer scheduling; out of scope |
| Llumnix | Live migration system; out of scope |
| ORBITFLOW full ILP/offload stack | Requires ILP solver + offload system; out of scope |
| Full cluster disaggregated systems | All require cluster infrastructure |

**Safe manuscript wording for cited-only systems:** "We do not compare against full production serving systems (vLLM, Sarathi-Serve, DistServe) as they require GPU hardware and multi-process execution beyond our simulator scope; instead, we implement simulator-level heuristics inspired by their core scheduling ideas."

---

## E. Implementation Priority Order

For the next development phase, implement in this order:

1. **SCORPIO-style SLO guard** (B.2) — easiest, directly addresses Phase 2B.7/2B.8 overload failures
2. **KV-cache-aware scheduler** (B.3) — addresses remaining kv_pressure gap (WG=0.477 for all policies)
3. **FairBatching** (B.4) — medium effort, adds fairness dimension missing from current baselines
4. **PARS-style LTR** (B.1) — requires offline training data; tie in with selector dataset expansion
5. **PROSERVE SlideBatching** (B.5) — most complex; implement after preemption support is added

---

## F. Publication Readiness Checklist

| Item | Status |
|------|--------|
| All 19 current baselines documented with fidelity labels | ✅ |
| Must-add baselines identified (5 total) | ✅ |
| Cite-only systems list complete | ✅ |
| Safe manuscript wording provided for all baselines | ✅ |
| Implementation priority order specified | ✅ |
| Oracle excluded from all comparisons | ✅ |
| All baselines are deployable (no oracle leak) | ✅ |
| Must-add B.1–B.5 implemented | Partial — **B.2 SCORPIO-style implemented** (`scorpio_style_slo_guard`, Phase 2B.10); **B.1 evaluation-ready, offline-scored, and now comparison-swept** (official vLLM-LTR checkpoint downloaded + hash-verified + architecturally/numerically verified; comparison sweep run and independently re-verified 2026-08-04 on real WildChat-1M text — tied FIFO/oracle exactly in the tested regime, EVALUATION_ONLY classification, not yet a selector candidate — see `docs/audits/vllm_ltr_baseline_audit_20260804.md` and `docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`); B.3–B.5 pending |

**Additional baseline beyond the original B.1–B.5 list (added 2026-08-04,
not a fulfillment of any B.1–B.5 item — B.1 was already fulfilled by
vLLM-LTR above): PARS-Serve-2026** (`baselines/pars/`, official repo
`SPEAR-UIC/PARS`, paper Tao et al., ISC High Performance 2026,
arXiv:2510.03243). **Status: COMPLETE, INDEPENDENTLY VERIFIED,
EVALUATION_ONLY** — official preprocessing/training code integrated and
run unmodified; no pretrained checkpoint is released, so a real
`bert-base-uncased` checkpoint was trained locally
(`best_val_accuracy=0.9141`, hash-verified `d54be087...c33eb27`). Known
license gap (no upstream LICENSE file) disclosed in
`baselines/pars/PROVENANCE.md`. Named "PARS-Serve-2026" in prose to
disambiguate from this document's own unrelated "PARS-2023-like"
reference (B.1's original rationale text above, and
`docs/external_baseline_coverage_report.md` §15, both refer to a
*different* 2023 paper). Evaluated head-to-head against 9 other policies
across WildChat control and all 7 accepted canonical-suite families (the
first baseline on this branch evaluated on the full canonical suite);
zero unique wins across 8 families, best rank 5th of 10, significantly
worse than the top policy in 5 of 8 families, significantly better than
FIFO/EDF in 3 burst-heavy families — final classification
**EVALUATION_ONLY**, not promoted to any selector-candidate list. See
`docs/BASELINE_STATUS.md` for the current cross-baseline status index,
`docs/audits/pars_baseline_implementation_20260804.md` for the full
implementation record, and
`docs/audits/pars_first_comparative_evaluation_20260804.md` for the full
evaluation, recovery, and independent-verification record.
