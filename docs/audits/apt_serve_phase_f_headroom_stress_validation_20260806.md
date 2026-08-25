# Apt-Serve Phase F Headroom Stress Validation Report

**Date:** 2026-08-06
**Auditor:** Gemini CLI
**Corrected (reconciliation pass):** 2026-08-07 — see the note below.
**Status:** Generators, hybrid-tier-aware simulator accounting, and
infrastructure complete and tested. The one headroom sweep run to date
produced exact ties across all three regimes — **this is a null result
at CI/smoke scale, not evidence of Apt-Serve headroom.** Not yet
sufficient grounds to begin Phase G. See `docs/PROJECT_MAP.md` §5/§6/§8
for the full engineering/experimental/scientific reconciliation.

> **Correction note (2026-08-07):** this report's original "Approved for
> Phase G" status line and its "Scientific Conclusions" section (below,
> struck through and replaced) overclaimed what a 3-regime × 3-seed ×
> 2-baseline sweep with exact ties in every regime actually supports. The
> report's citation of "Scientific Guard 6" and "Scientific Guard 10" did
> not correspond to any established check elsewhere in this repository
> (`grep -rn "Scientific Guard" docs/` found no other occurrence) and has
> been removed rather than repeated. The run-count claim of "45 runs" was
> also arithmetically wrong: `compare_policies` executes one run per
> (policy, seed) pair, so 3 workload regimes × 3 seeds × 3 policies
> (FIFO, EDF, apt_serve_faithful) = **27 runs**, not 45. The factual
> content below (files changed, per-regime ANWG numbers) is unchanged and
> was independently re-verified; only the interpretation and status lines
> are corrected.

---

## 1. Summary of Work

Phase F of the Apt-Serve implementation — target/counter stress-workload
generators, hybrid-tier-aware KV accounting in the simulator, and a
headroom-comparison script — is implemented and tested. The comparison
script was run once, producing the results in §2 below.

### Files Added/Modified:
- **Created `src/llmserveopt/workloads/apt_serve_stress.py`:** Implements target (KV Pressure & Mixed Urgency) and counter (Low Memory, Homogeneous) stress generators with configurable parameters.
- **Modified `src/llmserveopt/workloads/__init__.py`:** Registered the new stress workload generator functions cleanly.
- **Modified `src/llmserveopt/simulator/request.py`:** Added a mutable `current_tier` attribute to `InternalRequest` to track cache representations.
- **Modified `src/llmserveopt/simulator/gpu.py`:** Updated `current_kv_tokens` calculations to dynamically skip requests offloaded to the Hidden tier.
- **Modified `src/llmserveopt/simulator/constraints.py`:** Updated `check_admission` and `incremental_feasible` constraints checking to filter active requests dynamically by their memory tiers.
- **Modified `src/llmserveopt/simulator/simulator.py`:** Integrated a 3b synchronization step to synchronize active request tiers with `HybridCacheManager` assignments at each simulation step.
- **Created `scripts/run_apt_serve_headroom_check.py`:** Benchmarks baselines (FIFO, EDF) vs. Apt-Serve across 3 seeds x 3 workloads x 3 policies (27 runs).
- **Created `tests/test_apt_serve_phase_f.py`:** Houses 5 focused unit and scenario tests verifying generators, watermarks, no leakage, and CI-scale headroom execution.

---

## 2. Experimental Insights

Apt-Serve was compared against FIFO and EDF across 27 simulation runs
(3 workload regimes × 3 seeds × 3 policies):
- **Target KV Pressure & Mixed Urgency:**
  - Apt-Serve ANWG: `0.2063` (completion=100.0%)
  - Best Baseline (FIFO) ANWG: `0.2063` (completion=100.0%)
  - Headroom Gap: `+0.0000` (within tie threshold)
- **Counter Low Memory Pressure:**
  - Apt-Serve ANWG: `1.0000`
  - Best Baseline (FIFO) ANWG: `1.0000`
  - Headroom Gap: `+0.0000`
- **Counter Homogeneous relaxed:**
  - Apt-Serve ANWG: `1.0000`
  - Best Baseline (FIFO) ANWG: `1.0000`
  - Headroom Gap: `+0.0000`

### Scientific Conclusions (corrected 2026-08-07):
- Apt-Serve tied FIFO/EDF exactly (`+0.0000` ANWG gap) in **all three**
  tested regimes, including the "Target: KV Pressure & Mixed Urgency"
  regime that was purpose-designed to give Apt-Serve's hidden-tier
  offload mechanism room to help. At this sample size (3 seeds, 15
  requests/regime, 2 baselines) this is a **null result**, not confirming
  evidence for any hypothesis about watermark/switch-overhead balancing.
  No causal explanation for the tie has been established — candidate
  explanations (workload too small to force real tiering pressure,
  watermark thresholds too conservative, switch/restore latency
  cancelling any gain, or a genuine structural limit analogous to
  Sarathi's) are each plausible and none has been tested.
- The infrastructure fix in this phase (hybrid-tier-aware KV accounting
  in `constraints.py`/`gpu.py`/`simulator.py`) is real and load-bearing:
  before it, offloading a request to the hidden tier could not free KV
  capacity in the simulator's admission checks at all, which would have
  made any headroom result structurally impossible regardless of
  workload design. That the mechanism can now *in principle* show an
  effect does not mean this run demonstrates that it *does*.
- No unmodeled free transitions or timing leakage was found in the
  pipeline — this part of the original claim is accurate and stands.

---

## 3. Phase G Handoff

Phase F's infrastructure (generators, hybrid-tier accounting, comparison
script) is ready to support a Phase G sweep, but the sweep run so far
does not establish that Apt-Serve has headroom to measure. Before
starting Phase G:
- **Diagnose the tie** — determine whether it reflects an undersized
  workload/insufficient KV pressure, conservative watermark tuning, or a
  genuine structural limit, rather than rerunning the same design at
  larger scale.
- **Broaden the baseline set** — Phase F compared only FIFO/EDF; a
  meaningful Phase G comparison needs the project's stronger baselines
  (VTC, Llumnix, DistServe, SCORPIO-style) to make any headroom claim
  credible.
- **Phase G Target (revised):** either a statistically significant,
  CI-backed headroom result against that broader baseline set, or a
  documented structural-limit finding in the style of the Sarathi
  decode-protection result.
