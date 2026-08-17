# Resume Here

**Shortest current operational entrypoint.** For the research roadmap, read
[`docs/PROJECT_MAP.md`](../PROJECT_MAP.md). For detailed status, read
[`WORK_STATUS.md`](WORK_STATUS.md). For ordered next actions, read
[`NEXT_ACTIONS.md`](NEXT_ACTIONS.md).

## Current State

| Field | Value |
|---|---|
| Repository | `llm-serving-heuristic-evolution` |
| Branch | `contextual-compositional-heuristics-20260731` |
| Last reconciled SHA | see `git rev-parse HEAD` (MF-PSD v1 build after `dc5757b`) |
| Remote | `origin/contextual-compositional-heuristics-20260731` |
| Expected Git state | clean, 0 ahead / 0 behind after `git fetch --prune origin` |
| Canonical roadmap | `docs/PROJECT_MAP.md` |
| Cluster PSD worktree | `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1` |

Resume commands:

```bash
cd /home/soroush/llm-serving-heuristic-evolution
git fetch --prune origin
git status --short --branch
git rev-parse HEAD
git rev-list --left-right --count @{u}...HEAD
python scripts/check_project_handoff_consistency.py
```

## What This Project Is

This project builds toward a verified contextual compositional scheduler system
for LLM inference serving:

```text
policy-separating workloads -> complementary policy library -> contextual selection (multi-family)
        -> mechanism attribution -> bounded envelope
```

The current primary metric is `arrival_normalized_weighted_goodput` (ANWG).

Typed DSL / module-composition infrastructure exists in-repo but is deferred; **within-scenario composition and synthesis have been demoted** as a central hypothesis.
 
 

## Most Recently Completed Work (Structural Reassessment + MF-PSD)

**The higher-level structural reassessment of the composition hypothesis is
COMPLETE.**

- Audit: [`../audits/reassessment_composition_hypothesis_20260817.md`](../audits/reassessment_composition_hypothesis_20260817.md)
- Verdict: **`COMPOSITION_DEMOTED`**. Within-scenario composition/synthesis
  is now exploratory future work, not the project's central hypothesis.
- Revised roadmap: `policy-separating workloads -> complementary policy
  library -> contextual selection (multi-family) -> mechanism attribution
  -> bounded envelope`.

**MF-PSD v1 (Multi-Family Policy Separation Dataset) — revised roadmap Step
1 — is COMPLETE.**

- Audit: [`../audits/multi_family_policy_separation_dataset_v1_20260817.md`](../audits/multi_family_policy_separation_dataset_v1_20260817.md)
- Artifacts: `experiments/mf_psd_v1/` (`mf_psd_long_v1.csv`,
  `mf_psd_scenarios_v1.csv`, `mf_psd_schema_v1.json`,
  `mf_psd_provenance_v1.json`, `mf_psd_build_manifest_v1.json`); builder
  `src/llmserveopt/policy_separation/mf_psd.py`; CLI
  `scripts/build_mf_psd_v1.py`; tests `tests/test_mf_psd_v1.py` (31/31
  passing).
- Verdict: **`MF_PSD_READY`**. Unifies the three `_COMPOSITION_READY`-gate
  sources (Family A v2 fairness/starvation, Family B v2 prefill/decode
  contention, Family C/KV v2 admission control) into one canonical
  long-form utility table (496 rows) and scenario-context table (176
  scenarios), with an explicit machine-readable learnable-feature
  allowlist/forbidden-field denylist, full source-row/scenario
  conservation, zero duplicates, deterministic byte-for-byte rebuild, and
  zero mutation of any frozen source artifact.
- The six-anchor policy matrix is **sparse, not dense** (each family only
  evaluated its own 2 anchors on its own scenarios) — the audit documents
  exactly what Step 2 (unified six-policy utility-matrix evaluation) would
  require to make it dense. **This task did not build the dense matrix,
  train a selector, or run any composition/synthesis experiment** — data
  unification only, per explicit task scope.
- Next roadmap step (**not started**): Step 2, unified six-policy
  utility-matrix evaluation (see audit §M/§Q for the exact evaluation list).

## Most Recently Completed Work (WS-P / Policy Separation)

**Family B v2 prefill/decode TTFT-contention refinement is COMPLETE.**

- Audit: [`../audits/policy_separation_prefill_decode_pilot_v2_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v2_20260817.md)
- Provenance: [`../../experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/`](../../experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/)
- Family verdict: **`FAMILY_B_COMPOSITION_READY`**
- Two anchors only (`full_prefill` vs `chunked_prefill_small`): 16/15 practical wins at ε=0.01, near-tie 3.1% (v1 was 96%), mean \|Δ\|=0.131, seed agree 0.875, held-out seed bidirectional, mechanism = class TTFT.
- Frozen Family B v1 remains `USEFUL_BUT_NEEDS_REFINEMENT` / `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED` ([`../audits/policy_separation_prefill_decode_pilot_v1_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v1_20260817.md)); do not rewrite that CSV.
- Next WS-P step: **smallest two-parent PrefillControl composition falsification** (not GP / MAP-Elites / LLM synth). Do not run it as part of the v2 audit.

**PrefillControl composition falsification (`full_prefill` vs `chunked_prefill_small`) is now COMPLETE.**

- Audit: [`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md)
- Provenance: [`../../experiments/prefill_control_composition_v2_20260817T154633Z/`](../../experiments/prefill_control_composition_v2_20260817T154633Z/) (32 scenarios, train=16/val=8/test=4/ood=4, 120/120 success)
- Verdict: **`SELECTION_SUFFICIENT_FOR_THIS_PAIR`**
- A real TRAIN/VAL-fitted contextual top-1 selector reaches the two-parent oracle envelope exactly (0 regret) on both TEST and OOD. The genuinely per-step-dynamic `prefill_control_child` policy (verified not to collapse to any fixed baseline) never beats that selector and never expands the oracle envelope on held-out data. Symbolic distillation / broader module composition / MAP-Elites are **not** justified from this pair alone — see the audit's mechanism analysis for why a different per-step rule remains untested, not falsified.

**Family C v1 KV-pressure reserve pairwise-separation pilot (new mechanism family) is now COMPLETE.**

- Design: [`../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md)
- Audit: [`../audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md)
- Provenance: [`../../experiments/kv_pressure_pilot_v1_20260817T162650Z/`](../../experiments/kv_pressure_pilot_v1_20260817T162650Z/) (32 scenarios, 64/64 success)
- Parents: `kv_constrained_online` (soft KV-occupancy admission reserve) vs `least_laxity_first` (KV-blind laxity-greedy)
- Verdict: **`KV_FAMILY_USEFUL_NEEDS_REFINEMENT`** (5/6 gates pass: bidirectional wins 9-vs-4/32, mechanism activates 28,695 logged deferrals, no twin; tie-rate gate 59.4% did not clear its <50% bound)
- **This is the first family (of ESTF/WFS, PrefillControl, KV-pressure) to demonstrate genuine within-scenario mechanism opportunity**, not just a scenario-level contrast: KV-constrained's advantage over LLF on urgent-tenant SLO attainment is 2× larger when urgent tenants arrive after KV pressure has built up vs before (0.125 vs 0.0625 mean ANWG delta, matched cells) — exactly the structural precondition ESTF/WFS and PrefillControl lacked.
- **This is a pairwise-separation pilot only — no composition work was started or is currently justified.** Next step is refining this family (larger pilot to test whether the tie-rate gate clears with more power), not a composition falsification and not MAP-Elites/GP/distillation/LLM synthesis.

**Family C v2 KV-pressure reserve refinement is now COMPLETE — `KV_FAMILY_COMPOSITION_READY`.**

- Design: [`../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md)
- Audit: [`../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md)
- Provenance: [`../../experiments/kv_pressure_pilot_v2_20260817T165053Z/`](../../experiments/kv_pressure_pilot_v2_20260817T165053Z/) (72 scenarios, 144/144 success; v1's frozen run untouched)
- v1's tie-rate gap (59.4%) diagnosed to two root causes: coarse ANWG resolution at the v1 population size, and an accidental confound where bulk "background" tenants were themselves often classified urgent by the policy's own threshold. v2 fixed both (population roughly doubled; bulk slack recalibrated) and added a third arrival-phase level — all changes justified against the diagnosis, not tuned toward a preferred outcome (design doc §1-2 documents the full reasoning, including a case where a further "fix" was tried and rejected because it didn't change the qualitative picture).
- **All 10 preregistered gates pass**, including two new ones beyond v1's set: G6 (the within-scenario timing pattern replicates on 2 held-out seeds never used in any calibration decision — it does, at comparable-or-larger magnitude) and G10 (6 of 16 matched scenario cells show a *different practical winner* depending purely on when urgent tenants arrive within the same scenario, holding everything else fixed).
- **This is the first family, of the three studied, to reach `_COMPOSITION_READY`** — stronger motivating evidence for composition than ESTF/WFS or PrefillControl v2 produced, neither of which ever showed a within-scenario-timing dependency (both were already `SELECTION_SUFFICIENT_FOR_THIS_PAIR`, meaning a scenario-level selector was sufficient).
- **Important precision (audit §S):** this shows the *scenario-level optimal parent choice* depends on within-trajectory timing, and that a scenario-level selector alone therefore has less headroom to be sufficient here than in the other two families — it does **not** yet prove a state-dependent child would beat *both* fixed parents on the *same* trajectory. That is exactly what a composition falsification would test.
- **No composition work was started in that task**, per explicit scope. The audit stated what the smallest next composition falsification would look like without running it.

**KV-aware composition falsification v1 is now COMPLETE — `KV_COMPOSITION_INCONCLUSIVE`.**

- Design: [`../design/KV_COMPOSITION_FALSIFICATION_V1.md`](../design/KV_COMPOSITION_FALSIFICATION_V1.md)
- Audit: [`../audits/kv_composition_falsification_v1_20260817.md`](../audits/kv_composition_falsification_v1_20260817.md)
- Provenance: [`../../experiments/kv_composition_falsification_v1_20260817T172446Z/`](../../experiments/kv_composition_falsification_v1_20260817T172446Z/) (72 scenarios, 576/576 success)
- The child (`KVAdaptiveReserveChildPolicy`) delegates every step, unmodified, to `kv_constrained_online` or `least_laxity_first`, chosen from a single online-observable trigger (count of currently-waiting urgent-classified requests ≥ a TRAIN/VAL-fit `tau_urgent`). No new admission logic.
- **This is a qualitatively different outcome from ESTF/WFS and PrefillControl v2's `SELECTION_SUFFICIENT_FOR_THIS_PAIR` verdicts** — 6/8 gates pass with real signal (positive TEST envelope gain, 5/12 TEST scenarios beat both parents by >ε, genuine non-degenerate within-trajectory mode-switching on 24/36 held-out scenarios, directionally-consistent OOD replication), but **G7 (safety) fails**: on 6/36 (16.7%) held-out scenarios the child's peak KV utilization exceeds `max(parent peak utilizations)` by 0.013-0.033 — a composition-specific risk (mode-switching history creates KV states neither pure parent alone reaches) that a pairwise-separation pilot structurally cannot surface. Per the frozen decision rule, G7 failing forces `KV_COMPOSITION_INCONCLUSIVE` regardless of G1-G6.
- **Independent, important finding surfaced during this task's cross-checks (not part of any gate):** re-running the original, unmodified KV v2 pilot runner in the current environment reproduces itself perfectly (0/144 mismatch across independent reruns) but does **not** reproduce the historical frozen KV v2 CSV (99/144 rows mismatch, up to 0.25 ANWG). This falsification's own gates remain valid (all methods compared were computed from one internally-consistent run). **Forensic follow-up complete:** see below.
- **Per task scope, this outcome does not license escalating to a more complex child, MAP-Elites, or synthesis.** The smallest defensible next step (not started) would be a narrowly-rescoped child adding a transition-aware admission cap, re-run through the identical frozen procedure.

**KV v2 reproducibility forensic audit is COMPLETE — `REPRODUCIBILITY_GAP_BOUNDED`.**

- Audit: [`../audits/kv_v2_reproducibility_forensic_20260817.md`](../audits/kv_v2_reproducibility_forensic_20260817.md)
- Root cause **not demonstrated**. Ruled out/narrowed: code drift (zero diff on the entire KV v2 execution path between the historical launch commit `6be526e` and current HEAD), runtime/multiprocessing nondeterminism (current environment is byte-identical-SHA-256-reproducible across independent reruns and across `--workers 1` vs `--workers 4`), and both locally available BurstGPT dataset files (neither reproduces the historical CSV; their derived sampling pools are nearly but not exactly identical — filtered `[1024,3072)` pool length 7335 vs 7337 — demonstrating the pipeline's sensitivity to even a 2-row pool difference without pinning down which file, if either, the historical run actually used).
- Historical mismatch is scientifically material, not bit-level noise: 99/144 cells differ (max `|ΔANWG|`=0.25), and the practical (ε=0.01) parent winner flips on 17/72 (24%) scenarios.
- **Three questions kept explicitly separate:** (1) exact historical KV v2 reproducibility is weakened; (2) the composition falsification's internal validity is **not** weakened — every method it compares was evaluated in one single current-environment run; (3) any future cross-run comparison against the historical v2 numbers requires caution.
- **No historical CSV, verdict, or audit conclusion was rewritten.** `KV_FAMILY_COMPOSITION_READY` (v2) and `KV_COMPOSITION_INCONCLUSIVE` (composition falsification) both stand as originally recorded, with this document added as a standing provenance caveat.
- **Forward-looking guard added:** `scripts/run_policy_separation_kv_pressure_pilot_v1.py` now records additive provenance (git SHA/dirty, config+dataset SHA-256, library versions, result-CSV SHA-256, timestamp) in every future run's `final_summary.json` — behavior-neutral, 14 new focused tests pass, does not affect scenario generation, RNG order, or metrics.

**ESTF↔WFS minimal composition falsification remains COMPLETE.**

- Audit: [`../audits/estf_wfs_composition_falsification_v1_20260816.md`](../audits/estf_wfs_composition_falsification_v1_20260816.md)
- Provenance: [`../../experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/`](../../experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/)
- Verdict: **`SELECTION_SUFFICIENT_FOR_THIS_PAIR`**
- Contextual rank composition does not beat contextual top-1 on TEST; parent
  envelope gain is 0. Symbolic distillation / MAP-Elites / LLM synthesis are
  **not** justified from this pair alone.

Family A v2 Job 1182377 remains validated complementary-parent evidence
(`USEFUL_BUT_NEEDS_REFINEMENT`):
[`../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md).

Family A v1 Job 1182306 remains frozen diagnostic evidence
(`USEFUL_DIAGNOSTIC_ONLY` / `REDESIGN_REQUIRED`; historical CSV `anwg` =
unweighted SLO-success, not canonical ANWG):
[`../audits/policy_separation_fairness_starvation_pilot_v1_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v1_20260816.md).

## Latest Major Result (Apt-Serve/CC thread)

**Apt-Serve Phase G completed.**

- Collection: complete.
- Posthoc analysis: complete with wrapper `exit_code=0`.
- Canonical collection output:
  `results/apt_serve_phase_g_resume_20260807_174028/`.
- Preserved failed SS15 source run:
  `results/apt_serve_phase_g_overnight_20260807_011542/`.
- Canonical analysis output:
  `results/apt_serve_phase_g_analysis_20260809_190000/`.
- Audit:
  [`../audits/apt_serve_phase_g_analysis_20260809.md`](../audits/apt_serve_phase_g_analysis_20260809.md).

Supported interpretation:

- The Phase G dataset is structurally valid.
- Apt-Serve has positive leave-one-out marginal contribution to the policy
  portfolio: mean `0.025219`, grouped bootstrap CI `[0.004099, 0.057757]`.
- Global Apt-vs-best-fixed superiority is not established: mean gap
  `0.012032`, grouped bootstrap CI `[-0.013237, 0.046700]`.
- The best fixed baseline by mean ANWG is `scorpio_style_slo_guard`.
- Apt-Serve is one evaluated external scheduler family and a potential source
  of cache/tier-transition modules, not the whole project.

## Current Project Position

- CC0-CC5: complete; CC5 remains `COMPLETE_REGIME_SPECIFIC`.
- CC6: not started; requires explicit authorization and a scoped design.
- External baselines: current status is centralized in
  [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md).
- Apt-Serve: Phase G analysis is complete; no new Apt-Serve collection job is
  queued.
- WS-P: Family A v2 analyzed; ESTF↔WFS composition =
  `SELECTION_SUFFICIENT_FOR_THIS_PAIR`; Family B (the next mechanism family
  after ESTF/WFS) v1 is `USEFUL_BUT_NEEDS_REFINEMENT` /
  `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`; v2 is `FAMILY_B_COMPOSITION_READY`;
  PrefillControl composition falsification on the v2 pair = `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
  ([`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md));
  Family C v2 KV-pressure reserve refinement = `KV_FAMILY_COMPOSITION_READY`
  ([`../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md));
  v1 pilot remains `KV_FAMILY_USEFUL_NEEDS_REFINEMENT` (frozen, superseded
  by v2, not rewritten). The KV-pressure composition falsification on that
  pair is complete: `KV_COMPOSITION_INCONCLUSIVE`
  ([`../audits/kv_composition_falsification_v1_20260817.md`](../audits/kv_composition_falsification_v1_20260817.md))
  — real envelope-gain signal, blocked specifically by a composition-induced
  KV-safety gate failure, not by absence of signal.

## Exact Next Tasks (two independent threads)

1. **WS-P:** Family B v2 analysis is complete
   ([`../audits/policy_separation_prefill_decode_pilot_v2_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v2_20260817.md)).
   Verdict `FAMILY_B_COMPOSITION_READY`. ESTF↔WFS composition pilot verdict:
   `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
   ([`../audits/estf_wfs_composition_falsification_v1_20260816.md`](../audits/estf_wfs_composition_falsification_v1_20260816.md)).
   PrefillControl composition falsification (`full_prefill` vs
   `chunked_prefill_small`) is COMPLETE, verdict
   `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
   ([`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md)).
   **Family C v2 KV-pressure reserve** (`kv_constrained_online` vs
   `least_laxity_first`) reached `KV_FAMILY_COMPOSITION_READY`
   ([`../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md);
   design [`../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md))
   — the first of the three families studied to justify a composition
   falsification. **That falsification has since been run to completion:**
   `KV_COMPOSITION_INCONCLUSIVE`
   ([`../audits/kv_composition_falsification_v1_20260817.md`](../audits/kv_composition_falsification_v1_20260817.md);
   design [`../design/KV_COMPOSITION_FALSIFICATION_V1.md`](../design/KV_COMPOSITION_FALSIFICATION_V1.md)).
   A minimal state-dependent child (delegates every step, unmodified, to one
   of the two frozen parents based on an online-observable urgent-queue-depth
   trigger) showed real signal — positive TEST envelope gain, 5/12 TEST
   scenarios beating both parents by >ε, genuine non-degenerate
   within-trajectory mode-switching on 24/36 held-out scenarios,
   directionally-consistent OOD replication — but the frozen safety gate
   (G7) failed: on 6/36 held-out scenarios the child's peak KV utilization
   exceeded `max(parent peak utilizations)`, a composition-specific risk
   (mode-switching history creates KV states neither pure parent alone
   reaches) that a pairwise-separation pilot cannot surface. Per the frozen
   decision rule this forces `KV_COMPOSITION_INCONCLUSIVE` regardless of the
   otherwise-favorable G1-G6 results. **Do not** escalate to a more complex
   child, MAP-Elites, symbolic distillation, or LLM synthesis from this
   result — per its own audit §Z, the only defensible next step (not
   started) is a narrowly-rescoped child adding a transition-aware admission
   cap, re-run through the identical frozen procedure. **Separately,** this
   task surfaced an unresolved reproducibility gap in the whole KV v1/v2
   evidentiary chain (audit §P) — the current environment cannot reproduce
   the historical frozen KV v2 CSV bit-for-bit even by re-running the
   original unmodified runner; root cause not identified, flagged for a
   dedicated follow-up.
2. **Apt-Serve/CC:** Perform the post-Phase-G module-envelope interpretation and
   decide the next module-decomposition/compositional-learning step.

## Do Not Do By Default

- Do not claim Apt-Serve globally beats the best fixed baseline.
- Do not treat Apt-Serve as the project endpoint.
- Do not start CC6 without explicit authorization.
- Do not delete Phase G artifacts or historical negative-result audits.
- Do not start MAP-Elites, selector retraining, or broad synthesis from PSD yet.
- Do not train selectors on Family A v1 rows.
- Do not rewrite Job 1182306 CSV rows.
- Do not use local `results/` absence as proof an experiment never ran; check
  the audit trail.

## Navigation

- Public overview: [`../../README.md`](../../README.md)
- Research roadmap: [`../PROJECT_MAP.md`](../PROJECT_MAP.md)
- Detailed status: [`WORK_STATUS.md`](WORK_STATUS.md)
- Prioritized next actions: [`NEXT_ACTIONS.md`](NEXT_ACTIONS.md)
- External-baseline index: [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md)
- Documentation index: [`../README.md`](../README.md)
