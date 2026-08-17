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
| Last reconciled SHA | see `git rev-parse HEAD` (Family B v2 audit after `ecc0422`) |
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
context -> performance/marginal-contribution learning -> DSL/module selection
        -> verified composition/synthesis -> evaluation -> envelope expansion
```

The current primary metric is `arrival_normalized_weighted_goodput` (ANWG).

Typed DSL / module-composition infrastructure exists in-repo and is valuable,
but it does **not** by itself complete policy-separation decision-boundary
characterization (WS-P).

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
- **No composition work was started in this task**, per explicit scope. The audit states what the smallest next composition falsification would look like (§T) without running it.

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
  by v2, not rewritten).

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
   `least_laxity_first`) is now `KV_FAMILY_COMPOSITION_READY`
   ([`../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md);
   design [`../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md))
   — all 10 preregistered gates pass, including held-out-seed replication
   (G6) and within-scenario winner-flip evidence (G10) neither prior family
   produced. **This is the first family, of the three studied, to justify a
   composition falsification** — but that falsification was **not** run in
   this task (explicit scope). The audit (§T) states what the smallest next
   step should be: a two-parent composition falsification structured like
   the Family B v2 PrefillControl one, with a genuinely state-dependent
   KV-admission child compared against a TRAIN/VAL-fitted scenario-level
   selector on held-out TEST/OOD. Do **not** start it, MAP-Elites, symbolic
   distillation, or LLM synthesis without explicit authorization — a
   `_COMPOSITION_READY` verdict recommends the next falsification, it does
   not launch it.
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
