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
  ([`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md)).

## Exact Next Tasks (two independent threads)

1. **WS-P:** Family B v2 analysis is complete
   ([`../audits/policy_separation_prefill_decode_pilot_v2_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v2_20260817.md)).
   Verdict `FAMILY_B_COMPOSITION_READY`. ESTF↔WFS composition pilot verdict:
   `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
   ([`../audits/estf_wfs_composition_falsification_v1_20260816.md`](../audits/estf_wfs_composition_falsification_v1_20260816.md)).
   PrefillControl composition falsification (`full_prefill` vs
   `chunked_prefill_small`) is now COMPLETE, verdict
   `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
   ([`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md)).
   Next on this thread: select the next mechanism family / parent pair per
   the roadmap (`../PROJECT_MAP.md`). Do **not** start MAP-Elites, symbolic
   distillation, LLM synthesis, or QD work on the strength of this result —
   two independent pairs (ESTF/WFS and PrefillControl) have now both landed
   `SELECTION_SUFFICIENT_FOR_THIS_PAIR`, not `COMPOSITION_GO`.
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
