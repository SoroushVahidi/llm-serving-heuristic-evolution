# SBS_OVERRIDE_FRESH_OOD_SUPPORT_EXPANSION_V1

## Purpose

Diagnose why Azure 2024 and Bailian/Qwen produced zero SBS-vs-P6 canonical
action support, and search already staged external traces for natural OOD
action support before any terminal labels are generated.

This stage is outcome-blind. It may inspect source metadata, arrivals, online
request attributes, SBS trajectories, P6 native actions, action equality, and
state structural features. It must not inspect terminal utility under
alternative actions, `Q_SBS`, `A_SBS`, beneficial/harmful labels, learned
selector performance, or closed-loop behavior.

## Duplication Gate

Prior local/Wulver search found related public-trace policy-separation
artifacts but no equivalent SBS-trajectory P6-vs-SBS support manifests for
BurstGPT v2 or Azure 2023. TraceLab has a documented related HF-derived
policy-separation sweep with near-saturated separation, but it is not
equivalent to this SBS support question and should not be reused as fresh
confirmatory evidence without re-deriving windows from raw TraceLab.

## Frozen Candidate Sources

The new OOD support manifest is:

`experiments/sbs_override_fresh_ood_support_expansion_v1/fresh_ood_expansion_candidate_manifest.csv`

It includes:

- BurstGPT v2 from the already frozen LSSP window cache: 40 windows x load
  factors `1.0` and `2.0`.
- Azure 2023 public-trace replay augmented scenarios: Azure conversation and
  code windows x load factors `1.0` and `2.0`.

Selection is based only on source/window identity and preregistered load
factors. No terminal outcomes or selector results are used.

## Choice State Rate

`CHOICE_STATE_RATE` is preregistered as:

`waiting_queue_count >= 2`

Binding-capacity choice is recorded separately as:

`waiting_queue_count > total_free_sequence_slots`

This distinguishes "no meaningful scheduling choice" from "policies agree
despite available choices."

## Native Replay vs Controlled Stress

This run reports `NATIVE_REPLAY_SUPPORT` under the existing project replay and
controlled-annotation semantics. If a source has zero support because overlays
or capacity make the replay unconstrained, that is not fixed here by tuning
synthetic stress parameters. Any future stress-tuned variant must be reported
as `CONTROLLED_STRESS_EXTENSION`, not natural OOD confirmation.
