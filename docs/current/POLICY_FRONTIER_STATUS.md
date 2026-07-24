# Policy Frontier Status

Current status as of 2026-07-22.

## Completed Frontier-Oriented Workflows

| Workflow | Root | Final report | Current interpretation |
| --- | --- | --- | --- |
| Policy Frontier Cartography and Adversarial Discriminative Workload Mining | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z` | `reports/FRONTIER_FINAL_REPORT.md` | Frontier regions exist, but many regions remain near-tie or simulator-limited. Use as historical map evidence, not as a reason to launch broad synthesis immediately. |
| Policy Library V2 Expanded Frontier | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z` | `reports/FINAL_POLICY_LIBRARY_REPORT.md` | Synthetic/frontier expansion was modest; V2 adds useful behavior but does not solve selector/combiner readiness by itself. |
| V2 Real-OOD 27-Policy Library Audit | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/v2_real_ood_library_20260721T222521Z` | `reports/FINAL_REPORT.md` | Strongest positive library result: V2 oracle gain `0.008904` ANWG, about `3.54%` relative, CI `[0.008191, 0.009646]`. |
| SwissAI V2 Policy Sweep | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/swissai_v2_policy_sweep_20260722T184451Z` | No final report due reporting-stage `kv_proxy_p95` failure; complete matrix exists at `combined/policy_vectors.csv`. | Novel KV/cache/reuse features did not create policy separation under current simulator/objective; zero strict V2 marginal gain. |
| TraceLab V2 Policy Sweep | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/tracelab_v2_policy_sweep_20260722T214129Z` | `reports/FINAL_TRACELAB_V2_SWEEP_REPORT.md` | Long-context/agentic/prefix novelty saturated ANWG and produced zero strict V2 marginal gain. |
| SLO/Deadline Augmented V2 Sweep | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/slo_deadline_augmented_v2_sweep_20260722T194529Z` | `reports/FINAL_SLO_DEADLINE_AUGMENTATION_REPORT.md` | Synthetic SLO pressure exposes useful EDF/SCORPIO/admission/laxity signals and partially improves class balance. |

## Current Scientific Status

The frontier evidence now points away from broad policy-library expansion or
generic data collection as the immediate next step. V2 is valuable in real-OOD
policy-vector data, but SwissAI and TraceLab show that raw workload novelty can
fail to translate into reward separation. The current bottleneck is simulator
and objective discriminative power.

## Next Frontier Use

After simulator calibration, rerun small controlled subsets from:

- V2 real-OOD;
- SwissAI;
- TraceLab;
- SLO/deadline augmentation;
- selected frontier/cartography windows.

Use those bounded reruns to verify that resource pressure and policy separation
improve for scientifically defensible reasons before retraining selectors or
resuming module-combination/synthesis work.
