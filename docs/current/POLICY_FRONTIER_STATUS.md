# Policy Frontier Status

Current status as of the Query 2 integration pass on 2026-07-21.

## Running Workflows

Two frontier-oriented workflows remain active and protected:

| Workflow | Root | Status |
| --- | --- | --- |
| Policy Frontier Cartography and Adversarial Discriminative Workload Mining | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z` | broad sweep array running; targeted sweep and downstream reports pending |
| Policy Library v2 Expanded Frontier | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z` | expanded frontier array running; downstream combine/model/report jobs pending |

Do not modify these roots while active jobs or dependency chains remain in SLURM.

## Expected Outputs

Policy Frontier Cartography should eventually answer:

- where WSP beats SCORPIO;
- where SCORPIO beats WSP;
- whether other policies dominate both;
- whether frontier boundaries are simple, fragmented, feature-limited, or data-coverage-limited;
- whether targeted boundary augmentation helps.

Policy Library v2 Expanded Frontier should eventually answer:

- whether the 7 new policies expand the oracle envelope;
- which new policies produce meaningful unique wins;
- whether expanded library selectors improve practical held-out performance;
- whether policy-library incompleteness is still a bottleneck.

## Current Scientific Status

No final conclusions should be drawn for these two workflows until their final reports exist. Current committed docs should label both as running.
