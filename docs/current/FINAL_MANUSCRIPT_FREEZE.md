# Final Manuscript Freeze

## Identity

- Title: When Does LLM-Serving Scheduler Adaptation Matter? Action Opportunity and Causal Headroom in Production-Derived Replay
- Branch: release/peva-v1-20260920
- Source commit before freeze: a9574da757a828a58619dd0d52aaf5a62d11bc5e
- Canonical source: paper/performance_evaluation/main.tex
- Canonical PDF: paper/when_does_llm_serving_scheduler_adaptation_matter.pdf

## Frozen Artifact Hashes

- PDF SHA-256: beceddddc845675ff05dfe2da9c4546f9f2aff67c49213aa15647fc3a2e01d94
- Source SHA-256: a28aeccb210182eb3554238303dc534adfdc4951cb2b1f202f3c8a41cff5742f
- PDF size: 470857 bytes

## Manuscript Counts

- Page count: 30
- Figure count: 4
- Table count: 6
- Reference count: 34
- Claim-manifest count: 62
- Abstract length: 243 words
- Keywords: 6

## Readiness State

- Literature positioning ready: YES
- Contribution statement ready: YES
- Related Work ready: YES
- Manuscript frozen: YES
- Canonical PDF ready: YES
- Submission package current: NO
- Release package current: NO
- Zenodo v1.1 published: NO

## Validation

- Clean LaTeX build completed from paper/performance_evaluation/main.tex.
- Claim manifest: 62 claims, 0 failures.
- Relevant manuscript tests: 111 passed.
- No undefined citations or references were detected in the final build log.

## No-New-Experiment Confirmation

- Simulator runs: NONE
- New counterfactual runs: NONE
- vLLM runs: NONE
- New workload generation: NONE
- New baseline evaluation: NONE

Only rebuilds, hashes, citation/reference checks, tests, and inspection of existing artifacts were performed.

## Known Scientific Limitations

- Reference-scheduler dependence.
- Simple six-policy diagnostic portfolio.
- Exact-output-length information.
- Loose deadlines and uniform priority.
- Lightly loaded arrival-scaling configurations.
- BurstGPT fresh-window null result under the evaluated construction.
- Uncalibrated simulator.
- One-step oracle scope.
- Concentrated opportunity and finite statistical support.
- Single bounded vLLM correspondence probe.
- Version-controlled and pre-specified, but not externally preregistered.

## Stale Manuscript and Package Material Removed

- paper/performance_evaluation/submission/
- paper/performance_evaluation/submission_docs/
- release/peva_submission_final/
- release/performance_evaluation_v1_1_0/
- release/performance_evaluation_v1_1_0.zip

## Packaging Still Required

- Regenerate the Elsevier submission package from the frozen source.
- Regenerate the v1.1.0 release package from the frozen source.
- Prepare and publish/update the Zenodo v1.1.0 archive only after integration steps authorize it.

## Zenodo State

- Concept DOI: 10.5281/zenodo.22865293
- Published v1.0 DOI: 10.5281/zenodo.22865294
- Invalid/fake DOI checked and absent from live repository references: YES
- v1.1.0 DOI/archive: not yet published in this query.
