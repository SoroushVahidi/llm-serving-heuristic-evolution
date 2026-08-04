# Resume Contextual Composition

Use this file to resume the contextual-compositional heuristic branch.

## Checkout

```bash
git checkout contextual-compositional-heuristics-20260731
```

- Authoritative branch: `contextual-compositional-heuristics-20260731`
- Query 6 checkpoint SHA: `f6b4be9dc15fc4f13286f23b5aae39f48fbd01fb`
- CC2 (Query 8) checkpoint SHA: verify against the CC2 primitive interface
  report's `New SHA` with `git rev-parse HEAD`
- Starting SHA before the CC2 checkpoint commit:
  `4d806c8b1be0c4c9e202bbc7a20b3455c9c510b8`
- CC3 (Query 9) checkpoint SHA: verify against the CC3 DSL/verifier
  report's `New SHA` with `git rev-parse HEAD`
- Starting SHA before the CC3 checkpoint commit:
  `ed85e585bb42a37f47530939b1d2d11bb1ea0b3e`
- CC4 (Query 10) checkpoint SHA: verify against the CC4 oracle dataset
  report's `New SHA` with `git rev-parse HEAD`
- Starting SHA before the CC4 checkpoint commit:
  `19708f741d0bfb944b4a11ff34572a811df94d66`
- CC5 (Query 11) checkpoint SHA: verify against the CC5 predictor
  report's `New SHA` with `git rev-parse HEAD` (CC5's exit gate did NOT
  pass -- this checkpoint is an attempted-and-INCONCLUSIVE checkpoint, not
  a completion checkpoint)
- Starting SHA before the CC5 checkpoint commit:
  `db143fc7aef5cb604ed56b778b948b5d4f271891`
- CC4b/CC5-retry (Query 12) checkpoint SHA: verify against the CC4b/CC5
  retry report's `New SHA` with `git rev-parse HEAD`
- Starting SHA before the CC4b/CC5-retry checkpoint commit:
  `c17208079ef50368103f1feca992ac91f52ff4cb`
- CC5 uncertainty/regime (Query 13) checkpoint SHA: verify against the
  uncertainty/regime report's `New SHA` with `git rev-parse HEAD`
- Starting SHA before the uncertainty/regime checkpoint commit:
  `7718214119e7eff8f242ff974aad00d37063906a`
- CC5 final operating envelope (Query 14) checkpoint SHA: verify against
  the final operating envelope report's `New SHA` with `git rev-parse HEAD`
  (this checkpoint closes CC5 `COMPLETE_REGIME_SPECIFIC` and queues CC6,
  restricted scope)
- Starting SHA before the CC5 final operating envelope checkpoint commit:
  `f5a4f82d54111a656e5f49c554c2b41974de5349`

## Read In Order

1. [START_HERE_CONTEXTUAL_COMPOSITION.md](START_HERE_CONTEXTUAL_COMPOSITION.md)
2. [contextual_composition_roadmap.md](contextual_composition_roadmap.md)
3. [contextual_composition_decisions.md](contextual_composition_decisions.md)
4. [audits/contextual_composition_pause_checkpoint_20260731.md](audits/contextual_composition_pause_checkpoint_20260731.md)
5. [audits/contextual_composition_query5_discriminativeness_review_20260731.md](audits/contextual_composition_query5_discriminativeness_review_20260731.md)
6. [audits/contextual_composition_query7_final_pause_readiness_20260731.md](audits/contextual_composition_query7_final_pause_readiness_20260731.md)
7. [architecture/contextual_composition_primitives.md](architecture/contextual_composition_primitives.md)
8. [audits/contextual_composition_cc2_primitive_interface_report_20260802.md](audits/contextual_composition_cc2_primitive_interface_report_20260802.md)
9. [architecture/contextual_composition_dsl.md](architecture/contextual_composition_dsl.md)
10. [audits/contextual_composition_cc3_dsl_verifier_report_20260803.md](audits/contextual_composition_cc3_dsl_verifier_report_20260803.md)
11. [audits/contextual_composition_cc4_oracle_dataset_report_20260803.md](audits/contextual_composition_cc4_oracle_dataset_report_20260803.md)
12. [audits/contextual_composition_cc5_predictor_report_20260803.md](audits/contextual_composition_cc5_predictor_report_20260803.md)
13. [audits/contextual_composition_cc4b_cc5_retry_report_20260803.md](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md)
14. [audits/contextual_composition_cc5_uncertainty_regime_report_20260803.md](audits/contextual_composition_cc5_uncertainty_regime_report_20260803.md)
15. [audits/contextual_composition_cc5_final_operating_envelope_20260803.md](audits/contextual_composition_cc5_final_operating_envelope_20260803.md) -- **read this one for current status**
16. GitHub issue #6

## Verify State

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse --abbrev-ref --symbolic-full-name @{u}
git rev-list --left-right --count @{u}...HEAD
python scripts/check_contextual_composition_status.py
python scripts/check_contextual_composition_status.py --resume-readiness
python -m pytest tests/test_contextual_composition_status_checker.py tests/test_cc1_composition_opportunity.py tests/test_policy_composition.py tests/test_score_and_reciprocal_rank_composition.py tests/test_primitive_interface.py tests/test_primitive_reconstructed_policies.py tests/test_contextual_composition_cc3_dsl.py tests/test_cc4_oracle_composition_dataset.py tests/test_cc5_contextual_predictor.py tests/test_cc5_uncertainty_regime.py tests/test_cc5_final_operating_envelope.py -q
```

The expected state is a clean working tree, upstream
`origin/contextual-compositional-heuristics-20260731`, `0` ahead and `0`
behind, and a passing contextual-composition status checker.

## Current Phase

- Current phase: `CC6 - Dynamic adaptation and stability`
- Current status: `NEXT` (restricted scope -- CC5 is `COMPLETE`,
  `COMPLETE_REGIME_SPECIFIC`)
- Decision gate: CC4's exit gate passed and CC5 was attempted, retried, and
  finalized across four evidence stages. First attempt (n=6 held-out): the
  predictor tied best fixed policy and was beaten by
  `best_global_composition` -- data-scarcity finding. **Retry** (n=76
  held-out): the predictor clearly beat best fixed policy and was
  competitive with the hard selector, but did not clearly beat
  `best_global_composition` under independent-CI comparison. **Uncertainty/
  regime refinement**: attached model-agnostic conformal uncertainty,
  still `REGIME_SPECIFIC_ONLY`. **Final operating envelope** (this
  checkpoint): a frozen, development-evidence-only envelope (7 of 12
  regimes) evaluated once on held-out data reaches ANWG 0.4044, and a
  **paired** statistical analysis (not independent-CI comparison) shows
  this statistically beats best fixed (p<0.0001) and the hard selector
  (p=0.021), while its edge over `best_global_composition` is **not**
  statistically distinguishable from zero (p=0.5654) -- verdict
  `COMPLETE_REGIME_SPECIFIC`. See the final operating envelope report for
  full evidence.

## Exact Next Implementation Task

CC5 is closed. Per the final operating envelope report: CC6 is now queued,
but **restricted** to the CC5 trusted envelope (`burst_transition`,
`kv_pressure`, `long_output`, `prediction_noise`, `saturated`,
`selective_admission_trap`, `underloaded`). Evaluate controlled temporal
adaptation only inside this envelope, with hysteresis and fallback; do not
enable contextual switching in unsupported regimes
(`azure_conversation_like`, `burstgpt_derived`, `long_prompt`, `mixed_slo`,
`priority_conflict`). Do not begin CC6 implementation until a future query
explicitly authorizes it.

## Do Not Start Prematurely

Do not begin CC6 implementation in this query. When a future query does
begin it, keep it restricted to the CC5 trusted envelope above -- do not
enable contextual switching in unsupported regimes. Do not begin CC7
hardening, CC8 real-serving validation, hosted API jobs, GPU jobs,
real-vLLM jobs, evolutionary/QD library-expansion work, LLM-guided
synthesis work, or other new experiments before the roadmap gates allow
them -- see the roadmap's "Future Research Directions" section for what
remains unimplemented future work, not current capability.

## CC1b Evidence

Primary local result directory:

```text
results/cc1b_composition_discriminative/query5_cc1b_full_20260731/
```

Key files:

- `manifest.json`
- `verdict.json`
- `method_comparison.csv`
- `per_window_summary.csv`
- `composition_weights.csv`
- `cc1_report.md`

Regenerate only if needed:

```bash
python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1b_composition_discriminative.yaml \
  --dry-run

python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1b_composition_discriminative.yaml \
  --full-run
```

Do not use live APIs, GPU jobs, or real-vLLM jobs for this evidence.

## CC4 Evidence

Primary local result directory (untracked, per repository convention --
regenerate via `replay_commands.sh` inside it):

```text
results/cc4_oracle_composition_dataset/20260803T170735Z/
```

Key files: `manifest.json`, `dataset_card.md`, `oracle_labels.parquet`,
`regret_matrix.parquet`, `causal_features.parquet`, `search_summary.csv`.

Regenerate only if needed:

```bash
python scripts/run_cc4_oracle_composition_dataset.py \
  --config configs/cc4_oracle_composition_dataset.yaml \
  --dry-run

python scripts/run_cc4_oracle_composition_dataset.py \
  --config configs/cc4_oracle_composition_dataset.yaml \
  --full-run --allow-dirty --timestamp <new_timestamp>
```

Do not use live APIs, GPU jobs, or real-vLLM jobs for this evidence.

## CC5 Evidence

Primary local result directory (untracked, per repository convention --
regenerate via `replay_commands.sh` inside it):

```text
results/cc5_contextual_composition_predictor/20260803T175456Z/
```

Key files: `manifest.json`, `verdict.json`, `model_card.md`,
`cv_model_selection.csv`, `per_window_predictions.csv`,
`uncertainty_ood_diagnostics.csv`, `fallback_analysis.csv`.

This is the **first-attempt** run only (n=6 held-out, verdict
`INCONCLUSIVE`). The retry against the CC4b-expanded dataset (n=76
held-out, verdict `REGIME_SPECIFIC_ONLY`) lives in
`results/cc5_contextual_composition_predictor_retry/20260803T192246Z/` --
see the "CC4b/CC5 Retry Evidence" section below.

Regenerate only if needed (against the same CC4 dataset used originally):

```bash
python scripts/run_cc5_contextual_predictor.py \
  --dataset-dir results/cc4_oracle_composition_dataset/20260803T170735Z \
  --dry-run

python scripts/run_cc5_contextual_predictor.py \
  --dataset-dir results/cc4_oracle_composition_dataset/20260803T170735Z \
  --full-run --timestamp <new_timestamp>
```

Do not use live APIs, GPU jobs, or real-vLLM jobs for this evidence.

## CC4b/CC5 Retry Evidence

Primary local result directories (untracked, per repository convention):

```text
results/cc4b_oracle_composition_expansion/20260803T182426Z/     # 106 windows, 3604 executions
results/cc5_contextual_composition_predictor_retry/20260803T192246Z/  # verdict REGIME_SPECIFIC_ONLY
```

Config: `configs/cc4b_oracle_composition_expansion.yaml` (generated by
`scripts/generate_cc4b_expansion_config.py`; 76 held-out windows across 10
synthetic regime templates + real-trace variants, reusing CC4's exact
candidate-search config for direct comparability). Quality gates
(`scripts/check_cc4b_quality_gates.py <dataset_dir>`) all passed: 76
held-out (>=50), 35 non-near-tie (>=20), no dominant oracle family (39%
max share, <=70%), completion accounting consistent. See
[audits/contextual_composition_cc4b_cc5_retry_report_20260803.md](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md)
for the full status and verdict.

Do not use live APIs, GPU jobs, or real-vLLM jobs for this evidence.

## GitHub

GitHub issue #5
([#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5))
is **CLOSED** -- CC5 finalized `COMPLETE_REGIME_SPECIFIC`. Continue with
GitHub issue #6
([#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6)),
which is ready but marked restricted scope / not started: evaluate
controlled temporal adaptation only inside the CC5 trusted envelope, with
hysteresis and fallback. Issue #1 is the completed CC1/CC1b evidence gate;
issue #2 is the completed CC2 primitive interface gate; issue #3 is the
completed CC3 DSL/verifier gate; issue #4 is the completed CC4 oracle
dataset gate.


## CC5 Uncertainty / Regime Refinement Evidence

```bash
results/cc5_uncertainty_regime_refinement/20260803T202108Z/
docs/audits/contextual_composition_cc5_uncertainty_regime_report_20260803.md
logs/cc5_uncertainty_regime_20260803_195020.log
```

Superseded by the final operating envelope below (still valid provenance
for the uncertainty-calibration work).

## CC5 Final Operating Envelope Evidence

```bash
results/cc5_final_operating_envelope/20260804T024524Z/
docs/audits/contextual_composition_cc5_final_operating_envelope_20260803.md
logs/cc5_uncertainty_regime_finalization_20260804_024058.log
```

This is the **final, authoritative** CC5 evidence: paired statistical
analysis, the frozen development-evidence-only operating envelope, and the
single held-out evaluation of the frozen system. Verdict
`COMPLETE_REGIME_SPECIFIC`. Continue with GitHub issue #6 (restricted
scope, not started).
