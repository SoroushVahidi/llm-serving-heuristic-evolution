# Algorithm Stress-Test Library — Catalog Summary

Human-readable rendering of
`configs/stress_tests/algorithm_stress_test_catalog.yaml` (the
machine-readable source of truth — this document summarizes it, and can
go stale; regenerate the table below with the snippet at the bottom
rather than hand-editing it out of sync). 29 entries across 12
algorithms as of 2026-08-05 (Sarathi-Serve section added this date, 7
new entries; 22 entries existed before).

| stress_test_id | algorithm_id | role | evidence_class |
|---|---|---|---|
| `fifo_target_homogeneous_low_contention` | fifo | TARGET | DOCUMENTED_LIMITATION |
| `fifo_counter_head_of_line_blocking` | fifo | COUNTER | PAPER_MOTIVATING_STRESS_CASE |
| `sof_target_mixed_lengths_accurate_prediction` | shortest_output_first | TARGET | PROVEN_WORST_CASE |
| `sof_counter_long_job_starvation` | shortest_output_first | COUNTER | PROVEN_WORST_CASE |
| `estf_target_accurate_alpha_beta_estimate` | estimated_service_time_first | TARGET | PROVEN_WORST_CASE |
| `estf_counter_reasoning_prompt_length_misprediction` | estimated_service_time_first | COUNTER | PROVEN_WORST_CASE |
| `wsp_target_priority_length_balance` | weighted_shortest_processing | TARGET | PROVEN_WORST_CASE |
| `wsp_counter_priority_service_time_conflict` | weighted_shortest_processing | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME |
| `edf_target_feasible_heterogeneous_deadlines` | edf | TARGET | DOCUMENTED_LIMITATION |
| `edf_counter_domino_effect_transient_overload` | edf | COUNTER | PROVEN_WORST_CASE |
| `llf_target_service_time_heterogeneity` | least_laxity_first | TARGET | DOCUMENTED_LIMITATION |
| `llf_counter_laxity_instability_under_prediction_error` | least_laxity_first | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME |
| `priority_target_bounded_high_priority_load` | aging_priority | TARGET | DOCUMENTED_LIMITATION |
| `priority_counter_continuous_high_priority_starves_low` | aging_priority | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME |
| `scorpio_target_overload_selective_admission` | scorpio_style_slo_guard | TARGET | DOCUMENTED_LIMITATION |
| `scorpio_counter_false_rejection_near_threshold` | scorpio_style_slo_guard | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME |
| `selector_target_in_distribution_regime` | regression_anwg | TARGET | INTERNAL_EMPIRICAL_FINDING |
| `selector_counter_out_of_distribution_regime_shift` | regression_anwg | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME |
| `vllm_ltr_target_predictive_prompt_semantics` | vllm_ltr | TARGET | DOCUMENTED_LIMITATION |
| `vllm_ltr_counter_reasoning_domain_shift` | vllm_ltr | COUNTER | PAPER_MOTIVATING_STRESS_CASE |
| `pars_target_alpaca_style_instruction_prompts` | pars_semantic_reference | TARGET | DOCUMENTED_LIMITATION |
| `pars_counter_reasoning_domain_shift` | pars_semantic_reference | COUNTER | INTERNAL_EMPIRICAL_FINDING |
| **`sarathi_counter_long_prompt_moderate_output`** | sarathi_faithful | COUNTER | EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE |
| **`sarathi_target_active_decode_plus_arriving_prefill`** | sarathi_faithful | TARGET | EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE |
| **`sarathi_counter_prefill_heavy_burst`** | sarathi_faithful | COUNTER | EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE |
| **`sarathi_counter_mixed_prompt_lengths`** | sarathi_faithful | COUNTER | EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE |
| **`sarathi_target_kv_pressure`** | sarathi_faithful | TARGET | EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE |
| **`sarathi_counter_short_prompt_decode_dominated_regime`** | sarathi_faithful | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME |
| **`sarathi_counter_long_context_attention_recompute`** | sarathi_faithful | COUNTER | PAPER_MOTIVATING_STRESS_CASE |

## The new `evidence_class` value

`EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE` (added 2026-08-05, catalog
header comment) is a strictly stronger evidentiary tier than
`INTERNAL_EMPIRICAL_FINDING` — reserved for claims backed by actual GPU
hardware execution of the real system(s) involved (here: the 5-trial
repeated Wulver A100 validation of real Sarathi-Serve vs. real vLLM
0.24.0), not a simulator-only internal finding. Used only for the 5
Sarathi entries directly mirroring that validation; see
`docs/wulver_sarathi_vllm_repeated_validation.md` for the underlying
data and `COVERAGE_MATRIX.md` for how (and how much) that real-hardware
backing does and does not carry over into this simulator's own
acceptance gates for those same 5 entries.

## See also

- [`COVERAGE_MATRIX.md`](COVERAGE_MATRIX.md) — per-algorithm rollup,
  plus the full Sarathi-Serve detail table (real hardware / simulator
  support / local-GPU / Wulver-requirement / commit-fidelity columns).
- [`SARATHI_MECHANISM_CALIBRATION_20260805.md`](SARATHI_MECHANISM_CALIBRATION_20260805.md) —
  why the 5 real-hardware Sarathi entries' gates test a coarser
  mechanism than the one real hardware validated.
- [`SARATHI_COMMIT_DRIFT_20260805.md`](SARATHI_COMMIT_DRIFT_20260805.md) —
  whether the Wulver validation's commit matches `sarathi_faithful.py`'s
  pinned commit (it doesn't, exactly — classified `MECHANISM_LEVEL`).
- [`../../audits/sarathi_stress_test_catalog_completion_20260805.md`](../../audits/sarathi_stress_test_catalog_completion_20260805.md) —
  the full completion record for this task.

## Regenerating the table above

```python
import yaml
d = yaml.safe_load(open("configs/stress_tests/algorithm_stress_test_catalog.yaml"))
for e in d["stress_tests"]:
    print(f"| `{e['stress_test_id']}` | {e['algorithm_id']} | {e['test_role']} | {e['evidence_class']} |")
```
