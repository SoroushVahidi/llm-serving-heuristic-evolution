# Algorithm Stress-Test Library — Coverage Matrix

Per-algorithm summary of the catalog's TARGET/COUNTER coverage,
execution status, and (for Sarathi-Serve specifically, per the
2026-08-05 stress-test-catalog-completion task) real-hardware/local-GPU/
Wulver requirements and commit-level fidelity. Generated from
`configs/stress_tests/algorithm_stress_test_catalog.yaml` (29 entries as
of this update); regenerate by hand when the catalog changes rather than
letting this drift silently.

## All algorithms

| Algorithm | Target entries | Counter entries | Executable | Evidence classes used |
|---|---:|---:|---|---|
| fifo | 1 | 1 | 2/2 | DOCUMENTED_LIMITATION |
| shortest_output_first | 1 | 1 | 2/2 | DOCUMENTED_LIMITATION |
| estimated_service_time_first | 1 | 1 | 2/2 | DOCUMENTED_LIMITATION |
| weighted_shortest_processing | 1 | 1 | 2/2 | DOCUMENTED_LIMITATION |
| edf | 1 | 1 | 2/2 | DOCUMENTED_LIMITATION |
| least_laxity_first | 1 | 1 | 2/2 | DOCUMENTED_LIMITATION |
| aging_priority | 1 | 1 | 2/2 | DOCUMENTED_LIMITATION, HYPOTHESIZED_ADVERSARIAL_REGIME |
| scorpio_style_slo_guard | 1 | 1 | 2/2 | (see LITERATURE_RESEARCH_20260805.md) |
| regression_anwg | 1 | 1 | 0/2 (needs trained artifact) | (see LITERATURE_RESEARCH_20260805.md) |
| vllm_ltr | 1 | 1 | 0/2 (offline-scored, no corpus) | DOCUMENTED_LIMITATION, PAPER_MOTIVATING_STRESS_CASE |
| pars_semantic_reference | 1 | 1 | 0/2 (offline-scored, no corpus) | DOCUMENTED_LIMITATION, INTERNAL_EMPIRICAL_FINDING |
| **sarathi_faithful** | **2** | **5** | **6/7** (1 spec-only, structurally NOT_REPRESENTABLE) | **EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE (5), HYPOTHESIZED_ADVERSARIAL_REGIME (1), PAPER_MOTIVATING_STRESS_CASE (1)** |

Totals: 12 algorithms, 29 catalog entries, 22 executable this pass
(6 sarathi + 16 pre-existing), 7 explicitly out-of-scope/non-executable
(all disclosed via `real_system_followup_required`/
`requires_new_scoring_pass`/generator `NotImplementedError`, never
silently skipped).

## Sarathi-Serve detail (added 2026-08-05)

| Entry | Role | Real hardware | Simulator support | Local GPU (RTX 5060 Ti) | Wulver required | Commit-level fidelity |
|---|---|---|---|---|---|---|
| `sarathi_target_active_decode_plus_arriving_prefill` | TARGET | ROBUST (5/5, CI [0.990,1.036]s) | Representable, but mechanism NOT currently distinguishable in-simulator (see SARATHI_MECHANISM_CALIBRATION) | Insufficient (no nvcc, CUDA/Blackwell gap) | Yes, for any further real-hardware work | MECHANISM_LEVEL (see SARATHI_COMMIT_DRIFT) |
| `sarathi_target_kv_pressure` | TARGET | ROBUST (5/5, CI [0.769,0.903]s) | Same as above | Insufficient | Yes | MECHANISM_LEVEL |
| `sarathi_counter_long_prompt_moderate_output` | COUNTER | NOT_REPRODUCED-for-Sarathi = robust vLLM win (5/5, CI [-0.298,-0.213]s) | Same as above | Insufficient | Yes | MECHANISM_LEVEL |
| `sarathi_counter_prefill_heavy_burst` | COUNTER | robust vLLM win (5/5, CI [-0.157,-0.137]s) | Same as above | Insufficient | Yes | MECHANISM_LEVEL |
| `sarathi_counter_mixed_prompt_lengths` | COUNTER | robust vLLM win (5/5, CI [-0.257,-0.161]s) | Same as above | Insufficient | Yes | MECHANISM_LEVEL |
| `sarathi_counter_short_prompt_decode_dominated_regime` | COUNTER | Not tested on real hardware (hypothesis) | Representable, but degenerate at smoke scale (all 4 policies tie -- see catalog `calibration_note`) | Insufficient | Yes, to test the hypothesis at all | N/A (no real-hardware claim made) |
| `sarathi_counter_long_context_attention_recompute` | COUNTER | Paper's own sidestepped regime (not this project's data) | **NOT_REPRESENTABLE** -- no attention-cost scaling term in this simulator's timing model at all | Insufficient (16GB VRAM cannot hold a 32K-context KV cache at usable batch sizes for 7B+ models even if the software gap were fixed) | Yes, and even then needs the real engines, not this simulator | N/A (spec-only entry) |

**Key finding, prominent by design:** the 5 real-hardware entries'
`acceptance_gates` do NOT test the fine decode-protection-vs-shared-
contention distinction real hardware validated -- that distinction was
found to be structurally unreproducible in this simulator under FCFS-
strict admission (`SARATHI_MECHANISM_CALIBRATION_20260805.md`). The
gates instead test a coarser, genuinely-distinguishing mechanism
(chunked vs. non-chunked admission/completion of long prompts), all 5 of
which pass honestly. This is disclosed here rather than implied away by
a passing gate.

## How to regenerate

```bash
python scripts/stress_tests/run_stress_test_smoke.py --json /tmp/full_smoke.json
python scripts/stress_tests/run_sarathi_headroom_check.py
```

Cross-check entry counts against
`python -c "import yaml; d=yaml.safe_load(open('configs/stress_tests/algorithm_stress_test_catalog.yaml')); print(len(d['stress_tests']))"`.
