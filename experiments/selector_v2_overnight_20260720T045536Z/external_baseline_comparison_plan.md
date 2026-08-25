# External-baseline comparison plan (Selector v2 contention validation pilot)

Per architecture-native Protocol C (docs/external_baseline_integration.md), the
trained selector prototype is compared ONLY against policies in its own valid
topology class unless resource/topology normalization is explicitly applied.

## Monolithic selector comparison (the selector's own class)
- vllm_faithful
- vllm_chunked_prefill_faithful
- sarathi_faithful
- historical monolithic baselines (fifo, edf, scorpio_style_slo_guard, admission_control, weighted_shortest_processing, estimated_service_time_first, best_fit, multi_bin_batching)
- per-scenario oracle

## Disaggregated reference comparison (reported separately, NOT as a flat baseline)
- distserve_faithful
- tetriinfer_paper_reimplementation

## Migratory reference comparison (reported separately, NOT as a flat baseline)
- llumnix_faithful

These non-monolithic policies are structurally incompatible with the monolithic
selector's own topology and are NOT run head-to-head against it as if they were
peers; they are reported under their own architecture-native Protocol C configs
for context only, per docs/external_baseline_integration.md.
