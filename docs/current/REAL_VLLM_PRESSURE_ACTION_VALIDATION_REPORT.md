# REAL_VLLM_PRESSURE_ACTION_VALIDATION_REPORT

## 1. Infrastructure

The clean audit worktree was pinned to `b196c3e27d1e5a0d43fd0664e24e51090ec8e442`. The dirty primary worktree was not modified. The available execution environment is one NVIDIA GeForce RTX 5060 Ti (16,311 MiB), driver 580.173.02, CUDA 13.0, vLLM 0.27.1, PyTorch 2.13.0+cu130, and Qwen/Qwen2.5-0.5B-Instruct from the local cache. The isolated environment is `/home/soroush/.venvs/vllm_real_validation_v1`.

The repository already contained a working startup/replay harness, streaming TTFT and completion timing, deterministic workload generation, warmup exclusion, scheduler instrumentation, and request-level output. An exact one-step scheduler intervention hook was not present. Prometheus scraping was available in the first local run, but the native budget probe's metric parser produced empty dictionaries; queue/running evidence in that probe therefore comes from scheduler traces.

The standalone preregistration artifact in this report is explicitly retrospective because the scientific runs predated this packaging task. The earlier `workload_design.json`, feasibility design, and run-level integrity records were retained and audited; no new scientific run was launched.

## 2. Preregistration and frozen matrix

The bounded matrix was:

| Validation | Matrix | Measured runs |
| --- | --- | ---: |
| R1/R2 prefill comparison | FULL versus CHUNKED, 4 deterministic contention regimes, 5 repetitions | 40 |
| R1/R2 native budget probe | T512 versus T4096, 2 selected contention regimes, 5 repetitions | 20 |

The workload is `TOKEN_SHAPE_REPLAY`: deterministic token-length-shaped requests representing Azure 2023 code/conversation families, not exact production prompts. Warmups were excluded. The model, server settings, request order, repetitions, metrics, and nearest-observable action signature are frozen in `experiments/real_vllm_pressure_action_validation_v1/PREREGISTRATION_V1.json`.

## 3. Execution and completeness

The existing local run completed 40/40 measured regime runs with 300/300 successful requests, zero request failures, zero prompt-generation mismatches, clean server logs, and no active vLLM servers afterward. The native budget probe completed 20/20 runs with 150/150 successful requests and zero failures. No Wulver job was used for this validation, and no recovery reruns were needed.

## 4. Pressure validation

The run produced real queueing and concurrency pressure: the prefill comparison observed maximum running sequences of 4, maximum waiting sequences of 7, KV usage up to 0.0284, and zero preemptions. In the native budget probe, both treatments reached four running sequences; T512 reached seven waiting sequences versus five for T4096. The probe was valid but did not reach high KV occupancy, so this is a contention/action-pressure validation rather than a high-KV-capacity validation.

## 5. Action-opportunity validation

The nearest observable `REAL_ACTION_SIGNATURE` contains treatment, scheduler step, scheduled token budget, prefill/decode token counts, mixed-prefill/decode flag, partial-prefill count, scheduled request IDs, and waiting/running counts. Native T512 versus T4096 changed scheduler behavior: T512 had 1,680 scheduled steps, 165 mixed prefill/decode steps, and 194 partial-prefill items; T4096 had 1,536 steps, 55 mixed steps, and 9 partial-prefill items. This is clear R2 action differentiation, but it is not an exact mapping to the simulator's canonical action.

## 6. Latency results

The direct FULL versus CHUNKED comparison did not reproduce the simulator's preregistered qualitative reversal. Across the four regimes, the late-class TTFT difference `CHUNKED - FULL` ranged from 0.0161 s to 0.0355 s in the late-tight cases, rather than favoring CHUNKED; the hog-class E2E difference in hog-tight cases was -0.0165 s and -0.0079 s, without the required stable FULL advantage. This sub-experiment was correctly classified `PREFILL_REAL_VALIDATION_NO_GO` for the simulator Family-B correspondence.

The native vLLM budget probe nevertheless showed a reproducible native tradeoff. For `T4096 - T512`, late TTFT was -0.0306 s [ -0.0380, -0.0228 ] in `late_tight_low_late` and -0.0025 s [ -0.0077, 0.0018 ] in `late_tight_high_late`; hog E2E was +0.0163 s [0.0092, 0.0235] and +0.0233 s [0.0156, 0.0318], respectively. These are whole-run native budget effects, not one-step causal effects.

## 7. Controlled causal validation

`REAL_ONE_STEP_CAUSAL_INTERVENTION_IMPLEMENTED = NO`. The harness cannot force one alternative scheduler action while holding the live state, arrivals, random state, request population, and continuation fixed. Changing the complete native prefill budget is useful R1/R2 evidence but is not substituted for R3.

## 8. Simulator/real-system correspondence

| Property | Simulator | Real vLLM | Match |
| --- | --- | --- | --- |
| Low-pressure operation | abundant resources suppress choice | warm, low queue baseline | qualitative |
| Resource contention | KV/active pressure creates disagreement | waiting queue and four active sequences observed | qualitative |
| Action differentiation | canonical SBS/P6 branch | native token-budget and prefill/decode signatures | partial |
| Latency sensitivity | one-step SBS-relative headroom | whole-run TTFT/E2E budget tradeoff | partial, non-causal |
| Exact causal action | controlled branch continuation | unavailable | no |

## 9. Practical magnitude

The native budget effect was tens of milliseconds in late TTFT and hog E2E, larger than the fresh simulator mean headroom of approximately 1.996 ms. However, it is configuration-level native vLLM behavior and should not be interpreted as the simulator's one-step causal headroom transferring numerically.

## 10. Validation verdict

`REAL_SYSTEM_VALIDATION = PARTIAL_SUPPORT`. R1 and R2 are supported by valid real-vLLM execution and scheduler traces. The direct simulator Family-B qualitative prediction was a no-go in this harness, KV telemetry was incomplete in the native probe, and R3 was unavailable. The result supports the narrower claim that real serving resource contention can alter observable scheduling behavior and latency, not the stronger claim of real-system causal action headroom.

## 11. Fourth trace, readiness, and manuscript authorization

`FOURTH_TRACE_REQUIRED = NO`: the existing evidence is sufficient to state the real-system boundary and its limitations; another trace would not repair the missing one-step hook. The updated FGCS contribution readiness is 88/100 with 84% contribution-strength confidence. The real-system credibility gate is partial rather than fully closed. The full manuscript rewrite is authorized, with the real-vLLM result presented as bounded R1/R2 support and an explicit R3 limitation.

## 12. Shortest remaining path

Integrate this bounded real-system result, its no-go/partial-support distinction, and its reproducibility artifacts into the manuscript; then perform final consistency and artifact-packaging checks. Do not add another trace or selector experiment as a substitute for the absent causal hook.

## 13. Result freeze

Compact artifacts are under `experiments/real_vllm_pressure_action_validation_v1/`. Their SHA-256 manifest is `REAL_VLLM_ARTIFACT_HASHES_V1.json`. The source runs and raw logs remain under `experiments/real_vllm_mechanism_validation_v1/`; no prior result artifacts were modified.
