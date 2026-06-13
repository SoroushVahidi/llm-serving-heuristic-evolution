# Roadmap

## Phase 1 (current) — Baseline Infrastructure

- [x] Deterministic iteration-level simulator
- [x] Synthetic workload generators (Poisson, bursty, heavy-tail)
- [x] 8 classical baseline policies
- [x] Oracle policy (hindsight SRTF, non-deployable)
- [x] Evaluation metrics (latency, P95/P99, SLO violation, throughput, utilization)
- [x] Experiment runner with YAML configs
- [x] Result tables and figures
- [x] Unit tests
- [x] Documentation

## Phase 2 — LLM-Generated Restricted Policy Code

- [ ] Define a restricted policy DSL (whitelisted Python subset)
- [ ] Integrate Cohere / CloudRift API calls for code generation
- [ ] Build a sandboxed executor for LLM-generated policies
- [ ] Implement automated feasibility checking before execution
- [ ] Initial LLM-generated policy experiments on Phase 1 traces
- [ ] Compare LLM-generated policies against all Phase 1 baselines

## Phase 3 — Verifier and Sandbox

- [ ] Formal constraint verifier: check that generated policies respect all capacity
  constraints by static analysis or exhaustive small-trace testing
- [ ] Sandboxed execution environment (resource limits, import restrictions)
- [ ] Automated test suite for generated policies
- [ ] Rejection / repair mechanism for invalid generated policies

## Phase 4 — LLM-Evolution Loop

- [ ] Iterative prompt engineering: use metric feedback to guide re-generation
- [ ] Evolutionary search over policy code (mutation, crossover via LLM)
- [ ] Population-based optimization across multiple seeds
- [ ] Integration with Cohere Command R+ and CloudRift inference endpoints
- [ ] Track Pareto front: latency vs. SLO violation rate vs. throughput

## Phase 5 — Shifted Workload Evaluation and Paper Write-up

- [ ] Evaluate evolved policies on out-of-distribution workloads (shifted arrival rates,
  different length distributions)
- [ ] Compare against production vLLM baselines (requires additional validation)
- [ ] Ablation studies on policy complexity vs. performance
- [ ] Write-up: problem formulation, method, simulator, results, limitations
- [ ] Reproducibility package: frozen traces, policy code, result logs
- [ ] Submission target: MLSys, OSDI, or similar venue

## Known gaps before Phase 2

1. **Prefill cost**: Phase 1 treats prefill as free; realistic serving includes a
   non-trivial prefill step proportional to prompt length.
2. **KV cache paging**: No eviction/preemption means long-running requests can hold
   KV capacity indefinitely; real systems use block-level KV managers.
3. **Throughput saturation**: In Phase 1, decode throughput is constant regardless
   of batch size; real GPUs slow down under memory pressure.
4. **Real workload traces**: Phase 1 uses synthetic traces only.
5. **LLM policy interface**: The generated policy API needs to be frozen and
   documented before Phase 2 begins.
