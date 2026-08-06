# Llumnix — Official Artifact, Scientific, Implementation, Stress-Test, and Compute-Requirement Audit

**Date:** 2026-08-06
**Scope:** Audit only. No repository files were read-write-touched at the time this audit was produced; no branches switched; no commits; no pushes; no Wulver jobs submitted; no other baseline begun. Originally produced as a standalone research pass, then independently re-verified and integrated into the repository during the Query 2 reconciliation pass of the 2026-08-06 project-pause sequence (see `docs/audits/project_pause_reconciliation_query2_20260806.md`).
**Isolation:** Performed strictly read-only against this repository on branch `contextual-compositional-heuristics-20260731`, originally in parallel with a separate, unrelated Apt-Serve Wulver probe task.

**Integration note (added during Query 2 re-verification):** every load-bearing claim below was independently re-checked against the current repository before this document was committed: `src/llmserveopt/policies/llumnix_faithful.py` exists and is registered in `external_baselines_registry.py`; `tests/test_llumnix_faithful_scheduler.py` collects 36 tests; running `pytest tests/test_llumnix_faithful_scheduler.py tests/test_external_baseline_integration.py` passes 188/188; `docs/current/BASELINES.md` already documents `llumnix_faithful` correctly; `docs/BASELINE_STATUS.md`'s Llumnix row was confirmed stale (last touched by an unrelated Sarathi commit, `74c7fc8`, predating nothing Llumnix-related — the row has been wrong since `4bb54b5` landed on 2026-07-18) and is corrected as part of this same reconciliation (see `docs/BASELINE_STATUS.md`). No comparative evaluation of `llumnix_faithful` exists anywhere in this repository as of this commit — that remains the exact next action.

---

## 1. Repository and active-job state (captured, not modified)

| Field | Value |
|---|---|
| Branch | `contextual-compositional-heuristics-20260731` |
| Local SHA (at original audit time) | `f967c095826900aed0eb0326d3d1f3ea60936261` |
| Upstream SHA | `f967c095826900aed0eb0326d3d1f3ea60936261` |
| Ahead/behind | 0 / 0 |
| Working tree | Clean — no staged, unstaged, or untracked files |
| Active tmux sessions | None |
| Active repo-related processes | None found |

Per `docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md` (the tip of the branch at the time of this audit), the Apt-Serve Strategy C Wulver probe is prepared but blocked on Wulver authentication — no SLURM job was ever submitted, so there was no live conflict to avoid with this parallel, unrelated Llumnix audit in the first place.

---

## 2. The authoritative Llumnix artifact

| Field | Value |
|---|---|
| Paper title | "Llumnix: Dynamic Scheduling for Large Language Model Serving" |
| Authors | Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, Wei Lin (Alibaba Group) |
| Venue/year | 18th USENIX OSDI 2024 |
| arXiv | 2406.03243 (v1 only, submitted 2024-06-05) |
| DOI | 10.5555/3691938.3691948 (ACM DL cross-reference) |
| Artifact-evaluation badges | **All three**: Artifacts Available, Artifacts Functional, **Results Reproduced** (confirmed against the official OSDI'24 AEC results table; OSDI'24 had 34 submissions, 34/31/25 across the three badges respectively — Llumnix cleared the full bar) |
| Official repo | `github.com/alibaba/llm-scheduling-artifact` |
| License | Apache-2.0 |
| Pinned commit | `a90824307249573f9c7548645c22994c65f83a08` — verified live, still HEAD of `main`, `pushed_at: 2024-06-05T13:11:12Z`, zero commits since. 65 stars, 8 forks, 0 open issues/PRs, no tags/releases. |

**Repository landscape — three genuinely distinct codebases exist, and this matters:**

1. **`alibaba/llm-scheduling-artifact`** — the frozen OSDI Artifact-Evaluation snapshot (the pin above). A direct fork of vLLM (traceable back to vLLM v0.1.7/v0.2.0-era commits from 2023-07-26) with Llumnix's changes applied straight into the `vllm/` tree, not a separate importable package. **This is the only repository that is "the paper."**
2. **`llumnix-project/llumnix-ray`** ("v0", Ray-based) — what used to live at `AlibabaPAI/llumnix`; GitHub has since renamed/transferred that org/repo. 564 stars, 50 forks, one tagged release (v0.1.0, 2024-12-05). Its last real code commit was 2025-09-03; the only commit since (2026-03-12) is a docs-only "upgrade to v1" pointer. Functionally dormant for ~11 months as of this audit, despite the highest star count of the three.
3. **`llumnix-project/llumnix`** ("v1", cloud-native rewrite) — created 2026-02-24, first release v0.1.0 on 2026-04-22. Latest commit 2026-05-26. Also quiet for the ~2.5 months leading up to this audit. 37 stars, 6 forks.

**Determination:** neither of the two continuously-branded "Llumnix" repos (v0/v1) is the OSDI 2024 artifact, and both have themselves gone quiet. This project's existing pin (§8) correctly targets repo #1 directly. The exact version corresponding to the OSDI paper is unambiguous and independently confirmed: `alibaba/llm-scheduling-artifact @ a908243`.

---

## 3. Commit-drift analysis

Because the pinned artifact repository has had zero commits since the day of the arXiv submission, drift analysis against "current main of that repo" is trivially zero. The only drift question that is scientifically meaningful is: *artifact repo vs. the currently-branded "Llumnix" projects (v0/v1)*.

- **`alibaba/llm-scheduling-artifact` vs. `llumnix-project/llumnix-ray` (v0):** different repositories entirely (not a fork/branch relationship in the git-history sense) — v0 is its own maintainers' description "a better choice for local deployments," a parallel continuation, not the artifact's next commits. Scheduler-relevant surface not independently diffed line-by-line in this pass.
- **`alibaba/llm-scheduling-artifact` vs. `llumnix-project/llumnix` (v1):** explicitly, per the projects' own documentation, "a new architecture designed to be more modular and cloud-native" — a different codebase, not a refactor of the same one.

**Classification: UNRESOLVED for artifact-vs-v0/v1** — not because evidence is missing, but because the two things being compared are not the same lineage. **EXACT_PAPER_IMPLEMENTATION for artifact-repo-vs-paper** (frozen same day as the arXiv submission, three official AE badges including Results Reproduced).

This project's `llumnix_faithful.py` (see §8) was built by reading the pinned artifact repo's exact source files at the pinned commit, not from v0/v1 — so it inherits the "EXACT_PAPER_IMPLEMENTATION" pin and is unaffected by the v0/v1 drift question.

---

## 4. Reproducibility audit

| Requirement | Value (from the artifact repo) |
|---|---|
| OS | Not separately pinned beyond Docker base image |
| Python | vLLM v0.1.7/v0.2.0-era (Python 3.8–3.10 range typical for that vLLM generation) |
| CUDA / PyTorch | Whatever vLLM v0.1.7/v0.2.0 required (2023-era: CUDA 11.8/12.1, PyTorch ~2.0–2.1) — not independently re-verified against a live pip/conda resolution in this pass |
| vLLM | Two pins: v0.1.7 for the main experiments, v0.2.0 specifically for the migration benchmarks |
| Ray | Required (cluster orchestration for multi-instance dispatch/migration) |
| Other | NCCL/gloo (migration backend, configurable via `migrate_backend`); no FlashAttention/Triton pin found beyond whatever vLLM v0.1.7/v0.2.0 bundles |
| Cluster topology (paper scale) | 4 nodes × 4 NVIDIA A10 GPUs = 16 GPUs |
| Expected setup + runtime | Docker build + Ray bring-up; per-figure scripts: Fig.10 (migration efficiency) ~1h, Fig.11 (serving performance) ~40h, Fig.13 (priorities) ~1h, Fig.14 (auto-scaling) ~42h; AE evaluators were advised to budget ~4 days for a full run; a lighter path (pre-recorded Alibaba-OSS log downloads + plotting only) exists as a fallback to full reruns |

**Provided:** Docker (single `Dockerfile`, nvidia-docker, host networking/IPC sharing), `requirements.txt`/`requirements-dev.txt` (no conda file), per-figure scripts with matching `./plot/` subdirectories, pre-recorded historical logs as an alternative to full reruns. **Not confirmed in this pass:** presence of unit/smoke tests distinct from the figure-reproduction scripts (would require a clone per §16 to check).

**Classification: FULLY_REPRODUCIBLE.** Evidence: the paper independently earned all three OSDI'24 Artifact Evaluation badges, including Results Reproduced, from the AEC itself (external, adversarial verification, not a self-report). Docker + per-figure scripts + pre-recorded expected-output logs are all present.

**Hardware note:** the paper-era vLLM v0.1.7/v0.2.0 stack predates Blackwell-generation GPU support by roughly two years of CUDA/PyTorch releases. This project's local RTX 5060 Ti (Blackwell, confirmed via `nvidia-smi`: 16311 MiB) is essentially certain to be incompatible with that stack for real execution — the same CUDA/Blackwell blocker pattern this project's Sarathi-Serve, Apt-Serve, and VTC audits independently found (`docs/audits/sarathi_official_artifact_audit_20260805.md`, `docs/audits/apt_serve_official_artifact_audit_20260805.md`). Not independently re-verified by attempting an install in this pass — already established three times over for structurally identical vLLM-fork-era artifacts.

---

## 5. Architectural analysis

This project already produced a detailed, file-cited architecture analysis on 2026-07-18 (`docs/llumnix_faithful_scheduler_reference.md`, 445 lines, built by reading `vllm/core/request_scheduler.py`, `vllm/core/scheduler.py`, `vllm/engine/llm_engine_manager.py`, `vllm/engine/llm_engine.py`, `vllm/instance_info.py`, `vllm/worker/cache_engine.py` at the pinned commit, not from memory). This audit independently re-confirmed the pin is unchanged (§2/§3), so that analysis remains current. Summary:

**Classes and files:**
- `LLMEngineManager` (`vllm/engine/llm_engine_manager.py`, 714 lines) — top-level Ray/asyncio orchestrator; owns the periodic migration-trigger check and executes approved migration pairs. Default `generate_mode='callback'`.
- `RequestScheduler` (`vllm/core/request_scheduler.py`, 674 lines) — a Ray remote actor; the cluster-level scheduler. Owns per-instance `InstanceInfo`, initial dispatch policy, migration-pair selection, auto-scaling.
- `Scheduler` (`vllm/core/scheduler.py`, 558 lines) — one per instance ("Llumlet"); a near-verbatim vLLM v0.1.x local scheduler: FCFS prompt admission bounded by token/seq budgets, then swap-based preemption (not recompute).
- `LLMEngine` (`vllm/engine/llm_engine.py`, 852 lines) — per-instance; `migrate_out`/`migrate_in` (GPU-to-GPU block transfer via `cache_engine.send_gpu_cache`/`recv_gpu_cache`).
- `InstanceInfo`/`InstanceLoadInfo` (`vllm/instance_info.py`) — the load-metric formulas.

**Flow:**
```
request arrival
  → dispatch ('naive' default: round-robin on FIRST request of a session,
     sticky to the same instance for all subsequent requests in that session;
     load-oblivious by design)
  → per-instance local FCFS scheduler (swap-based preemption when out of KV blocks)
  → periodic cluster-level instance-load monitoring
     (every need_migrate_frequency=4 scheduling rounds; 'consumed_speed' metric:
      -1 * free_gpu_blocks / num_running_requests, more negative = more loaded)
  → migrate-out/migrate-in candidate detection
     (migrate-out: num_killed_request>0 OR load>threshold(1.5);
      migrate-in: num_killed_request==0 AND load<threshold, same threshold both sides)
  → pairing (i-th most-loaded out ↔ i-th least-loaded in) + benefit-projection check
     (reject pairs that wouldn't actually improve balance, using the last-scheduled
      request's block footprint as the assumed migrated-request size)
  → LCFS source-candidate selection (default strategy: scan running list from the
     END; first request with output_len()>0 [decoding phase only] AND non-priority)
  → destination admission check (reject outright if destination has ANY local
     waiting-queue backlog; else KV-capacity + seq-count check)
  → whole-block-table GPU-to-GPU KV transfer (request-level, not chunked;
     NCCL/gloo depending on migrate_backend; "multi-stage" overlap variant is a
     transfer-mechanism optimization only, not a scheduling-semantics change)
  → resumed execution on destination (decode continues; decoded-token count and
     admission_time preserved across the move)
```

**Priority/SLO:** `priority_type` flag exists; priority requests are never migration sources under the default LCFS path. A more elaborate `enable_load_control_prefill` variant (reserved block quotas, SLO-aware load formula) exists but is non-default and not part of the paper's headline mechanism.

**Fragmentation:** handled implicitly by the load-balancing migration mechanism — the pinned source defines no separate, dedicated fragmentation metric or trigger. This is a property of the paper itself, not an omission in this project's reading of it.

---

## 6. Scientific claim audit

- **Primary objective:** dynamic, load-aware rescheduling (live migration) across independent LLM-serving instances, to relieve queuing-delay/preemption hotspots and improve resource utilization without disrupting in-progress generation.
- **Strongest contribution:** a near-zero-overhead live-migration mechanism for LLM inference, exploiting the append-only property of KV cache, combined into one scheduler with dispatch and auto-scaling.
- **Workloads/baselines/metrics:** production-style traces; vLLM (no migration) and other static-dispatch configurations as baselines; latency (P99), request-completion/preemption counts, throughput/utilization. Exact numeric results were not re-extracted in this pass — that would require a full re-read of the paper's results tables, out of scope for an artifact/reproducibility/architecture audit.
- **Documented limitations (independently verified from the arXiv HTML full text, not a secondary summary):**
  - No dedicated "limitations" section and no quantified sensitivity analysis of migration frequency (`need_migrate_frequency`) or the load threshold (`migrate_out_threshold`) — a documented absence, not a proven bound.
  - Bandwidth dependence is only implicit: the paper's migration-efficiency argument leans on tensor-parallelism staying intra-node; it does not characterize what happens if that assumption breaks (cross-node TP, lower-bandwidth links, larger models).
  - The 64-instance scalability result replaces real GPU execution with a `sleep` command — that specific data point is a scheduling-only simulation, not a real-GPU migration benchmark at that scale (explicitly disclosed by the authors).
  - Two explicit future-work items: (a) no exploration of global/local scheduling interplay; (b) no multi-model-type support.
- **No proven worst case exists** in the paper or in later literature for Llumnix's migration mechanism specifically — every "migration cost exceeds benefit," "oscillating load," "topology asymmetry" scenario is a hypothesized adversarial regime relative to the primary literature, not a paper-motivating stress case (the paper doesn't construct any of these itself) and not a proven worst case (this project's own stress-test catalog does not exercise them yet either — see §12).

---

## 7. Follow-up literature (2024–2026)

| Paper | Venue/Year | Relationship to Llumnix | Code |
|---|---|---|---|
| DistServe (arXiv:2401.09670) | OSDI 2024 | Contemporaneous alternative: disaggregates prefill/decode onto separate GPU pools rather than live-migrating across homogeneous instances. Already implemented in this project as `distserve_faithful` — a genuinely different mechanism, frequently used as a contrast point for Llumnix in surveys. | `LLMServe/DistServe`, public |
| Mooncake (arXiv:2407.00079) | FAST 2025, Best Paper | Generalizes "move state, not just requests" into a global KVCache-as-storage-tier architecture (production system behind Kimi/Moonshot AI). Broader scope than Llumnix's per-instance migration. | `kvcache-ai/Mooncake`, public |
| BlitzScale (arXiv:2412.17246) | OSDI 2025 | Extends live-migration-style thinking to autoscaling specifically: fine-grained layer-level migration/offload during scale-out, explicitly positioned against coarser instance-level approaches (the class Llumnix belongs to). | `blitz-serving/blitz-scale`, public |
| STAR (arXiv:2510.13668) | HPDC 2025/26 (DOI 10.1145/3806645.3807813) | Direct extension of the "rescheduling" idea to decode-phase-specific, length-prediction-driven rebalancing in PD-disaggregated settings — a scenario the original Llumnix design does not target. | Not confirmed public |
| BanaServe (arXiv:2510.13223) | Software: Practice & Experience, 2026 | Unifies KV-cache migration and layer-weight migration under one "Global KV Cache Store" — thematically a generalization of Llumnix's single-mechanism (KV-only) migration; abstract does not explicitly cite Llumnix. | Not confirmed public |

No paper found that runs an explicit, direct adversarial/negative-result stress test against Llumnix's migration mechanism specifically.

---

## 8. Existing repository work audit (verified from code/tests/git history, not from status docs alone)

**This is the single most important finding of this audit: Llumnix is far more built-out in this repository than "audit only," and the project's own primary status index was stale about it.**

Verified via `git log`, direct file reads, and `pytest`:

| Artifact | Path | Status |
|---|---|---|
| Policy implementation | `src/llmserveopt/policies/llumnix_faithful.py` (385 lines) | Implemented 2026-07-18, commit `4bb54b5` |
| Reference/provenance doc | `docs/llumnix_faithful_scheduler_reference.md` (445 lines) | Same commit; file-level, line-cited against the pinned artifact source |
| Tests | `tests/test_llumnix_faithful_scheduler.py` | 36 tests. Re-verified during Query 2 integration: `pytest tests/test_llumnix_faithful_scheduler.py tests/test_external_baseline_integration.py` → **188 passed, 0 failed** (36 Llumnix-specific + 152 cross-baseline integration, run together since the Llumnix tests include regression checks against 5 other faithful baselines). Covers the migration primitive, dispatch, migration-pair selection, LCFS candidate selection, destination admission, end-to-end completion, determinism, regression against 5 other faithful baselines, and paper-level qualitative sanity checks (migration relieves a hotspot vs. static round-robin). |
| Registry | `src/llmserveopt/policies/external_baselines_registry.py` | `llumnix_faithful` registered, `FidelityClass.FAITHFUL`, `TopologyClass.MULTI_INSTANCE_MIGRATORY`, pin string embedded in code (`alibaba/llm-scheduling-artifact commit a90824307249573f9c7548645c22994c65f83a08`) |
| Config generator | `src/llmserveopt/evaluation/external_baseline_configs.py` | `multi_instance_migratory_config()` / `"llumnix_faithful"` branch present |
| New shared simulator primitive | `Action.migrate`, `Simulator._relocating`, `RequestPhase.RELOCATING`, `ObservableGPUState.incoming_migrations`, `GPUState.evict(preserve_progress=True)` extension, `GPUState.admit(is_relocation=True)` | All implemented — a genuinely new, fourth action verb distinct from `swap`, with an explicit, documented rationale for why the DistServe/TetriInfer bridge-queue primitive was not reused |
| Status in `docs/current/BASELINES.md` (updated 2026-07-23) | "Execution-health clean" | Correct — reflects that tests pass |
| Status in `docs/BASELINE_STATUS.md` (as it read before this reconciliation) | "Not integrated... Reference doc only... Unverified in this pass" | Was incorrect/stale — corrected as part of this same Query 2 pass (see `docs/BASELINE_STATUS.md`) |
| Stress-test catalog | `configs/stress_tests/algorithm_stress_test_catalog.yaml`, `docs/research/algorithm_stress_tests/STRESS_TEST_CATALOG.md` | Zero Llumnix entries in either. |
| Stress-test inventory (distinct doc) | `docs/research/algorithm_stress_tests/ALGORITHM_INVENTORY_20260805.md`, row 17 | Notes the existing `llumnix_faithful.py`, classifies representability as "Full" — this is candidate-identification, not catalog coverage. |
| Comparative evaluation | (searched `experiments/`, `docs/audits/`, `results/`) | **None found.** Unlike VTC, PARS-Serve-2026, and Sarathi-Serve — each of which has a dedicated comparative-evaluation audit doc with headroom-gated sweeps and independent re-verification — Llumnix has never been run against other policies in a scored comparison. "Execution-health clean" means it runs without crashing, not that it has been benchmarked. |

**Classification: FAITHFUL_SIMULATOR_IMPLEMENTATION.** Implementation and fidelity-test coverage are complete; comparative evaluation is a genuine open gap (not a documentation problem — no evaluation exists at all yet, correct or otherwise).

---

## 9. Simulator-compatibility analysis

**FULLY_REPRESENTABLE (already built and unit-tested):**
- Multiple serving instances (N independent `role=None` GPUs, one per Llumnix instance)
- Per-instance queues (`GPUState._active`)
- Cross-instance placement/dispatch (naive round-robin, session-sticky, via `_assign_instance`)
- Request migration (`Action.migrate`, `Simulator._relocating`, `RequestPhase.RELOCATING`)
- KV-cache migration (whole block-table transfer, re-checked against the destination's own `KVBlockSpaceManager` at transfer-completion time — a genuine two-stage design matching the pinned source, not an approximation)
- Migration latency/delay (configurable `llumnix_migration_delay`)
- Priority isolation for migration-source exclusion (`priority_exempt_threshold`)
- Dynamic load (an equivalent of `'consumed_speed'`)

**PARTIALLY_REPRESENTABLE:**
- **Migration bandwidth** — modeled only as a flat, configurable delay scalar, not a function of KV bytes transferred, actual link bandwidth, or contention.
- **Concurrent transfers / migration concurrency** — `_relocating` can hold multiple in-flight migrations simultaneously, but every transfer proceeds at the same fixed delay regardless of how many are concurrent; there is no shared-link-contention model.
- **Memory fragmentation** — handled only in aggregate (via `KVBlockSpaceManager` utilization), with no dedicated external-fragmentation metric — this matches the paper's own scope exactly, so it is a faithful limitation, not a simulator shortfall.

**NOT_REPRESENTABLE:**
- Topology / real network contention (multi-node interconnect asymmetry, NCCL-vs-gloo backend differences) — no topology model exists anywhere in this simulator.
- Cost-minimization / elastic auto-scaling — excluded by explicit design decision, matching every other faithful baseline in this project.
- Fault/failure recovery (Ray actor fault tolerance) — infrastructure concern, out of scope by established project convention.
- Non-default dispatch/migration strategies (`'unbalanced'`/`'balanced'`/`'load'`/`'block'` dispatch; `'SJF'`/`'LJF'` migration-candidate order; global `FFIT`/`FCFS`/`BE`/`SJF`/`LJF` modes) — deliberately unimplemented; only the verified default path (`'naive'` + `'LCFS'` + `'consumed_speed'`) exists.

**Required extensions to close the two closeable partial gaps** (both localized, small-to-medium scope):
1. Replace the flat `llumnix_migration_delay` scalar with a byte-size-aware function (`delay = kv_bytes / bandwidth + fixed_overhead`) in `ServiceModel`.
2. A per-link (or per-node) migration-bandwidth budget consumed by simultaneous in-flight transfers, checked in `Simulator._apply_migrations`.

Estimated scope, by this project's own historical conventions (e.g., FairBatching was estimated "~150 LOC + tests"): similarly small, likely 1–2 focused work sessions each. **Not implemented in this audit — audit only.**

---

## 10. Integration-strategy decision

**Official real-system track:** the pinned artifact is a monolithic vLLM fork (Llumnix's changes applied directly into the `vllm/` tree, not a separately importable package) — structurally the same class of artifact as Sarathi-Serve (this project's own prior finding: "no standalone algorithmic component to adapter-wrap"), and the opposite of VTC (whose `VTCReqQueue` is a standalone, dynamically-importable class this project already runs unmodified). This makes Strategy B (official-code-with-thin-adapter) implausible for the same structural reason it was ruled out for Sarathi.

**Simulator track:** already exists (§8/§9) as an independent, line-cited reimplementation of the exact default algorithmic path — this is Strategy D (FAITHFUL_INDEPENDENT_REIMPLEMENTATION), but only its "reimplementation" half. The "PLUS_OFFICIAL_VALIDATION" half — actually running the real `alibaba/llm-scheduling-artifact` code (e.g., on Wulver, Docker-based) to cross-check the simulator's migration-benefit numbers — has not happened. This is the same gap Sarathi-Serve closed with a real Wulver A100 repeated-trial comparison (`docs/wulver_sarathi_vllm_repeated_validation.md`); Llumnix has no equivalent.

**Dynamic-import feasibility:** LOW (monolithic fork, not a package — same reasoning as Sarathi).
**Source-vendoring legality:** Apache-2.0 permits vendoring with attribution + license notice; safe to clone into an isolated, non-project cache for read-only inspection; this project's own convention (Sarathi, VTC) is to run official code externally (Docker/Wulver) rather than vendor it into the repo, and that convention transfers cleanly here.

**Preliminary classification: D (current state) + A (recommended complement).** Complete the reimplementation-plus-validation pattern already used for Sarathi: keep `llumnix_faithful.py` as the simulator-track baseline, and add a Docker+Wulver real-artifact two-GPU migration run as the real-system validation track. Do not call the current simulator-only implementation "the official baseline" in any manuscript claim without that validation step, or without an explicit "faithful reimplementation, not executed official code" qualifier (matching this project's existing safe-wording conventions for `sarathi_faithful`/`distserve_faithful`).

---

## 11. Local workstation versus Wulver

| Capability | Where |
|---|---|
| Source inspection | Local CPU — done (this audit + the 2026-07-18 reference doc) |
| Unit tests (36, all pass) | Local CPU — pure-Python discrete-event simulator, no GPU needed regardless of scale |
| Environment build (real artifact) | Would need Wulver — local Python is 3.12 (per this project's Apt-Serve finding: vLLM 0.5.x-era wheels only ship for cp38–cp311; the same version-mismatch pattern likely applies to Llumnix's earlier vLLM v0.1.7/v0.2.0 pins too) |
| Single-instance smoke (real artifact) | Wulver single GPU |
| Two-instance migration (real artifact) | Wulver multi-GPU (same node) — the critical missing validation step (§10) |
| Paper-scale reproduction (Fig.11/14, 16 GPUs) | Wulver multi-node — only needed if full paper-scale fidelity is itself a claim being made |
| Network-sensitive experiments (bandwidth/topology stress) | Wulver multi-node, and only meaningful once the simulator-side bandwidth extension (§9) exists |

**Local Blackwell GPU (RTX 5060 Ti, 16GB) compatibility with the paper-era environment: NOT compatible** — same conclusion this project has already reached three times for structurally similar 2023-era vLLM-fork artifacts (Sarathi, Apt-Serve, VTC).

**Recommended SLURM configurations** (following this project's own established conventions — `account=ikoutis`, `qos=standard`, `partition=general` for CPU-only, `partition=gpu` + `--gres=gpu:a100:N` for GPU; none of these are submitted):

| # | Purpose | Partition | Nodes | GPUs | CPUs | RAM | Wall time | Expected output |
|---|---|---|---|---|---|---|---|---|
| 1 | CPU/environment probe | `general` | 1 | 0 | 8 | 16G | 00:45:00 | Docker/pip environment build log; confirm whether a prebuilt wheel resolves for vLLM v0.1.7/v0.2.0 at whatever Python version Wulver's default module provides |
| 2 | One-GPU import/smoke | `gpu` | 1 | 1× a100 | 8 | 32G | 00:30:00 | Single vLLM+Llumnix engine boots inside the official Docker image, serves one request; no migration exercised |
| 3 | Two-GPU same-node migration | `gpu` | 1 | 2× a100 | 16 | 64G | 01:00:00 | Reduced-scale reproduction of the paper's Fig.10 migration-efficiency experiment; real migration triggered and measured, cross-checked against `llumnix_faithful.py`'s `llumnix_migration_delay` assumption |
| 4 | Multi-node migration (only if full paper-scale fidelity becomes a scientific requirement) | `gpu` | 4 | 16× a100 (4/node) | 64 | 256G | 06:00:00 | Fig.11-equivalent serving-performance numbers at paper scale |

---

## 12. Stress-test library mapping

**Coverage classification: MISSING.** Zero entries for Llumnix in `configs/stress_tests/algorithm_stress_test_catalog.yaml` or `docs/research/algorithm_stress_tests/STRESS_TEST_CATALOG.md`. The only existing artifact is a single candidate-identification row in `ALGORITHM_INVENTORY_20260805.md` (row 17), which is not catalog coverage.

| Regime | Simulator-compatible today? | Wulver required? | Notes |
|---|---|---|---|
| Target: persistent multi-instance imbalance | Yes — `test_rescheduling_load_hotspot_triggers_migration`/`test_sanity_imbalanced_arrivals_across_instances` already probe this at unit-test level | No | Nearly promotable to a catalog entry as-is |
| Target: memory fragmentation | Only as aggregate KV pressure (per §9) | No | Catalog entry must be honest it tests aggregate pressure, not block-level external fragmentation |
| Target: heterogeneous SLO isolation | Partially — priority-exemption logic exists, no heterogeneous-SLO-class generator | No | Gap: workload generator |
| Target: priority acceleration | No dedicated mechanism (priority only excludes migration-source selection, matching the paper) | No | Not a coverage gap — should be marked N/A |
| Target: sustained imbalance with favorable migration payback | Partially — `test_sanity_migration_reduces_hotspot_vs_static_round_robin` is the closest existing (qualitative) evidence | No | Needs promotion to a quantitative, headroom-gated entry |
| Target: cost-minimizing placement | Not representable — no cost-minimization mechanism exists in Llumnix's default path at all | N/A | Should be marked N/A for this baseline specifically |
| Counter: rapidly oscillating load | Partially — `need_migrate_frequency` knob exists, no adversarial-oscillation generator | No | Gap: generator |
| Counter: short-lived imbalance | Partially — same trigger-frequency knob | No | Gap: generator |
| Counter: large KV state, low bandwidth | Partially only — no true bandwidth model (§9) | No (Wulver would strengthen evidence class, not enable it) | Needs the §9 bandwidth extension first |
| Counter: simultaneous migration contention | Not representable yet — no shared-link-contention model (§9) | No | Needs the §9 concurrency-budget extension first |
| Counter: migration cost > remaining service | Yes — comparable today via `llumnix_migration_delay` vs. predicted remaining service time | No | Cheapest gap to close — no generator built yet |
| Counter: overreaction to noisy metrics | Yes, with synthetic noise injection into `instance_load` reporting (not built) | No | Gap: generator |
| Counter: control-loop delay | Yes — directly `need_migrate_frequency` | No | Cheapest gap to close — no generator built yet |
| Counter: topology asymmetry | Not representable — no topology model at all | Would require real hardware | Out of scope until a topology model exists |
| Counter: tiny requests, migration overhead dominates | Yes — short `predicted_output_tokens` vs. fixed delay | No | Cheapest gap to close — no generator built yet |

**Evidence class today, for all rows: NONE (0 entries generated).**

---

## 13. Evaluation plan (future tracks — not executed)

**Track 1 — official real-system Llumnix:** Docker image from the pinned artifact; models/traces as specified by the artifact's own Fig.10/11 configs; start at 1 node × 2 A100 GPUs (§11 config 3); migration policy = default; compared against vLLM with migration disabled; ≥3 seeds; raw per-request completion logs + migration event logs as artifacts; independent re-verification of any headline number before citing it.

**Track 2 — simulator/mechanism-level migration policy:** `llumnix_faithful` vs. the existing deployable policy set plus `static_round_robin_test` (promoted from a test-only helper to a real named policy); workloads = the canonical suite plus new imbalance/fragmentation/oscillation generators from §12; multi-instance configs (2, 3, 4, 8 GPUs) via `multi_instance_migratory_config`; ≥3 seeds; metrics: TTFT, TPOT, throughput, ANWG, completion rate, SLO attainment, per-instance queue-imbalance trace, migrations count, migration duration, migration success/failure rate, migration payback, preemptions, tail latency (P95/P99).

**Track 3 — algorithm-specific stress tests:** the §12 target/counter matrix, generator-by-generator, each independently re-verified before being called "validated."

---

## 14. Implementation roadmap (future work — not started)

| Phase | Content | Files/LOC (est.) | Wulver | Runtime | Risk |
|---|---|---|---|---|---|
| A | Official artifact/environment probe (Docker build + Ray bring-up on Wulver) | 0 repo LOC (external) | CPU/env probe (§11 #1) | <1h | Low eng, low sci |
| B | Fix `docs/BASELINE_STATUS.md`'s stale Llumnix row | ~1 row | None | Minutes | None — **completed as part of this Query 2 reconciliation** |
| C | Official two-instance real-migration smoke (Wulver, §11 config 3) | 0 repo LOC (external) | 2-GPU probe | ~1h wall | Medium eng (auth/env), low sci |
| D | Simulator bandwidth + concurrency-contention extension (§9) | ~150–300 LOC + tests | None | N/A | Low-medium eng, low sci |
| E | Stress-test generators for the §12 matrix (cheapest-first: control-loop delay, migration-cost-exceeds-benefit, tiny-request-overhead) | ~200–400 LOC | None (all simulator-only) | N/A | Low eng, medium sci |
| F | Comparative evaluation (Track 2) — headroom-gated sweep vs. the deployable policy set | 0 new source, evaluation scripts only | Optional (real-hardware cross-check) | Hours (sweep) | Low eng, medium sci — **this is the actual missing evaluation from §8, and the cheapest of the remaining phases** |
| G | Foundational-library decision | N/A | N/A | N/A | Decision-only; gated on F's actual results |

**Stop conditions:** if Phase A's Docker build fails outright on Wulver's available toolchain, fall back to citing the artifact as CODE_ONLY-with-external-badge-evidence rather than attempting a from-scratch environment reconstruction.

---

## 15. Security, license, and artifact review

- **Artifact repo license:** Apache-2.0 (confirmed live). No redistribution restriction beyond standard attribution/license-notice requirements.
- **This project's own license:** MIT (`LICENSE`, root) — compatible, no conflict for citing/comparing against Apache-2.0 work.
- **Models/datasets:** none cloned in this audit.
- **Docker image:** not pulled in this pass.
- **No credentials, cluster paths, or secrets are recorded here.**

---

## 16. Optional isolated source clone

Not performed in this pass. All provenance facts needed for this audit were obtainable via GitHub API metadata reads without a full clone. If a future pass needs file-content-level drift verification, the recommended target remains an isolated, non-repository cache directory outside version control, pinning `a90824307249573f9c7548645c22994c65f83a08` (paper-era) and current `main` of whichever comparison repo is relevant.

---

## 17. Read-only validation performed

- `git status`, `git log`, `git branch --show-current`, `git rev-parse` — repository-state capture, read-only.
- `grep` across the repository — existing-work discovery, read-only.
- `git log --follow`/`git log -1` on specific files — history/staleness verification, read-only.
- `pytest tests/test_llumnix_faithful_scheduler.py --collect-only -q` — 36 tests collected, zero collection errors.
- **Re-verified during Query 2 integration:** `pytest tests/test_llumnix_faithful_scheduler.py tests/test_external_baseline_integration.py` — 188 passed, 0 failed.
- `nvidia-smi` — local hardware confirmation only.

No files were modified, staged, committed, or pushed by the original audit pass. This document itself is the first commit touching this content.

---

## 18. Central risks

1. **Documentation drift (now resolved):** `docs/BASELINE_STATUS.md`'s Llumnix row was wrong and misleading anyone who followed `docs/INDEX.md`'s "start here" pointer — corrected in this same reconciliation pass.
2. **Validation gap (open):** `llumnix_faithful.py` has never been run in a scored comparison against any other policy, and has no real-hardware cross-check — any manuscript claim about it needs both before it can be treated as "evaluated," not just "implemented and execution-health clean."
3. **Bandwidth/concurrency fidelity gap (open):** two named counter-regimes (large-KV/low-bandwidth, simultaneous-migration-contention) are not genuinely testable until the §9 extensions exist.
4. **Repo landscape confusion risk:** the "Llumnix" name now resolves, by search-engine/star-count popularity, to a dormant v0/v1 fork family that is not the paper — any future contributor googling "Llumnix github" without reading this project's own pinning doc first would likely land on the wrong repository.

---

## Summary

**Central conclusion:** Llumnix is not a green-field baseline for this project — it already has a faithful, line-cited, unit-tested simulator implementation (36 tests, all passing) built against a verified-unchanged, triple-badged (Available/Functional/Results Reproduced) OSDI 2024 artifact. What is actually missing is an evaluation: neither a real-hardware validation run nor a comparative sweep against this project's other policies has ever been performed, despite the code being ready for both. The stress-test catalog has zero entries for it, though most of the target/counter regime matrix is already representable with only two small, localized simulator extensions (migration-bandwidth modeling, concurrent-transfer contention) needed to close the remainder.

**Exact next action:** run the missing Phase F comparative evaluation (`llumnix_faithful` vs. the existing deployable policy set on multi-instance configs) — this is cheap (no new code, no Wulver, no new dependencies) and is the most load-bearing gap identified in this audit. Real-hardware Wulver validation and new stress-test generator work are secondary to it.

**Classification: implementation status COMPLETE; fidelity class FAITHFUL (independent reimplementation, OSDI-badge-verified pin, not executed official code); comparative-evaluation status NOT RUN; foundational-library classification UNESTABLISHED pending that evaluation.**
