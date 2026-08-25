# Apt-Serve Strategy C Wulver Probe — 2026-08-06

Phase A of `docs/audits/apt_serve_official_artifact_audit_20260805.md`
§12's implementation roadmap: determine whether Apt-Serve's official
scheduler can be imported and reused as a separable Python component
(Strategy C) rather than defaulting to a from-scratch reimplementation
(Strategy D) without having actually tested C's precondition.

**Status: EXECUTED. Strategy C confirmed viable (with bounded
limitations) from real, executed evidence on Wulver.** The Wulver
authentication blocker recorded below (§0) was specific to the earlier
audit pass's environment and did not recur in this execution pass — a
real interactive Wulver session was available this time, and the full
probe (§5-9) ran to completion, including a working differential
micro-trace against the real, unmodified, patched
`vllm.core.scheduler.Scheduler`. See §5-9 (updated with actual results)
and the new §9b (Strategy C/D decision, task step 9) below. Compact
result artifacts: `results/provenance/apt_serve_strategy_probe/`.

## 0. Repository state at start (task step 1)

- Branch: `contextual-compositional-heuristics-20260731`
- Starting SHA: `db4ba0f72d97abc83ba2de5269ae454221c72ab1`
- Upstream: 0 ahead / 0 behind, working tree clean
- Status checker + resume-readiness: both passed
- `docs/audits/apt_serve_official_artifact_audit_20260805.md` read in
  full; its findings (categorical CUDA/Blackwell blocker for local
  execution, Strategy C left genuinely open but unresolved, Strategy D
  as the interim default) are treated here as hypotheses this probe
  exists to test, not re-asserted.

**Wulver access blocker (discovered this pass, not present in the prior
audit's scope):** SSH to the configured Wulver login node
(`login02` → `login02.tartan.njit.edu`, per `~/.ssh/config`, user
`sv96`) reaches the host (NJIT login banner displays correctly) but
authentication fails: `Permission denied
(gssapi-keyex,gssapi-with-mic,keyboard-interactive)`. `klist` reports
"No credentials cache found" — this environment has no active Kerberos
ticket, and `kinit` requires an interactive password this session
cannot supply non-interactively. **No SLURM job has been submitted.**
Everything in §1-4 and the "prepared, not yet run" material in §5-10
was produced without cluster access, using only GitHub API reads (for
commit/file provenance) and local, non-Wulver dependency-resolution
checks (for the environment plan). This blocker, and what remains once
it is resolved, is the primary content of this document's conclusion.

## 1. Official artifact pin (task step 3)

Re-confirmed via `gh api` (not re-derived from memory of the prior
audit):

| Field | Value |
|---|---|
| Repository | `github.com/eddiegaoo/Apt-Serve` |
| Pinned commit (full SHA) | `c953217988274a761da35cf06c01033b18dadf68` |
| Commit tree SHA | `c7cde79f368f861196f0e5fac5055591f3c23600` |
| Commit date | 2025-06-09T09:29:08Z |
| License | `null` (confirmed again via `gh api repos/eddiegaoo/Apt-Serve --jq .license` — no LICENSE file exists) |

Git blob hashes of the exact files this probe reads/copies, at this
pinned commit (`gh api repos/eddiegaoo/Apt-Serve/contents/<path>?ref=c953217988`):

| File | Git blob SHA | Size |
|---|---|---|
| `additional_designs/core/aptserve_scheduler.py` | `3ecf74f19b820d15dc2f2f1c09d0303df4c3d768` | 98,271 bytes |
| `additional_designs/core/aptserve_block_manager.py` | `7fa580a2b4340b01827097dab45417818087cf84` | 47,973 bytes |
| `additional_designs/core/aptserve_interfaces.py` | `10c8e2e76727a9d877786e65614316c11d9e7eaf` | 3,100 bytes |
| `additional_designs/aptserve_sequence.py` | `cea9a50132a51c969afddbf9bb92b18294f54593` | 33,360 bytes |
| `additional_designs/aptserve_block.py` | `7826377cc5ca9442d25fc2ac190a01386c136914` | 4,063 bytes |
| `additional_designs/mixed_cache_kernels/mixed_cache.cu` | `01ffe4502467d4cbf081d7ddfacbaa1bbbc9df89` | 9,175 bytes (not used this probe — see §2) |
| `additional_designs/insert_designs.sh` | `13cefdc567ebd49aceddc95d97ba36fb5ed8a236` | 1,533 bytes (not applied verbatim — see §2) |

**No upstream source is vendored or redistributed in this repository or
in any file this pass commits** — these are hashes of files that will
be read/copied only inside an isolated Wulver scratch/vendor directory
at job-run time, per the license disclosure above. The job script
(§4) re-verifies these hashes against what it actually copies at
runtime (`sha256sum` of the copied-over vLLM files), so provenance is
checked twice: once here (git blob hash, pre-run), once on Wulver
(sha256 of the files as landed in the isolated env, post-copy).

## 2. Minimum official source subset for this probe (task step 7, partial)

Rather than apply the full 13-file `insert_designs.sh` replacement
(which would pull in the worker/model-executor/attention layers — code
this probe has no need to exercise and that plausibly has harder CUDA
dependencies, muddying the specific question this probe asks), the
prepared job applies **only the 5 files the scheduler class itself
imports or is constructed alongside**:

- `aptserve_scheduler.py` → `vllm/core/scheduler.py` (the class under test)
- `aptserve_block_manager.py` → `vllm/core/block_manager_v1.py` (constructed inside `Scheduler.__init__`)
- `aptserve_interfaces.py` → `vllm/core/interfaces.py` (the `BlockSpaceManager`/`AllocStatus` contract)
- `aptserve_sequence.py` → `vllm/sequence.py` (the `Sequence`/`SequenceGroup` objects the scheduler operates on)
- `aptserve_block.py` → `vllm/block.py` (block/slot data structures referenced by the above)

This is a **deliberate, disclosed deviation** from the official install
procedure, made explicit here (not silently substituted) — it narrows
what "Apt-Serve imports/constructs cleanly" can mean in this probe's
results: success here demonstrates the **scheduling algorithm** is
separable, not that the full serving stack is. The custom
`mixed_cache_ops` CUDA kernel (`mixed_cache.cu`) is deliberately NOT
built in this probe — it is invoked from the worker/cache-engine layer
during actual token generation, never from the scheduler's own
admission/scheduling decisions, so it is out of scope for a
scheduler-separability question specifically (this narrows, but does
not by itself resolve, task step 6's "whether custom CUDA kernels are
required" — this probe answers that only for the scheduler path, not
for end-to-end serving).

## 3. Environment reconstruction plan (task step 4) — designed, one genuinely new finding

Re-examining `docs/audits/apt_serve_official_artifact_audit_20260805.md`
§9's local probe with fresh eyes (per this task's "treat findings as
hypotheses to verify" instruction) surfaced something the prior audit
did not check: **why**, exactly, did the local `pip install
vllm==0.5.0.post1` dry-run trigger a from-source build? Re-checking
PyPI's file listing for `vllm==0.5.0.post1` directly:

```
vllm-0.5.0.post1-cp38-cp38-manylinux1_x86_64.whl
vllm-0.5.0.post1-cp39-cp39-manylinux1_x86_64.whl
vllm-0.5.0.post1-cp310-cp310-manylinux1_x86_64.whl
vllm-0.5.0.post1-cp311-cp311-manylinux1_x86_64.whl
vllm-0.5.0.post1.tar.gz  (source only, no cp312 wheel)
```

**Prebuilt wheels exist for Python 3.8-3.11 — not 3.12.** The local
workstation's Python is 3.12.3, which is *why* the prior probe hit a
from-source build requiring `torch==2.3.0` as a build dependency before
any CUDA question even arose. This is a **materially different finding**
from "vLLM 0.5.0.post1 cannot be installed here" — it can, cleanly, on
Python 3.11, independent of the (separately real) CUDA/Blackwell
question for actual GPU execution. The prepared Wulver environment
therefore specifies **Python 3.11** deliberately, not copied from the
local workstation's Python version — this alone should let `pip install
vllm==0.5.0.post1` resolve a prebuilt wheel with zero build step, on
Wulver's CPU-only partition, with no CUDA toolkit module needed at all
for the *install* (a real GPU/CUDA toolkit is still required for actual
*execution* of GPU kernels, unaffected by this finding — see §4).

Prepared reconstruction (not yet run):

| Component | Version | Install method |
|---|---|---|
| Python | 3.11 (mamba-created env) | `mamba create -p <env> python=3.11` |
| PyTorch | `torch==2.3.0` | `pip install torch==2.3.0` (prebuilt wheel expected at cp311) |
| vLLM | `vllm==0.5.0.post1` | `pip install vllm==0.5.0.post1` (prebuilt wheel expected at cp311 — the genuinely new finding above) |
| xformers | `xformers==0.0.26.post1` | `pip install` |
| vllm-flash-attn | `vllm-flash-attn==2.5.9` | `pip install` |
| CUDA toolkit | Not requested via `module load` for this probe | Import/construction-only testing does not need `nvcc`; Wulver's module system offers CUDA/12.6.0 and 12.8.0 (confirmed available from this project's own prior Wulver sessions, `docs/wulver_gpu_validation_handoff.md`-adjacent scripts) if a later phase needs it |
| Compiler | Not invoked | No source build expected given the cp311 wheel finding above |

Full commands are in `scripts/slurm/wulver_apt_serve_strategy_c_cpu_probe.sbatch`
(committed this pass, not hand-copied into this document to avoid two
sources of truth drifting apart).

## 4. Wulver job preparation (task step 2) — prepared, not submitted

| Field | Value |
|---|---|
| Job (primary) | `scripts/slurm/wulver_apt_serve_strategy_c_cpu_probe.sbatch` |
| Partition | `general` (CPU-only — matches this project's own established CPU-partition convention, e.g. `wulver_repeated_trials_postprocess.sbatch`) |
| Account / QOS | `ikoutis` / `standard` (matches every prior Wulver job in this repo) |
| CPU/RAM | 8 cpus-per-task, 16GB mem |
| Wall-time | 00:45:00 |
| GPU | None requested (CPU-only, per this task's explicit preference) |
| Module environment | `module purge` then a self-contained mamba env — no system modules loaded for this probe (see §3) |
| Job (fallback, GPU) | `scripts/slurm/wulver_apt_serve_strategy_c_gpu_fallback.sbatch` — **not to be submitted** unless the CPU probe's own results show a CUDA-initialization failure specifically at import time (not construction time); single `gpu:a100:1`, 30-minute wall-time, otherwise identical environment |
| Output/error logs | `logs/%x.%j.{out,err}` (repo-relative, matching existing convention) + a per-job scratch output directory `/mmfs1/scratch/ikoutis/sv96/apt_serve_strategy_c_probe_<jobid>/` containing `manifest.txt`, `pip_install.log`, `pip_freeze.txt`, `import_probe_vanilla.json`, `copied_file_hashes.txt`, `import_probe_patched.json`, `micro_trace.json` (if reached) |

## 5-9. Import probe / separability / coupling / micro-traces (task steps 5-8) — EXECUTED

Three jobs were submitted this pass, on `login02` (interactive Wulver
session, no auth blocker this time), account `ikoutis`, QOS `standard`,
partition `general`, all CPU-only (`--gres` unset), all completed with
SLURM exit code `0:0`:

| Job | Elapsed | Node | Outcome |
|---|---|---|---|
| `1163456` | 00:02:17 | n0121 | **Environment-construction bug, not a scientific result.** `mamba create` failed (`command not found` — the Anaconda3/2023.09-0 module's `conda.sh` does not ship a `mamba` binary; only Miniforge3 does) and, because the driver script runs under `set -uo pipefail` (deliberately not `-e`, see the `.sbatch` header), execution silently fell through to `conda activate` on a directory that was never created (`EnvironmentLocationNotFound`), which itself failed and ALSO fell through — every subsequent `pip install` then targeted the shared, read-only Anaconda module `site-packages` instead of an isolated env, and failed with `PermissionError` (`/apps/easybuild/software/Anaconda3/2023.09-0/lib/python3.11/site-packages/...`). **Zero packages installed; the read-only filesystem permission on the shared module path prevented any lasting damage** (independently re-verified post-hoc: `typing_extensions.py`'s owner/mtime in that shared path unchanged, still `userapps`/2023). Classified: **packaging/environment** failure, task step 5's first bucket — not import-only, not CUDA-runtime, not any Apt-Serve-specific finding at all. |
| `1163782` | 00:09:35 | n0121 | **Corrected resubmission (fix: `mamba create` → `conda create`, plus a loud `CONDA_PREFIX` verification abort added right after activation so this exact failure mode cannot recur silently).** Full environment build succeeded: Python 3.11.15, `torch==2.3.0+cu121`, `vllm==0.5.0.post1`, `xformers==0.0.26.post1`, `vllm-flash-attn==2.5.9` — **all four exact pins resolved as prebuilt wheels, zero source builds, zero compiler invoked** (confirms the prior audit's Python-3.11-vs-3.12 wheel-availability finding). Vanilla import probe: 9/9 `OK`. Patched import probe: 7/7 `OK`. Micro-trace: 0/3 `OK` — all three scenarios failed identically with `TypeError: Sequence.__init__() got an unexpected keyword argument 'prompt'`, a **type/interface** classification (task step 5's bucket) in the *probe script's* synthetic-object construction, not in the scheduler under test — the probe's own defensive introspection (`inspect.signature`) correctly recorded the real signature (`Sequence(seq_id, inputs: LLMInputs, block_size, eos_token_id=None, lora_request=None)`, an `LLMInputs` TypedDict, not separate `prompt`/`prompt_token_ids` kwargs) even as construction failed, exactly as the script's own docstring said it would. |
| `1164406` | 00:01:20 | n0041 | **Corrected resubmission (fix: `apt_serve_micro_trace.py`'s `_try_build_seq_group` rewritten to the introspected signature from job 1163782 — `LLMInputs(prompt_token_ids=..., prompt=...)` passed as `inputs=`, plus an explicit `SamplingParams(max_tokens=16)` instead of `None`), reusing the already-built env/vendor checkout (no reinstall, hence the short elapsed time).** Vanilla probe: 9/9 `OK` (byte-identical to 1163782 except timestamp/hostname — confirms reproducibility across two different compute nodes, `n0121` vs `n0041`). Patched probe: 7/7 `OK`. **Micro-trace: 3/3 `OK`** — real scheduling decisions from the actual, unmodified, patched `Scheduler.schedule()`. |

Per task step 11 ("submit at most one corrected replacement per
distinct diagnosed issue; never launch repeated blind retries"): three
jobs, two distinct diagnosed issues (the `mamba`/environment-isolation
bug; the micro-trace script's `Sequence` constructor-shape guess), one
corrected resubmission each — not blind retries.

### Import probe detail (task step 5's checklist, all from job 1164406's `import_probe_{vanilla,patched}.json`, copied to `results/provenance/apt_serve_strategy_probe/`)

- **Does `vllm.core.scheduler` import (vanilla)?** Yes — `import_vllm_core_scheduler` `OK`, along with `vllm.sequence`, `vllm.core.interfaces`, `vllm.config`, `vllm.lora.request`, `vllm.core.policy`, and vanilla `Scheduler(SchedulerConfig, CacheConfig, lora_config=None)` construction with synthetic config objects — all `OK`, all on CPU.
- **Do the Apt-Serve-patched modules import?** Yes — `import_patched_scheduler_module` `OK`, and directly confirms the loaded class is genuinely the Apt-Serve algorithm, not a stale/cached vanilla import: `has_greedy_selection_prefill: true`, `has_greedy_selection_decode: true`, `has_dynamic_priority: true` (all three are Apt-Serve-specific method names from the base audit's §3 source read, absent from vanilla vLLM's own `Scheduler`). `import_patched_block_manager`, `import_patched_sequence` (`sequence_group_has_use_hidden: true` — direct confirmation the patched `SequenceGroup`/`Sequence` carry the hybrid-cache `use_hidden` field the base audit's §3 predicted from source reading), `import_patched_block` — all `OK`.
- **Does import require CUDA shared libraries / initialize CUDA?** **No.** `env_fingerprint_torch` (recorded fresh in both vanilla and patched phases): `cuda_available: false`, `cuda_device_count: 0`, `torch_version: "2.3.0+cu121"` (the `+cu121` suffix means the wheel was *built against* CUDA 12.1 headers, not that CUDA was touched at runtime — no `nvcc`, no CUDA driver, no GPU allocated anywhere on this CPU-only node; the job requested no `--gres`). Zero CUDA-related exceptions anywhere in either probe.
- **Does patched `Scheduler.__init__` succeed with synthetic, GPU-free config?** **Yes** — `patched_scheduler_construct_synthetic_config` `OK`, `block_manager_type: "<class 'vllm.core.block_manager_v1.BlockSpaceManagerV1'>"` (the Apt-Serve-patched `AptServeBlockManager`, installed over vLLM's own `block_manager_v1.py` per the base audit's §3 file-mapping), using a purely synthetic `SchedulerConfig(max_num_batched_tokens=2048, max_num_seqs=16, max_model_len=2048)` / `CacheConfig(block_size=16, ..., num_gpu_blocks=100, num_cpu_blocks=100)` — no real model, no real GPU memory, no engine.
- **Are model weights required?** No — never referenced anywhere in either probe or the micro-trace; construction and scheduling both operate purely on integer block/token counts.
- **Is Ray required?** `import_ray` `OK` (`ray==2.56.1`, a normal pip dependency of `vllm==0.5.0.post1` itself, already present in the resolved environment) — **but not exercised or required by scheduler construction/scheduling** in this probe; vLLM's own `Scheduler` class does not itself instantiate a Ray actor/cluster at the construction or `schedule()` call granularity tested here (Ray is vLLM's *distributed-worker* orchestration layer, invoked at `LLMEngine`/`Worker` level, which this probe never reaches — consistent with the base audit's own scoping of "engine lifecycle" as a separate, untested layer).
- **Are custom CUDA kernels required?** No — `mixed_cache_ops` (the `mixed_cache.cu` kernel) was deliberately not built (per §2's disclosed minimal-file-subset scope) and never imported or referenced by anything that succeeded; this probe answers the question only for the **scheduler/admission-decision path**, not end-to-end token generation, exactly as scoped in §2.
- **Is the hidden-state-cache module an import-time dependency?** No separate hidden-cache *module* exists to import — the hybrid-cache concept is carried entirely as a field (`use_hidden`) on the already-tested `Sequence`/`SequenceGroup` objects (confirmed present, `sequence_group_has_use_hidden: true`) and as branching logic inside the already-tested `aptserve_scheduler.py`/`aptserve_block_manager.py`, not a separate importable component.

**No exception was ever classified as CUDA-runtime, engine-lifecycle, cache-manager, or model-execution** in either probe — every failure that did occur (job 1163456's total environment failure; job 1163782's micro-trace `TypeError`) was **packaging/environment** or **type/interface**, both fully diagnosed and corrected within this same pass, not treated as evidence of semantic inseparability (per task step 5's explicit "do not treat an import failure as proof of semantic inseparability until the failure is classified").

### Scheduler separability / micro-trace detail (task steps 6-7, from job 1164406's `micro_trace.json`)

All three hand-constructed scenarios ran against the real, unmodified,
patched `Scheduler` (seed `20260806`, `num_gpu_blocks=64`,
`block_size=16`, `max_num_seqs=16`, `max_num_batched_tokens=2048`,
environment hash `159abed20e53debe`):

| Scenario | Requests (prompt-token counts) | Scheduled | Waiting after | Running after |
|---|---|---|---|---|
| `three_requests_two_fit_memory_budget` | req-a=200, req-b=200, req-c=4000 | req-a, req-b | 1 (req-c) | 2 |
| `homogeneous_low_contention` | 4× req-N=100 | all 4 | 0 | 4 |
| `single_oversized_request_extreme_case` | req-small-1=50, req-small-2=50, req-huge=8000 | req-small-1, req-small-2 | 1 (req-huge) | 2 |

All three results are **exactly the qualitatively expected behavior**
from the base audit's §3 algorithm read (greedy, memory-budget-bounded
admission; the third scenario directly exercises the appendix's own
"extreme case" the double-check step exists to bound — a single very
large candidate correctly excluded rather than crowding out the small
ones, matching Theorem 1's guarantee rather than violating it).
**Initialization, request insertion (`add_seq_group`), value
computation, greedy selection, capacity checks, and state update
(waiting/running queue transitions) are all confirmed working end to
end** via this single `schedule()` call path — this probe's synthetic
scenarios did not separately exercise SLO-penalty deprioritization,
explicit cache-type-switch/preemption events, or tie-breaking among
equal-value candidates (all three would require either
`ttft_slo`/`tbt_slo`-bearing `SamplingParams`/request metadata or a
multi-`schedule()`-call sequence deliberately engineered to produce
ties — not attempted this pass, a disclosed scope limit, not a
failure). Deterministic replay: not separately re-run with a second
process this pass beyond the two independent job runs (1163782→1164406)
already showing byte-identical vanilla/patched import results across
two different nodes; the micro-trace itself was run fresh only once
under the corrected script (job 1164406) — a second identical
re-invocation to confirm bit-for-bit scenario-level determinism is a
natural follow-up, not completed here.

**Official micro-traces**: `results/provenance/apt_serve_strategy_probe/micro_trace.json`,
schema `apt_serve_strategy_c_probe.v1`, containing full input state,
constructor introspection, and output state per scenario, plus the
environment/commit provenance chain — see that directory's `README.md`
for the full field-by-field description (task step 7).

## 9b. Strategy C/D decision (task step 9)

**Classification: `STRATEGY_C_VIABLE_WITH_LIMITATIONS`.**

This is a scoped decision, not a blanket "Apt-Serve works" claim. It
answers exactly the question Phase A was opened to answer — is the
official scheduler separable and reusable, as opposed to needing a
from-scratch reimplementation to even evaluate — and it answers that
question yes, with five specific, disclosed limitations rather than
none.

- **Technical reuse feasibility — established.** The official,
  unmodified `aptserve_scheduler.py` (plus the 4 files it depends on:
  block manager, interfaces, sequence, block) imports cleanly, patches
  cleanly over a stock `vllm==0.5.0.post1` install via simple file copy,
  constructs cleanly against a purely synthetic `SchedulerConfig`/
  `CacheConfig` (no engine, no model, no GPU memory), and executes real
  `Scheduler.schedule()` calls that reproduce the qualitatively expected
  greedy, memory-budget-bounded admission behavior (§9's three
  scenarios). This was demonstrated with the real vendor source, not
  inferred from reading it — job 1164406, reproduced structurally by
  1163782 on a different node. Reuse is via a pinned external checkout
  plus a thin compatibility shim (the "apply 5 files over vLLM's own
  module paths" mechanism this probe used, or an equivalent adapter
  layer at implementation time) — not a fork, not a copy-paste port.
- **Legal redistribution constraint — binding, not advisory.**
  `eddiegaoo/Apt-Serve` has no LICENSE file (confirmed twice via `gh api
  ... --jq .license` → `null`, §1 and the prior official-artifact
  audit). No Apt-Serve source may be vendored, copied, or committed into
  this repository at any point, including at implementation time. Any
  real integration must fetch the pinned commit into scratch/vendor
  space outside the git tree at build/run time, exactly as this probe's
  job scripts already do — this is a hard constraint on *how* Strategy C
  can be implemented, not a caveat on whether it can.
- **Pinned-environment requirement — real, and non-trivial to
  reconcile.** The only environment in which this was demonstrated is an
  exactly-pinned, comparatively old stack: Python 3.11 specifically (3.12
  forces a from-source build, §3), `torch==2.3.0`, `vllm==0.5.0.post1`,
  `xformers==0.0.26.post1`, `vllm-flash-attn==2.5.9`, isolated in its own
  conda env. This project's other external-baseline integrations are not
  guaranteed to share this exact pin set. Whatever consumes the Apt-Serve
  scheduler at implementation time will need either strict environment
  isolation (a subprocess/adapter boundary, not a plain in-process
  import alongside this project's own simulator dependencies) or a
  deliberate, separately-justified dependency-compatibility check — not
  assumed compatible by default.
- **Remaining simulator gap — Strategy C/D does not resolve this.** This
  probe confirms the patched `Sequence`/`SequenceGroup` carry the
  hybrid-cache `use_hidden` field and that the scheduler constructs
  around it, but no scenario in §9 exercised actual dual-tier
  GPU/CPU-hidden-state cache promotion, eviction, or a memory-pressure-
  triggered tier switch — the three scenarios test admission/capacity
  logic, not cache-tier transitions. Building Apt-Serve's actual
  contribution into this project's own discrete-event simulator still
  requires a new dual-tier cache-state model in the simulator itself,
  **regardless of whether the scheduler is reused via Strategy C or
  reimplemented via Strategy D** — this decision narrows *how much
  scheduler code* must be written, not whether the cache-model work is
  needed.
- **Full-system validation status — not established, and not claimed
  here.** This probe validates component-level separability under
  synthetic, CPU-only, no-model, single-`schedule()`-call conditions. It
  is not a comparative performance evaluation, does not run token
  generation, does not exercise SLO-penalty deprioritization or
  preemption, and never required (or ran) the GPU fallback. Before
  Apt-Serve could receive any `FOUNDATIONAL_CANDIDATE` / `EVALUATION_ONLY`
  / deployable classification matching this project's convention for
  every other external baseline (Sarathi-Serve, VTC, vLLM-LTR,
  PARS-Serve-2026), it still needs: a thin in-repo adapter (or a faithful
  reimplementation informed by these traces, if version drift makes the
  adapter route too brittle), the dual-tier cache simulator extension
  above, fidelity-differential tests against more of these official
  traces, and only then a comparative evaluation against this project's
  deployable policy set. None of that is started by this probe or by
  this classification — this probe closes the *reuse-feasibility*
  question, not the *implementation* or *evaluation* questions.

## 10. What remains — exact next action

**COMPLETED (this pass, executed evidence, not projected):**

- Wulver SSH/auth resolved and a real interactive session used (the §0
  blocker was specific to the earlier, non-interactive audit pass and
  did not recur).
- Isolated Python 3.11 environment reconstruction, with the exact pins
  (`torch==2.3.0`, `vllm==0.5.0.post1`, `xformers==0.0.26.post1`,
  `vllm-flash-attn==2.5.9`), all as prebuilt wheels, zero source builds.
- Official vanilla `vllm.core.scheduler` import probe (9/9 checks OK).
- Official Apt-Serve-patched scheduler/block-manager/interfaces/sequence/
  block import probe (7/7 checks OK, confirmed genuinely Apt-Serve via
  `has_greedy_selection_prefill/decode`, `has_dynamic_priority`,
  `sequence_group_has_use_hidden`).
- Patched `Scheduler.__init__` construction against a purely synthetic,
  GPU-free config.
- Real, unmodified, patched `Scheduler.schedule()` micro-traces (3/3
  scenarios OK), reproduced structurally across two different compute
  nodes (`n0121`, `n0041`).
- Strategy C/D decision (§9b): `STRATEGY_C_VIABLE_WITH_LIMITATIONS`.
- Source and environment provenance chain: git blob hashes (pre-run) and
  sha256 hashes of the copied files as landed in the isolated env
  (post-run), pinned-commit re-verification (`PINNED_COMMIT` ==
  `ACTUAL_COMMIT`), full `pip freeze`, compact result artifacts in
  `results/provenance/apt_serve_strategy_probe/`.

**NOT COMPLETED (genuine gaps, not yet started):**

- A thin, in-repo compatibility adapter mapping this project's own
  simulator abstractions to the pinned vLLM's `Scheduler`/
  `SequenceGroup`/`BlockSpaceManager` interfaces (this probe used a
  standalone isolated-env file-copy, not an adapter integrated into
  `src/llmserveopt/`).
- The dual-tier (GPU/CPU-hidden-state) hybrid cache simulator extension
  — needed regardless of Strategy C vs. D (see §9b).
- Fidelity-differential tests against a larger set of official traces
  than this pass's 3 hand-constructed scenarios (no SLO-penalty
  deprioritization, preemption, or tie-breaking scenario has been run).
- An Apt-Serve stress-test catalog entry set (target/counter regimes),
  matching this project's convention for every other external baseline.
- A comparative evaluation against this project's deployable policy set.
- Real-system (GPU, real model, real token generation) performance
  evaluation — the GPU fallback job was never needed and was never run.
- A `FOUNDATIONAL_CANDIDATE` / `EVALUATION_ONLY` / deployable
  classification — premature until the adapter, cache extension, and
  comparative evaluation above exist.

**Exact next Apt-Serve action:** design the thin external-checkout
adapter and the minimal dual-tier cache interface specification —
*design only*, not implementation, and not this query. This is separate
from, and does not block, this project's separately-tracked next local
action (Llumnix stress-test coverage and first comparative evaluation,
per `docs/current/RESUME_HERE.md` §E).

## 11. `docs/BASELINE_STATUS.md`

Updated this pass (same commit as this document): Apt-Serve now has an
explicit row reflecting §9b's `STRATEGY_C_VIABLE_WITH_LIMITATIONS`
classification, replacing the prior "Not implemented" listing. It is
**not** classified `FOUNDATIONAL_CANDIDATE`, `EVALUATION_ONLY`, or
deployable — per §9b/§10, the adapter, dual-tier cache extension, and
comparative evaluation remain outstanding, and the row says so
explicitly rather than implying more progress than has actually
happened.
