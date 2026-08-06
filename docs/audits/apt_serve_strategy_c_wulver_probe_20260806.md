# Apt-Serve Strategy C Wulver Probe — 2026-08-06

Phase A of `docs/audits/apt_serve_official_artifact_audit_20260805.md`
§12's implementation roadmap: determine whether Apt-Serve's official
scheduler can be imported and reused as a separable Python component
(Strategy C) rather than defaulting to a from-scratch reimplementation
(Strategy D) without having actually tested C's precondition.

**Status: prepared, execution blocked pending Wulver authentication.**
Everything that can be done without an authenticated Wulver session is
complete (commit pinning with hashes, environment plan, job scripts,
probe scripts, all syntax-validated). The actual job submission and
result-gathering steps (§5-9 below) have not run yet — see §0.

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

## 5-9. Import probe / separability / coupling / micro-traces (task steps 5-8) — NOT YET RUN

Cannot be completed without Wulver authentication (§0). The scripts
that will produce this evidence are committed and syntax-validated
(`python -m py_compile` clean for both; `bash -n` clean for both
`.sbatch` files) but have never executed against a real vLLM install —
this is disclosed explicitly, not glossed over:

- `scripts/wulver_probes/apt_serve_import_probe.py` — implements
  exactly the checklist in task step 5 (vanilla vLLM scheduler import →
  Apt-Serve-patched module import → CUDA-at-import detection →
  synthetic-config scheduler construction → Ray import check), phased
  so a hard crash in the patched phase cannot destroy the vanilla
  phase's already-recorded results (run as separate `--phase vanilla`
  / `--phase patched` invocations, in separate Python processes, per
  the script's own docstring).
- `scripts/wulver_probes/apt_serve_micro_trace.py` — implements task
  step 8's requested differential traces (3 hand-constructed scenarios:
  mixed-size requests testing genuine memory-budget contention, a
  homogeneous low-contention control, and a single-oversized-request
  case directly probing the appendix's own "extreme case" double-check
  logic — see `apt_serve_official_artifact_audit_20260805.md` §4's
  discussion of this exact scenario). **Disclosed limitation**: this
  script's `Sequence`/`SequenceGroup` construction calls were written
  from general knowledge of vLLM 0.5.x's typical constructor shape,
  *not verified against a working local install* (none exists — see
  the base audit's §9). The script is defensive specifically because of
  this: it introspects and records the actual constructor signatures
  via `inspect.signature` regardless of whether construction succeeds,
  so a wrong assumption produces a diagnostic, not a lost job.

**Because none of this has run, task steps 9 (Strategy C vs. D
decision) and 10 (refined simulator-extension scope) cannot be
completed with executed evidence yet either** — per this task's own
explicit instruction ("Base the decision on executed evidence. Do not
choose based only on code reading."), no decision is made in this
document. `apt_serve_official_artifact_audit_20260805.md`'s own
placeholder position (Strategy D as the interim default, Strategy C
left genuinely open) **stands unchanged** until this probe actually
runs.

## 10. What remains — exact next action

1. **User action required**: authenticate this environment's SSH access
   to Wulver — run `kinit sv96@<realm>` (exact Kerberos realm not
   independently confirmed by this pass; `sv96` is the confirmed
   username from `~/.ssh/config`) interactively, or otherwise refresh
   whatever credential the `login02` GSSAPI login depends on. Suggested:
   type `! kinit sv96` in the prompt (adjusting the realm if the
   default one `kinit` picks is wrong) so the ticket lands in this
   session.
2. Once authenticated: `sbatch scripts/slurm/wulver_apt_serve_strategy_c_cpu_probe.sbatch`,
   monitor via `squeue`/`sacct` (use `tmux` if the monitoring session
   would run past ~15 minutes, per this task's own instruction — a
   45-minute wall-time job likely warrants this), then retrieve
   `import_probe_vanilla.json`/`import_probe_patched.json`/
   `micro_trace.json` from the job's scratch output directory.
3. Only submit the GPU fallback job if step 2's results show a genuine
   CUDA-at-import failure (not a missing-package or wrong-signature
   failure, which should be fixed and rerun on CPU instead).
4. Update this document's §5-9 with the actual results, make the
   Strategy C vs. D decision in §9 (this pass leaves that
   **undetermined**, explicitly, rather than guessing), and refine the
   simulator-extension scope estimate in §10 of the base audit using
   whatever the real constructor signatures/coupling turned out to be.

## 11. `docs/BASELINE_STATUS.md`

Not updated this pass — the integration-strategy classification has not
changed (still whatever `apt_serve_official_artifact_audit_20260805.md`
recorded, since no execution evidence exists yet to justify a change),
and Apt-Serve was never given its own `BASELINE_STATUS.md` row in the
first place (it remains listed as "Not implemented").
