# Apt-Serve Strategy C Wulver Probe — Result Artifacts

Compact, executed-evidence artifacts from the Strategy C (official-scheduler
dynamic-import) feasibility probe run on Wulver, 2026-08-05/06. Full narrative,
methodology, and the Strategy C/D decision are in
`docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md`; this directory
holds only the machine-generated evidence files themselves.

**Schema version:** `apt_serve_strategy_c_probe.v1`

**No upstream Apt-Serve source is present in this directory or anywhere in
this repository.** `eddiegaoo/Apt-Serve` has no LICENSE file (confirmed via
`gh api repos/eddiegaoo/Apt-Serve --jq .license` → `null`). All files here are
either this project's own probe-script output (JSON reports, hashes, package
manifests) or generic package-manager output (`pip freeze`); none contain
Apt-Serve source text.

## Provenance chain

| Field | Value |
|---|---|
| Official repository | `https://github.com/eddiegaoo/Apt-Serve` |
| Pinned commit | `c953217988274a761da35cf06c01033b18dadf68` |
| License | None (no LICENSE file; disclosed, not vendored) |
| Vendor checkout location (Wulver, outside repo tree) | `/mmfs1/project/ikoutis/sv96/vendor/apt-serve` |
| Isolated environment location (Wulver, outside repo tree) | `/mmfs1/project/ikoutis/sv96/conda_env/apt-serve-strategy-c-probe` |
| Environment hash (torch\|vllm\|xformers version fingerprint) | `159abed20e53debe` (`torch=2.3.0+cu121\|vllm=0.5.0.post1\|xformers=0.0.26.post1`) |

## Job manifests

- `job_manifest_1163782.txt` — first successful full environment build + probe run (job 1163782, 2026-08-05 23:17–23:27 EDT, `n0121`, exit 0). Micro-trace scenarios failed here on a `Sequence` constructor-shape mismatch in the probe script itself (see audit doc §5-9's job table) — scheduler construction itself succeeded.
- `job_manifest_1164406.txt` — corrected-script rerun (job 1164406, 2026-08-05 23:29–23:31 EDT, `n0041`, exit 0), reusing the already-built env/vendor checkout. All 3 micro-trace scenarios passed. **This run's `import_probe_*.json`/`micro_trace.json` are the ones copied into this directory** (byte-identical import-probe results to 1163782 except timestamp/hostname — confirms reproducibility across two different compute nodes).
- A prior job, 1163456, failed entirely on an environment-construction bug (`mamba` not available from the Anaconda3 module) and produced no valid signal; not included here — see the audit doc's job-failure log for the full post-mortem (this is why 1163782/1164406 exist as corrected resubmissions, each fixing one distinct diagnosed issue, not blind retries).

## Files

- `import_probe_vanilla.json` — vanilla (pre-Apt-Serve-patch) vLLM 0.5.0.post1 import/construction probe. 9/9 checks `OK`.
- `import_probe_patched.json` — Apt-Serve-patched (5-file minimal scheduler subset) import/construction probe. 7/7 checks `OK`, including `patched_scheduler_construct_synthetic_config`.
- `micro_trace.json` — 3 hand-constructed differential scheduling scenarios run against the real, unmodified, patched `vllm.core.scheduler.Scheduler`. 3/3 `OK`. Contains full input state (per-request prompt lengths), constructor introspection, and output state (scheduled/waiting/running request IDs) for each scenario — suitable as a differential-testing oracle for a future faithful reimplementation or adapter.
- `environment_pip_freeze.txt` — full `pip freeze` of the isolated probe environment (Python 3.11.15).
- `copied_file_sha256_hashes.txt` — sha256 of the 5 Apt-Serve files as landed inside the isolated vLLM install on Wulver, cross-checkable against the git blob hashes recorded in the audit doc (provenance checked twice: pre-run git blob hash, post-run sha256 of the copied file).

## Raw logs (Wulver-local, not copied into the repo)

Full `pip install` logs, `env_create.log`, `git_clone.log`, `file_copy.log` for
both runs remain at (Wulver-local paths, non-portable, subject to scratch
purge policy — not a durable reference):

- `/mmfs1/scratch/ikoutis/sv96/apt_serve_strategy_c_probe_1163782/`
- `/mmfs1/scratch/ikoutis/sv96/apt_serve_strategy_c_probe_1164406/`

SLURM stdout/stderr (kept in-repo, small): `logs/apt-serve-strategy-c-cpu-probe.{1163456,1163782,1164406}.{out,err}`.
