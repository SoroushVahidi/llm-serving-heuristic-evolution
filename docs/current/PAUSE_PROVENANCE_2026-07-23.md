# Pause Provenance — 2026-07-23

> **SUPERSEDED FOR CURRENT STATUS.**
> See [`docs/current/RESUME_HERE.md`](RESUME_HERE.md) for authoritative current state.
> This snapshot (2026-07-23) predates MF-PSD, all NO_GOs, hierarchical routing, live re-evaluation, and Family-B replication prep.

Durable evidence record for the pause state established by
`docs/current/PROJECT_HANDOFF_2026-07-23.md`. Written specifically because
the handoff originally cited a session-scratch `/tmp` log path that will not
survive logout/reboot — this document is the durable replacement for that
citation. Large raw logs are NOT copied into git; this file records the
concise, verified result plus exactly how it was verified, so a future agent
can re-verify without needing the original ephemeral file.

---

## Repository identity — FINAL pause checkpoint (2026-07-23)

- **Authoritative repo path:** `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-final-integration`
- **Branch:** `wulver-final-integration-20260721`
- **Checkpoint commit (final HEAD at pause):** `8c9cedbca171d44030a16cf630f81f99d15d729f`
  ("feat: add faithful SLAI baseline and preserve paused research state")
- **Parent commit (pre-checkpoint HEAD):** `d1d5f12a0752a061e563f87dcf3e3289bee2e4bb`
- **Upstream:** `origin/wulver-final-integration-20260721` — **pushed and synchronized: 0 ahead, 0 behind** at final pause time.
- **All SLAI-related work is committed** as of the checkpoint commit above —
  no uncommitted work remains. (Below this point, "at pause time" in
  earlier subsections of this document refers to the intermediate state
  captured during the pause sequence's own audit steps, before the
  checkpoint commit was created — kept for provenance, not because
  anything is still uncommitted.)
- History leading to the checkpoint: `8c9cedb` ← `d1d5f12` ("chore: ignore
  generated Wulver artifacts") ← `bacea0a` ("docs: consolidate current
  research state and roadmap") ← `e8bd759` (last commit that was already on
  `origin` before this pause sequence began).

---

## Full non-hardware test suite — final verified result

**Result: 2501 passed, 88 skipped, 26 deselected, 0 failed, exit code 0.**

**How this was verified:** run interactively via
```
cd /mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-final-integration
PYTHONPATH=$(pwd)/src python -m pytest tests/ -q \
  --deselect tests/test_compare_simulator_to_real_llm_latency.py \
  --deselect tests/test_calibration_gpu.py \
  --deselect tests/test_gpu_external_validity_audit.py
```
using the `repo-env` conda environment (`source $(conda info --base)/etc/profile.d/conda.sh; conda activate repo-env`), completing in 678.73s (0:11:18). This run tested the new `Action.hold_decode` simulator primitive in isolation, **before** `slai_faithful` itself was added to the tree — it is evidence that `hold_decode` introduced zero regressions across the entire existing suite, not evidence about `slai_faithful` specifically (that has its own 22-test dedicated suite, `tests/test_slai_faithful_scheduler.py`, verified separately and passing).

**Reproducing this result if the original log is gone:** re-run the exact
command above. The three `--deselect` flags exclude real-GPU-hardware tests
that require GPU allocation/CUDA and are not relevant to this CPU-only
simulator work. Expect the same 2501/88/26/0 split as long as no source
file under `src/llmserveopt/` changed between now and then in a way that
would add/remove/skip tests.

**Original (non-durable) log location, for reference only — may no longer
exist:**
`/tmp/claude-511190/-mmfs1-home-sv96/a2c3b791-b7e9-4e13-b4bb-8007e812fe25/scratchpad/full_test_run_final.log`

---

## SLAI bounded pilot — durable record

- **Slurm job ID:** `1129769`
- **Final state (verified via `sacct`):** `COMPLETED`, exit code `0:0`, elapsed `00:01:47`, MaxRSS 38224K
- **Durable pilot root (NOT ephemeral — lives on the shared project data
  volume, same tier as every other experiment in `docs/current/EXPERIMENT_INDEX.md`):**
  `/mmfs1/project/ikoutis/sv96/llmserveopt-data/slai_faithful_bounded_pilot_20260723T033609Z/`
  - `scripts/bounded_pilot.py` — the pilot script (entry point)
  - `sbatch/bounded_pilot.sbatch` — the exact submitted job script
  - `logs/pilot_1129769.{out,err}` — full stdout/stderr (stderr is empty; no errors)
  - `results/pilot_results.json` — all 12 windows' full per-policy metrics
  - `results/pilot_summary.json` — aggregate summary

### Key pilot conclusions (verified from `results/pilot_summary.json` and `results/pilot_results.json` directly)

- 12 windows tested: 3 each from Azure, BurstGPT, SwissAI, TraceLab.
- **Oracle-envelope gain from adding `slai_faithful` to the comparison set:
  exactly `0.0` in every one of the 12 windows** — `mean_oracle_envelope_gain_from_slai: 0.0` in `pilot_summary.json`.
- Azure/BurstGPT/SwissAI: every one of the 5 compared policies
  (`slai_faithful`, `sarathi_faithful`, `vllm_chunked_prefill_faithful`,
  `weighted_shortest_processing`, `scorpio_style_slo_guard`) tied at
  ANWG=1.0 in all 9 of those windows (underloaded pilot windows — consistent
  with the pre-existing, independently-documented simulator discriminative-
  power finding at
  `/mmfs1/project/ikoutis/sv96/llmserveopt-data/simulator_discriminative_audit_20260722T223236Z`,
  not a new or SLAI-specific issue).
- TraceLab (the only loaded regime in this pilot): `slai_faithful` lost
  clearly to `weighted_shortest_processing`/`scorpio_style_slo_guard` in all
  3 windows (ANWG 0.51/0.36/0.47 vs. 0.80/0.68/0.79) and was roughly tied-or-
  slightly-worse against the other two chunked-prefill-family faithful
  baselines.
- Decode-hold activation (the mechanism `Action.hold_decode` exists to
  support) fired on 0%–16.4% of steps across the 12 windows (mean 8.2%) —
  confirmed live/exercised on real data, not dead code.
- **`FULL_SWEEP_RECOMMENDATION = NO_GO`** — zero oracle-envelope gain
  everywhere tested, plus a clear loss on the one genuinely load-
  differentiated dataset, does not justify a full four-dataset production
  sweep at this time.

---

## Upstream SLAI source — durable citation

- **Paper:** "Optimal Scheduling Algorithms for LLM Inference: Theory and
  Practice" — Agrim Bari, Parikshit Hegde, Gustavo de Veciana (UT Austin).
  ACM SIGMETRICS 2026 / *Proc. ACM Meas. Anal. Comput. Syst.*, Vol. 9, No. 3,
  Article 59. arXiv:2508.01002 (v1 2025-08-01, v2 2025-12-01).
- **Official repository:** `github.com/agrimUT/SLAI`
- **Pinned commit:** `5098a7aba05e3edbcfa3a509d6cc9cd248fc4380` (`main`,
  "Update README.md", 2025-08-14)
- **License:** Apache License 2.0
- Full algorithm-to-source mapping, disclosed adaptations, and explicit
  exclusions: `docs/slai_faithful_scheduler_reference.md` (committed,
  durable — not dependent on this provenance file).

---

## Important Slurm job IDs to trace the current research state (pre-existing, not from this session unless noted)

| Job ID(s) | Experiment | Status |
|---|---|---|
| 1115576–1115604 (several) | Selector Dataset v2 Overnight Scale | COMPLETE |
| 1117627–1117728 (several) | Selector v2 Conclusive OOD Investigation | COMPLETE |
| 1117863–1117876 (several) | Selector v3 Multi-Domain Causal-State | COMPLETE |
| 1118186–1118197 (several) | Policy Frontier Cartography | COMPLETE |
| 1118781–1118789 (several) | Policy Library v2 Expanded Frontier | COMPLETE |
| 1120495, 1120496 | V2 Real-OOD 27-Policy Library Audit | COMPLETE |
| 1122788–1122795 (several) | Module Intervention / Structural Credit | COMPLETE |
| 1119434 | Composition Readiness Harness | COMPLETE |
| 1120123 | Native Composition Pilot | COMPLETE (`NO_GO`) |
| 1120181 | Structural Synthesis Readiness | COMPLETE |
| 1126581–1126610 (several) | 27-Policy Selector/Regret Benchmark | COMPLETE |
| 1127224–1127228 | SwissAI Staging | COMPLETE |
| 1127593–1127600 | SwissAI V2 27-Policy Sweep | PARTIAL (reporting step failed; data matrix complete) |
| 1127791–1127798 | TraceLab Staging | COMPLETE |
| 1128662–1128670 (several) | TraceLab V2 27-Policy Sweep | COMPLETE |
| 1127940–1127976 (several) | SLO/Deadline Augmented V2 Sweep | COMPLETE (see stale pending remnants below) |
| 1129057, 1129069 | Simulator Discriminative-Power Audit | COMPLETE |
| **1129769** | **SLAI Bounded Pilot (this pause cycle)** | **COMPLETE** |

Full index with per-job breakdowns: `docs/current/EXPERIMENT_INDEX.md`
(pre-existing; not modified this session — the SLAI pilot row above is not
yet added there, tracked as a follow-up in the handoff's resume checklist).

### Stale pending jobs at pause time (verified via `squeue -u sv96`, 2026-07-23)

10 jobs remain in the queue, **all `PENDING`, none `RUNNING`**:
`1127958_[0-39%6]`, `1127959`–`1127964` (six `slo_aug_*` jobs, state
`Dependency`), `1127943`, `1127944_[0-39%6]`, `1127950`, `1127943`
(additional `slo_aug_*` jobs, state `Dependency`/`DependencyNeverSatisfied`),
and `1127600` (`swissai_v2_report`, `DependencyNeverSatisfied`). All belong
to experiments already marked COMPLETE (SLO/Deadline Augmented V2 Sweep) or
PARTIAL-with-complete-data (SwissAI V2 Sweep) in `EXPERIMENT_INDEX.md` —
their upstream dependencies already finished or failed, so Slurm has marked
several `DependencyNeverSatisfied`, meaning **these jobs will never run and
cannot modify any output directory.** Not cancelled in this pause pass per
explicit instruction; safe to cancel in a future cleanup session (see
`PROJECT_HANDOFF_2026-07-23.md`'s Slurm pause-state section).
