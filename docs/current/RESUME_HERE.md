# Resume Here

**Read this document first.** It is the single entry point for resuming
this project after a multi-month pause. Readable in 5–10 minutes.

---

## Project state

- **Paused:** 2026-07-23
- **Authoritative repo path:** `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-final-integration`
- **Branch:** `wulver-final-integration-20260721`
- **Checkpoint commit:** `8c9cedbca171d44030a16cf630f81f99d15d729f`
  ("feat: add faithful SLAI baseline and preserve paused research state")
- **GitHub remote:** `origin/wulver-final-integration-20260721`
  (`github.com/SoroushVahidi/llm-serving-heuristic-evolution.git`) —
  pushed and synchronized (0 ahead, 0 behind) at pause time.

### Post-pause composition reconciliation (2026-07-24)

After the pause checkpoint, the composition/selector-integrated lineage also
added typed reciprocal-rank and score-aggregation operators plus decision
tracing. Those operators are unit-tested only; they are **not**
performance-validated at scale, and **no Wolverine oracle-mixture sweep was
launched**. Composition launch remains **BLOCKED** pending read-only recovery
of native pilot job `1120123` numeric artifacts (locally only a qualitative
`NO_GO` / approximately Level B summary is available). Structural synthesis
remains empirically `NOT_READY`. See
`COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md`,
`COMPOSITION_IMPLEMENTATION_STATUS.md`, and
`WOLVERINE_ORACLE_MIXTURE_HANDOFF.md`. The standing simulator-calibration
bottleneck above is unchanged.

## Read these first (in order)

1. This document.
2. `docs/current/PROJECT_HANDOFF_2026-07-23.md` — full resume-from-scratch handoff, all detail.
3. `docs/current/project_handoff_state.json` — same information, machine-readable.
4. `docs/current/PROJECT_STATUS.md` — the pre-SLAI canonical scientific-state document (still accurate for everything it covers).
5. `docs/current/BASELINES.md` and `docs/current/POLICY_LIBRARY.md` — exact current policy/baseline inventory.
6. `docs/slai_faithful_scheduler_reference.md` — the newest baseline's full fidelity record.

## What the project does

This project builds a deterministic discrete-event **LLM-serving scheduling
simulator** and studies how to select, combine, or synthesize
request-scheduling policies (admission order, batching, KV/cache
management) for LLM inference serving. The long-term goal is a
**state-conditioned mechanism** that estimates the suitability of scheduling
components for a given serving scenario and uses that evidence to combine
or synthesize a new policy — evaluated fairly against fixed, adaptive, and
real-system-derived ("faithful external") baselines. The primary metric is
**ANWG** (arrival-normalized weighted goodput).

## Current state in one paragraph

The repository registers **34 scheduling policies** (20 historical + 7
Policy Library v2 + 7 faithful external baselines, plus 1 hindsight-oracle
reference-only policy) — verify this count live against
`src/llmserveopt/policies/registry.py` /
`src/llmserveopt/policies/external_baselines_registry.py` before trusting
any cached number, including this one. The learned **selector is useful but
not solved**: it beats some fixed baselines in-distribution but does not
reliably capture the oracle-envelope gain out-of-distribution, and is
currently frozen for retraining. **External baseline coverage was just
strengthened** with a new faithful `slai_faithful` baseline (source-grounded
against `github.com/agrimUT/SLAI`, Apache-2.0). The project's **primary,
standing bottleneck is simulator/objective discriminative power** — the
simulator/ANWG objective often collapses diverse real workloads to
near-identical policy rewards (confirmed independently by three separate
audits, most recently the SLAI bounded pilot below). **Composition/structural
synthesis is the most promising longer-term research direction** but is
infrastructure-ready, not scientifically validated yet (a native composition
pilot returned `NO_GO`) — do not resume it before the discriminative-power
bottleneck is addressed.

## Most important recent result: SLAI bounded pilot

Slurm job **1129769** (COMPLETED), root:
`/mmfs1/project/ikoutis/sv96/llmserveopt-data/slai_faithful_bounded_pilot_20260723T033609Z/`.
Compared `slai_faithful` against `sarathi_faithful`,
`vllm_chunked_prefill_faithful`, `weighted_shortest_processing`, and
`scorpio_style_slo_guard` on 12 small windows (3 each from Azure, BurstGPT,
SwissAI, TraceLab):

- **Oracle-envelope gain from adding SLAI: exactly `0.0` in every one of the
  12 windows.**
- Azure/BurstGPT/SwissAI windows were all underloaded (every policy tied at
  ANWG=1.0) — consistent with, not contradicting, the standing simulator-
  discriminative-power bottleneck.
- On TraceLab (the one loaded regime tested), `slai_faithful` **lost
  clearly** to WSP/SCORPIO.
- Decode-hold (the new mechanism this baseline required) **did activate
  meaningfully** on real data (0–16.4% of steps) — the implementation
  works and is exercised, it just didn't win here.
- **`FULL_SWEEP_RECOMMENDATION = NO_GO`.**

**Why this is not necessarily final:** the pilot was small (12 windows),
3 of 4 datasets were underloaded by pilot-window construction (not
necessarily a property of the datasets themselves), and the TraceLab loss
may be partly a `max_steps` simulation-horizon artifact rather than a clean
algorithmic result — neither was confirmed. This is a real negative result
worth taking seriously, not a reason to distrust the implementation (which
is source-grounded, tested, and behaviorally distinct from the pre-existing
`slai_style_phase_aware` approximation) — but also not a reason to write
off `slai_faithful` permanently without a properly load-calibrated re-test.

## First three actions when resuming

1. **Re-validate current Git/data availability and load-calibrated
   benchmark construction.** Confirm this checkpoint is still what's on
   `origin`, confirm the durable data roots listed below still exist, then
   fix the underlying issue the SLAI pilot re-exposed: Azure/BurstGPT/SwissAI
   window construction needs genuinely loaded regimes, not just larger
   sample counts, before any policy comparison on them means anything
   (Stage 2 of `docs/current/RESEARCH_ROADMAP.md` — the project's own
   standing highest-priority item, independent of SLAI).
2. **Run a realistic, discriminative external-baseline evaluation and
   regenerate reward vectors** — once (1) is fixed, re-run the SLAI bounded
   pilot (same script,
   `.../slai_faithful_bounded_pilot_20260723T033609Z/scripts/bounded_pilot.py`)
   plus a broader pass across the other 6 faithful baselines, on properly
   loaded windows, via `sbatch` (never interactively for anything beyond a
   smoke test).
3. **Perform one final selector benchmark, then decide whether to freeze
   selector work and shift to composition/synthesis** — only after (1)
   and (2) produce trustworthy reward separation, per the project's own
   stated go/no-go criterion (`docs/current/SELECTOR_STATUS.md`).

## Do not do first

- **Do not blindly retrain the selector.** It's explicitly frozen pending
  simulator calibration — retraining on today's rewards repeats known,
  already-diagnosed weak-signal problems.
- **Do not run a full four-dataset SLAI sweep** before fixing load
  calibration and the TraceLab horizon question — the bounded pilot's
  `NO_GO` was specifically about *this*, not about SLAI as an algorithm.
- **Do not confuse style/inspired baselines with faithful baselines.**
  `scorpio_style_slo_guard`, `sarathi_style`, `slai_style_phase_aware`, etc.
  are original heuristics with suggestive names, not pinned-commit
  reproductions. Only the 7 names in `EXTERNAL_BASELINE_REGISTRY`
  (`src/llmserveopt/policies/external_baselines_registry.py`) are faithful.
- **Do not treat the other linked worktree
  (`.../llm-serving-heuristic-evolution`, branch
  `wulver-policy-composition-readiness`) as authoritative.** It was fully
  audited during this pause and confirmed to hold no unique work, but it is
  still not the checkout to build on.
- **Do not run long jobs interactively on the login node.** Use `sbatch`
  for anything beyond a trivial smoke test — the login node has 1 visible
  CPU and no active allocation.

## Useful durable paths

- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/slai_faithful_bounded_pilot_20260723T033609Z/` — SLAI pilot
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/simulator_discriminative_audit_20260722T223236Z/` — the standing-bottleneck audit
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/v2_selector_regret_benchmark_20260722T134925Z/` — strongest selector result to date
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/v2_real_ood_library_20260721T222521Z/` — real-OOD oracle-gain evidence
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/swissai_v2_policy_sweep_20260722T184451Z/` and `tracelab_v2_policy_sweep_20260722T214129Z/` — the two most recent full dataset sweeps
- Full index with every job ID: `docs/current/EXPERIMENT_INDEX.md`

## Resume verification commands (read-only, non-destructive)

```bash
# Repo identity and cleanliness
cd /mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-final-integration
git status --short
git log --oneline -n 3
git rev-parse --abbrev-ref --symbolic-full-name @{u}
git rev-list --left-right --count @{u}...HEAD

# Live policy counts (never trust a cached number without this)
source $(conda info --base)/etc/profile.d/conda.sh && conda activate repo-env
PYTHONPATH=src python3 -c "
from llmserveopt.policies.registry import BASELINE_NAMES, POLICY_LIBRARY_V2_NEW_NAMES, SELECTOR_CANDIDATE_NAMES
from llmserveopt.policies.external_baselines_registry import EXTERNAL_BASELINE_NAMES
print('total:', len(BASELINE_NAMES)+len(POLICY_LIBRARY_V2_NEW_NAMES)+len(EXTERNAL_BASELINE_NAMES))
print('selector candidates:', len(SELECTOR_CANDIDATE_NAMES))
print('faithful external:', len(EXTERNAL_BASELINE_NAMES), EXTERNAL_BASELINE_NAMES)
"

# Current Slurm queue (check before assuming anything is idle)
squeue -u "$USER" -o '%i|%j|%T|%M|%D|%R'

# Handoff JSON sanity check
python3 -c "import json; print(json.load(open('docs/current/project_handoff_state.json'))['recommended_next_step'])"
```
