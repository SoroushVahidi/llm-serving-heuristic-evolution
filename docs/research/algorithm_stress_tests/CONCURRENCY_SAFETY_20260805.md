# Algorithm Stress-Test Library — Concurrency Safety Check (2026-08-05)

## Repository state at start

- Branch: `contextual-compositional-heuristics-20260731`
- SHA: `07c79d904693d30ab51247158846030e639cfd48`
- Upstream: `origin/contextual-compositional-heuristics-20260731`, 0 ahead / 0 behind
- Working tree: clean

## VTC finalization job status

**Already complete, not running.** Checked directly: no tmux sessions
(`tmux ls` → "no server running"), no `pytest`/`python .../vtc*`/
`python .../sweep*` processes in the process table, working tree clean,
local SHA matches remote SHA exactly. The VTC fairness-validation work
(commit `07c79d9`, "baseline: validate VTC fairness under matched
admission") was committed and pushed in the immediately preceding task,
before this one began. There is therefore no live concurrency conflict to
avoid — but the exclusion list below is kept anyway, both as a safety
margin against a VTC follow-up landing mid-session and because this task's
own scope has no legitimate reason to touch VTC's files.

## Resource headroom (checked directly)

- CPU: 20 cores, idle
- RAM: 62Gi total, 59Gi available
- Disk: 700G filesystem, 286G available (57% used)
- GPU: RTX 5060 Ti, 16311 MiB total, 15 MiB used, 0% utilization — fully free

## Duplicate-task check

No other `stress_test`-related process or tmux session found. This is the
only active instance of this task.

## Exclusion list — this task must not modify or commit

- `baselines/vtc/**` (all VTC adapter code, provenance, workloads, sweep/smoke results)
- `docs/audits/vtc_*.md` (all four VTC audit documents)
- `scripts/*vtc*.py` (`run_vtc_smoke_evaluation.py`,
  `decompose_vtc_smoke_confound.py`, `check_vtc_fairness_headroom.py`,
  `run_vtc_fairness_comparative_sweep.py`, `verify_vtc_fairness_sweep.py`)
- `tests/test_vtc_*.py` (`test_vtc_baseline_adapter.py`,
  `test_vtc_fairness_headroom.py`, `test_vtc_micro_traces.py`)
- `docs/BASELINE_STATUS.md`, `docs/baselines.md`, `docs/roadmap.md`,
  `docs/external_baseline_decision.md` — these carry VTC's finalized
  status text; this task reads them (for the algorithm inventory) but
  does not edit them
- `src/llmserveopt/core/types.py`, `src/llmserveopt/core/action.py`,
  `src/llmserveopt/simulator/simulator.py` — CC5/CC6-adjacent core
  infrastructure; this task's workload generators and stress-test
  policies must be additive (new files) and must not require editing
  these
- `benchmarks/canonical_suite/**` — the accepted canonical benchmark
  suite; this task's stress-test workloads are a clearly separate,
  labeled extension, exactly as `baselines/vtc/fairness_workloads.py`
  already established as this project's pattern for non-canonical
  workload additions
- Any `configs/cc*.yaml` file (CC5/CC6 experiment configs)

## Working paths for this task

Per the task's own suggested layout (all new, none overlapping the
exclusion list above):

```
docs/research/algorithm_stress_tests/   # literature research + inventory
configs/stress_tests/                   # machine-readable catalog YAML
scripts/stress_tests/                   # workload generators + runner
tests/stress_tests/                     # tests for the generators
results/stress_test_catalog/            # any smoke-run output
```

No staging under `/tmp` was necessary — the working paths above do not
conflict with anything VTC-related or with canonical/CC5/CC6 files, so
canonical repository paths are safe to use directly and commits are not
blocked pending another task's completion.
