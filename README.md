# llm-serving-heuristic-evolution

Research code for **contextual, compositional scheduler synthesis for LLM
inference serving**.

The repository studies when a serving system should select, combine, or
synthesize scheduling policies under changing workload pressure. It includes a
GPU-calibrated discrete-event simulator, a library of internal policies,
faithful external scheduler integrations, a typed scheduling DSL, contextual
performance models, and reproducible experiment/audit artifacts.

## Start Here

Documentation authority is intentionally narrow:

1. [`README.md`](README.md) - public project overview and navigation.
2. [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md) - canonical long-term roadmap.
3. [`docs/current/RESUME_HERE.md`](docs/current/RESUME_HERE.md) - shortest current operational handoff.
4. [`docs/current/WORK_STATUS.md`](docs/current/WORK_STATUS.md) - detailed current status table.
5. [`docs/current/NEXT_ACTIONS.md`](docs/current/NEXT_ACTIONS.md) - prioritized next actions.
6. [`docs/BASELINE_STATUS.md`](docs/BASELINE_STATUS.md) - external-baseline status index.
7. [`docs/audits/`](docs/audits/) - immutable point-in-time audit trail.

If a status claim elsewhere conflicts with these files, treat it as historical
until reconciled.

## Research Objective

The project is not just a fixed-policy benchmark and not just an Apt-Serve
reproduction. The target system is a verified contextual compositional
hyper-heuristic:

```text
workload/state context
  -> policy/module performance modeling
  -> uncertainty, pairwise advantage, marginal contribution
  -> typed DSL / AST
  -> parent and module selection
  -> structural composition / symbolic synthesis
  -> verification
  -> evaluation
  -> policy-library envelope expansion
  -> iteration
  -> real-system validation
```

The primary metric for current work is
`arrival_normalized_weighted_goodput` (ANWG): weighted SLO goodput normalized
by all arriving requests. Completion-conditioned quality is tracked only as a
secondary diagnostic.

## Current Checkpoint

Current branch: `contextual-compositional-heuristics-20260731`.

As of the latest reconciliation, the most recent major local experiment is
**Apt-Serve Phase G**:

- collection: complete;
- posthoc analysis: complete with exit code 0;
- canonical analysis artifact:
  `results/apt_serve_phase_g_analysis_20260809_190000/`;
- scientific audit:
  [`docs/audits/apt_serve_phase_g_analysis_20260809.md`](docs/audits/apt_serve_phase_g_analysis_20260809.md).

The supported Phase G result is deliberately narrow: Apt-Serve has a positive
leave-one-out marginal contribution to the policy portfolio with a bootstrap CI
excluding zero, but global superiority over the best fixed baseline is **not**
established because the Apt-vs-best-fixed CI crosses zero. Apt-Serve remains one
external scheduler family and a source of cache/tier-transition mechanisms, not
the whole project or proof that compositional synthesis works.

The canonical next task is to reconcile the completed Phase G interpretation
into the broader module-decomposition and library-envelope roadmap, then return
to contextual composition work rather than launching another Apt-Serve sweep.

## Repository Layout

```text
src/llmserveopt/    library code: simulator, policies, DSL, selector, workloads
tests/              pytest suite, including historical phase regression tests
scripts/            experiment runners, analysis scripts, maintenance tools
configs/            YAML/JSON experiment and calibration configs
baselines/          official-code adapters and provenance for external baselines
benchmarks/         canonical workload suites
experiments/        small committed experiment artifacts and curated provenance
docs/               roadmap, current status, design docs, historical audits
data/               local datasets; raw/processed data are gitignored
results/            local generated outputs; gitignored except selected provenance
logs/               local runtime logs; gitignored
```

See [`docs/README.md`](docs/README.md), [`scripts/README.md`](scripts/README.md),
and [`configs/README.md`](configs/README.md) for detailed navigation.

## Install

```bash
python3 -m pip install -e ".[dev]"
python3 -c "import llmserveopt, pandas"
```

Optional extras:

```bash
python3 -m pip install -e ".[selector]"   # selector/suitability models
python3 -m pip install -e ".[vllm_ltr]"   # checkpoint-loading tests only
```

`requirements.txt` and `requirements-selector.txt` are convenience equivalents
for environments that do not install from `pyproject.toml`.

## Test

```bash
python3 -m pytest tests/test_project_handoff_consistency.py tests/test_apt_serve_phase_g_analysis.py -q
python3 -m pytest --collect-only -q
```

GPU/checkpoint tests are opt-in:

```bash
LLMSERVEOPT_RUN_GPU_TESTS=1 python3 -m pytest -m gpu
```

## Reproduce Key Workflows

- Phase G collection runner: `python scripts/run_apt_serve_phase_g.py --help`
- Phase G analysis runner: `python scripts/analyze_apt_serve_phase_g.py --help`
- Status consistency check: `python scripts/check_project_handoff_consistency.py`
- General smoke test: `python scripts/smoke_test.py`

Most full experiment runs write to `results/` and should be launched in tmux or
the cluster scheduler. See the relevant audit before rerunning any major
experiment; generated results are intentionally not version-controlled.

## License

MIT. See [`LICENSE`](LICENSE).
