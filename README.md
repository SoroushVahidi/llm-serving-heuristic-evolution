# llm-serving-heuristic-evolution

Research code for **online LLM-inference serving scheduler portfolios**: when
should a serving system switch between (rather than re-combine) scheduling
policies as workload pressure changes?

The repository contains a GPU-calibrated discrete-event simulator, a library of
internal scheduling policies, faithful external-scheduler integrations
(Apt-Serve, VTC, PARS, and others), trace-based counterfactual evaluation on
public workload corpora, contextual policy-selection models, and instrumented
real-vLLM mechanism validation. The primary metric throughout is
`arrival_normalized_weighted_goodput` (ANWG): weighted SLO goodput normalized
by all arriving requests.

## Manuscript

A finalized 15-page LNCS manuscript, **"The Exploitability Gap in
LLM-Serving Scheduler Portfolios"**, is included at
[`paper/llm2026/`](paper/llm2026/) (source, PDF, figures, and figure
regeneration scripts). The manuscript was submitted to the LLM 2026
conference on August 25, 2026. The paper's core
results are summarized under *Key Findings* below.

## Research Problem

LLM serving workloads stress different scheduling mechanisms — prompt
processing, urgent-deadline protection, KV-cache capacity — so no single
simple rule dominates all regimes. This project asks, in the language of
algorithm selection:

1. How much headroom does a scheduler **portfolio** (virtual-best per-scenario
   choice, VBS) hold over the best fixed single policy (SBS) on realistic
   joint workloads?
2. Can lightweight **online contextual selectors** capture that headroom
   without per-request overhead or unsafe policy switching?
3. Does **within-scenario composition/synthesis** of parent policies add
   value beyond scenario-level selection?
4. Do simulator-based conclusions transfer to a **real serving engine**?

## Key Findings

From the joint-240 scenario suite (paper and `docs/audits/`):

- Per-scenario best-policy choice improves ANWG by **0.0190 over the best
  fixed policy (SBS)**; lightweight online adapters fall *below* SBS, and a
  stronger nonlinear cost-sensitive utility selector reaches near-SBS ANWG
  but recovers only ~2.5% of the SBS→VBS headroom (gain CI includes zero).
- Terminal one-step counterfactuals are sparse and concentrated, and native
  policy disagreement predicts them only moderately; exact critical-state
  identity is not invariant across continuation policies.
- **Within-scenario composition was demoted after a structural reassessment**
  (2026-08-17 audit): for the policy families studied, a scenario-level
  selector matched the parent oracle (Families A/B), while the only
  composition gain observed (KV family) relied on violating parent safety
  invariants and vanished once safety constraints were enforced. Composition
  and typed-DSL synthesis remain exploratory future work, not the central
  hypothesis.
- The revised roadmap is: policy-separating workloads → complementary policy
  library → contextual selection (multi-family) → mechanism attribution →
  bounded envelope.
- Apt-Serve shows a positive leave-one-out marginal contribution to the
  policy portfolio (bootstrap CI excluding zero), but global superiority over
  the best fixed baseline is **not** established.

## Evidence Types

The repository deliberately separates two kinds of evidence:

- **Trace-based counterfactual evaluation** — GPU-calibrated discrete-event
  simulation and replay of public trace corpora (e.g., Public Trace Corpus
  v1) under alternative scheduler policies. Cheap, controlled, and
  reproducible, but simulator-based.
- **Real-system validation** — instrumented local vLLM runs (e.g., Qwen2.5
  models under vLLM) checking whether simulator-level mechanism conclusions
  hold on a real engine. These runs revealed a simulator–engine semantic
  mismatch and a native token-budget tradeoff, so simulator results are not
  presented as production measurements.

Calibration provenance (cluster GPU profiles) is documented under
`configs/calibration/`.

## Repository Layout

```text
src/llmserveopt/    library code: simulator, policies, DSL, selector, workloads
tests/              pytest suite, including historical phase regression tests
scripts/            experiment runners, analysis scripts, maintenance tools
configs/            YAML/JSON experiment and calibration configs
baselines/          official-code adapters and provenance for external baselines
benchmarks/         canonical workload suites
experiments/        committed experiment artifacts and curated provenance
docs/               roadmap, current status, design docs, historical audits
data/               local datasets; raw/processed data are gitignored
results/            local generated outputs; gitignored except selected provenance
paper/llm2026/      finalized manuscript package (LaTeX source, PDF, figures)
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
- Paper figures (from frozen artifacts, no new experiments):
  `python3 paper/llm2026/scripts/plot_joint_complementarity.py` and
  `python3 paper/llm2026/scripts/plot_vllm_semantic_validation.py`

Most full experiment runs write to `results/` and should be launched in tmux or
the cluster scheduler. See the relevant audit under `docs/audits/` before
rerunning any major experiment; generated results are intentionally not
version-controlled. A template for API credentials is provided in
`.env.example`; real credentials are never committed.

## Documentation

Long-term roadmap: [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md). The
shortest current operational entry point is
[`docs/current/RESUME_HERE.md`](docs/current/RESUME_HERE.md); detailed
status, next actions, and the working-branch audit trail live under
[`docs/current/`](docs/current/) and [`docs/audits/`](docs/audits/). If a
status claim elsewhere conflicts with those files, treat it as historical
until reconciled. External-baseline status:
[`docs/BASELINE_STATUS.md`](docs/BASELINE_STATUS.md).

## License

MIT. See [`LICENSE`](LICENSE).
