# Repository Map

Public-facing map of major directories. Prefer this over treating every folder
under `docs/current/` or `experiments/` as equally canonical.

## Top-level

| Path | Role |
|---|---|
| `src/llmserveopt/` | **Canonical library:** simulator, policies, workloads, selector/router, GP scaffolding |
| `tests/` | Automated tests (CPU default; GPU marked opt-in) |
| `scripts/` | Runners, analysis, and maintenance utilities (mix of current and historical) |
| `experiments/` | Per-study runners + frozen summaries used by audits and the manuscript |
| `paper/llm2026/` | LNCS manuscript, figures, compile artifacts |
| `docs/` | Roadmaps, design notes, audits, public indexes |
| `configs/` | YAML/JSON configs for experiments and calibration |
| `baselines/` | External baseline adapters + provenance notes |
| `benchmarks/` | Canonical workload suite definitions |
| `data/` | Local datasets; bulky raw/processed files gitignored |
| `results/` | Local generated outputs; mostly gitignored |
| `logs/` | Runtime logs; gitignored |
| `artifacts/` | Bundled transfer packages (e.g. cluster sweep bundles); not all public-facing |
| `datasets/` | Derived dataset roots used by some Family-A pipelines |
| `external/` | Notes / pointers for external resources |
| `tools/` | Small helper utilities |

## Documentation tiers

| Path | Treat as |
|---|---|
| `README.md`, `REPRODUCIBILITY.md` | Public entry points |
| `docs/RESULTS_INDEX.md` | Paper-relevant frozen evidence index |
| `docs/PUBLIC_RELEASE_CHECKLIST.md` | Release / anonymity checklist |
| `docs/PROJECT_MAP.md` | Long-term research roadmap (may lag manuscript freeze) |
| `docs/audits/` | Immutable historical audits |
| `docs/current/` | Working notes, analysis writeups, handoffs—**mixed freshness**; operational handoffs are **excluded** from the public allowlist |
| `docs/DATA_RELEASE_POLICY.md`, `docs/PUBLIC_RELEASE_*` | Public-release classification, data policy, author decisions |
| `docs/current/RESUME_HERE.md`, `WORK_STATUS.md`, `NEXT_ACTIONS.md` | Internal operational status — **excluded** from public allowlist |
| `docs/DATA_RELEASE_POLICY.md`, `docs/PUBLIC_RELEASE_*` | Public-release classification and author decisions |

## Experiment roles (paper chain)

**Canonical manuscript evidence** (see `docs/RESULTS_INDEX.md`):

- `experiments/unified_utility_matrix_v2/`
- `experiments/joint_multimechanism_generalization_v1/`
- `experiments/multifamily_contextual_selector_v1/` (and related selector audits)
- `experiments/hierarchical_regime_router_live_reeval_v1/`
- `experiments/family_a_wulver_dev_support_eval_v1/`
- `experiments/family_a_mechanism_composite_rule_static_feasibility_v1/`
- `experiments/portfolio_guided_typed_gp_screen_v1/`
- `experiments/real_vllm_mechanism_validation_v1/`
- `experiments/public_trace_replay_v1/`

**Supporting / precursor studies:** Family A/B/C pilots, MF-PSD, composition
falsifications, smoke diagnoses—useful provenance, not all cited as main
results.

**Historical / optional:** `experiments/real_llm/` hosted-API calibrations,
many Wulver-era sweeps documented primarily via audits and external cluster
storage.

## Generated vs source

| Keep / publish carefully | Usually local-only |
|---|---|
| Frozen JSON/CSV summaries under `experiments/` | `results/*` bulk outputs |
| `paper/llm2026/main.pdf` + figures | `logs/`, `*.log`, coverage caches |
| Provenance manifests | HF / model caches, virtualenvs |
| Selected `results/provenance/` | Raw third-party traces |

## Scratch / confusing roots (do not treat as canonical)

These exist in the working tree and may be tracked historically:

- Root `p2_config.yaml`, `p3_chunk_control.py`, `p5_analysis_chunk_comp.py`,
  `p7_runner.py`, `p8_test_runner.py` — one-off scratch, not public entry points
- `opencode.json` — local agent/provider UI config
- `.claude/` — local agent memory (gitignored)
- `docs/current/*RESUME*`, `AGENT_HANDOFF.md` — internal continuity notes
