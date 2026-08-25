# Public Release Manifest

Date: 2026-08-24  
Pass: 2 (classification for a future clean public tree)

This manifest does **not** delete, push, or publish anything. It classifies
content for an allowlist-based public release.

Classification codes:

| Code | Meaning |
|---|---|
| `PUBLIC_CANONICAL` | Essential to understand/reproduce paper-relevant work |
| `PUBLIC_SUPPORTING` | Useful context; not the core evidence chain |
| `PUBLIC_ARCHIVE` | Historical scientific material safe and useful to preserve |
| `EXCLUDE_INTERNAL` | Operational/agent/scratch workflow; exclude from public tree |
| `EXCLUDE_SENSITIVE` | Credentials, token audits, private endpoints, secrets |
| `AUTHOR_DECISION` | Reasonable either way; author must choose |

## Top-level and core

| Path / pattern | Class | Rationale | Release action |
|---|---|---|---|
| `README.md` | PUBLIC_CANONICAL | Public entry | INCLUDE |
| `REPRODUCIBILITY.md` | PUBLIC_CANONICAL | Install/smoke/GPU guidance | INCLUDE |
| `LICENSE` | PUBLIC_CANONICAL | MIT | INCLUDE |
| `CITATION.cff` | PUBLIC_CANONICAL | Software citation | INCLUDE (anonymize under Strategy B) |
| `CONTRIBUTING.md` | PUBLIC_CANONICAL | Contribution norms | INCLUDE |
| `.gitignore` | PUBLIC_CANONICAL | Hygiene | INCLUDE |
| `.env.example` | PUBLIC_SUPPORTING | Placeholder-only provider vars | INCLUDE |
| `pyproject.toml` | PUBLIC_CANONICAL | Package metadata / deps | INCLUDE |
| `requirements.txt` | PUBLIC_CANONICAL | Convenience install | INCLUDE |
| `requirements-selector.txt` | PUBLIC_SUPPORTING | Optional selector deps | INCLUDE |
| `src/` | PUBLIC_CANONICAL | Simulator, policies, workloads | INCLUDE |
| `tests/` | PUBLIC_CANONICAL | Validation / regression | INCLUDE (GPU tests remain opt-in) |
| `scripts/smoke_test.py` | PUBLIC_CANONICAL | Documented smoke path | INCLUDE |
| `scripts/` (remainder) | PUBLIC_SUPPORTING | Mixed current/historical runners | INCLUDE selected; exclude overnight/HPC-only if desired — default INCLUDE tracked scripts except scratch |
| `configs/` | PUBLIC_SUPPORTING | Experiment configs | INCLUDE |
| `baselines/` | PUBLIC_SUPPORTING | External baseline adapters + provenance | INCLUDE |
| `benchmarks/` | PUBLIC_SUPPORTING | Workload suite defs | INCLUDE |
| `tools/` | PUBLIC_SUPPORTING | Small helpers | INCLUDE |
| `external/` | PUBLIC_ARCHIVE | External resource notes | INCLUDE |
| `.github/PULL_REQUEST_TEMPLATE.md` | PUBLIC_SUPPORTING | PR template | INCLUDE |

## Documentation

| Path / pattern | Class | Rationale | Release action |
|---|---|---|---|
| `docs/REPOSITORY_MAP.md` | PUBLIC_CANONICAL | Directory roles | INCLUDE |
| `docs/RESULTS_INDEX.md` | PUBLIC_CANONICAL | Paper evidence index | INCLUDE |
| `docs/PUBLIC_RELEASE_CHECKLIST.md` | PUBLIC_SUPPORTING | Release checklist | INCLUDE |
| `docs/PUBLIC_RELEASE_MANIFEST.md` | PUBLIC_SUPPORTING | This file | INCLUDE |
| `docs/PUBLIC_RELEASE_ALLOWLIST.txt` | PUBLIC_SUPPORTING | Allowlist | INCLUDE |
| `docs/PUBLIC_RELEASE_EXCLUDELIST.txt` | PUBLIC_SUPPORTING | Excludelist | INCLUDE |
| `docs/DATA_RELEASE_POLICY.md` | PUBLIC_CANONICAL | Data redistribution policy | INCLUDE |
| `docs/PUBLIC_RELEASE_AUTHOR_DECISIONS.md` | PUBLIC_SUPPORTING | Author checkboxes | INCLUDE |
| `docs/README.md` | PUBLIC_SUPPORTING | Docs index | INCLUDE |
| `docs/PROJECT_MAP.md` | PUBLIC_ARCHIVE | Long-term roadmap (may lag) | INCLUDE |
| `docs/audits/` | PUBLIC_ARCHIVE | Immutable scientific audits | INCLUDE |
| `docs/architecture/`, `docs/design/`, `docs/experiments/`, `docs/milestones/`, `docs/paper/`, `docs/research/`, `docs/templates/` | PUBLIC_ARCHIVE | Design/historical docs | INCLUDE |
| `docs/current/llm2026_*` | PUBLIC_SUPPORTING | Manuscript number/claim/provenance ledgers | INCLUDE |
| `docs/current/*_analysis_202608*.md` (scientific analyses) | PUBLIC_SUPPORTING | Frozen analysis writeups linked from results index | INCLUDE |
| `docs/current/RESUME_HERE.md` | EXCLUDE_INTERNAL | Operational handoff | EXCLUDE |
| `docs/current/NEXT_ACTIONS.md` | EXCLUDE_INTERNAL | Prioritized next actions | EXCLUDE |
| `docs/current/WORK_STATUS.md` | EXCLUDE_INTERNAL | Live status table | EXCLUDE |
| `docs/current/AGENT_HANDOFF.md` | EXCLUDE_INTERNAL | Agent continuity | EXCLUDE |
| `docs/current/*HANDOFF*` | EXCLUDE_INTERNAL | Handoff notes | EXCLUDE |
| `docs/current/project_handoff_state.json` | EXCLUDE_INTERNAL | Machine status blob | EXCLUDE |
| `docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md` | EXCLUDE_INTERNAL | Internal protection list | EXCLUDE |
| `docs/START_HERE_CONTEXTUAL_COMPOSITION.md`, `docs/RESUME_CONTEXTUAL_COMPOSITION.md`, `docs/CONTEXTUAL_COMPOSITION_BRANCH.md` | EXCLUDE_INTERNAL | Branch/resume workflow | EXCLUDE |
| `docs/current/public_repository_readiness_audit_20260824.md` | PUBLIC_SUPPORTING | Release audit trail | INCLUDE |

## Experiments and paper evidence

| Path / pattern | Class | Rationale | Release action |
|---|---|---|---|
| `experiments/joint_multimechanism_generalization_v1/` | PUBLIC_CANONICAL | Core complementarity | INCLUDE (summaries + runner; skip `__pycache__`) |
| `experiments/unified_utility_matrix_v2/` | PUBLIC_CANONICAL | 176×6 matrix | INCLUDE |
| `experiments/public_trace_replay_v1/` | PUBLIC_CANONICAL | Public-trace saturation | INCLUDE (manifests/provenance; not raw traces) |
| `experiments/multifamily_contextual_selector_v1/` | PUBLIC_CANONICAL | Selector NO_GO | INCLUDE |
| `experiments/hierarchical_regime_router_live_reeval_v1/` | PUBLIC_CANONICAL | Router live gate | INCLUDE |
| `experiments/family_a_wulver_dev_support_eval_v1/` | PUBLIC_CANONICAL | Support gate | INCLUDE |
| `experiments/family_a_mechanism_composite_rule_static_feasibility_v1/` | PUBLIC_CANONICAL | Guarded composition | INCLUDE |
| `experiments/portfolio_guided_typed_gp_screen_v1/` | PUBLIC_CANONICAL | Typed GP screen | INCLUDE |
| `experiments/real_vllm_mechanism_validation_v1/` | PUBLIC_CANONICAL | Real-vLLM chain | INCLUDE (summaries/configs/runners; exclude bulky server logs if present) |
| Other `experiments/*` with scientific audits | PUBLIC_ARCHIVE / PUBLIC_SUPPORTING | Precursor studies | INCLUDE when tracked and non-sensitive |
| `experiments/**/server.log` | EXCLUDE_SENSITIVE / generated | Runtime logs | EXCLUDE |
| `experiments/**/__pycache__/` | EXCLUDE_INTERNAL | Cache | EXCLUDE |

## Manuscript

| Path / pattern | Class | Rationale | Release action |
|---|---|---|---|
| `paper/llm2026/main.tex`, `references.bib`, `figures/`, `scripts/`, template/cls | AUTHOR_DECISION | Identified manuscript package | INCLUDE under Strategy A; sanitize/exclude under Strategy B |
| `paper/llm2026/main.pdf` | AUTHOR_DECISION | Compiled PDF already on `origin/main` | INCLUDE under Strategy A |
| `paper/llm2026/main.log`, `main.blg`, aux | EXCLUDE_INTERNAL | Build junk | EXCLUDE |

## Data / artifacts / results

| Path / pattern | Class | Rationale | Release action |
|---|---|---|---|
| `data/README.md` | PUBLIC_CANONICAL | Data layout + BurstGPT MIT note | INCLUDE |
| `data/public_trace_corpus_v1/manifest.json`, `schema.json`, `distribution_stats.json`, `source_coverage.csv` | PUBLIC_CANONICAL | Corpus metadata | INCLUDE |
| `data/public_trace_corpus_v1/*/records.parquet` | AUTHOR_DECISION | Derived tables; bulky; redistribution TBD | EXCLUDE from default dry-run; author check |
| `data/raw/**` | AUTHOR_DECISION / EXCLUDE | Raw BurstGPT/Azure CSVs present locally | **EXCLUDE** from public tree by default; download-from-source |
| `data/processed/**` | EXCLUDE_INTERNAL | Generated; gitignored | EXCLUDE |
| `datasets/` | AUTHOR_DECISION | Family-A oracle/relabel roots | EXCLUDE default dry-run |
| `artifacts/` | AUTHOR_DECISION | Wulver transfer bundles | EXCLUDE default dry-run |
| `dataset_staging/`, `.local_data/`, `hf_*`, `llmserveopt-data/` | EXCLUDE_INTERNAL | Caches/staging | EXCLUDE |
| `results/` (bulk) | EXCLUDE_INTERNAL | Generated | EXCLUDE |
| `results/provenance/` | PUBLIC_ARCHIVE | Selected committed provenance | INCLUDE if present |
| `logs/`, `*.log`, `crash.log`, `run.log` | EXCLUDE_INTERNAL | Runtime logs | EXCLUDE |

## Scratch / agent / sensitive

| Path / pattern | Class | Rationale | Release action |
|---|---|---|---|
| `p2_config.yaml`, `p3_chunk_control.py`, `p5_analysis_chunk_comp.py`, `p7_runner.py`, `p8_test_runner.py` | EXCLUDE_INTERNAL | Root scratch | EXCLUDE |
| `opencode.json` | EXCLUDE_SENSITIVE | Local agent provider config | EXCLUDE |
| `.claude/` | EXCLUDE_INTERNAL | Agent memory | EXCLUDE |
| `.env` | EXCLUDE_SENSITIVE | Secrets | EXCLUDE (should not exist in release) |
| `results/baseline_api_audit/**` | EXCLUDE_SENSITIVE | Masked token audit | EXCLUDE |
| `.coverage`, `__pycache__/`, `.pytest_cache/` | EXCLUDE_INTERNAL | Tooling | EXCLUDE |

## Notes on provenance absolute paths

Many included experiment JSON files contain `/home/soroush/...` or `/mmfs1/...` as
**run-site provenance**. Classification: `PROVENANCE_SAFE` for scientific
records; do not treat as install instructions. Public docs must remain portable
(already true for README / REPRODUCIBILITY).
