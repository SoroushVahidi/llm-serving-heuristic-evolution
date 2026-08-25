# Public Repository Readiness Audit

Date: 2026-08-24

Scope: repository-quality / publication-readiness. No new scientific
experiments, no manuscript scientific edits, no visibility changes, no push.

## Initial repository state

| Field | Value |
|---|---|
| Branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `2987b7181efa2bc550d8a894c537eca8f6393eb6` |
| Upstream | `origin/contextual-compositional-heuristics-20260731` |
| Ahead/behind | 0 ahead / 2 behind |
| Worktrees | single research worktree |
| Locks | none |
| Active scientific jobs / tmux | none observed |
| Dirty/untracked | large pre-existing set (docs/current analyses, experiments/, paper/, scripts, …) — preserved |

## Public-release risks found

1. **Stale public README** centered on Apt-Serve Phase G and internal handoff
   docs; poor outsider orientation for the LLM 2026 evidence chain.
2. **Identifying metadata** throughout (author, institution, GitHub URL,
   acknowledgments) — fine for open release, conflict risk for double-blind.
3. **Internal operational docs** (`RESUME_HERE`, `NEXT_ACTIONS`, agent handoffs)
   mixed into `docs/current/` alongside scientific analyses.
4. **Root scratch scripts** (`p2_*`…`p8_*`) tracked and confusing.
5. **Hard-coded `/home/soroush/...`** in two shell scripts and many provenance
   JSON files.
6. **Gitignored but sensitive-adjacent** masked API-key audit log under
   `results/baseline_api_audit/` (must stay unpublished).
7. **Third-party traces** redistribution not auto-cleared for public upload.
8. **`paper/llm2026/README.md`** still said “scaffold only” despite completed
   manuscript PDF.

## Secrets / sensitive-data findings (redacted)

| Finding | Classification | Action |
|---|---|---|
| No full `sk-` / `ghp_` / PEM private-key material in working-tree text scan | Clear for now | Re-scan before visibility flip |
| `.env` absent; `.env.example` placeholder-only | OK | Keep gitignored `.env` |
| `results/baseline_api_audit/token_env_audit.log` contains **masked** provider key prefixes/lengths | Gitignored; should never be published | Author confirm stays local |
| `opencode.json` CloudRift public base URL, no key | Local agent config | Added to `.gitignore` for future; still tracked historically |
| `.claude/` agent memory | Already gitignored | Keep ignored |

## Portability findings

| Class | Examples | Disposition |
|---|---|---|
| Public scripts with absolute `cd` | `scripts/phase17c_postprocess.sh`, `scripts/run_family_a_v2_postrelabel_overnight.sh` | **Fixed** to repo-relative `ROOT` |
| Provenance JSON absolute paths | many under `experiments/`, `data/`, `artifacts/` | Retain as run-site provenance; documented as non-install paths |
| HPC paths in historical docs | `/mmfs1/project/...` in `EXPERIMENT_INDEX` | Historical; author may filter on release branch |

## Documentation problems (pre-change)

- README contradiction: Apt-Serve “current checkpoint” vs “checkpoint moved”
- `docs/PROJECT_MAP.md` HEAD/date lag vs manuscript freeze
- `paper/llm2026/README.md` outdated scaffold wording
- No public results index / reproducibility doc / release checklist at
  docs root

## Confusing / internal artifacts (preserved)

| Item | Decision |
|---|---|
| `docs/current/RESUME_*`, `NEXT_ACTIONS`, `AGENT_HANDOFF` | Keep; label as internal via repository map / checklist |
| Root `p2`–`p8` scratch | Keep; author decide move/delete later |
| `artifacts/`, `datasets/` untracked bundles | Preserve; author decide inclusion |
| Historical audits under `docs/audits/` | Keep (reproducibility evidence) |
| Manuscript audits under `docs/current/llm2026_*` | Keep |

## Changes actually made

- Rewrote public `README.md`
- Added `REPRODUCIBILITY.md`
- Added `docs/REPOSITORY_MAP.md`, `docs/RESULTS_INDEX.md`,
  `docs/PUBLIC_RELEASE_CHECKLIST.md`
- Updated `CITATION.cff`, `CONTRIBUTING.md`, `paper/llm2026/README.md`,
  `docs/README.md`
- Updated `LICENSE` copyright line to name author + 2025–2026
- Improved `.gitignore` (explicit crash/run logs; `opencode.json`)
- Portable `ROOT` in two shell scripts

## Files deliberately preserved

All pre-existing dirty/untracked scientific work, manuscript tree, experiment
artifacts, and internal status docs.

## Blockers requiring author decision

See `docs/PUBLIC_RELEASE_CHECKLIST.md` sections **NEEDS AUTHOR DECISION** and
**BLOCKER BEFORE PUBLIC RELEASE**.

## Validation results

- `python3 scripts/smoke_test.py` — PASSED
- `python3 -m py_compile` on smoke + figure scripts — OK
- `pytest tests/test_project_handoff_consistency.py` — 8 passed
- `git diff --check` on edited files — clean
- Secret-pattern rescan — no full secret matches in scanned tree (excluding
  gitignored bulk results contents beyond known masked audit)
- Public entry docs contain no `/home/soroush` paths
- README-linked canonical artifact paths exist on disk

## Exact recommended next step

Author-led **public-release decision pass**: (1) confirm secrets/data license,
(2) decide whether operational `docs/current` handoffs and root scratch ship,
(3) choose public-now vs post-acceptance visibility, (4) only then create a
clean public tag/branch and flip visibility—without merging unfinished
research-branch dirt blindly into `main`.

---

## Pass 2 — Clean release classification and dry-run (2026-08-24)

### Release manifest result

Created `docs/PUBLIC_RELEASE_MANIFEST.md` classifying paths into
`PUBLIC_CANONICAL`, `PUBLIC_SUPPORTING`, `PUBLIC_ARCHIVE`, `EXCLUDE_INTERNAL`,
`EXCLUDE_SENSITIVE`, and `AUTHOR_DECISION`.

Allowlist / excludelist:

- `docs/PUBLIC_RELEASE_ALLOWLIST.txt`
- `docs/PUBLIC_RELEASE_EXCLUDELIST.txt`

Author-only checklist:

- `docs/PUBLIC_RELEASE_AUTHOR_DECISIONS.md`

### Third-party data result

Created `docs/DATA_RELEASE_POLICY.md`.

- Default public tree **excludes** raw BurstGPT/Azure CSVs and parquet tables.
- Local evidence: BurstGPT MIT (`data/README.md`); Azure CC-BY-4.0 claimed in
  `docs/PROJECT_MAP.md` only → both remain
  `AUTHOR_EXTERNAL_LICENSE_CHECK_REQUIRED` before any raw redistribution.
- Manuscript-safe stance: ship manifests + download-from-source instructions.

### History-sensitive result

- Tracked history scan: only `.env.example` for env templates; **no** committed
  `.env`, credential files, PEM/private keys, or `token_env_audit` logs.
- Working-tree: no full live secrets in scanned text; masked token audit remains
  gitignored under `results/baseline_api_audit/`.
- `opencode.json` is gitignored and not in commit history (status: ignored).
- **No history rewrite required** based on this scan; author should still
  confirm before visibility flip.

### Anonymity strategy

Documented as Strategy A (identified) vs Strategy B (anonymized) in the final
report / author decisions. Venue policy **not** determined from local evidence.

### Dry-run result

Local dry-run tree:

`/tmp/llm-serving-public-release-dryrun-20260824`

- ~1950 files copied via allowlist
- Absent: handoffs, root scratch, `artifacts/`, `datasets/`, raw BurstGPT CSV,
  `opencode.json`, `docs/INDEX.md`
- Present: README, REPRODUCIBILITY, LICENSE, src/, tests/, canonical
  experiments, paper package, data manifests, release docs

### Clean-environment result

Inside dry-run:

- `PYTHONPATH=src python3 scripts/smoke_test.py` — PASSED
- Public entry Markdown relative links — no missing targets
- No full secret patterns found
- Absolute `/home/soroush` and `/mmfs1` paths remain inside some provenance
  JSON/historical docs (`PROVENANCE_SAFE`); public entry docs are portable

### Remaining author decisions

See `docs/PUBLIC_RELEASE_AUTHOR_DECISIONS.md` (checkboxes only).

### Technical readiness

**Technically ready for publication pending author decisions.** Remaining
blockers are author/venue/license/visibility choices, not unfinished
engineering of the clean release content set.
