# FGCS Repository Finalization Report

Date: 2026-09-20

This report closes `FGCS_REPOSITORY_FINALIZATION_QUERY_3` for the canonical
branch `contextual-compositional-heuristics-20260731`. It records repository
reconciliation and validation only; no scientific result was changed.

## 1. Canonical State

- Repository: `/home/soroush/llm-serving-heuristic-evolution`
- Branch: `contextual-compositional-heuristics-20260731`
- Pre-resume HEAD: `a8139e645f2a811dca5f1706d1af97e05b589fb3`
- Final HEAD: `PENDING_FINAL_COMMIT`
- Origin state before final commit: local and origin both at
  `a8139e645f2a811dca5f1706d1af97e05b589fb3`, `+0/-0`.
- Query-2 anchor: `946319c759b5d75203674321837c475d0fb7743a`
- Query-2 ancestry: verified as an ancestor of the canonical branch.
- Post-Query-2 commits before this finalization commit:
  - `e6197fed57c2d13c6030bbf531be4845c7be2644` -
    `docs: finalize FGCS repository provenance reconciliation`
  - `a8139e645f2a811dca5f1706d1af97e05b589fb3` -
    `docs: restore canonical resume entry point`

## 2. Primary Checkout

The resume audit found exactly one dirty tracked path: `.gitignore`.

The `.gitignore` change is retained. It adds narrow ignore rules for known
local-only Family-A provenance restored from the Query-3 archive:

- `datasets/family_a_oracle_policy_pilot_v1/`
- `datasets/family_a_oracle_policy_v1/`
- `experiments/family_a_contested_request_value_diagnosis/`
- `experiments/family_a_horizon_stability_v1/`
- `experiments/family_a_observability_continuation_v1/`

These rules are path-specific. They do not hide manuscript sources,
documentation, scripts, provenance manifests, broad `experiments/` content, or
the external Query-3 archive. They prevent known local raw/generated Family-A
artifact directories from reappearing as accidental repository dirt.

Final clean status is recorded after this report is committed.

## 3. Provenance Preservation

The external Query-3 provenance archive is retained outside Git:

`/home/soroush/llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/`

Authoritative local archive records include:

- `LOCAL_PROVENANCE_ARCHIVE_MANIFEST_V1.json`
- `LOCAL_PROVENANCE_ARCHIVE_MANIFEST_V1.md`
- `LOCAL_PROVENANCE_ARCHIVE_FILE_INVENTORY.tsv`
- `primary_checkout/git_status_porcelain_v2.txt`
- `primary_checkout/primary_tracked_changes.patch`
- `primary_checkout/primary_tracked_changes_binary.patch`
- `primary_checkout/primary_staged_changes_binary.patch`
- `primary_checkout/untracked_file_inventory.tsv`
- `primary_checkout/untracked_file_hashes.tsv`
- `primary_checkout/important_artifact_hashes.sha256`
- `fresh_latency_and_joint240_file_hashes.sha256`
- dataset, OOD-support, result, checkpoint, and worktree inventories.

The archive records 7,666 primary-checkout untracked/ignored inventory rows, 39
selected untracked file hashes, and four important large-artifact hashes. The
primary untracked archive is about 9.1 GB. The fresh-latency and joint240 raw
provenance archives are retained separately under the same archive root.

Retained refs/branches include:

- `repo-finalization/fgcs-cleanup-20260920` at
  `e6197fed57c2d13c6030bbf531be4845c7be2644`
- local and remote joint240 experiment branches listed in Section 5.

Final provenance conclusion: material identified by the Query-2/Query-3
protection manifests is either represented in canonical compact Git artifacts,
retained as local raw artifacts, retained as historical branches, or archived
with hashes outside Git. No protected scientific provenance was deleted during
this finalization.

`PROVENANCE_PRESERVATION = VERIFIED`

## 4. Fresh-Latency Reconciliation

The previous fresh-latency worktree paths are no longer active worktrees.
Canonical compact scientific outputs are represented in Git under
`experiments/fresh_production_latency_headroom_confirmatory_v1/` and linked by
current FGCS status documentation.

Raw fresh-latency provenance is preserved externally:

- Archive path:
  `/home/soroush/llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/fresh_latency/`
- Approximate size: 2.8 MB
- File count: 299 files
- Hash manifest:
  `/home/soroush/llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/fresh_latency_and_joint240_file_hashes.sha256`

No scientifically required fresh-latency commit remains outside canonical
history. The raw run material is retained for audit and reproducibility
provenance.

Final classification: `LOCAL_RAW_ARTIFACTS_RETAIN`.

## 5. Joint240 Reconciliation

The active repository has no separate joint240 worktree. Raw joint240
provenance is preserved externally:

- Archive path:
  `/home/soroush/llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/joint240/`
- Approximate size: 4.1 GB
- File count: 929 files
- Hash manifest:
  `/home/soroush/llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/fresh_latency_and_joint240_file_hashes.sha256`

Retained joint240 branches:

| Branch | SHA | Unique commits vs canonical | Disposition |
|---|---|---:|---|
| `experiment/joint240-dense-sbs-state-action-v1` | `11e9ac10898242a5c5fa95ac27174b9bba85339e` | 1 | `HISTORICAL_PROVENANCE_RETAIN` |
| `experiment/joint240-sbs-disagreement-scan-v1` | `072e1a555e75ec7c732e6bf888026ef9850c6c54` | 2 | `HISTORICAL_PROVENANCE_RETAIN` |
| `experiment/joint240-sbs-targeted-terminal-label-pilot-v1` | `e308b335776df2858941ebe2fcf00d9ce2e9d438` | 3 | `HISTORICAL_PROVENANCE_RETAIN` |
| `experiment/joint240-sbs-targeted-terminal-label-full-v1` | `0900ee7b565ee1cbfffe86272e30e4d0db13483c` | 6 | `HISTORICAL_PROVENANCE_RETAIN` |
| `experiment/joint240-sbs-advantage-learnability-v1` | `ddd94c0685f8c345e3d166337044c7b3caf15153` | 9 | `HISTORICAL_PROVENANCE_RETAIN` |

The branch-only commits add historical experiment designs, scripts, analysis
code, tests, and post-run artifacts. They are preserved as historical
scientific provenance. The current FGCS manuscript and repository-side
submission manifest do not require these branch-only files to support the
current paper state.

`DOES_CANONICAL_BRANCH_REQUIRE_ANY_JOINT240_COMMIT_TO_SUPPORT_THE_CURRENT_FGCS_MANUSCRIPT = NO_CANONICAL_INTEGRATION_REQUIRED`

Final classification: `HISTORICAL_PROVENANCE_RETAIN`.

## 6. Tests

Authoritative core command:

`PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider`

Final clean-run result: `PENDING_CLEAN_FULL_SUITE_RERUN`

The earlier dirty-tree run was intentionally interrupted after confirming the
only observed failure was the repository clean-status checker rejecting the
known uncommitted `.gitignore` change.

## 7. FGCS Figures

Generation script:
`paper/llm2026/scripts/plot_fgcs_figures.py`

The resume audit raster-compared regenerated figures from the external
Query-3 validation directory against current canonical figures. All five
matched exactly at 150 DPI:

- `paper/llm2026/figures/fgcs_pipeline.pdf` - PASS
- `paper/llm2026/figures/fgcs_disagreement_rates.pdf` - PASS
- `paper/llm2026/figures/fgcs_pressure_transition.pdf` - PASS
- `paper/llm2026/figures/fgcs_fresh_headroom.pdf` - PASS
- `paper/llm2026/figures/fgcs_regime_map.pdf` - PASS

`FGCS_FIGURE_VALIDATION = PASS`

## 8. FGCS Manuscript Build

Fresh isolated build command:

`latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/home/soroush/llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/query3_validation/fresh_latex_build_20260920_resume /home/soroush/llm-serving-heuristic-evolution/paper/llm2026/main.tex`

Build log:

`/home/soroush/llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/query3_validation/fresh_latex_build_20260920_resume/latexmk_stdout_stderr.log`

Output PDF:

`/home/soroush/llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/query3_validation/fresh_latex_build_20260920_resume/main.pdf`

Result:

- Build exit code: 0
- Page count: 13
- BibTeX: completed
- Undefined citations: none detected
- Undefined references: none detected
- Missing assets: none detected
- Significant warnings: nonblocking overfull/underfull box layout warnings.

The fresh output PDF hash differs from the checked-in canonical PDF hash due to
fresh PDF metadata, while the build succeeds and preserves the expected page
count.

`FGCS_MANUSCRIPT_BUILD = PASS`

## 9. Submission Manifest

Repository-side manifest:
`paper/llm2026/FGCS_SUBMISSION_PACKAGE_MANIFEST_V1.md`

Repository-finalization validation:

- Canonical source files exist:
  `main.tex`, `references.bib`, `llncs.cls`, `splncs04.bst`.
- All five current FGCS figures exist.
- Figure-generation script exists.
- Canonical PDF exists at `paper/llm2026/main.pdf`.
- Build manifest exists at `experiments/FGCS_MANUSCRIPT_BUILD_V1.json`.
- Reproducibility and artifact-boundary docs exist.
- No repository-side dependency points into removed worktrees or the external
  local provenance archive.
- Secret scan over current FGCS package/status surfaces found no API keys,
  private keys, password material, or common token variables.
- No multi-GB raw artifact is included in `paper/llm2026/`.

Deferred next-phase material:

- highlights, if required by FGCS;
- cover letter;
- declarations and final data/code availability wording;
- final FGCS format and metadata check;
- final journal upload/source archive assembly.

These are `NEXT_PHASE_SUBMISSION_MATERIAL`, not Query-3 blockers.

`SUBMISSION_MANIFEST_VALIDATION = PASS_FOR_REPOSITORY_FINALIZATION`

## 10. Git Integrity

Final gate commands:

- `git fsck --full`
- unresolved-conflict-marker scan
- staged-file check
- tracked-large-file review
- Query-2 ancestry check
- local/origin sync check

Final result: `PENDING_FINAL_GIT_INTEGRITY_CHECK`

## 11. Remaining Local Scientific Provenance

Retained outside Git:

- Query-3 external archive:
  `/home/soroush/llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/`
- Primary untracked archive, about 9.1 GB.
- Fresh-latency raw archive, about 2.8 MB.
- Joint240 raw archive, about 4.1 GB.
- Local raw/processed trace data inventories.
- Historical selector checkpoints.
- Historical ignored result bundles.
- Retained joint240 experiment branches.

Retained local provenance is indexed and hashed where practical and is not a
repository-finalization defect.

## 12. Remaining Warnings

Blocking:

- `PENDING_CLEAN_FULL_SUITE_RERUN`
- `PENDING_FINAL_GIT_INTEGRITY_CHECK`
- final commit/push/tag verification pending.

Nonblocking:

- Large local scientific provenance is retained outside Git.
- Fresh manuscript build has minor layout warnings.
- Branch-only joint240 work remains historical provenance rather than
  canonical integration.

Deferred to `FINAL_FGCS_SUBMISSION_AND_FORMAT_COMPLIANCE_AUDIT`:

- highlights, cover letter, declarations, journal metadata, and final upload
  package assembly.

## 13. Final Status

`REPOSITORY_FINALIZATION = PENDING_FINAL_COMMIT_AND_CLEAN_VALIDATION`

`CANONICAL_BRANCH = PENDING_FINAL_COMMIT_AND_PUSH`

`PRIMARY_CHECKOUT_RECONCILIATION = COMPLETE`

`PROVENANCE_PRESERVATION = VERIFIED`

`FRESH_LATENCY_RECONCILIATION = COMPLETE`

`JOINT240_RECONCILIATION = COMPLETE_HISTORICAL_RETAIN`

`WORKTREE_RECONCILIATION = COMPLETE_WITH_RETAINED_HISTORICAL_BRANCHES`

`LOCAL_LARGE_ARTIFACTS = RETAINED_AND_INDEXED`

`CORE_TESTS = PENDING_CLEAN_FULL_SUITE_RERUN`

`FGCS_FIGURE_VALIDATION = PASS`

`FGCS_MANUSCRIPT_BUILD = PASS`

`SUBMISSION_MANIFEST_VALIDATION = PASS_FOR_REPOSITORY_FINALIZATION`

`GIT_INTEGRITY = PENDING_FINAL_GIT_INTEGRITY_CHECK`

`FINALIZATION_REPORT = CREATED`

`SCIENTIFIC_CONTENT_CHANGED = NO`

`HISTORICAL_RESULTS_REWRITTEN = NO`

`NEW_EXPERIMENTS_RUN = NO`

`FORCE_PUSH_USED = NO`

`MILESTONE_TAG = PENDING_FINAL_GATE`

`READY_FOR_FINAL_FGCS_AUDIT = PENDING_FINAL_GATE`
