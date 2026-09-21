#!/usr/bin/env python3
"""Build the deterministic Performance Evaluation reproducibility release bundle.

The bundle mirrors the repository layout for every included path, so the paper's own scripts run inside it
unchanged.  It is a pure function of (git commit, local continuation-shard archive, DOI arguments): file order,
zip timestamps and metadata carry no wall-clock time.  Only *tracked* repository files are copied (via
``git ls-files``), so caches, logs, worktrees and untracked local material can never enter the bundle.

    python3 scripts/build_performance_evaluation_release.py \
        --version 1.1.0 --tag performance-evaluation-v1.1.0 \
        --archive-root ../llm-serving-heuristic-evolution-local-provenance

Writes ``release/performance_evaluation_v<version>/`` and ``release/performance_evaluation_v<version>.zip``.
Excluded on purpose: the partial ``fresh-latency-confirmatory-v1`` provenance directory (not canonical), raw
third-party traces, credentials, logs, and every untracked file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TITLE = ("When Does LLM-Serving Scheduler Adaptation Matter? Action Opportunity and Causal Headroom in "
         "Production-Derived Replay")
GITHUB = "https://github.com/SoroushVahidi/llm-serving-heuristic-evolution"
SHARD_RUN_REL = "fgcs-finalization-20260920/fresh_latency/fresh-latency-execution-v1/fresh_latency_causal_run_v1"
SHARD_BUNDLE_ROOT = "provenance_archive"

# (path or directory, role); directories are expanded through ``git ls-files``.
INCLUDE = [
    ("paper/performance_evaluation/main.tex", "manuscript_source"),
    ("paper/performance_evaluation/references.bib", "manuscript_source"),
    ("paper/performance_evaluation/main.bbl", "manuscript_source"),
    ("paper/performance_evaluation/README.md", "documentation"),
    ("paper/performance_evaluation/FINAL_CLAIM_MANIFEST.json", "claim_manifest"),
    ("paper/performance_evaluation/figures", "figure"),
    ("paper/performance_evaluation/scripts", "figure_and_table_script"),
    ("experiments/industry_realism_action_opportunity_phase_a_v1", "frozen_primary_artifact"),
    ("experiments/industry_realism_action_opportunity_phase_b_v2", "frozen_primary_artifact"),
    ("experiments/fresh_production_latency_headroom_confirmatory_v1", "frozen_primary_artifact"),
    ("experiments/fresh_production_latency_headroom_confirmatory_v1_robustness", "robustness_output"),
    ("experiments/fresh_production_latency_headroom_confirmatory_v1_corrected", "corrected_derivative"),
    ("experiments/public_trace_replay_v1/layer3_checkpoint.jsonl", "supporting_existing_artifact"),
    ("experiments/joint240_same_distribution_adaptive_exploitability_v1/per_scenario_oof_results.csv", "supporting_existing_artifact"),
    ("experiments/real_vllm_mechanism_validation_v1/native_vllm_chunk_budget_semantics_probe_v1/mechanism_summary.json", "vllm_probe_summary"),
    ("experiments/real_vllm_mechanism_validation_v1/native_vllm_chunk_budget_semantics_probe_v1/statistical_summary.json", "vllm_probe_summary"),
    ("experiments/real_vllm_mechanism_validation_v1/runtime_environment.json", "vllm_probe_summary"),
    ("experiments/real_vllm_mechanism_validation_v1/vllm_install.json", "vllm_probe_summary"),
    ("experiments/real_vllm_mechanism_validation_v1/probe_config.json", "vllm_probe_summary"),
    ("experiments/real_vllm_pressure_action_validation_v1/REAL_VLLM_VALIDATION_RESULT_V1.json", "vllm_probe_summary"),
    ("experiments/real_vllm_pressure_action_validation_v1/PREREGISTRATION_V1.json", "preregistration"),
    ("scripts/industry_realism_action_opportunity_phase_a_v1.py", "experiment_code"),
    ("scripts/industry_realism_action_opportunity_phase_b_v2.py", "experiment_code"),
    ("scripts/industry_realism_causal_headroom_phase_d_v1.py", "experiment_code"),
    ("scripts/industry_realism_causal_headroom_phase_d_v1_execute.py", "experiment_code"),
    ("scripts/fresh_production_support_mapping_v1.py", "experiment_code"),
    ("scripts/fresh_latency_causal_confirmatory_v1.py", "experiment_code"),
    ("scripts/fresh_causal_robustness_v1.py", "analysis_code"),
    ("scripts/fresh_causal_correct_state_level_v1.py", "correction_code"),
    ("scripts/build_performance_evaluation_manuscript.sh", "build_script"),
    ("scripts/verify_performance_evaluation_release.py", "verification_script"),
    ("src/llmserveopt", "simulator_code"),
    ("tests/test_manuscript_claim_manifest.py", "test"),
    ("tests/test_manuscript_robustness_numbers.py", "test"),
    ("tests/test_manuscript_figures.py", "test"),
    ("tests/test_fresh_causal_robustness.py", "test"),
    ("tests/test_fresh_causal_state_level_correction.py", "test"),
    ("docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md", "documentation"),
    ("docs/FRESH_CAUSAL_ROBUSTNESS_REPORT.md", "documentation"),
    ("docs/current/PERFORMANCE_EVALUATION_REPRODUCIBILITY.md", "documentation"),
    ("LICENSE", "license"),
    ("pyproject.toml", "environment"),
    ("requirements.txt", "environment"),
]
# Never copy these even when they sit under an included tree.
EXCLUDE_PARTS = ("__pycache__", ".pytest_cache", ".DS_Store")
EXCLUDE_SUFFIX = (".pyc", ".log", ".aux", ".out", ".blg", ".spl")
ENV_PACKAGES = ["numpy", "pandas", "matplotlib", "scipy", "pyarrow", "PyYAML", "pytest", "pymupdf", "pillow"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def tracked(spec: str) -> list[str]:
    out = git("ls-files", "-z", "--", spec)
    files = [p for p in out.split("\0") if p]
    return [p for p in files if not any(x in Path(p).parts for x in EXCLUDE_PARTS) and not p.endswith(EXCLUDE_SUFFIX)]


def collect(archive_root: Path | None) -> list[tuple[Path, str, str]]:
    """Return (source path, bundle-relative path, role) for every bundle file except generated metadata."""
    items: dict[str, tuple[Path, str]] = {}
    for spec, role in INCLUDE:
        files = tracked(spec)
        if not files:
            raise SystemExit(f"included path has no tracked files: {spec}")
        for rel in files:
            items[rel] = (ROOT / rel, role)
    if archive_root is not None:
        shard_dir = archive_root / SHARD_RUN_REL / "continuation_shards"
        if not shard_dir.is_dir():
            raise SystemExit(f"continuation shard directory not found: {shard_dir}")
        for p in sorted(shard_dir.iterdir()):
            if p.is_file():
                items[f"{SHARD_BUNDLE_ROOT}/{SHARD_RUN_REL}/continuation_shards/{p.name}"] = (p, "continuation_shard")
    return [(src, rel, role) for rel, (src, role) in sorted(items.items())]


def dirty_included_paths() -> list[str]:
    specs = [s for s, _ in INCLUDE]
    out = git("status", "--porcelain", "--", *specs)
    return [line for line in out.splitlines() if line]


def environment() -> dict:
    import importlib.metadata as md
    pk = {}
    for name in ENV_PACKAGES:
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    try:
        tex = subprocess.check_output(["pdflatex", "--version"], text=True).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        tex = None
    return {"python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(),
            "packages": pk, "pdflatex": tex,
            "note": "Recorded on the build machine; the pinned lower bounds are in pyproject.toml and requirements.txt."}


def release_readme(a: argparse.Namespace, commit: str, n_files: int, n_shards: int) -> str:
    return f"""# Reproducibility package, release {a.version}

{TITLE}

* v1.0.0 (published record): [10.5281/zenodo.22865294](https://doi.org/10.5281/zenodo.22865294)
* v1.1.0 (published record): [10.5281/zenodo.22866983](https://doi.org/10.5281/zenodo.22866983)
* Source repository: {GITHUB}
* Built from git commit `{commit}` on branch `release/peva-v1-20260920` (final Git tag not yet created)
* Author: Soroush Vahidi, New Jersey Institute of Technology
* License: MIT (see `LICENSE`). Values derived from third-party traces keep the upstream attribution terms below.

This archive is a snapshot of the repository at the commit above, restricted to what the paper needs. It mirrors the
repository layout, so every script runs from the archive root exactly as it does from a repository checkout.
`MANIFEST.json` lists every file with its size, SHA-256 and role; `SHA256SUMS.txt` is the same in `sha256sum` format.

## What changed since v1.0.0

v1.0.0 held the frozen confirmatory artifacts and an older figure script. This release adds the post hoc robustness
outputs, the corrected derivative, the continuation shards, the current figure code, the claim manifest, the simulator
code and the manuscript source. **No frozen artifact and no primary result changed**: the
files under the three frozen `experiments/` directories are byte-identical to v1.0.0.

## Layout

| Path | Content |
|---|---|
| `paper/performance_evaluation/` | Manuscript source (`main.tex`, `references.bib`, `main.bbl`), current and legacy figure assets, figure/table scripts, `FINAL_CLAIM_MANIFEST.json`. |
| `experiments/industry_realism_action_opportunity_phase_a_v1/` | **Frozen.** Native replay (Phase A) with its preregistration. |
| `experiments/industry_realism_action_opportunity_phase_b_v2/` | **Frozen.** Resource-pressure map (Phase B v2). |
| `experiments/fresh_production_latency_headroom_confirmatory_v1/` | **Frozen.** Preregistered fresh causal latency-headroom study: protocols, preregistration, states, actions, result, bootstrap, artifact hashes. |
| `experiments/fresh_production_latency_headroom_confirmatory_v1_robustness/` | Post hoc robustness and concentration outputs (paper Section 7, Tables 3-5, Figure 6). |
| `experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/` | Corrected derivative of two descriptive state-level columns, audit, extracted SBS reference rows, provenance. |
| `experiments/real_vllm_*` | Summaries of the bounded vLLM probe (only the files the paper reads). |
| `{SHARD_BUNDLE_ROOT}/{SHARD_RUN_REL}/continuation_shards/` | The {n_shards} hash-verified continuation-shard artifacts (96 shards plus their manifests) of the fresh causal run. |
| `scripts/`, `src/llmserveopt/` | Experiment, robustness and correction code and the simulator. |
| `tests/` | The manuscript, robustness and correction tests. |
| `docs/` | Correction and robustness reports, reproducibility guide. |
| `verify_release.py`, `ENVIRONMENT.json`, `PROVENANCE_MAP.md`, `SHARD_PUBLICATION_DECISION.md` | Self-check, build environment, claim-to-artifact map, shard decision. |

## Verify the archive

```bash
pip install numpy pandas matplotlib scipy pyarrow pyyaml pytest   # see pyproject.toml / requirements.txt
python3 verify_release.py            # checksums, frozen-artifact hashes, claim manifest, figures, correction replay, tests
python3 verify_release.py --latex    # additionally rebuild the manuscript PDF (needs pdflatex, bibtex, elsarticle)
```

The checks are: SHA-256 of every file against `MANIFEST.json`; the frozen files and all 289 shard artifacts against
`FRESH_LATENCY_ARTIFACT_HASHES_V1.json`; the 62 claims in `FINAL_CLAIM_MANIFEST.json` recomputed from the artifacts and
checked against `main.tex`; every number of the robustness tables recomputed by `robustness_numbers.py`; current manuscript figures regenerated
into a temporary directory and compared with the shipped PNGs; the correction replayed from the shards and compared
byte-for-byte with the shipped corrected derivative; and the bundled tests.

Re-running the simulator (Phases A, B and the fresh causal execution) is not required to check the paper's numbers and
needs the raw upstream traces (below) and, for the fresh execution, an HPC allocation; the executor code is included.

## Artifact correction (disclosure)

The frozen `FRESH_LATENCY_STATE_LEVEL_V1.csv` is preserved unchanged. Two of its **descriptive** columns,
`mean_ref_latency` and `p95_ref_latency`, were populated from the first counterfactual branch of each state instead of
the SBS reference branch. **No reported scientific result depended on those columns**: the primary endpoint, the
opportunity counts, the bootstrap confidence interval and every table and figure are computed from other fields. A
corrected derivative (`..._corrected/FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv`) supplies the SBS reference values,
extracted from the shards (`SBS_REFERENCE_ROWS_V1.csv`). The primary endpoint and its confidence interval are unchanged.
Details: `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`.

## Third-party data

Raw traces are **not** redistributed. Obtain them from the upstream providers and follow their licence and attribution
terms: the Azure LLM inference trace 2023 (Azure Public Dataset, <https://github.com/Azure/AzurePublicDataset>) and
BurstGPT (Wang et al., KDD 2025, doi:10.1145/3711896.3737413). The artifacts here are derived scheduling outcomes and
window/state identifiers, not trace content. Please credit the trace authors when reusing them.

## How to cite

Cite this archive by its Zenodo persistent DOI after publication (v1.0.0: `10.5281/zenodo.22865294`; the v1.1.0
version DOI is assigned by Zenodo on publication). Do not treat a reserved draft DOI as published.

## Contents

{n_files} files (excluding this README, `MANIFEST.json` and `SHA256SUMS.txt`).
"""


def provenance_map(a: argparse.Namespace, commit: str) -> str:
    return f"""# Provenance map

| Paper element | Produced by | Frozen input(s) | Verified by |
|---|---|---|---|
| Native action-null result (Section 4, Figure 2) | `scripts/industry_realism_action_opportunity_phase_a_v1.py` | `experiments/industry_realism_action_opportunity_phase_a_v1/` | claim manifest `native.*`, `figure2.*` |
| Pressure-induced disagreement (Section 5, Figures 2-3) | `scripts/industry_realism_action_opportunity_phase_b_v2.py` | `experiments/industry_realism_action_opportunity_phase_b_v2/` | claim manifest `pressure.*`, `figure3.*` |
| Fresh causal headroom, primary endpoint (Section 6) | `scripts/fresh_latency_causal_confirmatory_v1.py` | `experiments/fresh_production_latency_headroom_confirmatory_v1/` | claim manifest `fresh.*`; `FRESH_LATENCY_ARTIFACT_HASHES_V1.json` |
| Robustness and concentration (Section 7, Tables 3-5, Figure 6) | `scripts/fresh_causal_robustness_v1.py` | `..._v1_robustness/` | claim manifest `robust.*`; `paper/performance_evaluation/scripts/robustness_numbers.py` |
| Corrected derivative | `scripts/fresh_causal_correct_state_level_v1.py` | continuation shards under `{SHARD_BUNDLE_ROOT}/` | `verify_release.py` (byte replay) |
| Current manuscript figures | `paper/performance_evaluation/scripts/plot_*.py` (shared style `figstyle.py`) | frozen CSVs above | `verify_release.py`, `tests/test_manuscript_figures.py` |
| vLLM correspondence probe (Section 9) | (probe summaries only) | `experiments/real_vllm_*` | claim manifest `vllm.*` |
| All 62 quoted numbers | `paper/performance_evaluation/scripts/build_claim_manifest.py` | all of the above | `build_claim_manifest.py --check` |

Built from git commit `{commit}` (branch `release/peva-v1-20260920`). Original execution commit of the fresh causal run: `38e02bcdaeb8f9cec2289b05dea9a1406318d14f`
(executor script sha256 `e4ef52c38ce0412674319cdf2997f1934ab5385c031b227010eb6a913f24815b`).
"""


def shard_decision(n_shards: int) -> str:
    return f"""# Continuation-shard publication decision

**Decision: publish both** the full canonical continuation shards *and* the compact extracted SBS reference rows.

* The {n_shards} shard artifacts (96 shard CSVs, their JSON manifests and correctness CSVs, and `RUN_SUMMARY.json`)
  total under 3 MB, are pure derived simulator outputs (latency statistics, state identifiers and request-id hashes;
  no trace content, prompts, credentials or local paths), and every file matches the frozen manifest
  `FRESH_LATENCY_ARTIFACT_HASHES_V1.json`. Neither size, privacy nor licensing argues against publishing them.
* They are the only artifact holding the `SBS_REFERENCE` rows, so they are the sole source from which the corrected
  derivative can be regenerated and the byte-identity proof of the correction replayed. The compact
  `SBS_REFERENCE_ROWS_V1.csv` alone lets a reader check the corrected columns but not the replay.
* The partial directory `fresh-latency-confirmatory-v1` of the local provenance archive (3 files, from an
  interrupted earlier attempt) is **not** canonical and is not released; only `fresh-latency-execution-v1` is.
"""


def build(a: argparse.Namespace) -> None:
    if not a.allow_dirty:
        dirty = dirty_included_paths()
        if dirty:
            raise SystemExit("refusing to build from a dirty tree (included paths changed):\n" + "\n".join(dirty))
    commit = git("rev-parse", "HEAD")
    commit_date = git("show", "-s", "--format=%cI", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    out = ROOT / "release" / f"performance_evaluation_v{a.version.replace('.', '_')}"
    zip_path = out.with_suffix(".zip")
    if out.exists():
        shutil.rmtree(out)
    files = collect(a.archive_root)
    for src, rel, _ in files:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    n_shards = sum(1 for _, _, r in files if r == "continuation_shard")
    n_generated = 7
    (out / "README.md").write_text(release_readme(a, commit, len(files) - n_generated, n_shards))
    (out / "PROVENANCE_MAP.md").write_text(provenance_map(a, commit))
    (out / "SHARD_PUBLICATION_DECISION.md").write_text(shard_decision(n_shards))
    (out / "ENVIRONMENT.json").write_text(json.dumps(environment(), indent=1) + "\n")
    shutil.copyfile(ROOT / "scripts" / "verify_performance_evaluation_release.py", out / "verify_release.py")
    (out / "CITATION.cff").write_text(citation_cff(a, commit))
    (out / ".zenodo.json").write_text(json.dumps(zenodo_metadata(a, commit), indent=2, ensure_ascii=False) + "\n")

    generated = {"README.md": "release_readme", "PROVENANCE_MAP.md": "documentation", "SHARD_PUBLICATION_DECISION.md": "documentation",
                 "ENVIRONMENT.json": "environment", "verify_release.py": "verification_script", "CITATION.cff": "citation_metadata",
                 ".zenodo.json": "archive_metadata"}
    entries = []
    roles = {rel: role for _, rel, role in files}
    roles.update(generated)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            rel = p.relative_to(out).as_posix()
            entries.append({"path": rel, "size_bytes": p.stat().st_size, "sha256": sha256_file(p), "role": roles[rel]})
    manifest = {
        "schema": "peva_release_manifest_v1",
        "archive_name": out.name,
        "release": {"version": a.version, "tag": a.tag, "git_commit": commit, "git_commit_date": commit_date, "built_from_branch": branch,
                    "included_paths_clean_at_build": not a.allow_dirty},
        "identifiers": {"v1_doi": a.prior_doi, "concept_doi": "10.5281/zenodo.22865293 (concept of record 22865294)", "v1_1_doi": "10.5281/zenodo.22866983", "repository": GITHUB},
        "title": TITLE,
        "builder": {"script": "scripts/build_performance_evaluation_release.py",
                    "script_sha256": sha256_file(Path(__file__)), "python": platform.python_version()},
        "excluded_by_design": ["fresh-latency-confirmatory-v1 (partial, non-canonical provenance directory)", "raw third-party traces",
                               "credentials and logs", "untracked files", "unrelated worktrees and selector-study artifacts"],
        "total_files": len(entries),
        "files": entries,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    lines = [f"{e['sha256']}  {e['path']}" for e in entries] + [f"{sha256_file(out / 'MANIFEST.json')}  MANIFEST.json"]
    (out / "SHA256SUMS.txt").write_text("\n".join(sorted(lines, key=lambda s: s.split('  ', 1)[1])) + "\n")

    make_zip(out, zip_path, commit_date)
    print(f"bundle: {out.relative_to(ROOT)}  files={manifest['total_files']}  commit={commit}")
    print(f"zip:    {zip_path.relative_to(ROOT)}  sha256={sha256_file(zip_path)}  bytes={zip_path.stat().st_size}")


def make_zip(out: Path, zip_path: Path, commit_date: str) -> None:
    y, mo, d = (int(x) for x in commit_date[:10].split("-"))
    stamp = (max(y, 1980), mo, d, 0, 0, 0)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zi = zipfile.ZipInfo(f"{out.name}/{p.relative_to(out).as_posix()}", date_time=stamp)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = (0o755 if p.suffix == ".py" and p.name == "verify_release.py" else 0o644) << 16
                z.writestr(zi, p.read_bytes(), compresslevel=9)


def citation_cff(a: argparse.Namespace, commit: str) -> str:
    return f"""cff-version: 1.2.0
message: "If you use this archive, please cite the paper's reproducibility package (Zenodo record below)."
type: dataset
title: "Reproducibility package for \\"{TITLE}\\""
version: "{a.version}"
date-released: "{a.release_date}"
license: MIT
repository-code: "{GITHUB}"
authors:
  - family-names: "Vahidi"
    given-names: "Soroush"
    affiliation: "New Jersey Institute of Technology"
identifiers:
  - type: doi
    value: "{a.prior_doi}"
    description: "Published v1.0.0 record (Zenodo); v1.1.0 is archived in the same concept record only after Zenodo publication succeeds"
  - type: url
    value: "{GITHUB}/tree/{commit}"
    description: "Git commit {commit} (branch release/peva-v1-20260920; final tag not yet created)"
keywords:
  - LLM serving
  - request scheduling
  - performance evaluation
  - workload replay
  - causal evaluation
  - reproducibility
"""


def zenodo_metadata(a: argparse.Namespace, commit: str) -> dict:
    desc = (
        "<p>Reproducibility package for the paper <em>" + TITLE + "</em> (submitted to <em>Performance Evaluation</em>).</p>"
        "<p>The package supports a trace-driven performance evaluation of when scheduler adaptation matters in LLM serving: a "
        "deterministic simulator replaying production-derived workloads (Azure LLM inference 2023, BurstGPT), a resource-pressure "
        "map, and a preregistered fresh causal study of one-step latency headroom. It contains the <strong>frozen confirmatory "
        "artifacts</strong> (unchanged from v1.0.0), the <strong>post hoc robustness and concentration analysis</strong>, a "
        "<strong>corrected derivative</strong> of two descriptive state-level columns (the original frozen file is preserved; no "
        "reported result depended on those columns), the 96 hash-verified continuation shards from which the derivative is computed, "
        "the simulator, experiment, robustness, correction and figure code, the claim manifest that recomputes every quoted number, "
        "and the manuscript source.</p>"
        "<p>Raw third-party traces are not redistributed; see README.md. Run <code>python3 verify_release.py</code> from the archive root "
        "to check the package.</p>")
    return {
        "title": f"{TITLE} (Reproducibility Package)",
        "upload_type": "dataset",
        "description": desc,
        "creators": [{"name": "Vahidi, Soroush", "affiliation": "New Jersey Institute of Technology"}],
        "access_right": "open",
        "license": "mit-license",
        "keywords": ["LLM serving", "request scheduling", "performance evaluation", "workload replay", "causal headroom",
                     "resource pressure", "reproducibility", "robustness analysis"],
        "related_identifiers": [
            {"relation": "isSupplementTo", "identifier": f"{GITHUB}/tree/{commit}", "resource_type": "software", "scheme": "url"},
        ],
        "notes": f"Built from git commit {commit} (branch release/peva-v1-20260920; final Git tag not yet created). v1.0.0 is {a.prior_doi}; this bundle is the published v1.1.0 version (record 22866983, concept DOI 10.5281/zenodo.22865293).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", default="1.1.0")
    ap.add_argument("--tag", default="performance-evaluation-v1.1.0")
    ap.add_argument("--version-doi", default=None)
    ap.add_argument("--concept-doi", default="10.5281/zenodo.22865293")
    ap.add_argument("--prior-doi", default="10.5281/zenodo.22865294")
    ap.add_argument("--release-date", default="2026-09-21")
    ap.add_argument("--archive-root", type=Path, default=ROOT.parent / "llm-serving-heuristic-evolution-local-provenance")
    ap.add_argument("--allow-dirty", action="store_true", help="build although included tracked paths have uncommitted changes (validation builds only)")
    build(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
