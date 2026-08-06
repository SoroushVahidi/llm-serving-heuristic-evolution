"""Focused tests for the Apt-Serve Strategy C Wulver probe's compact,
committed evidence artifacts (results/provenance/apt_serve_strategy_probe/)
and the resulting Strategy C/D decision recorded in
docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md.

These are consistency/regression checks over already-executed, already-
committed evidence -- they do not run vLLM, do not touch Wulver, and do
not require GPU/CUDA. See that audit doc for the full narrative.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = ROOT / "results" / "provenance" / "apt_serve_strategy_probe"
AUDIT_DOC = ROOT / "docs" / "audits" / "apt_serve_strategy_c_wulver_probe_20260806.md"
BASELINE_STATUS = ROOT / "docs" / "BASELINE_STATUS.md"

PINNED_COMMIT = "c953217988274a761da35cf06c01033b18dadf68"
CLASSIFICATION = "STRATEGY_C_VIABLE_WITH_LIMITATIONS"

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"gho_[A-Za-z0-9]{30,}"),
    re.compile(r"krbtgt"),
]


def _load(name: str) -> dict:
    return json.loads((PROVENANCE_DIR / name).read_text(encoding="utf-8"))


def test_provenance_dir_exists_and_is_compact():
    assert PROVENANCE_DIR.is_dir()
    files = sorted(p.name for p in PROVENANCE_DIR.iterdir() if p.is_file())
    assert files == [
        "README.md",
        "copied_file_sha256_hashes.txt",
        "environment_pip_freeze.txt",
        "import_probe_patched.json",
        "import_probe_vanilla.json",
        "job_manifest_1163782.txt",
        "job_manifest_1164406.txt",
        "micro_trace.json",
    ]
    # No raw source, no compiled artifacts, nothing but small text/JSON.
    for path in PROVENANCE_DIR.iterdir():
        assert path.suffix in {".md", ".txt", ".json"}, path
        assert path.stat().st_size < 200_000, f"{path} unexpectedly large for a compact artifact"


def test_vanilla_import_probe_all_ok():
    data = _load("import_probe_vanilla.json")
    results = data["results"]
    assert len(results) == 9
    assert all(r["status"] == "OK" for r in results)


def test_patched_import_probe_all_ok():
    data = _load("import_probe_patched.json")
    results = data["results"]
    assert len(results) == 7
    assert all(r["status"] == "OK" for r in results)
    labels = {r["label"] for r in results}
    assert "patched_scheduler_construct_synthetic_config" in labels
    # Confirms this is genuinely the Apt-Serve-patched scheduler, not a
    # stale/cached vanilla import.
    patched_scheduler = next(r for r in results if r["label"] == "import_patched_scheduler_module")
    assert patched_scheduler["detail"]["has_greedy_selection_prefill"] is True
    assert patched_scheduler["detail"]["has_dynamic_priority"] is True


def test_micro_trace_all_scenarios_ok_and_commit_pinned():
    data = _load("micro_trace.json")
    assert data["apt_serve_commit"] == PINNED_COMMIT
    scenarios = data["scenarios"]
    assert len(scenarios) == 3
    assert all(s["status"] == "OK" for s in scenarios)
    names = {s["name"] for s in scenarios}
    assert names == {
        "three_requests_two_fit_memory_budget",
        "homogeneous_low_contention",
        "single_oversized_request_extreme_case",
    }


def test_copied_file_hashes_cover_the_five_scheduler_files():
    text = (PROVENANCE_DIR / "copied_file_sha256_hashes.txt").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 5
    for expected in ("block.py", "sequence.py", "block_manager_v1.py", "interfaces.py", "scheduler.py"):
        assert any(expected in line for line in lines), f"missing hash line for {expected}"
    # sha256sum output: "<64 hex chars>  <path>"
    for line in lines:
        digest = line.split()[0]
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


def test_no_upstream_apt_serve_source_committed():
    """License is disclosed as absent (docs/audits/...): no Apt-Serve
    source may be vendored/redistributed anywhere in this repository."""
    assert not any(PROVENANCE_DIR.glob("*.py")), "no Python source belongs in the compact provenance dir"
    assert not (ROOT / "vendor").exists(), "Apt-Serve is fetched to Wulver-local /mmfs1 scratch, never into the repo tree"


def test_provenance_artifacts_contain_no_obvious_secrets():
    for path in PROVENANCE_DIR.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            assert not pattern.search(text), f"possible secret pattern {pattern.pattern!r} in {path}"


def test_strategy_classification_consistent_across_docs():
    audit_text = AUDIT_DOC.read_text(encoding="utf-8")
    baseline_text = BASELINE_STATUS.read_text(encoding="utf-8")
    assert f"`{CLASSIFICATION}`" in audit_text or CLASSIFICATION in audit_text
    assert CLASSIFICATION in baseline_text
    # The decision must be scoped, not overclaimed as full integration/evaluation.
    assert "9b" in audit_text
    assert "not been evaluated" not in audit_text.lower() or "not evaluated" in baseline_text.lower()


def test_pinned_commit_consistent_across_sbatch_and_docs():
    sbatch = (ROOT / "scripts" / "slurm" / "wulver_apt_serve_strategy_c_cpu_probe.sbatch").read_text(encoding="utf-8")
    assert f'PINNED_COMMIT="{PINNED_COMMIT}"' in sbatch
    assert PINNED_COMMIT in AUDIT_DOC.read_text(encoding="utf-8")
