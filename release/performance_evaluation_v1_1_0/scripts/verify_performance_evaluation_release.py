#!/usr/bin/env python3
"""Verify the Performance Evaluation reproducibility release bundle (or a repository checkout).

Run from the archive root as ``python3 verify_release.py`` (the builder copies this script there), or from a
repository checkout as ``python3 scripts/verify_performance_evaluation_release.py --archive-root <provenance root>``.

Checks (each prints PASS/FAIL; the exit status is non-zero if any fails):
  1. every file against MANIFEST.json, and no undeclared file (bundle mode only);
  2. the frozen compact artifacts and all continuation-shard artifacts against FRESH_LATENCY_ARTIFACT_HASHES_V1.json;
  3. the 62 claims of FINAL_CLAIM_MANIFEST.json recomputed from the artifacts and checked against main.tex;
  4. the robustness numbers behind Tables 3-5 recomputed from the frozen artifacts;
  5. the correction replayed from the shards into a temporary directory and compared byte-for-byte with the shipped
     corrected derivative;
  6. the bundled tests (which also regenerate the current manuscript figures and compare them with the shipped figures);
  7. with --latex: the manuscript rebuilt with pdflatex/bibtex (needs a TeX distribution with elsarticle).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = (HERE / "MANIFEST.json").exists()
ROOT = HERE if BUNDLE else HERE.parent
SHARD_RUN_REL = "fgcs-finalization-20260920/fresh_latency/fresh-latency-execution-v1/fresh_latency_causal_run_v1"
FROZEN = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1"
CORRECTED = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1_corrected"
PAPER = ROOT / "paper" / "performance_evaluation"

results: list[tuple[str, bool, str]] = []


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), MPLBACKEND="Agg")
    e.update(env or {})
    return subprocess.run(cmd, cwd=cwd, env=e, text=True, capture_output=True)


def check_manifest() -> None:
    m = json.loads((ROOT / "MANIFEST.json").read_text())
    bad, seen = [], set()
    for e in m["files"]:
        p = ROOT / e["path"]
        seen.add(e["path"])
        if not p.is_file() or p.stat().st_size != e["size_bytes"] or sha256(p) != e["sha256"]:
            bad.append(e["path"])
    extra = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
             and p.relative_to(ROOT).as_posix() not in seen and p.name not in ("MANIFEST.json", "SHA256SUMS.txt")
             and "__pycache__" not in p.parts and not p.name.endswith(".pyc")]
    record("MANIFEST.json: every file present with matching size and SHA-256", not bad and not extra,
           f"{len(m['files'])} files; bad={bad[:3]} extra={extra[:3]}")


def check_frozen_hashes(shard_dir: Path | None) -> None:
    h = json.loads((FROZEN / "FRESH_LATENCY_ARTIFACT_HASHES_V1.json").read_text())
    bad = [n for n, d in h["compact_artifacts"].items() if not (FROZEN / n).is_file() or sha256(FROZEN / n) != d]
    record("frozen compact artifacts match FRESH_LATENCY_ARTIFACT_HASHES_V1.json", not bad, f"{len(h['compact_artifacts'])} files; bad={bad}")
    if shard_dir is None or not shard_dir.is_dir():
        record("continuation shards match the frozen shard manifest", False, "shard directory not available")
        return
    bad = [n for n, d in h["shard_artifacts"].items() if not (shard_dir / n).is_file() or sha256(shard_dir / n) != d]
    extra = [p.name for p in shard_dir.iterdir() if p.is_file() and p.name not in h["shard_artifacts"]]
    record("continuation shards match the frozen shard manifest", not bad and not extra,
           f"{len(h['shard_artifacts'])} artifacts; bad={bad[:3]} extra={extra[:3]}")


def check_claims() -> None:
    r = run([sys.executable, str(PAPER / "scripts" / "build_claim_manifest.py"), "--check"])
    record("claim manifest: 62 claims recomputed and matched to main.tex", r.returncode == 0 and "62 claims checked, 0 problem(s)" in r.stdout,
           (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else (r.stdout + r.stderr)[-300:])


def check_robustness_numbers() -> None:
    r = run([sys.executable, str(PAPER / "scripts" / "robustness_numbers.py")])
    record("robustness numbers (Tables 3-5) recomputed from the frozen artifacts", r.returncode == 0 and len(r.stdout) > 500, (r.stderr or "")[-200:])


def check_correction_replay(archive_root: Path | None) -> None:
    if archive_root is None or not (archive_root / SHARD_RUN_REL).is_dir():
        record("correction replay from the shards", False, "shard archive root not available")
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "corrected"
        r = run([sys.executable, str(ROOT / "scripts" / "fresh_causal_correct_state_level_v1.py"), "--archive-root", str(archive_root), "--out", str(out)])
        if r.returncode != 0:
            record("correction replay from the shards", False, (r.stdout + r.stderr)[-300:])
            return
        diffs = []
        for p in sorted(CORRECTED.iterdir()):
            q = out / p.name
            if not q.is_file() or q.read_bytes() != p.read_bytes():
                diffs.append(p.name)
        diffs += [q.name for q in out.iterdir() if not (CORRECTED / q.name).exists()]
        frozen_untouched = all(sha256(FROZEN / n) == d for n, d in json.loads((FROZEN / "FRESH_LATENCY_ARTIFACT_HASHES_V1.json").read_text())["compact_artifacts"].items())
        record("correction replay reproduces the shipped corrected derivative byte-for-byte; frozen files untouched", not diffs and frozen_untouched,
               f"{len(list(CORRECTED.iterdir()))} files; differing={diffs}")


def check_tests() -> None:
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    if not tests:
        record("bundled tests", False, "no tests directory")
        return
    test_env = {"PEVA_ARCHIVE_ROOT": os.environ.get("PEVA_ARCHIVE_ROOT", "")}
    if BUNDLE:
        # The tracked release tree may be verified before commit while it is
        # nested inside the source repository. Do not let bundle tests inherit
        # that parent .git directory; frozen-file integrity is checked above by
        # explicit hashes.
        test_env["GIT_DIR"] = str(ROOT / ".no-parent-git")
    r = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], env=test_env)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    record("bundled tests (claim manifest, robustness numbers, figure regeneration, correction)", r.returncode == 0, tail)


def check_latex() -> None:
    if not shutil.which("pdflatex") or not shutil.which("bibtex"):
        record("manuscript rebuild", False, "pdflatex/bibtex not found")
        return
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        shutil.copytree(PAPER / "figures", t / "figures")
        for n in ("main.tex", "references.bib"):
            shutil.copyfile(PAPER / n, t / n)
        r = None
        for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                    ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
            r = subprocess.run(cmd, cwd=t, text=True, capture_output=True)
        ok = (t / "main.pdf").is_file() and "Output written on main.pdf" in r.stdout and "undefined" not in r.stdout.lower()
        record("manuscript rebuild (pdflatex, bibtex, pdflatex x2; no undefined references)", ok,
               next((l for l in r.stdout.splitlines() if l.startswith("Output written")), "no output"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive-root", type=Path, default=None,
                    help="root containing the continuation shards (bundle mode uses ./provenance_archive automatically)")
    ap.add_argument("--latex", action="store_true", help="also rebuild the manuscript PDF")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    archive = (ROOT / "provenance_archive") if BUNDLE and a.archive_root is None else a.archive_root
    os.environ["PEVA_ARCHIVE_ROOT"] = str(archive) if archive else ""
    if archive:
        os.environ["FRESH_ARCHIVE_ROOT"] = str(archive)  # lets the bundled shard-dependent tests run
    print(f"verifying {'bundle' if BUNDLE else 'repository checkout'} at {ROOT}")
    if BUNDLE:
        check_manifest()
    check_frozen_hashes(archive / SHARD_RUN_REL / "continuation_shards" if archive else None)
    check_claims()
    check_robustness_numbers()
    check_correction_replay(archive)
    if not a.skip_tests:
        check_tests()
    if a.latex:
        check_latex()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
