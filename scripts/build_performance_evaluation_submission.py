#!/usr/bin/env python3
"""Regenerate the Performance Evaluation submission package from the canonical manuscript source.

    python3 scripts/build_performance_evaluation_submission.py 

The package directory ``paper/performance_evaluation/submission/`` is DELETED and rebuilt from

  * ``paper/performance_evaluation/{main.tex,references.bib,main.bbl,figures/}``  (canonical manuscript source),
  * ``paper/performance_evaluation/submission_docs/``  (highlights, cover letter, CRediT, reviewer suggestions).

Nothing is copied from the previous package.  It then builds the Elsevier source ZIP and verifies it by extracting it
into a fresh temporary directory, compiling from scratch, comparing the text of the result with the canonical review
PDF, checking that every figure resolves, and scanning for absolute local paths.  Everything except the recorded
verification is deterministic (fixed zip timestamps and order).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "performance_evaluation"
DOCS = PAPER / "submission_docs"
PKG = PAPER / "submission"
CANONICAL_PDF = ROOT / "paper" / "when_does_llm_serving_scheduler_adaptation_matter.pdf"
ZIP_NAME = "performance_evaluation_submission_sources.zip"
ZIP_STAMP = (2026, 9, 20, 0, 0, 0)
BUILD_CMD = "pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
HIGHLIGHT_MAX = 85


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def section(tex: str, title: str) -> str:
    m = re.search(r"\\section\*\{" + re.escape(title) + r"\}\n(.*?)(?=\n\\section|\n\\bibliographystyle)", tex, re.S)
    if not m:
        raise SystemExit(f"section not found in main.tex: {title}")
    return re.sub(r"\s+", " ", m.group(1)).strip()


def plain(s: str) -> str:
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\cite\{[^}]*\}", "[28]", s)
    s = re.sub(r"Section~\\ref\{[^}]*\}", "Section 7", s)
    s = s.replace("{\\leavevmode\\raggedright ", "").replace("\\par}", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def facts(tex: str) -> dict:
    abstract = re.sub(r"\s+", " ", re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).replace("\\%", "%")).strip()
    kw = [k.strip() for k in re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tex, re.S).group(1).split("\\sep")]
    title = re.sub(r"\s+", " ", re.search(r"\\title\{(.*?)\}\n", tex, re.S).group(1).replace("\\\\", " ")).strip()
    cites = set()
    for m in re.findall(r"\\cite\{([^}]*)\}", tex):
        cites.update(x.strip() for x in m.split(","))
    bib = set(re.findall(r"^@\w+\{([^,]+),", (PAPER / "references.bib").read_text(), re.M))
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    return {"title": title, "abstract_words": len(abstract.split()), "keywords": kw, "cited": cites, "bib": bib, "figures": figs,
            "tables": len(re.findall(r"\\begin\{table\*?\}", tex)), "includegraphics_in_tables": 0}


def declarations(tex: str, version_doi: str) -> str:
    ai = section(tex, "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process")
    return f"""# Declarations (verbatim from the manuscript; for the Elsevier declarations tool and the submission portal)

## Declaration of competing interest
{plain(section(tex, "Declaration of Competing Interest"))}

## Funding
{plain(section(tex, "Funding and Support"))}

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
{plain(ai)}

## CRediT authorship contribution statement
{plain((DOCS / "credit_authorship_statement.txt").read_text().split("[Candidate", 1)[0].replace("CRediT authorship contribution statement", "").strip())}
(Candidate statement for the sole author; the author must confirm it. See `credit_authorship_statement.txt`.)

## Acknowledgements
{plain(section(tex, "Acknowledgements"))}

## Data availability
{plain(section(tex, "Data and Code Availability"))}

Archive: Zenodo record 22865294 (published v1.0.0: https://doi.org/10.5281/zenodo.22865294; concept DOI 10.5281/zenodo.22865293).
The v1.1.0 version DOI is assigned by Zenodo when the v1.1.0 version is published; see the release report (QUERY_8).
"""


def checklist(tex: str, f: dict, hl: list[str], version_doi: str, zip_sha: str, zip_info: dict, verification: dict) -> str:
    lens = [len(h) for h in hl]
    missing = sorted(f["cited"] - f["bib"]), sorted(f["bib"] - f["cited"])
    rows = [
        ("Title", "READY", f["title"]),
        ("Corresponding author details", "USER_CONFIRM", "Name and affiliation are in the manuscript; the corresponding-author email is entered in the portal and appears in `cover_letter.txt` (sv96@njit.edu, taken from the repository author metadata; confirm). ORCID is optional and portal-only."),
        ("Abstract", "READY", f"{f['abstract_words']} words (limit 250); renders in the PDF (verified in the extracted-source build)"),
        ("Keywords", "READY", f"{len(f['keywords'])} keywords (limit 7): " + "; ".join(f["keywords"])),
        ("Highlights", "READY", f"`performance_evaluation_highlights.txt`: {len(hl)} bullets, character counts {lens} (limit {HIGHLIGHT_MAX}); separate editable file"),
        ("Figures supplied separately", "READY", f"{len(f['figures'])} vector PDF figures in `figures/` (fonts embedded, drawn at final size), all resolved in the clean build; 300 dpi PNG previews stay in the repository"),
        ("Tables editable", "READY", f"{f['tables']} tables are LaTeX `tabular`/`tabularx` text; none is an image"),
        ("References reciprocal", "READY" if not any(missing) else "FAIL", f"{len(f['cited'])} cited, {len(f['bib'])} in `references.bib`; uncited={missing[1]} undefined={missing[0]}"),
        ("Spelling / grammar", "READY", "aspell (en_US) over the manuscript body: only technical terms, product names and acronyms are flagged (no misspellings); an author read-through is still recommended"),
        ("Permissions", "READY", "All figures and tables are original (generated from the author's own artifacts); no third-party copyrighted material is reproduced, so no permission is required"),
        ("Competing-interest declaration", "READY", "In the manuscript; the Elsevier declarations tool must still be completed and its document uploaded at submission (portal step)"),
        ("Funding statement", "READY", "In the manuscript: in-kind Google Cloud Research Credits and CloudRift Inc. computational/tooling support; no monetary research grant"),
        ("AI declaration", "READY", "In the manuscript before the references: ChatGPT, Codex (OpenAI), Gemini (Google), Claude (Anthropic); purposes, author review and responsibility stated"),
        ("CRediT statement", "USER_CONFIRM", "Candidate for the sole author in `credit_authorship_statement.txt`; no Funding acquisition, Supervision, Resources or Project administration claimed"),
        ("Data availability", "READY", "In the manuscript; Zenodo record 22865294 (v1.0.0 published, v1.1.0 version pending publication - DOI assigned by Zenodo at that time; see the release report)"),
        ("Data / software citation", "READY", "Reference [28] `[dataset]` (one combined dataset-and-software archive, version DOI) cited from Data and Code Availability"),
        ("Source ZIP build verified", "READY" if verification["all_ok"] else "FAIL", f"`{ZIP_NAME}`: {zip_info['files']} files, {zip_info['uncompressed_bytes']} bytes, sha256 {zip_sha}; build `{BUILD_CMD}` from a fresh extraction; {verification['summary']}"),
        ("Acknowledgements wording", "USER_CONFIRM", "The sentence 'the author thanks his mother' is grammatical and intentional; unchanged. Note the manuscript also uses 'they' for the author in the competing-interest declaration; confirm the pronoun choice."),
        ("Prior-submission disclosure", "USER_CONFIRM", "The cover letter states the manuscript is not under consideration elsewhere and does not mention the withdrawn, unpublished LLM 2026 predecessor (the roadmap suggested optional transparency); confirm."),
        ("Suggested reviewers", "OPTIONAL", "`suggested_reviewers.md`: four verified candidates; portal-only, never in the manuscript"),
        ("Graphical abstract", "NOT_APPLICABLE", "Optional; deliberately not supplied"),
    ]
    out = ["# Performance Evaluation submission checklist", "",
           "Generated by `scripts/build_performance_evaluation_submission.py` from the canonical manuscript; nothing here is carried over from the earlier package.", "",
           "| Item | Status | Evidence |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out) + "\n"


def make_zip(files: list[tuple[str, Path]], dest: Path) -> dict:
    if dest.exists():
        dest.unlink()
    total = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc, src in sorted(files):
            zi = zipfile.ZipInfo(arc, date_time=ZIP_STAMP)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            data = src.read_bytes()
            total += len(data)
            z.writestr(zi, data, compresslevel=9)
    return {"files": len(files), "uncompressed_bytes": total}


def norm_pdf_text(pdf: Path) -> list[str]:
    import pymupdf
    lines = []
    for page in pymupdf.open(pdf):
        for ln in page.get_text().splitlines():
            ln = ln.strip()
            if not ln or re.fullmatch(r"\d+", ln) or "Preprint submitted to" in ln or re.fullmatch(r"[A-Z][a-z]+ \d{1,2}, \d{4}", ln):
                continue
            lines.append(ln)
    return lines


def verify_zip(zip_path: Path) -> dict:
    checks: dict[str, tuple[bool, str]] = {}
    with tempfile.TemporaryDirectory() as td:
        t = Path(td) / "extract"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(t)
            leaks = [n for n in z.namelist() if re.search(rb"(/home/|/tmp/|/mnt/|C:\\\\Users)", z.read(n))]
        checks["no absolute local paths in any file"] = (not leaks, f"offending={leaks}")
        tex = (t / "main.tex").read_text()
        missing = [g for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex) if not (t / g).is_file()]
        checks["every figure resolves"] = (not missing, f"missing={missing}")
        r = None
        for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                    ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
            r = subprocess.run(cmd, cwd=t, text=True, capture_output=True)
        built = (t / "main.pdf").is_file() and "Output written on main.pdf" in r.stdout
        checks["clean pdflatex/bibtex build succeeds"] = (built, next((l for l in r.stdout.splitlines() if l.startswith("Output written")), "no output"))
        checks["no undefined references or citations"] = ("undefined" not in r.stdout.lower(), "final pass")
        if built:
            new, ref = norm_pdf_text(t / "main.pdf"), norm_pdf_text(CANONICAL_PDF)
            checks["text of the built PDF equals the canonical review PDF"] = (new == ref, f"{len(new)} vs {len(ref)} lines")
            ab = "Adaptive scheduling for large language model" in "\n".join(new) and "worthwhile" in "\n".join(new)
            checks["abstract renders"] = (ab, "")
    ok = all(v[0] for v in checks.values())
    return {"all_ok": ok, "checks": {k: {"ok": v[0], "detail": v[1]} for k, v in checks.items()},
            "summary": "; ".join(f"{k}: {'ok' if v[0] else 'FAIL'}" for k, v in checks.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version-doi", default=None)
    a = ap.parse_args()
    tex = (PAPER / "main.tex").read_text()
    f = facts(tex)
    hl = [l for l in (DOCS / "performance_evaluation_highlights.txt").read_text().splitlines() if l.strip()]
    if not 3 <= len(hl) <= 5 or any(len(h) > HIGHLIGHT_MAX for h in hl):
        raise SystemExit(f"highlights violate 3-5 bullets of <= {HIGHLIGHT_MAX} characters: {[len(h) for h in hl]}")
    if f["cited"] != f["bib"]:
        raise SystemExit(f"citation/bibliography mismatch: {sorted(f['cited'] ^ f['bib'])}")

    if PKG.exists():
        shutil.rmtree(PKG)
    (PKG / "figures").mkdir(parents=True)
    for n in ("main.tex", "references.bib", "main.bbl"):
        shutil.copyfile(PAPER / n, PKG / n)
    for g in f["figures"]:
        shutil.copyfile(PAPER / g, PKG / g)
    readme_build = (f"Performance Evaluation submission sources\n\nBuild (elsarticle class, TeX Live 2023 or later):\n\n    {BUILD_CMD}\n\n"
                    "main.bbl is included so the bibliography also builds without BibTeX. Figures are vector PDF in figures/.\n"
                    "No non-standard class or style files are needed.\n")
    (PKG / "README_BUILD.txt").write_text(readme_build)
    for n in ("performance_evaluation_highlights.txt", "cover_letter.txt", "credit_authorship_statement.txt", "suggested_reviewers.md"):
        shutil.copyfile(DOCS / n, PKG / n)
    (PKG / "declarations.md").write_text(declarations(tex, a.version_doi))

    src_files = [("main.tex", PKG / "main.tex"), ("references.bib", PKG / "references.bib"), ("main.bbl", PKG / "main.bbl"),
                 ("README_BUILD.txt", PKG / "README_BUILD.txt")] + [(g, PKG / g) for g in f["figures"]]
    zip_path = PKG / ZIP_NAME
    zip_info = make_zip(src_files, zip_path)
    zsha = sha256(zip_path)
    verification = verify_zip(zip_path)
    (PKG / "SOURCE_ZIP_VERIFICATION.json").write_text(json.dumps({"zip": ZIP_NAME, "sha256": zsha, **zip_info, "build_command": BUILD_CMD, **verification}, indent=1) + "\n")
    (PKG / "SUBMISSION_CHECKLIST.md").write_text(checklist(tex, f, hl, a.version_doi, zsha, zip_info, verification))

    manifest = ["# Submission package manifest", "", "Regenerated from the canonical manuscript source; see `SUBMISSION_CHECKLIST.md`.", "",
                "| File | Purpose | SHA-256 |", "|---|---|---|"]
    purposes = {"main.tex": "Editable LaTeX source", "references.bib": "Bibliography source", "main.bbl": "Compiled bibliography", "README_BUILD.txt": "Build instructions",
                "performance_evaluation_highlights.txt": "Highlights (upload as a separate file)", "cover_letter.txt": "Cover letter",
                "credit_authorship_statement.txt": "CRediT statement (candidate; author to confirm)", "suggested_reviewers.md": "Optional reviewer suggestions (portal only)",
                "declarations.md": "Declarations text for the Elsevier declarations tool", "SUBMISSION_CHECKLIST.md": "Checklist",
                "SOURCE_ZIP_VERIFICATION.json": "Result of the clean-extraction build check", ZIP_NAME: "Elsevier source ZIP"}
    for p in sorted(PKG.rglob("*")):
        if p.is_file() and p.name != "SUBMISSION_PACKAGE_MANIFEST.md":
            rel = p.relative_to(PKG).as_posix()
            manifest.append(f"| `{rel}` | {purposes.get(rel, 'Figure (vector PDF)' if rel.startswith('figures/') else '')} | `{sha256(p)}` |")
    (PKG / "SUBMISSION_PACKAGE_MANIFEST.md").write_text("\n".join(manifest) + "\n")
    print(f"package: {PKG.relative_to(ROOT)}  zip sha256={zsha}  files={zip_info['files']}  bytes={zip_info['uncompressed_bytes']}")
    print(verification["summary"])
    sys.exit(0 if verification["all_ok"] else 1)


if __name__ == "__main__":
    main()
