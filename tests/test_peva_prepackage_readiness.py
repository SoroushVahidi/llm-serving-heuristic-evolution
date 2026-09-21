"""Pre-package readiness guards: declarations, Data Availability accuracy, README/build accuracy, checklist consistency."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "performance_evaluation"
TEX = PAPER / "main.tex"
README = PAPER / "README.md"
READINESS = ROOT / "docs" / "PEVA_PREPACKAGE_READINESS.md"
BUILD = ROOT / "scripts" / "build_performance_evaluation_manuscript.sh"
OLD_DOI = "10.5281/zenodo.22865294"

pytestmark = pytest.mark.skipif(not TEX.exists(), reason="manuscript source not present")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text()


def _section(tex, title):
    m = re.search(r"\\section\*\{" + re.escape(title) + r"\}\n(.*?)(?=\n\\section|\n\\bibliographystyle)", tex, re.S)
    assert m, title
    return re.sub(r"\s+", " ", m.group(1))


def test_generative_ai_declaration_names_only_the_tools_actually_used(tex):
    d = _section(tex, "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process")
    for tool in ("ChatGPT (OpenAI)", "Codex (OpenAI)", "Gemini (Google)", "Claude (Anthropic)"):
        assert tool in d, tool
    assert not re.search(r"Copilot|Grammarly|DeepL|Llama|Mistral|Cohere|Bard|Perplexity", d)
    for needed in ("the author used", "in order to assist with", "the author reviewed and edited the content", "takes full responsibility"):
        assert needed in d, needed
    for forbidden in ("autonomous", "decided", "made the scientific", "conducted the experiments"):
        assert forbidden not in d, forbidden
    # deterministic plotting code does not receive its own AI note
    assert "AI" not in re.sub(r"\s+", " ", "".join(re.findall(r"\\caption\{.*?\}\s*\\label", tex, re.S)))


def test_data_availability_makes_no_false_claim_about_the_existing_archive(tex):
    d = _section(tex, "Data and Code Availability")
    assert "https://github.com/SoroushVahidi/llm-serving-heuristic-evolution" in d
    if OLD_DOI in d:  # the archive predating the robustness work must not be presented as containing it
        assert "predates the post hoc robustness analysis" in d and "available in the GitHub repository only" in d
        assert "The derived research data and reproducibility package have been permanently archived" not in d
    assert "TODO" not in tex and "Query 8" not in tex and "QUERY_8" not in tex  # the marker lives in repository documentation only


def test_zenodo_todo_marker_is_resolved():
    doc = READINESS.read_text()
    assert "TODO(QUERY_8): ZENODO_NEW_VERSION" not in doc
    assert "TODO(QUERY_8)" not in README.read_text()


def test_readme_names_the_real_build_and_figure_commands():
    r = README.read_text()
    for cmd in ("scripts/build_performance_evaluation_manuscript.sh", "plot_performance_evaluation_figures.py", "plot_regime_figures.py", "plot_robustness_figures.py",
                "build_claim_manifest.py", "pdflatex", "bibtex", "figstyle.py", "FINAL_CLAIM_MANIFEST.json", "when_does_llm_serving_scheduler_adaptation_matter.pdf"):
        assert cmd in r, cmd
    for path in re.findall(r"`((?:tests|scripts|docs)/[A-Za-z0-9_./-]+\.(?:py|sh|md))`", r):
        assert (ROOT / path).exists() or (PAPER / path).exists(), f"README names a missing file: {path}"
    b = BUILD.read_text()
    for script in ("plot_performance_evaluation_figures.py", "plot_regime_figures.py", "plot_robustness_figures.py", "build_claim_manifest.py", "--check"):
        assert script in b, script
    assert (ROOT / "paper" / "when_does_llm_serving_scheduler_adaptation_matter.pdf").exists()


def test_readiness_checklist_is_complete_and_consistent():
    doc = READINESS.read_text()
    verdict = re.findall(r"^FINAL_VERDICT: (\S+)$", doc, re.M)
    assert verdict in (["MANUSCRIPT_READY_FOR_ARCHIVE_AND_PACKAGING"], ["NOT_READY"])
    rows = re.findall(r"^\| ([^|]+?) \| (READY|BLOCKED_FOR_QUERY_8) \|", doc, re.M)
    status = dict(rows)
    manuscript = ["Title", "Abstract", "Keywords", "Introduction", "Related Work", "Methods", "Results", "Discussion", "Conclusion", "References", "AI declaration",
                  "Acknowledgements", "Data Availability"]
    visuals = [f"Figure {i}" for i in range(1, 7)] + ["Tables 1-6"]
    provenance = ["Frozen primary artifacts", "Corrected derivative", "Robustness outputs", "Claim manifest"]
    deferred = ["Merge / integration into `main`", "Zenodo new version", "Dataset / software citation", "`submission/main.tex` regeneration", "Source zip", "Highlights",
                "Cover letter", "Submission checklist", "Elsevier declarations tool", "Suggested reviewers"]
    for item in manuscript + provenance + deferred:
        assert item in status, item
    for i in range(1, 7):
        assert any(k.startswith(f"Figure {i} ") for k in status), i
    assert "Tables 1-6" in status
    assert all(status[k] == "BLOCKED_FOR_QUERY_8" for k in deferred)
    if verdict == ["MANUSCRIPT_READY_FOR_ARCHIVE_AND_PACKAGING"]:
        assert all(v == "READY" for k, v in status.items() if k not in deferred), [k for k, v in status.items() if v != "READY" and k not in deferred]


def test_no_unreleased_claims_in_the_manuscript_about_submission_status(tex):
    body = tex[tex.index(r"\section{Introduction}"):tex.index(r"\bibliographystyle")]
    for bad in ("Query 7", "Query 8", "Zenodo version", "highlights", "cover letter"):
        assert bad not in body, bad
