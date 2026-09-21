"""Presentation, terminology and compliance guards for the Performance Evaluation manuscript source."""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "performance_evaluation"
TEX = PAPER / "main.tex"
BIB = PAPER / "references.bib"
MOD = PAPER / "scripts" / "robustness_numbers.py"
FROZEN = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1"

pytestmark = pytest.mark.skipif(not TEX.exists(), reason="manuscript source not present")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text()


@pytest.fixture(scope="module")
def flat(tex):
    return re.sub(r"\s+", " ", tex)


def _sect(flat, a, b):
    return flat[flat.index(a):flat.index(b)]


# ------------------------------------------------------------------ terminology
def test_no_terminology_drift(tex):
    body = tex[tex.index(r"\section{Introduction}"):tex.index(r"\bibliographystyle")]
    for bad in ("single-best", "Resource-Regime", "real-vLLM", "Real-vLLM", "validation", "The earlier The earlier"):
        assert bad not in body, bad
    # the long form is used only where SBS is (re)defined or the abstract stands alone
    assert len(re.findall(r"single best scheduler", tex)) <= 4
    assert "single best scheduler (SBS)" in tex
    assert r"\newcommand{\sbs}{\mathrm{SBS}}" in tex and r"\newcommand{\anwg}{\mathrm{ANWG}}" in tex


def test_vllm_named_as_system_correspondence_probe(tex, flat):
    assert r"\section{A Bounded vLLM System Correspondence Probe}" in tex
    sec = _sect(flat, r"\section{A Bounded vLLM", r"\section{Discussion}")
    assert "The simulator predicted a reversal between the two classes" in sec and "did not reproduce this reversal" in sec
    assert "does not estimate Eq.~\\ref{eq:latency} in the real system" in sec
    assert "Level 3" in sec and "is not established" in sec


def test_regime_and_condition_defined_consistently(flat):
    assert "A pressure condition is a workload window plus one fixed one-axis intervention" in flat
    assert "five pressure \\emph{regimes}, each a workload, pressure axis, and pressure setting pooled over windows" in flat


# ------------------------------------------------------------------ Table 3 / Figures 4-5 consistency
def test_table3_rows_match_verified_regime_numbers(flat):
    spec = importlib.util.spec_from_file_location("robustness_numbers", MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    R = sorted(m.compute()["regimes"], key=lambda r: (r["workload"] != "azure_2023_code", r["axis"] != "active_sequence_capacity",
                                                     -int(r["setting"].split("_")[1]) if r["axis"] == "active_sequence_capacity" else 0))
    assert len(R) == 5
    for i, r in enumerate(R, start=1):
        cells = f"{r['states']} & {r['windows']} & {100 * r['P_D']:.3g} & {100 * r['P_B']:.1f} & {r['mean_ms']:.3f} & {100 * r['share']:.2f} \\\\"
        assert cells in flat, cells
        assert f"({i}) Azure " in flat
    assert sum(r["states"] for r in R) == 720
    assert "All selected regimes & 720 & 36 & --- & 81.9 & 1.996 & 100.00" in flat


def test_table_and_figure_style(tex):
    tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.S)
    assert len(tables) == 6
    for t in tables:
        assert r"\caption{" in t and r"\label{" in t and r"\toprule" in t and r"\bottomrule" in t
        assert "|" not in re.search(r"\\begin\{tabular[x]?\}.*?\n", t).group(0).replace(r"\mid", "")  # no vertical rules
    assert "records the scope honestly" not in tex and "We do not claim" not in tex
    assert "Source win" not in tex


def test_figure_files_exist_are_vector_and_embed_fonts(tex):
    files = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    assert len(files) == 4
    for f in files:
        pdf = PAPER / f
        assert pdf.exists() and pdf.suffix == ".pdf", f
        assert pdf.with_suffix(".png").exists(), f
    if shutil.which("pdffonts"):
        for f in ("figures/pe_pipeline.pdf", "figures/pe_regime_map.pdf", "figures/pe_robustness.pdf"):
            out = subprocess.check_output(["pdffonts", str(PAPER / f)], text=True).splitlines()[2:]
            assert out and all(line.split()[-5] == "yes" for line in out if line.strip()), f"unembedded fonts in {f}"


# ------------------------------------------------------------------ Related Work and PEVA references
def test_peva_references_present_cited_and_metadata_pinned(tex):
    bib = BIB.read_text()
    for key, vol, page, doi in (("yildiz2026dispatching", "172", "102551", "10.1016/j.peva.2026.102551"),
                                ("liang2026lilou", "171", "102539", "10.1016/j.peva.2025.102539"),
                                ("choudhury2025job", "167", "102463", "10.1016/j.peva.2024.102463")):
        entry = re.search(r"@article\{" + key + r",(.*?)\n\}", bib, re.S).group(1)
        assert "Performance Evaluation" in entry and f"volume = {{{vol}}}" in entry and f"pages = {{{page}}}" in entry and doi in entry, key
        assert f"\\cite{{{key}}}" in tex or key in ",".join(re.findall(r"\\cite\{([^}]*)\}", tex)), key
    rel = tex[tex.index(r"\section{Related Work"):tex.index(r"\section{Methods}")]
    assert "yildiz2026dispatching" in rel and "liang2026lilou" in rel


def test_related_work_is_not_defensive(tex):
    rel = tex[tex.index(r"\section{Related Work"):tex.index(r"\section{Methods}")]
    for bad in ("narrow any generic claim", "honestly", "do not claim", "not unimplemented baselines", "novelty claim is not"):
        assert bad not in rel, bad
    assert "Yes & No" not in rel and "Partial" not in rel  # descriptive table, not a scorecard


def test_introduction_equation_is_split_across_three_lines(tex):
    eq = re.search(r"\\begin\{equation\}\s*\\begin\{aligned\}\s*\\text\{resource pressure\}.*?\\label\{eq:chain\}", tex, re.S).group(0)
    assert eq.count(r"\\") == 2 and r"\longrightarrow\text{predictability and deployment}" in eq


# ------------------------------------------------------------------ journal / declarations
def test_generative_ai_declaration_elements(tex):
    d = re.search(r"\\section\*\{Declaration of generative AI[^}]*\}\n(.*?)\n\n", tex, re.S).group(1)
    for s in ("the author used", "in order to", "the author reviewed and edited the content", "takes full responsibility"):
        assert s in d, s
    assert d.index("the author used") < d.index("reviewed and edited")
    assert tex.index(r"\section*{Declaration of generative AI") < tex.index(r"\bibliographystyle")
    assert tex.index(r"\section*{Acknowledgements}") < tex.index(r"\bibliographystyle")


def test_preamble_uses_standard_packages(tex):
    assert r"\usepackage{tools/array}" not in tex and r"\usepackage{array}" in tex
    assert r"\bibliographystyle{elsarticle-num}" in tex


# ------------------------------------------------------------------ abstract claims traceable to the body
def test_abstract_claims_are_supported_in_body(flat, tex):
    ab = _sect(flat, r"\begin{abstract}", r"\end{abstract}")
    body = flat[flat.index(r"\section{Introduction}"):flat.index(r"\bibliographystyle")]
    # about one million action-null decision states = sum of the three native counts in Section 4
    counts = [99992, 461985, 440461]  # native decision-state counts reported in Section 4
    assert "99,992" in body and "461,985" in body and "440,461" in body and 0.95e6 < sum(counts) < 1.05e6 and "about one million" in ab
    assert "Arrival scaling through $8\\times$ produced no disagreement state" in body and "up to eight times" in ab
    assert "590/720=0.8194" in body and "590 (81.9\\%)" in ab
    assert "[0.1909, 3.5462]" in body and "0.19--3.55" in ab
    assert re.search(r"median (?:is )?0\.20~ms", body) and "the median is 0.20 ms" in ab
    assert "88.4\\%" in body and "88\\%" in ab
    assert "did not reproduce this reversal" in body and "does not validate the simulated latency effect" in ab
