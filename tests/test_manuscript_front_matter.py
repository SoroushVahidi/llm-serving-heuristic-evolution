"""Guards for the Performance Evaluation manuscript front matter and the numbers quoted in it.

* Journal constraints: abstract <= 250 words, 1-7 keywords, abstract without citations.
* Structural: title/abstract/keywords sit in the elsarticle ``frontmatter`` (a bare ``\\maketitle`` silently drops
  the abstract and keywords from the PDF).
* Numeric: every result quoted in the abstract/Introduction matches the frozen confirmatory artifacts and the
  post hoc robustness outputs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "performance_evaluation" / "main.tex"
BIB = ROOT / "paper" / "performance_evaluation" / "references.bib"
FROZEN = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1"
ROBUST = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1_robustness"

pytestmark = pytest.mark.skipif(not TEX.exists(), reason="manuscript source not present")


def _tex() -> str:
    return TEX.read_text()


def _block(name: str) -> str:
    m = re.search(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (name, name), _tex(), re.S)
    assert m, f"{name} environment missing"
    return m.group(1).strip()


def _plain(s: str) -> str:
    s = re.sub(r"\\%", "%", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return re.sub(r"[{}~]", " ", s)


def _intro() -> str:
    t = _tex()
    return t[t.index(r"\section{Introduction}"):t.index(r"\section{Related Work and Positioning}")]


def test_front_matter_is_inside_frontmatter_environment():
    t = _tex()
    assert r"\begin{frontmatter}" in t and r"\end{frontmatter}" in t
    assert r"\maketitle" not in t  # would drop abstract and keywords from the compiled PDF
    fm = t[t.index(r"\begin{frontmatter}"):t.index(r"\end{frontmatter}")]
    for env in (r"\title{", r"\author{", r"\begin{abstract}", r"\begin{keyword}"):
        assert env in fm, env


def test_abstract_within_journal_limit_and_has_no_citations():
    ab = _block("abstract")
    assert r"\cite" not in ab and "[" not in ab
    words = len(_plain(ab).split())
    assert 150 <= words <= 250, words


def test_keywords_between_one_and_seven():
    kws = [k.strip() for k in _block("keyword").split(r"\sep") if k.strip()]
    assert 1 <= len(kws) <= 7, kws
    assert all(len(k.split()) <= 3 for k in kws)


def test_all_cited_keys_exist_in_bibliography():
    keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text()))
    cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", _tex()) for k in grp.split(",")}
    assert cited <= keys, cited - keys


def test_five_research_questions_and_five_contributions():
    intro = _intro()
    assert len(re.findall(r"\\item\[RQ\d", intro)) == 5
    contrib = intro[intro.index(r"\subsection{Contributions}"):]
    assert contrib.count(r"\item ") == 5


# --------------------------------------------------------------- numbers quoted in abstract / Introduction
@pytest.fixture(scope="module")
def result():
    return json.loads((FROZEN / "FRESH_LATENCY_CAUSAL_RESULT_V1.json").read_text())["primary"]


@pytest.mark.skipif(not (FROZEN / "FRESH_LATENCY_CAUSAL_RESULT_V1.json").exists(), reason="frozen artifacts absent")
def test_primary_numbers_match_frozen_result(result):
    fm = _plain(_block("abstract")) + " " + _plain(_intro())
    assert result["states"] == 720 and result["beneficial_states"] == 590
    assert "720" in fm and "590" in fm
    assert f"{100 * result['P_B_given_D']:.1f}" == "81.9" and "81.9" in fm
    assert f"{1e3 * result['mean_oracle_headroom']:.2f}" == "2.00" and "2.00 ms" in fm
    ci = result["clustered_ci95"]
    lo, hi = 1e3 * ci["mean_oracle_headroom_ci95_low"], 1e3 * ci["mean_oracle_headroom_ci95_high"]
    assert f"{lo:.2f}" == "0.19" and f"{hi:.2f}" == "3.55" and "0.19--3.55" in _tex()
    assert ci["clusters"] == 36 and "36" in fm


@pytest.mark.skipif(not (ROBUST / "composition_summary.json").exists(), reason="robustness outputs absent")
def test_robustness_numbers_match_post_hoc_outputs():
    import csv

    comp = json.loads((ROBUST / "composition_summary.json").read_text())
    dist = list(csv.DictReader((ROBUST / "distribution_summary.csv").open()))[0]
    thr = {(r["scope"], float(r["threshold_ms"])): r for r in csv.DictReader((ROBUST / "practical_thresholds.csv").open())}
    tol = json.loads((ROBUST / "numerical_tolerance_audit.json").read_text())
    txt = _plain(_block("abstract")) + " " + _plain(_intro())
    assert f"{100 * comp['top_window_share']:.0f}" == "88" and "88%" in txt
    assert f"{float(dist['median_ms']):.2f}" == "0.20" and "0.20 ms" in txt
    assert f"{100 * float(thr[('global', 2.0)]['fraction']):.1f}" == "22.5" and "22.5\\%" in _tex()  # reported in Section 6, not in the Introduction
    assert tol["beneficial_states_with_eps_guard"] == 589 and tol["canonical_beneficial_states_strict_gt0"] == 590
    assert "589/720" in _tex()  # the guarded count is reported with the primary result (Section 5), not in the Introduction


def test_scope_statements_present_once_and_no_deployment_claim():
    intro = _plain(_intro())
    assert intro.count("opportunity bound") == 1
    assert "not a scheduler" in intro
    forbidden = ("we propose a new scheduler", "production deployment", "deployment-ready")
    assert not any(f in intro.lower() for f in forbidden)
