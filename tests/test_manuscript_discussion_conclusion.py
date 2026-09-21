"""Guards for the Discussion, limitations, practical implications and Conclusion.

Numbers quoted there are checked against the frozen artifacts and the post hoc robustness outputs (via
``paper/performance_evaluation/scripts/robustness_numbers.py``); wording guards keep the interpretation consistent
with the evidence (sign stable, magnitude concentrated, the 1.9958 ms mean never presented as robust or typical).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "performance_evaluation" / "main.tex"
MOD = ROOT / "paper" / "performance_evaluation" / "scripts" / "robustness_numbers.py"
FROZEN = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1"
ROBUST = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1_robustness"

pytestmark = pytest.mark.skipif(not (TEX.exists() and (FROZEN / "FRESH_LATENCY_STATE_LEVEL_V1.csv").exists() and ROBUST.exists()),
                                reason="manuscript or artifacts not present")


@pytest.fixture(scope="module")
def N():
    spec = importlib.util.spec_from_file_location("robustness_numbers", MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = m.compute()
    out["_W11"], out["_KV"] = m.W11, m.CODE_KV
    return out


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text()


def _ws(s):
    return re.sub(r"\s+", " ", s)


@pytest.fixture(scope="module")
def discussion(tex):
    return _ws(tex[tex.index(r"\section{Discussion}"):tex.index(r"\section{Conclusion}")])


@pytest.fixture(scope="module")
def conclusion(tex):
    return _ws(tex[tex.index(r"\section{Conclusion}"):tex.index(r"\section*{Declaration of generative")])


def pct(x):
    return f"{100 * x:.1f}\\%"


def test_discussion_structure(discussion):
    for h in ("Sign and magnitude of the headroom", "Opportunity, prediction, and deployment", "Resource pressure and action support",
              "System correspondence", "Implications for practice", "Limitations and threats to validity"):
        assert r"\subsection{" + h + "}" in discussion, h
    for lim in ("Policy portfolio", "Workloads", "Controlled interventions", "One-step oracle", "Concentration", "System correspondence",
                "Objective", "External validity", "Reference scheduler", "Workload overlays"):
        assert r"\item[" + lim + ".]" in discussion, lim


def test_discussion_numbers_match_artifacts(discussion, N):
    P, D = N["primary"], N["dist"]
    th = {r.get("tau_ms"): r for r in N["thresholds"]}
    W = {w["name"].split(" (")[0]: w for w in N["weighting"]}
    c, L, g = N["conc"], N["loo"], N["diag"]
    w11, kv = L["window"][N["_W11"]], L["regime"][N["_KV"]]
    expect = [
        f"{P['beneficial']} of {P['n']} disagreement states ({pct(P['beneficial'] / P['n'])})",
        f"{P['mean_ms']:.2f}~ms", f"[{P['ci_ms'][0]:.2f}, {P['ci_ms'][1]:.2f}]~ms", f"is {D['median_ms']:.2f}~ms",
        f"{W['equal-workload']['mean_ms']:.2f}, {W['equal-regime']['mean_ms']:.2f}, and {W['equal-window']['mean_ms']:.2f}~ms",
        f"{pct(th[0.5]['share'])} of disagreement states offer more than 0.5~ms", f"{pct(th[2.0]['share'])} more than 2~ms",
        f"carries {pct(c['w11_headroom_share'])}", f"carries {pct(c['code_kv_headroom_share'])}", f"is {c['eff_windows']:.2f}",
        f"{w11['mean_ms']:.2f}~ms (95\\% CI [{w11['ci_ms'][0]:.2f}, {w11['ci_ms'][1]:.2f}]~ms)",
        f"{kv['mean_ms']:.2f}~ms (95\\% CI [{kv['ci_ms'][0]:.2f}, {kv['ci_ms'][1]:.2f}]~ms)",
        f"{N['thresholds'][0]['windows']} of the 36 windows contain a state with positive headroom",
        f"the median disagreement state offers only {D['median_ms']:.2f}~ms",
    ]
    for s in expect:
        assert s in discussion, s
    assert g["jack_ci_ms"][0] < 0 < g["jack_ci_ms"][1] and g["large_ci_ms"][0] > 0 and g["bca_ci_ms"][0] > 0


def test_conclusion_numbers_and_ending(conclusion, N):
    P, D, c = N["primary"], N["dist"], N["conc"]
    W = {w["name"].split(" (")[0]: w for w in N["weighting"]}
    for s in (f"{P['beneficial']} of {P['n']} under the preregistered criterion", f"{P['mean_ms']:.2f}~ms", f"offers {D['median_ms']:.2f}~ms",
              f"{W['equal-window']['mean_ms']:.2f}~ms when windows are weighted equally", f"holds {pct(c['w11_headroom_share'])} of the total"):
        assert s in conclusion, s
    last = re.split(r"(?<=\.) ", conclusion.strip())[-1].lower()
    assert "not by its average" in last and "future work" not in last
    paragraphs = [p for p in re.split(r"\n\s*\n", re.sub(r"[ \t]+", " ", TEX.read_text()[TEX.read_text().index(r"\section{Conclusion}"):TEX.read_text().index(r"\section*{Declaration of generative")])) if p.strip()]
    assert 3 <= len(paragraphs) - 1 <= 4  # heading block + 3 paragraphs


def test_mean_never_called_robust_or_typical(discussion, conclusion):
    for sent in re.split(r"(?<=[.;]) ", discussion + " " + conclusion):
        if "1.9958" in sent or "2.00~ms" in sent:
            low = sent.lower()
            assert "robust" not in low, sent
            assert not re.search(r"\b(consistent|reliable) (2|two)", low), sent
    low = (discussion + conclusion).lower()
    for bad in ("robust 2 ms", "consistent 2 ms", "adaptation produces", "production gain", "deployment benefit", "outlier", "anomalous"):
        assert bad not in low, bad


def test_defensive_patterns_removed(discussion, conclusion):
    low = (discussion + " " + conclusion).lower()
    for bad in ("we do not claim", "we make no claim", "no claim", "we caution", "it is important to emphasize", "should not be interpreted"):
        assert bad not in low, bad


def test_vllm_characterized_as_probe_not_validation(discussion, conclusion, tex):
    assert "bounded system correspondence probe" in discussion
    low = (discussion + " " + conclusion).lower()
    assert "strong validation" not in low and "validated" not in low
    assert "vllm validation" not in low
    for s in ("did not reproduce the simulator", "does not validate the simulated headroom magnitude"):
        assert s in discussion, s
    assert "vLLM pressure/action validation" not in tex


def test_terminology_and_scope_definitions(discussion, conclusion):
    assert "single-best" not in (discussion + conclusion).lower()
    assert "single best scheduler" not in discussion  # SBS is defined in Methods; the Conclusion redefines it once
    assert conclusion.count("single best scheduler (SBS)") == 1
    for s in ("one-step oracle headroom under SBS continuation", "after the outcomes are observed", "production-derived workloads under controlled resource interventions"):
        assert s in discussion.replace("~", " ") or s in discussion, s
    assert "staged" in discussion.lower() or "stages" in discussion.lower()


def test_sign_stable_magnitude_concentrated_stated(discussion):
    low = discussion.lower()
    assert "the sign is stable" in low and "the magnitude is not" in low
    assert "highly localized scheduling opportunity" in low
    assert "criterion on the sign of the causal effect, not on its operational size" in low
