"""Guards for the post hoc robustness integration in the Performance Evaluation manuscript.

Every number quoted in the new Methods/Results text and tables is recomputed by
``paper/performance_evaluation/scripts/robustness_numbers.py`` from the frozen artifacts (cross-checked against the
robustness outputs) and searched for in ``main.tex``.  The preregistered primary values must remain unchanged.
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
    return m.compute()  # asserts internal consistency with the frozen and robustness artifacts


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text()


@pytest.fixture(scope="module")
def robust_section(tex):
    main = tex[tex.index(r"\section{Concentration, Sensitivity, and Reference Dependence}"):tex.index(r"\section{A Bounded vLLM System Correspondence Probe}")]
    s = main + " " + tex[tex.index(r"\section*{Appendix A."):tex.index(r"\section*{Declaration of generative")]  # Appendix A carries the interval diagnostics
    return re.sub(r"\s+", " ", s)  # LaTeX source wraps lines mid-phrase


@pytest.fixture(scope="module")
def fresh_section(tex):
    s = tex[tex.index(r"\section{Causal Headroom on Untouched Windows}"):tex.index(r"\section{Concentration, Sensitivity, and Reference Dependence}")]
    return re.sub(r"\s+", " ", s)


def pct(x):
    return f"{100 * x:.1f}\\%"


# ----------------------------------------------------------------------------- preregistered result preserved
def test_primary_result_unchanged_and_stated_first(fresh_section, N):
    assert "590/720=0.8194\\ (81.9\\%)" in fresh_section
    assert "0.001995791 s (1.9958 ms)" in fresh_section
    assert "[0.000190884, 0.003546196] s ([0.1909, 3.5462] ms)" in fresh_section
    assert N["primary"]["beneficial"] == 590 and abs(N["primary"]["mean_ms"] - 1.9958) < 5e-5
    assert N["primary"]["replicates"] == 2000 and N["primary"]["seed"] == 20260920 and N["primary"]["clusters"] == 36
    assert "2,000" in TEX.read_text() and "20260920" in TEX.read_text()
    assert fresh_section.index("590 beneficial") < fresh_section.index("post hoc analyses")  # primary result precedes the qualification


def test_numerical_residue_sentence(fresh_section, N):
    r = N["residue"]
    assert r["ulps"] == 1.0 and r["identical_completed_requests"]
    assert f"{r['advantage_s'] * 1e17:.1f}" == "5.6" and "$5.6\\times10^{-17}$" in fresh_section
    assert f"{r['ref_latency_s']:.3f}" == "0.357" and "0.357~s" in fresh_section
    assert f"{N['primary']['guarded_beneficial']}/720 ({100 * 589 / 720:.1f}\\%)" in fresh_section
    assert "590/720" in fresh_section and "589/720" in fresh_section  # both reported; 590 remains the preregistered count


# ----------------------------------------------------------------------------- tables
def test_threshold_table_rows(robust_section, N):
    for r in N["thresholds"]:
        cells = f"{r['n']} & {pct(r['share'])} & {r['windows']} & {r['in_code_kv']} \\\\"
        assert cells in robust_section, cells
    assert N["thresholds"][0]["n"] == 590 and N["thresholds"][1]["n"] == 589


def test_sensitivity_table_rows(robust_section, N):
    for w in N["weighting"]:
        cells = f"{w['mean_ms']:.3f} & [{w['ci_ms'][0]:.3f}, {w['ci_ms'][1]:.3f}] & {w['P']:.3f} \\\\"
        assert cells in robust_section, cells
    L = N["loo"]
    w11 = L["window"]["azure_2023_code:w11"]
    ck = L["regime"]["azure_2023_code|kv_capacity|kv_16000"]
    for v in (w11, ck):
        cells = f"{v['n']} states & {v['mean_ms']:.3f} & [{v['ci_ms'][0]:.3f}, {v['ci_ms'][1]:.3f}] & {v['P']:.3f} \\\\"
        assert cells in robust_section, cells


def test_weighting_claims_hold(N):
    means = {w["name"]: w["mean_ms"] for w in N["weighting"]}
    assert all(v > 0 for v in means.values())
    assert all(w["ci_ms"][0] > 0 for w in N["weighting"])
    ratio = N["weighting"][0]["mean_ms"] / N["weighting"][3]["mean_ms"]
    assert 5.5 < ratio < 6.5  # "a factor of about six"
    assert N["loo"]["all_ci_exclude_zero"] and N["loo"]["n_windows"] == 36 and N["loo"]["n_regimes"] == 5


# ----------------------------------------------------------------------------- prose numbers
def test_distribution_prose(robust_section, N):
    d = N["dist"]
    for s in (f"{d['median_ms']:.3f}~ms", f"{d['mean_ms']:.3f}~ms", pct(d["frac_le_mean"]), f"{d['p90_ms']:.1f} and {d['p95_ms']:.1f}~ms", f"{d['max_ms']:.1f}~ms", f"{d['n_zero']} states ({100 * d['frac_zero']:.1f}\\%)",
              f"{d['n_pos']} states with positive headroom the median is {d['pos_median_ms']:.2f}~ms"):
        assert s in robust_section, s
    assert 0.09 < d["median_ms"] / d["mean_ms"] < 0.11  # "about one tenth"


def test_threshold_prose(robust_section, N):
    th = {r.get("tau_ms"): r for r in N["thresholds"] if "tau_ms" in r}
    for tau in (0.1, 0.5, 2.0, 5.0):
        assert pct(th[tau]["share"]) in robust_section
    assert f"{th[0.5]['in_code_kv']} of the {th[0.5]['n']} states" in robust_section
    assert f"only {th[2.0]['windows']} of the 36 windows" in robust_section


def test_concentration_prose(robust_section, N):
    c = N["conc"]
    for s in (f"{c['w11_states']} of the 720 states ({100 * c['w11_state_share']:.1f}\\%)", f"{100 * c['w11_headroom_share']:.1f}\\% of the total summed headroom",
              f"{100 * c['top3_share']:.1f}\\%", f"{c['eff_windows']:.2f} ({c['eff_windows_by_states']:.1f} by state count)",
              f"{100 * c['code_kv_state_share']:.1f}\\% of the states and {100 * c['code_kv_headroom_share']:.1f}\\% of the headroom",
              f"{c['code_kv_contrib_ms']:.2f} of the {N['primary']['mean_ms']:.2f}~ms", f"{c['w11_code_kv_states']} states of that regime average {c['w11_code_kv_mean_ms']:.2f}~ms",
              f"Of the {c['n_gt5']} states above 5~ms, {c['n_gt5_in_w11']} are in w11", f"{c['n_gt2_in_w11']} of the {c['n_gt2']} states above 2~ms"):
        assert s in robust_section, s
    assert "highly concentrated" in robust_section


def test_leave_one_out_prose(robust_section, N):
    L = N["loo"]
    lo_w, hi_w = L["other_windows_range_ms"]
    lo_r, hi_r = L["other_regimes_range_ms"]
    assert f"between {lo_w:.2f} and {hi_w:.2f}~ms" in robust_section
    assert f"between {lo_r:.2f} and {hi_r:.2f}~ms" in robust_section
    assert f"survives all {L['n_windows'] + L['n_regimes']} omissions" in robust_section
    assert f"All {L['n_windows']} single-window and {L['n_regimes']} single-regime omissions leave a positive mean" in robust_section


def test_diagnostics_prose(robust_section, N):
    g = N["diag"]
    for s in (f"[{g['large_ci_ms'][0]:.3f}, {g['large_ci_ms'][1]:.3f}]~ms", f"{g['mc_low_sd_ms']:.3f}~ms", f"across {g['mc_seeds']} seeds",
              f"[{g['bca_ci_ms'][0]:.3f}, {g['bca_ci_ms'][1]:.3f}]~ms", f"{100 * g['frac_without_w11']:.0f}\\%", f"{g['mean_without_w11_ms']:.2f}~ms",
              f"{g['mean_with_w11_ms']:.2f}~ms", f"[$-{abs(g['jack_ci_ms'][0]):.2f}$, {g['jack_ci_ms'][1]:.2f}]~ms", f"({g['jack_se_ms']:.2f}~ms)"):
        assert s in robust_section, s
    assert g["jack_ci_ms"][0] < 0 < g["jack_ci_ms"][1] and g["frac_boot_le_0"] == 0.0
    assert g["large_ci_ms"][0] > 0 and g["bca_ci_ms"][0] > 0


# ----------------------------------------------------------------------------- structure and editorial artifacts
def test_section_order_and_references(tex):
    order = [r"\section{Where Disagreement Appears}", r"\section{Causal Headroom on Untouched Windows}",
             r"\section{Concentration, Sensitivity, and Reference Dependence}", r"\section{A Bounded vLLM System Correspondence Probe}"]
    pos = [tex.index(o) for o in order]
    assert pos == sorted(pos)
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = {r for grp in re.findall(r"\\ref\{([^}]+)\}", tex) for r in [grp]}
    assert refs <= labels, refs - labels
    assert {"sec:robust", "sec:posthoc", "tab:thresholds", "tab:sensitivity", "fig:robust"} <= labels


def test_methods_define_posthoc_analyses(tex):
    m = re.sub(r"\s+", " ", tex[tex.index(r"\subsection{Statistical analysis}"):tex.index(r"\section{Where Disagreement Appears}")]).lower()
    for s in ("state-weighted mean", "equal-window", "equal-regime", "equal-workload", "leave-one-out", "bca", "jackknife", "10^{-12}", "0.1,0.25,0.5,1,2,5",
              "not alternative estimates", "diagnostics, not replacements", "weight $1/u$", "weight $1/|d|$"):
        assert s in m, s


def test_no_known_editing_artifacts_in_touched_sections(tex):
    for bad in ("The earlier The earlier", "old The earlier", "arrival-normalized weighted goodput Saturation", "six-policy is a fixed portfolio",
                "The earlier's all-terminal"):
        assert bad not in tex, bad
    assert r"\newcommand{\sbs}{\mathrm{SBS}}" in tex and r"\newcommand{\anwg}{\mathrm{ANWG}}" in tex


def test_no_representativeness_or_deployment_language_in_new_section(robust_section):
    low = robust_section.lower()
    for bad in ("robust 2 ms", "consistent 2 ms", "adaptation produces", "production gain", "deployment benefit", "anomalous"):
        assert bad not in low, bad
    assert "post hoc" in low and "pre-specified" in low
