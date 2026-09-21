"""Technical and visual-consistency guards for the six Performance Evaluation manuscript figures.

The figures are drawn at their final size and included at scale 1, so the point sizes checked here are the sizes a
reader sees.  Encoding must be grayscale-safe (no hue is ever required), fonts must be embedded, and the plotted
values must come from the frozen artifacts rather than from numbers typed into a script.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "performance_evaluation"
TEX = PAPER / "main.tex"
FIG = PAPER / "figures"
SCRIPTS = PAPER / "scripts"
NAMES = ["pe_pipeline", "pe_disagreement_rates", "pe_regime_map", "pe_robustness"]
LINEWIDTH_PT = 390.0  # \linewidth of the elsarticle preprint 12 pt class
MIN_TEXT_PT = 7.0
MATH_SUBSCRIPT = {"LAT"}  # mathtext subscripts are drawn at 70 % of the parent size (about 5.3-5.6 pt)

pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.skipif(not TEX.exists(), reason="manuscript source not present")


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text()


def test_manuscript_includes_the_four_figures_at_natural_size(tex):
    inc = re.findall(r"\\includegraphics\[([^\]]*)\]\{figures/([^}]+)\.pdf\}", tex)
    assert [n for _, n in inc] == NAMES  # order of appearance = Figures 1-4
    for opts, name in inc:
        assert opts.strip() == "scale=1", f"{name}: figures are drawn at final size and must not be rescaled ({opts})"


@pytest.mark.parametrize("name", NAMES)
def test_figure_pdf_is_vector_embedded_and_fits_the_column(name):
    pdf = FIG / f"{name}.pdf"
    doc = pymupdf.open(pdf)
    assert len(doc) == 1
    page = doc[0]
    assert page.rect.width <= LINEWIDTH_PT + 1e-6, f"{name}: {page.rect.width:.1f} pt exceeds the text width"
    assert page.get_images() == [], f"{name}: contains raster images (text or marks must be vector)"
    assert page.get_text("text").strip(), f"{name}: no live text"
    if shutil.which("pdffonts"):
        rows = [ln.split() for ln in subprocess.check_output(["pdffonts", str(pdf)], text=True).splitlines()[2:] if ln.strip()]
        assert rows and all(r[-5] == "yes" for r in rows), f"{name}: unembedded fonts"
        assert not any("Type" in r[1:3] and "3" in r[1:3] for r in rows), f"{name}: Type 3 fonts"


@pytest.mark.parametrize("name", NAMES)
def test_figure_text_is_legible_at_actual_size(name):
    page = pymupdf.open(FIG / f"{name}.pdf")[0]
    spans = [s for b in page.get_text("dict")["blocks"] if b["type"] == 0 for ln in b["lines"] for s in ln["spans"] if s["text"].strip()]
    assert spans
    small = [(round(s["size"], 2), s["text"]) for s in spans if s["size"] < MIN_TEXT_PT]
    assert all(t in MATH_SUBSCRIPT for _, t in small), f"{name}: text below {MIN_TEXT_PT} pt: {small}"
    assert max(s["size"] for s in spans) <= 9.0, f"{name}: text larger than the common figure scale"
    assert {s["font"].split("+")[-1] for s in spans} <= {"DejaVuSans", "DejaVuSans-Oblique", "DejaVuSans-Bold", "DejaVuSans-BoldOblique"}, f"{name}: mixed typefaces"


@pytest.mark.parametrize("name", NAMES)
def test_figure_png_is_high_resolution_and_grayscale_safe(name):
    Image = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    im = Image.open(FIG / f"{name}.png")
    width_pt = pymupdf.open(FIG / f"{name}.pdf")[0].rect.width
    assert im.size[0] >= 0.99 * width_pt / 72.0 * 300, f"{name}: PNG below 300 dpi"
    rgb = np.asarray(im.convert("RGB")).astype(int)
    assert int((rgb.max(axis=2) - rgb.min(axis=2)).max()) == 0, f"{name}: uses colour; every figure must read in grayscale"


def test_figure_scripts_use_the_shared_style_and_no_hard_coded_data():
    style = (SCRIPTS / "figstyle.py").read_text()
    assert '"pdf.fonttype": 42' in style and "DejaVu Sans" in style
    for f in ("plot_performance_evaluation_figures.py", "plot_regime_figures.py", "plot_robustness_figures.py"):
        src = (SCRIPTS / f).read_text()
        assert "figstyle.py" in src, f
        assert "plt.rcParams.update" not in src and "bbox_inches" not in src, f"{f}: bypasses the shared style/export"
    fig23 = (SCRIPTS / "plot_performance_evaluation_figures.py").read_text()
    assert not re.search(r"\b0\.0009987|0\.0059532|0\.0000670|\"BurstGPT\\nactive_8\"", fig23), "Figure 2/3 values must be read from the Phase A/B artifacts"
    assert "PHASE_A_WORKLOAD_SUMMARY_V1.csv" in fig23 and "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv" in fig23


def test_figure_2_uses_the_artifact_value_for_burstgpt_cap_8():
    """Regression: the earlier Figure 2 plotted a hand-typed 0.0003 for BurstGPT cap 8; the artifact says 1.71e-4."""
    import csv
    rows = list(csv.DictReader((ROOT / "experiments/industry_realism_action_opportunity_phase_b_v2/PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv").open()))
    r = next(x for x in rows if x["source_dataset"] == "burstgpt" and x["axis"] == "active_sequence_capacity" and float(x["axis_value"]) == 8)
    assert round(float(r["disagreement_rate"]), 6) == 0.000171
    tokens = pymupdf.open(FIG / "pe_disagreement_rates.pdf")[0].get_text("text").split()
    assert "0.0171" in tokens and "0.03" not in tokens


def test_figures_regenerate_from_the_frozen_artifacts(tmp_path):
    """Regenerating into a scratch directory reproduces the committed figures (same size, same text)."""
    env = dict(os.environ, PEVA_FIG_DIR=str(tmp_path), MPLBACKEND="Agg")
    for f in ("plot_performance_evaluation_figures.py", "plot_regime_figures.py", "plot_robustness_figures.py"):
        subprocess.run([sys.executable, str(SCRIPTS / f)], check=True, env=env, capture_output=True, cwd=ROOT)
    for name in NAMES:
        new, old = pymupdf.open(tmp_path / f"{name}.pdf")[0], pymupdf.open(FIG / f"{name}.pdf")[0]
        assert (round(new.rect.width, 1), round(new.rect.height, 1)) == (round(old.rect.width, 1), round(old.rect.height, 1)), name
        assert new.get_text("text") == old.get_text("text"), f"{name}: committed figure is stale relative to the artifacts"


def test_captions_use_consistent_figure_terminology(tex):
    flat = re.sub(r"\s+", " ", tex)
    caps = re.findall(r"\\begin\{figure\}.*?\\caption\{(.*?)\}\s*\\label", flat)
    assert len(caps) == 4
    for c in caps:
        assert "canonical disagreement rate" not in c.lower() and "16k" not in c and "1e-3" not in c
    assert "Disagreement prevalence $P(D)$" in caps[1]
    assert "disagreement prevalence $P(D)$" in caps[2]
