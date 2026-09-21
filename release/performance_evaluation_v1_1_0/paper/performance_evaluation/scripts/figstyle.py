"""Shared typography, encoding and export rules for all six manuscript figures.

Every figure is drawn at its FINAL size: the PDF page is at most the journal text width (390 pt = 5.4 in) and the
manuscript includes it at scale 1, so the point sizes below are the sizes a reader sees.  Encoding is grayscale-safe;
no colour is used and no distinction depends on hue:

* dark gray fill  = KV-capacity pressure axis
* white fill      = active-sequence-cap pressure axis
* hatching / line style carry any further distinction.

Fonts are embedded as TrueType (Type 42), so text stays live and selectable in the vector PDF.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# PEVA_FIG_DIR redirects the output (used by the tests to regenerate the figures without touching the committed files)
FIG_DIR = Path(os.environ.get("PEVA_FIG_DIR") or Path(__file__).resolve().parents[1] / "figures")
# \linewidth is 390 TeX pt = 5.396 in; 5.39 in (388.1 bp = 389.5 TeX pt) never exceeds it.
TEXT_WIDTH_IN = 5.39

DARK, LIGHT = "0.30", "white"
WORKLOAD = {"azure_2023_code": "Azure code", "azure_2023_conv": "Azure conv.", "burstgpt": "BurstGPT"}

# Minimum rendered text size (pt).  Checked by tests/test_manuscript_figures.py.
MIN_FONT_PT = 7.0


def apply() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.6, "patch.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.5,
        "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "0.7",
        "savefig.dpi": 300,
    })


def new_figure(height_in: float, width_in: float = TEXT_WIDTH_IN, **kw):
    return plt.subplots(figsize=(width_in, height_in), **kw)


def assert_inside(fig, name: str, tol_pt: float = 0.5) -> None:
    """Fail if any drawn artist (text, legend, tick labels) extends beyond the figure canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = fig.get_tightbbox(r)  # inches, includes every artist
    w, h = fig.get_size_inches()
    tol = tol_pt / 72.0
    if box.x0 < -tol or box.y0 < -tol or box.x1 > w + tol or box.y1 > h + tol:
        raise AssertionError(f"{name}: artists exceed the canvas ({box.x0:.3f},{box.y0:.3f})-({box.x1:.3f},{box.y1:.3f}) vs {w:.3f}x{h:.3f} in")


def fit(fig, pad_pt: float = 1.5) -> None:
    """tight_layout, then widen the subplot margins until every artist (including 2-line y tick labels) is inside the canvas."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # gridspec-width figures: the margin loop below does the real fitting
        fig.tight_layout()
    w, h = fig.get_size_inches()
    for _ in range(8):
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        pad = pad_pt / 72.0
        dl, dr, db, dt = max(0.0, pad - box.x0), max(0.0, box.x1 - (w - pad)), max(0.0, pad - box.y0), max(0.0, box.y1 - (h - pad))
        if max(dl, dr, db, dt) < 1e-4:
            return
        sp = fig.subplotpars
        fig.subplots_adjust(left=sp.left + dl / w, right=sp.right - dr / w, bottom=sp.bottom + db / h, top=sp.top - dt / h)


def save(fig, name: str) -> None:
    """Write <name>.pdf (vector) and <name>.png (300 dpi) at the drawn size, after checking nothing is clipped."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    assert_inside(fig, name)
    # no timestamps/producer strings, so regenerating a figure from the same data is byte-identical
    fig.savefig(FIG_DIR / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    print("wrote", FIG_DIR / f"{name}.pdf")
