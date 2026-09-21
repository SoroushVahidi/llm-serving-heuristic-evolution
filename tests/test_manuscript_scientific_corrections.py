"""Guards for the QUERY_11 scientific corrections to the Performance Evaluation manuscript.

They check wording constraints (no unsupported 'strong default', no general 'load does not matter', overlays and BurstGPT
mechanism disclosed, VBS claim weakened) and that the claim manifest covers every newly quoted fact.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "performance_evaluation"
TEX = PAPER / "main.tex"
MANIFEST = PAPER / "FINAL_CLAIM_MANIFEST.json"

pytestmark = pytest.mark.skipif(not TEX.exists(), reason="manuscript source not present")


@pytest.fixture(scope="module")
def flat():
    return re.sub(r"\s+", " ", TEX.read_text())


def _section(flat, a, b):
    return flat[flat.index(a):flat.index(b)]


def test_no_unsupported_strength_or_load_claims(flat):
    low = flat.lower()
    assert "strong default" not in low
    for bad in ("load does not matter", "load is unimportant", "arrival rate does not matter", "not load alone"):
        assert bad not in low, bad
    # the arrival result is described as a light-load negative result
    assert "negative result under light load" in low
    assert "does not show that arrival intensity cannot create opportunity" in low or "not evidence that load is irrelevant" in low


def test_reference_scheduler_is_described_and_conditional(flat):
    methods = _section(flat, r"\section{Methods}", r"\section{Where Disagreement Appears}")
    for s in ("KV-constrained online policy", "reserve of 0.82", "not tuned to these workloads or to latency", "not selected, tuned, or changed using fresh outcomes",
              "not claimed to be optimal or strong here", "earlier benchmark of 240 multi-mechanism scenarios",
              "imposed capacity setting", "physical binding", "reference's own reserve threshold"):
        assert s in methods, s
    assert "conditional on it" in flat and ("reference-conditional" in flat or "reference dependence" in flat.lower())


def test_overlays_and_capacity_are_disclosed_in_methods_and_limitations(flat):
    methods = _section(flat, r"\section{Methods}", r"\section{Where Disagreement Appears}")
    for s in ("loose service-level objective (SLO) deadline of arrival plus 1000~s", "uniform priority of 1.0", "a single class",
              "predicted output length equal to the true length", "a 1~ms step"):
        assert s in methods, s
    assert "hidden from policies" not in flat  # policies see the true length as their prediction
    lim = _section(flat, r"\subsection{Limitations and threats to validity}", r"\section{Conclusion}")
    assert r"\item[Workload overlays.]" in lim and "1000~s after arrival" in lim and r"\item[Reference scheduler.]" in lim


def test_burstgpt_mechanism_is_explained_not_merely_unselected(flat):
    fresh = _section(flat, r"\section{Causal Headroom on Untouched Windows}", r"\section{Concentration, Sensitivity")
    for s in ("All 360 untouched BurstGPT conditions were valid", "none had a binding capacity constraint", "never queued more than three requests"):
        assert s in fresh, s
    assert "contributed no selected regime" not in flat


def test_vbs_claim_is_weakened_to_an_illustration(flat):
    disc = _section(flat, r"\section{Discussion}", r"\section{Conclusion}")
    assert "overstates opportunity at every step" not in disc
    assert "without measuring a six-policy gap" in disc and "not evaluated in the faithful view" in disc


def test_relative_effect_and_harm_structure_reported(flat):
    assert r"\label{tab:secondary}" in flat and "Every alternative increases latency & 102 (14.2\\%)" in flat
    assert "12.8\\% of states exceed 10\\%" in flat or "92 states (12.8\\%) exceed 10\\%" in flat


def test_manifest_covers_new_facts():
    ids = {c["id"] for c in json.loads(MANIFEST.read_text())["claims"]}
    need = {"burstgpt.fresh_mechanism", "arrival.original_light_load", "arrival.fresh_light_load", "native.mean_queue_length", "overlay.faithful_view",
            "overlay.max_mean_latency", "reference.sbs_selection", "reference.all_five_differ", "reference.kv_without_physical_binding",
            "portfolio.functional_diversity", "secondary.state_and_action_structure", "secondary.relative_headroom", "vbs.whole_window_illustration",
            "fresh.support_transfer"}
    assert need <= ids, need - ids


def test_reference_numbers_recompute_from_artifacts():
    spec = importlib.util.spec_from_file_location("reference_policy_numbers", PAPER / "scripts" / "reference_policy_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    R = mod.compute()  # asserts internally: action-level vs corrected reference latency, preregistered per-regime relative headroom
    assert R["burst"]["valid"] == 360 and R["burst"]["disagreement_states"] == 0
    assert R["reference"]["all_five_differ"] == 512 and R["structure"]["all_harmful"] == 102
    assert round(100 * R["relative"]["gt10"] / 720, 1) == 12.8  # not the 15.4% obtained from the mislabeled frozen reference-latency column
