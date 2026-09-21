"""The final claim manifest is verified against the canonical artifacts and the manuscript text.

FINAL_CLAIM_MANIFEST.json is provenance metadata (value, source artifact, source field, manuscript location).  The
values are recomputed from the frozen artifacts by scripts/build_claim_manifest.py, and every claim's manuscript
text is derived from the recomputed value, so an edited number or a changed artifact fails here.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "performance_evaluation"
SCRIPT = PAPER / "scripts" / "build_claim_manifest.py"
MANIFEST = PAPER / "FINAL_CLAIM_MANIFEST.json"

pytestmark = pytest.mark.skipif(not (PAPER / "main.tex").exists(), reason="manuscript source not present")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("build_claim_manifest", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def committed():
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def fresh(mod):
    claims = mod.build_claims()
    problems = mod.evaluate(claims, mod.TEX.read_text())
    return claims, problems


REQUIRED_IDS = {
    # native action-null counts
    "native.azure_2023_code.action_null", "native.azure_2023_conv.action_null", "native.burstgpt.action_null", "native.total_decision_states",
    # pressure campaign
    "pressure.conditions", "pressure.condition_classes", "pressure.disagreement_states", "pressure.arrival_scaling_action_null", "pressure.transition_table",
    # fresh causal
    "fresh.states", "fresh.beneficial_preregistered", "fresh.beneficial_guarded", "fresh.mean_headroom", "fresh.mean_headroom_ci", "fresh.regime_table",
    # robustness and practical thresholds
    "robust.median", "robust.equal_window", "robust.equal_regime", "robust.equal_workload", "robust.w11_headroom_share", "robust.regime_headroom_share",
    "robust.thresholds_used", "robust.threshold_0.1ms", "robust.threshold_0.25ms", "robust.threshold_0.5ms", "robust.threshold_1ms", "robust.threshold_2ms", "robust.threshold_5ms",
    # vLLM
    "vllm.setup", "vllm.run_counts", "vllm.trace_counts", "vllm.queue_lengths", "vllm.latency_differences",
}


def test_manifest_schema_and_required_claims(committed):
    assert committed["schema"] == "peva_final_claim_manifest_v1"
    ids = [c["id"] for c in committed["claims"]]
    assert len(ids) == len(set(ids)) == committed["n_claims"]
    assert REQUIRED_IDS <= set(ids), sorted(REQUIRED_IDS - set(ids))
    for c in committed["claims"]:
        for field in ("id", "claim", "value", "source_artifact", "source_field", "manuscript_text", "manuscript_location"):
            assert c.get(field) not in (None, "", []), f"{c['id']}: empty {field}"
        for loc in c["manuscript_location"]:
            assert {"section", "line", "text"} <= set(loc)
    assert "no independent scientific data" in committed["purpose"]


def test_every_manuscript_text_is_found_in_main_tex(fresh):
    _, problems = fresh
    assert problems == []


def test_recomputed_values_equal_the_committed_manifest(fresh, committed):
    claims, _ = fresh
    old = {c["id"]: c for c in committed["claims"]}
    assert {c["id"] for c in claims} == set(old)
    for c in claims:
        assert json.loads(json.dumps(c["value"])) == json.loads(json.dumps(old[c["id"]]["value"])), c["id"]
        assert c["manuscript_text"] == old[c["id"]]["manuscript_text"], c["id"]


def test_source_artifacts_exist(committed):
    import re
    for c in committed["claims"]:
        paths = re.findall(r"(?:experiments|scripts)/[A-Za-z0-9_./-]+", c["source_artifact"])
        assert paths, f"{c['id']}: no source path recognised in {c['source_artifact']!r}"
        for path in paths:
            assert (ROOT / path).exists() or (PAPER / path).exists(), f"{c['id']}: missing source {path}"


def test_headline_numbers_of_the_freeze(committed):
    v = {c["id"]: c["value"] for c in committed["claims"]}
    assert (v["native.azure_2023_code.action_null"]["sbs_decision_states"], v["native.azure_2023_conv.action_null"]["sbs_decision_states"],
            v["native.burstgpt.action_null"]["sbs_decision_states"]) == (99992, 461985, 440461)
    assert v["pressure.conditions"] == 1080 and v["pressure.disagreement_states"] == 11328
    assert v["pressure.arrival_scaling_action_null"]["max_multiplier"] == 8.0 and v["pressure.arrival_scaling_action_null"]["disagreement_states"] == 0
    assert v["fresh.states"] == 720 and v["fresh.beneficial_preregistered"]["beneficial"] == 590 and v["fresh.beneficial_guarded"]["guarded"] == 589
    assert round(v["fresh.mean_headroom"]["ms"], 4) == 1.9958
    assert [round(x, 4) for x in v["fresh.mean_headroom_ci"]["ms"]] == [0.1909, 3.5462]
    assert round(v["robust.median"], 3) == 0.197
    assert round(v["robust.equal_window"]["mean_ms"], 3) == 0.327 and round(v["robust.equal_regime"]["mean_ms"], 3) == 0.759 and round(v["robust.equal_workload"]["mean_ms"], 3) == 1.360
    assert round(100 * v["robust.w11_headroom_share"]["headroom_share"], 1) == 88.4 and round(100 * v["robust.regime_headroom_share"]["headroom_share"], 1) == 96.8
    assert v["robust.thresholds_used"] == [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


def test_a_drifting_manuscript_number_is_detected(mod):
    tex = mod.TEX.read_text()
    assert "0.327 & [0.114, 0.644]" in tex
    problems = mod.evaluate(mod.build_claims(), tex.replace("0.327 & [0.114, 0.644]", "0.328 & [0.114, 0.644]", 1))
    assert any("robust.equal_window" in p for p in problems)
