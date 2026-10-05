from __future__ import annotations

import pytest

from scripts import fresh_production_support_mapping_v1 as fresh

# These tests rebuild replay windows from the normalized public-trace corpus,
# which is derived from third-party traces and is not committed (see
# docs/DATA_RELEASE_POLICY.md). Skip them when the corpus has not been built
# locally. The paper's numbers stay verifiable from tracked artifacts via
# paper/performance_evaluation/scripts/build_claim_manifest.py --check.
_TRACE_CORPUS = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "public_trace_corpus_v1"
requires_trace_corpus = pytest.mark.skipif(
    not all((_TRACE_CORPUS / w / "records.parquet").is_file() for w in ("azure_2023_code", "azure_2023_conv", "burstgpt")),
    reason="needs the locally built public-trace corpus (data/public_trace_corpus_v1/*/records.parquet), which is not redistributed",
)



@requires_trace_corpus
def test_frozen_fresh_windows_are_disjoint_and_balanced():
    windows = fresh.frozen_windows()
    assert {source: len(rows) for source, rows in windows.items()} == {
        "azure_2023_code": 20,
        "azure_2023_conv": 20,
        "burstgpt": 20,
    }
    assert sum(len(row["source_record_id_range"]) == 2 for rows in windows.values() for row in rows) == 60
    assert fresh.verify_preflight()["fresh_request_overlap_with_phase_a_b_d"] == 0


@requires_trace_corpus
def test_fresh_condition_matrix_is_exactly_1080_and_one_axis_isolated():
    jobs = fresh.condition_records(fresh.fresh_records())
    assert len(jobs) == 1080
    for job in jobs:
        fresh.phase_b.validate_one_axis_isolation(job["condition"])


def test_regime_selection_is_latency_blind_and_deterministic():
    rows = [
        {
            "source_dataset": "azure_2023_code",
            "axis": "kv_capacity",
            "condition_id": "kv_8000000",
            "pressure_order": 0,
            "validity_classes": "VALID_UNCONSTRAINED",
            "canonical_disagreement_states": 0,
            "windows_with_disagreement": 0,
        },
        {
            "source_dataset": "azure_2023_code",
            "axis": "kv_capacity",
            "condition_id": "kv_16000",
            "pressure_order": 1,
            "validity_classes": "VALID_STRONGLY_CONSTRAINED",
            "canonical_disagreement_states": 3,
            "windows_with_disagreement": 2,
        },
    ]
    selected = fresh.select_regimes(rows)
    assert len(selected) == 1
    assert selected[0]["condition_id"] == "kv_16000"
    assert set(selected[0]["roles"]) == {"onset", "sustained", "strongest-valid"}
    assert all(row["latency_outcomes_used_for_regime_selection"] is False for row in selected)
