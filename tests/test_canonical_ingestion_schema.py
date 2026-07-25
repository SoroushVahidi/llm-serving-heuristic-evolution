"""Canonical ingestion schema and observable-field leakage guards."""
import pytest

from llmserveopt.core.types import ObservableRequest, Request
from llmserveopt.workloads.canonical_schema import (
    CanonicalIngestRecord,
    FieldProvenance,
    OBSERVABLE_REQUEST_FIELDS,
    assert_no_actual_output_leakage,
    default_provenance,
    observable_request_fields,
    replay_label_for_time_scale,
    scale_interarrivals,
    validate_canonical_record,
    validate_canonical_records,
)


def _valid_record(**overrides):
    base = dict(
        request_id=0,
        arrival_time=0.0,
        prompt_tokens=10,
        actual_output_tokens=5,
        predicted_output_tokens=6,
        slo_deadline=2.0,
        priority=1.0,
        class_id="interactive",
        field_provenance=default_provenance(),
    )
    base.update(overrides)
    return CanonicalIngestRecord(**base)


def test_observable_fields_exclude_actual_output():
    assert "actual_output_tokens" not in OBSERVABLE_REQUEST_FIELDS
    assert "actual_output_tokens" not in observable_request_fields()
    assert_no_actual_output_leakage()
    obs = ObservableRequest.from_request(
        Request(
            request_id=0,
            arrival_time=0.0,
            prompt_tokens=8,
            predicted_output_tokens=4,
            actual_output_tokens=9,
            slo_deadline=1.0,
            priority=1.0,
            class_id="interactive",
        )
    )
    assert not hasattr(obs, "actual_output_tokens") or "actual_output_tokens" not in obs.__dict__
    assert "actual_output_tokens" not in ObservableRequest.__dataclass_fields__


def test_validate_rejects_missing_provenance():
    rec = _valid_record(field_provenance={})
    with pytest.raises(ValueError, match="missing provenance"):
        validate_canonical_record(rec)


def test_validate_rejects_unavailable_slo():
    prov = default_provenance(slo_deadline=FieldProvenance.UNAVAILABLE.value)
    rec = _valid_record(field_provenance=prov)
    with pytest.raises(ValueError, match="slo_deadline"):
        validate_canonical_record(rec)


def test_validate_rejects_non_chronological():
    r0 = _valid_record(request_id=0, arrival_time=0.0)
    r1 = _valid_record(request_id=1, arrival_time=1.0)
    r2 = _valid_record(request_id=2, arrival_time=0.5)
    with pytest.raises(ValueError, match="chronologically"):
        validate_canonical_records([r0, r1, r2])


def test_scale_interarrivals_and_labels():
    scaled = scale_interarrivals([10.0, 12.0, 15.0], time_scale=2.0)
    assert scaled[0] == 0.0
    assert abs(scaled[1] - 4.0) < 1e-9
    assert abs(scaled[2] - 10.0) < 1e-9
    assert replay_label_for_time_scale(1.0) == "natural_trace_replay"
    assert replay_label_for_time_scale(0.05) == "trace-derived, time-scaled"


def test_metadata_discloses_synthesized_fields():
    rec = _valid_record()
    md = rec.to_metadata()
    assert "predicted_output_tokens" in md["synthetic_fields"]
    assert "slo_deadline" in md["synthetic_fields"]
    assert "priority" in md["synthetic_fields"]
    assert "prompt_tokens" in md["observed_fields"]
    assert "actual_output_tokens" in md["observed_fields"]


def test_malformed_negative_tokens():
    with pytest.raises(ValueError):
        validate_canonical_record(_valid_record(prompt_tokens=0))
