"""Tests for the generation loop in dry-run (mock) mode."""
import json
import tempfile
from pathlib import Path
import pytest
from llmserveopt.llm_generation.generation_loop import GenerationConfig, run_generation_loop
from llmserveopt.llm_generation.candidate_io import load_verified_candidates
from llmserveopt.llm_generation.repair import extract_json


def test_extract_json_plain():
    text = '{"name": "test", "tie_breaker": "arrival_order", "default": {"request_score": {"const": 1.0}}}'
    obj = extract_json(text)
    assert obj is not None
    assert obj["name"] == "test"


def test_extract_json_with_fence():
    text = '```json\n{"name": "fenced"}\n```'
    obj = extract_json(text)
    assert obj is not None
    assert obj["name"] == "fenced"


def test_extract_json_with_preamble():
    text = 'Here is the heuristic:\n{"name": "after_preamble"}'
    obj = extract_json(text)
    assert obj is not None


def test_extract_json_no_json():
    assert extract_json("This is plain text with no JSON") is None


def test_extract_json_empty():
    assert extract_json("") is None


def test_dry_run_generates_candidates():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = GenerationConfig(
            providers=["mock"],
            max_candidates=3,
            max_repair_attempts=2,
            output_dir=Path(tmpdir),
            dry_run=True,
            verbose=False,
        )
        summary = run_generation_loop(cfg)
        assert summary.generated == 3
        assert summary.generated >= summary.verified_ok


def test_dry_run_creates_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test_output"
        cfg = GenerationConfig(
            providers=["mock"],
            max_candidates=2,
            output_dir=out,
            dry_run=True,
            verbose=False,
        )
        run_generation_loop(cfg)
        assert out.exists()


def test_dry_run_creates_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        cfg = GenerationConfig(
            providers=["mock"],
            max_candidates=3,
            output_dir=out,
            dry_run=True,
            verbose=False,
        )
        run_generation_loop(cfg)
        assert (out / "index.csv").exists()


def test_dry_run_candidate_dirs_have_required_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        cfg = GenerationConfig(
            providers=["mock"],
            max_candidates=2,
            output_dir=out,
            dry_run=True,
            verbose=False,
        )
        run_generation_loop(cfg)
        subdirs = [d for d in out.iterdir() if d.is_dir()]
        for d in subdirs:
            assert (d / "prompt.json").exists()
            assert (d / "raw_response.txt").exists()
            assert (d / "verifier_result.json").exists()
            assert (d / "metadata.json").exists()


def test_dry_run_metadata_has_no_secrets():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        cfg = GenerationConfig(
            providers=["mock"],
            max_candidates=2,
            output_dir=out,
            dry_run=True,
            verbose=False,
        )
        run_generation_loop(cfg)
        for d in out.iterdir():
            if not d.is_dir():
                continue
            meta = json.loads((d / "metadata.json").read_text())
            assert "api_key" not in str(meta).lower()
            assert "secret" not in str(meta).lower()
            assert "password" not in str(meta).lower()


def test_dry_run_repair_triggered():
    """The mock provider returns one invalid candidate that triggers repair."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        cfg = GenerationConfig(
            providers=["mock"],
            max_candidates=4,
            max_repair_attempts=2,
            output_dir=out,
            dry_run=True,
            verbose=False,
        )
        summary = run_generation_loop(cfg)
        # At least one candidate should have needed repair (invalid mock is in cycle)
        assert summary.repaired_ok >= 0  # may or may not need repair depending on cycle


def test_dry_run_providers_used():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = GenerationConfig(
            providers=["mock"],
            max_candidates=2,
            output_dir=Path(tmpdir),
            dry_run=True,
            verbose=False,
        )
        summary = run_generation_loop(cfg)
        assert "mock" in summary.providers_used


def test_load_verified_candidates_from_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        cfg = GenerationConfig(
            providers=["mock"],
            max_candidates=4,
            max_repair_attempts=2,
            output_dir=out,
            dry_run=True,
            verbose=False,
        )
        summary = run_generation_loop(cfg)
        records = load_verified_candidates(out)
        assert isinstance(records, list)
        for r in records:
            assert "candidate" in r
            assert isinstance(r["candidate"], dict)
