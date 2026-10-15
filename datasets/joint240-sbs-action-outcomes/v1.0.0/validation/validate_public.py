#!/usr/bin/env python3
"""Validate a downloaded public Joint-240 dataset using only relative files."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "data/state_context.csv": 8886,
    "data/alternative_action_outcomes.csv": 21854,
    "metadata/scenario_metadata.csv": 236,
    "metadata/policy_metadata.csv": 6,
}
STATE_HEADER = ["state_id", "scenario_id", "fold", "step", "sim_time"]
OUTCOME_HEADER = ["state_id", "scenario_id", "fold", "step", "sim_time", "canonical_action_id"]


def read(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    tables = {rel: read(rel) for rel in EXPECTED}
    assert {rel: len(rows) for rel, rows in tables.items()} == EXPECTED
    states = tables["data/state_context.csv"]
    outcomes = tables["data/alternative_action_outcomes.csv"]
    assert all(column in states[0] for column in STATE_HEADER)
    assert all(column in outcomes[0] for column in OUTCOME_HEADER)
    state_ids = {row["state_id"] for row in states}
    state_keys = {(row["scenario_id"], row["step"]) for row in states}
    outcome_keys = [(row["state_id"], row["canonical_action_id"]) for row in outcomes]
    assert len(state_ids) == len(states) == 8886
    assert len(state_keys) == len(states)
    assert len(set(outcome_keys)) == len(outcomes)
    assert all(row["state_id"] in state_ids for row in outcomes)
    assert all(row["is_sbs_action"] in ("0", "false", "False") for row in outcomes)
    assert {row["scenario_id"] for row in states} == {
        row["scenario_id"] for row in tables["metadata/scenario_metadata.csv"]
    }
    for row in outcomes:
        for metric in ("q_sbs_anwg", "q0_sbs_anwg", "a_sbs_anwg"):
            float(row[metric])
        assert abs(float(row["a_sbs_anwg"]) -
                   (float(row["q_sbs_anwg"]) - float(row["q0_sbs_anwg"]))) <= 1e-10
    signs = (
        sum(float(row["a_sbs_anwg"]) > 1e-12 for row in outcomes),
        sum(float(row["a_sbs_anwg"]) < -1e-12 for row in outcomes),
        sum(abs(float(row["a_sbs_anwg"])) <= 1e-12 for row in outcomes),
    )
    assert signs == (2573, 3624, 15657)
    checksum_file = ROOT / "PUBLIC_RELEASE_CHECKSUMS.sha256"
    checksums = {}
    for line in checksum_file.read_text().splitlines():
        if line.strip():
            checksum, rel = line.split("  ", 1)
            assert not rel.startswith("/") and rel != "PUBLIC_RELEASE_CHECKSUMS.sha256"
            checksums[rel] = checksum
    for rel in checksums:
        assert (ROOT / rel).is_file(), rel
        assert digest(ROOT / rel) == checksums[rel], rel
    assert set(EXPECTED).issubset(checksums)
    print({
        "status": "PASS",
        "rows": EXPECTED,
        "unique_states": len(state_ids),
        "unique_outcomes": len(set(outcome_keys)),
        "scenarios": len({row["scenario_id"] for row in states}),
        "effect_signs": {"positive": signs[0], "negative": signs[1], "null": signs[2]},
        "checksums": len(checksums),
    })


if __name__ == "__main__":
    main()
