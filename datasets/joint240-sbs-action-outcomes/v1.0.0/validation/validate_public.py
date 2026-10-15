#!/usr/bin/env python3
"""Validate a downloaded public Joint-240 dataset using only relative files."""
from __future__ import annotations

import csv
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "data/state_context.csv": 8886,
    "data/alternative_action_outcomes.csv": 21854,
    "metadata/scenario_metadata.csv": 236,
    "metadata/policy_metadata.csv": 6,
}
EXPECTED_SIGNS = (2573, 3624, 15657)
STATE_FEATURE_COUNT = 95
FROZEN_SCIENCE = {
    "data/state_context.csv": (
        "5152ec0a2e139ccad7b9edceab571d51ca5bd2a5b9f93c2275344d64b62bdb50"
    ),
    "data/alternative_action_outcomes.csv": (
        "af9e60fe749538cfbb1b0b7f1e999408ff3a5811c3f90ba5eb8b39c53b3a92fd"
    ),
    "metadata/scenario_metadata.csv": (
        "d3ccde80792a80f466c687cdef60eb94537181d7f99e7859fbec3f6f3b0c53e1"
    ),
    "metadata/policy_metadata.csv": (
        "9b46d5940a0d0ac898be114cc4be7a877172e01206f80488965baa7671300793"
    ),
}
STATE_HEADER_PREFIX = ["state_id", "scenario_id", "fold", "step", "sim_time"]
OUTCOME_HEADER_PREFIX = [
    "state_id",
    "scenario_id",
    "fold",
    "step",
    "sim_time",
    "canonical_action_id",
]
DIFF_METRICS = ("anwg", "soft", "wcg")
IMPROVEMENT_METRICS = ("wmt", "wnt")


def read(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError(f"non-finite value: {value!r}")
    return number


def main() -> int:
    tables = {rel: read(rel) for rel in EXPECTED}
    assert {rel: len(rows) for rel, rows in tables.items()} == EXPECTED

    states = tables["data/state_context.csv"]
    outcomes = tables["data/alternative_action_outcomes.csv"]
    scenarios = tables["metadata/scenario_metadata.csv"]
    policies = tables["metadata/policy_metadata.csv"]

    assert all(column in states[0] for column in STATE_HEADER_PREFIX)
    assert all(column in outcomes[0] for column in OUTCOME_HEADER_PREFIX)
    state_features = [c for c in states[0] if c.startswith("state__")]
    assert len(state_features) == STATE_FEATURE_COUNT
    assert len(policies) == 6

    state_ids = {row["state_id"] for row in states}
    state_keys = {(row["scenario_id"], row["step"]) for row in states}
    outcome_keys = [(row["state_id"], row["canonical_action_id"]) for row in outcomes]
    assert len(state_ids) == len(states) == 8886
    assert len(state_keys) == len(states)
    assert len(set(outcome_keys)) == len(outcomes)
    assert all(row["state_id"] in state_ids for row in outcomes)
    assert all(row["is_sbs_action"] in ("0", "false", "False") for row in outcomes)
    assert {row["scenario_id"] for row in states} == {
        row["scenario_id"] for row in scenarios
    }
    assert len({row["scenario_id"] for row in states}) == 236

    for row in states:
        finite(row["sim_time"])
        finite(row["step"])
        for feature in state_features:
            finite(row[feature])

    for row in outcomes:
        for metric in DIFF_METRICS:
            q = finite(row[f"q_sbs_{metric}"])
            q0 = finite(row[f"q0_sbs_{metric}"])
            a = finite(row[f"a_sbs_{metric}"])
            assert abs(a - (q - q0)) <= 1e-10
        for metric in IMPROVEMENT_METRICS:
            q = finite(row[f"q_sbs_{metric}"])
            q0 = finite(row[f"q0_sbs_{metric}"])
            a = finite(row[f"a_sbs_{metric}_improvement"])
            assert abs(a - (q0 - q)) <= 1e-10

    signs = (
        sum(finite(row["a_sbs_anwg"]) > 1e-12 for row in outcomes),
        sum(finite(row["a_sbs_anwg"]) < -1e-12 for row in outcomes),
        sum(abs(finite(row["a_sbs_anwg"])) <= 1e-12 for row in outcomes),
    )
    assert signs == EXPECTED_SIGNS

    for rel, expected in FROZEN_SCIENCE.items():
        assert digest(ROOT / rel) == expected, rel

    checksum_file = ROOT / "PUBLIC_RELEASE_CHECKSUMS.sha256"
    checksums: dict[str, str] = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            checksum, rel = line.split("  ", 1)
            assert not rel.startswith("/") and ".." not in Path(rel).parts
            assert rel != "PUBLIC_RELEASE_CHECKSUMS.sha256"
            checksums[rel] = checksum
    for rel, checksum in checksums.items():
        assert (ROOT / rel).is_file(), rel
        assert digest(ROOT / rel) == checksum, rel
    assert set(EXPECTED).issubset(checksums)
    assert "validation/validate_public.py" in checksums

    # Construct markers so this source file does not embed the literals it forbids.
    private_markers = (
        "/" + "home" + "/" + "soroush",
        "ghp" + "_",
        "hf" + "_",
        "Bearer" + " ",
    )
    for rel in checksums:
        if rel.endswith(".csv") or rel.endswith(".py"):
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for marker in private_markers:
            assert marker not in text, f"{rel} contains forbidden marker"

    result = {
        "status": "PASS",
        "rows": EXPECTED,
        "unique_states": len(state_ids),
        "unique_outcomes": len(set(outcome_keys)),
        "scenarios": 236,
        "state_features": len(state_features),
        "effect_signs": {
            "positive": signs[0],
            "negative": signs[1],
            "null": signs[2],
        },
        "checksums": len(checksums),
    }
    print(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print({"status": "FAIL", "error": str(exc) or repr(exc)}, file=sys.stderr)
        raise SystemExit(1)
