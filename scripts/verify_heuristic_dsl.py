#!/usr/bin/env python3
"""
Verify one or more heuristic DSL JSON files.

Usage:
    python scripts/verify_heuristic_dsl.py configs/heuristics/examples/edf_like.json
    python scripts/verify_heuristic_dsl.py configs/heuristics/examples/*.json
    python scripts/verify_heuristic_dsl.py configs/heuristics/examples/bad.json --strict
"""
import argparse
import json
import sys
from pathlib import Path

# Make sure src/ is on path when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.heuristics.verifier import verify_heuristic


def verify_file(path: Path, strict: bool) -> bool:
    try:
        with open(path) as f:
            doc = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[INVALID JSON] {path}: {e}")
        return False

    result = verify_heuristic(doc)
    status = "PASS" if result.valid else "FAIL"
    print(f"[{status}] {path.name}")
    for code, msg in result.errors:
        print(f"  ERROR {code}: {msg}")
    for w in result.warnings:
        print(f"  WARN:  {w}")

    if strict and result.warnings:
        return False
    return result.valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify heuristic DSL JSON files")
    parser.add_argument("files", nargs="+", type=Path, help="DSL JSON files to verify")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    all_pass = True
    for path in args.files:
        ok = verify_file(path, args.strict)
        if not ok:
            all_pass = False

    if not all_pass:
        print("\nSome heuristics failed verification.")
        sys.exit(1)
    else:
        print("\nAll heuristics verified successfully.")


if __name__ == "__main__":
    main()
