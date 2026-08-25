#!/usr/bin/env python3
"""Print the deterministic read-only Public Trace Replay v1 analysis summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmserveopt.analysis.public_trace_replay_v1_analysis import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_REPLAY_DIR,
    DEFAULT_UNIFIED_MATRIX,
    public_trace_science_summary,
    to_jsonable,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--unified-matrix", default=str(DEFAULT_UNIFIED_MATRIX))
    parser.add_argument("--skip-trajectories", action="store_true")
    parser.add_argument("--skip-action-traces", action="store_true")
    args = parser.parse_args()

    summary = public_trace_science_summary(
        Path(args.replay_dir),
        Path(args.corpus_dir),
        Path(args.unified_matrix),
        include_trajectories=not args.skip_trajectories,
        include_action_traces=not args.skip_action_traces,
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
