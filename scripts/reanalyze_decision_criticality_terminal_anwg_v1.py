#!/usr/bin/env python3
"""Re-analyze terminal-ANWG v1 branches (post-run / if runner analyze was stale)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "tanwg_runner", REPO / "scripts/run_decision_criticality_terminal_anwg_v1.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
analyze = _mod.analyze

OUT = REPO / "experiments/decision_criticality_terminal_anwg_v1"


def main() -> int:
    branches_path = OUT / "branches.csv"
    if not branches_path.exists():
        jl = OUT / "branches.jsonl"
        if not jl.exists() or jl.stat().st_size == 0:
            print("no branches", file=sys.stderr)
            return 1
        branches = pd.DataFrame([json.loads(l) for l in jl.read_text().splitlines() if l.strip()])
        branches.to_csv(branches_path, index=False)
    else:
        branches = pd.read_csv(branches_path)

    summary = analyze(branches)
    prev = {}
    sp = OUT / "summary.json"
    if sp.exists():
        prev = json.loads(sp.read_text())
    for k in (
        "started_utc",
        "finished_utc",
        "elapsed_s",
        "n_scenarios_attempted",
        "n_scenarios_succeeded",
        "n_scenarios_failed",
        "failures",
        "config",
    ):
        if k in prev:
            summary[k] = prev[k]
    summary["n_branch_rows"] = int(len(branches))
    summary["reanalyzed"] = True
    sp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    abs_d = branches["abs_delta_anwg"].sort_values(ascending=False).to_numpy(float)
    total = float(abs_d.sum())
    cum = np.cumsum(abs_d)
    frac_states = (np.arange(1, len(abs_d) + 1)) / max(len(abs_d), 1)
    frac_mass = cum / total if total > 0 else np.zeros_like(cum)
    pd.DataFrame({"frac_states": frac_states, "frac_abs_anwg_mass": frac_mass}).to_csv(
        OUT / "concentration_curve.csv", index=False
    )
    keep = (
        "prevalence",
        "by_acquisition",
        "by_family",
        "concentration_abs_delta_all_states",
        "scenario_concentration_abs_mass",
        "disagreement_as_criticality_proxy",
        "h10_proxy_join",
        "ref_replay",
    )
    print(json.dumps({k: summary[k] for k in keep if k in summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
