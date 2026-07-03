#!/usr/bin/env python3
"""
Persist a corrected-objective ("our method") selector artifact.

Context
-------
Phase 2B.15/2B.16 retrained several selectors under the corrected
`arrival_normalized_wg` objective (Phase 2B.14's fix for the completed-only
`weighted_goodput` denominator) but evaluated them entirely in-memory --
no `.joblib` model was ever written to disk. Every serialized selector
artifact that DOES exist on disk (results/phase2a2_selector_dataset/,
phase2a3_selector_eval/, phase2a4_2b4_final_eval/) predates the Phase
2B.14 correction and is trained/selected under the flawed objective.

This script closes that gap for exactly one selector: `regression_anwg`
(per-policy RandomForestRegressor, argmax at predict time), which
docs/result_claims.md (Phase 2B.16 safe claims) calls "the strongest
deployable selector under arrival-norm WG" (0.9856 on 174 fresh
held-out windows, +0.0170 vs always-SCORPIO, CI [0.0127, 0.0213]).

It does NOT re-run any simulation. It reuses:
  - results/phase2b13_selector_training_and_suspicion_audit/per_window.csv
    (the same 319-window training table Phase 2B.15 read) for fitting,
    with the same train-diversity-seed split Phase 2B.15 used.
  - results/phase2b16_fresh_corrected_objective_validation/fresh_per_window.csv
    (174 fresh windows, disjoint seeds/workloads never used for training)
    to verify the freshly-persisted model reproduces the previously
    published held-out number, rather than asserting it from memory.

Every feature at predict() time comes from `feat_*` columns only (see
llmserveopt/selector/features.py FEATURE_NAMES) -- no hindsight/oracle
fields are used to choose a policy.

Usage
-----
python scripts/persist_corrected_selector_artifact.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FEATURE_NAMES
from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector

# Import data-plumbing helpers (df_to_rows/relabel_rows/split_rows/_anwg) from
# the Phase 2B.15 script by file path (scripts/ has no __init__.py, so it
# isn't an importable package). These are plain functions, not pickled model
# classes, so no sys.modules registration is needed for them.
_B15_PATH = ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py"
_spec = importlib.util.spec_from_file_location("phase2b15_mod", _B15_PATH)
b15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b15)

TRAIN_CSV = ROOT / "results/phase2b13_selector_training_and_suspicion_audit/per_window.csv"
FRESH_CSV = ROOT / "results/phase2b16_fresh_corrected_objective_validation/fresh_per_window.csv"
OUT_DIR = ROOT / "results/corrected_selector_artifact_regression_anwg"

TRAIN_DIVERSITY_SEEDS = [6, 7, 8, 9, 10]
VAL_DIVERSITY_SEEDS = [11]

KV_UTILIZATION_NOTE = (
    "kv_utilization (feat index {idx}) has no honest client-side substitute "
    "in an external-admission harness talking to a real vLLM server over "
    "HTTP -- it requires scraping vLLM's /metrics endpoint, which is not "
    "implemented in scripts/run_vllm_external_baseline_comparison.py. The "
    "remaining 17/18 features ARE reconstructable from client-side request "
    "bookkeeping (queue state, prompt/output-length stats, slack, arrival "
    "timing, recent SLO violations)."
).format(idx=FEATURE_NAMES.index("kv_utilization"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return bool(out.strip())


def evaluate_on(rows: list[dict], selector, label_rows_source: str) -> dict:
    """Feature-only prediction; compare against always-SCORPIO / always-WSP / oracle."""
    preds = selector.predict(rows)
    sel_anwg = np.array([b15._anwg(r, p) for r, p in zip(rows, preds)])
    scorpio_anwg = np.array([b15._anwg(r, "scorpio_style_slo_guard") for r in rows])
    wsp_anwg = np.array([b15._anwg(r, "weighted_shortest_processing") for r in rows])
    oracle_anwg = np.array(
        [max(b15._anwg(r, p) for p in SELECTOR_CANDIDATES) for r in rows]
    )
    return {
        "source": label_rows_source,
        "n_windows": len(rows),
        "selector_mean_anwg": round(float(sel_anwg.mean()), 4),
        "always_scorpio_mean_anwg": round(float(scorpio_anwg.mean()), 4),
        "always_wsp_mean_anwg": round(float(wsp_anwg.mean()), 4),
        "oracle_mean_anwg": round(float(oracle_anwg.mean()), 4),
        "gap_vs_always_scorpio": round(float(sel_anwg.mean() - scorpio_anwg.mean()), 4),
        "gap_vs_always_wsp": round(float(sel_anwg.mean() - wsp_anwg.mean()), 4),
        "gap_vs_oracle": round(float(sel_anwg.mean() - oracle_anwg.mean()), 4),
        "chosen_policy_distribution": dict(pd.Series(preds).value_counts()),
    }


def main() -> None:
    if not TRAIN_CSV.exists():
        print(f"FATAL: training data not found at {TRAIN_CSV}", file=sys.stderr)
        sys.exit(1)
    if not FRESH_CSV.exists():
        print(f"FATAL: fresh validation data not found at {FRESH_CSV}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase A: load + relabel + split (identical to Phase 2B.15) ----
    train_df = pd.read_csv(TRAIN_CSV)
    all_rows = b15.relabel_rows(b15.df_to_rows(train_df))
    train_rows, val_rows, test_rows = b15.split_rows(
        all_rows, TRAIN_DIVERSITY_SEEDS, VAL_DIVERSITY_SEEDS
    )
    print(f"Loaded {len(all_rows)} rows from {TRAIN_CSV.name}: "
          f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    # ---- Phase B: fit regression_anwg (feature-only, no oracle) ----
    selector = PerPolicyRegressionAnwgSelector()
    selector.fit(train_rows)
    print(f"Fit regression_anwg on {len(train_rows)} windows, "
          f"{len(SELECTOR_CANDIDATES)} per-policy RF regressors")

    # ---- Phase C: verify against B13 heldout (33 windows) ----
    b13_heldout_eval = evaluate_on(
        test_rows, selector, "phase2b13_per_window.csv (heldout split, n=33)"
    )
    print("B13 heldout (33 windows):", b13_heldout_eval)

    # ---- Phase D: verify against B16 fresh validation (174 windows, disjoint) ----
    fresh_df = pd.read_csv(FRESH_CSV)
    fresh_rows = b15.df_to_rows(fresh_df)
    fresh_eval = evaluate_on(
        fresh_rows, selector, "phase2b16_fresh_per_window.csv (n=174, disjoint seeds/workloads)"
    )
    print("B16 fresh (174 windows):", fresh_eval)

    # Reproducibility check against the already-published Phase 2B.16 claim
    # (docs/result_claims.md: regression_anwg = 0.9856, +0.0170 vs SCORPIO).
    published_anwg = 0.9856
    published_gap = 0.0170
    reproduced = {
        "published_selector_mean_anwg": published_anwg,
        "published_gap_vs_scorpio": published_gap,
        "freshly_computed_selector_mean_anwg": fresh_eval["selector_mean_anwg"],
        "freshly_computed_gap_vs_scorpio": fresh_eval["gap_vs_always_scorpio"],
        "matches_published_claim": (
            abs(fresh_eval["selector_mean_anwg"] - published_anwg) < 0.001
            and abs(fresh_eval["gap_vs_always_scorpio"] - published_gap) < 0.001
        ),
    }
    print("Reproducibility check:", reproduced)

    # ---- Phase E: persist artifact + manifest ----
    artifact_path = OUT_DIR / "regression_anwg_selector.joblib"
    selector.save(str(artifact_path))
    print(f"Persisted selector to {artifact_path}")

    # Round-trip check: load it back and confirm predictions are identical.
    reloaded = PerPolicyRegressionAnwgSelector.load(str(artifact_path))
    reloaded_preds = reloaded.predict(fresh_rows)
    original_preds = selector.predict(fresh_rows)
    if reloaded_preds != original_preds:
        print("FATAL: reloaded selector predictions differ from in-memory selector", file=sys.stderr)
        sys.exit(3)
    print("Round-trip load check passed: reloaded selector predictions match exactly")

    manifest = {
        "artifact_type": "selector",
        "selector_name": "regression_anwg",
        "selector_class": "PerPolicyRegressionAnwgSelector",
        "description": (
            "Per-policy RandomForestRegressor (one regressor per candidate "
            "policy, predicting arrival_normalized_wg); predict() takes "
            "the argmax across regressors' predictions for each window. "
            "Feature-only at inference time -- no oracle/hindsight fields."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by_script": "scripts/persist_corrected_selector_artifact.py",
        "git_commit": git_commit(),
        "git_dirty_at_creation": git_dirty(),
        "objective_definition": {
            "name": "arrival_normalized_wg",
            "formula": "completion_fraction(policy, window) * completed_request_quality(policy, window)",
            "completed_request_quality_formula": (
                "sum(priority_i * slo_met_i for i in completed_requests) / "
                "sum(priority_i for i in completed_requests)"
            ),
            "denominator_note": (
                "arrival_normalized_wg's completion_fraction term folds ALL "
                "arrivals into the effective denominator: dropped/rejected/ "
                "unfinished requests count as zero via the completion_fraction "
                "multiplier, rather than being excluded outright as in the "
                "pre-correction completed_request_quality-only metric."
            ),
            "source_correction": "Phase 2B.14 metric audit (commit abf7989)",
        },
        "training": {
            "input_csv": str(TRAIN_CSV.relative_to(ROOT)),
            "input_csv_sha256": sha256_of(TRAIN_CSV),
            "input_csv_rows": len(all_rows),
            "train_diversity_seeds": TRAIN_DIVERSITY_SEEDS,
            "val_diversity_seeds": VAL_DIVERSITY_SEEDS,
            "n_train_windows": len(train_rows),
            "n_val_windows": len(val_rows),
            "n_test_windows_b13_heldout": len(test_rows),
            "model_params": {"n_estimators": 100, "max_depth": 8, "random_state": 42},
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        },
        "features": {
            "names": FEATURE_NAMES,
            "count": len(FEATURE_NAMES),
            "kv_utilization_caveat": KV_UTILIZATION_NOTE,
        },
        "baseline_set_candidates": SELECTOR_CANDIDATES,
        "held_out_performance": {
            "phase2b13_heldout_33_windows": b13_heldout_eval,
            "phase2b16_fresh_174_windows": fresh_eval,
            "reproducibility_check_vs_published_claim": reproduced,
        },
        "known_limitations": [
            "rf_anwg (a related but different selector) loses to always-SCORPIO "
            "on the fresh_targeted workload subset; regression_anwg does not "
            "(per docs/result_claims.md), but this was not independently "
            "re-verified per-workload-group in this script -- only the "
            "aggregate 174-window number was reproduced here.",
            "93.1% of the 174 fresh windows are near-ties (margin < 0.005); "
            "gains are concentrated in a minority of meaningful windows.",
            "kv_utilization is not client-observable in the real-vLLM "
            "external-admission harness (see features.kv_utilization_caveat) "
            "-- any live deployment must use a placeholder or /metrics scrape "
            "for that one feature.",
        ],
    }

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Wrote manifest to {manifest_path}")

    if not reproduced["matches_published_claim"]:
        print(
            "WARNING: freshly computed fresh-eval numbers do NOT match the "
            "previously published Phase 2B.16 claim within tolerance. Do not "
            "treat this artifact as 'the same' regression_anwg selector "
            "without investigating the discrepancy.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
