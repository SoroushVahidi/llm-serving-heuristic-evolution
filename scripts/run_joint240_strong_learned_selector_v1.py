#!/usr/bin/env python3
"""Run joint-240 strong learned selector v1 (preregistered).

Design: docs/design/JOINT240_STRONG_LEARNED_SELECTOR_V1.md
Does not overwrite parent same-distribution or joint-240 matrix artifacts.
Does not edit the manuscript.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.analysis import joint240_strong_learned_selector_v1 as sl  # noqa: E402
from llmserveopt.analysis.joint240_same_distribution_adaptive_v1 import (  # noqa: E402
    FEATURE_ALLOWLIST,
    P6,
)

OUT_DEFAULT = ROOT / "experiments" / "joint240_strong_learned_selector_v1"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--short"], text=True
            ).strip()
        )
    except Exception:
        return True


def run(out_dir: Path) -> int:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    log_path = out_dir / "logs" / "full_run.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    design_hash = sl.sha256_file(sl.DESIGN_DOC)
    frozen_hash_path = out_dir / "DESIGN_FROZEN.sha256"
    if frozen_hash_path.exists():
        expected = frozen_hash_path.read_text().split()[0]
        if design_hash != expected:
            raise RuntimeError(
                f"design hash drift: live={design_hash} frozen={expected}"
            )
    log(f"schema={sl.SCHEMA_VERSION} design_sha256={design_hash}")

    data = sl.build_feature_matrix()
    folds = sl.load_parent_folds()
    parent_oof = sl.load_parent_oof()

    # Align
    data = data.merge(folds[["scenario_id", "fold"]], on="scenario_id", how="inner")
    data = data.merge(
        parent_oof[
            [
                "scenario_id",
                "a_scen_policy",
                "a_scen_anwg",
                "a_live_anwg",
                "majority_policy",
                "majority_anwg",
                "vbs_policy",
                "vbs_anwg",
            ]
        ],
        on="scenario_id",
        how="inner",
        suffixes=("", "_parent"),
    )
    if len(data) != 240:
        raise RuntimeError(f"aligned data size {len(data)}")

    # Reproduce parent headline numbers
    sbs_policy = "kv_constrained_online"
    sbs_mean = float(data[sbs_policy].mean())
    vbs_mean = float(data["vbs_anwg"].mean())
    a_scen_mean = float(data["a_scen_anwg"].mean())
    a_live_mean = float(data["a_live_anwg"].mean())
    maj_mean = float(data["majority_anwg"].mean())
    log(
        f"repro SBS={sbs_mean:.12f} VBS={vbs_mean:.12f} headroom={vbs_mean-sbs_mean:.12f} "
        f"A_scen={a_scen_mean:.12f} A_live={a_live_mean:.12f} Majority={maj_mean:.12f}"
    )
    assert abs(sbs_mean - 0.31407166947293264) < 1e-12
    assert abs(vbs_mean - 0.33310550374603504) < 1e-12
    assert abs(a_scen_mean - 0.3059465519866274) < 1e-12
    assert abs(a_live_mean - 0.2839667616302265) < 1e-12
    assert abs(maj_mean - 0.2909341236540524) < 1e-12

    fold_ids = sorted(int(x) for x in data["fold"].unique())
    assert fold_ids == [0, 1, 2, 3, 4]

    config = {
        "schema_version": sl.SCHEMA_VERSION,
        "design_doc": str(sl.DESIGN_DOC.relative_to(ROOT)),
        "design_sha256": design_hash,
        "parent_experiment": str(sl.PARENT_EXP.relative_to(ROOT)),
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "p6": list(P6),
        "split_seed": 20260825,
        "model_seed": sl.MODEL_SEED,
        "bootstrap_seed": sl.BOOTSTRAP_SEED,
        "n_bootstrap": sl.N_BOOTSTRAP,
        "catastrophic_eps": 0.01,
        "primary_model": "HistGradientBoostingRegressor",
        "secondary_model": "ExtraTreesRegressor",
        "formulation": "pooled_utility_regression_onehot_policy",
        "hgb_grid_n": len(sl.HGB_GRID),
        "et_grid_n": len(sl.ET_GRID),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "hostname": platform.node(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "parent_oof_sha256": sl.sha256_file(sl.PARENT_EXP / "per_scenario_oof_results.csv"),
        "parent_folds_sha256": sl.sha256_file(sl.PARENT_EXP / "split_oof_folds.csv"),
        "utility_matrix_sha256": sl.sha256_file(
            ROOT / "experiments/joint_multimechanism_generalization_v1/utility_matrix_wide.csv"
        ),
    }
    _write_json(out_dir / "config.json", config)

    # Copy folds reference
    folds.to_csv(out_dir / "split_oof_folds_reference.csv", index=False)

    hp_logs: Dict[str, Any] = {"hgb": {}, "et": {}}
    pred_rows: List[Dict[str, Any]] = []

    for test_fold in fold_ids:
        train_folds = [f for f in fold_ids if f != test_fold]
        test_ids = data.loc[data["fold"] == test_fold, "scenario_id"].astype(str).tolist()
        train_ids = data.loc[data["fold"] != test_fold, "scenario_id"].astype(str).tolist()
        log(
            f"outer fold={test_fold} n_train={len(train_ids)} n_test={len(test_ids)}"
        )

        # Primary HGB
        hgb_params, hgb_score, hgb_log = sl.nested_select_hyperparams(
            data,
            train_folds,
            folds,
            model_family="hgb",
            grid=sl.HGB_GRID,
        )
        hp_logs["hgb"][str(test_fold)] = {
            "best_params": hgb_params,
            "best_inner_mean_anwg": hgb_score,
            "candidates": hgb_log,
        }
        hgb_model = sl.fit_selector(
            data, train_ids, model_family="hgb", params=hgb_params
        )
        hgb_pred_utils = sl.predict_policy_utilities(hgb_model, data, test_ids)
        hgb_selected = sl.select_policies_from_preds(hgb_pred_utils)

        # Secondary ET
        et_params, et_score, et_log = sl.nested_select_hyperparams(
            data,
            train_folds,
            folds,
            model_family="et",
            grid=sl.ET_GRID,
        )
        hp_logs["et"][str(test_fold)] = {
            "best_params": et_params,
            "best_inner_mean_anwg": et_score,
            "candidates": et_log,
        }
        et_model = sl.fit_selector(
            data, train_ids, model_family="et", params=et_params
        )
        et_pred_utils = sl.predict_policy_utilities(et_model, data, test_ids)
        et_selected = sl.select_policies_from_preds(et_pred_utils)

        lookup = data.set_index("scenario_id")
        for sid in test_ids:
            row = lookup.loc[sid]
            hgb_p = hgb_selected[sid]
            et_p = et_selected[sid]
            pred_rows.append(
                {
                    "scenario_id": sid,
                    "fold": int(test_fold),
                    "n_elevated_mechanisms": int(row["n_elevated_mechanisms"]),
                    "vbs_policy": str(row["vbs_policy"]),
                    "vbs_anwg": float(row["vbs_anwg"]),
                    "sbs_policy": sbs_policy,
                    "sbs_anwg": float(row[sbs_policy]),
                    "majority_policy": str(row["majority_policy"]),
                    "majority_anwg": float(row["majority_anwg"]),
                    "a_scen_policy": str(row["a_scen_policy"]),
                    "a_scen_anwg": float(row["a_scen_anwg"]),
                    "a_live_anwg": float(row["a_live_anwg"]),
                    "a_hgb_policy": hgb_p,
                    "a_hgb_anwg": float(row[hgb_p]),
                    "a_hgb_pred_util": float(hgb_pred_utils[sid][hgb_p]),
                    "a_et_policy": et_p,
                    "a_et_anwg": float(row[et_p]),
                    "a_et_pred_util": float(et_pred_utils[sid][et_p]),
                    "vbs_gain": float(row["vbs_anwg"] - row[sbs_policy]),
                    "correct_hgb": bool(hgb_p == row["vbs_policy"]),
                    "correct_et": bool(et_p == row["vbs_policy"]),
                    "correct_scen": bool(row["a_scen_policy"] == row["vbs_policy"]),
                    **{p: float(row[p]) for p in P6},
                }
            )
        log(
            f"fold={test_fold} hgb_inner={hgb_score:.6f} params={hgb_params} "
            f"et_inner={et_score:.6f} params={et_params}"
        )

    preds = pd.DataFrame(pred_rows).sort_values("scenario_id").reset_index(drop=True)
    if len(preds) != 240 or preds["scenario_id"].nunique() != 240:
        raise RuntimeError("prediction table integrity failure")
    if preds.groupby("scenario_id")["fold"].nunique().max() != 1:
        raise RuntimeError("duplicate fold assignment")
    # No train/test overlap by construction; also verify fold partition covers all
    assert set(preds["fold"].unique()) == {0, 1, 2, 3, 4}

    preds.to_csv(out_dir / "predictions.csv", index=False)
    _write_json(out_dir / "hyperparameter_selection.json", hp_logs)

    # Metrics
    sbs = preds["sbs_anwg"].to_numpy(float)
    vbs = preds["vbs_anwg"].to_numpy(float)
    summaries = {
        "SBS": sl.method_summary(sbs, sbs, vbs),
        "VBS": sl.method_summary(vbs, sbs, vbs),
        "Majority": sl.method_summary(
            preds["majority_anwg"].to_numpy(float), sbs, vbs,
            selected=preds["majority_policy"], vbs_policy=preds["vbs_policy"],
        ),
        "A_scen": sl.method_summary(
            preds["a_scen_anwg"].to_numpy(float), sbs, vbs,
            selected=preds["a_scen_policy"], vbs_policy=preds["vbs_policy"],
        ),
        "A_live": sl.method_summary(preds["a_live_anwg"].to_numpy(float), sbs, vbs),
        "A_hgb": sl.method_summary(
            preds["a_hgb_anwg"].to_numpy(float), sbs, vbs,
            selected=preds["a_hgb_policy"], vbs_policy=preds["vbs_policy"],
        ),
        "A_et": sl.method_summary(
            preds["a_et_anwg"].to_numpy(float), sbs, vbs,
            selected=preds["a_et_policy"], vbs_policy=preds["vbs_policy"],
        ),
    }

    # Pairwise
    pairwise = {
        "A_hgb_minus_A_scen": sl.paired_bootstrap_diff(
            preds["a_hgb_anwg"].to_numpy(float),
            preds["a_scen_anwg"].to_numpy(float),
            seed=sl.BOOTSTRAP_SEED + 11,
        ),
        "A_hgb_minus_A_live": sl.paired_bootstrap_diff(
            preds["a_hgb_anwg"].to_numpy(float),
            preds["a_live_anwg"].to_numpy(float),
            seed=sl.BOOTSTRAP_SEED + 12,
        ),
        "A_et_minus_A_scen": sl.paired_bootstrap_diff(
            preds["a_et_anwg"].to_numpy(float),
            preds["a_scen_anwg"].to_numpy(float),
            seed=sl.BOOTSTRAP_SEED + 13,
        ),
        "A_et_minus_A_live": sl.paired_bootstrap_diff(
            preds["a_et_anwg"].to_numpy(float),
            preds["a_live_anwg"].to_numpy(float),
            seed=sl.BOOTSTRAP_SEED + 14,
        ),
        "A_hgb_minus_A_et": sl.paired_bootstrap_diff(
            preds["a_hgb_anwg"].to_numpy(float),
            preds["a_et_anwg"].to_numpy(float),
            seed=sl.BOOTSTRAP_SEED + 15,
        ),
    }
    _write_json(out_dir / "bootstrap.json", {"pairwise": pairwise, "methods": {
        k: {
            "bootstrap_gain_vs_sbs": v.get("bootstrap_gain_vs_sbs"),
            "bootstrap_gap_closure": v.get("bootstrap_gap_closure"),
        }
        for k, v in summaries.items()
        if k in ("A_hgb", "A_et", "A_scen", "A_live", "Majority")
    }})

    # Fold-level
    fold_rows = []
    for f, g in preds.groupby("fold"):
        fold_rows.append(
            {
                "fold": int(f),
                "n": int(len(g)),
                "SBS": float(g["sbs_anwg"].mean()),
                "VBS": float(g["vbs_anwg"].mean()),
                "Majority": float(g["majority_anwg"].mean()),
                "A_scen": float(g["a_scen_anwg"].mean()),
                "A_live": float(g["a_live_anwg"].mean()),
                "A_hgb": float(g["a_hgb_anwg"].mean()),
                "A_et": float(g["a_et_anwg"].mean()),
            }
        )
    per_fold = pd.DataFrame(fold_rows).sort_values("fold")
    per_fold.to_csv(out_dir / "per_fold_metrics.csv", index=False)

    # Failure analysis
    cm_hgb = sl.confusion_matrix(preds["a_hgb_policy"], preds["vbs_policy"])
    cm_et = sl.confusion_matrix(preds["a_et_policy"], preds["vbs_policy"])
    cm_hgb.to_csv(out_dir / "confusion_hgb_vs_vbs.csv")
    cm_et.to_csv(out_dir / "confusion_et_vs_vbs.csv")

    preds["regret_hgb"] = preds["vbs_anwg"] - preds["a_hgb_anwg"]
    preds["regret_scen"] = preds["vbs_anwg"] - preds["a_scen_anwg"]
    preds["catastrophic_hgb"] = preds["a_hgb_anwg"] < (preds["sbs_anwg"] - 0.01)

    # VBS gain tertiles
    try:
        preds["vbs_gain_tertile"] = pd.qcut(
            preds["vbs_gain"], 3, labels=["low", "mid", "high"], duplicates="drop"
        )
    except ValueError:
        preds["vbs_gain_tertile"] = "all"

    cond_rows = []
    for key, col in [
        ("n_elevated", "n_elevated_mechanisms"),
        ("vbs_winner", "vbs_policy"),
        ("vbs_gain_tertile", "vbs_gain_tertile"),
    ]:
        for val, g in preds.groupby(col):
            cond_rows.append(
                {
                    "conditioning": key,
                    "level": str(val),
                    "n": int(len(g)),
                    "A_hgb": float(g["a_hgb_anwg"].mean()),
                    "A_scen": float(g["a_scen_anwg"].mean()),
                    "SBS": float(g["sbs_anwg"].mean()),
                    "VBS": float(g["vbs_anwg"].mean()),
                    "gain_hgb_vs_sbs": float(g["a_hgb_anwg"].mean() - g["sbs_anwg"].mean()),
                    "accuracy_hgb": float(g["correct_hgb"].mean()),
                    "accuracy_scen": float(g["correct_scen"].mean()),
                    "catastrophic_hgb": int(g["catastrophic_hgb"].sum()),
                }
            )
    pd.DataFrame(cond_rows).to_csv(out_dir / "failure_analysis_conditioned.csv", index=False)
    preds.loc[preds["catastrophic_hgb"], [
        "scenario_id", "fold", "vbs_policy", "a_hgb_policy", "a_scen_policy",
        "a_hgb_anwg", "a_scen_anwg", "sbs_anwg", "vbs_anwg", "vbs_gain",
        "n_elevated_mechanisms",
    ]].to_csv(out_dir / "catastrophic_hgb_scenarios.csv", index=False)

    regret_summary = {
        "hgb_regret_mean": float(preds["regret_hgb"].mean()),
        "hgb_regret_median": float(preds["regret_hgb"].median()),
        "hgb_regret_p90": float(preds["regret_hgb"].quantile(0.9)),
        "scen_regret_mean": float(preds["regret_scen"].mean()),
        "scen_regret_median": float(preds["regret_scen"].median()),
        "frac_hgb_better_anwg_than_scen": float(
            (preds["a_hgb_anwg"] > preds["a_scen_anwg"]).mean()
        ),
        "frac_hgb_worse_anwg_than_scen": float(
            (preds["a_hgb_anwg"] < preds["a_scen_anwg"]).mean()
        ),
    }
    _write_json(out_dir / "failure_analysis_summary.json", regret_summary)

    recovery_hgb = sl.classify_recovery(summaries["A_hgb"])
    recovery_et = sl.classify_recovery(summaries["A_et"])
    clear_vs_scen = bool(pairwise["A_hgb_minus_A_scen"]["ci95_low"] > 0)
    clear_et_vs_scen = bool(pairwise["A_et_minus_A_scen"]["ci95_low"] > 0)

    # Manuscript decision heuristic
    if recovery_hgb == "STRONG_RECOVERY" or (
        recovery_hgb == "PARTIAL_RECOVERY" and clear_vs_scen
    ):
        manuscript_decision = "PROMOTE"
    elif recovery_hgb in ("NO_RECOVERY", "PARTIAL_RECOVERY"):
        manuscript_decision = "REPORT_NEGATIVE"
    else:
        manuscript_decision = "INVALID"

    summary = {
        "schema_version": sl.SCHEMA_VERSION,
        "design_sha256": design_hash,
        "n_scenarios": 240,
        "methods": summaries,
        "pairwise": pairwise,
        "interpretation": {
            "A_hgb_vs_SBS": recovery_hgb,
            "A_et_vs_SBS": recovery_et,
            "CLEAR_IMPROVEMENT_OVER_A_SCEN_hgb": clear_vs_scen,
            "CLEAR_IMPROVEMENT_OVER_A_SCEN_et": clear_et_vs_scen,
        },
        "manuscript_decision": manuscript_decision,
        "ltr_relationship_note": (
            "This selector learns scenario-level policy utility for portfolio "
            "selection. It does not reproduce Fu et al. Learning-to-Rank, which "
            "ranks requests by predicted relative output length for SJF-like "
            "scheduling."
        ),
        "integrity": {
            "n_predictions": int(len(preds)),
            "unique_scenarios": int(preds["scenario_id"].nunique()),
            "folds": {str(k): int(v) for k, v in preds["fold"].value_counts().sort_index().items()},
            "parent_means_unchanged": {
                "SBS": sbs_mean,
                "VBS": vbs_mean,
                "A_scen": a_scen_mean,
                "A_live": a_live_mean,
                "Majority": maj_mean,
            },
        },
        "wall_s": float(time.perf_counter() - t0),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "provenance.json", config)

    log(
        f"DONE A_hgb={summaries['A_hgb']['R_anwg']:.6f} "
        f"gain={summaries['A_hgb']['realized_gain']:.6f} "
        f"label={recovery_hgb} vs_scen_clear={clear_vs_scen} "
        f"decision={manuscript_decision} wall_s={summary['wall_s']:.1f}"
    )
    (out_dir / "DONE").write_text("ok\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    return run(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
