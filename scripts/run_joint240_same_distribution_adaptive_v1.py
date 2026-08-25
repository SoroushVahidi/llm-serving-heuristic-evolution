#!/usr/bin/env python3
"""Run joint-240 same-distribution adaptive exploitability v1.

Preregistration: docs/design/JOINT240_SAME_DISTRIBUTION_ADAPTIVE_EXPLOITABILITY_V1.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from llmserveopt.analysis.joint240_same_distribution_adaptive_v1 import (
    BOOTSTRAP_SEED,
    CATASTROPHIC_EPS,
    FEATURE_ALLOWLIST,
    N_BOOTSTRAP,
    N_FOLDS,
    P6,
    PROBE_POLICY,
    SCHEMA_VERSION,
    SPLIT_SEED,
    collect_probe_telemetry,
    fit_live_stage1,
    freeze_oof_folds,
    freeze_reference_split,
    generator_feature_table,
    load_utility_matrix,
    majority_policy,
    rebuild_all_scenarios,
    run_live_router_anwg,
    select_scen_model_on_val,
    sha256_file,
    summarize_oof,
    verify_matrix_vs_live,
    predict_policies,
    fit_scen_selector,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "experiments" / "joint240_same_distribution_adaptive_exploitability_v1"
JOINT_DIR = ROOT / "experiments" / "joint_multimechanism_generalization_v1"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _inner_train_val(
    train_ids: List[str], fold: int
) -> tuple[List[str], List[str]]:
    rng = np.random.default_rng(SPLIT_SEED + 100 + fold)
    ids = list(train_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(0.20 * len(ids))))
    if len(ids) - n_val < 1:
        n_val = max(0, len(ids) - 1)
    val_ids = ids[:n_val]
    tr_ids = ids[n_val:]
    return tr_ids, val_ids


def run(out_dir: Path, *, smoke: bool, smoke_n: int, skip_live: bool) -> int:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    log_path = out_dir / "logs" / ("smoke.log" if smoke else "full_run.log")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(f"schema={SCHEMA_VERSION} smoke={smoke} skip_live={skip_live}")
    matrix = load_utility_matrix()
    scenarios = rebuild_all_scenarios()
    scen_by_id = {s.scenario_id: s for s in scenarios}
    feats = generator_feature_table(scenarios)
    data = matrix.merge(feats, on="scenario_id", how="inner")
    if len(data) != 240:
        raise RuntimeError(f"feature join size {len(data)}")

    if smoke:
        keep = data.sort_values("scenario_id").head(smoke_n)["scenario_id"].tolist()
        data = data[data["scenario_id"].isin(keep)].reset_index(drop=True)
        scenarios = [scen_by_id[i] for i in keep]
        scen_by_id = {s.scenario_id: s for s in scenarios}
        log(f"smoke subset n={len(data)}")

    # Integrity: reproduce P6 utilities on a tiny smoke-vs-live check
    global_means = {p: float(data[p].mean()) for p in P6}
    sbs_policy_global = max(global_means, key=global_means.get)
    sbs_mean = float(global_means[sbs_policy_global])
    vbs_mean = float(data["vbs_anwg"].mean())
    log(
        f"subset SBS_mean={sbs_mean:.6f} ({sbs_policy_global}) "
        f"VBS_mean={vbs_mean:.6f} headroom={vbs_mean - sbs_mean:.6f}"
    )

    folds = freeze_oof_folds(
        data["scenario_id"].tolist(),
        data["n_elevated_mechanisms"].astype(int).tolist(),
        n_folds=min(N_FOLDS, max(2, len(data) // 2)) if smoke else N_FOLDS,
        seed=SPLIT_SEED,
    )
    ref_split = freeze_reference_split(
        data["scenario_id"].tolist(),
        data["n_elevated_mechanisms"].astype(int).tolist(),
        seed=SPLIT_SEED,
    )
    folds.to_csv(out_dir / "split_oof_folds.csv", index=False)
    ref_split.to_csv(out_dir / "split_reference_tvt.csv", index=False)

    allow = {
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist_notes": [
            "policy ANWG columns",
            "VBS/SBS labels at inference",
            "actual_output_tokens",
            "post-policy metrics",
        ],
        "online_features": [
            "contention_score_v2",
            "priority_skew",
            "kv_pressure",
            "queue_length",
        ],
        "p6": list(P6),
        "split_seed": SPLIT_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "catastrophic_eps": CATASTROPHIC_EPS,
        "probe_policy": PROBE_POLICY,
    }
    _write_json(out_dir / "config" / "frozen_protocol.json", allow)
    _write_json(
        out_dir / "feature_allowlist.json",
        {"allowlist": list(FEATURE_ALLOWLIST)},
    )
    _write_json(
        out_dir / "feature_denylist.json",
        {
            "denylist": allow["feature_denylist_notes"],
            "online_denylist": [
                "actual_output_tokens",
                "scenario labels",
                "utilities",
                "VBS winners at inference",
            ],
        },
    )

    # Smoke integrity vs live
    check_ids = data["scenario_id"].head(1 if smoke else 2).tolist()
    integrity = verify_matrix_vs_live(scen_by_id, data, check_ids)
    _write_json(out_dir / "matrix_live_integrity.json", integrity)
    if not integrity["ok"]:
        raise RuntimeError(f"matrix/live mismatch: {integrity['max_abs_err']}")
    log(f"matrix_live_integrity ok max_abs_err={integrity['max_abs_err']:.3e}")

    X_all = data[list(FEATURE_ALLOWLIST)].to_numpy(dtype=float)
    y_all = data["vbs_policy"].astype(str).to_numpy()
    id_all = data["scenario_id"].astype(str).to_numpy()
    id_to_row = {sid: i for i, sid in enumerate(id_all)}

    oof_rows: List[Dict[str, Any]] = []
    model_meta: List[Dict[str, Any]] = []

    n_folds = int(folds["fold"].max()) + 1
    for fold in range(n_folds):
        test_ids = folds.loc[folds["fold"] == fold, "scenario_id"].tolist()
        train_pool = folds.loc[folds["fold"] != fold, "scenario_id"].tolist()
        tr_ids, val_ids = _inner_train_val(train_pool, fold)
        tr_idx = [id_to_row[i] for i in tr_ids]
        val_idx = [id_to_row[i] for i in val_ids]
        te_idx = [id_to_row[i] for i in test_ids]

        name, val_score, model = select_scen_model_on_val(
            X_all[tr_idx],
            y_all[tr_idx],
            X_all[val_idx],
            [id_all[i] for i in val_idx],
            data,
        )
        # Refit on train+val for OOF test predictions
        tv_idx = tr_idx + val_idx
        model = fit_scen_selector(X_all[tv_idx], y_all[tv_idx], C=1.0 if "1.0" in name else 0.5)
        preds = predict_policies(model, X_all[te_idx])
        maj = majority_policy(y_all[tv_idx])

        live_stage1 = None
        telemetry_n = 0
        if not skip_live:
            tele = []
            for sid in tr_ids + val_ids:
                s = scen_by_id[sid]
                vbs = str(data.set_index("scenario_id").loc[sid, "vbs_policy"])
                tele.extend(collect_probe_telemetry(s, vbs))
            telemetry_n = len(tele)
            live_stage1 = fit_live_stage1(tele)

        for sid, pred in zip(test_ids, preds):
            row = data.set_index("scenario_id").loc[sid]
            a_scen = float(row[pred])
            maj_u = float(row[maj])
            live_u = float("nan")
            n_switch = 0
            if live_stage1 is not None:
                live_u, n_switch, _traj = run_live_router_anwg(scen_by_id[sid], live_stage1)
            rec = {
                "scenario_id": sid,
                "fold": fold,
                "n_elevated_mechanisms": int(row["n_elevated_mechanisms"]),
                "vbs_policy": str(row["vbs_policy"]),
                "vbs_anwg": float(row["vbs_anwg"]),
                "majority_policy": maj,
                "majority_anwg": maj_u,
                "a_scen_policy": str(pred),
                "a_scen_anwg": a_scen,
                "a_live_anwg": live_u,
                "a_live_n_switches": n_switch,
                "realized_gain_scen_vs_rowmax": float(a_scen - float(row[list(P6)].max())),
                "exploitability_gap_scen": float(row["vbs_anwg"] - a_scen),
                "correct_winner_scen": bool(pred == row["vbs_policy"]),
            }
            for p in P6:
                rec[p] = float(row[p])
            oof_rows.append(rec)
        model_meta.append(
            {
                "fold": fold,
                "selected_model": name,
                "val_mean_anwg": val_score,
                "n_train": len(tr_ids),
                "n_val": len(val_ids),
                "n_test": len(test_ids),
                "majority_policy": maj,
                "telemetry_rows": telemetry_n,
                "live_enabled": not skip_live,
            }
        )
        log(
            f"fold={fold} model={name} val_anwg={val_score:.6f} "
            f"n_test={len(test_ids)} telemetry_rows={telemetry_n}"
        )

    oof = pd.DataFrame(oof_rows)
    oof.to_csv(out_dir / "per_scenario_oof_results.csv", index=False)
    _write_json(out_dir / "model_metadata.json", model_meta)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "smoke": smoke,
        "n_scenarios": int(len(oof)),
        "p6": list(P6),
        "global_best_fixed_on_subset": sbs_policy_global,
        "A_scen": summarize_oof(oof, "a_scen_anwg"),
        "majority": summarize_oof(oof, "majority_anwg"),
        "classification_A_scen": {
            "macro_f1_vs_vbs_winner": float(
                f1_score(
                    oof["vbs_policy"],
                    oof["a_scen_policy"],
                    average="macro",
                    labels=list(P6),
                    zero_division=0,
                )
            ),
            "accuracy": float(np.mean(oof["correct_winner_scen"])),
            "mean_regret_when_correct": float(
                (oof.loc[oof["correct_winner_scen"], "vbs_anwg"] - oof.loc[oof["correct_winner_scen"], "a_scen_anwg"]).mean()
            )
            if oof["correct_winner_scen"].any()
            else float("nan"),
            "mean_regret_when_incorrect": float(
                (oof.loc[~oof["correct_winner_scen"], "vbs_anwg"] - oof.loc[~oof["correct_winner_scen"], "a_scen_anwg"]).mean()
            )
            if (~oof["correct_winner_scen"]).any()
            else float("nan"),
        },
        "stratified_by_n_elevated_A_scen": {},
        "wall_s": float(time.perf_counter() - t0),
    }
    if not skip_live and oof["a_live_anwg"].notna().all():
        summary["A_live"] = summarize_oof(oof, "a_live_anwg")
        summary["A_live"]["mean_switches"] = float(oof["a_live_n_switches"].mean())

    for label, mask in [
        ("0_1", oof["n_elevated_mechanisms"] <= 1),
        ("2", oof["n_elevated_mechanisms"] == 2),
        ("3", oof["n_elevated_mechanisms"] == 3),
        ("ge4", oof["n_elevated_mechanisms"] >= 4),
    ]:
        sub = oof.loc[mask]
        if len(sub) == 0:
            continue
        summary["stratified_by_n_elevated_A_scen"][label] = summarize_oof(sub, "a_scen_anwg")

    bootstrap = {
        "seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "A_scen_minus_SBS": summary["A_scen"]["bootstrap_adaptive_minus_sbs"],
        "VBS_minus_A_scen": summary["A_scen"]["bootstrap_vbs_minus_adaptive"],
    }
    if "A_live" in summary:
        bootstrap["A_live_minus_SBS"] = summary["A_live"]["bootstrap_adaptive_minus_sbs"]
        bootstrap["VBS_minus_A_live"] = summary["A_live"]["bootstrap_vbs_minus_adaptive"]

    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "bootstrap.json", bootstrap)

    provenance = {
        "joint_utility_matrix": str(JOINT_DIR / "utility_matrix_wide.csv"),
        "joint_utility_sha256": sha256_file(JOINT_DIR / "utility_matrix_wide.csv"),
        "joint_manifest": str(JOINT_DIR / "scenario_manifest.csv"),
        "joint_manifest_sha256": sha256_file(JOINT_DIR / "scenario_manifest.csv"),
        "design_doc": "docs/design/JOINT240_SAME_DISTRIBUTION_ADAPTIVE_EXPLOITABILITY_V1.md",
        "schema_version": SCHEMA_VERSION,
    }
    _write_json(out_dir / "provenance.json", provenance)
    _write_json(
        out_dir / "run_manifest.json",
        {
            "out_dir": str(out_dir),
            "smoke": smoke,
            "skip_live": skip_live,
            "wall_s": summary["wall_s"],
            "n_scenarios": summary["n_scenarios"],
            "n_folds": n_folds,
        },
    )
    (out_dir / "DONE").write_text(
        json.dumps({"ok": True, "wall_s": summary["wall_s"]}, indent=2) + "\n"
    )
    log(f"DONE wall_s={summary['wall_s']:.2f}")
    log(
        "A_scen: "
        f"R={summary['A_scen']['R_a_scen_anwg']:.6f} "
        f"gain={summary['A_scen']['realized_gain']:.6f} "
        f"gap={summary['A_scen']['exploitability_gap']:.6f} "
        f"closure={summary['A_scen']['gap_closure']:.4f}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=8)
    ap.add_argument("--skip-live", action="store_true")
    args = ap.parse_args()
    return run(args.out_dir, smoke=args.smoke, smoke_n=args.smoke_n, skip_live=args.skip_live)


if __name__ == "__main__":
    raise SystemExit(main())
