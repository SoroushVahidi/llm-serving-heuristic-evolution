#!/usr/bin/env python3
"""Replay frozen joint-240 terminal forks with per-request utility traces."""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from llmserveopt.analysis.decision_criticality_terminal_anwg_joint240_v1 import (
    ANWG_EQ_ATOL,
    fit_oof_alive_stage1_models,
    load_frozen_joint240_context,
)
from llmserveopt.analysis.decision_criticality_terminal_utility_joint240_v1 import (
    BOOTSTRAP_SEED,
    MEANINGFUL_EPS,
    N_BOOTSTRAP,
    PARENT_EXP,
    PRACTICAL,
    SCHEMA_VERSION,
    bootstrap_scenario_stats,
    concentration_curve,
    load_frozen_parent_branches,
    run_scenario_utility_replay,
    scenario_top_k_share_mult,
)

DESIGN = REPO / "docs/design/DECISION_CRITICALITY_TERMINAL_UTILITY_JOINT240_V1.md"
OUT = REPO / "experiments/decision_criticality_terminal_utility_joint240_v1"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def summarize_effect(series: pd.Series, name: str) -> Dict[str, Any]:
    x = series.to_numpy(dtype=float)
    ax = np.abs(x)
    return {
        "metric": name,
        "n": int(len(x)),
        "frac_exact_zero": float((ax <= 1e-12).mean()) if len(x) else None,
        "frac_nonzero_1e12": float((ax > 1e-12).mean()) if len(x) else None,
        "frac_above_1e9": float((ax > MEANINGFUL_EPS).mean()) if len(x) else None,
        "frac_practical_0p001": float((ax >= PRACTICAL).mean()) if len(x) else None,
        "frac_positive_1e12": float((x > 1e-12).mean()) if len(x) else None,
        "frac_negative_1e12": float((x < -1e-12).mean()) if len(x) else None,
        "mean_abs": float(ax.mean()) if len(x) else None,
        "median_abs": float(np.median(ax)) if len(x) else None,
        "p90_abs": float(np.quantile(ax, 0.90)) if len(x) else None,
        "p95_abs": float(np.quantile(ax, 0.95)) if len(x) else None,
        "p99_abs": float(np.quantile(ax, 0.99)) if len(x) else None,
        "concentration_abs": concentration_curve(ax),
        "top5_scenario_mass": None,  # filled by caller
    }


def cross_metric_overlap(df: pd.DataFrame) -> Dict[str, Any]:
    anwg = df["abs_delta_anwg_live"] > MEANINGFUL_EPS
    wmt = df["delta_wmt_improvement"].abs() > MEANINGFUL_EPS
    soft = df["delta_soft"].abs() > MEANINGFUL_EPS
    def jacc(a, b):
        u = int((a | b).sum())
        return float((a & b).sum()) / u if u else None
    return {
        "eps": MEANINGFUL_EPS,
        "n_anwg": int(anwg.sum()),
        "n_wmt": int(wmt.sum()),
        "n_soft": int(soft.sum()),
        "anwg_and_wmt": int((anwg & wmt).sum()),
        "anwg_not_wmt": int((anwg & ~wmt).sum()),
        "wmt_not_anwg": int((~anwg & wmt).sum()),
        "jaccard_anwg_wmt": jacc(anwg, wmt),
        "jaccard_anwg_soft": jacc(anwg, soft),
        "spearman_abs_anwg_wmt": float(
            pd.Series(df["abs_delta_anwg_live"]).corr(
                df["delta_wmt_improvement"].abs(), method="spearman"
            )
        )
        if len(df) > 2
        else None,
        "topk_overlap_abs": {},
    }


def topk_overlap(df: pd.DataFrame, col_a: str, col_b: str, frac: float) -> float:
    n = len(df)
    k = max(1, int(np.ceil(frac * n)))
    a = set(df.nlargest(k, col_a).index)
    b = set(df.nlargest(k, col_b).index)
    return float(len(a & b) / k)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--limit-scenarios", type=int, default=0)
    ap.add_argument("--scenario-ids", nargs="*", default=[])
    ap.add_argument("--parent-branches", type=Path, default=PARENT_EXP / "branches.csv")
    args = ap.parse_args()

    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    log = out / "logs" / "full_run.log"
    t0 = time.time()
    start = datetime.now(timezone.utc).isoformat()

    def logline(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        with log.open("a") as f:
            f.write(line + "\n")

    logline(f"start {start} schema={SCHEMA_VERSION}")
    if DESIGN.exists():
        (out / "DESIGN_FROZEN.md").write_text(DESIGN.read_text())

    parent = load_frozen_parent_branches(args.parent_branches)
    ctx = load_frozen_joint240_context()
    scenarios = ctx["scenarios"]
    folds = ctx["folds"]
    matrix = ctx["matrix"]
    manifest = ctx["manifest"].set_index("scenario_id")

    scen_list = sorted(parent["scenario_id"].unique().tolist())
    if args.scenario_ids:
        scen_list = [s for s in scen_list if s in set(args.scenario_ids)]
    if args.limit_scenarios > 0:
        scen_list = scen_list[: args.limit_scenarios]
    parent = parent[parent["scenario_id"].isin(scen_list)].copy()

    needed_folds = sorted(
        folds.loc[folds["scenario_id"].isin(scen_list), "fold"].astype(int).unique().tolist()
    )
    logline(f"scenarios={len(scen_list)} branches={len(parent)} folds={needed_folds}")

    config = {
        "schema_version": SCHEMA_VERSION,
        "parent_experiment": str(PARENT_EXP.relative_to(REPO)),
        "n_scenarios": len(scen_list),
        "n_parent_branches": int(len(parent)),
        "anwg_atol": ANWG_EQ_ATOL,
        "meaningful_eps": MEANINGFUL_EPS,
        "practical_threshold": PRACTICAL,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "design": str(DESIGN.relative_to(REPO)),
    }
    _write_json(out / "config.json", config)

    logline("Fitting OOF Alive Stage-1...")
    models = fit_oof_alive_stage1_models(
        scenarios, matrix, folds, fold_ids=needed_folds
    )
    logline(f"Fitted folds={sorted(models)}")

    trace_path = out / "request_traces.jsonl.gz"
    branch_path = out / "branches.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    if branch_path.exists():
        branch_path.unlink()

    trace_fh = gzip.open(trace_path, "wt")

    def write_trace(rows: List[Dict[str, Any]]) -> None:
        for r in rows:
            trace_fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")

    fold_map = folds.set_index("scenario_id")["fold"].to_dict()
    scen_rows = []
    failures = []
    n_branches = 0

    for i, sid in enumerate(scen_list):
        fold = int(fold_map[sid])
        seed = int(manifest.loc[sid, "seed"])
        fr = parent[parent["scenario_id"] == sid]
        logline(f"[{i+1}/{len(scen_list)}] {sid} fold={fold} n_frozen={len(fr)}")
        try:
            res = run_scenario_utility_replay(
                scenarios[sid],
                stage1=models[fold],
                fold=fold,
                seed=seed,
                frozen_rows=fr,
                write_trace=write_trace,
            )
            with branch_path.open("a") as bf:
                for br in res["branch_rows"]:
                    bf.write(json.dumps(br, sort_keys=True, default=str) + "\n")
                    n_branches += 1
            scen_rows.append({k: v for k, v in res.items() if k != "branch_rows"})
            logline(
                f"  matched={res['n_matched_steps']}/{res['n_frozen_expected']} "
                f"max_cf_anwg_err={res['max_cf_anwg_mismatch_vs_parent']:.3e}"
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            failures.append({"scenario_id": sid, "error": f"{type(e).__name__}: {e}"})
            logline(f"FAIL {sid}: {e}")
            logline(traceback.format_exc())

    trace_fh.close()
    pd.DataFrame(scen_rows).to_csv(out / "scenario_summaries.csv", index=False)

    branches = (
        pd.DataFrame([json.loads(l) for l in branch_path.read_text().splitlines() if l.strip()])
        if branch_path.exists() and branch_path.stat().st_size
        else pd.DataFrame()
    )
    if len(branches):
        branches.to_csv(out / "branches.csv", index=False)

    # Integrity vs parent
    integrity = {
        "n_parent": int(len(parent)),
        "n_replayed": int(len(branches)),
        "n_scenarios": int(len(scen_list)),
        "n_failures": int(len(failures)),
    }
    if len(branches):
        integrity.update(
            {
                "max_parent_cf_anwg_abs_err": float(branches["parent_cf_anwg_abs_err"].max()),
                "max_parent_ref_anwg_abs_err": float(branches["parent_ref_anwg_abs_err"].max()),
                "max_parent_delta_anwg_abs_err": float(branches["parent_delta_anwg_abs_err"].max()),
                "n_cf_anwg_match": int((branches["parent_cf_anwg_abs_err"] <= ANWG_EQ_ATOL).sum()),
                "n_delta_anwg_match": int((branches["parent_delta_anwg_abs_err"] <= ANWG_EQ_ATOL).sum()),
                "frac_policy_match": float(branches["policy_match"].mean()),
                "frac_alt_match": float(branches["alt_match"].mean()),
                "ref_replay_n": int(branches["ref_replay_anwg"].notna().sum())
                if "ref_replay_anwg" in branches.columns
                else 0,
                "ref_replay_n_match": int(branches.get("ref_replay_matches", pd.Series(dtype=bool)).fillna(False).sum())
                if "ref_replay_matches" in branches.columns
                else 0,
            }
        )
        integrity["anwg_reproduction_ok"] = bool(
            integrity["max_parent_delta_anwg_abs_err"] <= ANWG_EQ_ATOL
            and integrity["n_replayed"] == integrity["n_parent"]
            and integrity["n_failures"] == 0
        )
    else:
        integrity["anwg_reproduction_ok"] = False

    _write_json(out / "anwg_reproduction.json", integrity)
    logline(f"integrity={json.dumps(integrity)}")

    if not integrity.get("anwg_reproduction_ok", False) and len(branches) == len(parent) and not failures:
        # allow tiny float drift reporting but STOP analysis if material
        if integrity.get("max_parent_delta_anwg_abs_err", 1) > 1e-9:
            logline("STOP: material ANWG reproduction mismatch")
            _write_json(out / "INTEGRITY_FAIL", integrity)
            return 2

    # Summaries
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_utc": start,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.time() - t0,
        "config": config,
        "integrity": integrity,
        "failures": failures,
        "metrics": {},
        "bootstrap": {},
        "cross_metric_overlap": {},
    }

    if len(branches):
        metric_map = {
            "delta_anwg_live": "ANWG",
            "delta_wcg": "WCG",
            "delta_wmt_improvement": "WMT_improvement",
            "delta_wnt_improvement": "WNT_improvement",
            "delta_soft": "SoftGoodput",
        }
        for col, name in metric_map.items():
            sm = summarize_effect(branches[col], name)
            # scenario mass
            sc = branches.groupby("scenario_id")[col].apply(lambda s: float(np.abs(s).sum()))
            sm["top5_scenario_mass"] = scenario_top_k_share_mult(sc.to_numpy(float), 5)
            sm["scenario_concentration"] = concentration_curve(sc.to_numpy(float))
            summary["metrics"][name] = sm
            summary["bootstrap"][name] = bootstrap_scenario_stats(branches, effect_col=col)

        ov = cross_metric_overlap(branches)
        ov["topk_overlap_abs"] = {
            "top1pct_anwg_wmt": topk_overlap(
                branches.assign(a=branches["abs_delta_anwg_live"], b=branches["delta_wmt_improvement"].abs()),
                "a",
                "b",
                0.01,
            ),
            "top5pct_anwg_wmt": topk_overlap(
                branches.assign(a=branches["abs_delta_anwg_live"], b=branches["delta_wmt_improvement"].abs()),
                "a",
                "b",
                0.05,
            ),
            "top10pct_anwg_wmt": topk_overlap(
                branches.assign(a=branches["abs_delta_anwg_live"], b=branches["delta_wmt_improvement"].abs()),
                "a",
                "b",
                0.10,
            ),
        }
        summary["cross_metric_overlap"] = ov

        # Verdict
        wmt = summary["metrics"]["WMT_improvement"]
        top10 = wmt["concentration_abs"]["0.1"]["share"]
        prev = wmt["frac_above_1e9"]
        if not integrity.get("anwg_reproduction_ok", False) and integrity.get("max_parent_delta_anwg_abs_err", 1) > 1e-9:
            labels = ["TERMINAL_UTILITY_ROBUSTNESS_INCONCLUSIVE"]
        elif top10 is not None and top10 < 0.30:
            labels = ["ANWG_CRITICALITY_NOT_ROBUST_TO_CONTINUOUS_UTILITY"]
        elif prev is not None and prev >= 0.25 and top10 is not None and top10 >= 0.50:
            labels = ["ANWG_ZERO_RATE_STEP_FUNCTION_ARTIFACT_BUT_CONCENTRATION_ROBUST"]
        elif prev is not None and prev < 0.25 and top10 is not None and top10 >= 0.50:
            labels = ["TERMINAL_CRITICALITY_ROBUST_TO_CONTINUOUS_UTILITY"]
        else:
            labels = ["TERMINAL_UTILITY_ROBUSTNESS_INCONCLUSIVE"]
        summary["verdicts"] = labels

    _write_json(out / "summary.json", summary)
    _write_json(out / "bootstrap.json", summary.get("bootstrap", {}))
    _write_json(out / "cross_metric_overlap.json", summary.get("cross_metric_overlap", {}))

    if len(branches):
        pd.DataFrame(
            [{"metric": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}} for k, v in summary["metrics"].items()]
        ).to_csv(out / "summary.csv", index=False)

    (out / "DONE").write_text(
        json.dumps(
            {
                "ok": True,
                "elapsed_s": summary["elapsed_s"],
                "n_branches": int(len(branches)),
                "verdicts": summary.get("verdicts"),
                "anwg_reproduction_ok": integrity.get("anwg_reproduction_ok"),
            },
            indent=2,
        )
        + "\n"
    )
    logline(
        f"DONE elapsed={summary['elapsed_s']:.1f}s branches={len(branches)} "
        f"verdicts={summary.get('verdicts')} anwg_ok={integrity.get('anwg_reproduction_ok')}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
