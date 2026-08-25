#!/usr/bin/env python3
"""Run joint-240 SBS-continuation robustness for terminal criticality.

Parent Alive-continuation experiment is NOT overwritten.
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
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from llmserveopt.analysis import joint240_terminal_criticality_sbs_continuation_v1 as sbs  # noqa: E402
from llmserveopt.analysis.decision_criticality_terminal_anwg_joint240_v1 import (  # noqa: E402
    ANWG_EQ_ATOL,
    fit_oof_alive_stage1_models,
    load_frozen_joint240_context,
)

OUT_DEFAULT = REPO / "experiments" / "joint240_terminal_criticality_sbs_continuation_v1"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run(out_dir: Path, *, limit_scenarios: int = 0) -> int:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    log_path = out_dir / "logs" / "full_run.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    design_hash = sbs.sha256_file(sbs.DESIGN_DOC)
    frozen_path = out_dir / "DESIGN_FROZEN.sha256"
    if not frozen_path.exists():
        # Allow smoke subdirs to inherit the parent experiment design hash.
        frozen_path = OUT_DEFAULT / "DESIGN_FROZEN.sha256"
    frozen = frozen_path.read_text().split()[0]
    if design_hash != frozen:
        raise RuntimeError(f"design hash drift {design_hash} vs {frozen}")
    log(f"schema={sbs.SCHEMA_VERSION} design_sha256={design_hash}")

    parent_keys = sbs.load_parent_acquisition_keys()
    log(f"parent acquisition rows={len(parent_keys)} scenarios={parent_keys.scenario_id.nunique()}")

    ctx = load_frozen_joint240_context()
    scenarios = ctx["scenarios"]
    folds = ctx["folds"]
    matrix = ctx["matrix"]

    # Restrict scenarios if smoke
    scen_ids = sorted(parent_keys["scenario_id"].unique().tolist())
    if limit_scenarios > 0:
        scen_ids = scen_ids[:limit_scenarios]
        parent_keys = parent_keys[parent_keys["scenario_id"].isin(scen_ids)].copy()
        log(f"limit_scenarios={limit_scenarios} rows={len(parent_keys)}")

    fold_needed = sorted(
        folds.loc[folds["scenario_id"].isin(scen_ids), "fold"].astype(int).unique().tolist()
    )
    log(f"fitting OOF Alive Stage-1 for folds={fold_needed}")
    models = fit_oof_alive_stage1_models(
        scenarios, matrix, folds, fold_ids=fold_needed
    )

    config = {
        "schema_version": sbs.SCHEMA_VERSION,
        "design_sha256": design_hash,
        "parent_experiment": str(sbs.PARENT.relative_to(REPO)),
        "parent_branches_sha256": sbs.sha256_file(sbs.PARENT / "branches.csv"),
        "parent_summary_sha256": sbs.sha256_file(sbs.PARENT / "summary.json"),
        "sbs_policy": sbs.SBS_POLICY,
        "bootstrap_seed": sbs.BOOTSTRAP_SEED,
        "n_bootstrap": sbs.N_BOOTSTRAP,
        "limit_scenarios": limit_scenarios,
        "git_head": _git_head(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }
    _write_json(out_dir / "config.json", config)

    all_rows: List[Dict[str, Any]] = []
    missing_total = 0
    for i, sid in enumerate(scen_ids):
        fold = int(folds.set_index("scenario_id").loc[sid, "fold"])
        sub = parent_keys[parent_keys["scenario_id"] == sid]
        # Use parent-recorded seed for exact replay fidelity
        if "seed" in sub.columns and sub["seed"].notna().any():
            seed = int(sub["seed"].iloc[0])
        elif hasattr(scenarios[sid], "seed"):
            seed = int(scenarios[sid].seed)
        else:
            seed = int(scenarios[sid].params.get("seed", 20260824))
        res = sbs.run_scenario_sbs_continuation(
            scenarios[sid],
            stage1=models[fold],
            seed=seed,
            parent_rows=sub,
        )
        missing_total += int(res["n_missing"])
        all_rows.extend(res["branch_rows"])
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(scen_ids):
            log(
                f"progress {i+1}/{len(scen_ids)} sid={sid} "
                f"hit={res['n_hit']}/{res['n_targets']} missing={res['n_missing']}"
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "branches_sbs_continuation.csv", index=False)
    log(f"wrote {len(df)} SBS-continuation rows; missing_keys_total={missing_total}")

    if len(df) == 0:
        raise RuntimeError("no SBS continuation rows produced")

    # Primary SBS stats
    sbs_nz = df["sbs_nonzero"].to_numpy(dtype=bool)
    prev = sbs.scenario_clustered_bootstrap_prevalence(df, "sbs_nonzero")
    top1 = sbs.concentration_share(df["sbs_abs_delta_anwg"].to_numpy(float), 0.01)
    top5 = sbs.concentration_share(df["sbs_abs_delta_anwg"].to_numpy(float), 0.05)
    top10 = sbs.concentration_share(df["sbs_abs_delta_anwg"].to_numpy(float), 0.10)

    # Paired robustness
    both_nz = (df["alive_nonzero"] & df["sbs_nonzero"]).sum()
    only_alive = (df["alive_nonzero"] & ~df["sbs_nonzero"]).sum()
    only_sbs = (~df["alive_nonzero"] & df["sbs_nonzero"]).sum()
    both_z = (~df["alive_nonzero"] & ~df["sbs_nonzero"]).sum()
    agree_nz = float(((df["alive_nonzero"] == df["sbs_nonzero"]).mean()))

    dual = df[df["alive_nonzero"] & df["sbs_nonzero"]]
    if len(dual):
        sign_agree = float(
            ((np.sign(dual["alive_delta_anwg"]) == np.sign(dual["sbs_delta_anwg"])).mean())
        )
    else:
        sign_agree = float("nan")

    spear = spearmanr(df["alive_abs_delta_anwg"], df["sbs_abs_delta_anwg"])
    spear_r = float(spear.correlation) if spear.correlation is not None else float("nan")

    def top_set(abs_col: str, frac: float) -> Set[Tuple[str, int, str, str]]:
        vals = df[abs_col].to_numpy(float)
        k = max(1, int(np.ceil(frac * len(df))))
        idx = np.argsort(-vals)[:k]
        out = set()
        for i in idx:
            r = df.iloc[int(i)]
            out.add(
                (str(r.scenario_id), int(r.step), str(r.acquisition_type), str(r.alt_policy_id))
            )
        return out

    jac1 = sbs.jaccard(top_set("alive_abs_delta_anwg", 0.01), top_set("sbs_abs_delta_anwg", 0.01))
    jac5 = sbs.jaccard(top_set("alive_abs_delta_anwg", 0.05), top_set("sbs_abs_delta_anwg", 0.05))

    # Disagreement proxy under SBS
    y = df["sbs_nonzero"].astype(int).to_numpy()
    score = (df["acquisition_type"] == "DISAGREEMENT").astype(float).to_numpy()
    if y.min() != y.max() and score.min() != score.max():
        auroc = float(roc_auc_score(y, score))
        auprc = float(average_precision_score(y, score))
    else:
        auroc = float("nan")
        auprc = float("nan")

    summary: Dict[str, Any] = {
        "schema_version": sbs.SCHEMA_VERSION,
        "design_sha256": design_hash,
        "n_rows": int(len(df)),
        "n_scenarios": int(df["scenario_id"].nunique()),
        "n_missing_keys": int(missing_total),
        "sbs_nonzero_n": int(sbs_nz.sum()),
        "sbs_nonzero_prevalence": prev,
        "sbs_mean_abs_delta": float(df["sbs_abs_delta_anwg"].mean()),
        "sbs_top1pct_mass": top1,
        "sbs_top5pct_mass": top5,
        "sbs_top10pct_mass": top10,
        "alive_nonzero_n": int(df["alive_nonzero"].sum()),
        "alive_nonzero_prevalence": float(df["alive_nonzero"].mean()),
        "overlap": {
            "both_nonzero": int(both_nz),
            "only_alive_nonzero": int(only_alive),
            "only_sbs_nonzero": int(only_sbs),
            "both_zero": int(both_z),
            "nonzero_label_agreement": agree_nz,
            "sign_agreement_among_dual_nonzero": sign_agree,
        },
        "spearman_abs_alive_vs_sbs": spear_r,
        "jaccard_top1pct": jac1,
        "jaccard_top5pct": jac5,
        "disagreement_proxy_sbs": {
            "auroc": auroc,
            "auprc": auprc,
            "noskill_prevalence": float(y.mean()),
            "score": "binary acquisition_type==DISAGREEMENT",
        },
        "wall_s": float(time.perf_counter() - t0),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["continuation_verdict"] = sbs.classify_continuation(summary)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "continuation_robustness.json", {
        "overlap": summary["overlap"],
        "spearman_abs_alive_vs_sbs": spear_r,
        "jaccard_top1pct": jac1,
        "jaccard_top5pct": jac5,
        "continuation_verdict": summary["continuation_verdict"],
    })

    log(
        f"DONE n={len(df)} sbs_nz={summary['sbs_nonzero_n']} "
        f"prev={prev['mean']:.4f} spearman={spear_r:.3f} "
        f"jac5={jac5:.3f} verdict={summary['continuation_verdict']} "
        f"wall_s={summary['wall_s']:.1f}"
    )
    (out_dir / "DONE").write_text("ok\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--limit-scenarios", type=int, default=0)
    args = ap.parse_args()
    return run(args.out_dir, limit_scenarios=args.limit_scenarios)


if __name__ == "__main__":
    raise SystemExit(main())
