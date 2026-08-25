#!/usr/bin/env python3
"""Terminal-ANWG one-step decision criticality v1 — TRAIN/VAL runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from llmserveopt.analysis import decision_criticality_terminal_anwg_v1 as tan  # noqa: E402
from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm  # noqa: E402

DESIGN = REPO / "docs/design/DECISION_CRITICALITY_TERMINAL_ANWG_V1.md"
OUT = REPO / "experiments/decision_criticality_terminal_anwg_v1"
V1_EVENTS = REPO / "experiments/decision_criticality_timescale_trainval_v1/disagreement_and_divergence_events.csv"


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--short"], text=True).strip())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def concentration_curve(abs_vals: np.ndarray, fracs=(0.01, 0.05, 0.10, 0.20, 0.50)) -> dict:
    vals = np.asarray(abs_vals, dtype=float)
    if len(vals) == 0:
        return {str(f): {"k": 0, "share": None} for f in fracs}
    order = np.argsort(-vals)
    sorted_v = vals[order]
    total = float(sorted_v.sum())
    cum = np.cumsum(sorted_v)
    out = {}
    for f in fracs:
        k = max(1, int(np.ceil(f * len(sorted_v))))
        out[str(f)] = {
            "k": k,
            "share": float(cum[k - 1] / total) if total > 0 else 0.0,
        }
    return out


def analyze(branches: pd.DataFrame) -> dict:
    n = len(branches)
    abs_d = branches["abs_delta_anwg"].to_numpy(dtype=float)
    signed = branches["delta_anwg"].to_numpy(dtype=float)
    pos = np.maximum(signed, 0.0)

    prev = {
        "n_states": n,
        "frac_exact_zero": float((abs_d <= tan.ANWG_EQ_ATOL).mean()) if n else None,
        "frac_nonzero": float((abs_d > tan.ANWG_EQ_ATOL).mean()) if n else None,
        "frac_positive": float((signed > tan.ANWG_EQ_ATOL).mean()) if n else None,
        "frac_negative": float((signed < -tan.ANWG_EQ_ATOL).mean()) if n else None,
        "mean_abs_delta": float(abs_d.mean()) if n else None,
        "median_abs_delta": float(np.median(abs_d)) if n else None,
        "thresholds": {},
    }
    for t in tan.PRACTICAL_THRESHOLDS:
        prev["thresholds"][str(t)] = float((abs_d >= t).mean()) if n else None

    by_acq = {}
    for acq, g in branches.groupby("acquisition_type"):
        a = g["abs_delta_anwg"].to_numpy(dtype=float)
        by_acq[acq] = {
            "n": int(len(g)),
            "mean_abs": float(a.mean()),
            "median_abs": float(np.median(a)),
            "frac_nonzero": float((a > tan.ANWG_EQ_ATOL).mean()),
            "thresholds": {str(t): float((a >= t).mean()) for t in tan.PRACTICAL_THRESHOLDS},
        }

    # Disagreement indicator as criticality proxy (AUROC/AUPRC if both classes present)
    disagree_proxy = {"available": False}
    if "acquisition_type" in branches.columns and len(branches):
        y = (branches["abs_delta_anwg"] > tan.ANWG_EQ_ATOL).astype(int).to_numpy()
        s = (branches["acquisition_type"] == "DISAGREEMENT").astype(float).to_numpy()
        if y.min() != y.max() and s.min() != s.max():
            # Mann-Whitney AUROC without sklearn
            s_pos = s[y == 1]
            s_neg = s[y == 0]
            gt = float(np.mean(s_pos[:, None] > s_neg[None, :]))
            eq = float(np.mean(s_pos[:, None] == s_neg[None, :]))
            auroc = gt + 0.5 * eq
            # AUPRC for positive class with binary score
            order = np.argsort(-s)
            y_ord = y[order]
            tp = np.cumsum(y_ord)
            fp = np.cumsum(1 - y_ord)
            precision = tp / np.maximum(tp + fp, 1)
            recall = tp / max(float(y.sum()), 1.0)
            auprc = float(np.sum(precision * np.diff(np.concatenate([[0.0], recall]))))
            disagree_proxy = {
                "available": True,
                "auroc_disagreement_for_nonzero_abs_delta": auroc,
                "auprc_disagreement_for_nonzero_abs_delta": auprc,
                "n_positive": int(y.sum()),
                "n_negative": int((1 - y).sum()),
            }

    by_fam = {}
    for fam, g in branches.groupby("mechanism_family"):
        a = g["abs_delta_anwg"].to_numpy(dtype=float)
        by_fam[fam] = {
            "n": int(len(g)),
            "mean_abs": float(a.mean()),
            "frac_nonzero": float((a > tan.ANWG_EQ_ATOL).mean()),
        }

    # Scenario mass
    sc = branches.groupby("canonical_scenario_id")["abs_delta_anwg"].sum().sort_values(ascending=False)
    sc_vals = sc.to_numpy(dtype=float)
    sc_conc = concentration_curve(sc_vals)

    # Bootstrap scenario-grouped
    rng = np.random.default_rng(tan.BOOTSTRAP_SEED)
    scen_ids = sc.index.to_numpy()
    scen_mass = sc.to_numpy(dtype=float)
    boot_mean = []
    boot_top5 = []
    for _ in range(tan.N_BOOTSTRAP):
        idx = rng.integers(0, len(scen_mass), size=len(scen_mass))
        sample = scen_mass[idx]
        # state-level mean abs within resampled scenarios
        # approximate: use scenario mean mass / n_states_per_scenario — use total mass / n
        boot_mean.append(float(sample.mean()))
        order = np.argsort(-sample)
        sorted_s = sample[order]
        tot = float(sorted_s.sum())
        k = max(1, int(np.ceil(0.05 * len(sorted_s))))
        boot_top5.append(float(sorted_s[:k].sum() / tot) if tot > 0 else 0.0)

    # Join H10 proxy from v1 if available
    h10_join = {"available": False}
    if V1_EVENTS.exists() and n:
        # load only horizon==10 rows with completed_count
        parts = []
        for chunk in pd.read_csv(
            V1_EVENTS,
            usecols=["canonical_scenario_id", "step", "horizon", "completed_count_abs_diff"],
            chunksize=200_000,
        ):
            d = chunk[chunk["horizon"] == 10.0]
            if len(d):
                parts.append(d)
        if parts:
            h10 = pd.concat(parts, ignore_index=True)
            h10["h10_completion_critical"] = h10["completed_count_abs_diff"] > 0
            m = branches.merge(
                h10[["canonical_scenario_id", "step", "completed_count_abs_diff", "h10_completion_critical"]],
                on=["canonical_scenario_id", "step"],
                how="left",
            )
            both = m["h10_completion_critical"].notna()
            if both.any():
                sub = m[both]
                anwg_crit = sub["abs_delta_anwg"] > tan.ANWG_EQ_ATOL
                h10_crit = sub["h10_completion_critical"].astype(bool)
                h10_join = {
                    "available": True,
                    "n_joined": int(len(sub)),
                    "frac_h10_crit_also_anwg_crit": float((anwg_crit & h10_crit).sum() / max(h10_crit.sum(), 1)),
                    "frac_anwg_crit_also_h10_crit": float((anwg_crit & h10_crit).sum() / max(anwg_crit.sum(), 1)),
                    "frac_anwg_crit_missed_by_h10": float((anwg_crit & ~h10_crit).sum() / max(anwg_crit.sum(), 1)),
                    "spearman_abs_delta_vs_h10_completed": float(
                        pd.Series(sub["abs_delta_anwg"]).corr(
                            pd.Series(sub["completed_count_abs_diff"]), method="spearman"
                        )
                    )
                    if len(sub) > 2
                    else None,
                }

    # Temporal bursts of ANWG-critical disagreement steps
    crit = branches[
        (branches["acquisition_type"] == "DISAGREEMENT")
        & (branches["abs_delta_anwg"] > tan.ANWG_EQ_ATOL)
    ].sort_values(["canonical_scenario_id", "step"])
    bursts = []
    for sid, g in crit.groupby("canonical_scenario_id"):
        steps = g["step"].tolist()
        if not steps:
            continue
        prev_step = steps[0]
        length = 1
        for s in steps[1:]:
            if s == prev_step + 1:
                length += 1
                prev_step = s
            else:
                bursts.append(length)
                prev_step = s
                length = 1
        bursts.append(length)
    bursts_arr = np.asarray(bursts, dtype=int) if bursts else np.asarray([], dtype=int)

    return {
        "prevalence": prev,
        "by_acquisition": by_acq,
        "by_family": by_fam,
        "concentration_abs_delta_all_states": concentration_curve(abs_d),
        "concentration_positive_gain_all_states": concentration_curve(pos),
        "concentration_abs_delta_disagreement_only": concentration_curve(
            branches.loc[branches["acquisition_type"] == "DISAGREEMENT", "abs_delta_anwg"].to_numpy(float)
        ),
        "scenario_concentration_abs_mass": sc_conc,
        "bootstrap": {
            "n": tan.N_BOOTSTRAP,
            "seed": tan.BOOTSTRAP_SEED,
            "scenario_mean_abs_mass": {
                "mean": float(np.mean(boot_mean)),
                "ci95_low": float(np.quantile(boot_mean, 0.025)),
                "ci95_high": float(np.quantile(boot_mean, 0.975)),
            },
            "top5pct_scenario_mass_share": {
                "mean": float(np.mean(boot_top5)),
                "ci95_low": float(np.quantile(boot_top5, 0.025)),
                "ci95_high": float(np.quantile(boot_top5, 0.975)),
            },
        },
        "h10_proxy_join": h10_join,
        "disagreement_as_criticality_proxy": disagree_proxy,
        "temporal_anwg_critical_disagreement_bursts": {
            "n_bursts": int(len(bursts_arr)),
            "frac_length_1": float((bursts_arr == 1).mean()) if len(bursts_arr) else None,
            "median_length": float(np.median(bursts_arr)) if len(bursts_arr) else None,
        },
        "ref_replay": {
            "n_checks": int(branches["ref_replay_anwg"].notna().sum()),
            "n_match": int(branches.get("ref_replay_matches_reference", pd.Series(dtype=bool)).fillna(False).sum())
            if "ref_replay_matches_reference" in branches.columns
            else 0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-scenarios", type=int, default=0)
    ap.add_argument("--max-disagreement", type=int, default=tan.MAX_DISAGREEMENT_PER_SCENARIO)
    ap.add_argument("--max-agreement", type=int, default=tan.MAX_AGREEMENT_CONTROL_PER_SCENARIO)
    ap.add_argument("--families", nargs="*", default=[])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "run.log"
    t0 = time.time()
    start = datetime.now(timezone.utc).isoformat()

    def logline(msg: str) -> None:
        print(msg, flush=True)
        with log.open("a") as f:
            f.write(msg + "\n")

    logline(f"start {start}")
    dcm.assert_no_replication_module_imported()
    table = dcm.load_trainval_scenario_table()
    if args.families:
        table = table[table["mechanism_family"].isin(args.families)]
    if args.limit_scenarios > 0:
        table = table.head(args.limit_scenarios)

    config = {
        "schema_version": tan.SCHEMA_VERSION,
        "max_disagreement_per_scenario": args.max_disagreement,
        "max_agreement_control_per_scenario": args.max_agreement,
        "control_seed": tan.CONTROL_SEED,
        "bootstrap_seed": tan.BOOTSTRAP_SEED,
        "n_bootstrap": tan.N_BOOTSTRAP,
        "design_doc": str(DESIGN.relative_to(REPO)),
        "design_sha256": _sha(DESIGN) if DESIGN.exists() else None,
        "n_scenarios": int(len(table)),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "python": sys.executable,
        "python_version": platform.python_version(),
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    logline("Fitting frozen Stage-1/2...")
    stage1, stage2 = dcm.fit_frozen_models()

    branch_path = OUT / "branches.jsonl"
    if branch_path.exists():
        branch_path.unlink()
    scen_rows = []
    failures = []

    for i, (_, row) in enumerate(table.iterrows()):
        cid = row["canonical_scenario_id"]
        logline(f"[{i+1}/{len(table)}] {cid}")
        try:
            res = tan.run_scenario_terminal_anwg(
                row,
                stage1=stage1,
                stage2_selectors=stage2,
                max_disagreement=args.max_disagreement,
                max_agreement_control=args.max_agreement,
            )
            with branch_path.open("a") as f:
                for br in res["branch_rows"]:
                    f.write(json.dumps(br, sort_keys=True, default=str) + "\n")
            scen_rows.append({k: v for k, v in res.items() if k != "branch_rows"})
        except Exception as e:  # noqa: BLE001
            import traceback

            failures.append({"canonical_scenario_id": cid, "error": f"{type(e).__name__}: {e}"})
            logline(f"FAIL {cid}: {e}")
            logline(traceback.format_exc())

    scen_df = pd.DataFrame(scen_rows)
    scen_df.to_csv(OUT / "scenario_summaries.csv", index=False)

    branches = pd.DataFrame([json.loads(l) for l in branch_path.read_text().splitlines() if l.strip()]) if branch_path.exists() else pd.DataFrame()
    if len(branches):
        branches.to_csv(OUT / "branches.csv", index=False)
        summary = analyze(branches)
    else:
        summary = {"prevalence": {"n_states": 0}}

    summary.update({
        "started_utc": start,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.time() - t0,
        "n_scenarios_attempted": int(len(table)),
        "n_scenarios_succeeded": int(len(scen_rows)),
        "n_scenarios_failed": int(len(failures)),
        "failures": failures,
        "config": config,
    })
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Concentration curve CSV
    if len(branches):
        abs_d = branches["abs_delta_anwg"].sort_values(ascending=False).to_numpy(float)
        total = abs_d.sum()
        cum = np.cumsum(abs_d)
        frac_states = (np.arange(1, len(abs_d) + 1)) / len(abs_d)
        frac_mass = cum / total if total > 0 else np.zeros_like(cum)
        pd.DataFrame({"frac_states": frac_states, "frac_abs_anwg_mass": frac_mass}).to_csv(
            OUT / "concentration_curve.csv", index=False
        )

    logline(f"DONE elapsed={time.time()-t0:.1f}s branches={len(branches)} fails={len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
