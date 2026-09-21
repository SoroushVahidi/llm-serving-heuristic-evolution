#!/usr/bin/env python3
"""Read-only final metric table for the reference-reserve sensitivity experiment (post-derive).

Reads ONLY the hash-verified completed sensitivity results (disagreement_states.csv, result_summary.json,
bootstrap_summary.json, regime_summary.csv) and the frozen DENOMINATOR_SUMMARY_V1.json.  No simulation, no new
threshold and no new formula: every quantity is a count, sum, share or median of values already recorded.
Usage: final_metric_table_v1.py <existing_results_dir> <denominator_summary.json> <out_dir>
"""
import csv, json, math, statistics, sys
from collections import defaultdict
from pathlib import Path

results, summary_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
S = json.loads(summary_path.read_text())["reserves"]
RESERVES = ("0.82", "0.90", "1.00")
REG_ORDER = [("azure_2023_code", "active_sequence_capacity", "active_8"), ("azure_2023_code", "active_sequence_capacity", "active_4"),
             ("azure_2023_code", "kv_capacity", "kv_16000"), ("azure_2023_conv", "active_sequence_capacity", "active_4"),
             ("azure_2023_conv", "kv_capacity", "kv_16000")]
metrics, regimes = {}, []
for r in RESERVES:
    d = results / f"reserve_{r.replace('.', '')}"
    rows = list(csv.DictReader(open(d / "disagreement_states.csv")))
    H = [float(x["oracle_headroom"]) * 1000.0 for x in rows]                       # simulated ms
    pos = [h for h in H if h > 0]
    boot = json.loads((d / "bootstrap_summary.json").read_text())
    prim = json.loads((d / "result_summary.json").read_text())["primary"]
    mass = math.fsum(H)
    by_win, by_reg = defaultdict(float), defaultdict(float)
    for x, h in zip(rows, H):
        by_win[(x["source_dataset"], int(x["window_index"]))] += h
        by_reg[(x["source_dataset"], x["axis"], x["condition_id"])] += h
    wk, wm = max(by_win.items(), key=lambda kv: kv[1]); rk, rm = max(by_reg.items(), key=lambda kv: kv[1])
    s = S[r]
    metrics[r] = {
        "total_reference_decision_states": s["total_reference_decision_states"], "disagreement_states": s["disagreement_states"],
        "P_D_percent": s["P_D_percent"], "beneficial_states": s["beneficial_states"], "P_B_given_D_percent": 100 * s["P_B_given_D"],
        "mean_H_LAT_sim_ms": mass / len(H),
        "ci95_low_sim_ms": boot["mean_oracle_headroom_ci95_low"] * 1000, "ci95_high_sim_ms": boot["mean_oracle_headroom_ci95_high"] * 1000,
        "ci_bootstrap": f"{boot['bootstrap_replicates']} replicates, seed {boot['bootstrap_seed']}, {boot['clusters']} window clusters",
        "median_H_LAT_sim_ms": statistics.median(H), "positive_state_median_H_LAT_sim_ms": statistics.median(pos) if pos else None,
        "total_headroom_mass_sim_ms": mass, "opportunity_weighted_headroom_sim_ms_per_decision": s["opportunity_weighted_headroom_sim_ms_per_decision"],
        "beneficial_decision_rate_percent": s["beneficial_decision_rate_percent"],
        "all_alternatives_harmful_states": prim["all_harmful_states"], "all_alternatives_harmful_share_percent": 100 * prim["all_harmful_states"] / prim["states"],
        "dominant_window": f"{wk[0]}:w{wk[1]}", "dominant_window_headroom_share_percent": 100 * wm / mass if mass else None,
        "dominant_regime": "/".join(rk), "dominant_regime_headroom_share_percent": 100 * rm / mass if mass else None,
        "consistency_check_mass_vs_derive": abs(mass - s["total_headroom_mass_sim_ms"]) < 1e-9,
    }
    reg_sum = {(x["source_dataset"], x["axis"], x["condition_id"]): x for x in csv.DictReader(open(d / "regime_summary.csv"))}
    for pr in s["per_regime"]:
        key = (pr["source_dataset"], pr["axis"], pr["condition_id"]); rs = reg_sum.get(key)
        n = pr["disagreement_states"]
        regimes.append({"reserve": r, "source_dataset": key[0], "axis": key[1], "condition_id": key[2], "decision_states": pr["decision_states"],
                        "disagreement_states": n, "P_D_percent": 100 * pr["P_D"],
                        "beneficial_states": round(float(rs["P_B_given_D"]) * int(rs["states"])) if rs else 0,
                        "mean_conditional_H_LAT_sim_ms": float(rs["mean_oracle_headroom"]) * 1000 if rs else None,
                        "headroom_mass_sim_ms": (float(rs["mean_oracle_headroom"]) * int(rs["states"]) * 1000) if rs else 0.0})
regimes.sort(key=lambda x: (x["reserve"], REG_ORDER.index((x["source_dataset"], x["axis"], x["condition_id"]))))
out.mkdir(parents=True, exist_ok=True)
(out / "FINAL_RESERVE_METRICS_V1.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
with open(out / "FINAL_RESERVE_METRICS_BY_REGIME_V1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(regimes[0])); w.writeheader(); w.writerows(regimes)
print(json.dumps(metrics, indent=1)); print()
for x in regimes: print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()})
