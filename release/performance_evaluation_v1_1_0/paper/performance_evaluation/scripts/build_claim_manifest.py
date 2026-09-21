#!/usr/bin/env python3
"""Final claim manifest: provenance and verification metadata for the manuscript's quantitative claims.

Every value is RECOMPUTED here from the canonical artifacts (frozen Phase A / Phase B / fresh-causal artifacts, the
post hoc robustness outputs via ``robustness_numbers.compute()``, and the vLLM probe summaries).  Each claim also
carries the literal manuscript text that must contain that value, derived FROM the recomputed value, so a
manuscript edit that drifts from the artifacts (or an artifact change) is caught.  No scientific data is duplicated.

    python scripts/build_claim_manifest.py           # (re)write FINAL_CLAIM_MANIFEST.json
    python scripts/build_claim_manifest.py --check   # verify the committed manifest; exit 1 on any mismatch
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parents[1]
TEX = PAPER / "main.tex"
MANIFEST = PAPER / "FINAL_CLAIM_MANIFEST.json"
EXP = ROOT / "experiments"

_spec = importlib.util.spec_from_file_location("robustness_numbers", HERE / "robustness_numbers.py")
rn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rn)

PHASE_A = EXP / "industry_realism_action_opportunity_phase_a_v1"
PHASE_B = EXP / "industry_realism_action_opportunity_phase_b_v2"
FROZEN = "experiments/fresh_production_latency_headroom_confirmatory_v1"
ROBUST = "experiments/fresh_production_latency_headroom_confirmatory_v1_robustness"
VLLM_PROBE = EXP / "real_vllm_mechanism_validation_v1" / "native_vllm_chunk_budget_semantics_probe_v1"
VLLM_RESULT = EXP / "real_vllm_pressure_action_validation_v1" / "REAL_VLLM_VALIDATION_RESULT_V1.json"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def csv_rows(p: Path):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def comma(n: int) -> str:
    return f"{n:,}"


def ms(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


class Claims:
    def __init__(self):
        self.items = []

    def add(self, cid, claim, value, snippets, source, calc):
        assert cid not in {c["id"] for c in self.items}, cid
        self.items.append({"id": cid, "claim": claim, "value": value, "source_artifact": source, "source_field": calc, "manuscript_text": list(snippets)})


def build_claims():
    C = Claims()
    N = rn.compute()  # asserts agreement with the frozen and robustness artifacts

    # ------------------------------------------------------------------ native (Phase A)
    wl = {r["source_dataset"]: r for r in csv_rows(PHASE_A / "PHASE_A_WORKLOAD_SUMMARY_V1.csv")}
    src_a = rel(PHASE_A / "PHASE_A_WORKLOAD_SUMMARY_V1.csv")
    names = {"azure_2023_code": "Azure code", "azure_2023_conv": "Azure conversation", "burstgpt": "BurstGPT"}
    counts = {k: (int(v["true_canonical_disagreement_states"]), int(v["sbs_decision_states"])) for k, v in wl.items()}
    for k, (d, s) in counts.items():
        assert d == 0, k
    txt = {"azure_2023_code": f"Azure code had 0 disagreements in {comma(counts['azure_2023_code'][1])}",
           "azure_2023_conv": f"Azure conversation had 0 in {comma(counts['azure_2023_conv'][1])}",
           "burstgpt": f"BurstGPT had 0 in {comma(counts['burstgpt'][1])}"}
    for k in ("azure_2023_code", "azure_2023_conv", "burstgpt"):
        C.add(f"native.{k}.action_null", f"{names[k]}: native replay has 0 canonical disagreement states", {"disagreement_states": counts[k][0], "sbs_decision_states": counts[k][1]},
              [txt[k]], src_a, f"row source_dataset={k}: true_canonical_disagreement_states, sbs_decision_states")
    total = sum(s for _, s in counts.values())
    assert 0.95e6 < total < 1.05e6
    C.add("native.total_decision_states", "about one million action-null decision states in native replay", total, ["about one million"], src_a, "sum of sbs_decision_states over the three workloads")
    C.add("native.max_kv_utilization", "maximum KV utilization of native replay (Azure code, Azure conversation, BurstGPT)", [float(wl[k]["max_kv_utilization"]) for k in wl],
          [f"{ms(float(wl['azure_2023_code']['max_kv_utilization']), 4)}, {ms(float(wl['azure_2023_conv']['max_kv_utilization']), 4)}, and {ms(float(wl['burstgpt']['max_kv_utilization']), 4)}"], src_a, "max_kv_utilization, rounded to 4 decimals")
    assert all(int(v["active_sequence_capacity_binding_states"]) == 0 for v in wl.values())
    C.add("native.no_active_cap_binding", "no active-cap binding states in native replay", 0, ["no active-cap binding states"], src_a, "active_sequence_capacity_binding_states == 0 for all three rows")

    # ------------------------------------------------------------------ pressure campaign (Phase B v2)
    wc = csv_rows(PHASE_B / "PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv")
    summ = json.loads((PHASE_B / "PHASE_B_V2_RESULT_SUMMARY.json").read_text())
    src_wc = rel(PHASE_B / "PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv")
    assert len(wc) == summ["completed_window_conditions"] == summ["expected_window_conditions"] == 1080
    C.add("pressure.conditions", "pressure map evaluated 1,080 one-axis window conditions", len(wc), ["evaluated 1,080 one-axis conditions"], src_wc + "; " + rel(PHASE_B / "PHASE_B_V2_RESULT_SUMMARY.json"),
          "row count; completed_window_conditions == expected_window_conditions")
    cls = Counter(r["pressure_regime_class"] for r in wc)
    assert (cls["VALID_UNCONSTRAINED"], cls["VALID_PARTIALLY_CONSTRAINED"], cls["VALID_STRONGLY_CONSTRAINED"], cls["INVALID_HORIZON_TRUNCATED"]) == (926, 38, 96, 20)
    C.add("pressure.condition_classes", "926 valid unconstrained, 38 partially constrained, 96 strongly constrained, 20 horizon-invalid conditions", dict(cls),
          ["926 conditions were valid and unconstrained, 38 partially constrained, 96 strongly constrained, and 20 horizon-invalid"], src_wc, "count by pressure_regime_class")
    valid_dis = sum(int(r["true_canonical_disagreement_states"]) for r in wc if r["pressure_regime_class"].startswith("VALID"))
    assert valid_dis == 11328
    C.add("pressure.disagreement_states", "11,328 canonical disagreement states across valid conditions", valid_dis, ["11,328 canonical disagreement states"], src_wc,
          "sum of true_canonical_disagreement_states over pressure_regime_class starting VALID")
    arr = [r for r in wc if r["axis"] == "arrival_pressure"]
    assert max(float(r["arrival_multiplier"]) for r in arr) == 8.0 and all(int(r["true_canonical_disagreement_states"]) == 0 for r in arr)
    assert {r["source_dataset"] for r in arr} == set(wl)
    C.add("pressure.arrival_scaling_action_null", "arrival scaling through 8x is action-null for all three workloads", {"max_multiplier": 8.0, "conditions": len(arr), "disagreement_states": 0},
          ["Arrival scaling through $8\\times$ remained action-null", "up to eight times"], src_wc, "axis == arrival_pressure: max arrival_multiplier, sum of true_canonical_disagreement_states")

    tmap = csv_rows(PHASE_B / "PHASE_B_V2_TRANSITION_MAP.csv")
    axis_name = {"active_sequence_capacity": "Active-sequence cap", "kv_capacity": "KV capacity"}
    wname = {"azure_2023_code": "Azure code", "azure_2023_conv": "Azure conversation", "burstgpt": "BurstGPT"}

    def val(point):
        return comma(int(point.split("_")[1]))

    rows = []
    for w in ("azure_2023_code", "azure_2023_conv", "burstgpt"):
        for ax in ("active_sequence_capacity", "kv_capacity"):
            r = next(x for x in tmap if x["source_dataset"] == w and x["axis"] == ax)
            rows.append(f"{wname[w]} & {axis_name[ax]} & {val(r['first_binding_point'])} & {val(r['first_disagreement_point'])} & {val(r['sustained_disagreement_point'])} \\\\")
    C.add("pressure.transition_table", "Table 2: first binding, first disagreement and sustained disagreement settings per workload and pressure axis", rows, rows,
          rel(PHASE_B / "PHASE_B_V2_TRANSITION_MAP.csv"), "first_binding_point, first_disagreement_point, sustained_disagreement_point (sustained: windows_with_disagreement >= 2)")

    axis_rows = csv_rows(PHASE_B / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv")
    fig2 = {}
    for w, ax, v in (("azure_2023_code", "active_sequence_capacity", 4), ("azure_2023_code", "kv_capacity", 16000), ("azure_2023_conv", "active_sequence_capacity", 4), ("burstgpt", "active_sequence_capacity", 8)):
        r = next(x for x in axis_rows if x["source_dataset"] == w and x["axis"] == ax and int(float(x["axis_value"])) == v)
        fig2[f"{w}|{ax}|{v}"] = {"disagreement_rate": float(r["disagreement_rate"]), "disagreement_states": int(r["true_canonical_disagreement_states"]), "sbs_decision_states": int(r["sbs_decision_states"])}
    C.add("figure2.pressure_conditions", "Figure 2 pressure-condition prevalences (read from the artifact, not typed into the plot script)", fig2, ["figures/pe_disagreement_rates.pdf"],
          rel(PHASE_B / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv"), "disagreement_rate = true_canonical_disagreement_states / sbs_decision_states for the four plotted rows")
    f3 = {int(float(r["axis_value"])): float(r["disagreement_rate"]) for r in axis_rows if r["source_dataset"] == "azure_2023_code" and r["axis"] == "active_sequence_capacity"}
    C.add("figure3.azure_code_active_cap", "Figure 3 disagreement prevalence versus active-sequence cap (Azure code)", {str(k): v for k, v in sorted(f3.items(), reverse=True)}, ["figures/pe_pressure_transition.pdf"],
          rel(PHASE_B / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv"), "disagreement_rate for source_dataset=azure_2023_code, axis=active_sequence_capacity, ordered by pressure_order (cap decreasing = pressure increasing)")

    # ------------------------------------------------------------------ fresh causal (preregistered)
    P = N["primary"]
    res = json.loads((ROOT / FROZEN / "FRESH_LATENCY_CAUSAL_RESULT_V1.json").read_text())["primary"]
    boot = json.loads((ROOT / FROZEN / "FRESH_LATENCY_BOOTSTRAP_V1.json").read_text())
    src_res = f"{FROZEN}/FRESH_LATENCY_CAUSAL_RESULT_V1.json"
    src_state = f"{FROZEN}/FRESH_LATENCY_STATE_LEVEL_V1.csv"
    assert P["n"] == res["states"] == 720 and P["beneficial"] == res["beneficial_states"] == 590 and P["guarded_beneficial"] == 589
    C.add("fresh.states", "720 fresh disagreement states", 720, ["720 fresh disagreement states"], src_res, "primary.states (== rows of FRESH_LATENCY_STATE_LEVEL_V1.csv)")
    C.add("fresh.beneficial_preregistered", "590 of 720 states have positive one-step latency headroom (strict preregistered criterion), 81.9%", {"beneficial": 590, "share": 590 / 720},
          ["590/720=0.8194\\ (81.9\\%)", "590 (81.9\\%)"], src_res, "primary.beneficial_states; share = beneficial / states")
    C.add("fresh.beneficial_guarded", "589 of 720 states exceed the 1e-12 s floating-point guard, 81.8%", {"guarded": 589, "share": 589 / 720},
          ["589/720 (81.8\\%)", "589 excluding one floating-point-residue state"], src_state, "count of oracle_headroom > 1e-12 s")
    mean_s = res["mean_oracle_headroom"]
    lo_s, hi_s = boot["mean_oracle_headroom_ci95_low"], boot["mean_oracle_headroom_ci95_high"]
    C.add("fresh.mean_headroom", "state-weighted mean oracle headroom 0.001995791 s = 1.9958 ms", {"seconds": mean_s, "ms": 1e3 * mean_s},
          [f"{mean_s:.9f} s (1.9958 ms)", "2.00 ms"], src_res, "primary.mean_oracle_headroom")
    C.add("fresh.mean_headroom_ci", "95% window-clustered percentile-bootstrap CI [0.1909, 3.5462] ms", {"seconds": [lo_s, hi_s], "ms": [1e3 * lo_s, 1e3 * hi_s]},
          [f"[{lo_s:.9f}, {hi_s:.9f}] s", f"[{1e3 * lo_s:.4f}, {1e3 * hi_s:.4f}] ms", "0.19--3.55"], f"{FROZEN}/FRESH_LATENCY_BOOTSTRAP_V1.json",
          "mean_oracle_headroom_ci95_low / _high")
    C.add("fresh.bootstrap_design", "2,000 replicates, seed 20260920, 36 clusters (windows)", {"replicates": boot["bootstrap_replicates"], "seed": boot["bootstrap_seed"], "clusters": boot["clusters"]},
          ["2,000 replicates and seed 20260920", "36 windows"], f"{FROZEN}/FRESH_LATENCY_BOOTSTRAP_V1.json", "bootstrap_replicates, bootstrap_seed, clusters")
    assert (boot["bootstrap_replicates"], boot["bootstrap_seed"], boot["clusters"]) == (2000, 20260920, 36)
    order = lambda r: (r["workload"] != "azure_2023_code", r["axis"] != "active_sequence_capacity", -int(r["setting"].split("_")[1]) if r["axis"] == "active_sequence_capacity" else 0)
    R = sorted(N["regimes"], key=order)
    label = {"active_8": "active cap 8", "active_4": "active cap 4", "kv_16000": "KV 16,000"}
    short = {"azure_2023_code": "Azure code", "azure_2023_conv": "Azure conv."}
    trows = []
    for i, r in enumerate(R, 1):
        trows.append(f"({i}) {short[r['workload']]},\\newline {label[r['setting']]} & {r['states']} & {r['windows']} & {100 * r['P_D']:.3g} & {100 * r['P_B']:.1f} & {r['mean_ms']:.3f} & {100 * r['share']:.2f} \\\\")
    C.add("fresh.regime_table", "Table 3: per-regime states, windows, P(D), P(B|D), mean headroom and headroom share", trows, trows, f"{FROZEN}/FRESH_LATENCY_WORKLOAD_REGIME_V1.csv; " + src_state,
          "robustness_numbers.compute()['regimes'] (asserted equal to the regime CSV and to state-level aggregation)")
    resd = N["residue"]
    C.add("fresh.floating_point_residue", "one beneficial state has advantage 5.6e-17 s (1 ulp at 0.357 s); guard changes the mean by < 1e-19 s", resd,
          ["$5.6\\times10^{-17}$~s", "0.357~s", "less than $10^{-19}$~s"], f"{ROBUST}/numerical_tolerance_audit.json; experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/ROBUSTNESS_AND_590_589_RECHECK_V1.json",
          "residue advantage, reference latency, mean effect")
    assert abs(resd["advantage_s"] - 5.6e-17) < 0.05e-17 and abs(resd["ref_latency_s"] - 0.357) < 5e-4 and resd["mean_effect_s"] < 1e-19

    # ------------------------------------------------------------------ robustness (post hoc)
    D = N["dist"]
    src_rob = f"{ROBUST}/ via paper/performance_evaluation/scripts/robustness_numbers.py::compute (recomputes from {FROZEN}/FRESH_LATENCY_STATE_LEVEL_V1.csv and asserts agreement)"
    C.add("robust.median", "median headroom 0.197 ms", D["median_ms"], [f"{ms(D['median_ms'], 3)}~ms", f"median\nis {ms(D['median_ms'], 2)} ms".replace("\n", " "), f"median 0.20~ms"], src_rob, "distribution_summary.csv median_ms")
    C.add("robust.distribution", "quartiles, 90th/95th percentiles, maximum, zero and positive-state statistics of headroom",
          {k: D[k] for k in ("p25_ms", "p75_ms", "p90_ms", "p95_ms", "max_ms", "n_zero", "frac_zero", "pos_median_ms", "n_pos", "frac_le_mean")},
          [f"{ms(D['p25_ms'], 3)} and {ms(D['p75_ms'], 2)}~ms", f"{ms(D['p90_ms'], 1)} and {ms(D['p95_ms'], 1)}~ms", f"{ms(D['max_ms'], 1)}~ms",
           f"{D['n_zero']} states ({pct(D['frac_zero'])}\\%)", f"the median is {ms(D['pos_median_ms'], 2)}~ms", f"{pct(D['frac_le_mean'])}\\% of states lie at or below that mean"],
          src_rob, "distribution_summary.csv; quantiles via numpy 'linear' method")
    W = {w["name"]: w for w in N["weighting"]}
    eqw, eqr, eqk = W["equal-window"], W["equal-regime"], W["equal-workload (supplementary)"]
    for cid, w, key in (("robust.equal_window", eqw, "Equal-window"), ("robust.equal_regime", eqr, "Equal-regime"), ("robust.equal_workload", eqk, "Equal-workload")):
        C.add(cid, f"{key} mean headroom {ms(w['mean_ms'], 3)} ms with window-clustered 200,000-replicate CI", {"units": w["units"], "mean_ms": w["mean_ms"], "share": w["P"], "ci_ms": list(w["ci_ms"])},
              [f"{key} & {w['units']} {'windows' if w['units'] == 36 else 'regimes' if w['units'] == 5 else 'workloads'} & {ms(w['mean_ms'], 3)} & [{ms(w['ci_ms'][0], 3)}, {ms(w['ci_ms'][1], 3)}] & {ms(w['P'], 3)} \\\\"],
              f"{ROBUST}/weighting_sensitivity.csv; {ROBUST}/weighting_sensitivity_ci.csv", "equal-weight mean over units of the per-unit mean H_LAT")
    C.add("robust.aggregation_ratio", "abstract/discussion: equal-window 0.33 ms, equal-regime 0.76 ms, equal-workload 1.36 ms", [eqw["mean_ms"], eqr["mean_ms"], eqk["mean_ms"]],
          [f"{ms(eqk['mean_ms'], 2)}, {ms(eqr['mean_ms'], 2)}, and {ms(eqw['mean_ms'], 2)}~ms", f"from\n1.996~ms (state-weighted) to {ms(eqw['mean_ms'], 3)}~ms (equal-window)".replace("\n", " ")], src_rob, "see robust.equal_*")
    thr = {t["label"]: t for t in N["thresholds"]}
    for lab, tau in ((">0.1", "0.1"), (">0.25", "0.25"), (">0.5", "0.5"), (">1.0", "1"), (">2.0", "2"), (">5.0", "5")):
        t = thr[lab]
        C.add(f"robust.threshold_{tau}ms", f"states with H_LAT > {tau} ms: {t['n']} ({pct(t['share'])}%), in {t['windows']} windows, {t['in_code_kv']} in Azure-code KV 16,000", {k: t[k] for k in ("n", "share", "windows", "in_code_kv")},
              [f"$>{tau}$~ms & {t['n']} & {pct(t['share'])}\\% & {t['windows']} & {t['in_code_kv']} \\\\"], f"{ROBUST}/practical_thresholds.csv", "n_exceed, n_windows_with_exceedance; threshold tau + 1e-12 s guard")
    C.add("robust.thresholds_used", "practical thresholds epsilon in {0.1, 0.25, 0.5, 1, 2, 5} ms", list(rn.THRESHOLDS_MS), ["$\\epsilon\\in\\{0.1,0.25,0.5,1,2,5\\}$~ms"], "paper/performance_evaluation/scripts/robustness_numbers.py::THRESHOLDS_MS", "constant")
    Cn = N["conc"]
    C.add("robust.w11_headroom_share", "Azure-code window w11 carries 88.4% of total headroom with 36.5% of the states (263 of 720)", {"headroom_share": Cn["w11_headroom_share"], "state_share": Cn["w11_state_share"], "states": Cn["w11_states"]},
          [f"{pct(Cn['w11_state_share'])}\\%", f"{pct(Cn['w11_headroom_share'])}\\%", f"{Cn['w11_states']} of the 720 states", "88\\%"], src_rob, "composition_window.csv / composition_summary.json: share of sum(H_LAT) by window")
    C.add("robust.regime_headroom_share", "Azure-code KV 16,000 regime carries 96.8% of headroom and 59.9% of states; contributes 1.93 ms of the 2.00 ms mean",
          {"headroom_share": Cn["code_kv_headroom_share"], "state_share": Cn["code_kv_state_share"], "contrib_ms": Cn["code_kv_contrib_ms"]},
          [f"{pct(Cn['code_kv_state_share'])}\\% of the states and {pct(Cn['code_kv_headroom_share'])}\\% of the headroom", f"{ms(Cn['code_kv_contrib_ms'], 2)} of\nthe 2.00~ms".replace("\n", " ")], src_rob, "composition_regime.csv: share of sum(H_LAT) by regime")
    C.add("robust.concentration_other", "top-3 window share 95.3%; effective number of windows 1.27 (5.7 by state count); 205 w11 KV states average 6.16 ms; 128 of 129 states >5 ms and 140 of 162 states >2 ms in w11",
          {k: Cn[k] for k in ("top3_share", "eff_windows", "eff_windows_by_states", "w11_code_kv_states", "w11_code_kv_mean_ms", "n_gt5", "n_gt5_in_w11", "n_gt2", "n_gt2_in_w11")},
          [f"{pct(Cn['top3_share'])}\\%", f"{ms(Cn['eff_windows'], 2)} ({ms(Cn['eff_windows_by_states'], 1)} by state count)", f"{Cn['w11_code_kv_states']} states of that regime average {ms(Cn['w11_code_kv_mean_ms'], 2)}~ms",
           f"Of the {Cn['n_gt5']} states above 5~ms, {Cn['n_gt5_in_w11']} are in w11, as are {Cn['n_gt2_in_w11']} of the {Cn['n_gt2']}"], src_rob, "composition_window.csv / composition_regime.csv / composition_summary.json; effective windows = 1 / sum(p_w^2)")
    L = N["loo"]
    lw, lk = L["window"]["azure_2023_code:w11"], L["regime"]["azure_2023_code|kv_capacity|kv_16000"]
    C.add("robust.leave_one_out_w11", "omit Azure-code w11: 457 states, mean 0.365 ms, CI [0.158, 0.682], share 0.821", {k: lw[k] for k in ("n", "mean_ms", "P")} | {"ci_ms": list(lw["ci_ms"])},
          [f"Omit Azure-code window w11 & {lw['n']} states & {ms(lw['mean_ms'], 3)} & [{ms(lw['ci_ms'][0], 3)}, {ms(lw['ci_ms'][1], 3)}] & {ms(lw['P'], 3)} \\\\"], f"{ROBUST}/leave_one_window_out.csv; {ROBUST}/leave_one_window_out_remainder_ci.csv", "state-weighted mean of the remaining states")
    C.add("robust.leave_one_out_code_kv", "omit Azure-code KV 16,000: 289 states, mean 0.159 ms, CI [0.058, 0.234], share 0.664", {k: lk[k] for k in ("n", "mean_ms", "P")} | {"ci_ms": list(lk["ci_ms"])},
          [f"Omit Azure-code KV 16,000 & {lk['n']} states & {ms(lk['mean_ms'], 3)} & [{ms(lk['ci_ms'][0], 3)}, {ms(lk['ci_ms'][1], 3)}] & {ms(lk['P'], 3)} \\\\"], f"{ROBUST}/leave_one_regime_out.csv; {ROBUST}/leave_one_regime_out_remainder_ci.csv", "state-weighted mean of the remaining states")
    C.add("robust.leave_one_out_ranges", "all 36 + 5 omissions positive with CI excluding zero; other windows 2.00-2.26 ms, other regimes 2.02-2.39 ms",
          {"other_windows_range_ms": L["other_windows_range_ms"], "other_regimes_range_ms": L["other_regimes_range_ms"], "all_ci_exclude_zero": L["all_ci_exclude_zero"], "n_windows": L["n_windows"], "n_regimes": L["n_regimes"]},
          [f"between\n{ms(L['other_windows_range_ms'][0], 2)} and {ms(L['other_windows_range_ms'][1], 2)}~ms".replace("\n", " "), f"between {ms(L['other_regimes_range_ms'][0], 2)} and {ms(L['other_regimes_range_ms'][1], 2)}~ms", "all 41 omissions"],
          src_rob, "min/max of leave-one-out means excluding w11 / the Azure-code KV regime")
    assert L["all_ci_exclude_zero"] and L["n_windows"] + L["n_regimes"] == 41
    Dg = N["diag"]
    C.add("robust.interval_diagnostics", "10^6-replicate percentile CI [0.184, 3.543] ms; BCa [0.239, 4.049]; jackknife t [-1.36, 5.35], SE 1.65; MC sd 0.004 ms over 1000 seeds; 36% of replicates omit w11 (mean 0.36 vs 2.43 ms)",
          {k: Dg[k] for k in ("large_ci_ms", "bca_ci_ms", "jack_ci_ms", "jack_se_ms", "mc_low_sd_ms", "mc_seeds", "frac_without_w11", "mean_without_w11_ms", "mean_with_w11_ms")},
          [f"[{ms(Dg['large_ci_ms'][0], 3)}, {ms(Dg['large_ci_ms'][1], 3)}]~ms", f"[{ms(Dg['bca_ci_ms'][0], 3)}, {ms(Dg['bca_ci_ms'][1], 3)}]~ms", f"[$-{ms(-Dg['jack_ci_ms'][0], 2)}$, {ms(Dg['jack_ci_ms'][1], 2)}]~ms",
           f"({ms(Dg['jack_se_ms'], 2)}~ms)", f"{ms(Dg['mc_low_sd_ms'], 3)}~ms (standard deviation) across {Dg['mc_seeds']} seeds", f"{100 * Dg['frac_without_w11']:.0f}\\% of\nreplicates do not draw window w11".replace("\n", " "),
           f"{ms(Dg['mean_with_w11_ms'], 2)}~ms for those that do"], src_rob, "cluster_robustness.json: bootstrap 1e6, BCa, delete-one-window jackknife t, 1000 seed replications")

    # ------------------------------------------------------------------ vLLM probe
    ts = json.loads((VLLM_PROBE / "mechanism_summary.json").read_text())["trace_summary_by_treatment"]
    st = json.loads((VLLM_PROBE / "statistical_summary.json").read_text())["comparisons"]
    vr = json.loads(VLLM_RESULT.read_text())
    env = json.loads((EXP / "real_vllm_mechanism_validation_v1" / "runtime_environment.json").read_text())
    inst = json.loads((EXP / "real_vllm_mechanism_validation_v1" / "vllm_install.json").read_text())
    probe_cfg = json.loads((EXP / "real_vllm_mechanism_validation_v1" / "probe_config.json").read_text())
    src_v = rel(VLLM_PROBE / "mechanism_summary.json")
    assert "RTX 5060 Ti" in env["gpu"]["name"] and inst["versions"]["vllm"] == "0.27.1" and "Qwen2.5-0.5B-Instruct" in probe_cfg["model_path"]
    C.add("vllm.setup", "one RTX 5060 Ti GPU, vLLM 0.27.1, Qwen2.5-0.5B-Instruct", {"gpu": env["gpu"]["name"], "vllm": inst["versions"]["vllm"], "model": "Qwen2.5-0.5B-Instruct"},
          ["RTX 5060 Ti GPU, vLLM 0.27.1, Qwen2.5-0.5B-Instruct"], "experiments/real_vllm_mechanism_validation_v1/{runtime_environment,vllm_install,probe_config}.json", "gpu.name; versions.vllm; model_path")
    assert vr["prefill_decode"]["measured_runs"] == 40 and vr["native_budget_probe"]["measured_runs"] == 20
    C.add("vllm.run_counts", "40 full-versus-chunked runs and a 20-run native chunked-prefill budget probe", {"prefill_decode": 40, "native_budget": 20}, ["40 full-versus-chunked runs and\na 20-run native chunked-prefill budget probe".replace("\n", " ")],
          rel(VLLM_RESULT), "prefill_decode.measured_runs; native_budget_probe.measured_runs")
    t5, t4 = ts["T512"], ts["T4096"]
    C.add("vllm.trace_counts", "T512: 1,680 scheduler steps, 165 mixed prefill/decode steps, 194 partial-prefill items; T4096: 1,536, 55, 9", {"T512": [t5["scheduled_steps"], t5["mixed_prefill_decode_steps"], t5["partial_prefill_items"]], "T4096": [t4["scheduled_steps"], t4["mixed_prefill_decode_steps"], t4["partial_prefill_items"]]},
          [f"{comma(t5['scheduled_steps'])} scheduler steps, {t5['mixed_prefill_decode_steps']} mixed prefill/decode steps, and {t5['partial_prefill_items']} partial-prefill\nitems; T4096 produced {comma(t4['scheduled_steps'])}, {t4['mixed_prefill_decode_steps']}, and {t4['partial_prefill_items']}".replace("\n", " ")],
          src_v, "trace_summary_by_treatment.{T512,T4096}.{scheduled_steps,mixed_prefill_decode_steps,partial_prefill_items}")
    C.add("vllm.queue_lengths", "waiting queues reached 7 (T512) and 5 (T4096); running sequences reached 4 in both", {"waiting": [t5["max_waiting_after_schedule"], t4["max_waiting_after_schedule"]], "running": [t5["max_running_after_schedule"], t4["max_running_after_schedule"]]},
          [f"lengths of {t5['max_waiting_after_schedule']} and\n{t4['max_waiting_after_schedule']}, respectively, while running sequences reached {t5['max_running_after_schedule']} in both cases".replace("\n", " ")], src_v, "trace_summary_by_treatment.*.max_{waiting,running}_after_schedule")
    lo, hi = st["late_tight_low_late"], st["late_tight_high_late"]
    C.add("vllm.latency_differences", "low load: 30.6 ms late-TTFT difference, 16.3 ms prompt-heavy end-to-end; high load: smaller TTFT difference (2.5 ms, CI includes 0), 23.3 ms end-to-end",
          {"low_late_ttft_ms": 1e3 * lo["late_ttft_T4096_minus_T512_s_mean"], "low_hog_e2e_ms": 1e3 * lo["hog_e2e_T4096_minus_T512_s_mean"], "high_late_ttft_ms": 1e3 * hi["late_ttft_T4096_minus_T512_s_mean"],
           "high_late_ttft_ci_ms": [1e3 * x for x in hi["late_ttft_T4096_minus_T512_s_ci95"]], "high_hog_e2e_ms": 1e3 * hi["hog_e2e_T4096_minus_T512_s_mean"]},
          [f"a {abs(1e3 * lo['late_ttft_T4096_minus_T512_s_mean']):.1f}~ms difference in late time to first token", f"a {1e3 * lo['hog_e2e_T4096_minus_T512_s_mean']:.1f}~ms\nprompt-heavy end-to-end tradeoff".replace("\n", " "),
           f"a {1e3 * hi['hog_e2e_T4096_minus_T512_s_mean']:.1f}~ms prompt-heavy end-to-end"], rel(VLLM_PROBE / "statistical_summary.json"), "comparisons.<regime>.{late_ttft,hog_e2e}_T4096_minus_T512_s_mean")
    assert hi["late_ttft_T4096_minus_T512_s_ci95"][0] < 0 < hi["late_ttft_T4096_minus_T512_s_ci95"][1] and abs(hi["late_ttft_T4096_minus_T512_s_mean"]) < abs(lo["late_ttft_T4096_minus_T512_s_mean"])
    return C.items


# ------------------------------------------------------------------------------------------ manuscript location
def _pattern(snippet: str):
    return re.compile(r"\s+".join(re.escape(tok) for tok in snippet.split()))


def locate(tex: str, snippet: str):
    """Line number and human-readable location (numbered section / table / figure, or Abstract) of the first occurrence."""
    m = _pattern(snippet).search(tex)
    if not m:
        return None
    head = tex[:m.start()]
    line = head.count("\n") + 1
    # inside a float?  (the float's number is its order of appearance in the source)
    for kind in ("table", "figure"):
        opens = [x.start() for x in re.finditer(r"\\begin\{" + kind + r"\*?\}", head)]
        if opens and not re.search(r"\\end\{" + kind + r"\*?\}", head[opens[-1]:]):
            return {"section": f"{kind.capitalize()} {len(opens)}", "line": line}
    ab = head.rfind("\\begin{abstract}")
    if ab != -1 and "\\end{abstract}" not in head[ab:]:
        return {"section": "Abstract", "line": line}
    secs = list(re.finditer(r"\\section\{([^}]*)\}", head))
    if not secs:
        return {"section": "front matter", "line": line}
    n = len(secs)
    subs = [x for x in re.finditer(r"\\subsection\{([^}]*)\}", head[secs[-1].start():])]
    sec = f"Section {n} ({secs[-1].group(1)})"
    return {"section": sec + (f", subsection {n}.{len(subs)} ({subs[-1].group(1)})" if subs else ""), "line": line}


def evaluate(claims, tex):
    problems = []
    for c in claims:
        locs = []
        for s in c["manuscript_text"]:
            loc = locate(tex, s)
            if loc is None:
                problems.append(f"{c['id']}: manuscript text not found: {s!r}")
            else:
                locs.append({**loc, "text": s})
        c["manuscript_location"] = locs
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    tex = TEX.read_text()
    claims = build_claims()
    problems = evaluate(claims, tex)
    digest = hashlib.sha256(tex.encode()).hexdigest()
    doc = {
        "schema": "peva_final_claim_manifest_v1",
        "purpose": "Provenance/verification metadata for the manuscript's quantitative claims; contains no independent scientific data. Values are recomputed from the canonical artifacts by scripts/build_claim_manifest.py.",
        "manuscript": "paper/performance_evaluation/main.tex",
        "manuscript_sha256_at_generation": digest,
        "n_claims": len(claims),
        "claims": claims,
    }
    if a.check:
        old = json.loads(MANIFEST.read_text())
        by_id = {c["id"]: c for c in old["claims"]}
        for c in claims:
            o = by_id.get(c["id"])
            if o is None:
                problems.append(f"{c['id']}: missing from committed manifest")
            elif json.loads(json.dumps(o["value"])) != json.loads(json.dumps(c["value"])):
                problems.append(f"{c['id']}: value differs from committed manifest")
        problems += [f"{i}: in committed manifest but no longer produced" for i in by_id.keys() - {c["id"] for c in claims}]
        if old["manuscript_sha256_at_generation"] != digest:
            print("note: main.tex changed since the manifest was generated (all claims above were still verified against the current text)")
        for p in problems:
            print("FAIL", p)
        print(f"{len(claims)} claims checked, {len(problems)} problem(s)")
        sys.exit(1 if problems else 0)
    if problems:
        for p in problems:
            print("FAIL", p)
        sys.exit(1)
    MANIFEST.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {MANIFEST} ({len(claims)} claims)")


if __name__ == "__main__":
    main()
