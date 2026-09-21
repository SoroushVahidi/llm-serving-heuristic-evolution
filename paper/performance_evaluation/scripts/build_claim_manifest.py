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
import math
import re
import statistics
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

_spec2 = importlib.util.spec_from_file_location("reference_policy_numbers", HERE / "reference_policy_numbers.py")
rp = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(rp)

PHASE_A = EXP / "industry_realism_action_opportunity_phase_a_v1"
PHASE_B = EXP / "industry_realism_action_opportunity_phase_b_v2"
FROZEN = "experiments/fresh_production_latency_headroom_confirmatory_v1"
ROBUST = "experiments/fresh_production_latency_headroom_confirmatory_v1_robustness"
VLLM_PROBE = EXP / "real_vllm_mechanism_validation_v1" / "native_vllm_chunk_budget_semantics_probe_v1"
VLLM_RESULT = EXP / "real_vllm_pressure_action_validation_v1" / "REAL_VLLM_VALIDATION_RESULT_V1.json"
RESERVE = EXP / "reference_reserve_sensitivity_v1"
PHASE_D = EXP / "industry_realism_causal_headroom_phase_d_v1"


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
          ["Arrival scaling through $8\\times$ produced no disagreement state", "up to eight times"], src_wc, "axis == arrival_pressure: max arrival_multiplier, sum of true_canonical_disagreement_states")

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
    onset = {v: next(x for x in axis_rows if x["source_dataset"] == "azure_2023_code" and x["axis"] == "active_sequence_capacity" and int(float(x["axis_value"])) == v) for v in (8, 4)}
    o8, o4 = onset[8], onset[4]
    assert (int(o8["true_canonical_disagreement_states"]), int(o8["sbs_decision_states"])) == (1, 99992) and (int(o4["true_canonical_disagreement_states"]), int(o4["sbs_decision_states"])) == (100, 100127)
    assert f"{100 * float(o8['disagreement_rate']):.3f}" == "0.001" and f"{100 * float(o4['disagreement_rate']):.4f}" == "0.0999"
    C.add("pressure.azure_code_onset", "Azure code active-sequence cap: 1 of 99,992 SBS decision states at the onset cap 8 (0.001%) versus 100 of 100,127 (0.0999%) at cap 4",
          {"cap8": [1, 99992, float(o8["disagreement_rate"])], "cap4": [100, 100127, float(o4["disagreement_rate"])]},
          ["disagreement is a\nsingle state among 99,992 SBS decision states (0.001\\%)".replace("\n", " "), "100 of 100,127\n(0.0999\\%) at a cap of 4".replace("\n", " ")],
          rel(PHASE_B / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv"), "true_canonical_disagreement_states / sbs_decision_states for azure_2023_code, axis=active_sequence_capacity, axis_value in {8, 4}")
    C.add("figure2.pressure_conditions", "Figure 2 pressure-condition prevalences (read from the artifact, not typed into the plot script)", fig2, ["figures/pe_disagreement_rates.pdf"],
          rel(PHASE_B / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv"), "disagreement_rate = true_canonical_disagreement_states / sbs_decision_states for the four plotted rows")

    # ------------------------------------------------------------------ fresh causal (pre-specified)
    P = N["primary"]
    res = json.loads((ROOT / FROZEN / "FRESH_LATENCY_CAUSAL_RESULT_V1.json").read_text())["primary"]
    boot = json.loads((ROOT / FROZEN / "FRESH_LATENCY_BOOTSTRAP_V1.json").read_text())
    src_res = f"{FROZEN}/FRESH_LATENCY_CAUSAL_RESULT_V1.json"
    src_state = f"{FROZEN}/FRESH_LATENCY_STATE_LEVEL_V1.csv"
    assert P["n"] == res["states"] == 720 and P["beneficial"] == res["beneficial_states"] == 590 and P["guarded_beneficial"] == 589
    # support transfer: the pressure matrix repeated on fresh windows (recomputed from the frozen support conditions)
    src_sup = f"{FROZEN}/FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv"
    sup = [r for r in csv_rows(ROOT / src_sup) if r["validity_class"].startswith("VALID")]
    all_sup = csv_rows(ROOT / src_sup)
    onset = {}
    for wlname in ("azure_2023_code", "azure_2023_conv", "burstgpt"):
        for axis in ("active_sequence_capacity", "kv_capacity", "arrival_pressure"):
            rows = [r for r in sup if r["source_dataset"] == wlname and r["axis"] == axis and int(r["true_canonical_disagreement_states"]) > 0]
            onset[f"{wlname}:{axis}"] = None if not rows else min(rows, key=lambda r: float(r["pressure_order"]))["axis_value"]
    burst = [r for r in sup if r["source_dataset"] == "burstgpt"]
    assert len(all_sup) == 1080 and len(burst) == 360 and {r["source_dataset"] for r in burst} == {"burstgpt"}
    assert sum(int(r["true_canonical_disagreement_states"]) for r in burst) == 0
    assert onset["azure_2023_code:kv_capacity"] == "16000" and onset["azure_2023_conv:kv_capacity"] == "16000"
    assert onset["azure_2023_code:active_sequence_capacity"] == "8" and onset["azure_2023_conv:active_sequence_capacity"] == "4"
    assert all(onset[f"{w}:arrival_pressure"] is None for w in ("azure_2023_code", "azure_2023_conv", "burstgpt"))
    C.add("fresh.support_transfer", "pressure matrix repeated on 60 fresh windows: Azure onsets reproduce (KV 16,000; active cap 8 code, 4 conversation), arrival scaling action-null, BurstGPT has 0 disagreement states in all 360 (valid) conditions",
          {"conditions": len(all_sup), "onset_by_workload_axis": onset, "burstgpt_valid_conditions": len(burst), "burstgpt_disagreement_states": 0},
          ["repeated on 60 fresh windows", "reproduced for both Azure traces", "All 360 fresh BurstGPT conditions were valid, none produced a disagreement state, and none had a binding capacity constraint"],
          src_sup, "true_canonical_disagreement_states summed over valid conditions per workload/axis; onset = lowest-pressure setting with any disagreement")
    # ------------------------------------------------------------------ reference policy, load range, overlays, secondary outcomes
    R = rp.compute()
    B, AF, AO, NQ, OV, RF, ST, RL, WW = (R[k] for k in ("burst", "arrival_fresh", "arrival_original", "native_mean_queue", "overlay", "reference", "structure", "relative", "whole_window"))
    src_sup2 = f"{FROZEN}/FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv"
    assert (B["conditions"], B["valid"], B["windows"], B["disagreement_states"], B["binding_states"], B["max_queue_length"]) == (360, 360, 20, 0, 0, 3)
    C.add("burstgpt.fresh_mechanism", "BurstGPT fresh support: 360/360 conditions valid, 0 disagreement, 0 binding, max queue length 3 (caps never reached)", B,
          ["never queued more than three requests", "even the tightest caps (four active sequences, 8,000 KV tokens) were never reached"], src_sup2,
          "rows with source_dataset=burstgpt: validity_class, true_canonical_disagreement_states, active/kv binding states, max_queue_length")
    assert AO["peak_kv_utilization"] < 0.0071 and AO["max_workload_mean_queue"] < 0.09 and AO["max_active_sequences"] == 46 and AO["binding_states"] == 0 and AO["disagreement_states"] == 0
    C.add("arrival.original_light_load", "arrival scaling through 8x on the original windows: peak KV utilization < 0.71% of default capacity, <= 46 active sequences (limit 512), workload mean queue < 0.09, no binding",
          AO, ["peak KV utilization stayed below 0.71\\% of the default capacity", "at most 46 sequences were active against a limit of 512", "the mean queue length per workload stayed below 0.09"],
          rel(PHASE_B / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv"), "rows with axis containing 'arrival': max_kv_utilization, mean_queue_length, max_active_sequences, binding states")
    assert AF["peak_kv_utilization"] < 0.0045 and AF["max_workload_mean_queue"] < 0.091 and AF["binding_states"] == 0 and AF["disagreement_states"] == 0 and AF["default_active_cap"] == 512 and AF["default_kv_tokens"] == 8000000
    C.add("arrival.fresh_light_load", "arrival scaling through 8x on the fresh windows: peak KV utilization < 0.45%, no binding, no disagreement (default 512 sequences, 8,000,000 KV tokens)",
          AF, ["peak KV utilization below 0.45\\%", "arrival-rate scaling again\nproduced none".replace("\n", " ")], src_sup2, "valid arrival_pressure rows: max_kv_utilization, mean_queue_length, binding states, true_canonical_disagreement_states")
    assert [round(NQ[k], 4) for k in ("azure_2023_code", "azure_2023_conv", "burstgpt")] == [0.04, 0.0087, 0.0091]
    C.add("native.mean_queue_length", "native replay mean queue lengths 0.040 / 0.0087 / 0.0091 (Azure code / conversation / BurstGPT)", NQ,
          ["mean queue lengths were 0.040, 0.0087, and 0.0091 requests"], src_a, "mean_queue_length of PHASE_A_WORKLOAD_SUMMARY_V1.csv")
    C.add("overlay.faithful_view", "faithful view: deadline = arrival + 1000 s, uniform priority 1.0, single class, predicted output length = true length, one GPU (512 sequences, 512-token batch, 8,000,000 KV tokens), 1 ms step",
          OV, ["deadline of arrival plus 1000~s", "a uniform priority of 1.0, and a single class", "predicted output length equal to the true length", "512 active-sequence slots", "8,000,000 KV tokens by default", "with a 1~ms step"],
          rel(PHASE_A / "PHASE_A_REPLAY_SEMANTICS_V1.json"), "faithful_view.slo_deadline/priority/class_id/predicted_output_tokens/capacity_assignment/service_rate_assumptions")
    assert abs(RL["max_mean_reference_latency_s"] - 0.36) < 0.005
    C.add("overlay.max_mean_latency", "largest mean continuation latency L_SBS(s) over the 720 states is 0.36 s (far below the 1000 s deadline)", RL["max_mean_reference_latency_s"],
          ["the largest mean continuation latency in any causal population is 0.36~s"], "experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv",
          "max of mean_ref_latency (SBS-reference branch, corrected derivative)")
    # SBS selection provenance: best single policy by mean ANWG over the 240 joint-benchmark scenarios
    j = csv_rows(EXP / "joint240_same_distribution_adaptive_exploitability_v1" / "per_scenario_oof_results.csv")
    pol6 = ["full_prefill", "chunked_prefill_small", "estimated_service_time_first", "weighted_fair_share", "least_laxity_first", "kv_constrained_online"]
    means = {q: sum(float(r[q]) for r in j) / len(j) for q in pol6}
    assert len(j) == 240 and max(means, key=means.get) == "kv_constrained_online"
    C.add("reference.sbs_selection", "SBS = kv_constrained_online = best single policy (highest mean goodput) over 240 joint-benchmark scenarios; reserve 0.82, urgency slack 0.25 s", {"scenarios": len(j), "mean_anwg_by_policy": means, "target_kv_utilization": 0.82, "urgent_laxity_seconds": 0.25},
          ["highest mean goodput among the six policies on a separate earlier benchmark of 240 multi-mechanism scenarios", "reserve of 0.82 of the configured capacity", "slack under 0.25~s"],
          "experiments/joint240_same_distribution_adaptive_exploitability_v1/per_scenario_oof_results.csv; src/llmserveopt/policies/kv_constrained_online.py", "mean over scenarios of each policy column; defaults of KVConstrainedOnlinePolicy")
    assert (RF["all_five_differ"], RF["kv_states"], RF["kv_without_kv_binding"], RF["all_five_in_kv_regimes"], RF["all_five_single_alternative"]) == (512, 524, 414, 511, 445)
    assert abs(RF["all_five_headroom_share"] - 0.948) < 5e-4 and abs(RF["kv_nobind_headroom_share_of_kv"] - 0.896) < 5e-4
    C.add("reference.all_five_differ", "512/720 states have all five non-SBS policies differing; they carry 94.8% of headroom; 511 in the KV regimes; 445 with a single alternative action", 
          {k: RF[k] for k in ("all_five_differ", "all_five_headroom_share", "all_five_in_kv_regimes", "all_five_single_alternative")},
          ["In 512 of the 720 disagreement states (71\\%) all five other policies differ from the SBS", "these states carry 94.8\\% of the total headroom", "511 of them lie in the two KV-capacity regimes", "in 445 the five policies agree on a single alternative action",
           "95\\% of the headroom lies where all five other policies differ from the reference"],
          f"{FROZEN}/FRESH_ELIGIBLE_DISAGREEMENT_STATES_V1.csv; experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv", "p6_policies_with_non_sbs_action (count of policies), n_unique_non_sbs_actions, oracle_headroom")
    C.add("reference.kv_without_physical_binding", "524 KV-regime disagreement states, 414 without physical KV binding (waiting prompt tokens <= free KV tokens), holding 89.6% of the KV regimes' headroom",
          {k: RF[k] for k in ("kv_states", "kv_without_kv_binding", "kv_nobind_headroom_share_of_kv")},
          ["The KV-capacity regimes contain 524 disagreement states, of which 414 occur without physical KV binding", "hold 89.6\\% of those regimes' headroom"],
          f"{FROZEN}/FRESH_ELIGIBLE_DISAGREEMENT_STATES_V1.csv", "kv_binding = kv_capacity_binding_or_over_requested (waiting_prompt_token_mass > total_free_kv_tokens; scripts/industry_realism_action_opportunity_phase_a_v1.py)")
    assert RF["differs"]["estimated_service_time_first"] == RF["differs"]["weighted_fair_share"] == 519 and abs(RF["mean_distinct_alternatives"] - 1.15) < 0.005 and RF["single_alternative_states"] == 610
    C.add("portfolio.functional_diversity", "ESTF and WFS differ from SBS in the same 519 states; on average 1.15 distinct alternative actions per disagreement state; 610 states have exactly one", {k: RF[k] for k in ("differs", "mean_distinct_alternatives", "single_alternative_states")},
          ["exactly the same 519 states as ESTF", "1.15 distinct alternative actions", "610 states have exactly one"], f"{FROZEN}/FRESH_ELIGIBLE_DISAGREEMENT_STATES_V1.csv", "p6_policies_with_non_sbs_action; n_unique_non_sbs_actions")
    assert (ST["beneficial"], ST["mixed_beneficial_and_harmful"], ST["all_harmful"], ST["zero_headroom_other"], ST["positive"], ST["negative"], ST["zero"], ST["branches"]) == (590, 22, 102, 28, 660, 141, 30, 831)
    C.add("secondary.state_and_action_structure", "pre-specified secondary outcomes: 590 beneficial (22 mixed), 102 all-harmful (14.2%), 28 zero-headroom; 831 alternatives: 660 reduce, 141 increase, 30 unchanged", ST,
          ["Some alternative reduces latency & 590 (81.9\\%)", "of which some alternative also increases it & 22", "Every alternative increases latency & 102 (14.2\\%)", "Zero headroom, not every alternative harmful & 28 (3.9\\%)",
           "Reduce latency & 660 (79.4\\%)", "Increase latency & 141 (17.0\\%)", "Leave latency unchanged & 30 (3.6\\%)", "in 14.2\\% every alternative is worse than the reference"],
          f"{FROZEN}/FRESH_LATENCY_ACTION_LEVEL_V1.csv", "per-state max/min of a_lat over counterfactual branches; per-branch sign of a_lat")
    assert abs(RL["median"] - 0.0021) < 5e-5 and abs(RL["p90"] - 0.113) < 5e-4 and abs(RL["kv_code_p90"] - 0.161) < 5e-4 and (RL["gt1"], RL["gt5"], RL["gt10"], RL["kv_code_gt10"], RL["kv_code_states"]) == (211, 133, 92, 92, 431)
    C.add("secondary.relative_headroom", "H_REL = H_LAT / L_SBS: median 0.21%, p90 11.3% (Azure-code KV-16,000: 16.1%), 211/133/92 states above 1/5/10%, all 92 above 10% in the Azure-code KV-16,000 regime (21.3% of its 431 states)", RL,
          ["Median & 0.21\\%", "90th percentile & 11.3\\%", "90th percentile, Azure-code KV 16,000 only & 16.1\\%", "States above 1\\% & 211 (29.3\\%)", "States above 5\\% & 133 (18.5\\%)", "States above 10\\% & 92 (12.8\\%)", "92 states (12.8\\%) exceed 10\\% of the reference's mean latency", "where they are 21.3\\% of the states"],
          "experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv; " + f"{FROZEN}/FRESH_LATENCY_CAUSAL_RESULT_V1.json",
          "oracle_headroom / mean_ref_latency (SBS reference); cross-checked against action-level L_SBS = mean_latency + a_lat and the pre-specified per-regime mean_relative_headroom")
    PS = R["prespec"]
    C.add("prespec.protocol_timeline_and_cluster_correction", "protocol committed 2026-09-19 (b4e6c60), results committed ~2 h 14 min later (b196c3e); first bootstrap used 26 index clusters, corrected to 36 source windows", PS,
          ["on 19 September 2026", "committed 2 h 14 min later", "index shared across workloads (26 clusters)", "specified source window (36 clusters)"],
          f"{FROZEN}/PREREGISTRATION_V1.json (git history); experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv",
          "commit timestamps of the protocol-freeze and result commits; distinct window_index vs distinct (source_dataset, window_index) among the 720 states")
    assert WW["windows"] == 60 and WW["chunked_slower_windows"] == 60 and abs(WW["median"] - 0.142) < 5e-4 and WW["policies_in_faithful_view"] == ["chunked_prefill_small", "full_prefill"]
    C.add("vbs.whole_window_illustration", "60 native windows: small-chunk prefill has higher whole-window mean latency than full prefill in 60/60 (median +14.2%); only these two policies were run whole-window in the faithful view", WW,
          ["small-chunk prefill has a higher whole-window mean latency than full prefill in all 60 (median 14.2\\%)"], "experiments/public_trace_replay_v1/layer3_checkpoint.jsonl", "faithful-view rows: mean_latency of chunked_prefill_small vs full_prefill per window")
    C.add("fresh.states", "720 fresh disagreement states", 720, ["720 fresh disagreement states"], src_res, "primary.states (== rows of FRESH_LATENCY_STATE_LEVEL_V1.csv)")
    C.add("fresh.beneficial_preregistered", "590 of 720 states have positive one-step latency headroom (strict pre-specified criterion), 81.9%", {"beneficial": 590, "share": 590 / 720},
          ["590/720=0.8194\\ (81.9\\%)", "590 (81.9\\%)"], src_res, "primary.beneficial_states; share = beneficial / states")
    C.add("fresh.beneficial_guarded", "589 of 720 states exceed the 1e-12 s floating-point guard, 81.8%", {"guarded": 589, "share": 589 / 720},
          ["589/720 (81.8\\%)"], src_state, "count of oracle_headroom > 1e-12 s")
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
    C.add("robust.median", "median headroom 0.197 ms", D["median_ms"], [f"{ms(D['median_ms'], 3)}~ms", f"median headroom is {ms(D['median_ms'], 3)}~ms", f"median is {ms(D['pos_median_ms'], 2)}~ms"], src_rob, "distribution_summary.csv median_ms")
    C.add("robust.distribution", "quartiles, 90th/95th percentiles, maximum, zero and positive-state statistics of headroom",
          {k: D[k] for k in ("p25_ms", "p75_ms", "p90_ms", "p95_ms", "max_ms", "n_zero", "frac_zero", "pos_median_ms", "n_pos", "frac_le_mean")},
          [f"{ms(D['p90_ms'], 1)} and {ms(D['p95_ms'], 1)}~ms", f"{ms(D['max_ms'], 1)}~ms",
           f"{D['n_zero']} states ({pct(D['frac_zero'])}\\%)", f"the median is {ms(D['pos_median_ms'], 2)}~ms", f"{pct(D['frac_le_mean'])}\\% of states lie at or below that mean"],
          src_rob, "distribution_summary.csv; quantiles via numpy 'linear' method")
    W = {w["name"]: w for w in N["weighting"]}
    eqw, eqr, eqk = W["equal-window"], W["equal-regime"], W["equal-workload (supplementary)"]
    for cid, w, key in (("robust.equal_window", eqw, "Equal-window"), ("robust.equal_regime", eqr, "Equal-regime"), ("robust.equal_workload", eqk, "Equal-workload")):
        C.add(cid, f"{key} mean headroom {ms(w['mean_ms'], 3)} ms with window-clustered 200,000-replicate CI", {"units": w["units"], "mean_ms": w["mean_ms"], "share": w["P"], "ci_ms": list(w["ci_ms"])},
              [f"{key} & {w['units']} {'windows' if w['units'] == 36 else 'regimes' if w['units'] == 5 else 'workloads'} & {ms(w['mean_ms'], 3)} & [{ms(w['ci_ms'][0], 3)}, {ms(w['ci_ms'][1], 3)}] & {ms(w['P'], 3)} \\\\"],
              f"{ROBUST}/weighting_sensitivity.csv; {ROBUST}/weighting_sensitivity_ci.csv", "equal-weight mean over units of the per-unit mean H_LAT")
    C.add("robust.aggregation_ratio", "abstract/discussion: equal-window 0.33 ms, equal-regime 0.76 ms, equal-workload 1.36 ms", [eqw["mean_ms"], eqr["mean_ms"], eqk["mean_ms"]],
          [f"falls to {ms(eqw['mean_ms'], 2)}~ms when windows receive equal weight", f"from\n1.996~ms (state-weighted) to {ms(eqw['mean_ms'], 3)}~ms (equal-window)".replace("\n", " ")], src_rob, "see robust.equal_*")
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
    assert vr["direct_simulator_family_b_reversal"] == "NO_GO"
    C.add("vllm.direct_comparison", "direct full-versus-chunked comparison did not reproduce the simulator's predicted class reversal (verdict NO_GO)", {"verdict": vr["direct_simulator_family_b_reversal"]},
          ["vLLM did not reproduce this reversal"], rel(VLLM_RESULT), "direct_simulator_family_b_reversal")
    add_reserve_claims(C)
    add_design_claims(C)
    return C.items


def _sig3(x: float) -> str:
    """Three significant digits with fixed decimals (0.04012 -> 0.0401, 0.024006 -> 0.0240, 0.006733 -> 0.00673)."""
    return f"{x:.{max(0, 2 - math.floor(math.log10(x)))}f}"


def _sci(x: float) -> str:
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def add_reserve_claims(C):
    """Reference-reserve sensitivity (post hoc) and its denominator completion, recomputed from the preserved raw artifacts."""
    RES = {"0.82": "082", "0.90": "090", "1.00": "100"}
    src_states = "experiments/reference_reserve_sensitivity_v1/results_v1/reserve_{}/disagreement_states.csv"
    src_counts = "experiments/reference_reserve_sensitivity_v1/denominator_completion_v1/counts_reserve_{}.csv"
    R = {}
    for r, d in RES.items():
        st = csv_rows(RESERVE / "results_v1" / f"reserve_{d}" / "disagreement_states.csv")
        cn = csv_rows(RESERVE / "denominator_completion_v1" / f"counts_reserve_{d}.csv")
        H = [float(x["oracle_headroom"]) * 1000.0 for x in st]      # simulated ms
        dec = sum(int(x["decision_states"]) for x in cn)
        assert len(cn) == 100 and sum(int(x["disagreement_states"]) for x in cn) == len(st)
        ben = sum(int(x["beneficial_opportunity"]) for x in st)
        mass = math.fsum(H)
        by_reg = {}
        for x, h in zip(st, H):
            g = by_reg.setdefault((x["source_dataset"], x["condition_id"], x["axis"]), {"states": 0, "beneficial": 0, "mass": []})
            g["states"] += 1; g["beneficial"] += int(x["beneficial_opportunity"]); g["mass"].append(h)
        dec_reg = {}
        for x in cn:
            k = (x["source_dataset"], x["condition_id"], x["axis"])
            dec_reg[k] = dec_reg.get(k, 0) + int(x["decision_states"])
        boot = json.loads((RESERVE / "results_v1" / f"reserve_{d}" / "bootstrap_summary.json").read_text())
        R[r] = {"decisions": dec, "states": len(st), "beneficial": ben, "P_D_pct": 100 * len(st) / dec, "P_B_given_D_pct": 100 * ben / len(st),
                "mean_H_ms": mass / len(st), "mass_ms": mass, "BDR_pct": 100 * ben / dec, "OWH_ms_per_decision": mass / dec,
                "all_harmful": sum(int(x["all_alternatives_harmful"]) for x in st), "by_regime": by_reg, "dec_regime": dec_reg, "boot": boot}
    a, b, c = R["0.82"], R["0.90"], R["1.00"]
    S = json.loads((RESERVE / "denominator_completion_v1" / "DENOMINATOR_SUMMARY_V1.json").read_text())
    assert S["gates"]["DENOMINATOR_TRAJECTORY_INTEGRITY_GATE"] == "PASS" and S["gates"]["DENOMINATOR_REPRODUCTION_GATE"]["status"] == "PASS" and S["status"] == "DENOMINATOR_COMPLETION_SUCCESS"
    for r in R:  # the denominator-summary derivations agree with the independent recomputation from the raw files
        assert R[r]["decisions"] == S["reserves"][r]["total_reference_decision_states"] and R[r]["states"] == S["reserves"][r]["disagreement_states"]
        assert abs(R[r]["OWH_ms_per_decision"] - S["reserves"][r]["opportunity_weighted_headroom_sim_ms_per_decision"]) < 1e-12
    src = "; ".join([src_states.format(d) for d in RES.values()] + [src_counts.format(d) for d in RES.values()]
                     + ["experiments/reference_reserve_sensitivity_v1/denominator_completion_v1/DENOMINATOR_SUMMARY_V1.json"])
    fmt3 = lambda n: f"{n:,}"

    # denominators
    spread = 100 * (max(x["decisions"] for x in R.values()) - min(x["decisions"] for x in R.values())) / max(x["decisions"] for x in R.values())
    assert spread < 0.006
    C.add("reserve.decision_denominators", "total reference decision states 1,470,515 / 1,470,447 / 1,470,427 at reserves 0.82 / 0.90 / 1.00; spread < 0.006%",
          {"decisions": {r: R[r]["decisions"] for r in R}, "spread_pct": spread},
          [f"{fmt3(a['decisions'])}, {fmt3(b['decisions'])}, and {fmt3(c['decisions'])}", "under 0.006\\%"], src, "sum of decision_states over the 100 scenarios per reserve; (max-min)/max")
    # Table 7 rows
    rows = []
    for r in R:
        x = R[r]
        rows.append(f"{r} & {x['states']} & {_sig3(x['P_D_pct'])} & {x['P_B_given_D_pct']:.1f} & {x['mean_H_ms']:.3f} & {_sig3(x['BDR_pct'])} & {_sci(x['OWH_ms_per_decision'])}")
    C.add("reserve.table7_rows", "Table 7: states, P(D), P(B|D), mean H_LAT, beneficial-decision rate and opportunity-weighted headroom per reserve",
          {r: {k: R[r][k] for k in ("states", "beneficial", "P_D_pct", "P_B_given_D_pct", "mean_H_ms", "BDR_pct", "OWH_ms_per_decision")} for r in R}, rows, src,
          "P(D)=states/decisions; P(B|D)=beneficial/states; mean=fsum(H)/states; BDR=beneficial/decisions; OWH=fsum(H)/decisions (simulated ms)")
    # counts and gates
    C.add("reserve.counts_and_gates", "disagreement 720/466/198 and beneficial 590/353/99 at 0.82/0.90/1.00; the denominator completion passed both integrity gates, replayed 100 scenarios per reserve and reproduced the counts",
          {"states": {r: R[r]["states"] for r in R}, "beneficial": {r: R[r]["beneficial"] for r in R}, "scenarios_per_reserve": 100, "gates": {"trajectory": S["gates"]["DENOMINATOR_TRAJECTORY_INTEGRITY_GATE"], "reproduction_0.82": S["gates"]["DENOMINATOR_REPRODUCTION_GATE"]["status"]}},
          [f"reproduced the disagreement\ncounts ({a['states']}, {b['states']}, {c['states']})".replace("\n", " "), f"({b['states']} and {b['beneficial']}\nstates versus {a['states']} and {a['beneficial']})".replace("\n", " "), "replays the same 100 fresh scenarios"], src,
          "counts: rows of disagreement_states.csv and beneficial_opportunity flags; gates: DENOMINATOR_SUMMARY_V1.json")
    # normalized ratios
    pdr, bdr = 100 * b["P_D_pct"] / a["P_D_pct"], 100 * b["BDR_pct"] / a["BDR_pct"]
    owh90, owh100 = 100 * b["OWH_ms_per_decision"] / a["OWH_ms_per_decision"], 100 * c["OWH_ms_per_decision"] / a["OWH_ms_per_decision"]
    C.add("reserve.normalized_ratios", "reserve 0.90 vs 0.82: P(D) x0.647, beneficial-decision rate x0.598, opportunity-weighted headroom x0.980; reserve 1.00 opportunity-weighted headroom x0.0164",
          {"P_D_ratio_090": pdr / 100, "BDR_ratio_090": bdr / 100, "OWH_ratio_090": owh90 / 100, "OWH_ratio_100": owh100 / 100, "P_D_ratio_100": 100 * c["P_D_pct"] / a["P_D_pct"] / 100, "BDR_ratio_100": 100 * c["BDR_pct"] / a["BDR_pct"] / 100},
          [f"fall to {pdr:.0f}\\% and {bdr:.0f}\\% of their 0.82 values", f"keeps {owh90:.0f}\\%", f"leaving {owh100:.1f}\\% of the 0.82"], src, "ratios of the recomputed rates to their reserve-0.82 values")
    C.add("reserve.conditional_mean_caveat", "reserve 0.90 conditional mean H_LAT rises to 3.02 ms (0.82: 2.00 ms) while opportunity-weighted headroom is 98% of the 0.82 value",
          {"mean_H_ms": {r: R[r]["mean_H_ms"] for r in R}, "OWH_ratio_090": owh90 / 100}, [f"rises to {b['mean_H_ms']:.2f}~ms", f"({b['mean_H_ms']:.2f}~ms) is not more opportunity"], src, "mean = fsum(H)/states; OWH ratio as above")
    # KV regimes and active-cap invariance
    kc, kv_c = ("azure_2023_code", "kv_16000", "kv_capacity"), ("azure_2023_conv", "kv_16000", "kv_capacity")
    g = lambda r, k: R[r]["by_regime"].get(k, {"states": 0, "beneficial": 0, "mass": []})
    mean_kc = {r: (math.fsum(g(r, kc)["mass"]) / g(r, kc)["states"]) if g(r, kc)["states"] else 0.0 for r in R}
    assert [g(r, kc)["states"] for r in R] == [431, 224, 2] and [g(r, kv_c)["states"] for r in R] == [93, 46, 0] and g("1.00", kc)["beneficial"] == 0 and g("1.00", kv_c)["beneficial"] == 0
    C.add("reserve.kv_regimes", "Azure-code KV 16,000 disagreement states 431 / 224 / 2 (mean 3.23 / 6.16 / 0 ms; none beneficial at 1.00); Azure-conversation KV 16,000 93 / 46 / 0",
          {"code_kv_states": [g(r, kc)["states"] for r in R], "code_kv_mean_ms": mean_kc, "code_kv_beneficial": [g(r, kc)["beneficial"] for r in R], "conv_kv_states": [g(r, kv_c)["states"] for r in R]},
          [f"Azure code {g('0.82', kc)['states']} to {g('0.90', kc)['states']} states, mean {mean_kc['0.82']:.2f} to {mean_kc['0.90']:.2f}~ms", "two Azure-code states (neither beneficial)", "none in Azure conversation"], src,
          "per-regime rows of disagreement_states.csv (source_dataset, condition_id, axis)")
    act = [("azure_2023_code", "active_8", "active_sequence_capacity"), ("azure_2023_code", "active_4", "active_sequence_capacity"), ("azure_2023_conv", "active_4", "active_sequence_capacity")]
    inv = {}
    for k in act:
        tup = {r: (R[r]["dec_regime"][k], g(r, k)["states"], g(r, k)["beneficial"], round(math.fsum(g(r, k)["mass"]), 9)) for r in R}
        assert len(set(tup.values())) == 1, (k, tup)
        inv["/".join(k[:2])] = list(tup["0.82"])
    kv_mass_100 = math.fsum(g("1.00", kc)["mass"]) + math.fsum(g("1.00", kv_c)["mass"])
    act_mass_100 = math.fsum(x for k in act for x in g("1.00", k)["mass"])
    assert kv_mass_100 == 0.0 and abs(act_mass_100 - c["mass_ms"]) < 1e-9
    C.add("reserve.active_cap_invariance", "the three active-sequence-cap regimes are identical at every reserve (decisions, disagreements, beneficial, headroom mass) and supply all of the 23.6 ms remaining at reserve 1.00",
          {"active_regimes_decisions_states_beneficial_mass": inv, "mass_1.00_ms": c["mass_ms"], "kv_regime_mass_1.00_ms": kv_mass_100}, ["identical at every reserve", f"remaining {c['mass_ms']:.1f}~ms"], src,
          "per-regime tuples compared across the three reserves; sum of H over active-cap regimes == total mass at reserve 1.00")
    # sensitivity-run intervals (seed 42) kept distinct from the canonical primary interval
    ci = {r: [R[r]["boot"]["mean_oracle_headroom_ci95_low"] * 1000, R[r]["boot"]["mean_oracle_headroom_ci95_high"] * 1000] for r in R}
    assert all(R[r]["boot"]["bootstrap_seed"] == 42 and R[r]["boot"]["bootstrap_replicates"] == 2000 for r in R)
    fr = json.loads((EXP / "fresh_production_latency_headroom_confirmatory_v1" / "FRESH_LATENCY_BOOTSTRAP_V1.json").read_text())
    C.add("reserve.sensitivity_intervals_distinct_from_primary", "sensitivity-run window-clustered 95% intervals (2,000 replicates, seed 42) [0.182,3.563] / [0.067,4.547] / [0.042,0.232]; the primary confirmatory interval [0.1909, 3.5462] uses its own seed and is unchanged",
          {"sensitivity_ci_ms": ci, "sensitivity_seed": 42, "sensitivity_clusters": {r: R[r]["boot"]["clusters"] for r in R}, "primary_bootstrap_file": "FRESH_LATENCY_BOOTSTRAP_V1.json"},
          [f"(2,000\nreplicates, seed 42) are [{ci['0.82'][0]:.3f}, {ci['0.82'][1]:.3f}], [{ci['0.90'][0]:.3f}, {ci['0.90'][1]:.3f}], and\n[{ci['1.00'][0]:.3f}, {ci['1.00'][1]:.3f}]".replace("\n", " "), "primary confirmatory interval at 0.82 remains"], "; ".join(f"experiments/reference_reserve_sensitivity_v1/results_v1/reserve_{d}/bootstrap_summary.json" for d in RES.values()) + f"; {FROZEN}/FRESH_LATENCY_BOOTSTRAP_V1.json",
          "bootstrap_summary.json: mean_oracle_headroom_ci95_{low,high} * 1000; seed and replicates asserted")


def add_design_claims(C):
    """Setup statements not covered elsewhere, recomputed from the frozen artifacts and code."""
    wl = {r["source_dataset"]: r for r in csv_rows(PHASE_A / "PHASE_A_WORKLOAD_SUMMARY_V1.csv")}
    src_a = rel(PHASE_A / "PHASE_A_WORKLOAD_SUMMARY_V1.csv")
    assert all(int(v["windows"]) == 20 and int(v["requests"]) == 4000 for v in wl.values()) and len(wl) == 3
    C.add("design.native_windows", "faithful corpus: 20 windows of 200 requests from each of three workloads (60 windows, 4,000 requests per workload)",
          {"windows_per_workload": 20, "requests_per_window": 200, "workloads": 3}, ["20 windows of 200 requests from each of Azure 2023 code, Azure 2023 conversation, and BurstGPT"], src_a, "windows and requests columns")
    # prose restatements of Table 2 (onset / sustained settings) in Sections 4.2 and 5.1
    tmap = csv_rows(PHASE_B / "PHASE_B_V2_TRANSITION_MAP.csv")
    T = {(x["source_dataset"], x["axis"]): x for x in tmap}
    pt = lambda w, ax, col: int(T[(w, ax)][col].split("_")[1])
    A, K = "active_sequence_capacity", "kv_capacity"
    first_a = {w: pt(w, A, "first_disagreement_point") for w in wl}
    first_k = {pt(w, K, "first_disagreement_point") for w in wl}
    sus_a = {w: pt(w, A, "sustained_disagreement_point") for w in wl}
    sus_k = {w: pt(w, K, "sustained_disagreement_point") for w in wl}
    bind_conv_k = pt("azure_2023_conv", K, "first_binding_point")
    assert first_k == {16000} and sus_a["azure_2023_code"] == sus_a["azure_2023_conv"] and sus_k["azure_2023_code"] == sus_k["burstgpt"]
    C.add("pressure.onset_and_sustained_prose", "first disagreement at active-sequence cap 16 (BurstGPT) / 8 (Azure code) / 4 (Azure conversation) and KV capacity 16,000 in all three; sustained at caps 8 (BurstGPT), 4 (Azure code and conversation) and KV 16,000 (Azure code, BurstGPT) / 8,000 (Azure conversation); Azure-conversation KV binds first at 8,000",
          {"first_disagreement_active_cap": first_a, "first_disagreement_kv": sorted(first_k), "sustained_active_cap": sus_a, "sustained_kv": sus_k, "azure_conv_kv_first_binding": bind_conv_k},
          [f"active-sequence cap of {first_a['burstgpt']} (BurstGPT), {first_a['azure_2023_code']} (Azure code), or {first_a['azure_2023_conv']} (Azure conversation), and at a KV\ncapacity of {comma(first_k.copy().pop())} tokens in all three workloads".replace("\n", " "),
           f"at caps of {sus_a['burstgpt']} (BurstGPT) and {sus_a['azure_2023_code']} (Azure code and conversation)\nand at KV capacities of {comma(sus_k['azure_2023_code'])} (Azure code and BurstGPT) and {comma(sus_k['azure_2023_conv'])} (Azure\nconversation)".replace("\n", " "),
           f"disagreement appears at {comma(first_k.copy().pop())} tokens, before the constraint physically binds at {comma(bind_conv_k)}"],
          rel(PHASE_B / "PHASE_B_V2_TRANSITION_MAP.csv"), "first_disagreement_point / sustained_disagreement_point / first_binding_point by (source_dataset, axis)")
    q = int(wl["burstgpt"]["max_queue_length"])
    assert q == 31
    C.add("design.burstgpt_native_queue", "native replay of the original BurstGPT windows reached a queue of 31 requests", q, ["reached a queue of 31 requests"], src_a, "row source_dataset=burstgpt: max_queue_length")
    # smallest genuine advantage: one decode step over the largest continuation population
    fresh = csv_rows(EXP / "fresh_production_latency_headroom_confirmatory_v1" / "FRESH_LATENCY_STATE_LEVEL_V1.csv")
    H = [float(r["oracle_headroom"]) for r in fresh]
    pos = [h for h in H if h > 1e-12]
    floor = min(pos)
    assert f"{floor:.2e}".startswith("5.15e-06") and round(0.001 / floor) == 194
    C.add("design.numerical_floor", "no genuine advantage is smaller than about 5.2e-6 s (one 1 ms decode step over at most 194 requests)", {"min_positive_advantage_s": floor, "implied_population": round(0.001 / floor)},
          ["smaller than about $5.2\\times10^{-6}$~s"], f"{FROZEN}/FRESH_LATENCY_STATE_LEVEL_V1.csv", "min oracle_headroom above the 1e-12 guard; 0.001 / that")
    wins = {(r["source_dataset"], r["window_index"]) for r in fresh}
    wpos = {(r["source_dataset"], r["window_index"]) for r in fresh if float(r["oracle_headroom"]) > 0}
    assert (len(wins), len(wpos)) == (36, 32)
    C.add("design.windows_with_positive_headroom", "32 of the 36 windows contain a state with positive headroom", {"windows": len(wins), "positive": len(wpos)},
          ["32 of the 36 windows contain a state with positive headroom"], f"{FROZEN}/FRESH_LATENCY_STATE_LEVEL_V1.csv", "distinct (source_dataset, window_index) with oracle_headroom > 0")
    reg = csv_rows(EXP / "fresh_production_latency_headroom_confirmatory_v1" / "FRESH_LATENCY_WORKLOAD_REGIME_V1.csv")
    mx = max(float(r["P_D"]) for r in reg)
    assert mx < 0.005
    C.add("design.pd_below_half_percent", "P(D) is below 0.5% in every selected regime (maximum 0.456%)", {"max_P_D_pct": 100 * mx}, ["is below 0.5\\% in every selected regime"],
          f"{FROZEN}/FRESH_LATENCY_WORKLOAD_REGIME_V1.csv", "max P_D over the five regimes")
    # pilot ANWG saturation (Phase D, original windows)
    cont = csv_rows(PHASE_D / "PHASE_D_CONTINUATION_RESULTS_V1.csv")
    anwg = [float(r["q_sbs_anwg"]) for r in cont]
    assert anwg and all(abs(v - 1.0) < 1e-12 for v in anwg)
    ref, alts = {}, {}
    for r in cont:
        k = (r["state_id"],)
        if r["is_sbs_reference_branch"] in ("True", "true", "1"):
            ref[k] = float(r["mean_latency"])
    n = changed = 0
    for r in cont:
        k = (r["state_id"],)
        if r["is_sbs_reference_branch"] in ("True", "true", "1") or k not in ref:
            continue
        n += 1; changed += abs(float(r["mean_latency"]) - ref[k]) > 1e-12
    assert n > 0 and changed / n > 0.9
    C.add("design.pilot_anwg_saturation", "pilot causal analysis on the original windows: every terminal ANWG value is 1.0 although most counterfactual branches changed mean latency",
          {"anwg_values": len(anwg), "all_equal_1": True, "branches_compared": n, "share_changed_latency": changed / n}, ["every\nterminal ANWG value was 1.0".replace("\n", " "), "most counterfactual branches changed latency"],
          rel(PHASE_D / "PHASE_D_CONTINUATION_RESULTS_V1.csv"), "q_sbs_anwg over all continuation rows; per state_id, non-reference mean_latency vs the SBS reference branch")
    src_txt = (ROOT / "src" / "llmserveopt" / "policies" / "prefill_control_variants.py").read_text()
    chunk = int(re.search(r"^DEFAULT_CHUNK_SMALL\s*=\s*(\d+)", src_txt, re.M).group(1))
    assert chunk == 64
    C.add("design.small_chunk_size", "small-chunk prefill uses 64-token chunks", chunk, ["small-chunk prefill (64-token chunks)"],
          rel(PHASE_A / "PHASE_A_NATIVE_CONFIG_FREEZE_V1.json") + " (portfolio policy chunked_prefill_small); src/llmserveopt/policies/prefill_control_variants.py", "DEFAULT_CHUNK_SMALL; policy listed in p6_policy_portfolio")
    cfg = json.loads((PHASE_A / "PHASE_A_NATIVE_CONFIG_FREEZE_V1.json").read_text())
    g = cfg["gpu_config"]
    assert (g["max_active_sequences"], g["max_batch_tokens"], g["max_kv_tokens"], cfg["gpu_server_count"], cfg["service_model_kwargs"]["step_size"]) == (512, 512, 8_000_000, 1, 0.001)
    assert len(cfg["p6_policy_portfolio"]) == 6 and cfg["scheduler"] == "kv_constrained_online"
    C.add("design.gpu_config_and_portfolio", "one simulated GPU: 512 active-sequence slots, 512-token batch budget, 8,000,000 KV tokens, 1 ms step; six-policy portfolio with kv_constrained_online as reference",
          {"gpu_config": g, "gpu_server_count": 1, "step_size_s": cfg["service_model_kwargs"]["step_size"], "portfolio": cfg["p6_policy_portfolio"], "reference": cfg["scheduler"]},
          ["a 512-token batch budget", "The portfolio contains six policies implemented in the simulator"], rel(PHASE_A / "PHASE_A_NATIVE_CONFIG_FREEZE_V1.json"), "gpu_config, gpu_server_count, service_model_kwargs.step_size, p6_policy_portfolio, scheduler")
    proto = json.loads((EXP / "fresh_production_latency_headroom_confirmatory_v1" / "CAUSAL_PROTOCOL_V1.json").read_text())
    ptxt = json.dumps(proto)
    assert "exactly one forced alternative action" in ptxt and "immediately revert to fixed SBS" in ptxt
    C.add("design.one_step_forcing", "counterfactual = exactly one forced alternative action, then immediate reversion to the fixed SBS", {"protocol_phrases": ["exactly one forced alternative action", "immediately revert to fixed SBS"]},
          ["force exactly one action, and return to the SBS for the continuation"], f"{FROZEN}/CAUSAL_PROTOCOL_V1.json", "protocol text contains both phrases")


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
