#!/usr/bin/env python3
"""Non-destructive correction of the fresh-causal STATE-LEVEL artifact's SBS-reference columns.

Defect
------
``FRESH_LATENCY_STATE_LEVEL_V1.csv`` columns ``mean_ref_latency`` and ``p95_ref_latency`` hold the values
of the *first counterfactual branch* of each state, not those of the SBS reference branch.  Cause: in
``scripts/fresh_latency_causal_confirmatory_v1.py::analyze`` the state loop iterates ``cf.groupby("state_id")``
(counterfactual rows only) and reads ``meta = g.iloc[0]`` -> ``float(meta["mean_latency"])`` /
``float(meta["p95_latency"])``.  The primary endpoint never reads these columns (``a_lat`` uses the separate
``ref`` frame), so it is unaffected.

What this script does
---------------------
1. Verifies the raw continuation shards (the only artifact holding the SBS_REFERENCE rows) against the frozen
   hash manifest.
2. Replays the executor's aggregation on the shards and shows that it reproduces the frozen state-level,
   action-level and regime CSVs BYTE-FOR-BYTE (root-cause proof and validation of the replay model).
3. Shows that every downstream result is computed identically when the two columns are removed.
4. Writes a corrected DERIVATIVE to a new directory: identical rows/identifiers/columns, only the two
   proven-incorrect columns replaced by the SBS_REFERENCE values.  The frozen directory is never written.
5. Writes the audit table, the extracted SBS reference rows, and deterministic provenance metadata.

Everything is deterministic; provenance contains no timestamps and no absolute paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import fresh_causal_robustness_v1 as rob  # noqa: E402  (canonical bootstrap loop, helpers)

FROZEN = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1"
OUT = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1_corrected"
ROBUSTNESS_OUT = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1_robustness"
ARCHIVE_ROOT_DEFAULT = ROOT.parent / "llm-serving-heuristic-evolution-local-provenance"
ARCHIVE_RUN_REL = "fgcs-finalization-20260920/fresh_latency/fresh-latency-execution-v1/fresh_latency_causal_run_v1"

CORRECTED_COLUMNS = ("mean_ref_latency", "p95_ref_latency")
CORRECTED_NAME = "FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv"
AUDIT_NAME = "STATE_LEVEL_CORRECTION_AUDIT_V1.csv"
SBS_NAME = "SBS_REFERENCE_ROWS_V1.csv"
PROV_NAME = "CORRECTION_PROVENANCE_V1.json"
RECHECK_NAME = "ROBUSTNESS_AND_590_589_RECHECK_V1.json"
README_NAME = "README.md"

# fixed identifiers (no HEAD: provenance must be reproducible)
EXECUTION_COMMIT = "38e02bcdaeb8f9cec2289b05dea9a1406318d14f"
BOOTSTRAP_FIX_COMMIT = "a8fd735"
ROBUSTNESS_COMMIT = "5c8a78c4f1f4c9e92546c71bd3c0367cbb25b13a"

FROZEN_INPUTS = (
    "FRESH_LATENCY_STATE_LEVEL_V1.csv",
    "FRESH_LATENCY_ACTION_LEVEL_V1.csv",
    "FRESH_LATENCY_WORKLOAD_REGIME_V1.csv",
    "FRESH_LATENCY_CAUSAL_RESULT_V1.json",
    "FRESH_LATENCY_BOOTSTRAP_V1.json",
    "FRESH_LATENCY_ARTIFACT_HASHES_V1.json",
    "FRESH_LATENCY_EXECUTION_PROVENANCE_V1.json",
    "FRESH_SUPPORT_WORKLOAD_AXIS_SUMMARY_V1.csv",
    "FRESH_REGIME_SELECTION_V1.json",
    "PREREGISTRATION_V1.json",
    "METRIC_PROTOCOL_V1.json",
    "ARTIFACT_HASHES_V1.json",
)


# =============================================================================================
# CSV plumbing (byte-compatible with the executor's write_rows)
# =============================================================================================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return list(r.fieldnames or []), rows


def csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(fields))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def executor_write_rows_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Exactly the executor's ``write_rows``: header = sorted union of keys."""
    return csv_bytes(sorted({k for r in rows for k in r}), rows)


# =============================================================================================
# shards
# =============================================================================================
def frozen_shard_hashes(frozen: Path = FROZEN) -> dict[str, str]:
    return json.loads((frozen / "FRESH_LATENCY_ARTIFACT_HASHES_V1.json").read_text())["shard_artifacts"]


def resolve_run_root(archive_root: Path | None) -> Path | None:
    for cand in filter(None, (archive_root, Path(os.environ["FRESH_ARCHIVE_ROOT"]) if "FRESH_ARCHIVE_ROOT" in os.environ else None, ARCHIVE_ROOT_DEFAULT)):
        p = Path(cand) / ARCHIVE_RUN_REL
        if (p / "continuation_shards").is_dir():
            return p
    return None


def verify_shards(run_root: Path, frozen: Path = FROZEN) -> dict:
    exp = frozen_shard_hashes(frozen)
    bad, missing, n = [], [], 0
    for name, h in exp.items():
        f = run_root / "continuation_shards" / name
        if not f.exists():
            f = run_root / name
        if not f.exists():
            missing.append(name)
            continue
        n += 1
        if sha256_file(f) != h:
            bad.append(name)
    return {"expected": len(exp), "checked": n, "mismatched": bad, "missing": missing, "all_match": not bad and not missing}


def load_shard_rows(run_root: Path, n_shards: int = 96) -> list[tuple[str, dict[str, str]]]:
    """(shard_file, row) pairs in the executor's ``collect`` order (sorted shard files, file order)."""
    out = []
    for p in sorted((run_root / "continuation_shards").glob(f"shard_*_of_{n_shards:03d}.csv")):
        if p.name.endswith("_correctness.csv"):
            continue
        _, rows = read_rows(p)
        out.extend((p.name, r) for r in rows)
    return out


# =============================================================================================
# replay of the executor's aggregation (verbatim logic, from shard strings)
# =============================================================================================
def _frames(shard_rows: Sequence[tuple[str, dict[str, str]]]):
    df = pd.DataFrame([r for _, r in shard_rows])
    ref = df[df.branch_type == "SBS_REFERENCE"].set_index("state_id")
    cf = df[df.branch_type == "COUNTERFACTUAL"].copy()
    cf["a_lat"] = cf["state_id"].map(ref["mean_latency"].astype(float)) - cf["mean_latency"].astype(float)
    return df, ref, cf


def replay_state_rows(cf: pd.DataFrame, ref: pd.DataFrame, reference: str) -> list[dict[str, Any]]:
    """State-level rows.  ``reference='first_cf'`` reproduces the defect; ``'sbs'`` is the corrected logic."""
    assert reference in ("first_cf", "sbs")
    state = []
    for sid, g in cf.groupby("state_id", sort=True):
        adv = g.a_lat.to_numpy(float)
        meta = g.iloc[0]
        src = meta if reference == "first_cf" else ref.loc[sid]
        state.append({"state_id": sid, "source_dataset": meta.source_dataset, "window_index": int(meta.window_index), "axis": meta.axis, "condition_id": meta.condition_id, "regime_stage": meta.regime_stage, "num_non_sbs_branches": len(g), "max_a_lat": float(np.max(adv)), "oracle_headroom": float(max(0, np.max(adv))), "beneficial_opportunity": int(np.max(adv) > 0), "all_alternatives_harmful": int(np.all(adv < 0)), "all_alternatives_zero": int(np.all(adv == 0)), "mixed_beneficial_and_harmful": int(np.any(adv > 0) and np.any(adv < 0)), "mean_ref_latency": float(src["mean_latency"]), "p95_ref_latency": float(src["p95_latency"])})
    return state


def replay_overall(sdf: pd.DataFrame) -> dict:
    """The executor's ``overall`` block (reads only oracle_headroom / beneficial / harm-structure columns)."""
    pos = sdf.oracle_headroom > 0
    return {"states": len(sdf), "beneficial_states": int(sdf.beneficial_opportunity.sum()), "P_B_given_D": float(sdf.beneficial_opportunity.mean()), "mean_oracle_headroom": float(sdf.oracle_headroom.mean()), "positive_headroom_mean": float(sdf.loc[pos, "oracle_headroom"].mean()) if pos.any() else 0.0, "positive_headroom_median": float(sdf.loc[pos, "oracle_headroom"].median()) if pos.any() else 0.0, "all_harmful_states": int(sdf.all_alternatives_harmful.sum()), "all_zero_states": int(sdf.all_alternatives_zero.sum()), "mixed_states": int(sdf.mixed_beneficial_and_harmful.sum())}


def replay_regime_rows(sdf: pd.DataFrame, cf: pd.DataFrame, ref: pd.DataFrame, frozen: Path = FROZEN) -> list[dict[str, Any]]:
    support = pd.read_csv(frozen / "FRESH_SUPPORT_WORKLOAD_AXIS_SUMMARY_V1.csv")
    selected = json.loads((frozen / "FRESH_REGIME_SELECTION_V1.json").read_text())["selected_regimes"]
    out = []
    for sel in selected:
        key = (sel["source_dataset"], sel["axis"], sel["condition_id"])
        sg = sdf[(sdf.source_dataset == key[0]) & (sdf.axis == key[1]) & (sdf.condition_id == key[2])]
        ag = cf[(cf.source_dataset == key[0]) & (cf.axis == key[1]) & (cf.condition_id == key[2])].copy()
        sr = support[(support.source_dataset == key[0]) & (support.axis == key[1]) & (support.condition_id == key[2])].iloc[0]
        ag["relative_a_lat"] = ag["a_lat"] / ag["state_id"].map(ref["mean_latency"].astype(float))
        rel = ag.groupby("state_id")["relative_a_lat"].max()
        p95 = ag["state_id"].map(ref["p95_latency"].astype(float)) - ag["p95_latency"].astype(float)
        out.append({
            "source_dataset": key[0], "axis": key[1], "condition_id": key[2], "regime_roles": "+".join(sel["roles"]),
            "P_D": float(sr.canonical_disagreement_states) / float(sr.sbs_decision_states), "states": int(len(sg)),
            "contributing_windows": int(sg.window_index.nunique()), "P_B_given_D": float(sg.beneficial_opportunity.mean()),
            "mean_oracle_headroom": float(sg.oracle_headroom.mean()), "positive_headroom_mean": float(sg.loc[sg.oracle_headroom > 0, "oracle_headroom"].mean()) if (sg.oracle_headroom > 0).any() else 0.0,
            "positive_headroom_median": float(sg.loc[sg.oracle_headroom > 0, "oracle_headroom"].median()) if (sg.oracle_headroom > 0).any() else 0.0,
            "mean_relative_headroom": float(rel.clip(lower=0).mean()), "p95_effect_mean": float(p95.mean()),
            "anwg_effect": "NOT_CAPTURED_IN_EXECUTION_OUTPUT", "anwg_status": "secondary metric not used by primary verdict",
        })
    return out


# =============================================================================================
# corrected derivative construction (string-level: unaffected fields retained verbatim)
# =============================================================================================
def sbs_reference_table(shard_rows: Sequence[tuple[str, dict[str, str]]]) -> tuple[list[str], list[dict[str, str]]]:
    rows = []
    for shard, r in shard_rows:
        if r["branch_type"] == "SBS_REFERENCE":
            rows.append({**r, "source_shard_file": shard})
    rows.sort(key=lambda r: r["state_id"])
    fields = sorted({k for r in rows for k in r if k != "source_shard_file"}) + ["source_shard_file"]
    return fields, rows


def build_corrected_rows(orig_rows: Sequence[Mapping[str, str]], sbs_by_state: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    """Replace only ``mean_ref_latency`` / ``p95_ref_latency`` by the SBS_REFERENCE values; all else verbatim."""
    out = []
    for r in orig_rows:
        ref = sbs_by_state[r["state_id"]]
        n = dict(r)
        n["mean_ref_latency"] = repr(float(ref["mean_latency"]))
        n["p95_ref_latency"] = repr(float(ref["p95_latency"]))
        out.append(n)
    return out


def audit_rows(orig_rows: Sequence[Mapping[str, str]], corr_rows: Sequence[Mapping[str, str]], first_cf: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    out = []
    for o, c in zip(orig_rows, corr_rows):
        assert o["state_id"] == c["state_id"]
        om, cm, op, cp = float(o["mean_ref_latency"]), float(c["mean_ref_latency"]), float(o["p95_ref_latency"]), float(c["p95_ref_latency"])
        row = {"state_id": o["state_id"], "num_non_sbs_branches": o["num_non_sbs_branches"],
               "mean_ref_latency_original": o["mean_ref_latency"], "mean_ref_latency_corrected": c["mean_ref_latency"],
               "mean_ref_latency_delta_s": repr(om - cm), "mean_ref_latency_changed": om != cm,
               "p95_ref_latency_original": o["p95_ref_latency"], "p95_ref_latency_corrected": c["p95_ref_latency"],
               "p95_ref_latency_delta_s": repr(op - cp), "p95_ref_latency_changed": op != cp}
        if first_cf is not None:
            row["defect_source_branch_id"] = first_cf[o["state_id"]]
        out.append(row)
    return out


# =============================================================================================
# main procedure
# =============================================================================================
def frozen_hashes(frozen: Path = FROZEN) -> dict[str, str]:
    return {f: sha256_file(frozen / f) for f in FROZEN_INPUTS if (frozen / f).exists()}


def canonical_state_bootstrap(sdf_min: pd.DataFrame) -> dict:
    return rob.canonical_bootstrap_loop(sdf_min)


def replay_proof(shard_rows, frozen: Path = FROZEN) -> dict:
    """Byte-level replay of the frozen artifacts + proof that downstream results do not read the two columns."""
    _, ref, cf = _frames(shard_rows)
    proof: dict[str, Any] = {}
    defective = replay_state_rows(cf, ref, "first_cf")
    proof["defective_state_level_replay_is_byte_identical_to_frozen"] = executor_write_rows_bytes(defective) == (frozen / "FRESH_LATENCY_STATE_LEVEL_V1.csv").read_bytes()
    proof["action_level_replay_is_byte_identical_to_frozen"] = executor_write_rows_bytes(cf.to_dict("records")) == (frozen / "FRESH_LATENCY_ACTION_LEVEL_V1.csv").read_bytes()
    # downstream with the two columns REMOVED (KeyError if anything read them)
    sdf_nocols = pd.DataFrame(defective).drop(columns=list(CORRECTED_COLUMNS))
    reg = replay_regime_rows(sdf_nocols, cf, ref, frozen)
    proof["regime_csv_replay_without_ref_columns_is_byte_identical_to_frozen"] = executor_write_rows_bytes(reg) == (frozen / "FRESH_LATENCY_WORKLOAD_REGIME_V1.csv").read_bytes()
    overall = replay_overall(sdf_nocols)
    frozen_primary = json.loads((frozen / "FRESH_LATENCY_CAUSAL_RESULT_V1.json").read_text())["primary"]
    proof["primary_overall_replay_without_ref_columns_equals_frozen"] = all(overall[k] == frozen_primary[k] for k in overall)
    # The frozen (post-correction) CI was recomputed from the frozen CSV read with pandas' default float parser
    # (see BOOTSTRAP_CLUSTER_KEY_CORRECTION note), so the bit-exact replay reads the CSV the same way.
    fb = json.loads((frozen / "FRESH_LATENCY_BOOTSTRAP_V1.json").read_text())
    csv_df = pd.read_csv(frozen / "FRESH_LATENCY_STATE_LEVEL_V1.csv").drop(columns=list(CORRECTED_COLUMNS))
    ci = canonical_state_bootstrap(csv_df[["source_dataset", "window_index", "oracle_headroom"]])
    proof["bootstrap_ci_from_frozen_csv_without_ref_columns_equals_frozen_bit_exact"] = (ci["lo"] == fb["mean_oracle_headroom_ci95_low"] and ci["hi"] == fb["mean_oracle_headroom_ci95_high"] and ci["clusters"] == fb["clusters"])
    mem = canonical_state_bootstrap(sdf_nocols[["source_dataset", "window_index", "oracle_headroom"]])
    proof["_notes"] = {"bootstrap_ci_from_in_memory_floats_s": [mem["lo"], mem["hi"]],
                       "max_abs_diff_in_memory_vs_frozen_ci_endpoints_s": max(abs(mem["lo"] - fb["mean_oracle_headroom_ci95_low"]), abs(mem["hi"] - fb["mean_oracle_headroom_ci95_high"])),
                       "explanation": "pandas default read_csv float parser is not correctly rounded (448/720 oracle_headroom values differ from float() by <=7.7e-13 relative); float_precision='round_trip' reproduces the frozen mean/positive-mean/median bit-exactly"}
    corrected = replay_state_rows(cf, ref, "sbs")
    proof["corrected_replay_differs_from_defective_only_in_the_two_columns"] = all(
        {k: v for k, v in a.items() if k not in CORRECTED_COLUMNS} == {k: v for k, v in b.items() if k not in CORRECTED_COLUMNS} for a, b in zip(defective, corrected))
    return proof


def run(archive_root: Path | None = None, out: Path = OUT, frozen: Path = FROZEN) -> dict:
    run_root = resolve_run_root(archive_root)
    if run_root is None:
        raise SystemExit("continuation shards not found (set --archive-root or FRESH_ARCHIVE_ROOT)")
    before = frozen_hashes(frozen)
    frozen_hash_file = json.loads((frozen / "FRESH_LATENCY_ARTIFACT_HASHES_V1.json").read_text())["compact_artifacts"]
    frozen_ok = {f: sha256_file(frozen / f) == h for f, h in frozen_hash_file.items()}
    if not all(frozen_ok.values()):
        raise SystemExit(f"frozen artifacts do not match their hash file: {frozen_ok}")
    shard_check = verify_shards(run_root, frozen)
    if not shard_check["all_match"]:
        raise SystemExit(f"shard verification failed: {shard_check}")

    shard_rows = load_shard_rows(run_root)
    proof = replay_proof(shard_rows, frozen)
    if not all(v for k, v in proof.items() if not k.startswith("_")):
        raise SystemExit(f"replay proof failed: {proof}")

    orig_fields, orig_rows = read_rows(frozen / "FRESH_LATENCY_STATE_LEVEL_V1.csv")
    sbs_fields, sbs_rows = sbs_reference_table(shard_rows)
    sbs_by_state = {r["state_id"]: r for r in sbs_rows}
    assert len(sbs_rows) == len(orig_rows) == 720 and set(sbs_by_state) == {r["state_id"] for r in orig_rows}
    corr_rows = build_corrected_rows(orig_rows, sbs_by_state)
    corrected_bytes = csv_bytes(orig_fields, corr_rows)
    # cross-check: string-level correction == replay-with-corrected-logic
    _, ref_df, cf_df = _frames(shard_rows)
    assert corrected_bytes == executor_write_rows_bytes(replay_state_rows(cf_df, ref_df, "sbs")), "string-level and replay corrections disagree"

    first_cf = {}
    for _, r in shard_rows:
        if r["branch_type"] == "COUNTERFACTUAL":
            first_cf.setdefault(r["state_id"], r["branch_id"])
    audit = audit_rows(orig_rows, corr_rows, first_cf)

    out.mkdir(parents=True, exist_ok=True)
    (out / CORRECTED_NAME).write_bytes(corrected_bytes)
    (out / SBS_NAME).write_bytes(csv_bytes(sbs_fields, sbs_rows))
    audit_fields = list(audit[0].keys())
    (out / AUDIT_NAME).write_bytes(csv_bytes(audit_fields, audit))

    # validation of the derivative against the frozen original
    o = pd.read_csv(frozen / "FRESH_LATENCY_STATE_LEVEL_V1.csv", dtype=str)
    c = pd.read_csv(out / CORRECTED_NAME, dtype=str)
    unchanged_cols = [k for k in o.columns if k not in CORRECTED_COLUMNS]
    validation = {
        "row_count_original_corrected": [len(o), len(c)],
        "identical_column_order": list(o.columns) == list(c.columns),
        "identical_state_id_sequence": o.state_id.tolist() == c.state_id.tolist(),
        "unchanged_columns_byte_identical_as_text": bool((o[unchanged_cols] == c[unchanged_cols]).all().all()),
        "unchanged_columns": unchanged_cols, "corrected_columns": list(CORRECTED_COLUMNS),
        "n_rows_mean_ref_latency_changed": int(sum(a["mean_ref_latency_changed"] for a in audit)),
        "n_rows_p95_ref_latency_changed": int(sum(a["p95_ref_latency_changed"] for a in audit)),
        "n_rows_either_changed": int(sum(a["mean_ref_latency_changed"] or a["p95_ref_latency_changed"] for a in audit)),
        "n_rows_mean_changed_by_more_than_1e-12_s": int(sum(abs(float(a["mean_ref_latency_delta_s"])) > 1e-12 for a in audit)),
        "n_rows_p95_changed_by_more_than_1e-12_s": int(sum(abs(float(a["p95_ref_latency_delta_s"])) > 1e-12 for a in audit)),
        "n_original_values_equal_first_cf_branch_value": int(sum(
            float(r["mean_ref_latency"]) == float(cf_df[cf_df.branch_id == first_cf[r["state_id"]]].mean_latency.iloc[0]) and
            float(r["p95_ref_latency"]) == float(cf_df[cf_df.branch_id == first_cf[r["state_id"]]].p95_latency.iloc[0]) for r in orig_rows)),
        "corrected_equals_sbs_reference_rows": bool(all(float(x) == float(sbs_by_state[s]["mean_latency"]) for s, x in zip(c.state_id, c.mean_ref_latency)) and all(float(x) == float(sbs_by_state[s]["p95_latency"]) for s, x in zip(c.state_id, c.p95_ref_latency))),
        "primary_endpoint_original_vs_corrected": {
            "mean_oracle_headroom_s": [float(o.oracle_headroom.astype(float).mean()), float(c.oracle_headroom.astype(float).mean())],
            "beneficial_states": [int((o.beneficial_opportunity.astype(int) == 1).sum()), int((c.beneficial_opportunity.astype(int) == 1).sum())],
            "oracle_headroom_column_identical": bool((o.oracle_headroom == c.oracle_headroom).all())},
    }
    if not (validation["identical_column_order"] and validation["identical_state_id_sequence"] and validation["unchanged_columns_byte_identical_as_text"] and validation["corrected_equals_sbs_reference_rows"]):
        raise SystemExit(f"validation failed: {validation}")

    after = frozen_hashes(frozen)
    exec_prov = json.loads((frozen / "FRESH_LATENCY_EXECUTION_PROVENANCE_V1.json").read_text())
    prov = {
        "schema_version": "fresh_latency_causal_confirmatory_v1.state_level_correction.0.0",
        "status": "CORRECTED_DERIVATIVE_OF_FROZEN_ARTIFACT (frozen original unchanged and remains the confirmatory record)",
        "classification": "DERIVED_COLUMN_LABELING_DEFECT (no effect on primary endpoint, regime CSV, bootstrap, or manuscript numbers)",
        "defect": {"file": "FRESH_LATENCY_STATE_LEVEL_V1.csv", "columns": list(CORRECTED_COLUMNS),
                   "cause": "analyze(): state loop iterates counterfactual rows only; meta = g.iloc[0]; float(meta['mean_latency']) / float(meta['p95_latency'])",
                   "executor_script": "scripts/fresh_latency_causal_confirmatory_v1.py", "executor_script_sha256_at_execution": exec_prov["execution_script_sha256"],
                   "execution_commit": EXECUTION_COMMIT, "bootstrap_key_fix_commit": BOOTSTRAP_FIX_COMMIT, "robustness_commit": ROBUSTNESS_COMMIT},
        "sources": {
            "frozen_state_level_sha256": before["FRESH_LATENCY_STATE_LEVEL_V1.csv"],
            "frozen_action_level_sha256": before["FRESH_LATENCY_ACTION_LEVEL_V1.csv"],
            "frozen_hash_file_sha256": before["FRESH_LATENCY_ARTIFACT_HASHES_V1.json"],
            "sbs_reference_source": "SBS_REFERENCE rows of the 96 continuation shards (local provenance archive, fresh-latency-execution-v1)",
            "shards_verified_against_frozen_manifest": shard_check,
            "shard_manifest_entries_sha256_of_sorted_items": sha256_bytes(json.dumps(sorted(frozen_shard_hashes(frozen).items())).encode()),
        },
        "replay_proof": proof,
        "validation": validation,
        "outputs_sha256": {CORRECTED_NAME: sha256_bytes(corrected_bytes), SBS_NAME: sha256_file(out / SBS_NAME), AUDIT_NAME: sha256_file(out / AUDIT_NAME)},
        "frozen_inputs_sha256": before,
        "frozen_directory_unchanged_by_run": before == after,
        "generator": {"script": "scripts/fresh_causal_correct_state_level_v1.py", "script_sha256": sha256_file(Path(__file__)),
                      "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    rob.dump_json(prov, out / PROV_NAME)
    (out / README_NAME).write_text(README_TEXT)
    return {"corrected_sha256": prov["outputs_sha256"][CORRECTED_NAME], "replay_proof_all_true": all(v for k, v in proof.items() if not k.startswith("_")), "frozen_unchanged": prov["frozen_directory_unchanged_by_run"], "validation": {k: validation[k] for k in ("n_rows_mean_ref_latency_changed", "n_rows_p95_ref_latency_changed", "n_rows_either_changed")}}


README_TEXT = """# Corrected derivative: fresh causal state-level artifact

This directory is a **non-destructive derivative** of
`../fresh_production_latency_headroom_confirmatory_v1/FRESH_LATENCY_STATE_LEVEL_V1.csv`.
The frozen original is unchanged and remains the confirmatory record.

| File | Content |
|---|---|
| `FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv` | Same 720 rows, identifiers and column order as the original. Only `mean_ref_latency` and `p95_ref_latency` differ: they now hold the SBS-reference branch values. |
| `STATE_LEVEL_CORRECTION_AUDIT_V1.csv` | Per-state original vs corrected values, deltas, change flags, and the counterfactual branch the original value came from. |
| `SBS_REFERENCE_ROWS_V1.csv` | The 720 `SBS_REFERENCE` rows extracted from the hash-verified continuation shards (source of the corrected values). |
| `CORRECTION_PROVENANCE_V1.json` | Hashes (source, outputs, frozen inputs, shards), replay proof, validation, generator. |
| `ROBUSTNESS_AND_590_589_RECHECK_V1.json` | Recheck of the robustness commit and the 590 vs 589 classification. |

Regenerate: `python scripts/fresh_causal_correct_state_level_v1.py --archive-root <local-provenance-archive>`.
Documentation: `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`.
"""


# =============================================================================================
# robustness-commit recheck and 590 vs 589 (independent of the state-level CSV)
# =============================================================================================
def recheck(archive_root: Path | None = None, out: Path = OUT, frozen: Path = FROZEN, robustness_out: Path = ROBUSTNESS_OUT) -> dict:
    run_root = resolve_run_root(archive_root)
    if run_root is None:
        raise SystemExit("continuation shards not found")
    shard_rows = load_shard_rows(run_root)
    df, ref, cf = _frames(shard_rows)
    res: dict[str, Any] = {}

    # ---- 590 vs 589 from the RAW shards (independent of every compact artifact)
    best = cf.groupby("state_id")["a_lat"].max()
    step, eps = 0.001, 1e-12
    resid = best[(best > 0) & (best <= eps)]
    sid = resid.index[0]
    ref_mean = float(ref.loc[sid, "mean_latency"])
    cf_row = cf[cf.state_id == sid].iloc[0]
    ncount = int(ref.loc[sid, "population_count"])
    ulp = float(np.spacing(ref_mean))
    all_a = cf["a_lat"]
    tiny = all_a[(all_a.abs() > 0) & (all_a.abs() <= eps)]
    grid_dev = (all_a * cf["population_count"].astype(float) / step)
    grid_dev = (grid_dev - grid_dev.round()).abs()
    res["count_590_vs_589"] = {
        "source": "raw shards: A(s,a) = SBS_REFERENCE.mean_latency - COUNTERFACTUAL.mean_latency",
        "strict_gt_0_beneficial_states": int((best > 0).sum()),
        "guarded_gt_eps_beneficial_states": int((best > eps).sum()), "epsilon_s": eps, "states": int(len(best)),
        "fraction_strict": float((best > 0).mean()), "fraction_guarded": float((best > eps).mean()),
        "residue_state": sid, "residue_advantage_s": float(resid.iloc[0]),
        "ref_mean_latency_s": ref_mean, "cf_mean_latency_s": float(cf_row["mean_latency"]),
        "ulp_of_ref_mean_latency_s": ulp, "advantage_in_ulps": float(resid.iloc[0] / ulp),
        "sbs_and_cf_completed_request_id_hash_identical": bool(cf_row["completed_ids_hash"] == ref.loc[sid, "completed_ids_hash"]),
        "population_count": ncount, "advantage_on_step_over_N_grid_units": float(resid.iloc[0] * ncount / step),
        "smallest_genuine_nonzero_abs_A_s": float(all_a[all_a.abs() > eps].abs().min()),
        "all_sub_eps_nonzero_advantages_s": [float(x) for x in tiny.tolist()],
        "max_grid_deviation_units": float(grid_dev.max()),
        "primary_mean_headroom_effect_of_residue_s": float(resid.iloc[0] / len(best)),
    }

    # ---- robustness commit: reference used vs true reference
    a = rob.load_actions(frozen)
    tm = ref["mean_latency"].astype(float)
    rec = a.groupby("state_id")["ref_mean_latency"].mean()
    res["robustness_reference_recovery"] = {"max_abs_diff_recovered_vs_true_sbs_reference_s": float((rec - tm.reindex(rec.index)).abs().max()),
                                            "recovery_formula": "cf mean_latency + a_lat"}
    # relative-headroom summary recomputed with the TRUE SBS reference
    true_rel = (best / tm.reindex(best.index)).clip(lower=0)
    committed = json.loads((robustness_out / "relative_headroom_summary.json").read_text())
    recomputed = {"mean": float(true_rel.mean()), "median": float(true_rel.median()), "p90": float(true_rel.quantile(.9)),
                  "frac_gt_1pct": float((true_rel > 0.01 + 1e-12).mean()), "frac_gt_5pct": float((true_rel > 0.05 + 1e-12).mean()),
                  "mean_ref_latency_ms_state_weighted": float(tm.mean() * 1e3),
                  "pooled_mean_H_over_pooled_mean_ref_latency": float(best.clip(lower=0).mean() / tm.mean())}
    res["robustness_relative_headroom_recheck"] = {k: {"committed": committed[k], "true_reference": v, "abs_diff": abs(committed[k] - v)} for k, v in recomputed.items()}

    # ---- static: the robustness module never reads the affected columns
    src = (ROOT / "scripts" / "fresh_causal_robustness_v1.py").read_text()
    res["robustness_static_check"] = {"reads_state_level_mean_ref_latency_column": ('"mean_ref_latency"]' in src.replace("ref_mean_latency", "")) or ("['mean_ref_latency']" in src),
                                       "reads_state_level_p95_ref_latency_column": ('"p95_ref_latency"' in src) or ("'p95_ref_latency'" in src),
                                       "note": "the only occurrences are documentation strings/summary key names; recovered reference is `ref_mean_latency` from the action-level file"}

    # ---- dynamic: rerun the robustness pipeline with both columns REMOVED from the state-level CSV
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        design = td / "design"
        design.mkdir()
        for f in frozen.iterdir():
            if f.is_file():
                shutil.copy2(f, design / f.name)
        # drop the two columns at TEXT level so every other field keeps its exact bytes (no float re-rounding)
        fields, rows = read_rows(frozen / "FRESH_LATENCY_STATE_LEVEL_V1.csv")
        keep = [k for k in fields if k not in CORRECTED_COLUMNS]
        (design / "FRESH_LATENCY_STATE_LEVEL_V1.csv").write_bytes(csv_bytes(keep, [{k: r[k] for k in keep} for r in rows]))
        out_dir = td / "out"
        old_root = rob.ROOT
        rob.ROOT = td
        try:
            rob.run(out_dir, design)
        finally:
            rob.ROOT = old_root
        cmp = {}
        for p in sorted(robustness_out.iterdir()):
            if p.is_file() and p.suffix in (".csv", ".json", ".md") and p.name != "provenance.json":
                q = out_dir / p.name
                cmp[p.name] = q.exists() and q.read_bytes() == p.read_bytes()
    res["robustness_rerun_without_ref_columns"] = {"compared_files": len(cmp), "all_identical_to_committed_outputs": all(cmp.values()), "non_identical": [k for k, v in cmp.items() if not v]}
    out.mkdir(parents=True, exist_ok=True)
    rob.dump_json(res, out / RECHECK_NAME)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive-root", type=Path, default=None, help="local provenance archive root (contains fgcs-finalization-20260920/...)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-recheck", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.archive_root, a.out), indent=2, default=str))
    if not a.skip_recheck:
        r = recheck(a.archive_root, a.out)
        print(json.dumps({"count_590_vs_589": {k: r["count_590_vs_589"][k] for k in ("strict_gt_0_beneficial_states", "guarded_gt_eps_beneficial_states", "advantage_in_ulps")},
                          "robustness_rerun_without_ref_columns": r["robustness_rerun_without_ref_columns"]}, indent=2))


if __name__ == "__main__":
    main()
