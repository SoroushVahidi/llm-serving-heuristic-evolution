"""Multi-Family Policy Separation Dataset (MF-PSD) v1 builder.

DATA UNIFICATION ONLY. This module unifies the three frozen, independently
audited policy-separation source datasets that the higher-level structural
reassessment (`docs/audits/reassessment_composition_hypothesis_20260817.md`)
identified as having reached composition-readiness gates:

  1. Family A v2 -- fairness/starvation (ranking) mechanism
     `experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/`
  2. Family B v2 -- prefill/decode TTFT contention (chunking) mechanism
     `experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/`
  3. Family C / KV v2 -- KV-pressure admission-control (memory) mechanism
     `experiments/kv_pressure_pilot_v2_20260817T165053Z/`

into one canonical long-form utility table (one row per
(source scenario, evaluated policy)) plus one canonical scenario-level
context table (one row per canonical scenario).

This module does NOT train a selector, does NOT perform pairwise-regret
learning, does NOT do mechanism attribution, and does NOT modify any of the
three frozen source CSVs -- see
docs/audits/multi_family_policy_separation_dataset_v1_20260817.md for the
full audit and docs/audits/reassessment_composition_hypothesis_20260817.md
for the scientific context (`COMPOSITION_DEMOTED`, revised roadmap Step 1).

Determinism: given the same (unmodified) three source files, `build_mf_psd`
always produces byte-identical CSV/JSON output (stable sort order, no
wall-clock or RNG dependence in the transform itself -- only the recorded
`build_timestamp_utc` provenance field varies between rebuilds).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BUILDER_VERSION = "mf_psd_v1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Canonical policy inventory
# ---------------------------------------------------------------------------

#: The six canonical policies named by the revised roadmap
#: (docs/audits/reassessment_composition_hypothesis_20260817.md, section O,
#: step 2) as the intended anchor pair for each of the three mechanism
#: families. Every other policy present in a source dataset is real,
#: evaluated evidence but is NOT one of these six anchors.
CANONICAL_ANCHOR_POLICIES = (
    "estimated_service_time_first",
    "weighted_fair_share",
    "full_prefill",
    "chunked_prefill_small",
    "kv_constrained_online",
    "least_laxity_first",
)

FAMILY_A = "FAMILY_A_FAIRNESS_STARVATION_V2"
FAMILY_B = "FAMILY_B_PREFILL_DECODE_V2"
FAMILY_C = "FAMILY_C_KV_PRESSURE_V2"

MECHANISM_FAMILIES = (FAMILY_A, FAMILY_B, FAMILY_C)

PRIMARY_METRIC = "arrival_normalized_weighted_goodput"

_SEED_SUFFIX_RE = re.compile(r"\.s\d+$")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def group_key_for_scenario_id(mechanism_family: str, source_scenario_id: str) -> str:
    """Strip the trailing `.s<seed>` suffix shared by all three source
    families' scenario-ID naming convention, then prefix with the family, to
    get the "same underlying scenario configuration, different seed" grouping
    key used for seed-grouped / leave-one-group-out evaluation. This is
    AUDIT/GROUPING METADATA, never a learnable feature."""
    base = _SEED_SUFFIX_RE.sub("", source_scenario_id)
    return f"{mechanism_family}::{base}"


# ---------------------------------------------------------------------------
# Source specifications (hardcoded, verified against the frozen provenance
# in git_state.txt / run_manifest.json / final_summary.json of each source
# run -- see docs/audits/multi_family_policy_separation_dataset_v1_20260817.md
# section "Pre-Build Source Inventory" for how each field was verified).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    mechanism_family: str
    source_run_id: str
    run_dir: Path
    per_policy_results_path: Path
    scenario_features_path: Optional[Path]
    launch_git_sha: str
    launch_git_branch: str
    audit_doc: str
    design_doc: Optional[str]
    family_verdict: str
    canonical_anchor_policies: Tuple[str, ...]
    extra_policies: Tuple[str, ...]
    notes: str


def default_source_specs() -> Tuple[SourceSpec, ...]:
    run_a = REPO_ROOT / "experiments" / "policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377"
    run_b = REPO_ROOT / "experiments" / "policy_separation_prefill_decode_pilot_v2_20260817T024204Z"
    run_c = REPO_ROOT / "experiments" / "kv_pressure_pilot_v2_20260817T165053Z"
    return (
        SourceSpec(
            mechanism_family=FAMILY_A,
            source_run_id=run_a.name,
            run_dir=run_a,
            per_policy_results_path=run_a / "per_policy_results.csv",
            scenario_features_path=run_a / "scenario_features.csv",
            launch_git_sha="16ad5d3e5af2e02516dfc42cc0825fa8eb7cbf38",
            launch_git_branch="policy-separation-v1-wulver-20260809",
            audit_doc="docs/audits/policy_separation_fairness_starvation_pilot_v2_20260816.md",
            design_doc=None,
            family_verdict="USEFUL_BUT_NEEDS_REFINEMENT",
            canonical_anchor_policies=("estimated_service_time_first", "weighted_fair_share"),
            extra_policies=("fifo", "aging_priority"),
            notes=(
                "Job 1182377, predecessor_failed_job_id=1182373 (git_state.txt). "
                "Executed on Wulver cluster worktree "
                "llm-serving-heuristic-evolution-policy-separation-v1, copied into "
                "this repo's experiments/ tree. Verdict is USEFUL_BUT_NEEDS_REFINEMENT, "
                "not literally *_COMPOSITION_READY, but the structural reassessment "
                "(docs/audits/reassessment_composition_hypothesis_20260817.md, "
                "section J/K) treats Family A v2 as one of the three sources with "
                "'high-quality, verified, low-tie-rate scenario boundaries' sufficient "
                "for this unification step."
            ),
        ),
        SourceSpec(
            mechanism_family=FAMILY_B,
            source_run_id=run_b.name,
            run_dir=run_b,
            per_policy_results_path=run_b / "per_policy_results.csv",
            scenario_features_path=run_b / "scenario_features.csv",
            launch_git_sha="ecc0422286886c83d263e87655ed1123e62d2565",
            launch_git_branch="contextual-compositional-heuristics-20260731",
            audit_doc="docs/audits/policy_separation_prefill_decode_pilot_v2_20260817.md",
            design_doc="docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md",
            family_verdict="FAMILY_B_COMPOSITION_READY",
            canonical_anchor_policies=("full_prefill", "chunked_prefill_small"),
            extra_policies=(),
            notes="run_manifest.json git_head/git_branch read directly from the run directory.",
        ),
        SourceSpec(
            mechanism_family=FAMILY_C,
            source_run_id=run_c.name,
            run_dir=run_c,
            per_policy_results_path=run_c / "per_policy_results.csv",
            scenario_features_path=None,  # features are embedded directly in per_policy_results.csv
            launch_git_sha="6be526ebffe4c3eba6428eab27f9adae1835d320",
            launch_git_branch="contextual-compositional-heuristics-20260731",
            audit_doc="docs/audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md",
            design_doc="docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md",
            family_verdict="KV_FAMILY_COMPOSITION_READY",
            canonical_anchor_policies=("kv_constrained_online", "least_laxity_first"),
            extra_policies=(),
            notes=(
                "This run predates the provenance guard added in a later commit "
                "(scripts/run_policy_separation_kv_pressure_pilot_v1.py "
                "_collect_provenance, commit c757d00); it has no run_manifest.json "
                "of its own. launch_git_sha is the launch commit "
                "'feat: Family C v2 KV-pressure reserve refinement' (6be526e), "
                "identified from `git log --oneline -- configs/kv_pressure_pilot_v2.yaml "
                "docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md` and corroborated "
                "by docs/audits/kv_v2_reproducibility_forensic_20260817.md, which "
                "independently names 6be526e as the KV v2 launch commit. See that audit "
                "for the known (bounded, non-blocking for this task) reproducibility "
                "caveat on the historical KV v2 CSV."
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Long-form table
# ---------------------------------------------------------------------------

LONG_FORM_COLUMNS = [
    "mf_psd_row_id",
    "canonical_scenario_id",
    "source_scenario_id",
    "mechanism_family",
    "source_run_id",
    "source_result_path",
    "source_row_index",
    "group_key",
    "seed",
    "source_split_raw",
    "source_policy_name",
    "canonical_policy_id",
    "is_canonical_anchor",
    "status",
    "primary_utility_anwg",
    "secondary_completion_fraction",
    "secondary_unweighted_slo_success_rate",
    "source_row_json",
    "source_scenario_features_json",
    "builder_version",
]

# Columns that, per source CSV row, are provenance/context rather than
# outcome -- captured verbatim as source_row_json but never promoted to a
# top-level "shared" numeric column, to avoid inventing cross-family
# equivalence for family-specific secondary metrics (mean_ttft, TTFT
# percentiles, jains_fairness_index, peak_kv_utilization, etc.).
_CANONICAL_SHARED_SECONDARY = ("completion_fraction", "unweighted_slo_success_rate")


def _canonical_policy_id(source_policy_name: str) -> str:
    """Identity mapping: every source policy name observed across all three
    families is already a distinct, already-canonical identifier (no
    collisions, no renaming needed). Kept as an explicit function (rather
    than a bare passthrough) so a future source with non-canonical naming
    has one place to add a real mapping, and so this decision is documented
    rather than implicit."""
    return source_policy_name.strip()


def _scenario_feature_map(spec: SourceSpec) -> Dict[str, Dict[str, str]]:
    """Map source_scenario_id -> raw scenario-level feature dict for one
    source. Family A/B carry a separate scenario_features.csv (one row per
    scenario); Family C embeds its scenario-level fields (bulk_pressure,
    urgent_arrival_phase, urgent_tightness, seed, held_out) directly in
    per_policy_results.csv, so its "scenario features" are derived from the
    first policy row seen for each scenario_id (already validated invariant
    across policy rows for the same scenario by validate_scenario_table)."""
    if spec.scenario_features_path is not None:
        feat_rows = _read_csv_rows(spec.scenario_features_path)
        return {r["scenario_id"]: r for r in feat_rows}
    # No separate file: derive from per_policy_results.csv itself.
    pp_rows = _read_csv_rows(spec.per_policy_results_path)
    out: Dict[str, Dict[str, str]] = {}
    for r in pp_rows:
        out.setdefault(r["scenario_id"], r)
    return out


def build_long_form_rows(sources: Sequence[SourceSpec]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in sources:
        pp_rows = _read_csv_rows(spec.per_policy_results_path)
        # Family A/B carry a separate scenario_features.csv with seed and
        # other scenario-level fields not present in per_policy_results.csv
        # itself; Family C embeds them directly in per_policy_results.csv.
        feature_map = _scenario_feature_map(spec)
        for idx, raw in enumerate(pp_rows):
            source_scenario_id = raw["scenario_id"]
            source_policy_name = raw.get("policy_name") or raw.get("policy")
            canonical_scenario_id = f"{spec.mechanism_family}::{source_scenario_id}"
            mf_psd_row_id = f"{canonical_scenario_id}::{source_policy_name}"
            status = raw.get("status", "unknown") or "unknown"
            anwg = _to_float_or_nan(raw.get(PRIMARY_METRIC)) if status == "success" else float("nan")
            scenario_features_raw = feature_map.get(source_scenario_id, {})
            # 'seed' lives in per_policy_results.csv directly for Family C,
            # and only in scenario_features.csv for Family A/B.
            seed = raw.get("seed") or scenario_features_raw.get("seed", "")
            if spec.mechanism_family == FAMILY_C:
                source_split_raw = (
                    "held_out_eval_seed" if raw.get("held_out") == "True" else "calibration_seed"
                )
            else:
                source_split_raw = "NOT_DESIGNATED"
            row = {
                "mf_psd_row_id": mf_psd_row_id,
                "canonical_scenario_id": canonical_scenario_id,
                "source_scenario_id": source_scenario_id,
                "mechanism_family": spec.mechanism_family,
                "source_run_id": spec.source_run_id,
                "source_result_path": _repo_relpath(spec.per_policy_results_path),
                "source_row_index": idx,
                "group_key": group_key_for_scenario_id(spec.mechanism_family, source_scenario_id),
                "seed": seed,
                "source_split_raw": source_split_raw,
                "source_policy_name": source_policy_name,
                "canonical_policy_id": _canonical_policy_id(source_policy_name),
                "is_canonical_anchor": _canonical_policy_id(source_policy_name)
                in CANONICAL_ANCHOR_POLICIES,
                "status": status,
                "primary_utility_anwg": anwg,
                "secondary_completion_fraction": (
                    _to_float_or_nan(raw.get("completion_fraction")) if status == "success" else float("nan")
                ),
                "secondary_unweighted_slo_success_rate": (
                    _to_float_or_nan(raw.get("unweighted_slo_success_rate"))
                    if status == "success"
                    else float("nan")
                ),
                "source_row_json": json.dumps(raw, sort_keys=True),
                "source_scenario_features_json": json.dumps(scenario_features_raw, sort_keys=True),
                "builder_version": BUILDER_VERSION,
            }
            rows.append(row)
    # Deterministic ordering: family, then scenario, then policy.
    rows.sort(key=lambda r: (r["mechanism_family"], r["canonical_scenario_id"], r["canonical_policy_id"]))
    return rows


# ---------------------------------------------------------------------------
# Scenario-level context table
# ---------------------------------------------------------------------------

# Family-specific learnable context feature columns, each read from the raw
# source CSV(s) and re-emitted under a family-prefixed column name
# (feat_<family letter>__<original column>) so that superficially similarly
# named columns across families (e.g. `max_active_sequences` in both A and
# B) are never silently treated as the same feature. See the MF-PSD audit
# section "Learnable vs Audit-Only Fields" for the reasoning per column.
FAMILY_A_LEARNABLE_SOURCE_COLUMNS = (
    "target_utilization",
    "tenant_weight_skew",
    "favored_tenant_size",
    "other_tenant_size",
    "prediction_noise_sigma",
    "token_length_source",
    "size_priority_alignment",
    "max_active_sequences",
    "stress_control_relationship",
)
FAMILY_B_LEARNABLE_SOURCE_COLUMNS = (
    "hog_count",
    "late_pressure",
    "slo_emphasis",
    "n_total_jobs",
    "n_hog",
    "n_late",
    "step_token_budget",
    "max_active_sequences",
    "hog_prompt_median",
    "late_prompt_median",
    "output_median",
    "late_start_s",
    "slack_hog_s",
    "slack_late_s",
    "tbt_slo_s",
    "arrival_shape",
    "output_intervention",
    "token_sources",
    "mean_e2e_slack_hog",
    "mean_e2e_slack_late",
    "stress_control_relationship",
)
FAMILY_C_LEARNABLE_SOURCE_COLUMNS = (
    "bulk_pressure",
    "urgent_arrival_phase",
    "urgent_tightness",
)

_FAMILY_PREFIX = {FAMILY_A: "feat_A__", FAMILY_B: "feat_B__", FAMILY_C: "feat_C__"}
_FAMILY_LEARNABLE_COLUMNS = {
    FAMILY_A: FAMILY_A_LEARNABLE_SOURCE_COLUMNS,
    FAMILY_B: FAMILY_B_LEARNABLE_SOURCE_COLUMNS,
    FAMILY_C: FAMILY_C_LEARNABLE_SOURCE_COLUMNS,
}

#: Full set of learnable feature column names as they appear in the
#: scenario-level MF-PSD table (family-prefixed). This is the machine
#: readable ALLOWLIST -- selector training in a later step should default to
#: reading only these columns (filtered to the rows of the family/families
#: in scope) as model inputs.
LEARNABLE_FEATURE_ALLOWLIST: Tuple[str, ...] = tuple(
    f"{_FAMILY_PREFIX[fam]}{col}"
    for fam in MECHANISM_FAMILIES
    for col in _FAMILY_LEARNABLE_COLUMNS[fam]
)

SCENARIO_TABLE_IDENTITY_COLUMNS = (
    "canonical_scenario_id",
    "source_scenario_id",
    "mechanism_family",
    "source_run_id",
    "group_key",
    "seed",
    "source_split_raw",
    "n_policies_evaluated",
    "policies_evaluated_json",
    "builder_version",
)

#: Explicit machine-readable DENYLIST: fields that exist somewhere in the
#: unified tables (long-form or scenario-level) but must never become
#: default selector learnable inputs, per the anti-leakage requirements.
FORBIDDEN_AUDIT_ONLY_FIELDS: Tuple[str, ...] = (
    "mf_psd_row_id",
    "canonical_scenario_id",
    "source_scenario_id",
    "mechanism_family",
    "source_run_id",
    "source_result_path",
    "source_row_index",
    "group_key",
    "seed",
    "source_split_raw",
    "source_policy_name",
    "canonical_policy_id",
    "is_canonical_anchor",
    "status",
    "primary_utility_anwg",
    "secondary_completion_fraction",
    "secondary_unweighted_slo_success_rate",
    "source_row_json",
    "source_scenario_features_json",
    "builder_version",
    "n_policies_evaluated",
    "policies_evaluated_json",
)


def build_scenario_table_rows(long_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for r in long_rows:
        by_scenario.setdefault(r["canonical_scenario_id"], []).append(r)

    # Re-derive raw source rows (needed for feature columns not carried on
    # the long-form row) by re-parsing source_row_json.
    scenario_rows: List[Dict[str, Any]] = []
    for canonical_scenario_id, rows_for_scenario in by_scenario.items():
        first = rows_for_scenario[0]
        family = first["mechanism_family"]
        raw_dicts = [json.loads(r["source_scenario_features_json"]) for r in rows_for_scenario]

        out: Dict[str, Any] = {
            "canonical_scenario_id": canonical_scenario_id,
            "source_scenario_id": first["source_scenario_id"],
            "mechanism_family": family,
            "source_run_id": first["source_run_id"],
            "group_key": first["group_key"],
            "seed": first["seed"],
            "source_split_raw": first["source_split_raw"],
            "n_policies_evaluated": len(rows_for_scenario),
            "policies_evaluated_json": json.dumps(
                sorted(r["source_policy_name"] for r in rows_for_scenario)
            ),
            "builder_version": BUILDER_VERSION,
        }

        # Emit every family's learnable feature columns for every scenario
        # row (explicit missingness for families this scenario does not
        # belong to).
        for fam in MECHANISM_FAMILIES:
            prefix = _FAMILY_PREFIX[fam]
            for col in _FAMILY_LEARNABLE_COLUMNS[fam]:
                key = f"{prefix}{col}"
                if fam == family:
                    # Value must be invariant across all policy rows for
                    # this scenario; validated separately in validate_*.
                    out[key] = raw_dicts[0].get(col, "")
                else:
                    out[key] = ""  # explicit missingness: not this scenario's family

        scenario_rows.append(out)

    scenario_rows.sort(key=lambda r: (r["mechanism_family"], r["canonical_scenario_id"]))
    return scenario_rows


SCENARIO_TABLE_COLUMNS = list(SCENARIO_TABLE_IDENTITY_COLUMNS) + list(LEARNABLE_FEATURE_ALLOWLIST)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class MFPSDValidationError(Exception):
    pass


def validate_long_form(rows: Sequence[Dict[str, Any]], sources: Sequence[SourceSpec]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    # No duplicate (canonical_scenario_id, canonical_policy_id) cells.
    cells = [(r["canonical_scenario_id"], r["canonical_policy_id"]) for r in rows]
    dup_cells = len(cells) - len(set(cells))
    report["duplicate_scenario_policy_cells"] = dup_cells
    if dup_cells:
        raise MFPSDValidationError(f"{dup_cells} duplicate (scenario, policy) cells")

    # No duplicate mf_psd_row_id.
    ids = [r["mf_psd_row_id"] for r in rows]
    dup_ids = len(ids) - len(set(ids))
    report["duplicate_row_ids"] = dup_ids
    if dup_ids:
        raise MFPSDValidationError(f"{dup_ids} duplicate mf_psd_row_id values")

    # Exact source-row conservation: total row count == sum of source CSV
    # data-row counts.
    expected_total = 0
    per_family_expected = {}
    for spec in sources:
        n = len(_read_csv_rows(spec.per_policy_results_path))
        per_family_expected[spec.mechanism_family] = n
        expected_total += n
    report["expected_total_rows"] = expected_total
    report["actual_total_rows"] = len(rows)
    if expected_total != len(rows):
        raise MFPSDValidationError(
            f"row count mismatch: expected {expected_total} (sum of source CSVs), got {len(rows)}"
        )
    per_family_actual: Dict[str, int] = {}
    for r in rows:
        per_family_actual[r["mechanism_family"]] = per_family_actual.get(r["mechanism_family"], 0) + 1
    report["per_family_expected_rows"] = per_family_expected
    report["per_family_actual_rows"] = per_family_actual
    if per_family_expected != per_family_actual:
        raise MFPSDValidationError(
            f"per-family row count mismatch: expected {per_family_expected}, got {per_family_actual}"
        )

    # Finite ANWG for every 'success' row; NaN allowed only for non-success.
    non_finite_success = [
        r["mf_psd_row_id"]
        for r in rows
        if r["status"] == "success" and not math.isfinite(r["primary_utility_anwg"])
    ]
    report["non_finite_anwg_on_success_rows"] = len(non_finite_success)
    if non_finite_success:
        raise MFPSDValidationError(
            f"{len(non_finite_success)} success rows have non-finite ANWG: {non_finite_success[:5]}"
        )

    # Every row traces to a known mechanism family / source_run_id.
    known_run_ids = {s.source_run_id for s in sources}
    bad_traceability = [r["mf_psd_row_id"] for r in rows if r["source_run_id"] not in known_run_ids]
    report["untraceable_rows"] = len(bad_traceability)
    if bad_traceability:
        raise MFPSDValidationError(f"{len(bad_traceability)} rows do not trace to a known source run")

    report["forbidden_fields_present_in_long_form"] = sorted(
        c for c in LONG_FORM_COLUMNS if c in LEARNABLE_FEATURE_ALLOWLIST
    )
    if report["forbidden_fields_present_in_long_form"]:
        raise MFPSDValidationError("a forbidden/audit field leaked into the learnable allowlist")

    return report


def validate_scenario_table(
    scenario_rows: Sequence[Dict[str, Any]], long_rows: Sequence[Dict[str, Any]], sources: Sequence[SourceSpec]
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    ids = [r["canonical_scenario_id"] for r in scenario_rows]
    dup = len(ids) - len(set(ids))
    report["duplicate_scenario_ids"] = dup
    if dup:
        raise MFPSDValidationError(f"{dup} duplicate canonical_scenario_id values in scenario table")

    # Exact scenario conservation per family: count of distinct
    # canonical_scenario_id per family == distinct scenario_id count in the
    # source per_policy_results.csv for that family.
    expected_scenarios = {}
    for spec in sources:
        pp_rows = _read_csv_rows(spec.per_policy_results_path)
        expected_scenarios[spec.mechanism_family] = len({r["scenario_id"] for r in pp_rows})
    actual_scenarios: Dict[str, int] = {}
    for r in scenario_rows:
        actual_scenarios[r["mechanism_family"]] = actual_scenarios.get(r["mechanism_family"], 0) + 1
    report["expected_scenarios_per_family"] = expected_scenarios
    report["actual_scenarios_per_family"] = actual_scenarios
    if expected_scenarios != actual_scenarios:
        raise MFPSDValidationError(
            f"scenario count mismatch: expected {expected_scenarios}, got {actual_scenarios}"
        )

    # Every canonical_scenario_id in long_rows appears exactly once in
    # scenario_rows, and vice versa.
    long_scenario_ids = {r["canonical_scenario_id"] for r in long_rows}
    scenario_ids = set(ids)
    report["scenario_ids_only_in_long"] = sorted(long_scenario_ids - scenario_ids)
    report["scenario_ids_only_in_scenario_table"] = sorted(scenario_ids - long_scenario_ids)
    if long_scenario_ids != scenario_ids:
        raise MFPSDValidationError("scenario ID sets differ between long-form and scenario-level tables")

    # Scenario features invariant across policy rows for the same scenario
    # (re-checked directly against raw source rows, independent of the
    # scenario-table construction that already assumes this).
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for r in long_rows:
        by_scenario.setdefault(r["canonical_scenario_id"], []).append(r)
    variant_scenarios = []
    for canonical_scenario_id, rs in by_scenario.items():
        family = rs[0]["mechanism_family"]
        cols = _FAMILY_LEARNABLE_COLUMNS[family]
        raw_dicts = [json.loads(r["source_scenario_features_json"]) for r in rs]
        for col in cols:
            vals = {d.get(col) for d in raw_dicts}
            if len(vals) > 1:
                variant_scenarios.append((canonical_scenario_id, col, sorted(map(str, vals))))
    report["scenarios_with_non_invariant_features"] = len(variant_scenarios)
    if variant_scenarios:
        raise MFPSDValidationError(
            f"{len(variant_scenarios)} scenarios have a learnable feature that varies "
            f"across policy rows: {variant_scenarios[:5]}"
        )

    report["forbidden_fields_present_as_learnable"] = sorted(
        c
        for c in SCENARIO_TABLE_IDENTITY_COLUMNS
        if c in LEARNABLE_FEATURE_ALLOWLIST
    )
    if report["forbidden_fields_present_as_learnable"]:
        raise MFPSDValidationError("an audit-only scenario column leaked into the learnable allowlist")

    return report


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_mf_psd(
    output_dir: Path,
    sources: Optional[Sequence[SourceSpec]] = None,
) -> Dict[str, Any]:
    """Deterministically (re)build the MF-PSD v1 long-form + scenario-level
    tables plus schema/provenance manifests into `output_dir`. Returns a
    dict summary (also written as build_manifest.json)."""
    sources = tuple(sources) if sources is not None else default_source_specs()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_rows = build_long_form_rows(sources)
    scenario_rows = build_scenario_table_rows(long_rows)

    long_validation = validate_long_form(long_rows, sources)
    scenario_validation = validate_scenario_table(scenario_rows, long_rows, sources)

    long_path = output_dir / "mf_psd_long_v1.csv"
    scenario_path = output_dir / "mf_psd_scenarios_v1.csv"
    _write_csv(long_path, long_rows, LONG_FORM_COLUMNS)
    _write_csv(scenario_path, scenario_rows, SCENARIO_TABLE_COLUMNS)

    schema = {
        "builder_version": BUILDER_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "mechanism_families": list(MECHANISM_FAMILIES),
        "canonical_anchor_policies": list(CANONICAL_ANCHOR_POLICIES),
        "long_form_columns": LONG_FORM_COLUMNS,
        "scenario_table_columns": SCENARIO_TABLE_COLUMNS,
        "scenario_table_identity_audit_columns": list(SCENARIO_TABLE_IDENTITY_COLUMNS),
        "learnable_feature_allowlist": list(LEARNABLE_FEATURE_ALLOWLIST),
        "forbidden_audit_only_fields": list(FORBIDDEN_AUDIT_ONLY_FIELDS),
        "notes": (
            "learnable_feature_allowlist is the default selector-input column "
            "set for the SCENARIO-LEVEL table only. mechanism_family itself is "
            "deliberately excluded from the allowlist (it is retained as audit "
            "metadata for leave-one-family-out evaluation, never a default "
            "learnable input). All family-specific feature columns are "
            "family-prefixed (feat_A__/feat_B__/feat_C__) and are explicitly "
            "missing (empty string) for scenarios belonging to a different "
            "family -- see the MF-PSD audit doc for why superficially similar "
            "column names (e.g. max_active_sequences in both A and B) were NOT "
            "merged into one shared column."
        ),
    }
    schema_path = output_dir / "mf_psd_schema_v1.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2, sort_keys=True)
        f.write("\n")

    provenance = {
        "builder_version": BUILDER_VERSION,
        "build_git_head_sha": _git_head_sha(),
        "build_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": [
            {
                "mechanism_family": s.mechanism_family,
                "source_run_id": s.source_run_id,
                "run_dir": _repo_relpath(s.run_dir),
                "per_policy_results_path": _repo_relpath(s.per_policy_results_path),
                "per_policy_results_sha256": _sha256_of_file(s.per_policy_results_path),
                "scenario_features_path": (
                    _repo_relpath(s.scenario_features_path) if s.scenario_features_path else None
                ),
                "scenario_features_sha256": (
                    _sha256_of_file(s.scenario_features_path) if s.scenario_features_path else None
                ),
                "launch_git_sha": s.launch_git_sha,
                "launch_git_branch": s.launch_git_branch,
                "audit_doc": s.audit_doc,
                "design_doc": s.design_doc,
                "family_verdict": s.family_verdict,
                "canonical_anchor_policies": list(s.canonical_anchor_policies),
                "extra_policies": list(s.extra_policies),
                "notes": s.notes,
            }
            for s in sources
        ],
        "output_files": {
            "mf_psd_long_v1.csv": _sha256_of_file(long_path),
            "mf_psd_scenarios_v1.csv": _sha256_of_file(scenario_path),
            "mf_psd_schema_v1.json": _sha256_of_file(schema_path),
        },
    }
    provenance_path = output_dir / "mf_psd_provenance_v1.json"
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    manifest = {
        "builder_version": BUILDER_VERSION,
        "n_long_form_rows": len(long_rows),
        "n_scenarios": len(scenario_rows),
        "per_family_scenario_counts": scenario_validation["actual_scenarios_per_family"],
        "per_family_row_counts": long_validation["per_family_actual_rows"],
        "long_form_validation": long_validation,
        "scenario_table_validation": scenario_validation,
        "output_dir": _repo_relpath(output_dir),
    }
    manifest_path = output_dir / "mf_psd_build_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return manifest
