"""Public Trace Replay Scenarios v1 -- Layer 2 (canonical replay scenarios)
and Layer 3 (same-scenario multi-policy outcomes) builder.

Implements the frozen preregistration in
`docs/design/PUBLIC_TRACE_REPLAY_SCENARIOS_V1.md`. Turns the already-complete
Public Trace Corpus v1 (Layer 0/1) into `PolicySeparationScenario` objects and
evaluates them under the existing frozen six-policy portfolio, reusing the
existing `unified_utility_matrix.run_cell`-style evaluation pattern.

Two evidence-class views (design doc SS2), never pooled without the label:
  - PUBLIC_TRACE_FAITHFUL: only arrival_time/prompt_tokens/actual_output_tokens
    are real; only full_prefill/chunked_prefill_small (the two policies that
    read no synthesized field) are evaluated.
  - PUBLIC_TRACE_DERIVED_WITH_CONTROLLED_ANNOTATIONS: adds the deterministic,
    outcome-blind overlays of design doc SS3 (predicted_output_tokens via the
    project's existing apply_prediction_noise, slo_deadline via a
    workload-relative rule, uniform priority, class_id=source_dataset); all
    six canonical anchors are evaluated.

Window/scenario selection (design doc SS4) is fixed before any replay is run
and never depends on a replay outcome.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.types import GPUConfig
from ..policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA
from .builders import req
from .schema import PolicySeparationScenario
from .templates_fairness_starvation_v2 import apply_prediction_noise
from .unified_utility_matrix import CANONICAL_ANCHOR_IDS

BUILDER_VERSION = "public_trace_replay_v1.0.0"

ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = ROOT / "data" / "public_trace_corpus_v1"

SOURCES: Tuple[str, ...] = ("burstgpt", "azure_2023_conv", "azure_2023_code")

# -- frozen constants (design doc SS4/SS3) -----------------------------------
WINDOW_SIZE = 200
WINDOWS_PER_SOURCE = 20
SEED = 20260820
PREDICTION_NOISE_SIGMA = 0.30
SLACK_MULTIPLIER = 1.0
STEP_SIZE = 0.001

FAITHFUL = "PUBLIC_TRACE_FAITHFUL"
AUGMENTED = "PUBLIC_TRACE_DERIVED_WITH_CONTROLLED_ANNOTATIONS"
EVIDENCE_CLASSES = (FAITHFUL, AUGMENTED)
_EVIDENCE_CLASS_SHORT = {FAITHFUL: "faithful", AUGMENTED: "augmented"}

FAITHFUL_POLICIES: Tuple[str, ...] = ("full_prefill", "chunked_prefill_small")
AUGMENTED_POLICIES: Tuple[str, ...] = tuple(CANONICAL_ANCHOR_IDS)
assert set(FAITHFUL_POLICIES) <= set(AUGMENTED_POLICIES)

POLICIES_BY_EVIDENCE_CLASS: Dict[str, Tuple[str, ...]] = {
    FAITHFUL: FAITHFUL_POLICIES,
    AUGMENTED: AUGMENTED_POLICIES,
}

NATIVE = "NATIVE"
DETERMINISTIC_DERIVED = "DETERMINISTIC_DERIVED"
EXPERIMENTAL_CONTROLLED_ANNOTATION = "EXPERIMENTAL_CONTROLLED_ANNOTATION"

GPU_CONFIG_KWARGS = dict(max_active_sequences=512, max_batch_tokens=512, max_kv_tokens=8_000_000)
SERVICE_MODEL_KWARGS = dict(
    step_size=STEP_SIZE,
    enable_prefill_modeling=True,
    prefill_cost_per_token=1.0,
    step_token_budget=512,
    enable_decode_prefill_contention=True,
    decode_first=False,  # overridden per policy by the runner, matching Family-B v2's own pattern
)


# ---------------------------------------------------------------------------
# Layer 1 loading + window selection (deterministic, outcome-blind)
# ---------------------------------------------------------------------------

def load_source_records(source: str) -> pd.DataFrame:
    """Loads one source's Layer-1 parquet, verifying (not assuming) the
    monotonic-arrival-order invariant `schema.json` declares."""
    path = CORPUS_DIR / source / "records.parquet"
    df = pd.read_parquet(path)
    assert (df["relative_arrival_time"].diff().dropna() >= 0).all(), (
        f"{source}: relative_arrival_time is not monotonically non-decreasing"
    )
    return df.reset_index(drop=True)


def select_window_indices(n_records: int, window_size: int, n_windows: int) -> List[int]:
    """Design doc SS4: deterministic even-spacing across the source's full
    available window count, never based on any replay outcome."""
    n_available = n_records // window_size
    if n_available <= 0:
        return []
    n_windows = min(n_windows, n_available)
    stride = max(1, n_available // n_windows)
    return [i * stride for i in range(n_windows)][:n_windows]


def extract_window(df: pd.DataFrame, window_index: int, window_size: int) -> pd.DataFrame:
    start = window_index * window_size
    end = start + window_size
    window = df.iloc[start:end].reset_index(drop=True)
    assert len(window) == window_size, f"short window at index {window_index}: {len(window)} rows"
    return window


# ---------------------------------------------------------------------------
# Scenario construction (design doc SS3/SS4)
# ---------------------------------------------------------------------------

def _service_est(prompt_tokens: np.ndarray, predicted_output_tokens: np.ndarray) -> np.ndarray:
    return DEFAULT_ALPHA * prompt_tokens + DEFAULT_BETA * predicted_output_tokens


def build_scenario_from_window(
    window: pd.DataFrame, *, source: str, window_index: int, evidence_class: str,
) -> Tuple[PolicySeparationScenario, Dict[str, str]]:
    """Builds one `PolicySeparationScenario` for one (source, window,
    evidence_class), plus a per-field provenance map (design doc SS3)."""
    assert evidence_class in EVIDENCE_CLASSES

    arrival = window["relative_arrival_time"].to_numpy(dtype=float)
    arrival = arrival - arrival[0]  # design doc SS4: rebase so window starts at t=0
    prompt_tokens = window["prompt_tokens"].fillna(1).clip(lower=1).to_numpy(dtype=int)
    actual_output_tokens = window["output_tokens"].fillna(1).clip(lower=1).to_numpy(dtype=int)

    field_provenance = {
        "arrival_time": NATIVE,
        "prompt_tokens": NATIVE,
        "actual_output_tokens": NATIVE,
    }

    if evidence_class == FAITHFUL:
        predicted_output_tokens = actual_output_tokens.copy()
        priority = np.ones(len(window), dtype=float)
        class_id = np.full(len(window), "default", dtype=object)
        field_provenance["predicted_output_tokens"] = DETERMINISTIC_DERIVED
        field_provenance["slo_deadline"] = DETERMINISTIC_DERIVED  # unused by either faithful-view policy
        field_provenance["priority"] = DETERMINISTIC_DERIVED
        field_provenance["class_id"] = DETERMINISTIC_DERIVED
        slo_deadline = arrival + 1_000.0  # inert: no faithful-view policy reads it (design doc SS4 NO_PRESSURE_SLACK convention)
    else:
        rng = np.random.default_rng([SEED, hash((source, window_index)) & 0xFFFFFFFF])
        predicted_output_tokens = np.round(
            apply_prediction_noise(rng, actual_output_tokens, PREDICTION_NOISE_SIGMA)
        ).astype(int)
        service_est = _service_est(prompt_tokens.astype(float), predicted_output_tokens.astype(float))
        slo_deadline = arrival + service_est * (1.0 + SLACK_MULTIPLIER)
        priority = np.ones(len(window), dtype=float)
        class_id = np.full(len(window), source, dtype=object)
        field_provenance["predicted_output_tokens"] = EXPERIMENTAL_CONTROLLED_ANNOTATION
        field_provenance["slo_deadline"] = EXPERIMENTAL_CONTROLLED_ANNOTATION
        field_provenance["priority"] = EXPERIMENTAL_CONTROLLED_ANNOTATION
        field_provenance["class_id"] = EXPERIMENTAL_CONTROLLED_ANNOTATION

    requests = tuple(
        req(
            request_id=i,
            arrival_time=float(arrival[i]),
            prompt_tokens=int(prompt_tokens[i]),
            predicted_output_tokens=int(predicted_output_tokens[i]),
            actual_output_tokens=int(actual_output_tokens[i]),
            slo_deadline=float(slo_deadline[i]),
            priority=float(priority[i]),
            class_id=str(class_id[i]),
        )
        for i in range(len(window))
    )

    gpu_configs = (GPUConfig(gpu_id=0, **GPU_CONFIG_KWARGS),)
    short = _EVIDENCE_CLASS_SHORT[evidence_class]
    scenario_id = f"{source}::w{window_index}::{short}"

    scenario = PolicySeparationScenario(
        scenario_id=scenario_id,
        family="PUBLIC_TRACE_REPLAY_V1",
        template_name="public_trace_replay_v1.build_scenario_from_window",
        generator_version=BUILDER_VERSION,
        seed=SEED,
        params={
            "source": source,
            "window_index": window_index,
            "evidence_class": evidence_class,
            "window_size": WINDOW_SIZE,
        },
        requests=requests,
        gpu_configs=gpu_configs,
        service_model_kwargs=dict(SERVICE_MODEL_KWARGS),
        target_policy_family="PUBLIC_TRACE_REPLAY_V1",
        expected_qualitative_hypothesis=(
            "Natural policy separation under externally-grounded workload structure; "
            "not preregistered to favor any particular policy."
        ),
    )
    return scenario, field_provenance


def canonical_scenario_id(source: str, window_index: int, evidence_class: str) -> str:
    return f"PUBLIC_TRACE::{source}::w{window_index}::{_EVIDENCE_CLASS_SHORT[evidence_class]}"


def build_all_scenarios() -> List[Dict[str, Any]]:
    """Design doc SS4: 20 windows/source x 3 sources x 2 evidence classes =
    120 canonical scenario records. Deterministic, no replay performed here."""
    records: List[Dict[str, Any]] = []
    for source in SOURCES:
        df = load_source_records(source)
        window_indices = select_window_indices(len(df), WINDOW_SIZE, WINDOWS_PER_SOURCE)
        for window_index in window_indices:
            window = extract_window(df, window_index, WINDOW_SIZE)
            for evidence_class in EVIDENCE_CLASSES:
                scenario, field_provenance = build_scenario_from_window(
                    window, source=source, window_index=window_index, evidence_class=evidence_class,
                )
                records.append({
                    "canonical_scenario_id": canonical_scenario_id(source, window_index, evidence_class),
                    "source_dataset": source,
                    "window_index": window_index,
                    "scenario_evidence_class": evidence_class,
                    "scenario": scenario,
                    "field_provenance": field_provenance,
                    "applicable_policies": list(POLICIES_BY_EVIDENCE_CLASS[evidence_class]),
                })
    return records


# ---------------------------------------------------------------------------
# Layer 3 evaluation (reuses unified_utility_matrix's run_cell pattern)
# ---------------------------------------------------------------------------

class _TrajectoryLoggingPolicy:
    """Thin, generic per-step observation wrapper around any `BasePolicy`.

    Delegates every decision to `inner_policy.select_action` unchanged (pure
    observation -- the wrapped policy's actual scheduling behavior is never
    altered), and appends one summary row per step to `sink` (design doc SS6
    Layer-4 schema). Not specific to any one policy, unlike
    `LiveHierarchicalRouterPolicy` (which is router-specific); generic so it
    can wrap any of the six frozen baseline policies without reimplementing
    admission logic.
    """

    def __init__(self, inner_policy, *, canonical_scenario_id: str, source_dataset: str,
                 scenario_evidence_class: str, policy_id: str, sink: List[Dict[str, Any]]) -> None:
        self.inner_policy = inner_policy
        self.name = getattr(inner_policy, "name", policy_id)
        self._canonical_scenario_id = canonical_scenario_id
        self._source_dataset = source_dataset
        self._scenario_evidence_class = scenario_evidence_class
        self._policy_id = policy_id
        self._sink = sink

    def select_action(self, state):  # ObservableState -> Action
        action = self.inner_policy.select_action(state)
        n_active = sum(len(g.active_request_ids) for g in state.gpu_states)
        kv_utils = [g.current_kv_tokens / max(g.max_kv_tokens, 1) for g in state.gpu_states]
        n_admitted = sum(len(ids) for ids in action.admit.values())
        self._sink.append({
            "canonical_scenario_id": self._canonical_scenario_id,
            "source_dataset": self._source_dataset,
            "scenario_evidence_class": self._scenario_evidence_class,
            "policy_id": self._policy_id,
            "step": state.step,
            "time": state.time,
            "queue_length": len(state.waiting_queue),
            "active_request_count": n_active,
            "mean_kv_utilization": float(np.mean(kv_utils)) if kv_utils else 0.0,
            "max_kv_utilization": float(np.max(kv_utils)) if kv_utils else 0.0,
            "completed_count": state.completed_count,
            "admitted_count": n_admitted,
            "admitted_request_ids": sorted(
                rid for ids in action.admit.values() for rid in ids
            ),
        })
        return action


def evaluate_scenario_policy(
    scenario: PolicySeparationScenario, policy_id: str, *,
    capture_trajectory: bool = False,
    canonical_scenario_id: str = "",
    source_dataset: str = "",
    scenario_evidence_class: str = "",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One (scenario, policy) cell -- structurally identical to
    unified_utility_matrix.run_cell (reused, not re-derived) for the core
    admission/metrics path, extended with the fuller RunMetrics field set
    (design doc SS5/task SS3) and optional Layer-4 trajectory capture
    (design doc SS6). Never raises: failures are captured in the row's
    status/error fields. Returns (row, trajectory_rows); trajectory_rows is
    always [] when capture_trajectory=False."""
    from ..simulator.service_model import ServiceModel
    from ..simulator.simulator import Simulator, SimulatorConfig
    from .unified_utility_matrix import _build_policy

    row: Dict[str, Any] = {
        "canonical_scenario_id": canonical_scenario_id,
        "source_dataset": source_dataset,
        "scenario_evidence_class": scenario_evidence_class,
        "canonical_policy_id": policy_id,
        "builder_version": BUILDER_VERSION,
    }
    trajectory_rows: List[Dict[str, Any]] = []
    try:
        policy, sm_override = _build_policy(policy_id)
        if capture_trajectory:
            policy = _TrajectoryLoggingPolicy(
                policy,
                canonical_scenario_id=canonical_scenario_id,
                source_dataset=source_dataset,
                scenario_evidence_class=scenario_evidence_class,
                policy_id=policy_id,
                sink=trajectory_rows,
            )
        merged_sm = dict(scenario.service_model_kwargs)
        merged_sm.update(sm_override)
        sim = Simulator(SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**merged_sm),
        ))
        sim.load_trace(list(scenario.requests))
        metrics = sim.run(policy, workload_tag=scenario.scenario_id, seed=scenario.seed)
        row.update({
            "status": "success",
            "primary_utility_anwg": float(metrics.arrival_normalized_weighted_goodput),
            "secondary_completion_fraction": float(metrics.completion_fraction),
            "secondary_weighted_completion_fraction": float(metrics.weighted_completion_fraction),
            "num_total": int(metrics.num_total),
            "num_completed": int(metrics.num_completed),
            "num_dropped": int(metrics.num_dropped),
            "num_slo_violated": int(metrics.num_slo_violated),
            "slo_violation_rate": float(metrics.slo_violation_rate),
            "weighted_goodput": float(metrics.weighted_goodput),
            "mean_latency": float(metrics.mean_latency),
            "median_latency": float(metrics.median_latency),
            "p95_latency": float(metrics.p95_latency),
            "p99_latency": float(metrics.p99_latency),
            "mean_queuing_delay": float(metrics.mean_queuing_delay),
            "mean_ttft": float(metrics.mean_ttft),
            "mean_tpot": float(metrics.mean_tpot),
            "error": "",
        })
    except Exception as e:  # noqa: BLE001
        row.update({
            "status": "failed",
            "primary_utility_anwg": float("nan"),
            "secondary_completion_fraction": float("nan"),
            "secondary_weighted_completion_fraction": float("nan"),
            "num_total": None, "num_completed": None, "num_dropped": None, "num_slo_violated": None,
            "slo_violation_rate": float("nan"), "weighted_goodput": float("nan"),
            "mean_latency": float("nan"), "median_latency": float("nan"),
            "p95_latency": float("nan"), "p99_latency": float("nan"),
            "mean_queuing_delay": float("nan"), "mean_ttft": float("nan"), "mean_tpot": float("nan"),
            "error": f"{type(e).__name__}: {e}",
        })
        trajectory_rows = []
    return row, trajectory_rows


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Canonical cell keys / expected key set (design doc SS5 -- frozen)
# ---------------------------------------------------------------------------

def canonical_cell_key(canonical_scenario_id: str, policy_id: str) -> str:
    return f"{canonical_scenario_id}::{policy_id}"


def expected_cell_keys(records: List[Dict[str, Any]]) -> List[str]:
    """The exact, frozen 480-key set (design doc SS5), in the same
    deterministic order `build_all_scenarios()` already produces -- never
    reordered by any observed result."""
    keys: List[str] = []
    for r in records:
        for policy_id in r["applicable_policies"]:
            keys.append(canonical_cell_key(r["canonical_scenario_id"], policy_id))
    return keys


# ---------------------------------------------------------------------------
# Checkpoint I/O (JSONL, one line per completed cell -- design doc/task SS4)
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path) -> Dict[str, Dict[str, Any]]:
    """Reads an existing Layer-3 JSONL checkpoint into
    `{cell_key: row}`, keeping only the LAST well-formed entry per key (a
    later entry is assumed to be a resumed retry of an earlier, possibly
    interrupted, attempt). A truncated/corrupt trailing line (torn write)
    is skipped, not treated as the run's most current cell -- and is
    reported by the caller via `checkpoint_corruption_lines`, not silently
    dropped."""
    import json

    if not checkpoint_path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # corrupt/torn line; caller re-scans for these separately
            key = canonical_cell_key(row["canonical_scenario_id"], row["canonical_policy_id"])
            rows[key] = row
    return rows


def scan_checkpoint_corruption(checkpoint_path: Path) -> List[int]:
    """Returns 1-indexed line numbers that failed to parse as JSON (torn
    writes from an interrupted run) -- reported explicitly, never silently
    ignored."""
    import json

    if not checkpoint_path.exists():
        return []
    bad: List[int] = []
    with open(checkpoint_path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad.append(i)
    return bad


def append_checkpoint_row(checkpoint_path: Path, row: Dict[str, Any]) -> None:
    """Appends one cell result as a single JSON line, flushed and fsync'd
    immediately, so an interrupted process loses at most the one
    in-progress cell -- never corrupts already-completed cells (each is its
    own line, on its own disk write)."""
    import json

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        f.flush()
        import os
        os.fsync(f.fileno())


def is_valid_success_row(row: Dict[str, Any]) -> bool:
    """A resumable-skip is only valid for a well-formed, successful row --
    anything else (missing fields, status != success) must be recomputed,
    never silently skipped (task SS4/SS7)."""
    if row.get("status") != "success":
        return False
    required = (
        "canonical_scenario_id", "source_dataset", "scenario_evidence_class",
        "canonical_policy_id", "primary_utility_anwg", "secondary_completion_fraction",
    )
    if not all(k in row for k in required):
        return False
    anwg = row.get("primary_utility_anwg")
    if anwg is None or (isinstance(anwg, float) and (anwg != anwg)):  # NaN check, no numpy needed
        return False
    return True


def write_trajectory_parquet(traj_dir: Path, canonical_scenario_id: str, policy_id: str,
                              rows: List[Dict[str, Any]]) -> Optional[Path]:
    """Writes one cell's trajectory as its own parquet file (atomic via a
    temp-file + rename), rather than appending into one large file -- this
    makes each cell's trajectory independently resumable/verifiable and
    avoids parquet's lack of cheap incremental-append semantics. Returns
    None (writes nothing) if `rows` is empty (e.g. a failed cell)."""
    if not rows:
        return None
    traj_dir.mkdir(parents=True, exist_ok=True)
    safe_scenario = canonical_scenario_id.replace("::", "__")
    out_path = traj_dir / f"{safe_scenario}__{policy_id}.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    pd.DataFrame(rows).to_parquet(tmp_path, index=False)
    tmp_path.rename(out_path)  # atomic on the same filesystem
    return out_path


# ---------------------------------------------------------------------------
# Final integrity validation (design doc SS7 / task SS5)
# ---------------------------------------------------------------------------

def validate_full_result_set(
    expected_keys: List[str], checkpoint_rows: Dict[str, Dict[str, Any]], checkpoint_path: Path,
) -> Dict[str, Any]:
    """Design doc SS7 / task's finalizer requirement: refuses to claim a
    complete result set unless every expected cell is present and
    well-formed (either a valid success or an explicit recorded failure --
    a MISSING key, not present at all, is always an integrity failure)."""
    expected_set = set(expected_keys)
    present_set = set(checkpoint_rows.keys())
    missing = sorted(expected_set - present_set)
    unexpected = sorted(present_set - expected_set)
    duplicates_in_expected = len(expected_keys) != len(expected_set)
    corrupt_lines = scan_checkpoint_corruption(checkpoint_path)

    n_success = sum(1 for k in expected_set & present_set if checkpoint_rows[k].get("status") == "success")
    n_failed = sum(1 for k in expected_set & present_set if checkpoint_rows[k].get("status") == "failed")

    ok = (
        not missing
        and not unexpected
        and not duplicates_in_expected
        and not corrupt_lines
        and (n_success + n_failed) == len(expected_set)
    )
    return {
        "ok": ok,
        "n_expected": len(expected_set),
        "n_present": len(present_set),
        "n_success": n_success,
        "n_failed": n_failed,
        "n_missing": len(missing),
        "missing_keys": missing,
        "n_unexpected": len(unexpected),
        "unexpected_keys": unexpected,
        "duplicate_keys_in_expected_set": duplicates_in_expected,
        "corrupt_checkpoint_lines": corrupt_lines,
    }
