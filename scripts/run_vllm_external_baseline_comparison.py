#!/usr/bin/env python3
"""
External-admission-control comparison on top of a real vLLM server.

Compares this project's own scheduling policies (src/llmserveopt/policies/),
used here as CLIENT-SIDE admission controllers gating a fixed concurrency
budget, against naive direct submission — all issuing real HTTP requests to
the SAME vLLM server, over the IDENTICAL fixed request plan (arrivals,
prompts, priorities, deadlines, target output lengths) per policy, so
differences in outcome are attributable to admission ORDER, not workload
variance.

**What this is:** an external admission-control layer in front of vLLM.
This project's policies (`select_action(ObservableState) -> Action`) were
built for the discrete-event simulator, where a GPU's `ObservableGPUState`
reflects true internal batch/KV-cache state. Here, `ObservableGPUState` is
reconstructed from only what a client can observe: how many of ITS OWN
requests are currently in flight (bounded by a chosen concurrency budget).
This is a legitimate, different measurement — admission-order effects under
a client-side concurrency cap — not a reproduction of vLLM's own internal
batching/scheduling, which remains invisible from outside (see
docs/real_llm_simulator_integration_plan.md and
docs/vllm_real_serving_external_baseline_pilot.md for the exact boundary).

**What is NOT wired (do not fake):** the trained selector and any
"generated heuristic" from Phase 2B require simulator-internal features
(KV pressure, batch composition, etc.) that this client-side harness cannot
observe against a real server. Requesting them via --policies raises a
clear error rather than silently substituting a fixed baseline.

**"vllm_direct" vs. the policies:** vllm_direct submits requests in strict
arrival order bounded by the same concurrency cap, with NO ObservableState/
Action machinery at all. It is expected to behave identically to `fifo`
(which also admits oldest-arrived-first) — any observed difference between
them indicates overhead or a bug in the policy-admission loop itself, which
is itself a useful sanity signal, not noise to discard.

No hosted API is ever called (no Cohere/Gemini/OpenAI/Azure). The only
network target is a vLLM server (local subprocess or --server-url).

Usage (dry-run):
    python scripts/run_vllm_external_baseline_comparison.py \\
        --dry-run --output-dir experiments/real_llm/vllm_baseline_comparison_pilot_DRYRUN

Usage (mock, no network):
    python scripts/run_vllm_external_baseline_comparison.py \\
        --mock --output-dir /tmp/x

Usage (live, against an already-running vLLM server):
    python scripts/run_vllm_external_baseline_comparison.py \\
        --allow-live-server --server-url http://127.0.0.1:8000 \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output-dir experiments/real_llm/vllm_baseline_comparison_pilot_<timestamp>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402
from llmserveopt.core.types import (  # noqa: E402
    Request, ObservableRequest, ObservableGPUState, ObservableState,
)
from llmserveopt.policies.registry import make_policy  # noqa: E402
from llmserveopt.selector.candidates import SELECTOR_CANDIDATES  # noqa: E402
from llmserveopt.selector.features import extract_features, FeatureMode  # noqa: E402
from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector  # noqa: E402
from llmserveopt.workloads.synthetic import DEFAULT_SLO_CLASSES, SLOClass  # noqa: E402
import run_vllm_serving_baseline_pilot as vllm_mod  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Policies this harness can actually run against a real server today.
WIRED_POLICIES = (
    "vllm_direct", "fifo", "edf", "shortest_output_first",
    "least_laxity_first", "estimated_service_time_first",
)

# "selector" is neither always-wired nor never-wired: it requires
# --selector-artifact to point at a manifest-validated, corrected-objective
# artifact (see load_and_validate_selector_artifact). Requesting it without
# a valid artifact fails clearly before the benchmark starts.
CONDITIONAL_POLICIES = ("selector",)

# ---------------------------------------------------------------------------
# Selector action space vs. what the harness can dispatch
# ---------------------------------------------------------------------------
#
# The selector's output range is SELECTOR_CANDIDATES (every online-deployable
# registered baseline, oracle excluded). When --policies includes "selector",
# run_cell_for_policy dispatches each chosen sub-policy via make_policy(). For
# that to be honest and never fail mid-run, EVERY label the selector can emit
# must be constructible here. SELECTOR_DISPATCHABLE records exactly that set,
# and the import-time check below fails loudly if a candidate the selector
# could emit is not make_policy-constructible (e.g. a future registry change).
#
# This is the Option-1 fix for the Phase-2C selector action-space mismatch:
# the selector's action space stays honest and unchanged; the harness is made
# to execute the whole of it, and a preflight (preflight_selector_action_space)
# verifies this over the actual request plan BEFORE any live request is sent,
# so a request can never be dropped merely because a sub-policy adapter is
# missing. See docs/vllm_real_serving_scaled_comparison.md.
SELECTOR_DISPATCHABLE = tuple(SELECTOR_CANDIDATES)


def _policy_constructible(name: str) -> bool:
    """True iff make_policy(name) succeeds. Used by the import-time action-space
    check and the runtime preflight -- a selector may only emit labels for
    which this holds, or the harness could drop requests with no adapter."""
    try:
        make_policy(name)
        return True
    except Exception:  # noqa: BLE001 - any construction failure means not dispatchable
        return False


# Verified at import (not just at test time): every label the selector can emit
# must be dispatchable by this harness. If this ever fails, a selector run could
# hit an unconstructible sub-policy mid-benchmark -- catch it at import instead.
_unbuildable = [name for name in SELECTOR_DISPATCHABLE if not _policy_constructible(name)]
assert not _unbuildable, (
    "SELECTOR_DISPATCHABLE contains labels make_policy() cannot construct: "
    f"{_unbuildable}. The selector could emit these; every selector output "
    "must be dispatchable by this harness (see Option-1 action-space fix)."
)

# ---------------------------------------------------------------------------
# Arrival/SLO regimes
# ---------------------------------------------------------------------------
#
# Each regime is a (SLO-class config, arrival-timing pattern) pair. Honesty
# constraint: this harness is a client-side external-admission controller
# issuing real HTTP requests -- it cannot fabricate server-side batching
# dynamics, so "burstiness" here means genuine client-side arrival-time
# clustering (requests become admission-eligible at different real
# wall-clock offsets, checked against actual time.monotonic() in the
# admission loop), not a post-hoc relabeling of a single burst.
#
#   steady_moderate: arrivals spread as Poisson-process order statistics
#     over a short window (genuine light queueing), moderate/default SLOs.
#   bursty_tight: all arrivals become eligible simultaneously (maximal
#     burstiness -- the whole cell hits the admission queue at once) with
#     tighter deadlines than steady_moderate.
#   overloaded_mixed_priority: also simultaneous arrival (same reasoning as
#     bursty_tight), with the tightest deadlines and the widest priority
#     spread of the three regimes, producing more borderline SLO cases.
#     Overload pressure comes from deadline tightness + priority variance
#     at fixed concurrency, not from inflating request volume beyond what
#     was requested via --requests-per-cell.
REGIME_SLO_CLASSES: Dict[str, List[SLOClass]] = {
    "steady_moderate": DEFAULT_SLO_CLASSES,
    "bursty_tight": [
        SLOClass("tight", slo_slack=0.25, priority=3.0, weight=0.35),
        SLOClass("medium", slo_slack=1.0, priority=2.0, weight=0.45),
        SLOClass("loose", slo_slack=4.0, priority=1.0, weight=0.20),
    ],
    "overloaded_mixed_priority": [
        SLOClass("tight", slo_slack=0.2, priority=4.0, weight=0.40),
        SLOClass("medium", slo_slack=0.8, priority=2.0, weight=0.30),
        SLOClass("loose", slo_slack=3.0, priority=1.0, weight=0.30),
    ],
}
DEFAULT_REGIMES = ("steady_moderate",)

# Window (seconds) over which steady_moderate spreads a cell's arrivals --
# order statistics of Uniform(0, window), the correct conditional
# distribution of n Poisson-process arrival times given exactly n arrivals
# in [0, window]. Chosen to be the same order of magnitude as this model's
# observed per-request server latency (~0.7-1s), so real queueing pressure
# can appear without arrivals being so spread out that the cell never
# overlaps in time at all.
STEADY_ARRIVAL_WINDOW_S = 2.0


def _regime_arrival_times(regime: str, n: int, rng) -> List[float]:
    if regime == "steady_moderate":
        return sorted(rng.uniform(0.0, STEADY_ARRIVAL_WINDOW_S) for _ in range(n))
    if regime in ("bursty_tight", "overloaded_mixed_priority"):
        return [0.0] * n
    raise ValueError(f"Unknown regime: {regime!r}. Known: {sorted(REGIME_SLO_CLASSES)}")


# "vllm_default" is the natural name for "no external admission control at
# all" from the vLLM side; it is mechanically identical to vllm_direct here
# (see the module docstring's "vllm_direct vs. the policies" section).
POLICY_ALIASES = {"vllm_default": "vllm_direct", "llf": "least_laxity_first", "estf": "estimated_service_time_first"}


def normalize_policy_name(name: str) -> str:
    return POLICY_ALIASES.get(name, name)

# Requested-but-not-implemented policies and why, surfaced verbatim to the
# user rather than silently ignored or faked.
#
# UPDATE (corrected-objective selector-artifact persistence): the ML selector gap
# described below is CLOSED for one selector, `regression_anwg` ("strongest
# deployable selector under arrival-norm WG" per docs/result_claims.md),
# via scripts/persist_corrected_selector_artifact.py, which retrains it on
# the exact Phase 2B.15 train split and verifies it reproduces the
# published Phase 2B.16 held-out number (0.9856 ANWG on 174 fresh windows)
# before persisting. Pass --selector-artifact results/
# corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib
# and request --policies ...,selector to use it here. See
# load_and_validate_selector_artifact() for the manifest check that rejects
# any artifact not declared as trained under arrival_normalized_wg.
#
# `generated_heuristic`/`best_generated` remain NOT wired: investigated (not
# assumed) before this task concluded they aren't safely wirable today:
#   - Most of the selector's 18 features (src/llmserveopt/selector/
#     features.py: queue_length, active_sequence_count, free_sequence_ratio,
#     prompt/output-length stats, slack stats, arrival-rate/burstiness,
#     recent SLO violation rate) ARE reconstructable from this harness's own
#     client-side bookkeeping (see extract_features() usage below). Only
#     kv_utilization has no honest client-side substitute without scraping
#     vLLM's /metrics endpoint (not implemented here) -- it is passed as an
#     explicit, documented 0.0 placeholder (kv_utilization_available=False).
#   - The LLM-generated heuristic DSL shortlist (results/phase2a4_2b4_
#     final_eval/frozen_shortlist/) was selected by validation WG on
#     2026-06-25, one day BEFORE the Phase 2B.14 objective correction
#     (2026-06-26), and was never re-evaluated under arrival_normalized_wg.
#     Re-ranking it would require re-running the Phase 2A/2B evaluation
#     pipeline against the corrected objective, which was out of scope for
#     this task (the ML selector path above was sufficient to produce one
#     honestly corrected-objective "our method" artifact).
NOT_WIRED_POLICIES = {
    "generated_heuristic": (
        "The LLM-generated heuristic DSL shortlist predates the Phase 2B.14 "
        "objective correction and was never re-evaluated under "
        "arrival_normalized_wg (see module-level comment above). Use "
        "--policies ...,selector with --selector-artifact instead -- that "
        "path IS corrected-objective and wired."
    ),
    "best_generated": "alias of generated_heuristic -- see that entry.",
}


# ---------------------------------------------------------------------------
# Corrected-objective selector artifact loading
# ---------------------------------------------------------------------------

class SelectorArtifactError(Exception):
    """Raised when a selector artifact fails validation. Never caught silently
    -- callers must surface this and abort before starting the benchmark."""


def load_and_validate_selector_artifact(
    artifact_path: Path,
) -> Tuple[PerPolicyRegressionAnwgSelector, Dict[str, Any]]:
    """Load a selector .joblib plus its sibling manifest.json, refusing
    anything not explicitly declared as trained under the Phase
    2B.14-corrected `arrival_normalized_wg` objective.

    This rejects every pre-correction artifact on disk (results/
    phase2a2_selector_dataset/, phase2a3_selector_eval/, phase2a4_2b4_
    final_eval/) automatically: none of them ship a manifest.json, because
    they predate this validation contract entirely.
    """
    if not artifact_path.exists():
        raise SelectorArtifactError(f"Selector artifact not found: {artifact_path}")
    manifest_path = artifact_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise SelectorArtifactError(
            f"No manifest.json next to {artifact_path}. Refusing to load an "
            "undocumented selector artifact -- every pre-correction *.joblib "
            "on disk lacks a manifest for exactly this reason (see "
            "NOT_WIRED_POLICIES comment above)."
        )
    manifest = json.loads(manifest_path.read_text())
    objective = manifest.get("objective_definition", {}).get("name")
    if objective != "arrival_normalized_wg":
        raise SelectorArtifactError(
            f"Selector manifest at {manifest_path} declares "
            f"objective_definition.name={objective!r}, expected "
            "'arrival_normalized_wg' (the Phase 2B.14-corrected objective). "
            "Refusing to load a stale/pre-correction selector artifact."
        )
    selector_class = manifest.get("selector_class")
    if selector_class != "PerPolicyRegressionAnwgSelector":
        raise SelectorArtifactError(
            f"Selector manifest declares selector_class={selector_class!r}; "
            "this harness only knows how to load PerPolicyRegressionAnwgSelector."
        )
    selector = PerPolicyRegressionAnwgSelector.load(str(artifact_path))
    return selector, manifest


def selector_choose_subpolicy(
    selector: PerPolicyRegressionAnwgSelector,
    *,
    waiting_requests: List[Request],
    now: float,
    active_sequence_count: int,
    concurrency: int,
    recent_violation_rate: float,
) -> str:
    """Compute the 18 online-observable features for the current decision
    point and ask the selector which candidate policy to use next.

    Uses extract_features() in CAUSAL mode -- the same feature-extraction
    code path used to build the training data this selector was fit on
    (results/phase2b13_selector_training_and_suspicion_audit/per_window.csv
    and results/phase2b16_fresh_corrected_objective_validation/
    fresh_per_window.csv), so this is not a re-derived/approximate feature
    adapter. kv_utilization_available=False is an honest placeholder (see
    NOT_WIRED_POLICIES comment): this harness has no /metrics scrape, so
    that one feature is always 0.0 here, exactly as extract_features()
    documents for callers that cannot observe it.
    """
    free_sequence_ratio = 1.0 - (active_sequence_count / concurrency if concurrency > 0 else 0.0)
    features = extract_features(
        window_requests=waiting_requests,
        window_start_time=now,
        mode=FeatureMode.CAUSAL,
        prefix_requests=None,
        recent_violation_rate=recent_violation_rate,
        recent_violation_available=True,
        active_sequence_count=active_sequence_count,
        kv_utilization=0.0,
        kv_utilization_available=False,
        free_sequence_ratio=free_sequence_ratio,
        free_sequence_ratio_available=True,
    )
    feat_row = {f"feat_{k}": v for k, v in features.items()}
    chosen = selector.predict_one(feat_row)
    if chosen not in SELECTOR_CANDIDATES:
        raise SelectorArtifactError(
            f"Selector predicted unknown policy {chosen!r}, not in SELECTOR_CANDIDATES."
        )
    return chosen


class SelectorActionSpaceError(SelectorArtifactError):
    """Raised by the preflight when a selector could emit a label this harness
    cannot dispatch. A subclass of SelectorArtifactError so existing callers
    that abort on SelectorArtifactError also abort here -- always before any
    live vLLM request is sent."""


def preflight_selector_action_space(
    selector: PerPolicyRegressionAnwgSelector,
    plan: List["PlanRow"],
    concurrency_list: List[int],
) -> Dict[str, Any]:
    """Verify, BEFORE any live request, that every policy label the selector
    could emit over this exact request plan is dispatchable by the harness.

    Two layers of checking:

    1. Static superset guarantee. The selector's entire output range is
       SELECTOR_CANDIDATES (enforced in selector_choose_subpolicy). If
       SELECTOR_CANDIDATES is a subset of SELECTOR_DISPATCHABLE and every
       SELECTOR_DISPATCHABLE label is make_policy-constructible, then NO label
       the selector can ever emit -- on this plan or any other -- is
       unsupported. This is the strongest guarantee and the primary abort gate.

    2. Dynamic enumeration over the plan. We additionally invoke the selector at
       the first decision point of every (regime, bucket, target, concurrency)
       cell (all cell requests waiting, no in-flight work) and record the
       realized label distribution. This exercises the real feature-extraction
       + predict() path against the actual plan and catches any predict()-time
       error before live requests, plus reports which sub-policies this workload
       actually elicits. Any realized label not in SELECTOR_DISPATCHABLE is a
       hard failure.

    Returns a JSON-serializable report. Raises SelectorActionSpaceError if any
    label (static or dynamic) is not dispatchable -- callers must let this
    propagate and abort with no HTTP sent.
    """
    static_unsupported = sorted(set(SELECTOR_CANDIDATES) - set(SELECTOR_DISPATCHABLE))
    unbuildable = sorted(name for name in SELECTOR_DISPATCHABLE if not _policy_constructible(name))

    # Group plan into cells, exactly as the benchmark loop does.
    cells: Dict[Tuple[str, str, int, int], List["PlanRow"]] = {}
    for row in plan:
        cells.setdefault(
            (row.regime, row.prompt_bucket, row.target_output_tokens, row.concurrency_level), []
        ).append(row)

    emitted_counts: Dict[str, int] = {}
    dynamic_errors: List[str] = []
    for (regime, bucket, target, concurrency), cell_plan in cells.items():
        requests = [
            Request(
                request_id=row.request_id, arrival_time=row.arrival_time,
                prompt_tokens=row.intended_prompt_tokens,
                predicted_output_tokens=row.target_output_tokens,
                actual_output_tokens=row.target_output_tokens,
                slo_deadline=row.arrival_time + row.slo_slack_seconds,
                priority=row.priority, class_id=row.class_id,
            )
            for row in cell_plan
        ]
        try:
            chosen = normalize_policy_name(selector_choose_subpolicy(
                selector, waiting_requests=requests, now=0.0,
                active_sequence_count=0, concurrency=concurrency,
                recent_violation_rate=0.0,
            ))
        except Exception as exc:  # noqa: BLE001
            dynamic_errors.append(
                f"cell (regime={regime}, bucket={bucket}, target={target}, "
                f"concurrency={concurrency}): selector predict failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        emitted_counts[chosen] = emitted_counts.get(chosen, 0) + 1

    dynamic_unsupported = sorted(
        label for label in emitted_counts if label not in SELECTOR_DISPATCHABLE
    )

    report = {
        "selector_output_range": list(SELECTOR_CANDIDATES),
        "harness_dispatchable": list(SELECTOR_DISPATCHABLE),
        "labels_emitted_over_plan": dict(sorted(emitted_counts.items())),
        "labels_supported": sorted(set(emitted_counts) & set(SELECTOR_DISPATCHABLE)),
        "labels_unsupported_static": static_unsupported,
        "labels_unsupported_dynamic": dynamic_unsupported,
        "unbuildable_dispatchable_labels": unbuildable,
        "n_cells_enumerated": len(cells),
        "dynamic_predict_errors": dynamic_errors,
        "ok": not (static_unsupported or unbuildable or dynamic_unsupported or dynamic_errors),
    }

    if not report["ok"]:
        raise SelectorActionSpaceError(
            "Selector action-space preflight failed -- aborting before any live "
            "request. "
            f"static_unsupported={static_unsupported} "
            f"dynamic_unsupported={dynamic_unsupported} "
            f"unbuildable={unbuildable} "
            f"predict_errors={dynamic_errors}. "
            "Every label the selector can emit must be dispatchable by this "
            "harness (see SELECTOR_DISPATCHABLE / Option-1 action-space fix)."
        )
    return report


# ---------------------------------------------------------------------------
# Plan construction (identical across every policy)
# ---------------------------------------------------------------------------

@dataclass
class PlanRow:
    request_id: int
    prompt_bucket: str
    target_output_tokens: int
    concurrency_level: int
    request_index: int
    intended_prompt_tokens: int
    priority: float
    slo_slack_seconds: float
    class_id: str
    regime: str = "steady_moderate"
    arrival_time: float = 0.0
    prompt_text: str = field(default="", repr=False)


def _assign_slo_class(rng, slo_classes: List[SLOClass]) -> SLOClass:
    weights = [c.weight for c in slo_classes]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for c, w in zip(slo_classes, weights):
        acc += w
        if r <= acc:
            return c
    return slo_classes[-1]


def build_request_plan(
    prompt_buckets: List[str], target_output_tokens_list: List[int],
    concurrency_list: List[int], requests_per_cell: int, seed: int,
    slo_classes: List[SLOClass] = DEFAULT_SLO_CLASSES,
    regimes: Optional[List[str]] = None,
) -> List[PlanRow]:
    """Build the fixed request plan every policy replays identically.

    regimes=None (legacy/default): a single implicit regime, pure-burst
    arrival (arrival_time=0.0 for every row, matching every request plan
    built before regime support existed -- preserves exact behavior for
    existing callers/tests). PlanRow.regime is still labeled
    "steady_moderate" for schema consistency, but arrival timing is NOT
    the honest spread pattern _regime_arrival_times() computes for an
    explicit steady_moderate regime below.

    regimes=[...] (explicit): iterates regimes as an outer product
    dimension, each with its own SLO-class config (REGIME_SLO_CLASSES) and
    real arrival-time pattern (_regime_arrival_times) -- see the
    REGIME_SLO_CLASSES comment above for what each regime means and why.
    """
    import random
    from itertools import product

    rng = random.Random(seed)
    plan: List[PlanRow] = []
    request_id = 0

    if regimes is None:
        legacy_mode = True
        regime_list = ["steady_moderate"]
    else:
        legacy_mode = False
        unknown = set(regimes) - set(REGIME_SLO_CLASSES)
        if unknown:
            raise ValueError(f"Unknown regimes: {sorted(unknown)}. Known: {sorted(REGIME_SLO_CLASSES)}")
        regime_list = list(regimes)

    for regime in regime_list:
        regime_slo_classes = slo_classes if legacy_mode else REGIME_SLO_CLASSES[regime]
        for bucket, target, concurrency in product(prompt_buckets, target_output_tokens_list, concurrency_list):
            arrival_times = (
                [0.0] * requests_per_cell if legacy_mode
                else _regime_arrival_times(regime, requests_per_cell, rng)
            )
            for i in range(requests_per_cell):
                prompt_text = cc.build_length_targeted_prompt(bucket, target, seed, i)
                cls = _assign_slo_class(rng, regime_slo_classes)
                plan.append(PlanRow(
                    request_id=request_id, prompt_bucket=bucket, target_output_tokens=target,
                    concurrency_level=concurrency, request_index=i,
                    intended_prompt_tokens=cc.approx_token_count(prompt_text),
                    priority=cls.priority, slo_slack_seconds=cls.slo_slack, class_id=cls.class_id,
                    regime=regime, arrival_time=arrival_times[i],
                    prompt_text=prompt_text,
                ))
                request_id += 1
    return plan


def run_warmup(out_dir: Path, *, model: str, base_url: Optional[str], mock: bool, timeout_s: float) -> None:
    """One short/target=64 and one medium/target=128 request at
    concurrency=1, run before any measured policy loop, to absorb vLLM's
    one-time JIT kernel compilation latency spike under --enforce-eager
    (see experiments/real_llm/vllm_healthcheck_*/healthcheck.md). Written
    to its own files, never mixed into requests.jsonl / policy metrics."""
    warmup_plan = [
        PlanRow(
            request_id=-1, prompt_bucket="short", target_output_tokens=64, concurrency_level=1,
            request_index=0, intended_prompt_tokens=cc.approx_token_count(cc.build_length_targeted_prompt("short", 64, 0, 0)),
            priority=1.0, slo_slack_seconds=10.0, class_id="warmup",
            prompt_text=cc.build_length_targeted_prompt("short", 64, 0, 0),
        ),
        PlanRow(
            request_id=-2, prompt_bucket="medium", target_output_tokens=128, concurrency_level=1,
            request_index=0, intended_prompt_tokens=cc.approx_token_count(cc.build_length_targeted_prompt("medium", 128, 0, 0)),
            priority=1.0, slo_slack_seconds=10.0, class_id="warmup",
            prompt_text=cc.build_length_targeted_prompt("medium", 128, 0, 0),
        ),
    ]
    rows = []
    for planned_row in warmup_plan:
        t0 = time.monotonic()
        try:
            out = _dispatch(planned_row, model=model, base_url=base_url, mock=mock, timeout_s=timeout_s)
            rows.append({
                "request_id": planned_row.request_id, "prompt_bucket": planned_row.prompt_bucket,
                "target_output_tokens": planned_row.target_output_tokens, "status": "success",
                "ttft_seconds": out.get("ttft_seconds"),
                "server_request_latency_seconds": out.get("server_request_latency_seconds"),
                "total_wall_time_seconds": round(time.monotonic() - t0, 4),
                "output_tokens": out.get("output_tokens"), "finish_reason": out.get("finish_reason"),
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "request_id": planned_row.request_id, "prompt_bucket": planned_row.prompt_bucket,
                "target_output_tokens": planned_row.target_output_tokens, "status": "error",
                "error_type": type(exc).__name__, "error_message": str(exc)[:500],
                "total_wall_time_seconds": round(time.monotonic() - t0, 4),
            })

    with open(out_dir / "warmup_requests.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    lines = [
        "# Warm-up phase — NOT counted in policy metrics",
        "",
        "One short/target=64 and one medium/target=128 request at concurrency=1,",
        "run before any measured policy loop to absorb vLLM's one-time JIT kernel",
        "compilation latency spike under `--enforce-eager` (observed in",
        "`experiments/real_llm/vllm_healthcheck_*/healthcheck.md`: first request",
        "needed ~180s, subsequent requests ~0.3s). These requests are excluded",
        "from `requests.jsonl` and every `aggregate_by_*.csv` / policy metric.",
        "",
        "| request_id | bucket | target | status | ttft_s | server_latency_s | wall_s |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['request_id']} | {row['prompt_bucket']} | {row['target_output_tokens']} | "
            f"{row['status']} | {row.get('ttft_seconds')} | {row.get('server_request_latency_seconds')} | "
            f"{row['total_wall_time_seconds']} |"
        )
    (out_dir / "warmup_summary.md").write_text("\n".join(lines) + "\n")


def write_request_plan(plan: List[PlanRow], out_dir: Path) -> None:
    with open(out_dir / "request_plan.jsonl", "w") as f:
        for row in plan:
            f.write(json.dumps(asdict(row)) + "\n")


# ---------------------------------------------------------------------------
# Per-cell, per-policy external-admission execution against vLLM
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResultRow:
    policy: str
    request_id: int
    prompt_bucket: str
    target_output_tokens: int
    concurrency_level: int
    class_id: str
    priority: float
    arrival_time_s: float
    admission_time_s: Optional[float]
    completion_time_s: Optional[float]
    queuing_delay_s: float
    ttft_seconds: Optional[float]
    server_request_latency_seconds: Optional[float]
    total_wall_time_seconds: Optional[float]
    slo_deadline_s: float
    slo_violated: Optional[bool]
    output_tokens: Optional[float]
    status: str
    error_type: Optional[str]
    error_message: Optional[str]
    selector_chosen_policy: Optional[str] = None
    regime: str = "steady_moderate"


def _dispatch(planned_row: PlanRow, *, model: str, base_url: Optional[str], mock: bool, timeout_s: float) -> Dict[str, Any]:
    fake_planned = vllm_mod.VllmPlannedRequest(
        request_id=str(planned_row.request_id), prompt_bucket=planned_row.prompt_bucket,
        target_output_tokens=planned_row.target_output_tokens,
        concurrency_level=planned_row.concurrency_level, request_index=planned_row.request_index,
        intended_prompt_tokens=planned_row.intended_prompt_tokens, prompt_text=planned_row.prompt_text,
    )
    if mock:
        return vllm_mod.mock_call(fake_planned)
    return vllm_mod.query_vllm_completion(
        base_url, model, planned_row.prompt_text,
        max_tokens=planned_row.target_output_tokens * 2, timeout_s=timeout_s,
    )


def run_cell_for_policy(
    policy_name: str, cell_plan: List[PlanRow], concurrency: int, *,
    model: str, base_url: Optional[str], mock: bool, timeout_s: float,
    selector_model: Optional[PerPolicyRegressionAnwgSelector] = None,
) -> List[ComparisonResultRow]:
    is_meta_selector = policy_name == "selector"
    if is_meta_selector and selector_model is None:
        raise SelectorArtifactError("policy_name='selector' but no selector_model was provided.")
    policy = None if policy_name in ("vllm_direct", "selector") else make_policy(policy_name)
    subpolicy_cache: Dict[str, Any] = {}

    requests = [
        Request(
            request_id=row.request_id, arrival_time=row.arrival_time, prompt_tokens=row.intended_prompt_tokens,
            predicted_output_tokens=row.target_output_tokens,
            actual_output_tokens=row.target_output_tokens,  # placeholder; real value recorded post-hoc below
            # Request.slo_deadline is an ABSOLUTE time (core/types.py), not a
            # duration -- must be arrival_time + slack, or a policy that
            # reasons about deadlines (e.g. admission_control's laxity
            # check) sees an already-passed deadline for any row with
            # arrival_time > 0 (steady_moderate regime).
            slo_deadline=row.arrival_time + row.slo_slack_seconds, priority=row.priority, class_id=row.class_id,
        )
        for row in cell_plan
    ]
    by_id = {row.request_id: row for row in cell_plan}
    all_observable = [ObservableRequest.from_request(r) for r in requests]
    # Requests with arrival_time > 0 (regime-driven staggering, see
    # _regime_arrival_times) are not yet observable -- they move from
    # `pending` into `waiting` only once real elapsed wall-clock time
    # reaches their scheduled arrival, checked below every loop iteration.
    waiting: List[ObservableRequest] = [r for r in all_observable if r.arrival_time <= 0.0]
    pending: List[ObservableRequest] = sorted(
        (r for r in all_observable if r.arrival_time > 0.0), key=lambda r: r.arrival_time,
    )
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=concurrency, max_batch_tokens=10**9, max_kv_tokens=10**9,
        active_request_ids=[], active_requests_info=[], current_kv_tokens=0, tokens_decoded_per_request={},
    )

    results: List[ComparisonResultRow] = []
    active: Dict[int, Tuple[concurrent.futures.Future, float]] = {}
    admission_choice: Dict[int, Optional[str]] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency))
    t0 = time.monotonic()
    completed_count = 0
    step = 0

    def _recent_violation_rate() -> float:
        completed = [r for r in results if r.status == "success"]
        if not completed:
            return 0.0
        return sum(1 for r in completed if r.slo_violated) / len(completed)

    try:
        while pending or waiting or active:
            now = time.monotonic() - t0
            if pending:
                newly_arrived = [r for r in pending if r.arrival_time <= now]
                if newly_arrived:
                    waiting.extend(newly_arrived)
                    pending = [r for r in pending if r.arrival_time > now]
            chosen_name: Optional[str] = None
            if policy is not None or is_meta_selector:
                # Policies mutate gpu.active_request_ids/current_kv_tokens as
                # internal bookkeeping ("avoid over-admission within the same
                # action" — see e.g. fifo.py). That must land on a disposable
                # snapshot, never on our own authoritative gpu_state, or a
                # completed request's slot leaks permanently (only visible
                # once concurrency < requests pending, e.g. concurrency=1
                # with 2+ requests in a cell).
                gpu_snapshot = ObservableGPUState(
                    gpu_id=gpu_state.gpu_id, max_active_sequences=gpu_state.max_active_sequences,
                    max_batch_tokens=gpu_state.max_batch_tokens, max_kv_tokens=gpu_state.max_kv_tokens,
                    active_request_ids=list(gpu_state.active_request_ids),
                    active_requests_info=list(gpu_state.active_requests_info),
                    current_kv_tokens=gpu_state.current_kv_tokens,
                    tokens_decoded_per_request=dict(gpu_state.tokens_decoded_per_request),
                    prefilling_count=gpu_state.prefilling_count, decoding_count=gpu_state.decoding_count,
                )
                state = ObservableState(
                    time=now, waiting_queue=list(waiting), gpu_states=[gpu_snapshot],
                    completed_count=completed_count, step=step,
                )
                if is_meta_selector:
                    waiting_ids = {w.request_id for w in waiting}
                    waiting_requests = [r for r in requests if r.request_id in waiting_ids]
                    chosen_name = normalize_policy_name(selector_choose_subpolicy(
                        selector_model, waiting_requests=waiting_requests, now=now,
                        active_sequence_count=len(gpu_state.active_request_ids),
                        concurrency=concurrency, recent_violation_rate=_recent_violation_rate(),
                    ))
                    if chosen_name not in subpolicy_cache:
                        subpolicy_cache[chosen_name] = make_policy(chosen_name)
                    action = subpolicy_cache[chosen_name].select_action(state)
                else:
                    action = policy.select_action(state)
                admitted_ids = action.all_admitted_ids()
            else:  # vllm_direct: strict arrival-order admission bounded by concurrency
                free = gpu_state.max_active_sequences - len(gpu_state.active_request_ids)
                admitted_ids = {r.request_id for r in waiting[:max(0, free)]}

            for rid in list(admitted_ids):
                if rid in active:
                    continue
                idx = next((i for i, r in enumerate(waiting) if r.request_id == rid), None)
                if idx is None:
                    continue
                waiting.pop(idx)
                admission_time = time.monotonic() - t0
                fut = executor.submit(
                    _dispatch, by_id[rid], model=model, base_url=base_url, mock=mock, timeout_s=timeout_s,
                )
                active[rid] = (fut, admission_time)
                admission_choice[rid] = chosen_name
                gpu_state.active_request_ids.append(rid)

            if not active:
                if pending:
                    # Nothing admittable yet -- sleep until the next
                    # scheduled regime arrival instead of busy-spinning.
                    next_arrival = min(r.arrival_time for r in pending)
                    sleep_s = max(0.0, next_arrival - (time.monotonic() - t0))
                    time.sleep(min(sleep_s, 0.05))
                    step += 1
                    continue
                if waiting:
                    # Nothing active, nothing pending, but the chosen policy
                    # still refuses to admit everyone left in `waiting` (e.g.
                    # scorpio_style_slo_guard's laxity guard judging a deadline
                    # already unmeetable, or admission_control with a finite
                    # laxity_threshold). Record these explicitly as dropped --
                    # never silently lose a planned request from requests.jsonl
                    # -- so n_total stays equal to len(cell_plan) for every
                    # policy and compute_policy_metrics() counts them as zero
                    # (arrival-normalized-WG convention: all arrivals, dropped =
                    # zero credit).
                    #
                    # Error-type taxonomy (Part E of the action-space fix):
                    #   - PolicyDeclinedAdmission: the chosen policy IS
                    #     constructible/supported and deliberately admitted
                    #     nothing (intentional load-shed / laxity guard). This
                    #     is expected policy behavior, still zero-credit, and is
                    #     NOT a harness/adapter failure.
                    #   - PolicyNeverAdmitted: reserved for the (preflight-
                    #     prevented) case where the chosen label has no
                    #     dispatchable adapter at all. If we ever reach here with
                    #     an unconstructible policy, that is a genuine harness bug
                    #     and is labeled distinctly so it can never be confused
                    #     with intentional shedding.
                    declined_policy = chosen_name or policy_name
                    adapter_missing = not _policy_constructible(declined_policy)
                    if adapter_missing:
                        drop_error_type = "PolicyNeverAdmitted"
                        drop_message = (
                            f"Policy '{declined_policy}' has no dispatchable adapter in this "
                            "harness (make_policy failed). This should have been caught by the "
                            "selector action-space preflight before any live request -- reaching "
                            "it at runtime indicates a harness bug, not intentional load-shedding."
                        )
                    else:
                        drop_error_type = "PolicyDeclinedAdmission"
                        drop_message = (
                            f"Policy '{declined_policy}' evaluated the waiting queue and "
                            "deliberately admitted nothing (intentional load-shed / laxity guard: "
                            "the remaining requests were judged unmeetable). Counted as zero-credit "
                            "toward arrival-normalized WG; NOT a network/dispatch/adapter failure."
                        )
                    drop_time = time.monotonic() - t0
                    for w in waiting:
                        row = by_id[w.request_id]
                        results.append(ComparisonResultRow(
                            policy=policy_name, request_id=w.request_id, prompt_bucket=row.prompt_bucket,
                            target_output_tokens=row.target_output_tokens, concurrency_level=concurrency,
                            class_id=row.class_id, priority=row.priority, arrival_time_s=row.arrival_time,
                            admission_time_s=None, completion_time_s=None,
                            queuing_delay_s=round(drop_time - row.arrival_time, 4),
                            ttft_seconds=None, server_request_latency_seconds=None,
                            total_wall_time_seconds=None,
                            slo_deadline_s=row.slo_slack_seconds, slo_violated=True,
                            output_tokens=None, status="dropped", error_type=drop_error_type,
                            error_message=drop_message,
                            selector_chosen_policy=chosen_name if is_meta_selector else None,
                            regime=row.regime,
                        ))
                    waiting = []
                break  # nothing admitted and nothing pending — avoid a busy loop

            futures = [f for f, _ in active.values()]
            done, _ = concurrent.futures.wait(futures, timeout=timeout_s + 5, return_when=concurrent.futures.FIRST_COMPLETED)

            for rid, (fut, admission_time) in list(active.items()):
                if fut not in done:
                    continue
                row = by_id[rid]
                completion_time = time.monotonic() - t0
                try:
                    out = fut.result()
                    output_tokens = out.get("output_tokens")
                    results.append(ComparisonResultRow(
                        policy=policy_name, request_id=rid, prompt_bucket=row.prompt_bucket,
                        target_output_tokens=row.target_output_tokens, concurrency_level=concurrency,
                        class_id=row.class_id, priority=row.priority, arrival_time_s=row.arrival_time,
                        admission_time_s=round(admission_time, 4), completion_time_s=round(completion_time, 4),
                        queuing_delay_s=round(admission_time - row.arrival_time, 4),
                        ttft_seconds=out.get("ttft_seconds"),
                        server_request_latency_seconds=out.get("server_request_latency_seconds"),
                        total_wall_time_seconds=round(completion_time - row.arrival_time, 4),
                        slo_deadline_s=row.slo_slack_seconds,
                        slo_violated=(completion_time - row.arrival_time) > row.slo_slack_seconds,
                        output_tokens=output_tokens, status="success",
                        error_type=None, error_message=None,
                        selector_chosen_policy=admission_choice.get(rid),
                        regime=row.regime,
                    ))
                except Exception as exc:  # noqa: BLE001
                    status = "timeout" if "timed out" in str(exc).lower() else "error"
                    results.append(ComparisonResultRow(
                        policy=policy_name, request_id=rid, prompt_bucket=row.prompt_bucket,
                        target_output_tokens=row.target_output_tokens, concurrency_level=concurrency,
                        class_id=row.class_id, priority=row.priority, arrival_time_s=row.arrival_time,
                        admission_time_s=round(admission_time, 4), completion_time_s=None,
                        queuing_delay_s=round(admission_time - row.arrival_time, 4),
                        ttft_seconds=None, server_request_latency_seconds=None,
                        total_wall_time_seconds=round(completion_time - row.arrival_time, 4),
                        slo_deadline_s=row.slo_slack_seconds, slo_violated=None,
                        output_tokens=None, status=status, error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                        selector_chosen_policy=admission_choice.get(rid),
                        regime=row.regime,
                    ))
                del active[rid]
                gpu_state.active_request_ids.remove(rid)
                completed_count += 1
            step += 1
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_policy_metrics(rows: List[Dict[str, Any]], policy_wall_clock_s: float) -> Dict[str, Any]:
    total = len(rows)
    completed = [r for r in rows if r["status"] == "success"]
    failed = [r for r in rows if r["status"] in ("error", "timeout", "dropped")]
    dropped = [r for r in rows if r["status"] == "dropped"]
    # Distinguish intentional load-shed (a supported policy chose to admit
    # nothing) from a genuine missing-adapter drop. Post-preflight the latter
    # must be zero; reporting both makes any future regression obvious.
    n_declined_admission = sum(1 for r in dropped if r.get("error_type") == "PolicyDeclinedAdmission")
    n_never_admitted = sum(1 for r in dropped if r.get("error_type") == "PolicyNeverAdmitted")

    def stats(values: List[float]) -> Dict[str, Optional[float]]:
        values = [v for v in values if v is not None]
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
        return {
            "count": len(values), "mean": sum(values) / len(values),
            "p50": cc._percentile(values, 0.50), "p95": cc._percentile(values, 0.95),
            "p99": cc._percentile(values, 0.99),
        }

    ttft_stats = stats([r["ttft_seconds"] for r in completed])
    latency_stats = stats([r["server_request_latency_seconds"] for r in completed])
    wall_stats = stats([r["total_wall_time_seconds"] for r in completed])
    output_tokens = [r["output_tokens"] for r in completed if r.get("output_tokens")]

    weights = [r["priority"] if r["priority"] > 0 else 1.0 for r in completed]
    met = [0.0 if r["slo_violated"] else 1.0 for r in completed]
    total_weight = sum(weights)
    conditional_wg = (sum(w * m for w, m in zip(weights, met)) / total_weight) if total_weight > 0 else 0.0
    completion_fraction = (len(completed) / total) if total > 0 else float("nan")
    # Arrival-normalized weighted goodput: same convention as
    # scripts/run_phase2b14_metric_audit_scorpio_ablation.py's
    # arrival_normalized_wg = completion_fraction * conditional_WG — the
    # corrected objective (denominator over ALL arrivals, not completed-only).
    arrival_normalized_wg = completion_fraction * conditional_wg if total > 0 else None

    slo_violated_flags = [r["slo_violated"] for r in completed if r["slo_violated"] is not None]
    slo_violation_rate = (sum(slo_violated_flags) / len(slo_violated_flags)) if slo_violated_flags else None

    request_throughput = (len(completed) / policy_wall_clock_s) if policy_wall_clock_s > 0 else None
    token_throughput = (sum(output_tokens) / policy_wall_clock_s) if (policy_wall_clock_s > 0 and output_tokens) else None

    return {
        "n_total": total,
        "n_completed": len(completed),
        "n_failed": len(failed),
        "n_dropped": len(dropped),
        "n_declined_admission": n_declined_admission,
        "n_never_admitted": n_never_admitted,
        "completion_fraction": completion_fraction,
        "ttft_stats": ttft_stats,
        "server_latency_stats": latency_stats,
        "total_wall_time_stats": wall_stats,
        "mean_output_tokens": (sum(output_tokens) / len(output_tokens)) if output_tokens else None,
        "conditional_weighted_goodput": conditional_wg,
        "arrival_normalized_weighted_goodput": arrival_normalized_wg,
        "slo_violation_rate_among_completed": slo_violation_rate,
        "request_throughput_per_sec": request_throughput,
        "output_token_throughput_per_sec": token_throughput,
        "policy_wall_clock_seconds": policy_wall_clock_s,
    }


KENDALL_TAU_BASELINES = (
    "fifo", "edf", "least_laxity_first", "estimated_service_time_first",
    "shortest_output_first", "vllm_direct",
)


def _kendall_tau(order_a: List[int], order_b: List[int], common_ids: set) -> Optional[float]:
    """Pairwise concordant/discordant Kendall tau over the request_ids common
    to both admission orders. Pure Python (no scipy dependency) -- fine at
    the small per-cell n (requests_per_cell) this harness uses."""
    rank_a = {rid: i for i, rid in enumerate(order_a)}
    rank_b = {rid: i for i, rid in enumerate(order_b)}
    ids = [rid for rid in common_ids if rid in rank_a and rid in rank_b]
    n = len(ids)
    if n < 2:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = rank_a[ids[i]] - rank_a[ids[j]]
            db = rank_b[ids[i]] - rank_b[ids[j]]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom > 0 else None


def compute_decision_divergence(
    all_rows: List[Dict[str, Any]], selector_policy: str = "selector",
    baselines: Tuple[str, ...] = KENDALL_TAU_BASELINES,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Post-hoc divergence analysis using REAL, independently-executed
    admission orders and REAL SLO outcomes for every policy over the
    IDENTICAL request plan -- no shadow/simulated decisions. For each
    (regime, prompt_bucket, target_output_tokens, concurrency_level) cell,
    compares the selector's actual admission order (by admission_time_s)
    against each baseline's actual admission order:
      - n_rank_position_mismatches: how many admission ranks (1st admitted,
        2nd admitted, ...) picked a different request_id.
      - kendall_tau: rank-order agreement over requests both runs admitted.
      - n_slo_outcome_changed: requests whose slo_violated flag differs
        between the selector run and that baseline run.
    Returns (divergence_rows, example_rows) -- example_rows are individual
    requests whose SLO outcome differed, for selector_vs_baselines_examples.md.
    """
    import pandas as pd

    df = pd.DataFrame(all_rows)
    if df.empty or selector_policy not in set(df["policy"]):
        return [], []

    cell_cols = ["regime", "prompt_bucket", "target_output_tokens", "concurrency_level"]
    divergence_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    for cell_keys, group in df.groupby(cell_cols):
        cell_keys = cell_keys if isinstance(cell_keys, tuple) else (cell_keys,)
        cell_dict = dict(zip(cell_cols, cell_keys))
        sel = group[group["policy"] == selector_policy].sort_values("admission_time_s")
        if sel.empty:
            continue
        sel_order = sel["request_id"].tolist()
        sel_slo = dict(zip(sel["request_id"], sel["slo_violated"]))
        for baseline in baselines:
            base = group[group["policy"] == baseline].sort_values("admission_time_s")
            if base.empty:
                continue
            base_order = base["request_id"].tolist()
            base_slo = dict(zip(base["request_id"], base["slo_violated"]))
            common_ids = set(sel_order) & set(base_order)
            n_rank_mismatch = sum(
                1 for i in range(min(len(sel_order), len(base_order)))
                if sel_order[i] != base_order[i]
            )
            tau = _kendall_tau(sel_order, base_order, common_ids)
            slo_changed_ids = sorted(
                rid for rid in common_ids if sel_slo.get(rid) != base_slo.get(rid)
            )
            divergence_rows.append({
                **cell_dict, "baseline": baseline,
                "n_selector_admitted": len(sel_order), "n_baseline_admitted": len(base_order),
                "n_common_requests": len(common_ids),
                "n_rank_position_mismatches": n_rank_mismatch,
                "kendall_tau": tau,
                "n_slo_outcome_changed": len(slo_changed_ids),
                "slo_outcome_changed_request_ids": ";".join(str(r) for r in slo_changed_ids),
            })
            for rid in slo_changed_ids:
                example_rows.append({
                    **cell_dict, "baseline": baseline, "request_id": rid,
                    "selector_slo_violated": bool(sel_slo.get(rid)),
                    "baseline_slo_violated": bool(base_slo.get(rid)),
                    "selector_admission_rank": sel_order.index(rid),
                    "baseline_admission_rank": base_order.index(rid),
                })
    return divergence_rows, example_rows


def write_decision_divergence_outputs(
    out_dir: Path, divergence_rows: List[Dict[str, Any]], example_rows: List[Dict[str, Any]],
) -> None:
    import pandas as pd

    pd.DataFrame(divergence_rows).to_csv(out_dir / "decision_divergence.csv", index=False)

    lines = [
        "# Selector vs. Fixed-Baseline Decision Divergence — Examples",
        "",
        "Every row below is a request where the selector's actual admission "
        "run and a baseline's actual admission run (both real, over the "
        "identical request plan) produced a DIFFERENT SLO outcome for the "
        "same request_id. This is measured from real execution, not shadow "
        "evaluation: each policy including the selector ran independently "
        "against the real vLLM server over the identical plan.",
        "",
    ]
    if not example_rows:
        lines.append(
            "No SLO-outcome divergence found: every baseline that diverged in "
            "admission order still produced identical SLO outcomes for every "
            "shared request in this run (see decision_divergence.csv for "
            "order-only divergence, e.g. kendall_tau < 1.0)."
        )
    else:
        lines += [
            "| Regime | Bucket | Target tok | Concurrency | Baseline | Request | "
            "Selector SLO violated | Baseline SLO violated | Selector rank | Baseline rank |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for ex in example_rows[:10]:
            lines.append(
                f"| {ex['regime']} | {ex['prompt_bucket']} | {ex['target_output_tokens']} | "
                f"{ex['concurrency_level']} | {ex['baseline']} | {ex['request_id']} | "
                f"{ex['selector_slo_violated']} | {ex['baseline_slo_violated']} | "
                f"{ex['selector_admission_rank']} | {ex['baseline_admission_rank']} |"
            )
        if len(example_rows) > 10:
            lines.append(f"\n... and {len(example_rows) - 10} more (see decision_divergence.csv).")
    (out_dir / "selector_vs_baselines_examples.md").write_text("\n".join(lines) + "\n")


def compute_bootstrap_ci(
    all_rows: List[Dict[str, Any]], policies: List[str], selector_policy: str = "selector",
    n_boot: int = 2000, seed: int = 20260703,
) -> List[Dict[str, Any]]:
    """Paired bootstrap over request_id: for each policy, resample
    request_ids with replacement (paired across policies since every
    policy ran the identical plan) and recompute arrival-normalized WG each
    replicate. Also reports the paired difference (selector - each
    baseline) CI, using the SAME resampled request_ids for both policies in
    each replicate (correct paired bootstrap, not two independent CIs
    subtracted)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    by_policy: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in all_rows:
        by_policy.setdefault(r["policy"], {})[r["request_id"]] = r

    def _wg_for_ids(policy_rows: Dict[int, Dict[str, Any]], ids: List[int]) -> float:
        num, den = 0.0, 0.0
        for rid in ids:
            row = policy_rows.get(rid)
            if row is None:
                continue
            w = row["priority"] if row["priority"] > 0 else 1.0
            den += w
            if row["status"] == "success" and not row["slo_violated"]:
                num += w
        return num / den if den > 0 else 0.0

    rows_out: List[Dict[str, Any]] = []
    all_request_ids = sorted({r["request_id"] for r in all_rows if r["policy"] == policies[0]}) if policies else []
    n = len(all_request_ids)
    if n == 0:
        return rows_out

    boot_wg: Dict[str, np.ndarray] = {}
    for policy in policies:
        policy_rows = by_policy.get(policy, {})
        vals = np.empty(n_boot)
        for b in range(n_boot):
            sample_ids = rng.choice(all_request_ids, size=n, replace=True)
            vals[b] = _wg_for_ids(policy_rows, sample_ids.tolist())
        boot_wg[policy] = vals
        point = _wg_for_ids(policy_rows, all_request_ids)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows_out.append({
            "policy": policy, "point_estimate_wg": point,
            "ci_low_2.5pct": float(lo), "ci_high_97.5pct": float(hi), "n_boot": n_boot,
        })

    if selector_policy in boot_wg:
        for policy in policies:
            if policy == selector_policy:
                continue
            diff = boot_wg[selector_policy] - boot_wg[policy]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows_out.append({
                "policy": f"{selector_policy}_minus_{policy}", "point_estimate_wg": float(np.mean(diff)),
                "ci_low_2.5pct": float(lo), "ci_high_97.5pct": float(hi), "n_boot": n_boot,
            })
    return rows_out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(
    out_dir: Path, all_rows: List[Dict[str, Any]], per_policy_wall_clock: Dict[str, float], cfg: Dict[str, Any],
    *, decision_divergence_report: bool = False, bootstrap_ci: bool = False,
) -> Dict[str, Any]:
    import pandas as pd

    with open(out_dir / "requests.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    errors = [r for r in all_rows if r["status"] != "success"]
    with open(out_dir / "errors.jsonl", "w") as f:
        for row in errors:
            f.write(json.dumps(row) + "\n")

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    policies = sorted({r["policy"] for r in all_rows})
    regimes = sorted({r["regime"] for r in all_rows}) if all_rows else []

    per_policy_metrics = {
        policy: compute_policy_metrics([r for r in all_rows if r["policy"] == policy], per_policy_wall_clock.get(policy, 0.0))
        for policy in policies
    }
    per_policy_and_regime_metrics = {
        f"{policy}::{regime}": compute_policy_metrics(
            [r for r in all_rows if r["policy"] == policy and r["regime"] == regime],
            per_policy_wall_clock.get(policy, 0.0),
        )
        for policy in policies for regime in regimes
    }

    def _group_rows(group_cols: List[str]) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        records = []
        for keys, sub in df.groupby(group_cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            sub_success = sub[sub["status"] == "success"]
            rec = dict(zip(group_cols, keys))
            rec["n_total"] = len(sub)
            rec["n_success"] = len(sub_success)
            rec["n_failed"] = int((sub["status"].isin(["error", "timeout"])).sum())
            lat = sub_success["server_request_latency_seconds"].dropna().tolist()
            ttft = sub_success["ttft_seconds"].dropna().tolist()
            rec["mean_latency_s"] = (sum(lat) / len(lat)) if lat else None
            rec["mean_ttft_s"] = (sum(ttft) / len(ttft)) if ttft else None
            records.append(rec)
        return records

    by_policy = [{"policy": p, **per_policy_metrics[p]} for p in policies]
    pd.DataFrame(by_policy).to_csv(out_dir / "aggregate_by_policy.csv", index=False)

    by_policy_regime = [
        {"policy": p, "regime": r, **per_policy_and_regime_metrics[f"{p}::{r}"]}
        for p in policies for r in regimes
    ]
    pd.DataFrame(by_policy_regime).to_csv(out_dir / "aggregate_by_policy_and_regime.csv", index=False)

    pd.DataFrame(_group_rows(["policy", "concurrency_level"])).to_csv(out_dir / "aggregate_by_concurrency.csv", index=False)
    pd.DataFrame(_group_rows(["policy", "target_output_tokens"])).to_csv(out_dir / "aggregate_by_target_output_tokens.csv", index=False)
    pd.DataFrame(_group_rows(["policy", "prompt_bucket"])).to_csv(out_dir / "aggregate_by_prompt_bucket.csv", index=False)

    overall = {
        "total_records": len(all_rows),
        "policies": policies,
        "regimes": regimes,
        "per_policy": per_policy_metrics,
        "per_policy_and_regime": per_policy_and_regime_metrics,
    }

    if decision_divergence_report and "selector" in policies:
        divergence_rows, example_rows = compute_decision_divergence(all_rows)
        write_decision_divergence_outputs(out_dir, divergence_rows, example_rows)
        overall["decision_divergence_n_cells_compared"] = len({
            (r["regime"], r["prompt_bucket"], r["target_output_tokens"], r["concurrency_level"], r["baseline"])
            for r in divergence_rows
        })
        overall["decision_divergence_n_slo_outcome_changes"] = len(example_rows)
    else:
        pd.DataFrame([]).to_csv(out_dir / "decision_divergence.csv", index=False)
        (out_dir / "selector_vs_baselines_examples.md").write_text(
            "# Selector vs. Fixed-Baseline Decision Divergence — Examples\n\n"
            "Not computed for this run (--decision-divergence-report not passed, "
            "or 'selector' was not among the compared policies).\n"
        )

    if bootstrap_ci:
        ci_rows = compute_bootstrap_ci(all_rows, policies)
        pd.DataFrame(ci_rows).to_csv(out_dir / "bootstrap_confidence_intervals.csv", index=False)
        overall["bootstrap_ci"] = ci_rows

    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2))
    write_summary_md(out_dir, overall, cfg)
    return overall


def write_summary_md(out_dir: Path, overall: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    lines = [
        "# vLLM External-Admission Baseline Comparison — Summary",
        "",
        f"**Model:** `{cfg.get('model')}`",
        f"**Run status:** {cfg.get('run_status')}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    if cfg.get("run_status") != "completed":
        lines += [
            "**No real vLLM server was used for this run.** "
            f"See `run_config.json` (`run_status: {cfg.get('run_status')}`).",
            "",
        ]
    lines += [
        "## Policies compared",
        "",
        "| Policy | n_total | n_completed | n_failed | Declined (load-shed) | Never-admitted (adapter bug) | Arrival-norm. WG | SLO violation (completed) | Mean TTFT (s) | Mean server latency (s) | Req/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for policy, m in overall["per_policy"].items():
        anwg = m["arrival_normalized_weighted_goodput"]
        slo = m["slo_violation_rate_among_completed"]
        lines.append(
            f"| {policy} | {m['n_total']} | {m['n_completed']} | {m['n_failed']} | "
            f"{m.get('n_declined_admission', 0)} | {m.get('n_never_admitted', 0)} | "
            f"{anwg:.4f} | {(slo if slo is not None else float('nan')):.4f} | "
            f"{(m['ttft_stats']['mean'] or float('nan')):.4f} | "
            f"{(m['server_latency_stats']['mean'] or float('nan')):.4f} | "
            f"{(m['request_throughput_per_sec'] or float('nan')):.3f} |"
        )
    regimes = overall.get("regimes") or []
    if len(regimes) > 1 and overall.get("per_policy_and_regime"):
        lines += ["", "## Weighted goodput by policy and regime", "",
                   "| Policy | Regime | n_total | n_completed | Arrival-norm. WG | SLO violation |",
                   "|---|---|---|---|---|---|"]
        for key, m in overall["per_policy_and_regime"].items():
            policy, regime = key.split("::", 1)
            anwg = m["arrival_normalized_weighted_goodput"]
            slo = m["slo_violation_rate_among_completed"]
            lines.append(
                f"| {policy} | {regime} | {m['n_total']} | {m['n_completed']} | "
                f"{(anwg if anwg is not None else float('nan')):.4f} | "
                f"{(slo if slo is not None else float('nan')):.4f} |"
            )

    if overall.get("bootstrap_ci"):
        lines += ["", "## Bootstrap 95% confidence intervals (arrival-norm. WG)", "",
                   "| Policy | Point estimate | CI low | CI high |", "|---|---|---|---|"]
        for row in overall["bootstrap_ci"]:
            lines.append(
                f"| {row['policy']} | {row['point_estimate_wg']:.4f} | "
                f"{row['ci_low_2.5pct']:.4f} | {row['ci_high_97.5pct']:.4f} |"
            )

    if "decision_divergence_n_cells_compared" in overall:
        lines += [
            "", "## Decision divergence",
            "",
            f"- Cells compared (selector vs. each baseline): {overall['decision_divergence_n_cells_compared']}",
            f"- Requests where selector's real SLO outcome differed from a baseline's real SLO outcome: "
            f"{overall['decision_divergence_n_slo_outcome_changes']}",
            "- See `decision_divergence.csv` (per-cell Kendall tau / rank mismatches) and "
            "`selector_vs_baselines_examples.md` (worked examples).",
        ]

    lines += [
        "",
        "Policies not compared: " + ", ".join(NOT_WIRED_POLICIES.keys()) + " — see "
        "`docs/vllm_real_serving_external_baseline_pilot.md` for why.",
        "",
        "See `aggregate_by_policy.csv`, `aggregate_by_policy_and_regime.csv`, "
        "`aggregate_by_concurrency.csv`, `aggregate_by_target_output_tokens.csv`, "
        "`aggregate_by_prompt_bucket.csv` for breakdowns, and `errors.jsonl` for failure detail.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_manifest_and_repro(out_dir: Path, plan: List[PlanRow], cfg: Dict[str, Any]) -> None:
    git_info = vllm_mod._git_info(ROOT)
    (out_dir / "manifest.json").write_text(json.dumps({
        **git_info,
        "planned_requests_per_policy": len(plan),
        "policies": cfg["policies"],
        "regimes": sorted({r.regime for r in plan}),
        "cells": sorted({(r.regime, r.prompt_bucket, r.target_output_tokens, r.concurrency_level) for r in plan}),
    }, indent=2))
    vllm_mod.write_reproducibility_md(out_dir, cfg, git_info)


def capture_gpu_mem(path: Path) -> None:
    """Write `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu
    --format=csv,noheader` output to path. Never raises -- GPU memory
    capture is best-effort diagnostics, not correctness-critical."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        path.write_text(out.stdout.strip() + "\n" if out.returncode == 0 else f"nvidia-smi error: {out.stderr}\n")
    except Exception as exc:  # noqa: BLE001
        path.write_text(f"nvidia-smi unavailable: {exc}\n")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def csv_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-live-server", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--server-url", "--base-url", dest="server_url", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--policies", type=csv_str_list, default=list(WIRED_POLICIES))
    parser.add_argument("--prompt-buckets", type=csv_str_list, default=["short", "medium"])
    parser.add_argument("--target-output-tokens-list", type=cc.csv_int_list, default=[64, 128])
    parser.add_argument("--concurrency-list", type=cc.csv_int_list, default=[1, 2, 4])
    parser.add_argument("--requests-per-cell", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-total-requests", type=int, default=1000)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--stream", action="store_true",
        help="No-op: live requests always use vLLM's streaming SSE endpoint "
        "(needed to measure TTFT). Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--warmup", action="store_true",
        help="Run one short/target=64 and one medium/target=128 request at "
        "concurrency=1 before measurement, to absorb any one-time JIT "
        "kernel compilation latency spike. Not counted in policy metrics; "
        "written to warmup_requests.jsonl/warmup_summary.md instead.",
    )
    parser.add_argument(
        "--selector-artifact", default=None,
        help="Path to a corrected-objective selector .joblib (must have a "
        "sibling manifest.json declaring objective_definition.name == "
        "'arrival_normalized_wg'). Required if --policies includes "
        "'selector' or --require-our-method is passed.",
    )
    parser.add_argument(
        "--heuristic-artifact", default=None,
        help="Reserved for a future corrected-objective generated-heuristic "
        "artifact. No such artifact currently exists (see NOT_WIRED_POLICIES "
        "for generated_heuristic/best_generated) -- passing this today is a no-op.",
    )
    parser.add_argument(
        "--require-our-method", action="store_true",
        help="Fail before starting the benchmark unless a valid "
        "corrected-objective selector artifact loads successfully. Requires "
        "--policies to include 'selector' and --selector-artifact to be set.",
    )
    parser.add_argument(
        "--arrival-regimes", type=csv_str_list, default=None,
        help="One or more of: " + ", ".join(sorted(REGIME_SLO_CLASSES)) + ". "
        "Each regime adds an outer product dimension (its own SLO-class "
        "config and arrival-timing pattern -- see REGIME_SLO_CLASSES). "
        "Default (omitted): single implicit steady_moderate regime with "
        "legacy pure-burst arrival timing, matching pre-regime-support runs.",
    )
    parser.add_argument(
        "--decision-divergence-report", action="store_true",
        help="Write decision_divergence.csv and selector_vs_baselines_examples.md "
        "comparing the selector's real admission order/SLO outcomes against "
        "every fixed baseline's real admission order/SLO outcomes over the "
        "identical plan. No-op (empty report) unless 'selector' is among --policies.",
    )
    parser.add_argument(
        "--bootstrap-ci", action="store_true",
        help="Write bootstrap_confidence_intervals.csv: paired bootstrap 95%% "
        "CIs on arrival-normalized WG per policy, plus (selector - baseline) "
        "paired-difference CIs.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def repo_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else root / p


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.policies = [normalize_policy_name(p) for p in args.policies]

    unknown = (
        set(args.policies) - set(WIRED_POLICIES) - set(NOT_WIRED_POLICIES.keys()) - set(CONDITIONAL_POLICIES)
    )
    if unknown:
        print(f"ERROR: unknown policies: {sorted(unknown)}. Known: {WIRED_POLICIES + CONDITIONAL_POLICIES}", file=sys.stderr)
        return 2
    requested_not_wired = set(args.policies) & set(NOT_WIRED_POLICIES.keys())
    if requested_not_wired:
        print("ERROR: the following requested policies are not wired to a live server:", file=sys.stderr)
        for name in sorted(requested_not_wired):
            print(f"  {name}: {NOT_WIRED_POLICIES[name]}", file=sys.stderr)
        print(f"Use --policies with a subset of: {WIRED_POLICIES}", file=sys.stderr)
        return 8

    if not args.dry_run and not args.allow_live_server and not args.mock:
        print("ERROR: specify --dry-run, --mock, or --allow-live-server.", file=sys.stderr)
        return 2
    if args.allow_live_server and not args.mock and not args.server_url:
        print("ERROR: --allow-live-server requires --server-url.", file=sys.stderr)
        return 2

    if args.require_our_method and "selector" not in args.policies:
        print(
            "ERROR: --require-our-method requires --policies to include 'selector'.",
            file=sys.stderr,
        )
        return 9

    selector_model: Optional[PerPolicyRegressionAnwgSelector] = None
    selector_manifest: Optional[Dict[str, Any]] = None
    needs_selector = "selector" in args.policies or args.require_our_method
    if needs_selector:
        if not args.selector_artifact:
            print(
                "ERROR: --policies includes 'selector' (or --require-our-method was "
                "passed) but --selector-artifact was not given.",
                file=sys.stderr,
            )
            return 9
        try:
            selector_model, selector_manifest = load_and_validate_selector_artifact(
                repo_path(ROOT, args.selector_artifact)
            )
        except SelectorArtifactError as exc:
            print(f"ERROR: selector artifact validation failed: {exc}", file=sys.stderr)
            return 9
        print(
            f"  Loaded selector artifact: {args.selector_artifact} "
            f"(objective={selector_manifest['objective_definition']['name']}, "
            f"class={selector_manifest['selector_class']})"
        )

    out_dir = repo_path(ROOT, args.output_dir)
    requests_path = out_dir / "requests.jsonl"
    if out_dir.exists() and requests_path.exists() and requests_path.stat().st_size > 0:
        print(f"ERROR: output dir {out_dir} already has a non-empty requests.jsonl. Choose a new --output-dir.", file=sys.stderr)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.arrival_regimes:
        unknown_regimes = set(args.arrival_regimes) - set(REGIME_SLO_CLASSES)
        if unknown_regimes:
            print(
                f"ERROR: unknown --arrival-regimes: {sorted(unknown_regimes)}. "
                f"Known: {sorted(REGIME_SLO_CLASSES)}", file=sys.stderr,
            )
            return 2

    plan = build_request_plan(
        args.prompt_buckets, args.target_output_tokens_list, args.concurrency_list,
        args.requests_per_cell, args.seed, regimes=args.arrival_regimes,
    )
    total_planned = len(plan) * len(args.policies)
    if total_planned > args.max_total_requests:
        print(
            f"HARD CAP VIOLATION: planned {total_planned} requests ({len(plan)} x "
            f"{len(args.policies)} policies) exceeds --max-total-requests={args.max_total_requests}",
            file=sys.stderr,
        )
        return 4

    # Selector action-space preflight (Part E): before ANY live request, verify
    # every label the selector could emit over this exact plan is dispatchable.
    # If not, abort now -- no HTTP is sent, no server touched. --require-our-method
    # thereby fails fast on an unsupported selector output label.
    selector_preflight: Optional[Dict[str, Any]] = None
    if selector_model is not None:
        try:
            selector_preflight = preflight_selector_action_space(
                selector_model, plan, args.concurrency_list
            )
        except SelectorActionSpaceError as exc:
            (out_dir / "selector_action_space_preflight.json").write_text(
                json.dumps({"ok": False, "error": str(exc)}, indent=2)
            )
            print(f"ERROR: {exc}", file=sys.stderr)
            return 9
        (out_dir / "selector_action_space_preflight.json").write_text(
            json.dumps(selector_preflight, indent=2)
        )
        print(
            "  Selector action-space preflight OK: "
            f"{selector_preflight['n_cells_enumerated']} cells enumerated, "
            f"labels emitted={selector_preflight['labels_emitted_over_plan']}"
        )

    run_status = "planned_only"
    if args.allow_live_server and not args.mock:
        run_status = "completed"
    elif args.mock:
        run_status = "completed_mock"
    elif args.dry_run:
        run_status = "planned_only" if vllm_mod.vllm_cli_available() else "planned_only_vllm_not_installed"

    cfg = {
        "model": args.model, "policies": args.policies, "prompt_buckets": args.prompt_buckets,
        "target_output_tokens_list": args.target_output_tokens_list, "concurrency_list": args.concurrency_list,
        "requests_per_cell": args.requests_per_cell, "timeout_seconds": args.timeout_seconds,
        "max_total_requests": args.max_total_requests, "fail_fast": args.fail_fast, "seed": args.seed,
        "mock": args.mock, "server_url": args.server_url, "run_status": run_status,
        "not_wired_policies": NOT_WIRED_POLICIES,
        "selector_artifact": args.selector_artifact,
        "selector_manifest": selector_manifest,
        "require_our_method": args.require_our_method,
        "arrival_regimes": args.arrival_regimes or ["steady_moderate (legacy burst)"],
        "decision_divergence_report": args.decision_divergence_report,
        "bootstrap_ci": args.bootstrap_ci,
        "selector_action_space_preflight": selector_preflight,
    }
    write_request_plan(plan, out_dir)
    write_manifest_and_repro(out_dir, plan, cfg)
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    print("vLLM external-admission baseline comparison")
    print(f"  output_dir:        {out_dir}")
    print(f"  planned per policy: {len(plan)}")
    print(f"  policies:          {args.policies}")
    print(f"  run_status:        {run_status}")

    if args.dry_run and not args.allow_live_server and not args.mock:
        write_summary_md(out_dir, {"per_policy": {}}, cfg)
        (out_dir / "summary.json").write_text(json.dumps({"per_policy": {}}, indent=2))
        print("  No vLLM server was launched or queried (dry-run).")
        return 0

    if args.server_url:
        try:
            req = urllib.request.Request(f"{args.server_url.rstrip('/')}/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                (out_dir / "server_status.json").write_text(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            (out_dir / "server_status.json").write_text(json.dumps({"error": str(exc)}))

    if not args.mock:
        capture_gpu_mem(out_dir / "gpu_mem_before.txt")

    if args.warmup:
        print("  Running warm-up phase (not counted in policy metrics)...")
        run_warmup(out_dir, model=args.model, base_url=args.server_url, mock=args.mock, timeout_s=args.timeout_seconds)

    cells: Dict[Tuple[str, str, int, int], List[PlanRow]] = {}
    for row in plan:
        cells.setdefault((row.regime, row.prompt_bucket, row.target_output_tokens, row.concurrency_level), []).append(row)

    all_rows: List[Dict[str, Any]] = []
    per_policy_wall_clock: Dict[str, float] = {}
    fail_fast_triggered = False

    for policy_name in args.policies:
        policy_t0 = time.monotonic()
        for (regime, bucket, target, concurrency), cell_plan in cells.items():
            if fail_fast_triggered:
                break
            cell_results = run_cell_for_policy(
                policy_name, cell_plan, concurrency, model=args.model,
                base_url=args.server_url, mock=args.mock, timeout_s=args.timeout_seconds,
                selector_model=selector_model,
            )
            for r in cell_results:
                all_rows.append(asdict(r))
            if args.fail_fast:
                n = len(cell_results)
                n_failed = sum(1 for r in cell_results if r.status != "success")
                if n >= 10 and n_failed / n > 0.5:
                    fail_fast_triggered = True
                    print(f"FAIL-FAST: aborting after high failure rate in policy={policy_name} regime={regime}", file=sys.stderr)
                    break
        per_policy_wall_clock[policy_name] = time.monotonic() - policy_t0

    if not args.mock:
        capture_gpu_mem(out_dir / "gpu_mem_after.txt")

    overall = write_outputs(
        out_dir, all_rows, per_policy_wall_clock, cfg,
        decision_divergence_report=args.decision_divergence_report,
        bootstrap_ci=args.bootstrap_ci,
    )

    for policy_name, m in overall["per_policy"].items():
        print(f"  [{policy_name}] completed={m['n_completed']}/{m['n_total']} arrival_norm_wg={m['arrival_normalized_weighted_goodput']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
