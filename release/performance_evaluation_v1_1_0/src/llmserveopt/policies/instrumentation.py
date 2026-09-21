"""Optional decision-trace instrumentation for scheduling policies.

Records what a policy decided at each scheduling step -- candidates,
scores/ranks if the policy exposes them, admitted requests, and a compact
causal state summary -- without changing what the policy decides.

Disabled by default. When disabled, `InstrumentedPolicy.select_action` does
exactly what the wrapped policy's `select_action` does plus one boolean
check; no trace object is built and nothing is appended to any buffer or
file. This is the "negligible overhead when disabled" requirement: the only
added cost is `if not self._sink.enabled: return action`.

Schema
------
`DecisionTraceRecordV1.schema_version == "DecisionTraceV1"`. Any breaking
field change must introduce a new schema_version rather than silently
reinterpreting existing fields, so older recorded traces stay parseable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..core.action import Action
from ..core.types import ObservableState
from .base import BasePolicy
from .composition import CompositionDecisionLog, causal_context_features

DECISION_TRACE_SCHEMA_VERSION = "DecisionTraceV1"


@dataclass(frozen=True)
class DecisionTraceRecordV1:
    schema_version: str
    scenario_id: Optional[str]
    step: int
    sim_time: float
    policy_name: str
    candidate_request_ids: List[int]
    selected_by_gpu: Dict[int, List[int]]
    selected_request_ids: List[int]
    state_summary: Dict[str, float]
    raw_scores: Optional[Dict[int, float]] = None
    normalized_scores: Optional[Dict[int, float]] = None
    ranks: Optional[Dict[int, float]] = None
    expert_weights: Optional[Dict[str, float]] = None
    fallback_used: bool = False
    tie_breaker: str = "arrival_order_then_request_id"
    invalid_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionTraceSink:
    """In-memory buffer of decision traces with optional JSONL flush.

    `enabled=False` (the default) makes `record()` a no-op single branch;
    no record objects are constructed and the buffer never grows.
    """

    def __init__(self, *, enabled: bool = False, scenario_id: Optional[str] = None) -> None:
        self.enabled = enabled
        self.scenario_id = scenario_id
        self.records: List[DecisionTraceRecordV1] = []

    def record(self, rec: DecisionTraceRecordV1) -> None:
        if not self.enabled:
            return
        self.records.append(rec)

    def clear(self) -> None:
        self.records.clear()

    def write_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for rec in self.records:
                f.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")

    @staticmethod
    def read_jsonl(path: str | Path) -> List[DecisionTraceRecordV1]:
        out: List[DecisionTraceRecordV1] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("schema_version") != DECISION_TRACE_SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported decision trace schema_version {payload.get('schema_version')!r}"
                    )
                # JSON round-trips dict keys as strings; restore int request_ids.
                for int_keyed_field in ("raw_scores", "normalized_scores", "ranks"):
                    value = payload.get(int_keyed_field)
                    if value is not None:
                        payload[int_keyed_field] = {int(k): v for k, v in value.items()}
                payload["selected_by_gpu"] = {int(k): v for k, v in payload["selected_by_gpu"].items()}
                out.append(DecisionTraceRecordV1(**payload))
        return out


def _state_summary(state: ObservableState) -> Dict[str, float]:
    summary = causal_context_features(state)
    summary["queue_length_raw"] = float(len(state.waiting_queue))
    summary["step"] = float(state.step)
    return summary


class InstrumentedPolicy(BasePolicy):
    """Wraps any BasePolicy and optionally records its decisions.

    The wrapped policy's `select_action` return value is forwarded
    unmodified; instrumentation only reads state already produced by that
    call (the wrapped policy's own `decision_logs`, when present) and never
    feeds anything back into the decision.
    """

    def __init__(self, wrapped: BasePolicy, sink: DecisionTraceSink) -> None:
        self.wrapped = wrapped
        self.sink = sink
        self.name = f"instrumented:{wrapped.name}"

    def reset(self) -> None:
        self.wrapped.reset()

    def select_action(self, state: ObservableState) -> Action:
        action = self.wrapped.select_action(state)
        if not self.sink.enabled:
            return action

        candidate_ids = sorted(r.request_id for r in state.waiting_queue)
        selected_ids = sorted(action.all_admitted_ids())
        log = self._last_decision_log()
        rec = DecisionTraceRecordV1(
            schema_version=DECISION_TRACE_SCHEMA_VERSION,
            scenario_id=self.sink.scenario_id,
            step=state.step,
            sim_time=state.time,
            policy_name=self.wrapped.name,
            candidate_request_ids=candidate_ids,
            selected_by_gpu={gid: sorted(ids) for gid, ids in action.admit.items()},
            selected_request_ids=selected_ids,
            state_summary=_state_summary(state),
            normalized_scores=(
                {rid: sum(contrib.values()) for rid, contrib in log.expert_contributions.items()}
                if log is not None and log.expert_contributions
                else None
            ),
            expert_weights=dict(log.expert_weights) if log is not None else None,
            fallback_used=log.fallback_used if log is not None else False,
            invalid_reason=log.invalid_reason if log is not None else None,
        )
        self.sink.record(rec)
        return action

    def _last_decision_log(self) -> Optional[CompositionDecisionLog]:
        logs = getattr(self.wrapped, "decision_logs", None)
        if not logs:
            return None
        return logs[-1]
