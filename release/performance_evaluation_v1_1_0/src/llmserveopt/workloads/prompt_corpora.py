"""
Adapters for prompt / conversation / benchmark corpora.

These are **not** production serving traces. They supply prompt/response length
distributions (and optionally conversation metadata) that must be paired with
an explicit arrival process (synthetic or borrowed from a real trace).

Supported corpora
-----------------
* LMSYS-Chat-1M (gated) — metadata/length adapter only
* WildChat-1M (public) — metadata/length adapter only
* LongBench — long-context benchmark prompt-length adapter

Safety
------
Adapters must not persist raw conversation text into Git. Prefer whitespace or
explicit tokenizer length extraction into numeric features only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .augmentation import AugmentationConfig, augment_trace
from .canonical_schema import (
    CanonicalIngestRecord,
    DatasetType,
    FieldProvenance,
    ReplayLabel,
    default_provenance,
    records_to_requests_and_metadata,
    validate_canonical_records,
)
from ..core.types import Request


@dataclass
class PromptCorpusLengthRecord:
    """Numeric-only view of one conversation/turn or benchmark item."""

    source_record_id: str
    prompt_tokens: int
    actual_output_tokens: int
    n_turns: int = 1
    model_id: Optional[str] = None
    language: Optional[str] = None
    session_id: Optional[str] = None
    task: Optional[str] = None


@dataclass
class PromptCorpusConversionConfig:
    arrival_mode: str = "poisson"
    arrival_rate: float = 10.0
    max_requests: Optional[int] = None
    source_dataset: str = ""
    source_split: str = ""
    min_prompt_tokens: int = 1
    min_output_tokens: int = 1
    max_prompt_tokens: int = 131072
    max_output_tokens: int = 32768


@dataclass
class PromptCorpusConversionReport:
    rows_read: int
    rows_retained: int
    rows_dropped: int
    time_range_seconds: float
    mean_arrival_rate: float
    prompt_tokens_mean: float
    output_tokens_mean: float
    dataset_type: str
    replay_label: str
    source_dataset: str


def _whitespace_token_count(text: str) -> int:
    return max(1, len(text.split()))


def extract_chat_length_record(
    row: Dict[str, Any],
    source_record_id: str,
    *,
    conversation_key: str = "conversation",
) -> PromptCorpusLengthRecord:
    """Extract numeric length features from a chat-style HF row.

    Does not return or store conversation text.
    """
    conv = row.get(conversation_key) or []
    if not isinstance(conv, list) or not conv:
        raise ValueError("conversation missing or empty")

    prompt_text = None
    response_text = None
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("from") or "").lower()
        content = turn.get("content") or turn.get("value") or ""
        if role in {"user", "human"} and prompt_text is None:
            prompt_text = str(content)
        elif role in {"assistant", "gpt"} and prompt_text is not None and response_text is None:
            response_text = str(content)
            break

    if prompt_text is None or response_text is None:
        raise ValueError("could not extract user/assistant pair")

    return PromptCorpusLengthRecord(
        source_record_id=source_record_id,
        prompt_tokens=_whitespace_token_count(prompt_text),
        actual_output_tokens=_whitespace_token_count(response_text),
        n_turns=len(conv),
        model_id=str(row["model"]) if row.get("model") is not None else None,
        language=str(row["language"]) if row.get("language") is not None else None,
        session_id=str(row["conversation_id"]) if row.get("conversation_id") is not None else (
            str(row["conversation_hash"]) if row.get("conversation_hash") is not None else None
        ),
    )


def extract_longbench_length_record(
    row: Dict[str, Any],
    source_record_id: str,
    task: str = "",
) -> PromptCorpusLengthRecord:
    """Extract length features from a LongBench item without retaining text."""
    context = str(row.get("context") or "")
    inp = str(row.get("input") or "")
    answers = row.get("answers") or row.get("answer") or []
    if isinstance(answers, str):
        answer_text = answers
    elif isinstance(answers, list) and answers:
        answer_text = str(answers[0])
    else:
        answer_text = "ok"

    # Prefer dataset-provided length when present (token estimate).
    if isinstance(row.get("length"), (int, float)) and int(row["length"]) > 0:
        prompt_tokens = int(row["length"])
    else:
        prompt_tokens = _whitespace_token_count(context) + _whitespace_token_count(inp)

    return PromptCorpusLengthRecord(
        source_record_id=source_record_id,
        prompt_tokens=max(1, prompt_tokens),
        actual_output_tokens=_whitespace_token_count(answer_text),
        n_turns=1,
        task=task or str(row.get("dataset") or ""),
    )


def _generate_poisson_arrivals(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=float)
    gaps = rng.exponential(1.0 / rate, size=n)
    arrivals = np.cumsum(gaps)
    arrivals = arrivals - arrivals[0]
    return arrivals


def convert_prompt_lengths_to_records(
    lengths: Sequence[PromptCorpusLengthRecord],
    config: Optional[PromptCorpusConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
    dataset_type: str = DatasetType.PROMPT_CONVERSATION_CORPUS.value,
) -> Tuple[List[CanonicalIngestRecord], PromptCorpusConversionReport]:
    if config is None:
        config = PromptCorpusConversionConfig()
    if augmentation_config is None:
        augmentation_config = AugmentationConfig()

    rows_read = len(lengths)
    kept: List[PromptCorpusLengthRecord] = []
    dropped = 0
    for rec in lengths:
        if (
            rec.prompt_tokens < config.min_prompt_tokens
            or rec.actual_output_tokens < config.min_output_tokens
            or rec.prompt_tokens > config.max_prompt_tokens
            or rec.actual_output_tokens > config.max_output_tokens
        ):
            dropped += 1
            continue
        kept.append(rec)

    if config.max_requests is not None:
        kept = kept[: config.max_requests]

    replay_label = ReplayLabel.TRACE_CALIBRATED_SYNTHETIC_ARRIVALS.value
    if not kept:
        report = PromptCorpusConversionReport(
            rows_read=rows_read,
            rows_retained=0,
            rows_dropped=dropped,
            time_range_seconds=0.0,
            mean_arrival_rate=0.0,
            prompt_tokens_mean=0.0,
            output_tokens_mean=0.0,
            dataset_type=dataset_type,
            replay_label=replay_label,
            source_dataset=config.source_dataset,
        )
        return [], report

    if config.arrival_mode != "poisson":
        raise ValueError(
            f"Unsupported arrival_mode {config.arrival_mode!r}; "
            "prompt corpora require an explicit synthetic arrival process"
        )

    rng = np.random.default_rng(seed)
    arrival_times = _generate_poisson_arrivals(len(kept), config.arrival_rate, rng)
    prompt_tokens = np.array([r.prompt_tokens for r in kept], dtype=int)
    output_tokens = np.array([r.actual_output_tokens for r in kept], dtype=int)
    augmented = augment_trace(output_tokens, arrival_times, augmentation_config, rng)

    records: List[CanonicalIngestRecord] = []
    for i, src in enumerate(kept):
        prov = default_provenance(
            arrival_time=FieldProvenance.SYNTHESIZED.value,
            prompt_tokens=FieldProvenance.DERIVED.value,
            actual_output_tokens=FieldProvenance.DERIVED.value,
            session_id=(
                FieldProvenance.OBSERVED.value
                if src.session_id is not None
                else FieldProvenance.UNAVAILABLE.value
            ),
            model_id=(
                FieldProvenance.OBSERVED.value
                if src.model_id is not None
                else FieldProvenance.UNAVAILABLE.value
            ),
        )
        records.append(
            CanonicalIngestRecord(
                request_id=i,
                arrival_time=float(arrival_times[i]),
                prompt_tokens=int(prompt_tokens[i]),
                actual_output_tokens=int(output_tokens[i]),
                predicted_output_tokens=int(augmented["predicted_output_tokens"][i]),
                slo_deadline=float(augmented["slo_deadlines"][i]),
                priority=float(augmented["priorities"][i]),
                class_id=str(augmented["class_ids"][i]),
                session_id=src.session_id,
                model_id=src.model_id,
                source_dataset=config.source_dataset,
                source_split=config.source_split,
                source_record_id=src.source_record_id,
                field_provenance=prov,
                time_scale=1.0,
                replay_label=replay_label,
                dataset_type=dataset_type,
                extra={
                    "n_turns": src.n_turns,
                    "language": src.language,
                    "task": src.task,
                    "arrival_mode": config.arrival_mode,
                    "arrival_rate": config.arrival_rate,
                    "not_a_serving_trace": True,
                },
            )
        )

    validate_canonical_records(records)
    time_range = float(arrival_times[-1] - arrival_times[0]) if len(arrival_times) > 1 else 0.0
    report = PromptCorpusConversionReport(
        rows_read=rows_read,
        rows_retained=len(records),
        rows_dropped=dropped,
        time_range_seconds=time_range,
        mean_arrival_rate=(len(records) / time_range) if time_range > 0 else 0.0,
        prompt_tokens_mean=float(np.mean(prompt_tokens)),
        output_tokens_mean=float(np.mean(output_tokens)),
        dataset_type=dataset_type,
        replay_label=replay_label,
        source_dataset=config.source_dataset,
    )
    return records, report


def convert_prompt_lengths_to_requests(
    lengths: Sequence[PromptCorpusLengthRecord],
    config: Optional[PromptCorpusConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
    dataset_type: str = DatasetType.PROMPT_CONVERSATION_CORPUS.value,
) -> Tuple[List[Request], List[Dict[str, Any]], PromptCorpusConversionReport]:
    records, report = convert_prompt_lengths_to_records(
        lengths, config, seed, augmentation_config, dataset_type=dataset_type
    )
    requests, metadata = records_to_requests_and_metadata(records)
    return requests, metadata, report


def safe_hf_stream_sample(
    dataset_id: str,
    *,
    split: str = "train",
    limit: int = 20,
    token: Any = True,
) -> List[Dict[str, Any]]:
    """Stream a tiny HF sample. Raises a typed error if gated/unavailable.

    Never logs row content; callers must extract numeric fields only.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is required for HF streaming") from exc

    try:
        ds = load_dataset(dataset_id, split=split, streaming=True, token=token)
        return list(ds.take(limit))
    except Exception as exc:
        msg = str(exc).lower()
        if "gated" in msg or "403" in msg or "401" in msg:
            raise PermissionError(
                f"Dataset {dataset_id!r} is gated or access is not granted"
            ) from exc
        raise
