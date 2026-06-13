"""
ShareGPT conversation-derived workload.

ShareGPT provides real human-assistant conversation pairs.
Prompt and response lengths are derived by tokenization.

Real fields (derived by tokenizer):   prompt_tokens, actual_output_tokens
Synthetic fields:                      arrival_time, slo_deadline, priority, class_id,
                                       predicted_output_tokens
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.types import Request
from .augmentation import AugmentationConfig, augment_trace


@dataclass
class ShareGPTConversionConfig:
    arrival_mode: str = "poisson"
    arrival_rate: float = 10.0
    duration: float = 60.0
    burst_factor: float = 5.0
    burst_fraction: float = 0.2
    tokenizer_name: Optional[str] = None
    fallback_whitespace: bool = True
    max_requests: Optional[int] = None
    min_prompt_tokens: int = 1
    min_output_tokens: int = 1
    max_prompt_tokens: int = 8192
    max_output_tokens: int = 8192


@dataclass
class ShareGPTConversionReport:
    rows_read: int
    pairs_extracted: int
    pairs_skipped: int
    rows_retained: int
    time_range_seconds: float
    mean_arrival_rate: float
    prompt_tokens_mean: float
    prompt_tokens_p95: float
    output_tokens_mean: float
    output_tokens_p95: float
    tokenizer_used: str
    seed: int
    augmentation_config_summary: Dict[str, Any]


def load_sharegpt_raw(path: Union[str, Path]) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ShareGPT file not found: {path}")
    with open(path) as f:
        return json.load(f)


def extract_prompt_response_pairs(records: List[dict]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    human_roles = {"human", "user"}
    assistant_roles = {"gpt", "assistant"}

    for record in records:
        convs = record.get("conversations", record.get("conversation", []))
        if not convs:
            continue

        prompt_text = None
        response_text = None

        for turn in convs:
            role = turn.get("from", turn.get("role", "")).lower()
            value = turn.get("value", turn.get("content", ""))
            if role in human_roles and prompt_text is None:
                prompt_text = value
            elif role in assistant_roles and prompt_text is not None and response_text is None:
                response_text = value
                break

        if prompt_text is not None and response_text is not None:
            pairs.append((prompt_text, response_text))

    return pairs


def _whitespace_tokenize(text: str) -> int:
    return len(text.split())


def tokenize_pairs(
    pairs: List[Tuple[str, str]],
    tokenizer_name: Optional[str],
    fallback_whitespace: bool,
) -> Tuple[List[Tuple[int, int]], str]:
    tokenizer_used = "whitespace"

    if tokenizer_name is not None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            def _tok(text: str) -> int:
                return len(tokenizer.encode(text, add_special_tokens=False))

            tokenizer_used = tokenizer_name
        except ImportError:
            if not fallback_whitespace:
                raise ImportError(
                    "transformers is not installed. "
                    "Install it with: pip install transformers, "
                    "or set fallback_whitespace=True to use whitespace tokenization."
                )
            _tok = _whitespace_tokenize
        except Exception:
            if not fallback_whitespace:
                raise
            _tok = _whitespace_tokenize
    else:
        _tok = _whitespace_tokenize

    counts = [(_tok(p), _tok(r)) for p, r in pairs]
    return counts, tokenizer_used


def _generate_arrivals(
    n: int,
    mode: str,
    arrival_rate: float,
    duration: float,
    burst_factor: float,
    burst_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "poisson":
        interarrivals = rng.exponential(1.0 / arrival_rate, size=n)
        arrivals = np.cumsum(interarrivals)
        arrivals = arrivals - arrivals[0]
        return arrivals

    elif mode == "bursty":
        burst_rate = arrival_rate * burst_factor
        normal_rate = arrival_rate * (1.0 - burst_fraction * burst_factor) / (1.0 - burst_fraction)
        normal_rate = max(normal_rate, arrival_rate * 0.01)

        arrivals = []
        t = 0.0
        while len(arrivals) < n:
            in_burst = rng.random() < burst_fraction
            rate = burst_rate if in_burst else normal_rate
            gap = rng.exponential(1.0 / rate)
            t += gap
            arrivals.append(t)

        arr = np.array(arrivals[:n])
        arr = arr - arr[0]
        return arr

    elif mode == "mmpp":
        lam1, lam2 = arrival_rate * 0.2, arrival_rate * 1.8
        q12, q21 = 0.5, 0.5
        arrivals = []
        t = 0.0
        state = 0
        lams = [lam1, lam2]
        qs = [[0, q12], [q21, 0]]

        while len(arrivals) < n:
            lam = lams[state]
            next_arrival = rng.exponential(1.0 / lam)
            next_switch = rng.exponential(1.0 / (qs[state][1 - state]))
            if next_arrival < next_switch:
                t += next_arrival
                arrivals.append(t)
            else:
                t += next_switch
                state = 1 - state

        arr = np.array(arrivals[:n])
        arr = arr - arr[0]
        return arr

    else:
        raise ValueError(f"Unknown arrival_mode: {mode!r}")


def convert_sharegpt_to_requests(
    records: List[dict],
    config: Optional[ShareGPTConversionConfig] = None,
    seed: int = 0,
    aug_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], ShareGPTConversionReport]:
    if config is None:
        config = ShareGPTConversionConfig()
    if aug_config is None:
        aug_config = AugmentationConfig()

    rows_read = len(records)
    pairs = extract_prompt_response_pairs(records)
    pairs_extracted = len(pairs)
    pairs_skipped = rows_read - pairs_extracted

    if config.max_requests is not None:
        pairs = pairs[: config.max_requests]

    token_counts, tokenizer_used = tokenize_pairs(
        pairs, config.tokenizer_name, config.fallback_whitespace
    )

    rng = np.random.default_rng(seed)

    valid_counts = [
        (p, r) for p, r in token_counts
        if p >= config.min_prompt_tokens and r >= config.min_output_tokens
    ]

    if len(valid_counts) == 0:
        report = ShareGPTConversionReport(
            rows_read=rows_read,
            pairs_extracted=pairs_extracted,
            pairs_skipped=pairs_skipped,
            rows_retained=0,
            time_range_seconds=0.0,
            mean_arrival_rate=0.0,
            prompt_tokens_mean=0.0,
            prompt_tokens_p95=0.0,
            output_tokens_mean=0.0,
            output_tokens_p95=0.0,
            tokenizer_used=tokenizer_used,
            seed=seed,
            augmentation_config_summary={},
        )
        return [], report

    n = len(valid_counts)
    arrival_times = _generate_arrivals(
        n, config.arrival_mode, config.arrival_rate, config.duration,
        config.burst_factor, config.burst_fraction, rng,
    )

    prompt_tokens = np.clip(
        np.array([p for p, _ in valid_counts], dtype=int),
        config.min_prompt_tokens,
        config.max_prompt_tokens,
    )
    output_tokens = np.clip(
        np.array([r for _, r in valid_counts], dtype=int),
        config.min_output_tokens,
        config.max_output_tokens,
    )

    augmented = augment_trace(output_tokens, arrival_times, aug_config, rng)

    requests: List[Request] = []
    for i in range(n):
        req = Request(
            request_id=i,
            arrival_time=float(arrival_times[i]),
            prompt_tokens=int(prompt_tokens[i]),
            predicted_output_tokens=int(augmented["predicted_output_tokens"][i]),
            actual_output_tokens=int(output_tokens[i]),
            slo_deadline=float(augmented["slo_deadlines"][i]),
            priority=float(augmented["priorities"][i]),
            class_id=augmented["class_ids"][i],
        )
        requests.append(req)

    time_range = float(arrival_times[-1] - arrival_times[0]) if n > 1 else 0.0
    mean_rate = n / time_range if time_range > 0 else 0.0

    report = ShareGPTConversionReport(
        rows_read=rows_read,
        pairs_extracted=pairs_extracted,
        pairs_skipped=pairs_skipped,
        rows_retained=n,
        time_range_seconds=time_range,
        mean_arrival_rate=mean_rate,
        prompt_tokens_mean=float(np.mean(prompt_tokens)),
        prompt_tokens_p95=float(np.percentile(prompt_tokens, 95)),
        output_tokens_mean=float(np.mean(output_tokens)),
        output_tokens_p95=float(np.percentile(output_tokens, 95)),
        tokenizer_used=tokenizer_used,
        seed=seed,
        augmentation_config_summary={
            "noise_mode": aug_config.prediction_noise.mode,
            "slo_classes": [c.class_id for c in aug_config.slo.classes],
        },
    )
    return requests, report
