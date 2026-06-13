"""
Deterministic augmentation of missing trace fields.

Augments real trace data with synthetic fields that are not present in
public datasets. All augmentation is transparent, seeded, and documented.
Fields produced here are clearly labeled as synthetic in trace metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class PredictionNoiseConfig:
    mode: str = "lognormal"
    sigma: float = 0.35
    bias: float = 0.0
    min_tokens: int = 1
    max_tokens: int = 4096


@dataclass
class SLOClassConfig:
    class_id: str
    weight: float
    priority: float
    slo_slack: float


@dataclass
class SLOAugConfig:
    classes: List[SLOClassConfig]


DEFAULT_SLO_AUG = SLOAugConfig(classes=[
    SLOClassConfig("interactive", weight=0.50, priority=3.0, slo_slack=2.0),
    SLOClassConfig("standard",    weight=0.35, priority=2.0, slo_slack=6.0),
    SLOClassConfig("batch",       weight=0.15, priority=1.0, slo_slack=20.0),
])


def apply_prediction_noise(
    actual_tokens: np.ndarray,
    config: PredictionNoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    actual = np.asarray(actual_tokens, dtype=float)
    mode = config.mode

    if mode == "exact":
        predicted = actual.copy()
    elif mode == "lognormal":
        noise = np.exp(rng.normal(config.bias, config.sigma, size=len(actual)))
        predicted = np.round(actual * noise)
    elif mode == "biased_under":
        noise = np.exp(rng.normal(-0.3, config.sigma, size=len(actual)))
        predicted = np.round(actual * noise)
    elif mode == "biased_over":
        noise = np.exp(rng.normal(0.3, config.sigma, size=len(actual)))
        predicted = np.round(actual * noise)
    elif mode == "bucket":
        buckets = [
            (1,    64,   32),
            (64,   256,  128),
            (256,  1024, 512),
            (1024, 4096, 2048),
        ]
        predicted = np.full_like(actual, 4096, dtype=float)
        for lo, hi, mid in buckets:
            mask = (actual >= lo) & (actual < hi)
            predicted[mask] = mid
    else:
        raise ValueError(f"Unknown prediction noise mode: {mode!r}")

    predicted = np.clip(predicted, config.min_tokens, config.max_tokens)
    return predicted.astype(int)


def augment_slo_classes(
    n: int,
    config: SLOAugConfig,
    arrival_times: np.ndarray,
    rng: np.random.Generator,
) -> tuple[List[str], List[float], List[float]]:
    weights = np.array([c.weight for c in config.classes], dtype=float)
    weights /= weights.sum()
    indices = rng.choice(len(config.classes), size=n, p=weights)

    class_ids: List[str] = []
    priorities: List[float] = []
    slo_deadlines: List[float] = []

    for i, idx in enumerate(indices):
        cls = config.classes[idx]
        class_ids.append(cls.class_id)
        priorities.append(cls.priority)
        slo_deadlines.append(float(arrival_times[i]) + cls.slo_slack)

    return class_ids, priorities, slo_deadlines


@dataclass
class AugmentationConfig:
    prediction_noise: PredictionNoiseConfig = field(
        default_factory=PredictionNoiseConfig
    )
    slo: SLOAugConfig = field(default_factory=lambda: DEFAULT_SLO_AUG)
    synthetic_fields: List[str] = field(
        default_factory=lambda: [
            "predicted_output_tokens",
            "class_id",
            "priority",
            "slo_deadline",
        ]
    )


def augment_trace(
    actual_tokens: np.ndarray,
    arrival_times: np.ndarray,
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    predicted = apply_prediction_noise(actual_tokens, config.prediction_noise, rng)
    class_ids, priorities, slo_deadlines = augment_slo_classes(
        len(actual_tokens), config.slo, arrival_times, rng
    )
    return {
        "predicted_output_tokens": predicted,
        "class_ids": class_ids,
        "priorities": priorities,
        "slo_deadlines": slo_deadlines,
    }


def load_augmentation_config(path_or_dict: Union[str, dict]) -> AugmentationConfig:
    if isinstance(path_or_dict, dict):
        cfg = path_or_dict
    else:
        import yaml
        with open(path_or_dict) as f:
            cfg = yaml.safe_load(f)

    noise_cfg = cfg.get("prediction_noise", {})
    noise = PredictionNoiseConfig(
        mode=noise_cfg.get("mode", "lognormal"),
        sigma=float(noise_cfg.get("sigma", 0.35)),
        bias=float(noise_cfg.get("bias", 0.0)),
        min_tokens=int(noise_cfg.get("min_tokens", 1)),
        max_tokens=int(noise_cfg.get("max_tokens", 4096)),
    )

    slo_raw = cfg.get("slo_classes", None)
    if slo_raw is not None:
        slo_classes = [
            SLOClassConfig(
                class_id=c["class_id"],
                weight=float(c["weight"]),
                priority=float(c["priority"]),
                slo_slack=float(c["slo_slack"]),
            )
            for c in slo_raw
        ]
        slo = SLOAugConfig(classes=slo_classes)
    else:
        slo = DEFAULT_SLO_AUG

    return AugmentationConfig(prediction_noise=noise, slo=slo)
