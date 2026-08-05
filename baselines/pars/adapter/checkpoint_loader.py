"""Loader for the official PARS ``PairwiseRanker`` architecture + a locally
self-trained checkpoint (see PROVENANCE.md -- no pretrained checkpoint is
shipped by the official repository; a real BERT-based pairwise ranker was
trained here using their unmodified training script).

Does NOT vendor the official repository's source into this project. The
``PairwiseRanker`` class is dynamically imported, at runtime, directly from
the pinned local clone (see ``provenance.DEFAULT_OFFICIAL_CLONE_PATH``) --
the exact same class definition their own training and serving scripts use
(confirmed byte-for-byte identical between
``predictor_train/scripts/train_pairwise_bert.py`` and
``predictor_serving/scripts/serve_predictor_score.py`` via ``diff``), never
copy-pasted into this file.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import provenance
from .errors import (
    MissingCheckpointError,
    MissingDependencyError,
    MissingOfficialCloneError,
    StaleCloneCommitError,
)


def _require_torch_and_transformers():
    try:
        import torch
        import transformers
    except ImportError as e:
        raise MissingDependencyError(
            "PARS adapter requires torch and transformers. Install the "
            "'pars' extra or `pip install torch transformers`."
        ) from e
    return torch, transformers


def resolve_official_clone_path() -> str:
    return os.environ.get("PARS_OFFICIAL_CLONE_PATH", provenance.DEFAULT_OFFICIAL_CLONE_PATH)


def verify_official_clone(clone_path: Optional[str] = None) -> str:
    """Locate the pinned local clone and verify its HEAD commit matches
    ``provenance.PINNED_COMMIT``. Raises rather than silently using an
    unpinned/drifted copy of the official code."""
    path = clone_path or resolve_official_clone_path()
    if not os.path.isdir(os.path.join(path, ".git")):
        raise MissingOfficialCloneError(
            f"No git clone of {provenance.OFFICIAL_REPOSITORY} found at {path!r}. "
            f"Clone it first: git clone {provenance.OFFICIAL_REPOSITORY} {path}"
        )
    head = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != provenance.PINNED_COMMIT:
        raise StaleCloneCommitError(
            f"Local clone at {path!r} is at commit {head!r}, expected the "
            f"pinned commit {provenance.PINNED_COMMIT!r}. Re-clone or "
            f"`git checkout {provenance.PINNED_COMMIT}` before using this adapter."
        )
    return path


def _import_pairwise_ranker_class(clone_path: str):
    """Dynamically import the official, unmodified ``PairwiseRanker`` class
    definition directly from the pinned clone -- never duplicated here.

    Imports from ``predictor_train/scripts/train_pairwise_bert.py`` rather
    than ``predictor_serving/scripts/serve_predictor_score.py`` -- both
    define a functionally identical ``PairwiseRanker`` (see PROVENANCE.md),
    but the training script's module-level code is side-effect-free on
    import (all data loading/training is inside ``main()``, guarded by
    ``if __name__ == "__main__"``), whereas the serving script executes a
    FastAPI app + tokenizer + checkpoint load at import time -- avoiding an
    unnecessary ``fastapi`` dependency and avoiding having to load (or
    monkeypatch around) a real checkpoint just to obtain a class
    definition.
    """
    module_path = os.path.join(
        clone_path, "predictor_train", "scripts", "train_pairwise_bert.py"
    )
    if not os.path.exists(module_path):
        raise MissingOfficialCloneError(
            f"Expected official file not found: {module_path}. Clone may be "
            "incomplete or the official repository's layout has changed."
        )
    spec = importlib.util.spec_from_file_location("pars_official_train_pairwise_bert", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PairwiseRanker


@dataclass
class PARSPredictorHandle:
    model: object
    tokenizer: object
    max_length: int = provenance.MAX_LENGTH

    def score_batch(self, prompts: Sequence[str], batch_size: int = 8) -> List[float]:
        torch, _ = _require_torch_and_transformers()
        device = next(self.model.parameters()).device
        scores: List[float] = []
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_scores = self.model.score(inputs["input_ids"], inputs["attention_mask"])
            scores.extend(float(s) for s in batch_scores.detach().cpu().tolist())
        return scores

    def score(self, prompt: str) -> float:
        return self.score_batch([prompt])[0]


def load_pars_predictor(
    checkpoint_path: str,
    clone_path: Optional[str] = None,
    model_name: str = provenance.MODEL_NAME,
    max_length: int = provenance.MAX_LENGTH,
    device: Optional[str] = None,
) -> PARSPredictorHandle:
    torch, transformers = _require_torch_and_transformers()

    if not os.path.exists(checkpoint_path):
        raise MissingCheckpointError(f"No trained PARS checkpoint at {checkpoint_path!r}.")

    verified_clone_path = verify_official_clone(clone_path)
    PairwiseRanker = _import_pairwise_ranker_class(verified_clone_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = PairwiseRanker(model_name)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(resolved_device)

    return PARSPredictorHandle(model=model, tokenizer=tokenizer, max_length=max_length)
