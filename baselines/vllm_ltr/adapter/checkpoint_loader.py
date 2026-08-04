"""Loader for the official vLLM-LTR OPT predictor checkpoint.

Key finding this loader relies on (verified 2026-08-04 by diffing the pinned
commit's ``vllm/model_executor/models/opt.py::OPTForSequenceClassification``
against HuggingFace ``transformers``' own ``OPTForSequenceClassification``,
v4.30.0): the two classes are field-for-field identical --
``self.model = OPTModel(config)`` + ``self.score =
nn.Linear(config.word_embed_proj_dim, config.num_labels, bias=False)``, no
pooling/MLP/dropout added. The vLLM fork's custom class exists only to plug
into vLLM's paged-attention/logits-processor plumbing for serving
efficiency; it is not architecturally distinct from stock HF
``OPTForSequenceClassification``. This means the official checkpoint
(``LLM-ltr/OPT-Predictors`` on Hugging Face) should be loadable directly
through plain ``transformers``, with no custom weight-key remapping code --
UNTESTED ASSUMPTION: this has not been verified against the actual
checkpoint files (no network fetch of the ~GB-scale weights was performed
for this scaffold; see the audit doc's "Remaining work" section). If the
real checkpoint fails to load via ``AutoModelForSequenceClassification``,
that is the first thing to check.

torch/transformers are optional dependencies here (see the ``vllm_ltr``
extra in ``pyproject.toml``); the core simulator has no hard dependency on
either.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import provenance
from .errors import (
    MissingCheckpointError,
    MissingDependencyError,
    StaleArtifactError,
    VersionMismatchError,
)

PROVENANCE_SIDECAR_FILENAME = "vllm_ltr_provenance.json"


def _require_torch_and_transformers():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            "torch is required to load the vLLM-LTR OPT predictor but is not "
            "installed. Install the optional extra: pip install -e "
            "'.[vllm_ltr]' (or: pip install torch transformers)."
        ) from exc
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            "transformers is required to load the vLLM-LTR OPT predictor but "
            "is not installed. Install the optional extra: pip install -e "
            "'.[vllm_ltr]' (or: pip install torch transformers)."
        ) from exc
    return torch, transformers


def _major_minor(version_str: str) -> str:
    parts = version_str.split(".")
    return ".".join(parts[:2])


def _read_provenance_sidecar(checkpoint_dir: str) -> dict:
    sidecar_path = os.path.join(checkpoint_dir, PROVENANCE_SIDECAR_FILENAME)
    if not os.path.isfile(sidecar_path):
        raise StaleArtifactError(
            f"Checkpoint directory {checkpoint_dir!r} has no "
            f"{PROVENANCE_SIDECAR_FILENAME} sidecar recording which pinned "
            "vllm-ltr commit and library versions it was verified against. "
            "Refusing to load an unprovenanced artifact -- write one with "
            "write_local_provenance_sidecar() first."
        )
    with open(sidecar_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_local_provenance_sidecar(
    checkpoint_dir: str,
    *,
    pinned_commit: str,
    torch_version: str,
    transformers_version: str,
) -> str:
    """Write a ``vllm_ltr_provenance.json`` sidecar into ``checkpoint_dir``.

    Not called by any production path -- this exists so tests (and future
    real checkpoint downloads) can record the provenance this loader
    requires before a checkpoint is considered loadable.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    sidecar_path = os.path.join(checkpoint_dir, PROVENANCE_SIDECAR_FILENAME)
    payload = {
        "pinned_commit": pinned_commit,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return sidecar_path


def _validate_provenance_sidecar(sidecar: dict, torch_mod, transformers_mod) -> None:
    recorded_commit = sidecar.get("pinned_commit")
    if recorded_commit != provenance.PINNED_COMMIT:
        raise StaleArtifactError(
            f"Checkpoint sidecar records pinned_commit={recorded_commit!r}, "
            f"but this adapter is pinned to {provenance.PINNED_COMMIT!r}. "
            "Refusing to load a checkpoint verified against a different "
            "vllm-ltr commit."
        )
    recorded_torch = sidecar.get("torch_version")
    recorded_transformers = sidecar.get("transformers_version")
    installed_torch = torch_mod.__version__
    installed_transformers = transformers_mod.__version__
    if recorded_torch is None or _major_minor(recorded_torch) != _major_minor(installed_torch):
        raise VersionMismatchError(
            f"Checkpoint sidecar recorded torch=={recorded_torch!r}, but "
            f"installed torch=={installed_torch!r}. Refusing to load: "
            "numeric behavior of the score head is not guaranteed identical "
            "across torch minor versions."
        )
    if recorded_transformers is None or _major_minor(recorded_transformers) != _major_minor(
        installed_transformers
    ):
        raise VersionMismatchError(
            f"Checkpoint sidecar recorded transformers=={recorded_transformers!r}, "
            f"but installed transformers=={installed_transformers!r}. Refusing "
            "to load: OPTForSequenceClassification's pooling behavior has "
            "changed across transformers releases in the past."
        )


@dataclass
class OPTPredictorHandle:
    """A loaded, ready-to-score official OPT predictor.

    ``score()`` is deterministic: the underlying model is put in ``eval()``
    mode and every forward pass runs under ``torch.no_grad()`` -- no
    dropout, no sampling.
    """

    model: object
    tokenizer: object
    num_labels: int

    def score(self, prompt_text: str) -> float:
        torch, _ = _require_torch_and_transformers()
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits  # shape (1, num_labels)
        if self.num_labels > 1:
            return float(logits.argmax(dim=-1).item())
        return float(logits[0, 0].item())


def load_opt_predictor_from_local(checkpoint_dir: str) -> OPTPredictorHandle:
    """Load the official predictor from a local checkpoint directory.

    Requires a ``vllm_ltr_provenance.json`` sidecar in ``checkpoint_dir``
    (see ``write_local_provenance_sidecar``) recording the pinned commit and
    library versions the checkpoint was verified against; rejects the
    checkpoint outright if that sidecar is missing, stale, or records
    versions that don't match what's currently installed. See
    ``errors.py`` for the exact exception raised in each case.
    """
    if not os.path.isdir(checkpoint_dir):
        raise MissingCheckpointError(
            f"No checkpoint directory found at {checkpoint_dir!r}. Download "
            f"the official checkpoint from "
            f"https://huggingface.co/{provenance.CHECKPOINT_HF_REPO} first."
        )
    torch_mod, transformers_mod = _require_torch_and_transformers()
    sidecar = _read_provenance_sidecar(checkpoint_dir)
    _validate_provenance_sidecar(sidecar, torch_mod, transformers_mod)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    num_labels = int(model.config.num_labels)
    return OPTPredictorHandle(model=model, tokenizer=tokenizer, num_labels=num_labels)
