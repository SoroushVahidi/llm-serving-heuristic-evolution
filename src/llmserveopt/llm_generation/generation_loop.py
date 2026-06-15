"""
Main generation loop: generate → verify → repair → archive.

Usage
-----
cfg = GenerationConfig(providers=["cloudrift"], max_candidates=6, ...)
summary = run_generation_loop(cfg)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..heuristics.verifier import verify_heuristic
from .candidate_io import (
    CandidateRecord,
    _git_commit,
    _sha256,
    make_candidate_dir,
    save_candidate,
    save_repair_attempt,
    update_index,
)
from .prompt_templates import build_generation_messages, build_repair_messages
from .providers import build_providers
from .repair import extract_json, run_repair_loop, verify_and_collect_errors


@dataclass
class GenerationConfig:
    providers: List[str] = field(default_factory=lambda: ["mock"])
    models: List[str] = field(default_factory=lambda: ["auto"])
    max_candidates: int = 6
    max_repair_attempts: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000
    output_dir: Path = Path("results/phase2b2_llm_generation/candidates")
    dry_run: bool = False
    seed: int = 42
    verbose: bool = True


@dataclass
class GenerationSummary:
    generated: int = 0
    verified_ok: int = 0
    repaired_ok: int = 0
    failed: int = 0
    providers_used: List[str] = field(default_factory=list)
    candidate_dirs: List[str] = field(default_factory=list)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def run_generation_loop(cfg: GenerationConfig) -> GenerationSummary:
    """Run the full generation loop and return a summary."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    providers_to_use = (
        ["mock"] if cfg.dry_run else cfg.providers
    )

    provider_instances = build_providers(providers_to_use)
    available = [p for p in provider_instances if p.is_available()]

    if not available:
        print("[WARN] No available providers. Nothing generated.")
        return GenerationSummary()

    git_commit = _git_commit()
    summary = GenerationSummary()
    summary.providers_used = [p.name for p in available]

    prompt_messages = build_generation_messages()
    candidate_counter = 0

    for provider in available:
        n_from_this = max(1, cfg.max_candidates // len(available))
        remaining = cfg.max_candidates - summary.generated
        n_from_this = min(n_from_this, remaining)

        model = cfg.models[0] if cfg.models else "auto"

        for _ in range(n_from_this):
            candidate_counter += 1
            cid = f"c{candidate_counter:03d}"
            _log(cfg.verbose, f"\n[{provider.name}] Generating candidate {cid}...")

            t0 = time.monotonic()
            resp = provider.generate(
                prompt_messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                model=model,
            )
            gen_time = time.monotonic() - t0

            summary.generated += 1
            extracted = extract_json(resp.text)

            cand_dir = make_candidate_dir(output_dir, provider.name, resp.model, cid)

            if extracted is None:
                _log(cfg.verbose, f"  [FAIL] Could not parse JSON from response")
                vr_dict = {"valid": False, "errors": [["JSON_PARSE_ERROR", "Could not extract JSON"]]}
                record = CandidateRecord(
                    candidate_id=cid, provider=provider.name, model=resp.model,
                    temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                    generation_time=gen_time, repair_attempt_count=0,
                    verification_ok=False, sha256="none", git_commit=git_commit,
                )
                save_candidate(
                    cand_dir,
                    prompt_messages=prompt_messages,
                    raw_response=resp.text,
                    candidate_json=None,
                    verifier_result=vr_dict,
                    metadata=record,
                )
                update_index(output_dir, record, cand_dir)
                summary.failed += 1
                summary.candidate_dirs.append(str(cand_dir))
                continue

            valid, errors = verify_and_collect_errors(extracted)
            sha = _sha256(extracted)
            repair_count = 0

            if not valid:
                _log(cfg.verbose, f"  [VERIFY FAIL] {len(errors)} errors — attempting repair")
                repaired, repair_ok, repair_count = run_repair_loop(
                    extracted,
                    provider,
                    cfg.max_repair_attempts,
                    build_repair_messages,
                    lambda n, raw, cand: save_repair_attempt(cand_dir, n, raw, cand),
                    temperature=max(0.0, cfg.temperature - 0.3),
                    max_tokens=cfg.max_tokens,
                )
                if repair_ok:
                    extracted = repaired
                    valid = True
                    sha = _sha256(extracted)
                    _log(cfg.verbose, f"  [REPAIRED] in {repair_count} attempt(s)")
                    summary.repaired_ok += 1
                else:
                    _log(cfg.verbose, f"  [REPAIR FAIL] still invalid after {repair_count} attempt(s)")
                    summary.failed += 1
            else:
                _log(cfg.verbose, f"  [VERIFY OK] {extracted.get('name', cid)}")

            if valid:
                summary.verified_ok += 1

            vr_dict = {
                "valid": valid,
                "errors": list(errors),
            }
            record = CandidateRecord(
                candidate_id=cid,
                provider=provider.name,
                model=resp.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                generation_time=gen_time,
                repair_attempt_count=repair_count,
                verification_ok=valid,
                sha256=sha,
                git_commit=git_commit,
            )
            save_candidate(
                cand_dir,
                prompt_messages=prompt_messages,
                raw_response=resp.text,
                candidate_json=extracted if extracted else None,
                verifier_result=vr_dict,
                metadata=record,
            )
            update_index(output_dir, record, cand_dir)
            summary.candidate_dirs.append(str(cand_dir))

    return summary
