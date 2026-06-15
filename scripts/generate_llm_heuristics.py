#!/usr/bin/env python3
"""
Offline LLM heuristic candidate generation script.

LLMs propose candidate heuristics offline. No LLM is called at runtime.
Generated candidates are verified and archived; evaluation is a separate step.

Usage (dry-run):
    python scripts/generate_llm_heuristics.py \\
        --providers mock --models mock --max-candidates 4 \\
        --dry-run --output-dir results/phase2b2_llm_generation/mock_candidates

Usage (real API):
    python scripts/generate_llm_heuristics.py \\
        --providers cloudrift --models auto --max-candidates 6 \\
        --temperature 0.4 --max-tokens 2500 \\
        --output-dir results/phase2b2_llm_generation/real_api_candidates
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.llm_generation.generation_loop import GenerationConfig, run_generation_loop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline LLM scheduling heuristic candidates"
    )
    parser.add_argument("--providers", default="mock",
                        help="Comma-separated provider names: cloudrift,cohere,mistral,mock")
    parser.add_argument("--models", default="auto",
                        help="Comma-separated model names (or 'auto' to use provider defaults)")
    parser.add_argument("--max-candidates", type=int, default=6,
                        help="Total number of candidates to generate across all providers")
    parser.add_argument("--max-repair-attempts", type=int, default=3,
                        help="Max repair attempts for each invalid candidate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--output-dir", default="results/phase2b2_llm_generation/candidates")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock provider; do not call real APIs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    cfg = GenerationConfig(
        providers=providers,
        models=models if models else ["auto"],
        max_candidates=args.max_candidates,
        max_repair_attempts=args.max_repair_attempts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print(f"Providers:       {providers}")
    print(f"Max candidates:  {args.max_candidates}")
    print(f"Max repair:      {args.max_repair_attempts}")
    print(f"Temperature:     {args.temperature}")
    print(f"Dry-run:         {args.dry_run}")
    print(f"Output:          {cfg.output_dir}")
    print()

    summary = run_generation_loop(cfg)

    print()
    print("=" * 50)
    print("Generation Summary")
    print("=" * 50)
    print(f"  Providers used:        {summary.providers_used}")
    print(f"  Generated:             {summary.generated}")
    print(f"  Verified OK:           {summary.verified_ok}")
    print(f"  Repaired OK:           {summary.repaired_ok}")
    print(f"  Failed:                {summary.failed}")
    print(f"  Output dir:            {cfg.output_dir}")
    print(f"  Index:                 {cfg.output_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
