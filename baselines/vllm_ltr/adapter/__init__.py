"""Project-specific adapter code for the vLLM-LTR baseline.

Deliberately separate from ``baselines/vllm_ltr/official_reference/``, which
holds literal pinned-source citations only. Nothing in this package imports
or executes official vLLM-LTR code; it independently reproduces the two
pieces of official behavior verified in ``official_reference/`` (the
ranking/tie-break rule, and the predictor's backbone+head architecture),
against real weights when available.
"""
