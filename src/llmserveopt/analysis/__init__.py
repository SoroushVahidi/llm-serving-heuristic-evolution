"""Diagnostic/methodology analysis modules.

Distinct from `llmserveopt.policy_separation` and `llmserveopt.selector`:
modules here never define or fit a new router/selector/policy and never
compute or imply a new scientific TEST verdict. They inspect and
characterize the behavior of already-frozen systems (e.g. the hierarchical
regime router) via TRAIN/VAL-only diagnostics.
"""
