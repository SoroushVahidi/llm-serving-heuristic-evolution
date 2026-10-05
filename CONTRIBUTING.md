# Contributing

This repository is research code. Preserve reproducibility and provenance over
tidiness.

Before changing code or docs:

1. Read `README.md`, `REPRODUCIBILITY.md`, and `docs/current/README.md`. The
   documentation authority order is in `docs/README.md`.
2. Check `git status --short --branch`.
3. Never edit frozen artifacts in place. These include the confirmatory
   `experiments/` directories, the archives under `release/`, and the
   hash-locked `SBS_OVERRIDE_*` documents. Add a separate, documented
   correction instead.
4. Do not delete generated result directories or historical audits unless a
   cleanup task explicitly authorizes it. Mark outdated entry points as
   historical instead of rewriting them.

Validation for routine changes:

```bash
python3 scripts/check_project_handoff_consistency.py
python3 paper/performance_evaluation/scripts/build_claim_manifest.py --check
python3 -m pytest --collect-only -q
```

Run focused tests for the files you change. Launch long experiment or full-suite
validation jobs in tmux or the cluster scheduler.
