# Contributing

This repository is research code. Preserve reproducibility and provenance over
tidiness.

Before changing code or docs:

1. Read `README.md`, `docs/PROJECT_MAP.md`, and `docs/current/RESUME_HERE.md`.
2. Check `git status --short --branch`.
3. Do not delete generated result directories or historical audits unless a
   cleanup task explicitly authorizes it.
4. Keep current status claims in the canonical docs synchronized:
   `docs/PROJECT_MAP.md`, `docs/current/RESUME_HERE.md`,
   `docs/current/WORK_STATUS.md`, `docs/current/NEXT_ACTIONS.md`, and
   `docs/BASELINE_STATUS.md`.

Validation for routine changes:

```bash
python scripts/check_project_handoff_consistency.py
python3 -m pytest --collect-only -q
```

Run focused tests for the files you change. Launch long experiment or full-suite
validation jobs in tmux or the cluster scheduler with wrapper metadata.
