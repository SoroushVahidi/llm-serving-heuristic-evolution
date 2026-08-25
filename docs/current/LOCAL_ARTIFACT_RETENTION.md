# Local Artifact Retention

Generated outputs under `results/` and `logs/` are local-only and ignored by
Git unless a narrow provenance exception is explicitly listed in `.gitignore`.

Retention rules:

1. Do not delete provenance-bearing result directories during documentation or
   code cleanup.
2. Preserve failed runs when they explain a fix or safety stop.
3. Preserve canonical completed runs until their manifests, configs, code SHA,
   and summaries are documented.
4. Large scientifically relevant local result sets should receive a manifest or
   audit pointer before any compression, archival, or deletion decision.
5. Disposable caches (`__pycache__`, `.pytest_cache`, `.mypy_cache`,
   `.ruff_cache`, `*.egg-info`) may be removed when ignored and reproducible.

Known large local artifact requiring a separate decision:

- `results/module_credit_overnight/`: The ~108 GB folder of AutoML checkpoints has been formally audited. The resulting 10,000 trial-level evaluation metrics, hyperparameters, and negative-result conclusions have been published to `SoroushVahidi/module-intervention-credit` on Hugging Face. The 10,000 individual `models/` checkpoints (e.g. `.pt`, `.joblib`) offer no independent Hugging Face publication value due to their weak generalization status. **Permission is granted to safely delete the `models/` subdirectory to reclaim disk space.** The summary reports, JSON manifests, and CSV leaderboards MUST be retained locally for provenance.

Protected Phase G artifacts:

- `results/apt_serve_phase_g_overnight_20260807_011542/`
- `results/apt_serve_phase_g_resume_20260807_174028/`
- `results/apt_serve_phase_g_analysis_20260809_190000/`
