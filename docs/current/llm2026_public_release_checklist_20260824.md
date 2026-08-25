# LLM 2026 Public Release Checklist

Date: 2026-08-24

Repository planned for public release:
`https://github.com/SoroushVahidi/llm-serving-heuristic-evolution`

This checklist does not make the repository public, delete files, push commits,
or upload artifacts.

## Required Before Public Release

- [ ] Run a secrets scan over the repository, including untracked files.
- [ ] Confirm no API keys, tokens, `.env` files, cloud credentials, SSH private
      keys, cookies, service-account files, or paid-provider credentials are
      present.
- [ ] Confirm no private SSH material or machine-specific control sockets are
      included.
- [ ] Confirm no accidental personal files are included.
- [ ] Confirm no reviewer-confidential material is included.
- [ ] Confirm no unpublished third-party data is redistributed improperly.
- [ ] Confirm raw BurstGPT and Azure trace redistribution complies with their
      licenses and source terms; if uncertain, release only derived manifests
      and instructions rather than raw files.
- [ ] Confirm license and attribution files are present for project code and
      any redistributable derived artifacts.
- [ ] Confirm `README` explains how to reproduce paper-relevant analyses or
      points to exact scripts/artifacts.
- [ ] Identify exact paper branch, commit, or tag for submission.
- [ ] Label generated paper artifacts separately from source data and code.
- [ ] Confirm large local caches, model weights, virtual environments, logs,
      and transient build outputs are excluded or intentionally documented.
- [ ] Confirm Wulver/Vulver/local machine paths in released artifacts are
      either harmless provenance or redacted where necessary.
- [ ] Confirm public-trace derived artifacts clearly distinguish original
      external traces from project-generated annotations.
- [ ] Confirm the final PDF/source package is present under `paper/llm2026/`.

## Recommended Paper Artifact Package

The repository should expose, or clearly point to, a compact paper artifact
package containing:

- final manuscript source and figures;
- joint 240-scenario workload descriptors;
- six-policy utility matrices;
- SBS/VBS/oracle summaries;
- selector/router frozen-gate summaries;
- support-expansion, guarded-composition, and typed-GP summary outputs;
- real-vLLM aggregate measurements and scheduler-trace summaries;
- dataset provenance metadata;
- environment/provenance manifests;
- README with exact reproduction instructions.

## Items to Avoid Releasing Without Review

- Raw third-party traces if redistribution is not confirmed.
- Local model caches or downloaded weights.
- Virtual environments.
- Private API calibration logs unrelated to this paper.
- Wulver/SSH credentials or connection artifacts.
- Prior project datasets not used in the final manuscript.

