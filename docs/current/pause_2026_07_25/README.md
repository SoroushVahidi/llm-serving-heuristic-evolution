# Pause Snapshot 2026-07-25

This directory is the **durable GitHub pause snapshot** for the
llm-serving-heuristic-evolution project after real-trace staging, validated
window construction, and the repaired balanced discrimination pilot.

Wolverine `/mmfs1/...` storage is temporary and may be deleted. GitHub is the
durable source of truth for code, schemas, compact scientific summaries, and
reconstruction instructions.

## What is preserved in GitHub
- Reusable runners, selection logic, tests, Slurm templates
- Compact pilot/window/dataset/experiment summaries and JSON
- Part 1 audit derivatives under `audit/`
- Handoff, status, roadmap, and limitation docs (updated under `docs/current/`)

## What remains external
- Raw/processed request-level traces (~26 GB datasets root)
- Generated real-window collections (~237 MB+)
- Full per-window pilot matrices and Slurm logs
- Mooncake row-level data (redistribution prohibited until license clarified)

## How to resume
1. Read `../RESUME_HERE.md`
2. Read `PAUSE_HANDOFF.md` in this directory
3. Checkout `reality-grounded-dataset-expansion-20260724` (then sync per Part 3)
4. Reconstruct external artifacts if paths are gone (`REPRODUCTION_COMMANDS.md`)

## Authoritative documents
| Topic | Doc |
| --- | --- |
| Resume entry | `docs/current/RESUME_HERE.md` |
| This pause | `PAUSE_HANDOFF.md` + `pause_handoff_state.json` |
| Pilot | `REPAIRED_PILOT_SUMMARY.md` |
| Limitations | `KNOWN_LIMITATIONS.md` |
| Part 1 audit | `audit/PAUSE_AUDIT.md` |
