# Pause Handoff — 2026-07-25

## Project objective
Deterministic LLM-serving scheduling simulator; study selection, composition, and
synthesis of request-scheduling policies under realistic load. Primary metric: ANWG.

## Canonical branches / worktrees at pause
| Role | Path | Branch | SHA (Part 2 start) |
| --- | --- | --- | --- |
| Dataset / Part 2 work | `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion` | `reality-grounded-dataset-expansion-20260724` | `4dd97eadd16aa65512db61af07f7750596c08d14` |
| Final integration | `.../llm-serving-heuristic-evolution-final-integration` | `wulver-final-integration-20260721` | `b0768f28016442527d2ebe9dcbc9efdf24f26da0` |
| Legacy (dirty) | `.../llm-serving-heuristic-evolution` | `wulver-policy-composition-readiness` | `c8aee12` (dirty; deferred) |

## Staged real datasets
Tier 1 complete under external `datasets/` (~26 GB): BurstGPT, Azure 2023/2024,
Bailian/Qwen, Mooncake (internal OOD; redistribution prohibited).

## Real-window status
`ALL_COMPLETE_VALID` at `/mmfs1/project/ikoutis/sv96/llmserveopt-data/real_window_construction_20260725T035054Z` (SHA `4dd97eadd16aa65512db61af07f7750596c08d14`).

## Repaired-pilot result
`LOAD_DISCRIMINATION_PILOT = PARTIALLY_READY` (job 1143392).
Key: sat 0.072; exact-tie 0.604; near-tie 0.804; mean margin ~0.0125; 7 winners;
Mooncake included (50). Signal gates failed. **No full sweep.**

## Selector status
Still useful-not-solved / data-or-simulator limited. Repaired pilot does **not**
authorize unrestricted selector retraining or a full fingerprint sweep; it
supports targeted load/simulator discrimination work first.

## Composition / synthesis
Native composition pilot remains `NO_GO`. Structural synthesis empirically
`NOT_READY`. Repaired pilot does **not** reopen composition.

## Simulator gaps
Existing gaps remain open. Additional caveat: pilot tie diagnostics use outcome
signatures, not true action traces.

## Exact next scientific decision
**Targeted simulator / load-regime discrimination repair** using natural+busy
primary evidence (and scaled only as stress), before any full 27-policy sweep.
Do not treat aggregate PARTIALLY_READY as sweep authorization.

## First resume commands
```bash
cd /path/to/clone
git fetch origin --prune --tags
git checkout reality-grounded-dataset-expansion-20260724
# after Part 3 push: git pull --ff-only
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate repo-env
less docs/current/RESUME_HERE.md
less docs/current/pause_2026_07_25/PAUSE_HANDOFF.md
```
