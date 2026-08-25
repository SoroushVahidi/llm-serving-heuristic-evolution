# Joint-240 Terminal Criticality — Discovery Audit & Gate Decision (2026-08-25)

**Host:** `al-khwarizmi`  
**Branch:** `contextual-compositional-heuristics-20260731` @ `2987b718…`  
**Upstream:** ahead 2 / behind 0  
**Purpose:** Mandatory local + Wulver discovery before any new joint-240
terminal-criticality launch.

---

## A. Local discovery (manuscript-relevant)

| Experiment | Status | In manuscript? | Overlaps this task? |
|---|---|---|---|
| `decision_criticality_terminal_anwg_v1` (A/B/C TRAIN/VAL, 734/27) | **COMPLETED_AND_IN_MANUSCRIPT** (§4.3) | Yes | Parent comparison only |
| `decision_criticality_uncertainty_existing_data_v1` | **COMPLETED_NOT_IN_MANUSCRIPT** (CIs used in §4.3 rewrite) | Partially (CIs) | A/B/C uncertainty |
| **`decision_criticality_terminal_anwg_joint240_v1`** | **COMPLETED_NOT_IN_MANUSCRIPT** | **No** | **YES — primary question** |
| `decision_criticality_terminal_utility_joint240_v1` | COMPLETED_NOT_IN_MANUSCRIPT | No | Continuous utility, not SBS continuation |
| `…/utility_robustness_v1` | COMPLETED (analysis-only; continuous Δ unavailable) | No | Not SBS continuation |
| `joint240_same_distribution_adaptive_exploitability_v1` | COMPLETED_AND_IN_MANUSCRIPT | Yes | Parent folds / A_live |
| `joint240_strong_learned_selector_v1` | COMPLETED_NOT_IN_MANUSCRIPT | No | Post-hoc join only |
| `joint240_guarded_abstaining_selector_v1` | COMPLETED_NOT_IN_MANUSCRIPT | No | Related adaptive |
| `public_replay_load_scaling_v1/v2` | COMPLETED (local+Wulver) | Partially (limitations) | No |

**No local scientific processes / tmux / locks.** Dirty tree preserved.

### Existing joint-240 terminal-ANWG (authoritative)

- Path: `experiments/decision_criticality_terminal_anwg_joint240_v1/`
- Design: `docs/design/DECISION_CRITICALITY_TERMINAL_ANWG_JOINT240_V1.md`
- Caps: 10 disagreement + 5 agreement/scenario (cap 3600); **3541** acquired
- Nonzero: **206** (5.82%); top-1% mass **0.421**; AUROC **0.680**
- Verdict: **`JOINT240_TERMINAL_CRITICALITY_REPLICATED`**
- Continuation: **OOF A_live only** (cloned Alive)
- H10: **unavailable** on joint-240
- Analysis: `docs/current/decision_criticality_terminal_anwg_joint240_v1_analysis_20260825.md`
- Environment: **local** (not a Wulver job)

---

## B. Wulver / Wolverine discovery

- Control master `~/.ssh/cm/wulver.sock`: **live**
- Login: `login02` / `sv96`
- **`squeue -u sv96`:** empty (no RUNNING/PENDING)
- Recent completed (relevant):
  - `1194758` joint240_same_distribution_adaptive_v1 COMPLETED
  - `1195452` infra smoke COMPLETED
  - `1195488_*` public_replay_load_scaling_v1 COMPLETED
  - `1195618_*` public_replay_load_scaling_v2 COMPLETED
  - Family-A medium array `1193266_*` (older)
- **No Wulver job** for `terminal_anwg_joint240` / joint-240 criticality
- No uncollected remote criticality outputs found under home experiment listings

---

## C. Cross-environment reconciliation / GATE

| Question | Answer |
|---|---|
| Does an experiment already answer joint-240 sparse/concentrated terminal criticality? | **YES** — `decision_criticality_terminal_anwg_joint240_v1` |
| Is it in the manuscript? | **NO** (manuscript §4.3 still TRAIN/VAL A/B/C only) |
| Duplicate primary A_live criticality? | **FORBIDDEN by gate** |
| Still scientifically missing? | **SBS continuation-policy robustness** on the same acquisition points; post-hoc A_hgb join |

### Gate decision

1. **Do not relaunch** primary joint-240 Alive-continuation criticality.
2. **Do not increase acquisition caps** in a new primary run solely to chase more nonzero events (outcome-adaptive risk; existing n_nz=206 already ≫ 27).
3. **Proceed only** with:
   - analysis/reconciliation of the existing joint-240 result for the final report;
   - a **new, narrowly scoped** SBS-continuation robustness experiment on the
     **exact frozen acquisition keys** from the parent branches;
   - optional **post-hoc** A_hgb scenario join (analysis-only).

---

## D. Parent reproduction (spot-check)

| Quantity | Value |
|---:|
| SBS / VBS / headroom | 0.314072 / 0.333106 / 0.019034 |
| A_scen / A_live / A_hgb | 0.3059 / 0.2840 / 0.3145 |
| A/B/C: 734 states, 27 nonzero, top-1% 48.3%, AUROC 0.513 | confirmed |
| Joint-240 criticality: 3541 / 206 / 5.82% / top-1% 42.1% / AUROC 0.680 | confirmed |
