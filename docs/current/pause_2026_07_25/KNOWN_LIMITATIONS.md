# Known Limitations at Pause

1. **Outcome-signature diagnostics**: repaired-pilot “behavioral disagreement” and tie-cause labels use outcome signatures (completion/drops/rounded ANWG/SLO/batch), **not** true scheduler action traces.
2. **Natural-load discrimination weak**: near_tie_rate overall 0.804; signal gates failed (`near_tie_below_0_80`, `pct_margin_gt_0_02_above_0_20`).
3. **Scaled evidence ≠ natural proof**: stronger margins on time-scaled windows are stress evidence.
4. **No full fingerprint sweep authorized**.
5. **Mooncake**: license not explicitly specified; redistribution prohibited until clarified; internal OOD only.
6. **Simulator/heuristic gaps** remain (see `docs/current/KNOWN_SIMULATOR_HEURISTIC_GAPS.md`); dataset commits did not fix simulator semantics.
7. **Composition/synthesis**: native composition pilot remains `NO_GO`; structural synthesis empirically `NOT_READY`. Repaired pilot does **not** reopen composition.
8. **External storage ephemeral**: datasets (~26GB) and windows (~237MB+) may vanish with Wolverine deletion.
9. **Legacy dirty worktree** may still hold older drafts; inventoried in Part 1; resolution deferred to Part 3.
