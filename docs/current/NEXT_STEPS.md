# Next Steps

> **SUPERSEDED — see [`docs/current/NEXT_ACTIONS.md`](NEXT_ACTIONS.md) and
> [`docs/PROJECT_MAP.md`](../PROJECT_MAP.md).** This file is retained as
> historical context for an older simulator-calibration/selector planning
> thread. It is not the current prioritized action list.

Short and actionable. See `PROJECT_STATUS.md`, `ROADMAP_GAP_ANALYSIS.md`, and
`RESEARCH_ROADMAP.md` for context.

## Recommended Sequence

1. **Run bounded simulator calibration and discriminative-power validation.**
   This is the next major task.
2. Validate whether KV/cache reuse, prefix reuse, prefill/decode contention,
   overload/capacity, and SLO feasibility now create measurable policy-reward
   separation instead of near-ceiling ANWG ties.
3. Rerun small controlled subsets from:
   - V2 real-OOD;
   - SwissAI;
   - TraceLab;
   - SLO/deadline augmentation;
   - selected frontier/cartography windows.
4. If controlled reruns show improved, scientifically defensible separation,
   retrain the 27-policy suitability model with regret/listwise objectives,
   near-tie handling, uncertainty, and grouped OOD evaluation.
5. If selector/suitability quality improves, refresh targeted module-credit
   intervention design using reliable suitability and uncertainty signals.
6. If module-credit learning beats simple donor-policy baselines, launch only
   restricted, evidence-guided composition/synthesis.
7. Compare the final adaptive/synthesized method against fixed policies and
   external baselines only after the simulator evaluation target is calibrated
   enough to distinguish meaningful policy behavior.

## Stop Conditions

- Do not collect more generic datasets as the main next action while
  simulator/objective saturation is unresolved.
- Do not retrain selectors on weakly discriminative labels and treat that as
  progress.
- Do not launch broad structural synthesis while `COMBINER_TRAINING_SIGNAL =
  WEAK`.
- Do not treat SwissAI/TraceLab zero-gain results as natural FIFO optimality.
- Do not treat synthetic SLO augmentation as natural real-OOD evidence.

## Immediate Development Prompt

Design a bounded simulator calibration/discriminative-power task that:

- identifies the simulator code paths connecting prompt/context/reuse to
  service time and KV occupancy;
- defines controlled windows where each intended resource pressure should
  matter;
- verifies policy ranking changes under those pressures;
- preserves ANWG as the primary metric while auditing ceiling behavior;
- does not change validated historical experiment artifacts.
