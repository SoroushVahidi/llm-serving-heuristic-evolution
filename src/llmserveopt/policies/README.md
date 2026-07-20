# src/llmserveopt/policies

Policy implementations and their registries. Full inventory with exact
provenance: **[docs/current/BASELINES.md](../../../docs/current/BASELINES.md)**.

## Two separate registries -- do not confuse them

- **`registry.py`** -- `BASELINE_NAMES` (20 entries): the historical/internal
  policy portfolio (classical, packing, composite, and literature-inspired
  "style" policies). `BasePolicy` (`base.py`) is the shared ABC --
  `select_action(ObservableState) -> Action` is the one entry point every
  policy implements.
- **`external_baselines_registry.py`** -- `EXTERNAL_BASELINE_REGISTRY` /
  `EXTERNAL_BASELINE_NAMES` (6 entries): faithful reimplementations of real
  external systems (vLLM, vLLM-chunked-prefill, Sarathi-Serve, DistServe,
  TetriInfer, Llumnix), each pinned to an exact upstream commit. All have
  `selector_eligible=False` -- evaluation-only, never selector actions.

Neither registry imports the other.

## Selector eligibility

- The 20 policies in `registry.py` are the pool Selector v1 (`selector/`)
  draws from.
- Selector v2's actual trainable action space is a narrower, explicitly
  curated 8-policy subset -- `SELECTOR_V2_OPTION_B_POLICIES` in
  `selector/dataset_v2/candidates.py` -- not the full 20 and not the 6
  external baselines. See [selector/README.md](../selector/README.md).

## Known unregistered code

`earliest_feasible_gpu.py` (`EarliestFeasibleGPUPolicy`) is not in either
registry and has no external references -- see its own module docstring for
its status. Do not assume every `.py` file in this directory is registered
and selector-eligible; check `registry.py`/`external_baselines_registry.py`.

## Helper modules (not policies themselves)

`feasibility.py`, `scoring.py`, `tie_breaking.py` -- shared utilities used
by many policies. `tetriinfer_routing.py` / `tetriinfer_length_prediction.py`
-- composition helpers used only by `tetriinfer_paper_reimplementation.py`.
`oracle.py` -- `oracle_srtf`, a non-deployable hindsight upper bound,
excluded from both registries by design (see `make_oracle_policy`).
