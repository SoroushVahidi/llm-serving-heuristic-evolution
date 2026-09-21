# IMPLEMENTATION_V1.md

## Reference Reserve Sensitivity Analysis Implementation Context

### 1. Context of Pre-existing Execution
- Previous commit `102bee58433266b7b03ff5c5fc1343407ecf2dc8` contained only a minimal skeleton/placeholder and did not execute the actual scientific simulation.
- SLURM Jobs `1303667`, `1303668`, and `1303669` executed the placeholder and finished successfully in 2-5 seconds, producing zero scientific output files or outcomes.
- No sensitivity outcomes have been observed or analyzed prior to this implementation freeze.

### 2. Implementation & Code Reuse
- This complete implementation reuses the exact simulator, service model, and policy template definitions from the main codebase.
- Specifically, the deterministic state fork (`dcm.fork_from_live_simulator`), the sequential replay step, and the bootstrap confidence interval logic are imported directly from:
  * `scripts/industry_realism_action_opportunity_phase_a_v1.py`
  * `scripts/industry_realism_action_opportunity_phase_b_v2.py`
  * `scripts/industry_realism_causal_headroom_phase_d_v1_execute.py`
  * `scripts/fresh_latency_causal_confirmatory_v1.py`

### 3. Reserve Parameterization & Trajectory Scans
- The experimental reference scheduler `kv_constrained_online` is instantiated with the custom `--reserve` CLI parameter (using `target_kv_utilization` in the constructor).
- A custom `DynamicObserver` is implemented to:
  1. Replay each of the 20 selected windows for the 5 selected regimes (100 total scenarios).
  2. Query the shadow policy portfolio on each state to detect real-time action disagreement.
  3. Clone the exact state when disagreement is detected, forcing a 1-step counterfactual action before returning to the reference policy with the SAME reserve.
  4. Preserve all future arrivals, random seed state, and pressure characteristics.

### 4. Output Schema and Verification Gates
- Output directories: `results/reference_reserve_sensitivity_v1/reserve_{082,090,100}/`
- Key metrics recorded: total states, disagreement states, P(D), P(B_LAT|D), state-weighted mean H_LAT, bootstrap-clustered 95% CI.
- **Baseline reproduction gate**: A strict check is executed for reserve=0.82 to verify that it replicates the original fresh latency confirmatory results (720 disagreement states, 590 beneficial states, mean headroom ≈ 0.00199579s). The script exits with a non-zero code if it fails to pass this gate.
