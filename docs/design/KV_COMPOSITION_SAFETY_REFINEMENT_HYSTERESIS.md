# KV Composition Safety Refinement — Transition-Aware Hysteresis

## I. Diagnosis of the 6 Held-Out Unsafe Cases
During the first KV composition falsification run (`experiments/kv_composition_falsification_v1_20260817T172446Z/`), the child policy (`KVAdaptiveReserveChildPolicy`) exceeded the maximum parent peak KV utilization (`max(peak_KV(parent_A), peak_KV(parent_B))`) in exactly 6 out of the 36 held-out (TEST + OOD) scenarios:

1. `kvp2.bulk10.phaselate.tighttight.s20260913` (TEST) — Peak: 1.132 vs Max Parent: 1.117 (Overshoot: 0.015)
2. `kvp2.bulk10.phasemiddle.tighttight.s20260913` (TEST) — Peak: 1.124 vs Max Parent: 1.115 (Overshoot: 0.009)
3. `kvp2.bulk24.phaseearly.tightloose.s20260913` (TEST) — Peak: 1.161 vs Max Parent: 1.136 (Overshoot: 0.025)
4. `kvp2.bulk24.phasemiddle.tighttight.s20260913` (TEST) — Peak: 1.198 vs Max Parent: 1.166 (Overshoot: 0.032)
5. `kvp2.bulk10.phaselate.tightloose.s20260914` (OOD) — Peak: 1.040 vs Max Parent: 1.012 (Overshoot: 0.028)
6. `kvp2.bulk10.phaselate.tighttight.s20260914` (OOD) — Peak: 1.064 vs Max Parent: 1.024 (Overshoot: 0.040)

These overshoots occurred during decode-phase steps when no new admissions were being actively authorized, and when the child was in `reserve` mode (for some cases, already for >20 steps).

---

## II. History-of-Composition Mechanism
The overshoots are not a localized, single-step admission bug. Rather, they are caused by the **history of composition** over multiple conversational steps:
1. **Aggressive LLF State Pre-fill:** When in `llf` mode, the child policy acts like `LeastLaxityFirstPolicy` and admits requests greedily, filling up to 100% of nominal physical capacity (`max_kv_tokens`) without keeping any target reserve (unlike `KVConstrainedOnlinePolicy` which caps non-urgent requests at `target_kv_utilization = 0.82` to leave an 18% cushion).
2. **Urgent Arrival and Switch to Reserve:** When urgent requests arrive, the trigger `n_urgent >= tau_urgent` switches the child's mode to `reserve`.
3. **Reserve Mode Admission on Top of High Occupancy:** `reserve` mode (which acts as `KVConstrainedOnlinePolicy`) evaluates the urgent requests. Because they are urgent, the 0.82 target cap is bypassed. The requests are admitted immediately, subject only to the 1.0 strict physical capacity cap.
4. **Combined Footprint Expansion:** Because the system was already at $\approx 96\%\text{--}99\%$ capacity due to prior greedy LLF admissions of non-urgent requests, admitting these urgent requests on top of them creates an exceptionally high initial KV token footprint.
5. **Decentralized Peak Exceedance:** During subsequent decode steps, as both the newly-admitted urgent requests and the pre-existing LLF-admitted non-urgent requests generate output tokens, their combined token generation pushes the peak KV utilization to $\approx 1.12\text{--}1.20$. This exceeds the highest peak that either pure parent policy alone would ever reach on the same scenario trajectory.

Thus, the child's ANWG advantage (+0.036 on the overshoot subset) is direct downstream of using the 18% reserve buffer left idle by the conservative parent, but naive dynamic switching introduces a safety risk when switching back and forth under elevated KV pressure.

---

## III. Why Post-Admission Caps Were Rejected
We evaluated post-admission caps of the form `current_kv + projected_tokens <= safety_cap` where `projected_tokens = req.prompt_tokens + 0.25 * req.predicted_output_tokens` or `req.prompt_tokens + req.predicted_output_tokens`.
* **Head-of-Line (HOL) Blocking:** Because the underlying parent policies do not know that their admission decisions will be filtered/discarded post-hoc, they will continually prioritize and attempt to admit the same blocked top-of-queue request step after step.
* **Severe Starvation:** This results in complete scheduler stalls, dropping the child's ANWG to $0.46\text{--}0.55$, representing a catastrophic performance regression.

---

## IV. Why Dynamic-Capacity Experiments Were Rejected
We evaluated dynamically shrinking `gpu.max_kv_tokens` (e.g., to 95% or 98% of physical capacity) passed to the parent policies.
* **Scheduler Stalls & Suboptimality:** While this successfully avoids HOL blocking (because the parents natively skip requests that exceed the scaled-down limit), it either:
  1. Fails to eliminate the overshoot (non-linear decode expansion still pushes KV past the parent peak); or
  2. Artificially starves the child, completely collapsing the ANWG advantage below the parents' baseline level (dropping ANWG from $0.88$ to $0.78\text{--}0.81$ and failing G2/G4).

---

## V. Single Refinement Chosen: Transition Hysteresis
To prevent the child from switching dynamically into unsafe high-KV pressure states, we implement **Transition Hysteresis** based entirely on online-observable states.

* **LLF → reserve:**
  `mode = "reserve"` if `n_urgent >= tau_urgent` **AND** current KV occupancy is within a safe-transition threshold (`current_kv_utilization <= 0.90`).
  * If the LLF-created occupancy is already exceptionally high ($>90\%$), we *refuse* to transition to `reserve` mode. This prevents us from admitting urgent requests on top of an already bloated non-urgent buffer, which would trigger the overshoot during subsequent decodes.
  
* **reserve → LLF:**
  `mode = "llf"` only if `n_urgent < tau_urgent` **AND** `current_kv_utilization <= 0.82` (the release threshold, corresponding to the target utilization of the KV parent).
  * If the occupancy is still elevated ($>82\%$), we stay in `reserve` mode to ensure the buffer is drained before returning to the greedy LLF-like behavior. This prevents high-frequency mode thrashing under KV pressure.

This single, parameter-free invariant utilizes parent semantics ($0.82$ target utilization from the KV policy and the $0.90$ conservative safety cushion) to bound transition-induced overshoots while preserving HOL-free scheduling.
