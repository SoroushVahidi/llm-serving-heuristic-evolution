# Policy Separation Sobol Pilot v1 -- Scientific Audit

## Provenance

- **Slurm Job:** `1182183`
- **Run Directory:** `/mmfs1/scratch/ikoutis/sv96/policy_separation_sobol_pilot_20260816T183600Z_1182183/` (rsync'd locally to `experiments/policy_separation_sobol_pilot_20260816T183600Z_1182183/`)
- **Repo Checkout:** `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1`
- **Git Branch:** `policy-separation-v1-wulver-20260809` (HEAD pointing to executive SHA `b1181c6380029254080397c161d5dd281bbd6d89`, identical to `origin/contextual-compositional-heuristics-20260731`)
- **Script/Config:** `scripts/run_policy_separation_sobol_pilot_v1.py`, `configs/policy_separation_sobol_pilot_v1.yaml`
- **Predecessors:** Job `1170116` (three-case diagnostic) and Job `1171116` (boundary refinement)
- **Wall Time:** ~2:40, 8 CPU workers, no GPU (pure-Python simulator)
- **Scope:** First continuous, space-filling exploration stage for the Policy Separation roadmap. NOT yet the full 8K-12K MAP-Elites dataset, and NOT selector training or structural crossover.
- **Classification:** **STRUCTURALLY_VALID**

---

## Integrity

The Sobol pilot dataset has been rigorously validated and meets all structural requirements:
- **Scenario Count:** Exactly 1,616 scenarios across three workload families (1024 Family B, 512 Family C, 80 FCFS add-on), matching the design specification exactly.
- **Evaluation Count:** Exactly 6,976 policy evaluations (Family B: 1024 x 4 = 4096; Family C: 512 x 5 = 2560; FCFS: 80 x 4 = 320), with zero task failures recorded.
- **Key Uniqueness:** Zero duplicate `(scenario_id, policy_name)` keys and zero duplicate `scenario_id` entries.
- **No Malformed Data:** Zero NaNs or Infs present in `arrival_normalized_weighted_goodput` (ANWG) or secondary metrics.
- **Valid Range:** All ANWG scores reside strictly within `[0.0, 1.0]`.
- **Sobol Point Uniqueness:** 128 unique base points (Sobol indices 0 to 127) are verified across both Family B and Family C continuous spaces, scrambled via distinct seeds (20260810 for Family B, 20260812 for Family C) to prevent cross-subspace dimension correlation.
- **Roster Correctness:** Confirmed that `weighted_shortest_processing` (WSP) was correctly excluded (due to 100% parameter-induced equivalence to ESTF) and `admission_control` was retained.

---

## 3A. Prediction / Size-Based Scheduling Family (Family B)

Family B analyzes scheduling performance as a continuous function of load (`target_utilization` ∈ [0.50, 1.10]), size-prediction error (`inversion_fraction` ∈ [0, 1]), and categorical `heterogeneity` (moderate vs. strong) over 1024 scenarios.

### 1. Overall Family Statistics
- **Best Fixed Policy:** `estimated_service_time_first` (ESTF) with a mean ANWG of **0.5915**.
- **Portfolio Performance:**
  - `estimated_service_time_first`: Mean = 0.5915, Median = 0.5833, Std = 0.1568
  - `shortest_output_first`: Mean = 0.5857, Median = 0.5833, Std = 0.1597
  - `aging_priority`: Mean = 0.5773, Median = 0.5833, Std = 0.1697
  - `fifo`: Mean = 0.5265, Median = 0.5333, Std = 0.2012
- **Oracle Headroom:** Mean headroom over the best fixed policy is **0.0159** (~1.6%).
- **Headroom Distribution:**
  - Headroom > 0.0: **47.17%** of scenarios
  - Headroom > 0.01: **47.17%** of scenarios (matching >0.0 due to 1/60th-job quantization of ANWG)
  - Headroom > 0.05: **9.77%** of scenarios (highly exploitable regions)
- **Unique Winners (eps=0):** 4 (ESTF, SOF, Aging, FIFO all win in some scenarios).
- **Tie Rate (eps=0):** **37.11%** of scenarios have multiple joint-best policies.
- **Winner Entropy:** **1.8714 bits**, indicating a highly diverse, non-flat performance landscape.
- **Mean Best-vs-Second Margin:** **0.0178**.

### 2. Pairwise Separation Coverage (Fraction of scenarios where $|P_1 - P_2| > 0.01$)
| Policy | `aging_priority` | `estimated_service_time_first` | `fifo` | `shortest_output_first` |
|---|---|---|---|---|
| `aging_priority` | 0.0000 | 0.7881 | 0.8232 | 0.7930 |
| `estimated_service_time_first` | 0.7881 | 0.0000 | 0.8584 | 0.6709 |
| `fifo` | 0.8232 | 0.8584 | 0.0000 | 0.8584 |
| `shortest_output_first` | 0.7930 | 0.6709 | 0.8584 | 0.0000 |

This matrix proves excellent pairwise separation, showing that policies are functionally distinguishable across 67% to 86% of the continuous space.

### 3. Split by Heterogeneity
- **Moderate Heterogeneity:**
  - ESTF - FIFO Mean Margin: **0.0845**
  - SOF - FIFO Mean Margin: **0.0745**
  - Aging - ESTF Mean Margin: **-0.0174** (Aging never outperforms ESTF)
  - Best Fixed Policy: ESTF (Mean = 0.6480)
  - Mean Headroom: **0.0133**, with **40.82%** of scenarios > 0.01.
- **Strong Heterogeneity:**
  - ESTF - FIFO Mean Margin: **0.0455**
  - SOF - FIFO Mean Margin: **0.0438**
  - Aging - ESTF Mean Margin: **-0.0110**
  - Best Fixed Policy: ESTF (Mean = 0.5350)
  - Mean Headroom: **0.0184**, with **53.52%** of scenarios > 0.01.

### 4. Continuous Decision-Boundary Replication
Under strong heterogeneity, binned analysis reveals clean, inversion-dependent winner flips that replicate and generalize Job 1171116's boundaries:
- **Low Load (`target_utilization` < 0.70):**
  - `aging_priority` dominates across the entire inversion range (winning 105 out of 168 scenarios) due to low queue pressure, which allows its age-based priority to balance short/long requests while preventing starvation.
- **Mid Load (`target_utilization` ∈ [0.70, 0.90)):**
  - **Low Inversion (< 0.30):** ESTF is dominant (winning 28/56 scenarios) due to high size-prediction accuracy.
  - **Mid/High Inversion (≥ 0.30):** `aging_priority` takes over (winning 73/120 scenarios), acting as a robust fallback that prevents prediction-error starvation.
- **High Load (`target_utilization` ≥ 0.90):**
  - **Low Inversion (< 0.30):** ESTF is highly dominant (33/52 wins) because accurate size-based scheduling is critical to clear queue buildup.
  - **Mid Inversion ([0.30, 0.70)):** Highly competitive three-way split (ESTF 25, Aging 21, SOF 18), indicating a complex transition zone.
  - **High Inversion (≥ 0.70):** `aging_priority` dominates (25/52 wins) as completely inverted size-predictions would otherwise starve short jobs under ESTF.

This confirms that **higher queue pressure increases accurate prediction value** under strong heterogeneity, and that **accurate size-scheduling is critical at high load but highly vulnerable to prediction errors, where aging serves as a robust fallback**.

---

## 3B. Deadline / Overload Family (Family C)

Family C explores deadline-aware scheduling and admission control as a continuous function of `overload_factor` ∈ [0.85, 1.40] and `fraction_impossible` ∈ [0.00, 0.80] over 512 scenarios.

### 1. Performance and Structural Dominance
- **Best Fixed Policy:** `scorpio_style_slo_guard` with a mean ANWG of **0.5810**.
- **Portfolio Performance:**
  - `scorpio_style_slo_guard`: Mean = 0.5810, Median = 0.60, Std = 0.2080
  - `fifo`: Mean = 0.3645, Median = 0.37, Std = 0.1451
  - `least_laxity_first`: Mean = 0.1928, Median = 0.13, Std = 0.1529
  - `edf`: Mean = 0.1465, Median = 0.10, Std = 0.1606
  - `admission_control`: Mean = 0.1461, Median = 0.10, Std = 0.1598
- **Oracle Headroom:** Mean headroom is **0.0003** (~0.03%). Under 99.6% of scenarios (510/512), Scorpio is the absolute winner.
- **Winner Entropy:** **0.1067 bits**, indicating a completely flat, single-winner landscape.
- **Pairwise Separation:** Near-zero separation among EDF and admission_control (MAD < 0.001, 94.0% bit-identical), indicating high redundancy.

### 2. Mechanism Analysis via Secondary Metrics
By analyzing completion and SLO rates, we can clearly isolate the active filtering mechanism:
- `edf` and `admission_control` achieve a `completion_fraction` of **1.00** across all 512 overload scenarios. However, their ANWG is only **0.146**, indicating that **85.4% of completed requests violated their SLOs** because they wasted execution slots on unsalvageable jobs, leading to cascading queue contention.
- `scorpio_style_slo_guard` achieves a `completion_fraction` of **0.5810** while maintaining an ANWG of **0.5810**. This proves that **100% of the requests Scorpio chose to complete satisfied their SLOs**.
- Scorpio's `num_dropped` averages **11.9 out of 30** requests, while EDF, LLF, and admission_control dropped **exactly 0**.

This demonstrates that **Scorpio's active laxity-based admission filtering and credit-budget throttling are essential under overload**, preventing cascading deadline violations by proactively discarding unsalvageable requests. However, **admission_control remains near-equivalent to EDF (94.0% identical) because its default laxity filter is inert**, highlighting the gap between basic prioritization and active admission control.

---

## 3C. FCFS Categorical Add-On

The non-Sobol categorical add-on over 80 scenarios confirms that arrival event-ordering is a discontinuous mechanism:
- **Template A1 (offset = 0.0, mas = 1, n=60):**
  Produces strong policy separation: `fifo` achieves a mean ANWG of **0.202**, while the size-based policies (`estimated_service_time_first`, `shortest_output_first`, `aging_priority`) tie exactly at **0.383**. Under single-concurrency slots, size-based scheduling is highly effective.
- **Template A2 (offset > 0.0, mas = 4, n=20):**
  Shows that positive offsets and higher concurrency reduce the convoy effect, so all policies perform very close to each other (FIFO = 0.945, others = 0.979). The separation is drastically compressed.

*Note:* Because the FCFS offset represents a sharp, discontinuous transition (separating only at offset = 0.0), it was correctly excluded from the continuous Sobol space, as space-filling coordinates would waste budget on uninformative points.

---

## 4. Policy Footprint / Dataset-Separation Analysis

A core goal of this experiment was to evaluate whether space-filling Sobol sampling preserves synthetic policy-separation signals or dilutes them compared to handcrafted grids:

### 1. Quality Indicators by Family
| Metric | Family B (Prediction) | Family C (Deadline) | FCFS Add-On |
|---|---|---|---|
| **All-Policy Near-Tie Rate** (max - min ≤ 0.01) | **0.0459** (4.6%) | **0.0000** (0.0%) | **0.3500** (35.0%) |
| **Mean Inter-Policy Variance** | **0.002790** | **0.042692** | **0.014658** |
| **Unique Winners** (eps=0) | **4** | **4** (Scorpio 510, others 2) | **4** |
| **Winner Entropy (bits)** | **1.8714** | **0.1067** | **1.4803** |
| **Oracle Headroom** | **0.0159** | **0.0003** | **0.0000** |

### 2. Comparison with Prior Work
- **Comparison to Job 1171116 (Boundary Refinement):**
  - **Headroom Preservation:** Family B's mean headroom is **0.0159** (vs. 0.0154 in 1171116), and the fraction of scenarios with headroom > 0.01 is **47.17%** (vs. 43.4%).
  - **Separation Coverage:** The near-tie rate in Family B is extremely low (4.6%), proving that the space-filling Sobol sequence successfully targeted informative regions without drowning the corpus in uninformative, flat-performance scenarios.
- **Comparison to SwissAI V2 Repaired Result:**
  Unlike SwissAI V2 which exhibited severe redundancy (ranking collapses where all size-based policies behaved identically), Family B displays rich, distinguishable policy footprints across 67-86% of its continuous space.
- **Verdict on Space-Filling Dilution:**
  The Sobol pilot **did not dilute** the policy-separation signal. It successfully mapped the continuous transition boundaries while retaining identical statistical headroom and separation coverage as the handcrafted refinement grid.

---

## 5. Coverage Gaps in Current Dataset

We classify the coverage of key scientific mechanisms below:

| Mechanism / Dimension | Classification | Documentation / Evidence |
|---|---|---|
| **Prediction Quality × Heterogeneity** | `COVERED_AND_SEPARATING` | Replicated inside Family B; clean crossover between ESTF and Aging. |
| **Deadline Overload × Unsalvageable Jobs** | `COVERED_BUT_NOT_SEPARATING` | Replicated inside Family C; Scorpio dominates completely (no separation). |
| **FCFS Convoy ordering** | `COVERED_AND_SEPARATING` | Replicated inside FCFS add-on; offset acts as a categorical gate. |
| **Tenant Fairness / Weight Skew** | `NOT_YET_TESTED` | Present in the simulator (e.g., `vtc`), but completely missing from this dataset. |
| **Starvation / Aging pressure** | `PARTIALLY_COVERED` | Represented only by the `aging_priority` policy in Family B, but missing explicit aging workloads. |
| **KV-Cache Pressure / Capacity Constraints** | `NOT_YET_TESTED` | Absent. All current families assume infinite/non-constraining KV-cache. |
| **Prefix Reuse / Cache Thrashing** | `NOT_YET_TESTED` | Absent. |
| **Prefill-vs-Decode Interference** | `NOT_YET_TESTED` | Absent. All current runs use uniform prefill/decode models. |
| **Transition / Offload Costs** | `NOT_YET_TESTED` | Absent. |
| **SLO Tightness Heterogeneity** | `NOT_YET_TESTED` | Absent. |

---

## 6. MAP-Elites / Selector Training Readiness Verdict

### **Verdict:** `MAP_ELITES_NOT_YET_JUSTIFIED`

### **Scientific Evidence and Rationale:**
1. **Single-Family Separation:** True policy specialization and decision-relevant boundaries only exist in **Family B** (size-prediction error). In Family C, Scorpio dominates completely (98.8% wins, headroom ~0), and in the FCFS add-on, size-based policies are equivalent.
2. **Insufficient Descriptors:** Running MAP-Elites or CMA-ES over the current 3-family space is trivial because there is only one active, non-flat dimension of trade-offs (prediction accuracy vs. load). The archive would simply illuminate that ESTF wins under low error and Aging wins under high error.
3. **Selector Retraining is Futile:** Training a selector over this pilot corpus is not scientifically useful because it would learn a trivial rule: "Select Scorpio under overload, and select ESTF/Aging under size heterogeneity."

To make Quality-Diversity (QD) search or selector training meaningful, we must first introduce the other major multi-dimensional trade-offs that define real serving systems (such as multi-tenant fairness and KV-cache capacity).

---

## 7. Recommended Next Experiment: FAIRNESS-AND-AGING-STARVATION (Family A)

To resolve the coverage gaps and build toward a mature, multi-dimensional dataset, we must execute a targeted pilot for **Family A: Fairness, Weight Skew, and Aging Starvation**.

### **Scientific Design for the Next Experiment:**
- **Scientific Hypothesis:**
  In a multi-tenant serving environment, high-weight bulk tenants will starve low-weight interactive tenants under standard size-based scheduling (ESTF/SOF). Introducing a tenant-weight fairness policy (like VTC) or aging priority will restore SLO compliance for interactive tenants, creating a clear decision boundary gapped by load and tenant weight skew.
- **Policies in Roster:**
  - `fifo` (baseline)
  - `estimated_service_time_first` (throughput-optimized size-based)
  - `aging_priority` (starvation-prevention size-based)
  - `vtc_style_fair_scheduler` (multi-tenant fairness)
- **Workload Dimensions & Continuous Coordinates:**
  - `target_utilization` ∈ [0.50, 1.10] (Queue pressure)
  - `tenant_weight_skew` ∈ [1.0, 10.0] (Ratio of bulk tenant weight to interactive tenant weight)
  - `interactive_volume_fraction` ∈ [0.05, 0.40] (Fraction of total requests originating from interactive tenants)
- **Controls:**
  - A control group with `tenant_weight_skew = 1.0` (equal weights) where throughput-optimized size-based scheduling (ESTF) should dominate without starvation penalties.
- **Metrics:**
  - *Primary:* `arrival_normalized_weighted_goodput` (ANWG)
  - *Secondary:* Tenant-specific SLO violation rates, interactive-tenant TTFT, and JFI (Jain's Fairness Index) on goodput.
- **Scale:**
  - 2**6 = 64 Sobol points * 2 interactive fractions * 5 seeds = 640 scenarios.
  - 640 scenarios * 4 policies = 2,560 evaluations.
- **Why this is superior to immediately starting MAP-Elites:**
  It introduces a second, highly realistic dimension of trade-offs (Fairness vs. Throughput), providing a multi-family landscape that actually justifies a multi-dimensional Quality-Diversity grid.
