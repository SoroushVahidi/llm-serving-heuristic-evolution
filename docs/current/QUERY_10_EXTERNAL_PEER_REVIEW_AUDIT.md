# QUERY_10 — External peer-review and whole-paper audit (diagnosis only)

No manuscript, test, manifest, or release file was modified by this query. The target is the *working-tree* manuscript
including the uncommitted QUERY_9 edits. The pasted query was truncated at its Section 10 ("10. CROSS-"), so that
section was not seen; Sections 0–9 are covered.

## 0. Preserved target state

| Item | Value |
|---|---|
| Branch | `release/peva-v1-20260920` |
| HEAD | `77cdde98815fe14ae9fc49730640cda1a11cfc78` |
| `git status` | M `FINAL_CLAIM_MANIFEST.json`, M `main.bbl`, M `main.tex`, M `scripts/build_claim_manifest.py`, M `tests/test_manuscript_discussion_conclusion.py`, M `tests/test_manuscript_robustness_numbers.py`; ?? `docs/current/QUERY_9_MANUSCRIPT_COHERENCE_AUDIT.md` (this file is also new, uncommitted) |
| `git diff --stat` | 6 files, +331 / −296 (main.tex 300 lines changed) |
| SHA-256 `paper/performance_evaluation/main.tex` | `8d1661c491df4c615ff8b889f7ce13cb150ee1908d95d213f5a120fa38b3e40e` |
| SHA-256 `paper/performance_evaluation/main.pdf` (git-ignored, untracked) | `25defb21c984368730e166117e5f83a8bc7df662fb73546275a6f9fe0f799d08` |
| SHA-256 `main.bbl` | `1aeddba89db60a0ab67fdf58676b4dc13f6c21e85905ce9c79f268293d135cd3` |
| SHA-256 `FINAL_CLAIM_MANIFEST.json` | `92ba971289ada183c0dc4977544806969cc0fbe233cba4327533106f9735dcbe` |
| SHA-256 tracked (stale) `paper/when_does_llm_serving_scheduler_adaptation_matter.pdf` | `985e085de8ffce5301bef947c9921a8b23d488d9938b81803c4d75324b4d9d27` |

`main.pdf` has an mtime 9 s *older* than `main.tex` because a `git stash`/`stash pop` in QUERY_9 rewrote `main.tex`
with identical content. I verified equivalence by rebuilding `main.tex` in a scratch directory: `pdftotext` output is
byte-identical to `main.pdf`. So `main.pdf` renders the current text.

## 1. Evidence hierarchy and conflicts found

Hierarchy used: (1) `PREREGISTRATION_V1.json` and machine-readable primary artifacts (`FRESH_*_V1.csv/json`, Phase A/B
CSVs); (2) simulator/analysis code (`scripts/*_v1.py`, `src/llmserveopt/policies/*`); (3) generated Markdown reports;
(4) manuscript prose. I did not find a repository document that defines a different hierarchy (not exhaustively
searched). Every number below was recomputed from (1)/(2), not read from reports.

| # | Conflict | Canonical recomputation | Status in current manuscript |
|---|---|---|---|
| C1 | Prose implied fresh support transferred to BurstGPT | 360 BurstGPT fresh conditions, all VALID (357 unconstrained + 3 constrained), 20 windows, **0 disagreement states**, 0 active-cap binding, 0 KV binding, max queue length 3 | Now stated correctly (abstract, intro, C2, Sec. 6.1, Sec. 9.3, Limitations). The *cause* is not stated (see M5) |
| C2 | Report §4: "BurstGPT `kv_8000` invalid/truncated" | All 21 INVALID_HORIZON_TRUNCATED conditions are Azure `kv_capacity=8000`: 20 Azure code, 1 Azure conversation. BurstGPT has 0 invalid | Manuscript silent; report still wrong |
| C3 | Reproducibility guide lists BurstGPT upstream as `github.com/SoroushVahidi/BurstGPT` | Dataset owner repo is `github.com/HPMLL/BurstGPT` (CC-BY-4.0, KDD'25) | **Uncorrected** in `PERFORMANCE_EVALUATION_REPRODUCIBILITY.md` |
| C4 | Preregistered central question says "*realistic* resource pressure" | Caps are synthetic one-axis interventions | Manuscript says "controlled" — consistent with the data; the prereg wording is the outlier |
| C5 | Stale derived copies (`submission/`, `release/peva_submission_final/`) | Still contain pre-QUERY_9 text | Known, deliberately not regenerated |

## 2. First-pass reading

**Question.** Does a serving system reach states where another executable action exists, and is taking it worth anything?
**Contribution.** A staged measurement protocol (pressure → disagreement → one-step counterfactual benefit → magnitude
→ deployability) plus an in-simulator empirical result.
**Findings.** (i) Under the simulated abundant-resource baseline no alternative exists; (ii) capacity caps create
alternatives; (iii) forcing one lowers latency in 81.9% of 720 Azure states; (iv) the magnitude is concentrated (one
window, 88.4%).
**Reviewer view.** New: the canonical-action-level prevalence measure and the concentration/clustering analysis.
Implementation/evaluation work: the simulator, portfolio, vLLM probe.
**Reads as one paper?** Mostly yes after QUERY_9. Residual report flavour: the "faithful corpus / fresh / frozen /
regime" vocabulary is only intelligible to someone who followed the project.

## 3. Novelty and contribution

**Classification.** Primary: *new evaluation methodology* (staged decomposition; one-step forced-action counterfactual
with default continuation; window-clustered inference with concentration diagnostics). Secondary: *new empirical
finding*, valid only for this simulator and portfolio. Not an algorithm, system, or dataset contribution.

| Element | Genuinely new | Existing idea, new context | Established technique | Engineering |
|---|---|---|---|---|
| SBS/VBS reference points | | ✔ (Rice 1976; SATzilla; AutoFolio) | | |
| Canonicalized action disagreement `D(s)` over all decision states | ✔ | | | |
| One-step forced-action counterfactual, default continuation | | ✔ (rollout/one-step lookahead, hindsight-optimal benchmarks — cf. Jaillet et al.) | | |
| Cluster bootstrap by trace window | | | ✔ | |
| Concentration / leave-one-out reporting | | | ✔ | |
| Deterministic replay simulator, claim manifest, release scripts | | | | ✔ |

**Is the novelty clear?** Partly. The paper never shows *on its own data* that the staged view changes a conclusion
relative to the whole-policy VBS-gap view it criticizes: Sec. 9.2 says "a VBS gap … does not show that the live system
reaches states with a different executable action", but no VBS gap is reported. A single comparison of window-level
policy complementarity against `P(D)`, `P(B|D)`, `H` would make the methodological claim empirical. This may be
computable from existing artifacts; I did not verify.

**The hard reviewer question** ("why is this more than a sensible checklist?"). What makes the answer convincing: the
canonicalization (policy-score differences collapse to identical actions in 194/22/462 native states), the
denominator choice, and the clustered inference exposing that one window carries 88%. What makes it weak: the
decomposition itself is intuitive; the demonstration is on one simulator, one in-repo portfolio, and Azure only; and
the deployability stage is not measured.

**Consistency across sections.** Abstract, intro and Contribution 1–4 describe a *measurement* contribution at
consistent strength. Broader than the evidence: Conclusion ¶3 "scheduler adaptation should be justified stage by
stage" and Sec. 9.5 practices generalize from one simulator/portfolio; and "strong default" (Sec. 2, line 218) — see M3.

## 4. Baselines and comparisons

The portfolio `P6_POLICIES` (`scripts/industry_realism_action_opportunity_phase_a_v1.py:40`) and SBS
(`SBS_POLICY = "kv_constrained_online"`):

| Policy | What it is (code) | Published-system faithful? | Cited in manuscript? |
|---|---|---|---|
| `full_prefill` | in-repo heuristic | No | No |
| `chunked_prefill_small` | in-repo chunked-prefill variant | Sarathi-like idea, not a Sarathi-Serve reimplementation | No source given |
| `estimated_service_time_first` | in-repo SJF-style on predicted tokens | No | No |
| `weighted_fair_share` | in-repo | No | No |
| `least_laxity_first` | in-repo, uses `slo_deadline` | No | No |
| `kv_constrained_online` (**SBS**) | laxity/KV-cost ordering; admits only if post-admission KV utilization ≤ 0.82 unless laxity ≤ 0.25 s | No | No |

The manuscript gives names but no definitions, parameters, tuning, or citations for any of them. Faithful
reimplementations of Llumnix, DistServe, Apt-Serve and an Orca-style policy exist in `src/llmserveopt/policies/`
(`llumnix_faithful.py`, `distserve_faithful.py`, `apt_serve_faithful.py`, `orca_style.py`) but are **not** in the
portfolio, and the paper does not say why.

**Deadline/priority overlays are inert.** In the faithful view `slo_deadline = arrival + 1000.0` and `priority =
uniform 1.0` (Phase A script lines 476–477). So LLF and the SBS "urgent" branch cannot discriminate, WFS has no
weights to use, and goodput saturates at 1.0 by construction. The manuscript says overlays are "deterministic
simulator inputs" but never states these values. (My QUERY_9 rewrite also dropped the word "loose" that the original
had before "deadline"; it should be restored and the 1000 s stated.)

**Is the portfolio strong/recent/modern?** No: it is six simple heuristics with a hand-set KV reserve. That limits the
generality of "adaptation opportunity" to *this* portfolio, which the manuscript concedes ("Policy portfolio"
limitation) but without disclosing the inert overlays. Because the paper asks about action opportunity, not scheduler
superiority, implementing modern adaptive systems is *not* required; **but** the paper must justify the portfolio and
state that the result bounds the opportunity between these six rules only.

**Candidate absent systems**

| Candidate | Classification | Why |
|---|---|---|
| Sarathi-Serve (cited) | SHOULD DISCUSS as the origin of `chunked_prefill_small`; not experimentally required | Portfolio member is a chunked-prefill variant |
| Llumnix, SOLA, LMetric, Preble (cross-instance routing/migration) | NOT ACTUALLY COMPARABLE | They act above the single-engine step decision the paper measures |
| Faithful Llumnix/DistServe/Apt-Serve in repo | SHOULD DISCUSS why excluded | Already implemented; a reviewer will ask |
| Jaillet et al. KV-constrained online scheduling | SHOULD DISCUSS (missing ref #1) | Theory for exactly the KV-cap axis and a hindsight-optimal benchmark |
| A tuned-threshold SBS (KV reserve swept) | MUST COMPARE if the paper keeps "strong default" | See M3 |

## 5. Related work audit

Claims about other papers, checked against the source:

| Sentence / cell | Source opened | Verdict |
|---|---|---|
| Nixon et al.: year of production traffic; workload evolution, caching, load balancing | arXiv 2608.13573 abstract | **Verified** (one-year trace from "CompanyX"; caching/LB in title) |
| MorphServe: changes precision and KV capacity under pressure | arXiv 2506.02006 abstract | **Verified**. Bib defect: title now "…Runtime *Quantized* Layer Swapping…"; authors now 7 (adds Zeyu Zhang, Haiying Shen) |
| Strata: hierarchical context caching | arXiv 2508.18572 abstract | **Verified** |
| LMetric: KV-aware × load indicator; failure conditions derived; production-style workloads | arXiv 2603.15202 full text | **Verified** (Sec. 5.2 derives failure conditions; real H20 GPUs; Alibaba/Kimi traces; deployed at Bailian) |
| Libra: flexible request partitioning | USENIX NSDI'26 page | **Verified**; bib lacks pages 1243–1258 |
| ServeGen: characterizes/generates production workloads | USENIX page/arXiv 2505.09999 | **Verified**. Bib defect: 6 authors listed, paper has 8 (missing Yan Zhang, Jingren Zhou) |
| FastServe: preemptive scheduling | USENIX NSDI'26 page | **Verified** |
| SOLA: state-aware scheduling for SLO attainment | MLSys'25 page | **Verified** |
| OpenTela: unifies decentralized compute for heterogeneous LLM serving | search snippets only | Title-level **verified**. Table 1 cell "*Operational traces*" is **unverified** — use title-level wording |
| Lilou (PEVA 171:102539) | ScienceDirect PII S0166531625000732 | **Verified** exists in PEVA; latency prediction for GPU model serving (DNNs, MAPE ≈ 8.7%) |
| Ali et al. serverless ML training (PEVA 167:102451) | ScienceDirect PII S0166531624000567 | **Verified** exists in PEVA |
| Yildiz et al. dispatching (PEVA 172:102551) | DOI resolves to Elsevier PII S0166531626000118 | DOI resolves to a PEVA article; **title/authors not retrieved** — unverified |
| Choudhury et al. job assignment (PEVA 167:102463) | search returned nothing | **Unverified** by me |
| Orca, vLLM, Sarathi-Serve, DistServe, Splitwise, Mooncake, DynamoLLM | not re-opened | Not verified in this query (standard venues; bibliographic fields plausible) |

**Analysis vs listing.** §2 ¶1–2 is mostly a list of one-clause descriptions; it explains the *difference* only once
("The question here precedes them"). Table 1 has six rows but two of the closest works (LMetric, Llumnix/SOLA) are
described as "mechanism whose decision states this measurement could examine" — informative. **Closest prior work is
not identified**: the closest methodological neighbours (hindsight-optimal benchmarks for KV-constrained scheduling;
simulator-based what-if tools such as Vidur) are absent. **Citation clustering:** no `\cite` exceeds 4 keys (max 4).
Two blocks of 4 (`sarathi…fastserve`; `rice…autofolio`) do not say which paper supports which claim.

## 6. Important missing references (verified to exist)

1. **Jaillet, Jiang, Mellou, Molinaro, Podimata, Zhou — "Online Scheduling for LLM Inference with KV Cache
   Constraints."** arXiv:2502.07115 (v1 10 Feb 2025; v5 15 Jan 2026), cs.LG. Venue beyond arXiv: not confirmed.
   https://arxiv.org/abs/2502.07115. *Verified content:* online model of KV-constrained LLM inference scheduling; a
   polynomial-time algorithm; a proof that no deterministic online algorithm has constant competitive ratio under
   arbitrary arrivals; an integer program as a *hindsight-optimal benchmark*; synthetic and public-trace simulations
   (Llama2-70B/A100). *Relevance:* theory for the KV-cap axis and a direct precedent for an opportunity bound.
   *Placement:* Related Work + Discussion.
2. **Agrawal, Kedia, Mohan, Panwar, Kwatra, Gulavani, Ramjee, Tumanov — "VIDUR: A Large-Scale Simulation Framework for
   LLM Inference."** MLSys 2024 (PMLR/MLSys vol. 6). https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html.
   *Verified content:* operator profiling + predictive modelling; scheduler/batching what-if; reports <5% error on
   latency/throughput estimates (search summary also cites ≈9% average latency error; I did not reconcile the two).
   *Relevance:* the manuscript's headline effect is measured in an uncalibrated simulator; this is the
   fidelity standard a reviewer will invoke. *Placement:* Related Work + Limitations.
3. **Mitzenmacher, Shahout — "Queueing, Predictions, and Large Language Models: Challenges and Open Problems."**
   *Stochastic Systems* 2025, DOI 10.1287/stsy.2025.0106; arXiv:2503.07545. *Verified content:* position/survey
   linking queueing with predicted service times and LLM scheduling issues (variable inference time, KV memory,
   preemption). *Relevance:* the performance-evaluation community's framing of exactly the scheduling-with-predictions
   problem behind ESTF/KV policies. *Placement:* Related Work + Discussion.
4. **Srivatsa, He, Abhyankar, Li, Zhang — "Preble: Efficient Distributed Prompt Scheduling for LLM Serving."** ICLR
   2025, arXiv:2407.00023. *Verified content:* distributed scheduler co-optimizing KV-state reuse and load balance;
   reports 1.5–14.5× average and 2–10× p99 latency gains on real workloads; is itself a baseline in LMetric.
   *Relevance:* an adaptive scheduler in the same family as Llumnix/LMetric that the paper claims to be "complementary"
   to. *Placement:* Related Work only (not comparable to a single-engine step measurement).

**Performance Evaluation search.** I did not find a recent PEVA-journal paper on *LLM-serving* scheduling; the four
already cited are neighbours (dispatching, DNN latency prediction, serverless training, ML-inference job assignment).
The nearest queueing-theory work I found is 2026 arXiv material (e.g., arXiv:2605.04595 on stability of LLM inference
with KV constraints), not verified beyond its title.

## 7. Datasets and workloads

| Workload | Source | Real vs synthetic | What is real | What is synthetic |
|---|---|---|---|---|
| Azure 2023 code / conversation | `AzurePublicDataset/AzureLLMInferenceDataset2023.md`, CC-BY, collected 11 Nov 2023 | Real *token counts and timestamps only*; no prompt text | arrival timestamps, ContextTokens, GeneratedTokens | SLO deadline (arrival+1000 s), priority (1.0), `predicted_output_tokens` origin (not verified), all resource caps |
| BurstGPT | KDD'25; 10.31 M traces, regional Azure OpenAI GPT services, 213 days, CC-BY-4.0 | Real | timestamps, request/response token counts | same overlays and caps |
| Azure 2024 / Bailian | not referenced in the manuscript | — | — | — |

- **Windows.** Each window is 200 contiguous requests; native and pressure use 20 windows/workload; fresh windows are
  "the first 20 unused full contiguous windows by ascending window_index" after excluding earlier windows
  (`PREREGISTRATION_V1.json`, `fresh_trace_population.selection_rule`) — **trace-order, not random, not
  load-stratified**. The manuscript says "untouched" and "non-overlapping" but not this selection rule.
- **Time scale.** `step_size=0.001`, `step_token_budget=512`, `prefill_cost_per_token=1.0` (Phase A). One step = 1 ms;
  no calibration to hardware appears in the manuscript. The smallest possible headroom ≈ 1 step / request-population.
- **Wording.** Justified: "production-derived *traces*" for inputs (arrival and token lengths from real services).
  Not justified: "production workloads", "production-scale", or anything implying production execution. The
  manuscript mostly complies (I found no "realistic" or "production workload"). Caps are clearly labelled
  "controlled" in Methods/Limitations.
- **BurstGPT (recomputed).** Fresh: 360 conditions, all valid, 0 disagreement, 0 binding, max queue 3, in every axis
  including active cap 4 and KV 8,000 tokens. *Original* native BurstGPT had max queue length 31 and disagreement at
  caps 16/8. The untouched BurstGPT windows simply never queue. Why no causal regime was selected: the mechanical
  rule requires observed disagreement, and there was none (`FRESH_REGIME_SELECTION`).
- **Licensing/availability.** Both datasets CC-BY; derived windows are redistributed in `data/public_trace_corpus_v1/`
  (attribution required). The Azure README asks users to cite Splitwise (already in the bibliography) alongside the
  dataset entry.

## 8. Experimental rigor — findings, ranked

**Major**

- **M1. The native "action-null" result is close to tautological.** Native mean queue length is 0.040 / 0.0087 /
  0.0091 (Azure code / conv / BurstGPT); max KV utilization 0.27% / 0.17% / 0.38%; and `no_policy_choice_states ==
  sbs_decision_states` in every row. Across *all* fresh conditions 99.96% / 99.90% / 100% of decision states are
  no-choice engine iterations. "About one million states with no alternative" counts iterations, most with a single
  candidate. The informative denominator is states with ≥2 candidates.
- **M2. The arrival-scaling null is confounded with default capacity.** At 8× arrival the fresh runs reach max KV
  utilization 0.44% (default capacity 8,000,000 tokens), max 106 active sequences (cap 512), mean queue ≤ 0.09, and zero
  binding states on every workload. No resource binds, so "load ≠ contention; opportunity follows binding" (Sec. 9.3)
  is not identified by this experiment; it only shows that these arrival multipliers never reached the capacity
  limit of the simulated system.
- **M3. The default is untuned and chosen on a different problem, and the headroom is largely the default's own
  heuristic.** SBS was selected on the earlier "joint240" benchmark under utility R = 0.314, not on these windows or
  latency. Of 720 disagreement states, **512 (71%) have all five non-SBS policies differing from SBS and carry 94.8% of
  total headroom**; in the KV-16,000 regime **414 of 524 disagreement states occur without physical KV binding** (mean
  headroom 3.06 ms vs 1.34 ms with binding). Reading: much of the headroom is the SBS's hard-coded 0.82 post-admission
  KV reserve refusing admissions the others make — an SBS-tuning effect — and KV-axis disagreement is partly by
  construction because only the SBS reacts to KV utilization. "Strong default" (Sec. 2) is unsupported.
- **M4. Inert SLO/priority overlays** (Section 4 above) mean three of six policies degenerate.
- **M5. External validity.** Causal evidence: 2 Azure workloads, 5 regimes, 36 windows, effective 1.27 windows by headroom
  mass. Fresh window selection is trace-order. BurstGPT's fresh windows have no bursts, so non-reproduction is most
  plausibly a sampling outcome, not a property of BurstGPT; the manuscript's "onset settings are not workload
  constants" is correct but the mechanism (windows never queue) is not stated.
- **M6. Effect-size framing and selective reporting.** The paper reports absolute ms only. Recomputed relative headroom
  `H/L_ref`: state-weighted ratio of means 2.03%; median per-state 0.21%; Azure-code KV-16,000 (431 states) mean
  5.2%, p90 19.3%; **15.4% of all states offer >10% relative reduction**. "Generally modest" therefore depends on the
  unit, and the unit is an uncalibrated 1 ms step. The preregistered secondary outcomes — relative headroom, p95,
  harm/zero/mixed structure, action-level rates — are absent from the manuscript. From the state-level CSV: **102/720
  (14.2%) states have every alternative harmful**, 26 all-zero, 22 mixed. The harm structure bears directly on the
  deployability stage the paper says it does not measure.

**Moderate**

- **"Preregistered".** A version-controlled protocol in a private repository: design freeze commit `b4e6c60`
  (2026-09-19 22:12 −0400), support map frozen `38e02bc` (23:00), execution `b196c3e` (2026-09-20 00:26) — 2 h 14 min
  from design freeze to execution. Not an external registry. The manuscript never says where the registration lives.
  Wording such as "pre-specified in a time-stamped protocol (commit …)" would be verifiable. The latency endpoint
  was chosen after the earlier Phase-D data were seen; the protocol records this and moved confirmation to fresh
  windows (good practice, and the manuscript discloses it).
- **Statistical wording.** "Statistically supported" (Sec. 6) sits beside a delete-one-window jackknife interval that
  includes zero and a bimodal bootstrap; "met the preregistered decision rule" is the defensible phrase.
- **Ablations.** Present: aggregation/weighting, leave-one-window/regime, numerical guard, interval method,
  objective (goodput vs latency). Missing: SBS choice / KV-reserve sweep (M3), portfolio ablation (which alternative
  policy delivers the headroom), continuation policy (default-only), canonicalization sensitivity, TTFT/TPOT/p95.
- **Realism/challenge.** Native replay: idle system (M1). Pressure: caps are the only thing that binds (M2). Fresh
  support: BurstGPT lacks pressure (M5). Causal: one step, oracle. vLLM probe: different mechanism (token budget), one
  model/GPU, and no KV/preemption metrics — corroborates queueing, not headroom.
- **Simulator fidelity** is not validated against any real system or against Vidur-class simulators.

## 9. Reproducibility

| Result | Public input | Script | Command documented? | Seed/config | Reproducible now? | Gap |
|---|---|---|---|---|---|---|
| Native counts (Sec. 4) | derived windows `data/public_trace_corpus_v1/` | `scripts/industry_realism_action_opportunity_phase_a_v1.py` | Script named, no argument line | in artifacts | **Not re-run** (frozen) | Overlay values, timing model not in paper |
| Pressure map, Table 2, Figs 2–3 | same | `…phase_b_v2.py` | Script named | in artifacts | **Not re-run**; numbers re-derived from CSV by the manifest | — |
| Fresh support map | fresh windows | `fresh_latency_causal_confirmatory_v1.py` | "HPC (Wulver)" | `PREREGISTRATION_V1.json` | Not re-run | HPC cost unstated |
| Table 3, Sec. 6 headline | frozen state CSV | same | Yes | seed 20260920, 2,000 replicates | **Yes — manifest `--check`: 47 claims, 0 problems** | — |
| Robustness Tables 4–5, Sec. 7 | frozen state CSV | `paper/performance_evaluation/scripts/robustness_numbers.py`, `scripts/fresh_causal_robustness_v1.py` | Yes | seeds in outputs | **Yes** (78 manuscript tests pass) | — |
| Figs 1–6 | frozen artifacts | `plot_*` scripts | Yes (guide §1) | deterministic | Artifact-level checked; **not regenerated** | — |
| Table 1 | cited papers | none | N/A | — | Not reproducible; partly verified here | OpenTela cell |
| vLLM probe | model + GPU | probe scripts | Not audited | — | Needs RTX 5060 Ti class GPU | CUDA/driver version not in paper |

**Misleading or stale artifacts:** C2 (report vs CSV), C3 (BurstGPT URL), C5 (stale submission/release copies incl. PDF,
zip, checksums), and the tracked `paper/when_does…pdf` (hash above) predates QUERY_9. The repro guide states BurstGPT is
"2025/2026" without a version.

## 10. Technical clarity and correctness

- **`Q_SBS(s,a)` is defined once (Sec. 3.3) and never used**; the equations use `L_SBS(s)` and `L_CF(s,a)`, and
  `L_SBS`/`L_CF` are never defined as symbols. Replace `Q` with the `L` notation.
- Symbols `D`, `B_LAT`, `H_LAT`, `A_LAT`, `ANWG`, `\bar H`, `\epsilon`, `U`, `p_w` are defined before use. Units: `H_LAT` in
  seconds/ms; `P(D)` in % in tables but as a probability in equations — state once.
- **"Best".** "fixed best single policy" (Sec. 3.2) does not say best *by what, on what data*. Never implies global
  optimality; VBS/"virtual best solver" is used correctly.
- **"Causal".** Exact deterministic counterfactual replay inside a simulator; identification is not the issue, but the
  word can read as a field-level causal claim. "Counterfactual (simulated)" would be more accurate at first use.
- **"Oracle".** Used correctly (ex post best of ≤3 alternatives; 610/720 states have exactly 1).
- **"Robust".** Applied to the *sign*, not the mean (matches the tests).
- **"Production".** Compliant (see Section 7).
- **Eq. 4** (`P(D) × P(B|D) × magnitude`) is flagged in text as conceptual. One unnumbered display
  (`P(B_LAT|D)=590/720`) is a result, not a definition; acceptable.
- **Equation vs implementation.** `H = max(0, max_a A)` matches `PREREGISTRATION_V1.json` `primary_endpoint`
  (`mean_s max(0, max_a(L_SBS − L_CF))`, zeros included); the SBS continuation and one forced action match the
  `intervention` block. I did not re-execute the simulator.

## 11. Ranked reviewer risks (for the author's decision)

1. M3 — the headline headroom is largely the untuned SBS's own KV reserve; "strong default" unsupported.
2. M1/M2 — native and arrival nulls are consequences of an idle, effectively unlimited-capacity simulator.
3. M6 — absolute-ms-only, uncalibrated time scale; unreported preregistered secondaries (harm 14.2%; relative effect up to
   19% p90 in the KV regime). "Modest" is unit-dependent.
4. M4/portfolio — inert overlays; portfolio undefined/uncited in the paper.
5. M5 — Azure-only causal evidence; trace-order window sampling; BurstGPT lacks pressure.
6. Novelty demonstration — no VBS-gap-vs-staged comparison on the same data.
7. Bibliography defects (MorphServe, ServeGen, Libra pages, OpenTela cell, two unverified PEVA items) and C2/C3.

---

## Addendum (QUERY_11, 2026-09-21): corrections to this audit

Two statements in this audit were wrong and one was unverified. The manuscript now uses the corrected values.

1. **M1 evidence was partly circular.** I wrote that "99.5–100% of decision states are no-choice engine iterations".
   In the code (`scripts/industry_realism_action_opportunity_phase_a_v1.py`, `no_choice = n_distinct_canonical_p6_actions == 1`),
   `no_policy_choice_states` counts states where all six policies issue the same canonical action, i.e. exactly the
   complement of disagreement `D`. It is therefore not independent evidence that the engine was idle. The independent
   indicators are mean queue length (native 0.040 / 0.0087 / 0.0091), peak KV utilization (0.17–0.38% native), and
   zero binding states; the manuscript uses only those.
2. **M6 relative-effect numbers were computed from a mislabeled column.** `FRESH_LATENCY_STATE_LEVEL_V1.csv` holds
   counterfactual-branch values in `mean_ref_latency`/`p95_ref_latency` (see `experiments/..._corrected/README.md`). Using the
   corrected SBS-reference latency (cross-checked against `mean_latency + a_lat` in the action-level rows and against the
   preregistered per-regime `mean_relative_headroom` in `FRESH_LATENCY_CAUSAL_RESULT_V1.json`):

   | Quantity | This audit (wrong) | Corrected |
   |---|---|---|
   | median relative headroom | 0.208% | 0.208% |
   | 90th percentile | 12.8% | **11.3%** |
   | states above 10% | 111 (15.4%) | **92 (12.8%)** |
   | states above 5% / 1% | — / 212 | 133 (18.5%) / 211 (29.3%) |
   | Azure-code KV-16,000 p90 | 19.3% | **16.1%** |
   | ratio of means | 2.03% | 1.995% |

   The harm-structure figures (102/720 all-harmful, 22 mixed, 28 zero-headroom; 660/141/30 of 831 alternatives) were
   recomputed from the action-level rows and are unchanged.
3. **OpenTela "Operational traces" (Table 1 cell) was marked unverified; it is supported.** The OpenTela paper releases
   and analyzes an anonymized production trace (July 2024 – October 2025) from its own deployment (PDF text, Section 6).
4. **Choudhury et al. and Yildiz et al. (PEVA)** were marked unverified; Crossref confirms title, authors, volume, and
   article number for both (and for Lilou and Ali et al.). Abstracts were not available, so the manuscript keeps
   title-level descriptions.
5. **Additional finding.** The KV-constrained SBS's "urgent" exemption (slack under 0.25 s) never applies because
   deadlines are 1000 s and the largest mean continuation latency is 0.36 s; the reserve therefore acts as a hard cap.
