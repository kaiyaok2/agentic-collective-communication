# PPoPP'27 #1429 rebuttal experiments — results summary (7× trn1.32xlarge, ap-southeast-2c, 2026-10-13/14)

SDK on this cluster: neuronx-cc 2.22.12 / runtime 2.29.40 (submission: compiler 2.24). Every comparison re-measures its
own baseline; the paper's absolute numbers nevertheless reproduce. Detailed per-section documents: rebuttal_draft.md
(response text), sim_v4_changes.md (simulator fixes), sim_correlation_final.md, taxonomy_AQ5_attempted.md.

## 1. Headline reproductions (C-Q5, C7)
| Quantity | This cluster | Paper |
|---|---|---|
| OLMoE-10B baseline, 224 ranks, steady median (3 data seeds) | 4348.8 / 4347.6 / 4350.1 ms | 4404.3 |
| OLMoE-10B OverlayCCL (pack-and-gather), 3 seeds | 3127.4 / 3130.7 / 3122.1 ms → **1.39×** | 3139.8 → 1.40× |
| Llama-7B per_mb steady (3 runs) | 71.92 / 71.85 / 73.02 ms | 71.3 |
| Llama-7B bundled steady (3 runs) | 21.14 / 22.07 / 21.93 ms → 3.3× | 22.0 → 3.24× |
| Llama-7B 6000 steps incl. cold compile, wall | per_mb 593 / 596 s vs bundled 525 / 513 s → **1.145× net** | (not reported) |
| Llama-7B 6000 steps warm cache, wall | 595 vs 500 s → 1.19× | |
| Bundled fixed overhead (trace/load/warm-up, cache-independent) | ≈370 s; break-even ≈4,100 steps | ≈3.7 min; ≈4,400 |
| Llama-7B on a second 7-node cluster (cluster 2, 2026-10-14 23:30 UTC), 300 steps | per_mb 73.19 vs bundled 22.45 ms → **3.26×** | 71.3 / 22.0 → 3.24× |
| Llama-7B at 3 nodes, model fixed (D) | per_mb 65.7 vs bundled 21.0 ms (3.13×) | 67.8 vs 27.2 (2.49×) |
| OLMoE at 3 nodes (96 experts), 2 baseline runs | 2092.0 / 2090.2 vs 1377.5 ms (1.52×) | 2547 vs 1422 (1.79×) |

**Cluster-2 fabric note (2026-10-14 21:20 UTC onward):** after a stuck-collective hardware error (`NRT_EXEC_HW_ERR_COLLECTIVES`)
triggered by the DM=4096 attempt, large-payload collectives on cluster 2 fail with CCOM retransmit timeouts even after rebooting
all seven instances (EFA `retrans_timeout_events` on every adapter), while small-collective workloads (Llama-7B, per-problem
harnesses, Ring-KV pipeline runs) keep working. The OLMoE headline could therefore not be re-measured on cluster 2 (two attempts
each arm); its cross-cluster evidence is the Llama-7B reproduction plus the 6+3-seed OLMoE repeats on cluster 1.

## 2. Phase-1 probes: the shipped simulator used a stale snapshot (A-Q3)
Both the public artifact and the archived private commit route every Phase-1 `measure_*` tool to a static table. Live
probes on this cluster (single node, 32 ranks, cold cache) vs the snapshot: collective dispatch (4 KB AR in its own
graph) 1674 µs vs 100; back-to-back marginal ≈33 µs/AR → alphas (0.011, 0.027, 0.027) vs (0.30, 0.10, 0.02);
per-mark_step launch 150 µs vs 1500; dense permute copy 67 GB/s vs 200 (clone 124 GB/s); strided column extraction,
on the full-tensor basis the model charges, 297 GB/s vs 20; per-op in-graph marginal 10–70 µs for view ops (table 0.1),
200–1200 µs isolated for gather/scatter/index_select (table 29); cold compile+load 2–37 s for ≤4 MB all-gathers (an 8 MB
all-gather graph compiled >15 min and thrashed the host). Small-collective latencies differ ±2× between probe runs.
Sensitivity of the corrected model's rankings (agreement with the live-calibrated arm): network curves ×4 / ×0.25 → bundling problems 1.00 / 0.90–1.00, AllToAllV 0.93 / −0.21, dxe 0.81 / 0.93; local terms reverted to the snapshot → dxe 0.81, Ring KV 0.94, others 1.00. The paper's ρ=1.00 TPU swap holds only on the dispatch-count-dominated problems (arms sim_v4_v13_net4x / net025x / localsnap).

**Second cluster, same SDK (2026-10-14 11:40 UTC probe):** volume terms reproduce (HBM copy 292 vs 297 GB/s strided basis,
74 vs 67 GB/s dense; index_select 1M = 18.5 vs 17.9 ms; 4 MB all-gather 5.07 vs 5.18 ms; gather/index_select marginals
within 3%), while latency-floor terms move ±2× (dispatch 968 vs 1674 µs; 4 KB all-reduce 406 vs 1511 µs; back-to-back
alphas (0.33, 0.0) vs (0.011, 0.027)). This is the per-deployment variance the Phase-1 probes exist to absorb, and it lands
on exactly the terms that decide small-payload bundling decisions.

**Probe repeatability, four runs (cluster 1 ×1, cluster 2 ×3, same SDK):** volume terms are stable — dense copy 67–76 GB/s
(1.13× spread), clone 124–134 GB/s (1.08×), index_select 1M = 17.9–18.5 ms (1.03×), gather marginal 605–642 µs (1.06×),
4 MB all-gather 5.07–5.34 ms (1.05×) — while latency-floor terms are not: dispatch 968–1674 µs (1.73×), mark_step
102–201 µs (1.97×), 4 KB all-reduce 406–1692 µs (4.2×), 4 KB all-gather 1036–1817 µs (1.75×), back-to-back α₁ 0.01–0.33.
The corrected simulator's rankings rest on the stable terms (bytes moved × measured rates); the unstable terms only decide
small-payload bundling, where we amortize them, and this spread is the per-deployment variance Phase-1 probing exists to absorb.

## 3. Simulator vs hardware, final (A-Q2, C-Q1) — arm v16b, one configuration for all problems
74 candidates (seed templates, grammar variants, archived LLM strategies, deployed runtimes, baselines, search finds,
6 AllToAllV seed-library helpers), 3 repeats at 224-rank training scope; repeat sd ≤2%.
| Problem | n | ρ (sim vs measured step) | top-1 | regret |
|---|---|---|---|---|
| AllToAllV (per-primitive scope) | 8 (helpers n=3, seed helpers n=4 reps) | 0.90 | yes (vectorized AG+slice helper 24.2 ms) | 0 |
| Distributed CE | 8 | 0.73 | yes | 0 |
| FSDP prefetch | 7 | 0.81 | yes | 0 |
| Layer-block AR | 7 | 0.90 | yes | 0 |
| PP send/recv | 6 | 0.89 | yes | 0 |
| Ring KV | 6 | 0.75 | top-2 (HW tie 4.96 vs 5.14) | 3.7% |
| TP MLP | 16 | 0.90 | within 0.3% | 0.3% |
| Uniform A2A | 10 (3 reps, sd ≤6 ms) | 0.73 | yes (AG+slice extraction 1205 ms; AG+RS forms 1511–1526; harness baseline 1515; AG+index_select 5692) | 0 |
Pooled Spearman on speedup ratios 0.69 (n=68); direction agreement 31/33. Simulator arm = sim_v4_v16b (v13 + XLA dead-collective elimination + helper variants scored by the code the harness runs + volume terms paid fwd+bwd). Prediction errors: ratio-MAPE 8–16% (lbar, pp, fsdp, rkv), 24–27% (tp, a2av, dxe), 120% (ua2a: ranking right, gap magnitude over-predicted); affine-rescaled median error 0.1–1.9% (bundling), 13–15% (a2av, dxe, rkv), 35% (ua2a). Per-candidate predicted-vs-measured table (C-Q1 scatter data): results/sim_vs_hw_v16b.csv (68 rows, n_reps, measured sd); earlier arm in sim_vs_hw_v13.csv. Before fixes: dxe −0.77, AllToAllV −0.32; intermediate arm v13: a2av 0.56, dxe 0.56, rkv 0.58, others as above.
Cross-scope inversion reproduced by the simulator: per-primitive payload → AG+T+RS 19972 < AG+slice 29045/40760 (HW
30.0 / 24.2 / 38.0 ms); OLMoE payload → AG+slice 131603/143416 < AG+T+RS 271144 (HW 3127 vs 4338 ms).

**Which simulator terms are load-bearing (leave-one-out ablation of the fixes, same pool, same constants):**

| arm (fix removed) | a2av | dxe | fsdp | lbar | pp | rkv | tp | ua2a | pooled | direction |
|---|---|---|---|---|---|---|---|---|---|---|
| full (v16b) | 0.9 | 0.73 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.69 | 31/33 |
| − real harness shapes (toy shapes × multiplier) | 0.9 | 0.73 | 0.52 | 0.9 | 0.89 | 0.75 | 0.74 | 0.73 | 0.62 | 34/39 |
| − bytes actually moved (input bytes only) | 0.9 | 0.27 | 0.91 | 0.9 | 0.89 | 0.93 | 0.9 | 0.73 | 0.56 | 30/35 |
| − volume-charged local ops (flat per-op cost) | 0.9 | 0.27 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.53 | 28/33 |
| − measured collective curves (topology constants) | 0.29 | 0.73 | -0.41 | -0.2 | -0.8 | 0.75 | -0.26 | 0.73 | 0.27 | 16/21 |
| − fusion-aware dispatch runs (any local op resets) | 0.9 | 0.02 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.57 | 29/33 |
| − XLA concat-of-slices rule | 0.9 | 0.73 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.69 | 31/33 |
| − XLA dead-collective elimination | 0.9 | 0.73 | 0.81 | 0.9 | 0.89 | 0.58 | 0.9 | 0.73 | 0.64 | 30/33 |
| − forward+backward factor on volume terms | 0.9 | 0.56 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.64 | 28/30 |
| − scoring helpers by the code they run | 0.56 | 0.73 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.65 | 30/33 |

The measured collective latency curves carry the bundling problems (without them five problems invert); bytes-moved,
volume-charged local ops and fusion-aware dispatch runs carry distributed CE; real harness shapes carry FSDP/TP MLP;
dead-collective elimination carries the Ring-KV search find; helper-code scoring carries AllToAllV. The concat-of-slices
rule is redundant once helpers are scored by their own code. Removing bytes-moved raises FSDP/Ring KV (0.91/0.93) but
collapses dxe (0.27): the full model is the compromise that keeps every problem above 0.7.

**Cross-cluster check (cluster 2, 1–2 repeats of the same pool, same v16b simulator — no re-fitting):** ρ AllToAllV 0.90,
dxe 0.81, FSDP 0.85, Layer-block 0.90, PP 0.80, Ring KV 0.56, TP MLP 0.87, Uniform A2A 0.87; pooled 0.75, direction 29/30
(sim top-1 = HW best or within 0.1–5.8%). Hardware itself reproduces across clusters with rank agreement 0.77–0.96 and median
step-time difference 0.3–2.1% — except dxe (rank agreement 0.48 on the first repeat, 5.3%): its six all-reduce variants are
within hardware noise of each other, which bounds any simulator's ρ on that problem (the simulator keeps 1-AG last on both
clusters). The second cluster-2 repeat is partial (26/50) because large-payload collectives began hanging on that cluster's
fabric (see §1 note).

## 4. Leak-free non-LLM search (C-Q2)
Generic AST edits (swap primitive, mutate constant, delete/move, insert tensor op, stack/cat operands, loop unroll/fuse)
× {random walk, GA, beam, compiler passes}, same oracle, from each baseline.
- vs shipped simulator (budget 150): "wins" of 3.7–60× on PP/TP-MLP/FSDP = the microbatch loop unrolled by two — a
  detector exploit; hardware 1.00× (TP MLP 11.65–11.90 vs 11.75; FSDP 15.25 vs 14.5–15.3; PP variants crash). One real
  find: Ring KV parts-per-chunk 4→1 (GA), HW 6.82 vs 8.81 ms baseline (deployed per-slot 4.96).
- vs fixed simulator (budget 60, 2 seeds): **no strategy-level improvement on any of the 8 problems**; the only sim gains are compiler-pass-level rewrites of the PP loop (LICM of the zeros buffer: 0.985×; GA unroll + shared buffer: 0.990×); both fail the hardware gate (harness-incompatible CPU tensors inherited from the seed template).
  None of the methods produced pack-and-gather, the dxe recomposition, or stack/cat bundling.
- vs the FINAL simulator (v16b; budget 60, 2 seeds, 4 workers): random/beam find the Ring-KV parts-per-chunk reduction (sim 0.81–0.82×; hardware 1.29×, measured 6.8 vs 8.8 ms) — a real but minor parameter change of the baseline, not a new strategy; PP: compiler-pass LICM (0.97×, harness-incompatible); all other problems: no improvement. Still no pack-and-gather, dxe recomposition, or stack/cat bundling from any non-LLM method.
- Archived non-LLM Phase-3 arms ("none"/"random" over the seed library): reproduce OLMoE (1.41×), deploy the baseline on
  all four Llama problems.

## 5. Realistic OLMoE (A-Q1): compute fraction and speedup
| config (B=1, S=256, L=8, DM=2048, 224 experts) | AG+T+RS | pack-and-gather | no-collective stub | collective share | speedup |
|---|---|---|---|---|---|
| expert dim 1024 (paper) | 4347–4351 ms | 3121–3130 ms | 333 ms | 92% | 1.39× |
| expert dim 4096 | 4407 | 3168 | 365 | 92% | 1.39× |
| top-k 4 (expert dim 1024) | 2830 ms | 1993 ms | 316 ms | 89% / 84% (agent) | 1.42× |
| top-k 2 (expert dim 1024) | 2079 ms | 1677 ms | 310 ms | 85% / 82% (agent) | 1.24× |
| expert dim 16384 | device OOM | device OOM | device OOM | — | — |
| **top-k 2 + expert dim 8192** (3 runs, 2 seeds) | 2164 / 2163 / 2156 ms | 1740 / 1724 / 1718 ms | 365 ms | 83% / 79% (agent) | **1.25×** |
| top-k 4 + expert dim 8192 (2–3 runs) | 2923 / 2906 ms | 2046 / 2030 / 2031 ms | 381 ms | 87% / 81% (agent) | 1.43× |
| model width 4096 (L=8, 4, 2 variants) | device OOM (L=8); stuck-collective hardware error `NRT_EXEC_HW_ERR_COLLECTIVES` (L=4, L=2) | | | — | — |
| expert dim 8192 | device OOM | 3216 | 440 | 86% | — (AG+RS needs a second full buffer) |

(The top-k 2/4 + ex 8192 rows combine cluster-1 runs (seed 42 ×2) and cluster-2 runs (seeds 43/44); one cluster-1 repeat's JSON was lost when AWS reclaimed the block 35 min early — its numbers come from the run log.)
**Why tokens per step cannot change this share.** With all-gather-based AllToAllV over 224 experts the collective payload is
ws·T·k·D and the expert compute is T·k·D·EXDIM: both are linear in the token count T, so the collective share is T-invariant; only
the per-token compute (expert width, top-k, model width) moves it — which the table sweeps. B>1 attempts on cluster 2 with two harness rewrites (one-hot matmul gather/scatter; per-sample top-k keeping every sort at S=256): the T>256 routing graph either faults at runtime (`NRT_EXEC_OOB` in the indirect-DMA gather/scatter) or fails to compile (`NCC_ITEN404 MaskPropagation` internal compiler error), so larger-token steps are blocked by the Neuron stack, not by the harness logic. Every larger-token configuration (B=2/4, S=512, L=16) fails at NEFF load / device memory for both backends: the AG-based
AllToAllV buffer is ws×cap×DM·ws bytes (2 GB per rank at the paper shape, 8 GB at B=4). The regime is collective-bound by
construction of 224-way expert parallelism; the transferable number is the collective-component speedup 4016→2795 ms (1.44×).

## 6. Attribution / cross-scope (A-Q2, C-Q1, C-Q3)
AG+T+RS routed through the agent path at OLMoE scope = 4338 ms (inline baseline 4349) → the 1.39× is the algorithm, not the
harness path; at per-primitive scope the ordering inverts (AG+T+RS 30.0 vs pack-and-gather 38.0 ms). Megatron/NxD 3-AR
cross-entropy at 224 ranks 8.25–8.31 ms vs deployed 2-AR 8.3–8.6 vs 1-AG baseline 14.6–14.9 → dxe gain ≈1.0× vs the library baseline.


## Live-calibration reproduction (Phase 1 → 3 → 4 → 5 end to end, cluster 2, 2026-10-14)
Same pipeline (`experiments/run_search.py`, sonnet, strategy-enumerate, HW gates on 7 nodes), three Phase-1 arms:
(S) the shipped static snapshot table; (L) the live on-cluster probe table (`OVERLAY_HW_TABLE`); (L+v4) live table with
the corrected simulator structure. Deployed runtimes then measured 3× with the per-problem harness at 224 ranks.

| problem | arm | Phase-1 outcome | Phase-3 ranking (sim µs) | deployed | measured (ms) |
|---|---|---|---|---|---|
| dxe | S | dispatch 100, alphas (.3,.1,.1) | 1-AG 5207 < 2-AR 5528 < 3-AR 5668 | **1-AG (baseline)** | 13.0 / 13.3 / 13.6 |
| dxe | L | live constants | 1-AG 16166 < 2-AR 16818 < 3-AR 17406 | **1-AG (baseline)** | 12.7 / 13.0 / 13.4 |
| dxe | L+v4 | alphas (.33,.15,.14) | 2-AR async 15817 < 3-AR 15875 < fused 2-AR 16002 < **1-AG 21531 (last)** | **2-AR recomposition** | **8.6** (vs 13.0–13.6 for the 1-AG deployments of arms S and L) |
| alltoallv | S | alphas (.3,.1,.02) | ag_slice_cat 5979 < AG+T+RS 16695 (3 LLM strategies fail the oracle) | **pack-and-gather seed template `ag_slice_cat`** | plain helper 29.6 (= AG+T+RS, defect #10); wrapped: 47.0–47.6 per-primitive, **6199 ms at OLMoE scope** (slower than the 4349 baseline; the paper's archived runtime — a different loop+cat implementation — gives 3125) |
| alltoallv | L | alphas (.33,0,0) | permute_ring 11632 < ag_slice_cat 12787 | **collective_permute ring** (evo:permute_ring_3) | per-primitive 23.8–24.2 ms (correctness unverified by the harness); OLMoE scope: device OOM (2 GB buffer) |
| ring_kv | S | snapshot | per_slot 5264 < fused/batched 11075 (2 LLM strategies fail the oracle) | **per-slot AG** (= paper) | 5.08 / 5.08 / 5.28 ms (pool: per-slot 4.96) |
| ring_kv | L+v4 | live table + corrected structure | per_slot 89824 < bucket-fused 90353 < fused concat 90738 ≪ pipelined-chunked 861592 | **per-slot AG** (same as S) | same runtime class, 5.1 ms |
| alltoallv | L+v4 | alphas (.33,.15,.15) | LLM "segmented all-gather + strided indexing" (refined) 292448 < ag_slice_cat 315832; 3 LLM strategies fail the oracle | **LLM pack-and-gather variant** | per-primitive 47.2 (wrapped; = template 47.0); OLMoE scope 6198 ms via the generic wrapper (same as the template; the paper's hand-written uniform helper is what yields 3125) |

Reading: with the shipped model structure, live constants do not repair the dxe mis-ranking (1-AG still wins) and can
promote a hardware-infeasible permute ring on AllToAllV; with the corrected structure plus live constants the pipeline
deploys the all-reduce recomposition on dxe and ranks 1-AG last — calibration operating end to end, and necessary but not
sufficient without the structural terms (real shapes, bytes moved, volume-charged local ops). On Ring KV (a bundling
problem) all arms deploy the same per-slot all-gather — consistent with the sensitivity result that bundling problems do not
depend on the constants.

## 7. A-Q5: 10 strategy classes attempted, 4 deployed — see taxonomy_AQ5_attempted.md.

## 8. Artifact defects found
1. `runtime/trainium_alltoallv_7node.py` line 1 is a stray scp warning → SyntaxError on import.
2. Its `alltoallv()` helper is a later "vectorized" version that OOMs at the paper's OLMoE shape; the measured loop+cat helper works.
3. Phase-1 `measure_*` tools return a static snapshot table (public and 20fc5e3).
4. Deployed Layer-block runtime == baseline; the gain comes from the e2e harness's bundling.
5. Registered ring_kv `per_chunk_per_part_allgather` template fails its own reference.
6. Microbatch-loop graph detector only matched `range(M)` (gameable); fixed locally.
7. `$19/hour` should be ~$150/h; Table 5 7n-train (4410) vs Table 6 step (4404) come from different harnesses.
8. `collective_permute` deadlocks in CCOM bootstrap at ws=32 on SDK 2.22/2.29.
9. The Uniform-A2A harness imports `runtime/trainium_uniform_a2a.py` (not the `_7node` file); the shipped copy is a `collective_permute` ring that aborts at 224 ranks (`ENC_MAX_COMM_N`), so the public ua2a agent path is broken as shipped and every earlier "agent variant" measurement silently ran it. Fixed in deploy_candidate.py; pool re-measured (see §3).
10. Phase-5 codegen (`search/problems.py` alltoallv `public_api_code`, line ~409) emits an `alltoallv(x, ws, mc)` helper hard-coded to AG+transpose+RS ("matching make_baseline_alltoallv ... evolved_alltoallv is not exercised by OLMoE"): the shipped pipeline's deployed AllToAllV strategy never reaches the OLMoE / per-primitive harness path (a freshly generated pack-and-gather runtime measures 29.6 ms = the inline baseline). The paper-era runtime (r28 archive) carried a pack-and-gather helper; the template was overwritten by the r43 "inline match" experiment.
11. The paper-era runtime's uniform helper (`alltoallv(x, ws, mc)`: all-gather the pre-packed x, then 224 fixed-offset slices + cat; r28 archive) is hand-written and is what the OLMoE harness measured (3125 ms). Routing the identical seed template `ag_slice_cat` through a generic uniform wrapper (re-pack loop + per-source slices) costs 6199 ms at OLMoE scope — slower than the 4349 baseline. The shipped codegen emits neither (defect #10). The OLMoE headline therefore depends on the helper implementation, not only on the deployed algorithm class.
