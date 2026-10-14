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
| Llama-7B at 3 nodes, model fixed (D) | per_mb 65.7 vs bundled 21.0 ms (3.13×) | 67.8 vs 27.2 (2.49×) |
| OLMoE at 3 nodes (96 experts), 2 baseline runs | 2092.0 / 2090.2 vs 1377.5 ms (1.52×) | 2547 vs 1422 (1.79×) |

## 2. Phase-1 probes: the shipped simulator used a stale snapshot (A-Q3)
Both the public artifact and the archived private commit route every Phase-1 `measure_*` tool to a static table. Live
probes on this cluster (single node, 32 ranks, cold cache) vs the snapshot: collective dispatch (4 KB AR in its own
graph) 1674 µs vs 100; back-to-back marginal ≈33 µs/AR → alphas (0.011, 0.027, 0.027) vs (0.30, 0.10, 0.02);
per-mark_step launch 150 µs vs 1500; dense permute copy 67 GB/s vs 200 (clone 124 GB/s); strided column extraction,
on the full-tensor basis the model charges, 297 GB/s vs 20; per-op in-graph marginal 10–70 µs for view ops (table 0.1),
200–1200 µs isolated for gather/scatter/index_select (table 29); cold compile+load 2–37 s for ≤4 MB all-gathers (an 8 MB
all-gather graph compiled >15 min and thrashed the host). Small-collective latencies differ ±2× between probe runs.
Network constants do not matter for ranking (TPU swap ρ = 1.00 reproduced); local terms do.

## 3. Simulator vs hardware, final (A-Q2, C-Q1) — arm v13, one configuration for all problems
74 candidates (seed templates, grammar variants, archived LLM strategies, deployed runtimes, baselines, search finds,
6 AllToAllV seed-library helpers), 3 repeats at 224-rank training scope; repeat sd ≤2%.
| Problem | n | ρ (sim vs measured step) | top-1 | regret |
|---|---|---|---|---|
| AllToAllV (per-primitive scope) | 8 (seed-library helpers n=4 reps) | 0.56 | no (AG+T+RS 30.0 vs vectorized AG+slice 24.2 ms) | 23.8% |
| Distributed CE | 8 | 0.56 | yes | 0 |
| FSDP prefetch | 7 | 0.81 | yes | 0 |
| Layer-block AR | 7 | 0.90 | yes | 0 |
| PP send/recv | 6 | 0.89 | yes | 0 |
| Ring KV | 6 | 0.58 | top-2 (HW tie 4.96 vs 5.14) | 3.7% |
| TP MLP | 16 | 0.90 | within 0.3% | 0.3% |
| Uniform A2A | 10 (⟨n=3 reps pending⟩) | 0.73 | yes (AG+slice extraction 1205 ms; AG+RS forms 1511–1526; harness baseline 1515; AG+index_select 5692) | 0 |
Pooled Spearman on speedup ratios 0.56 (n=68); direction agreement 26/30. Simulator arm = sim_v4_v13 (v11c + index_select at its measured gather rate). Before fixes: dxe −0.77, AllToAllV −0.32.
Cross-scope inversion reproduced by the simulator: per-primitive payload → AG+T+RS 19972 < AG+slice 29045/40760 (HW
30.0 / 24.2 / 38.0 ms); OLMoE payload → AG+slice 131603/143416 < AG+T+RS 271144 (HW 3127 vs 4338 ms).

## 4. Leak-free non-LLM search (C-Q2)
Generic AST edits (swap primitive, mutate constant, delete/move, insert tensor op, stack/cat operands, loop unroll/fuse)
× {random walk, GA, beam, compiler passes}, same oracle, from each baseline.
- vs shipped simulator (budget 150): "wins" of 3.7–60× on PP/TP-MLP/FSDP = the microbatch loop unrolled by two — a
  detector exploit; hardware 1.00× (TP MLP 11.65–11.90 vs 11.75; FSDP 15.25 vs 14.5–15.3; PP variants crash). One real
  find: Ring KV parts-per-chunk 4→1 (GA), HW 6.82 vs 8.81 ms baseline (deployed per-slot 4.96).
- vs fixed simulator (budget 60, 2 seeds): **no strategy-level improvement on any of the 8 problems**; the only sim gains are compiler-pass-level rewrites of the PP loop (LICM of the zeros buffer: 0.985×; GA unroll + shared buffer: 0.990×); both fail the hardware gate (harness-incompatible CPU tensors inherited from the seed template).
  None of the methods produced pack-and-gather, the dxe recomposition, or stack/cat bundling.
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
| top-k 2 + expert dim 8192 | ⟨running⟩ | ⟨running⟩ | ⟨running⟩ | | |
| expert dim 8192 | device OOM | 3216 | 440 | 86% | — (AG+RS needs a second full buffer) |
Every larger-token configuration (B=2/4, S=512, L=16) fails at NEFF load / device memory for both backends: the AG-based
AllToAllV buffer is ws×cap×DM·ws bytes (2 GB per rank at the paper shape, 8 GB at B=4). The regime is collective-bound by
construction of 224-way expert parallelism; the transferable number is the collective-component speedup 4016→2795 ms (1.44×).

## 6. Attribution / cross-scope (A-Q2, C-Q1, C-Q3)
AG+T+RS routed through the agent path at OLMoE scope = 4338 ms (inline baseline 4349) → the 1.39× is the algorithm, not the
harness path; at per-primitive scope the ordering inverts (AG+T+RS 30.0 vs pack-and-gather 38.0 ms). Megatron/NxD 3-AR
cross-entropy at 224 ranks 8.25–8.31 ms vs deployed 2-AR 8.3–8.6 vs 1-AG baseline 14.6–14.9 → dxe gain ≈1.0× vs the library baseline.

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
