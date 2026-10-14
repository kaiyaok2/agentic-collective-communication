# PPoPP'27 #1429 — author response (DRAFT v2, 2026-10-14; ⟨⟩ = pending from runs still in flight)

We thank the reviewers. We answer the explicit questions first. All new numbers were measured on a fresh 7-node
trn1.32xlarge cluster (224 ranks; compiler 2.22 / runtime 2.29 vs 2.24 in the submission) with the submission's code,
unless marked "archived". Every table below will be added to the paper and the artifact.

## A-Q2 / C-Q1 — Simulator accuracy, and a correction

**Correction first.** While re-running Phase 1 we found that the released pipeline (and the commit the paper's runs used)
had replaced the on-device Phase-1 probes with a *snapshot* of an earlier probe campaign: every `measure_*` tool returned
values from a static table, so the Phase-1 "calibration" of the submitted runs was a stale constant set, not a live
measurement. We re-ran the real probes on the cluster (single node, 32 ranks, cold cache; collective latency curves,
back-to-back amortization, per-op in-graph costs, HBM copy rates, mark_step and NEFF-load costs). The snapshot was off by
large factors on this SDK (collective dispatch 1674 µs vs 100; per-mark_step 150 µs vs 1500; dense copy 67 GB/s vs 200;
the strided-extraction term over-charged `[:, rank, :].contiguous()` ≈15×; per-op costs 10–70 µs for views vs 0.1).
Small-collective latencies also vary ±2× between probe runs. We will state this in §3/§5 and release the probes.

**Faithfulness fixes (structural, no constant tuned against the hardware pool).** With live constants alone the two
payload-changing problems still mis-ranked (dxe ρ −0.72, AllToAllV −0.32) because the sandbox traced candidates at toy
shapes (dxe: a 4 KB all-gather × a hand multiplier, vs 147 KB in / 33 MB out on hardware) and charged local elementwise
ops a flat ~29 µs regardless of size. We now (i) trace at the harness's real training shapes, (ii) charge collectives on
the bytes they actually move (an all-gather receives (ws−1)×input) with measured latency curves, (iii) charge elementwise/
reduction/memset ops by volume at the measured HBM rate, (iv) let a dispatch run survive XLA-fusible local compute and
amortize only latency-bound (≤256 KB) collectives, (v) apply XLA's concat-of-consecutive-slices simplification, and
(vi) derive graph boundaries from the scope being scored. These changes are listed in the artifact (sim_v4_changes.md).

**Result (71-candidate pool over the 8 problems, each measured 3× at 224-rank training scope; repeat sd ≤2%):**
within-problem Spearman between simulator score and measured step time — AllToAllV **0.56** (8 cands), distributed CE
**0.56** (8; sim top-1 = HW best), FSDP **0.81** (7; top-1), Layer-block **0.90** (7; top-1), PP **0.89** (6; top-1),
Ring KV **0.58** (6; sim's pick within 3.7% of best, the two leaders tie within repeat noise), TP MLP **0.90** (16; within
0.3%), Uniform A2A: not measurable — every alternative implementation we could deploy (the three archived LLM variants and the three seed-library templates) aborts in the Neuron runtime (`ENC_MAX_COMM_N` assertion) at 224 ranks, so only the baseline runs; we report this as a limit of the SDK, not as a data point. Pooled Spearman on
speedup-vs-baseline ratios 0.60 (n = 58); where the sim predicts a change vs the baseline its direction agrees with
hardware 22/25 times. Before the fixes: TP MLP 0.74, Layer-block 0.90, PP 0.93, FSDP 0.60, Ring KV 0.56, dxe −0.77,
AllToAllV −0.32. We will replace the paper's simulator-accuracy claims with this table.

**The cross-scope inversion is reproduced by the simulator, not assumed.** For AllToAllV the same candidates scored at the
per-primitive harness's payload rank AG+T+RS first (sim 19972 < vectorized AG+slice 29045 < loop+cat 40760; hardware
30.0 / 24.2 / 38.0 ms), and at the OLMoE step's payload rank pack-and-gather first (sim AG+slice 131603 / 143416 <
AG+T+RS 271144; hardware 3127 vs 4338 ms). The driver is the AG+RS form's second collective over the full
ws×ws×cap×DM buffer plus its dense permute copy, both of which scale with payload.

## A-Q3 — Why per-deployment calibration rather than a one-time hand-fit

Reviewer A's suspicion was, for the submitted runs, literally correct (see the correction above) — and the data say why
it matters: the ranking is indeed insensitive to the *network* constants (swapping all six for TPU values changes no
ranking: ρ = 1.00, reproduced), but it is sensitive to the *local* terms — per-op in-graph costs, HBM copy rates,
collective latency curves and NEFF-load behavior — which moved 2–15× between the snapshot and this SDK and ±2× between
probe runs. Those are exactly the quantities the probes measure, and the quantities that decide the payload-changing
problems (dxe, AllToAllV). We will rewrite the §5 portability claim as: network constants are not what calibration is for;
local cost terms are, and they must be measured on the deployed SDK.

## C-Q2 / B — Is the LLM necessary? Non-LLM generators, leak-free

We built a generator with no knowledge of collective rewrites: generic AST edits (swap primitive, mutate constant,
delete/move statement, insert tensor op, stack/cat operands, loop unroll/fuse) driven by random walk, GA, beam search, and
a rule-based arm (standard compiler passes DCE/CSE/LICM/loop-fusion), same correctness oracle, equal budget, starting from
each developer baseline. Against the *shipped* simulator it reported 3.7–60× "wins" on PP/TP-MLP/FSDP: the baseline loop
unrolled by two (`range(0, M, 2)`), which the old graph-boundary detector did not recognize — a simulator exploit, not a
strategy. Hardware: TP MLP 11.65–11.90 ms vs baseline 11.75, FSDP 15.25 vs 14.5–15.3, PP variants fail the harness. The
fixed detector removes it. Against the fixed simulator (budget 60 evaluations per method, 2 seeds, all 8 problems) no method improves on any baseline except two compiler-pass-level rewrites of the PP loop (hoisting the zeros buffer out of the loop, 1.5% in the simulator; hardware check ⟨pending⟩). No method
produced pack-and-gather, the distributed-CE recomposition, or stack/cat bundling; the one real find (Ring KV, GA: fewer
parts per chunk) measures 6.82 ms vs 8.81 baseline and 4.96 deployed. Separately, the archived non-LLM Phase-3 arms
("none"/"random" over the seed library) reproduce the OLMoE headline (pack-and-gather is a seed template; 1.41×) but
deploy the baseline on all Llama problems — the bundling strategies behind the 3.24× are LLM-authored.

## A-Q4 — Call-level baselines

None is shape-tuned: each is the generic recipe with fixed loop structure and default primitive arguments; the only
shape-dependent parameter is the padding chunk `max_chunk` shared by both AllToAllV implementations. AllToAllV and Uniform
A2A = `all_gather` → `permute(1,0,2).contiguous()` → `reduce_scatter(SUM, 1/ws)` (the AWS AG+RS recipe); Ring KV = one
`all_gather` per (slot, head); distributed CE = `all_gather` of vocab-sharded logits + `F.cross_entropy`; PP = one masked
`all_reduce` per microbatch; TP MLP = one `all_reduce` per (microbatch, layer); FSDP = one `all_gather` per (microbatch,
layer); Layer-block = two `all_reduce`s per layer. One concession: the standard vocab-parallel cross-entropy in
Megatron-LM / NeuronX-Distributed uses three small all-reduces, not an all-gather; measured at 224 ranks it runs 8.25–8.31
ms vs 8.3–8.6 for the deployed 2-AR strategy and 14.6–14.9 for our 1-AG baseline, so against the library baseline the
distributed-CE gain is ≈1.0× and Table 5's dxe row will say so.

## A-Q5 — How many strategies: delivered vs attempted

Delivered: 8 runtimes spanning 4 mechanisms (algebraic collective elimination AG+RS→AG+slice; cross-call bundling by
stack/cat/flatten; algebraic recomposition changing collective type and payload (dxe); graph-boundary lifting), plus the
masked-AR substitute for `collective_permute`. Attempted (archived Phase-3 logs, 27 runs × K=5 = 135 LLM
implementations, ~65 distinct sketches, plus 70 refinement calls): **10 qualitatively distinct classes**. Six were never
deployed: hierarchical intra/inter-node two-level schemes (12 attempts), ring/butterfly/Bruck permutation schedules (10)
and explicit P2P/all_to_all substitution (4) all died at the correctness oracle; payload-changing recompositions other
than dxe's (26) mostly failed the oracle and the survivors lost on the simulator or the 224-rank shape gate; bucketing/
chunking (20) and async/pipelined orderings (14) passed the oracle but were slower in the simulator; AG + index_select
extraction (3) failed the 224-rank index gate. We will add this funnel (with per-class failure stage) as an appendix table.

## A-Q1 — Realistic OLMoE configuration

Measured directly. Replacing AllToAllV by a local identity stub gives the step's non-collective time; raising the expert
width raises compute without changing the AllToAllV payload:

| config (B=1, S=256, L=8, DM=2048, 224 experts) | AG+T+RS baseline | pack-and-gather | no-collective stub | collective share | speedup |
|---|---|---|---|---|---|
| expert dim 1024 (paper) | 4347–4351 ms | 3121–3130 ms | 333 ms | 92% | 1.39× |
| expert dim 4096 | 4407 ms | 3168 ms | 365 ms | 92% | 1.39× |
| expert dim 8192 | device OOM (second buffer) | 3216 ms | 440 ms | 86% | — |

Larger token counts are not reachable by either implementation: the AG-based AllToAllV allocates a ws×ws×cap×DM buffer
that grows linearly with tokens (2 GB per rank at the paper's shape; B=4 is 8 GB), so every B>1 / longer-sequence
configuration fails at load on this stack for both backends. We will therefore (i) state that the headline is the
collective-bound regime of 224-way expert parallelism, (ii) report the collective-component speedup (4016→2795 ms,
1.44×) as the quantity that transfers to other step compositions, and (iii) note the memory advantage of the deployed
strategy (one gathered buffer instead of two), which is why it still runs at expert dim 8192 where AG+RS does not.

## C-Q5 / C7 — Wall-clock, compile, repeats

Llama-7B 6000 steps from a cold cache: per_mb 593/596 s vs bundled 525/513 s → **1.145× net including compile**
(steady 71.9 vs 21.1–22.1 ms/step; paper 71.3/22.0 → 3.3×). Warm cache 595 vs 500 s (1.19×). Bundled fixed overhead ≈370 s,
cache-independent (trace/NEFF load/warm-up); break-even ≈4,100 steps (paper ≈4,400). We will label 3.24× steady-state.
OLMoE headline repeats: archived n=7 baseline / n=6 agent (4391.7–4404.3 vs 3123–3140 ms) plus 3 new data seeds
(4348.8/4347.6/4350.1 vs 3127.4/3130.7/3122.1 → 1.39×) under a different compiler/runtime (2.22/2.29 vs 2.24).

## C-Q4 — Accounting corrections
"$19/hour" is the LLM token cost (cluster ≈$150/h, which the $47k figure uses); Table 5's 7n-train column is a per-call
probe inside the OLMoE step and Table 6 a whole-step time from a different run — we will re-measure both in one run;
"~8 cluster-hours" = Phase-1 probes + Phase-4 gates + training validation, "56 min" = Phase-3 search wall only; "~30 min"
includes LLM turns, "8–12 min" the on-device probe campaign.

## C8 / D / misc
Archived Phase-4 gates: 125 candidates → 93 pass / 15 fail / 17 skipped (shape errors and `collective_permute` at
ws>64, which also deadlocks in CCOM bootstrap at ws=32 on this SDK). 3-node Llama-7B with the model fixed: per_mb 65.7 vs
bundled 21.0 ms (3.13×); bundled is flat across 3→7 nodes (21.0 vs 21.1–22.1) while per_mb improves as the cluster
shrinks, so the ratio compresses because the baseline speeds up. We will remove the H100 price comparison and the XLA
wording, expand jargon on first use, and resolve the §6 single-trajectory/strategy-enumerate inconsistency.

## Artifact defects we will fix
`runtime/trainium_alltoallv_7node.py` line 1 is a stray scp warning (SyntaxError on import); its vectorized helper OOMs at
the paper's OLMoE shape (the measured loop+cat helper works); the Phase-1 probe tools return a static table; the deployed
Layer-block runtime equals the baseline (gain is the e2e harness's bundling); the registered Ring-KV
`per_chunk_per_part_allgather` template fails its own reference; the microbatch-loop graph detector was gameable.
