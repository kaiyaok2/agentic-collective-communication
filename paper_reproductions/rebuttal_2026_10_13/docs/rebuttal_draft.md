# PPoPP'27 #1429 — author response (DRAFT v3, 2026-10-15 02:30 UTC)

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
amortize only latency-bound (≤256 KB) collectives, (v) apply XLA's concat-of-consecutive-slices simplification and dead-collective elimination, (vi) derive graph
boundaries from the scope being scored, and (vii) pay volume-proportional local terms in forward and backward, as the
collectives already were. These changes are listed in the artifact (sim_v4_changes.md).

**Result (74-candidate pool over the 8 problems, each measured 3× at 224-rank training scope; repeat sd ≤2%):**
within-problem Spearman between simulator score and measured step time — AllToAllV **0.90** (8 cands; sim top-1 = HW
best), distributed CE **0.73** (8; top-1), FSDP **0.81** (7; top-1), Layer-block **0.90** (7; top-1), PP **0.89** (6;
top-1), Ring KV **0.75** (6; sim's pick within 3.7% of best, the two leaders tie within repeat noise), TP MLP **0.90**
(16; within 0.3%), Uniform A2A **0.73** (10; sim top-1 = HW best; measured after fixing an artifact bug — the harness imports `runtime/trainium_uniform_a2a.py`, whose shipped copy is a `collective_permute` ring that aborts at 224 ranks, so earlier "agent" runs never executed the candidate). Pooled Spearman on
speedup-vs-baseline ratios 0.69 (n = 68); where the sim predicts a change vs the baseline its direction agrees with
hardware 31/33 times. Before the fixes: TP MLP 0.74, Layer-block 0.90, PP 0.93, FSDP 0.60, Ring KV 0.56, dxe −0.77,
AllToAllV −0.32. We will replace the paper's simulator-accuracy claims with this table.

**Prediction errors (C-Q1).** Simulator units are not milliseconds, so we report errors on predicted-vs-measured
speedup ratios against each problem's baseline (MAPE) and, after a per-problem affine rescaling, median relative error of
predicted step time: bundling problems (PP, TP MLP, FSDP, Layer-block, Ring KV) ratio-MAPE 8–24%, rescaled error 0.1–14%; payload-changing
problems where the inversion lives (AllToAllV, dxe) ratio-MAPE 24–27%, rescaled error 13–14%; Uniform A2A's ranking is
right but the magnitude of the AG+slice-vs-AG+RS gap is over-predicted (ratio-MAPE 120%). The ranking survives (ρ ≥ 0.73 everywhere) while magnitudes on the payload-changing problems
remain coarse — which is why the pipeline keeps the hardware gate. None of the pool candidates was used to fit anything:
the constants come from the Phase-1 probes, the structure is fixed.

**The cross-scope inversion is reproduced by the simulator, not assumed.** For AllToAllV the same candidates scored at the
per-primitive harness's payload rank AG+T+RS first (sim 19972 < vectorized AG+slice 29045 < loop+cat 40760; hardware
30.0 / 24.2 / 38.0 ms), and at the OLMoE step's payload rank pack-and-gather first (sim AG+slice 131603 / 143416 <
AG+T+RS 271144; hardware 3127 vs 4338 ms). The driver is the AG+RS form's second collective over the full
ws×ws×cap×DM buffer plus its dense permute copy, both of which scale with payload.

## A-Q3 — Why per-deployment calibration rather than a one-time hand-fit

Reviewer A's suspicion was, for the submitted runs, literally correct (see the correction above) — and the data say why
it matters: the ranking is indeed insensitive to the *network* constants (swapping all six for TPU values changes no
ranking: ρ = 1.00, reproduced), but it is sensitive to the *local* terms — per-op in-graph costs, HBM copy rates,
collective latency curves and NEFF-load behavior — which moved 2–15× between the snapshot and this SDK; across four probe runs on two clusters the latency-floor terms spread
1.7–4.2× (dispatch 968–1674 µs, 4 KB all-reduce 406–1692 µs) while volume terms spread only 3–13% (copy 67–76 GB/s,
index_select 17.9–18.5 ms, 4 MB all-gather 5.07–5.34 ms). Those are exactly the quantities the probes measure, and the quantities that decide the payload-changing
problems (dxe, AllToAllV). We measured this directly on the corrected model (74-candidate pool, rank agreement of within-problem orderings with
the live-calibrated model): scaling every collective curve's bandwidth by 4× or by 0.25× leaves the five bundling
problems' rankings unchanged (Spearman 1.00, 0.90–1.00) — that is the invariance behind the paper's ρ = 1.00, and it holds
because those problems are decided by dispatch counts, not bandwidth — but it does *not* hold for the payload-changing
problems: AllToAllV's ranking agreement drops to 0.93 (4× faster network) and **−0.21** (4× slower), distributed CE's to
0.81; and reverting only the *local* terms to the snapshot values changes distributed CE (0.81) and Ring KV (0.94) while
leaving the rest intact. On an independent second cluster, the same simulator (no re-fitting) keeps ρ ≥ 0.56 on all 8
problems against fresh repeats of the pool (dxe 0.81, AllToAllV 0.90, Uniform A2A 0.87; pooled 0.75, direction 29/30). So the TPU swap in §5 tested the problems where constants cannot matter and missed the ones where
they do. We will rewrite the §5 portability claim as: network constants are not what calibration is for on bundling
problems; on payload-changing problems both network and local terms move the ranking, and they must be measured on the
deployed SDK.

**Calibration operating, end to end (new).** On a second 7-node cluster we re-ran the full pipeline (Phase 1 LLM
probing → Phase 3 enumeration → Phase 4 gates → Phase 5 deployment, same prompts, budget and gates) three ways and
measured what each deployed: (S) the shipped snapshot table; (L) the live on-cluster probe table; (L+v4) the live table
with the corrected cost-model structure. Distributed CE: S and L both deploy the full-vocab all-gather (13.0–13.6 ms
measured); L+v4 ranks it last and deploys the two-all-reduce recomposition (8.6 ms, 1.55× over the S/L deployment).
AllToAllV: S deploys the pack-and-gather seed template (the paper's strategy class); L deploys an LLM collective_permute
ring (23.9 ms at per-primitive scope, device-OOM at OLMoE scope); L+v4 deploys an LLM pack-and-gather variant (47.2 ms
per-primitive; 6198 ms at OLMoE scope through the generic helper). One more honest finding from this exercise: the uniform helper the OLMoE harness calls is
not emitted by the shipped codegen (it hard-codes AG+T+RS), and the paper-era runtime's hand-written helper (all-gather
the pre-packed buffer, 224 fixed-offset slices + cat) is what produced 3125 ms; the same template through a generic
re-packing wrapper costs 6199 ms. We will ship the helper with the artifact and state that the headline depends on it.
On Ring KV (a bundling problem) every arm deploys the same per-slot all-gather (5.1 ms), matching the sensitivity
result that bundling decisions do not depend on the constants. So calibration is necessary (S → wrong dxe deployment) but
constants alone are not sufficient (L → still wrong); the structural terms the probes are meant to feed must be in the model.

## C-Q2 / B — Is the LLM necessary? Non-LLM generators, leak-free

We built a generator with no knowledge of collective rewrites: generic AST edits (swap primitive, mutate constant,
delete/move statement, insert tensor op, stack/cat operands, loop unroll/fuse) driven by random walk, GA, beam search, and
a rule-based arm (standard compiler passes DCE/CSE/LICM/loop-fusion), same correctness oracle, equal budget, starting from
each developer baseline. Against the *shipped* simulator it reported 3.7–60× "wins" on PP/TP-MLP/FSDP: the baseline loop
unrolled by two (`range(0, M, 2)`), which the old graph-boundary detector did not recognize — a simulator exploit, not a
strategy. Hardware: TP MLP 11.65–11.90 ms vs baseline 11.75, FSDP 15.25 vs 14.5–15.3, PP variants fail the harness. The
fixed detector removes it. Against the fixed simulator (budget 60 evaluations per method, 2 seeds, all 8 problems, repeated against the final
corrected arm) no method improves on any baseline except the Ring-KV parts-per-chunk reduction (a parameter change of the
baseline: 1.29× on hardware) and two compiler-pass-level rewrites of the PP loop (hoisting the zeros buffer out of the loop, 1.5% in the simulator; both fail the hardware gate — the seed template they descend from allocates CPU tensors the harness cannot run). No method
produced pack-and-gather, the distributed-CE recomposition, or stack/cat bundling. Separately, the archived non-LLM Phase-3 arms
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
| top-k 4 (expert dim 1024) | 2830 ms | 1993 ms | 316 ms | 89% / 84% (agent) | 1.42× |
| top-k 2 (expert dim 1024) | 2079 ms | 1677 ms | 310 ms | 85% / 82% (agent) | 1.24× |
| expert dim 16384 | device OOM | device OOM | device OOM | — | — |
| **top-k 2 + expert dim 8192** (3 runs, 2 seeds) | 2164 / 2163 / 2156 ms | 1740 / 1724 / 1718 ms | 365 ms | 83% / 79% (agent) | **1.25×** |
| top-k 4 + expert dim 8192 (2–3 runs) | 2923 / 2906 ms | 2046 / 2030 / 2031 ms | 381 ms | 87% / 81% (agent) | 1.43× |
| model width 4096 (L=8, 4, 2 variants) | device OOM (L=8); stuck-collective hardware error `NRT_EXEC_HW_ERR_COLLECTIVES` (L=4, L=2) | | | — | — |
| expert dim 8192 | device OOM (second buffer) | 3216 ms | 440 ms | 86% (agent) | — |

**Why tokens per step cannot change this share.** With all-gather-based AllToAllV over 224 experts the collective payload is
ws·T·k·D and the expert compute is T·k·D·EXDIM: both are linear in the token count T, so the collective share is T-invariant; only
the per-token compute (expert width, top-k, model width) moves it — which the table sweeps. Larger token counts are not reachable by either implementation: the AG-based AllToAllV allocates a ws×ws×cap×DM buffer
that grows linearly with tokens (2 GB per rank at the paper's shape; B=4 is 8 GB), so every B>1 / longer-sequence
configuration fails at load on this stack for both backends. We will therefore (i) state that the headline is the
collective-bound regime of 224-way expert parallelism, (ii) report the collective-component speedup (4016→2795 ms,
1.44×) as the quantity that transfers to other step compositions, and (iii) note the memory advantage of the deployed
strategy (one gathered buffer instead of two), which is why it still runs at expert dim 8192 where AG+RS does not.

## C — remaining numbered comments

**C1 (what the method buys; manual vs inferred model).** Across the archived Phase-3 runs, 125 candidates reached the
gates and 8 were trained end to end: the cost model plus gates avoid 117 of 125 training runs. What distinguishes this
from model-based autotuning is the search space (program rewrites above a closed API, with a correctness oracle) rather
than the model. Honest statement on the model: its structure and functional forms are manual; only its constants come
from probes — and, as corrected above, the submitted runs took them from a snapshot. **C3 (search-space size,
sensitivity).** The space before the seed library is the set of correct programs over 4 primitives × tensor rewrites,
not enumerable; we fixed K=5 strategies per enumeration and 2 refinement rounds and did not run a sensitivity sweep over
these; we will state this as a limitation. **C4 (FSDP fallback).** Correction: the appendix attributes the FSDP fallback to
the HBM gate; the archived logs show a harness error (`TypeError: missing argument 'torch'`) for all FSDP candidates in
those runs, and the fully-fused FSDP all-gather failed the correctness oracle; we will rewrite that paragraph.
**C8 (rejection categories).** Archived Phase-4: 125 candidates → 93 pass / 15 fail / 17 skipped; failures are training-
shape gate errors (RuntimeError/TypeError/IndexError on shape) and `collective_permute` at ws>64; on this SDK
`collective_permute` also deadlocks at ws=32. **C9.** Agreed; claims narrowed as stated below.

## B — comments

(i) *Is a better microbenchmark enough?* Our per-primitive harness at 224 ranks is that better microbenchmark — it uses
the real primitives, shapes and rank count — and it still inverts on AllToAllV (AG+T+RS 30.0 vs pack-and-gather 38.0 ms
per step in isolation; 4338 vs 3127 ms inside the OLMoE step). The inversion is a property of payload and graph context,
not of microbenchmark quality. (ii) *Regression instead of an LLM for calibration?* Yes for the constants: our corrected
model fits its collective curves by least squares from the probes, and we will say so; the LLM's contribution is Phase 3
(the rewrites a generic search does not find), not Phase 1. (iii) *Takeaway.* Three measured facts: the inversion; a
cost model faithful enough to rank at training scope (table above); and LLM-authored rewrites (bundling, the dxe
recomposition) that leak-free generic searches do not reach. We will expand `mark_step`, ICI/DCN and the other jargon
on first use.

## D — question and comments

*Portability.* The model's structure is hardware-specific in what it charges (dispatch runs, graph launches, NEFF loads,
HBM copies), and its constants are per-deployment; swapping TPU constants changes no ranking because rankings here are
driven by local terms, not network constants. We claim no TPU result and will say the structure's portability is
unvalidated. *Cluster-size experiment.* Repeated with the model fixed: 3-node Llama-7B per_mb 65.7 vs bundled 21.0 ms;
the bundled time is flat across 3→7 nodes (21.0 vs 21.1–22.1), so the agent strategy does not degrade — Table 9's 27.2
ms was a single run; the ratio compresses because the baseline speeds up. C2 is mitigated, not addressed — we will reword;
the H100 price sentence and the XLA phrasing will be removed.

## C-Q5 / C7 — Wall-clock, compile, repeats

Llama-7B 6000 steps from a cold cache: per_mb 593/596 s vs bundled 525/513 s → **1.145× net including compile**
(steady 71.9 vs 21.1–22.1 ms/step; paper 71.3/22.0 → 3.3×). Warm cache 595 vs 500 s (1.19×). Bundled fixed overhead ≈370 s,
cache-independent (trace/NEFF load/warm-up); break-even ≈4,100 steps (paper ≈4,400). We will label 3.24× steady-state.
On a second cluster (different instances, 2026-10-14) Llama-7B steady steps reproduce again: per_mb 73.2 vs bundled 22.5 ms
(3.26×). OLMoE headline repeats: archived n=7 baseline / n=6 agent (4391.7–4404.3 vs 3123–3140 ms) plus 3 new data seeds
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
`per_chunk_per_part_allgather` template fails its own reference; the microbatch-loop graph detector was gameable; the Uniform-A2A harness imports a runtime file the deployment step never writes (its shipped copy is a collective_permute ring that aborts at 224 ranks); and the Phase-5 codegen's uniform `alltoallv` helper is hard-coded to AG+transpose+RS, so a freshly generated AllToAllV runtime does not exercise the deployed algorithm on the training path (we measure deployed strategies through a wrapper that routes the helper to `evolved_alltoallv`).
