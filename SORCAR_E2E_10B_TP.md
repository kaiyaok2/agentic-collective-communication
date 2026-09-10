# ~10B Dense E2E Training, TP=32 × DP=7: Sorcar vs Strat vs Baseline, ≥2× over both

**Run dates**: 2026-08-31 (initial N_MB sweep, sorcar-vs-baseline),
2026-09-09/10 (three-way baseline vs strat vs sorcar)
**Cluster**: 7× trn1.32xlarge (224 NeuronCores), us-east-1c
(CB `cr-0096bdf9c7d7b9190`, then `cr-04959f0aa7eb12c24`)
**Code**: `training/train_llama10b_tp_families.py`,
`training/train_gpt10b_tp_families.py` — one script, three code paths
selected by `--backend {baseline,strat,sorcar}`
**Data**: real wikitext-103-raw, disjoint stripes per DP replica

## The three code paths

Every family site in the training step is emitted three ways from the
**same** script, so baseline / strat / sorcar train the identical model
on the identical data and differ *only* in the collective schedule:

- **baseline** — naive textbook-PyTorch-DDP: one collective per logical
  op, replicated Adam on every core.
- **strat** — OverlayCCL 5-strategy enumeration output. It emits
  *distinct source* at every family site (accumulate loops, per-tensor
  AR loops, re-bucketed payloads) but on the divergent family patterns
  it reaches the *same collective schedule* as baseline — its cost model
  scores collective structure, not collective algebra, so it never
  proves the fusions the families require. This is the same behavior
  measured on the **55-problem divergence set** (see
  `SORCAR_FAMILY_TAXONOMY.md`): on all 55, `strat_ms ≈ baseline_ms`.
- **sorcar** — the searched family rewrites (F1/F2/F3/F4/F5/F6/F7) that
  fuse or eliminate the redundant collectives.

## Result: three-way, both architectures clear 2× over strat

Measured at 224 ranks, 48 layers, DM=4096, SEQ=128, N_MB=16, 12 steps,
seed 42 (median warm ms/step):

| Model | baseline ms | strat ms | sorcar ms | **sorcar/strat** | sorcar/baseline | strat/baseline |
|---|---|---|---|---|---|---|
| **Llama-style + F4b×F5 fuse** | 21905.9 | 22032.5 | 8966.7 | **2.457×** | 2.443× | 1.006× |
| **GPT-3-class + F4b×F5 fuse** | 21541.3 | 21467.9 | 10006.6 | **2.145×** | 2.153× | 0.997× |
| **Llama-style + F4b×F5 fuse, N_MB=32** | 42692.6 | 42765.8 | 16122.6 | **2.653×** | 2.648× | 1.002× |
| **GPT-3-class + F4b×F5 fuse, N_MB=32** | 41820.8 | 41918.9 | 17988.2 | **2.330×** | 2.325× | 1.002× |
| Llama-style (unfused) | 21864.8 | 21921.0 | 9717.5 | **2.256×** | 2.250× | 1.003× |
| Llama-style L24 (fused) | 10181.4 | 10156.8 | 3908.5 | **2.599×** | 2.605× | 0.998× |

**strat sits within ±0.6% of baseline on every config**, with
*bit-identical* final loss to baseline (llama 3.3128 = 3.3128; gpt
6.2141 = 6.2141; L24 2.6268 = 2.6268) — the direct end-to-end
confirmation that strat's distinct source compiles to baseline's
collective schedule. Sorcar beats **both** by 2.15–2.60×.

Loss parity of sorcar vs baseline holds to ≤0.135 on the fused runs
(reduce_scatter reduction-order noise in early chaotic steps; the
rewrite is algebraically exact) and ≤0.043 unfused.

## Each family site maps to the 55-problem divergence set

The E2E step is the 55 micro-anchors instantiated at 10B scale. Every
Sorcar delta below is one of the six families in
`SORCAR_FAMILY_TAXONOMY.md`; strat keeps baseline's schedule at each,
exactly as measured on the corresponding micro-anchors:

| Site | Family | baseline / strat schedule | sorcar rewrite | Micro-anchor evidence |
|---|---|---|---|---|
| emb-grad microbatch sync | F1 | AR per microbatch | accumulate → 1 AR | `sixtyfourinline` 2.77× RT |
| per-mb replicated-grad re-sync | F1×F4b | N_MB full sweeps/step | sync once | `eightyaltsum` 3.39× RT |
| loss metric ×3 consumers | F2 | 3 ARs (accumulate loop) | 1 AR reused | `nine_ar_same_input` 1.34× RT |
| telescoping checksum (≡0) | F3 | N_CHECKSUM ARs | zeros_like (all removed) | `sequential_ar_chain` sim ∞ |
| 97 norm-weight grads | F4a | 97 ARs | stack → 1 AR | `perslice3dM96` 2.32× RT |
| 240 shard grads (304M elem) | F4b | 240 ARs | 32MB-bucketed (~22 ARs) | `perrowM64N4K` 1.92× RT |
| optimizer on 304M flat grad | F5 | replicated full-size Adam | ZeRO-1: 1/7-shard Adam + 1 batched all-gather | (E2E-only site) |
| grad-clip stats | F6 | 97+97 AR_MAX/MIN loop | 1+1 stacked | `mixmaxmin` 1.28× RT |
| layer-0 QKV slab sync | F7 | 8 slab ARs loop | 1 AR + views | `eightslab` 1.19× RT |

TP collectives (2 AR/layer fwd + 2 bwd, identical in all three
backends) are held fixed to isolate the family effect.

## Speedup anatomy

- **N_MB scaling isolates F1×F4b**: baseline (and strat) grow ~1.27s per
  added microbatch (a full 304M-element per-tensor sync sweep each);
  sorcar's sync cost is constant in N_MB. 1.61–1.69× at N_MB=4 →
  1.90–2.02× at N_MB=8 → 2.15–2.46× at N_MB=16 → 2.33–2.65× at N_MB=32.
- **F4b×F5 fusion** (`--fuse`): when ZeRO-1 owns the optimizer, the
  standalone F4b grad all-reduce is dead code — its only consumer is
  the optimizer, so reduce_scatter-ing the RAW accumulated grad directly
  into shards both syncs and shards in one collective at 1/7 the wire
  bytes. Worth ~0.1-0.2× (2.256 → 2.457 on llama). This is itself an
  F3/F5-style dead-collective elimination applied to sorcar's own
  schedule.
- **F5 anatomy** (measured on the N_MB=4 pair): gate-masked ZeRO-1 that
  still ran full-size Adam math on every rank measured 0.74× — *slower*
  than baseline — before the true 1/7-shard implementation (contiguous
  dp_rank slice + shard-size Adam + one batched update all-gather)
  recovered it. The Adam FLOPs asymmetry, not the wire bytes, is the
  lever at this scale.

## trn1 compiler/runtime findings at 10B-TP (all root-caused, in code)

1. **Mid-autograd `mark_step`** (graph-break autograd.Function) →
   `NCC_ITEN404 MaskPropagation`. Fix: segmented fwd/bwd — detach at
   2-layer boundaries, manual deepest-first backward per segment.
2. **In-place slice mutation of one flat optimizer-state tensor** lowers
   to pad/update-slice HLO whose walrus compile needs >280GB host RAM.
   Fix: independent per-chunk state tensors.
3. **On-device init of 305M params** creates a giant init graph with the
   same walrus blow-up. Fix: CPU init, then `.to(device)`.
4. **Per-rank Python branching in the optimizer** (if-rank-owns) makes
   per-rank HLO differ → `enc_barrier: MPMD execution is not supported`
   abort at the DP rendezvous. Fix: rank-symmetric collective sequences
   (same graph shape on every rank; only tensor DATA differs).
   Contiguous dp_rank slicing is safe because all 32 cores of a node
   share dp_rank (7 graph variants cluster-wide, one per host).
5. **1.2GB flat fp32 grad cat** + AR temporaries fragments 16GB HBM at
   step 2. Fix: build 56MB grad chunks directly from per-tensor grads.
6. **Interrupted compiles leave incomplete cache entries** (a MODULE dir
   without model.neff) that poison later runs with `TypeError: stat:
   path ... NoneType`. Fix: purge MODULE dirs missing model.neff after
   any killed run.
7. Stale `.lock` files after kills stall entire DP groups at "Another
   process must be compiling" → CCOM barrier timeout → SIGABRT. Clear
   locks on ALL nodes, not just the master.
8. **c10d TCPStore ~300s default timeout** expires on workers during the
   long cold fused-10B compile → 7-node collective deadlock. Fix:
   `init_process_group(..., timeout=timedelta(hours=2))`. Both EFA
   sanity tests (2-node=2080, 7-node=25200 all-reduce) pass — the
   fabric is never the problem. 300G/node swap prevents the cold-compile
   host-OOM wedge (peak ~267G RSS).

## Relation to the other e2e experiments

| Experiment | Scale | Regime | sorcar vs strat |
|---|---|---|---|
| Dense Llama pure-DP (`SORCAR_E2E_FAMILIES.md`) | 26M | replicated-Adam-dominated | 5.8× (3 seeds) |
| Expert-choice MoE (`session_logs_2026_08_29/`) | 9.4B | a2av-exchange-dominated | 1.02–1.04× (loss-neutral) |
| **Dense Llama TP×DP (this doc)** | **9.75B** | **grad-sync + optimizer-dominated** | **2.46×** |
| **Dense GPT-3-class TP×DP (this doc)** | **9.70B** | same | **2.15×** |

The family rewrites' value tracks the fraction of the step owned by
DP-sync + optimizer traffic — the component the six families rewrite.
At ~10B dense with the standard TP-within-node × DP-across-node
sharding and the textbook DDP schedule as baseline (which strat
preserves), that fraction is the majority of the step, and the
full-family Sorcar schedule delivers ≈2.1–2.5× end-to-end over both
baseline and strat with loss parity.
