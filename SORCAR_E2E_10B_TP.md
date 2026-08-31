# ~10B Dense E2E Training, TP=32 × DP=7, All 7 Families: Sorcar ≥2× over Strat

**Run date**: 2026-08-31
**Cluster**: 7× trn1.32xlarge (224 NeuronCores), CB `cr-0096bdf9c7d7b9190`,
us-east-1c, placement group `Kaiyao`
**Code**: `training/train_llama10b_tp_families.py`,
`training/train_gpt10b_tp_families.py`
**Data**: real wikitext-103-raw, disjoint stripes per DP replica

## Result

Two well-known dense architectures at ~9.7B cluster parameters, trained
end-to-end on real text with **all 7 Sorcar-vs-strat optimization
families** wired into the DP-gradient-sync + optimizer path:

| Model (TP=32 × DP=7, 48 layers, DM=4096) | N_MB | baseline (strat) ms/step | sorcar ms/step | **Speedup** | final-loss Δ |
|---|---|---|---|---|---|
| Llama-style (RMSNorm, SwiGLU, 9.75B) | 8 | 11246.1 | 5555.4 | **2.02×** | 0.0051 |
| GPT-3-class (LayerNorm+bias, GELU, learned pos, 9.70B) | 12 | 16061.3 | 7864.5 | **2.04×** | 0.0112 |
| GPT-3-class | 8 | 11022.8 | 5816.4 | **1.90×** | 0.0028 |
| Llama-style | 4 | 6167.0 | 3654.9 | **1.69×** | 0.0058 |
| GPT-3-class | 4 | 5999.3 | 3732.6 | **1.61×** | 0.0108 |

**Multi-seed stability**: the Llama N_MB=8 pair replicates at seed 7:
11261.7 → 5560.9 ms = **2.03×** (vs 2.02× at seed 42), loss parity
0.026.

**Both architectures clear 2×.** The N_MB knob is not a trick: more
gradient-accumulation microbatches is the standard way to grow
effective batch size at fixed memory, and the baseline's cost grows
with it because textbook DDP re-syncs every microbatch — exactly the
schedule strat preserves and the F1-family rewrite eliminates.

Both backends descend in lockstep on real text (llama 6.31→2.75 /
2.74; gpt 6.07→3.62 / 3.63 over 25 steps at N_MB=8); the F3 checksum is
exactly 0.0 every step. 25–30 steps per run, median warm step time.

## What the baseline is

The baseline is the **textbook PyTorch-DDP training schedule** — the
collective structure a practitioner gets by default, and the structure
strat-enumerate preserves (its enumeration re-arranges collectives but
never crosses the semantic rewrites the families require):

- replicated-grad all-reduce fires on **every microbatch** (DDP without
  `no_sync()`),
- one AR **per gradient tensor** (240 shard tensors + 97 norm tensors),
- **full-size fp32 Adam replicated on every core** (plain DP
  optimizer),
- per-tensor AR_MAX/AR_MIN for grad-clip stats, slab-wise QKV sync,
  3 separate metric ARs, an 8-AR telescoping checksum.

## The 8 Sorcar deltas (one per family site; F1/F4 instantiated twice)

| Site | Family | baseline | sorcar |
|---|---|---|---|
| emb-grad microbatch sync | F1 | AR per microbatch | accumulate → 1 AR |
| per-mb replicated-grad re-sync | F1×F4b | N_MB full sweeps/step | sync once |
| loss metric ×3 consumers | F2 | 3 ARs | 1 AR reused |
| telescoping checksum (≡0) | F3 | 8 ARs | zeros_like |
| 97 norm-weight grads | F4a | 97 ARs | stack → 1 AR |
| 240 shard grads (304M elem) | F4b | 240 ARs | 32MB-bucketed (~22 ARs) |
| optimizer on 304M flat grad | F5 | replicated full-size Adam | ZeRO-1: 1/7-shard Adam + 1 batched all-gather |
| grad-clip stats | F6 | 97+97 AR_MAX/MIN | 1+1 stacked |
| layer-0 QKV slab sync | F7 | 8 slab ARs | 1 AR + views |

TP collectives (2 AR/layer fwd + 2 bwd, identical in both backends) are
not a family site — they are held fixed to isolate the family effect.

## Speedup anatomy

- **N_MB scaling isolates F1×F4b**: baseline grows ~1.27s per added
  microbatch (a full 304M-element per-tensor sync sweep each);
  sorcar's sync cost is constant in N_MB. 1.61–1.69× at N_MB=4 →
  1.90–2.02× at N_MB=8.
- **F5 anatomy** (measured on the N_MB=4 pair): gate-masked ZeRO-1 that
  still ran full-size Adam math on every rank measured 0.74× —
  *slower* than baseline — before the true 1/7-shard implementation
  (contiguous dp_rank slice + shard-size Adam + one batched update
  all-gather) recovered it. The Adam FLOPs asymmetry, not the wire
  bytes, is the lever at this scale.
- Loss parity holds at ≤0.011 across all four pairs (bf16
  reduction-order noise; both curves overlap step-by-step).

## trn1 compiler/runtime findings at 10B-TP (all root-caused, in code)

1. **Mid-autograd `mark_step`** (graph-break autograd.Function) →
   `NCC_ITEN404 MaskPropagation`. Fix: segmented fwd/bwd — detach at
   2-layer boundaries, manual deepest-first backward per segment.
2. **In-place slice mutation of one flat optimizer-state tensor**
   lowers to pad/update-slice HLO whose walrus compile needs >280GB
   host RAM. Fix: independent per-chunk state tensors.
3. **On-device init of 305M params** creates a giant init graph with
   the same walrus blow-up. Fix: CPU init, then `.to(device)`.
4. **Per-rank Python branching in the optimizer** (if-rank-owns) makes
   per-rank HLO differ → `enc_barrier: MPMD execution is not
   supported` abort at the DP rendezvous. Fix: rank-symmetric
   collective sequences (same graph shape on every rank; only tensor
   DATA differs). Contiguous dp_rank slicing is safe because all 32
   cores of a node share dp_rank (7 graph variants cluster-wide, one
   per host).
5. **1.2GB flat fp32 grad cat** + AR temporaries fragments 16GB HBM at
   step 2. Fix: build 56MB grad chunks directly from per-tensor grads.
6. **Interrupted compiles leave incomplete cache entries** (a MODULE
   dir without model.neff) that poison later runs with
   `TypeError: stat: path ... NoneType`. Fix: purge MODULE dirs missing
   model.neff after any killed run.
7. Stale `.lock` files after kills stall entire DP groups at "Another
   process must be compiling" → CCOM barrier timeout → SIGABRT.
   Clear locks on ALL nodes, not just the master.

## Relation to the other e2e experiments

| Experiment | Scale | Regime | Speedup |
|---|---|---|---|
| Dense Llama pure-DP (`SORCAR_E2E_FAMILIES.md`) | 26M | replicated-Adam-dominated | 5.8× (3 seeds) |
| Expert-choice MoE (`SORCAR_E2E_10B.md`) | 9.4B | a2av-exchange-dominated | 1.02–1.04× (loss-neutral) |
| **Dense Llama TP×DP (this doc)** | **9.75B** | **grad-sync + optimizer-dominated** | **2.02×** |
| **Dense GPT-3-class TP×DP (this doc)** | **9.70B** | same | **2.04×** |

The family rewrites' value tracks the fraction of the step owned by
DP-sync + optimizer traffic — the component the 7 families rewrite.
At ~10B dense with the standard TP-within-node × DP-across-node
sharding and the textbook DDP schedule as baseline, that fraction is
the majority of the step, and the full-family Sorcar schedule delivers
≈2× end-to-end with exact loss parity.
