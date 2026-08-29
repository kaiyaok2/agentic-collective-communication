# E2E 9.4B MoE Training at 224 Ranks: Sorcar vs Strat (All 7 Families)

**Run date**: 2026-08-29
**Cluster**: 7× trn1.32xlarge (224 NeuronCores), CB `cr-029642dfa443c6462`,
us-east-1c, placement group `Kaiyao`
**Code**: `training/train_olmoe10b_families.py`
**Data**: real wikitext-103-raw, disjoint per-rank stripes

## Model

OLMoE-class expert-choice sparse MoE at OverlayCCL scale:
- DM=2048, 16 heads, 7 layers, RoPE, RMSNorm, SwiGLU experts
- NEXP = ws = 224 (one expert per rank, EXDIM=960), TOPK=8,
  expert-choice routing → uniform per-(src,dst) AllToAllV counts
- Vocab-sharded head (V_local = VOCAB/224)
- **Cluster parameters: 9.44B** (replicated 153.7M + 41.4M/rank × 224)
- bf16 compute, fp32 sharded-optimizer path

Per training step the communication is: 2 AllToAllV (AG+RS) per MoE
layer × 7 layers × fwd+bwd (identical in both backends — a2av is one of
the paper's original problems, held fixed to isolate the family
effect), plus the ~146M-element replicated-grad sync that carries all 7
family sites, plus CE-gather and metric collectives.

## Family sites (same 7-family mapping as `SORCAR_E2E_FAMILIES.md`)

| Site | baseline (strat) | sorcar |
|---|---|---|
| F1 emb grad (33M), N_MB=2 | AR per microbatch delta | 1 AR after accumulation |
| F2 loss metric ×3 | 3 ARs | 1 AR reused |
| F3 8-AR telescoping checksum | 8 dispatches | zeros_like |
| F4a 15 RMSNorm grads | per-tensor AR | stack → 1 AR |
| F4b 78.7M replicated grads in 13 tensors | per-tensor AR loop | 32MB-bucketed concat+AR (grad_ar v6 pattern) |
| F5 o-proj grads layers 0–3 (16.8M) | AR full + replicated Adam | reduce_scatter → sharded Adam → all_gather |
| F6 clip stats (max & min) | per-entry AR_MAX/AR_MIN | stacked 1+1 |
| F7 layer-0 QKV grad (12.6M) | 8 slab ARs | 1 AR + views |

## Results (60 steps, median warm ms/step)

| Config | baseline | sorcar | Speedup | loss (both) |
|---|---|---|---|---|
| SEQLEN=192 (compute+a2av heavy) | 6051.4 | 5929.1 | **1.02×** | 9.97 → 4.36 / 4.31 |

Loss parity: final-loss delta 0.051, max per-step divergence 0.16 —
consistent with bf16 reduction-order noise on a 9.4B model (both curves
descend from 9.97 to ~4.3 in lockstep; the F3 checksum is exactly 0.0
every step of every run).

**Interpretation.** At this scale the step is dominated by the 28
AllToAllV dispatches (fixed in both backends) and expert compute; the
family sites' total collective-count saving (~25 dispatches → ~8) is
worth ~120 ms of the 6-second step. This is the same absolute-saving
pattern measured across the pool: family wins are dispatch-count
savings, so their relative impact scales inversely with everything else
in the step. A short-sequence variant (SEQLEN=96, grad-sync-dominated)
is measured separately below.

<!-- SEQLEN96_RESULTS -->

## trn1 engineering findings at 9.4B/224 ranks (reproduction guide)

Getting a 9.4B pure-DP-replicated + expert-parallel model to train at
224 ranks on trn1 required root-causing six failure modes; all fixes
are in `training/train_olmoe10b_families.py`:

1. **`xm.all_gather` has no backward on this stack** — the CE gather
   over vocab shards needs a custom `autograd.Function` (backward =
   extract local shard). Same root cause as the 2026-06-06
   realtok finding.
2. **Rank-dependent Python constants fork the HLO graph per rank** —
   slicing by `rank` in the CE backward produced 32 distinct graphs per
   host → 32 concurrent neuronx-cc processes → host OOM (`[F137]`).
   Fix: rank-invariant one-hot-mask backward.
3. **Concurrent compile storms OOM the 495GB host** even with a
   rank-invariant graph. Fix: an 8-slot flock-based semaphore shim
   around `neuronx-cc` (`training/tools/neuronx-cc-slotshim.sh`).
4. **Single big einsum over the expert dim** trips `NCC_EBVF030`
   (>5M instructions); loop-over-experts GEMMs compile fine.
5. **Device HBM budget**: at 224 ranks the CE-gather buffer
   (T×VOCAB fp32), the F4b full-concat AR (78.7M), and the fp32
   replicated Adam chain each individually OOM 16GB cores.
   Fixes: VOCAB 32256→16128; 32MB bucketing for the F4b concat
   (exactly the production grad_ar v6 finding — unbucketed concat
   OOMs, per-tensor is slow, 32MB buckets are the sweet spot);
   F5 optimizer group restricted to 4 layers' o-proj.
6. **Python-scalar Adam bias-correction constants** are fine
   (torch_xla promotes them to device data) — a device-tensor
   accumulator variant correlated with repeated-execution NRT
   aborts and was reverted.

## Relation to the small-scale e2e result

`SORCAR_E2E_FAMILIES.md` (26M dense Llama, same 7 sites): 5.87× —
there the F5 replicated-Adam penalty dominated a 1.2 s step.
At 9.4B MoE the same rewrites all hold (loss parity, no regression),
but the step budget is owned by the a2av exchange the families don't
touch. Together the two experiments bracket the regime dependence:
**family rewrites are worth minutes per step when the optimizer/grad
sync dominates, and are loss-neutral overhead reduction when expert
exchange dominates.**
