# E2E #277 — Ideas & Rationale

**Goal:** E2E GPT+Llama training where **Sorcar > OverlayCCL** (measured, not
just > baseline), aim **≥2.0×** end-to-end at ~10B TP=32×DP=7, real text, loss
parity. Then ablations + research loop for further gains.

## Baseline understanding (established this session)
- In the E2E harness, `--backend baseline` **== strat** (strat-enum's outcome
  is the textbook-DDP schedule; confirmed at sim AND warm-RT: strat_ms ≈
  baseline_ms within noise on all 56 divergent problems).
- OverlayCCL sim **== current sim** (byte-identical minus 5 pow methods;
  0/143 scoring divergences). So "Sorcar-on-OverlayCCL" ≡ "Sorcar-on-current"
  at search level. The meaningful E2E delta is **Sorcar family-rewrites vs the
  strat/baseline schedule**.
- Prior E2E (2026-08-31, 2026-09-07): 2.02–2.47× llama, 2.04–2.22× gpt,
  replicated on 2 physical clusters. #277 = reproduce cleanly on THIS
  hardware + push higher via the research loop.
- RT-confirmed strongest family representatives (this session, warm-cache RT):
  - F1i many-AR-inline: eightyaltsum 3.39×, sixtyfourinline 2.77× (scales with
    AR count, monotone 1.54→3.39×)
  - F7 slab: perslice3dM96 2.32×
  - F4 per-row: perrowM64N4K 1.92×
  - F3 algebraic-zero: ten_ar_alt_sign_zero 20.5× vs baseline

## The 8 family sites already in the harness (baseline vs sorcar)
F1 emb-grad microbatch sync | F1×F4b per-mb resync | F2 loss metric ×3 |
F3 checksum≡0 | F4a 97 norm grads | F4b 240 shard grads (bucketed) |
F5 optimizer (replicated Adam → ZeRO-1) | F6 grad-clip stats | F7 QKV slab.

## Idea backlog (to be judged, implemented, measured)
Seed ideas — refine after baseline profile + web search:

1. **Reproduce headline** (N_MB=16 fused) on current+new CB, 3 seeds, both
   archs. Establishes the ≥2.0× claim on this hardware. [MUST-DO FIRST]
2. **N_MB scaling curve** — confirm F1×F4b isolation (baseline grows ~1.27s/mb,
   sorcar flat). Sweep nmb ∈ {4,8,12,16}.
3. **F4b bucket-size sweep** — the 32MB bucket cap is a tuned constant; try
   16/24/32/48/64MB to find the dispatch-vs-payload optimum at 224 ranks.
4. **F5 fusion depth** — reduce_scatter-of-raw-grad (`--fuse`) gave +0.1-0.2×.
   Can we fuse F4a norm grads into the same reduce_scatter? (cross-family
   F4a×F5).
5. **F7 generalization** — apply slab-fusion to ALL linear layers' grads, not
   just layer-0 QKV (currently 1 site). Every layer has a QKV slab.
6. **Overlap collectives with compute** — async reduce_scatter during backward
   of the next segment (bucketed pipelining, cf simulator cost-model memory).
7. **Larger effective model** — does the ≥2× hold at DM=5120 / 60 layers
   (bigger F5 optimizer term)? Or does HBM force smaller N_MB?

## Held-out / generalization plan
- Prove ≥2.0× on BOTH archs (llama + gpt) — gpt is the held-out arch relative
  to llama-tuned rewrites.
- Multi-seed (≥2 seeds) for step-time stability + loss-parity.
- Loss parity ≤~0.05 final (unfused) as the exactness gate.
