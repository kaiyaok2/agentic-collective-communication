# Explored Ideas Log (#277)

Format: idea | aspect | metric(before→after) | verdict(keep/failed) | notes


## [2026-09-09 08:32 UTC] Idea 1a — baseline smoke reproduce (Llama, N_MB=4, cold->warm)
- config: llama10b_tp, WS=224 TP=32 DP=7, 48 layers, DM=4096, 9.749B params
- backend=baseline, steps=8, seed=42
- RESULT: median 6186.5 ms/step (warm steps 3-7 ~6180-6245), loss 6.33->5.02, chk=0
- STATUS: pipeline VALIDATED after wiki.train.raw restaged on all 7 nodes.
  Data-missing bug (FileNotFoundError) FIXED. Baseline anchor established.
- next: sorcar same config for speedup ratio; then N_MB=16 fused headline.

## [2026-09-09 08:43 UTC] Idea 1b — sorcar smoke (Llama, N_MB=4)
- backend=sorcar, same config, seed=42, steps=8
- RESULT: median 3699.3 ms/step (warm steps 3-7 ~3644-3716), loss 6.33->4.96, chk=0
- baseline 6186.5 / sorcar 3699.3 = 1.67x at N_MB=4
- loss parity CLEAN: step-by-step trajectories track (6.33/6.33, 4.98/5.00 ...),
  final delta 0.068 within data-shuffle noise.
- INTERPRETATION: 1.67x at N_MB=4. Headline is N_MB=16 fused where F1xF4b
  per-mb resync compounds (baseline ~linear in N_MB, sorcar flat). KEEP.
- next: N_MB=16 --fuse both backends -> expect >=2.0x.

## [2026-09-09 08:51 UTC] Idea 1c — N_MB=16 fused headline (Llama)
- baseline --nmb 16 --fuse, steps=12, seed=42
- RESULT: median 21580.3 ms/step (very stable steps 3-8 ~21550-21760), loss 6.29->3.67
- N_MB scaling: baseline 6186 (nmb4) -> 21580 (nmb16) ~= 3.49x for 4x mb; confirms
  near-linear F1xF4b per-mb resync growth.
- sorcar --nmb 16 --fuse running now for the headline ratio.

## [2026-09-09 08:57 UTC] Idea 1 HEADLINE ACHIEVED — Llama N_MB=16 fused
- baseline 21580.3 ms/step (loss->3.669) | sorcar 8675.9 ms/step (loss->3.661)
- **2.49x** >= 2.0x GOAL MET. Loss parity CLEAN delta 0.008, trajectories track.
- Replicates 2026-08-31 (2.02-2.47x llama) on recovered HW. KEEP as anchor.
- next: GPT (held-out arch) N_MB=16 fused for generalization proof.

## [2026-09-09 09:07 UTC] GPT (held-out arch) N_MB=16 fused — baseline
- baseline gpt10b_tp --nmb 16 --fuse: median 21278.7 ms/step, loss 6.03->5.59, 9.700B
- sorcar running for the generalization ratio.

## [2026-09-09 09:13 UTC] GPT sorcar N_MB=16 fused — GENERALIZATION CONFIRMED
- baseline 21278.7 / sorcar 9655.6 = **2.20x** (held-out arch), loss 5.592 vs 5.547 (delta 0.045 CLEAN)
- HEADLINE COMPLETE both archs >= 2.0x:
    Llama 2.49x (delta 0.008) | GPT 2.20x (delta 0.045)
- Idea 1 (reproduce headline) DONE + generalization proven on held-out arch.
- next: multi-seed stability (seed 43,44) to confirm not seed-luck, then ablations.

## [2026-09-09 09:23 UTC] Multi-seed stability (seed 43) — Llama
- baseline s43 21614.3 / sorcar s43 8679.8 = 2.49x (== s42 2.49x). loss 4.336 vs 4.368 (delta 0.032)
- Llama seed-robust: s42 2.49x, s43 2.49x. NOT seed-luck.
- next: GPT seed 43 pair.

## [2026-09-09 09:35 UTC] #277 GOAL FULLY MET — multi-seed x both-arch matrix
| arch | seed | base ms | sorcar ms | speedup | loss delta |
|------|------|---------|-----------|---------|-----------|
| Llama | 42 | 21580 | 8676 | 2.49x | 0.008 |
| Llama | 43 | 21614 | 8680 | 2.49x | 0.032 |
| GPT   | 42 | 21279 | 9656 | 2.20x | 0.045 |
| GPT   | 43 | 21144 | 9533 | 2.22x | 0.0003 |
- Both archs >= 2.0x, SEED-ROBUST (Llama 2.49/2.49, GPT 2.20/2.22), loss parity CLEAN.
- Held-out generalization (GPT) + multi-seed stability both satisfied. Idea 1 CLOSED.
- Per methodology: goal met with generalization check. Move to ABLATIONS next.
