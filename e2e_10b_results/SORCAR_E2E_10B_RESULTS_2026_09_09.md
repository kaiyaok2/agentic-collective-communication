# Sorcar vs baseline E2E — 10B TP, 7-node trn1 (2026-09-09)

WS=224 (TP=32 x DP=7), 48 layers, DM=4096, N_MB=16, --fuse (F4b+F5), seq=128,
byte-level vocab (256). backend=baseline is the strat-enum / textbook-DDP
schedule; backend=sorcar applies the family collective rewrites. warm compile
cache, steps=12, warm median of steps 3-11.

## Headline: multi-seed x both-arch

| arch  | seed | baseline ms/step | sorcar ms/step | speedup | final-loss delta |
|-------|------|------------------|----------------|---------|------------------|
| Llama | 42   | 21580.3          | 8675.9         | 2.49x   | 0.008            |
| Llama | 43   | 21614.3          | 8679.8         | 2.49x   | 0.032            |
| GPT   | 42   | 21278.7          | 9655.6         | 2.20x   | 0.045            |
| GPT   | 43   | 21143.6          | 9532.9         | 2.22x   | 0.0003           |

- Both architectures clear the >=2.0x goal; seed-robust (Llama 2.49/2.49, GPT
  2.20/2.22). GPT is the held-out arch (rewrites tuned against Llama).
- Loss parity CLEAN: step-by-step trajectories track; final deltas within the
  per-step data-shuffle noise. checksum F3 gate == 0 every step.
- Params: Llama 9.749B, GPT 9.700B.

## N_MB scaling (Llama, seed 42) — the mechanism
| N_MB | baseline ms | sorcar ms | speedup |
|------|-------------|-----------|---------|
| 4    | 6186.5      | 3699.3    | 1.67x   |
| 16   | 21580.3     | 8675.9    | 2.49x   |

baseline grows ~near-linearly with N_MB (per-microbatch grad resync); sorcar
stays much flatter (F1xF4b fuses the per-mb collectives). The speedup therefore
rises with N_MB.

Raw per-run JSON: all_results.jsonl. Per-run master logs: logs/.
