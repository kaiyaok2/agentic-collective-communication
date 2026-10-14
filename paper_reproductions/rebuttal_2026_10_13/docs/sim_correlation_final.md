# Simulator vs 224-rank hardware — final numbers (rebuttal, 2026-10-14 00:30 UTC; arm v11c)

Pool: 71 candidates over the 8 problems (seed-library templates, every structural-grammar variant, archived LLM strategies
incl. non-deployed ones, the deployed runtimes, developer baselines, the leak-free search's finds, and 6 AllToAllV
seed-library helper variants), each measured 3x at 224-rank training scope with the per-problem harness (median of 130
steady steps; repeat sd <= 2% on 55/58 measured candidates; 10 candidates fail the hardware gate and are excluded).
Simulator = the shipped cost model with the fixes in sim_v4_changes.md (measured probe constants, real harness shapes,
bytes-moved + volume charging, measured collective curves, fusion-aware dispatch runs, XLA concat-of-slices rule,
scope-aware graph boundaries). One configuration for all problems; no per-problem tuning.

| Problem | n | Spearman rho (sim vs measured step) | sim top-1 = HW best? | regret of sim pick |
|---|---|---|---|---|
| AllToAllV (per-primitive scope) | 8 | **0.56** | no (AG+T+RS inline 30.0 ms vs vectorized AG+slice 24.2) | 23.8% |
| Distributed CE | 8 | **0.56** | yes (three_ar 8.31 ms) | 0 |
| FSDP prefetch | 7 | **0.81** | yes (fully bundled AG 13.37) | 0 |
| Layer-block AR | 7 | **0.90** | yes (stacked single AR 17.20) | 0 |
| PP send/recv | 6 | **0.89** | yes (stacked masked AR 111.4) | 0 |
| Ring KV | 6 | **0.58** | top-2 (joint 5.14 vs per-slot 4.96; HW gap within repeat noise) | 3.7% |
| TP MLP | 16 | **0.90** | within 0.3% (per-layer flatten 10.88 vs per-layer stack 10.85) | 0.3% |
| Uniform A2A | 1 measured (+3 seed candidates running) | pending | | |
Pooled Spearman on speedup-vs-baseline ratios (n=58): 0.60; direction agreement where the sim predicts a change: 22/25.

Before the fixes (shipped static table, toy shapes): TP MLP 0.74, Layer-block 0.90, PP 0.93, FSDP 0.60, Ring KV 0.56,
**dxe -0.77, AllToAllV -0.32**; the dxe failure was structural (4 KB sandbox all-gather x10 vs 147 KB in / 33 MB out on
hardware, flat 29 us per local op), not a constant.

## Cross-scope inversion reproduced by the simulator (AllToAllV)
Same candidates, same simulator, only the payload scope changes:
- per-primitive harness scope (CAP*DM = 1024 elements per pair): sim AG+T+RS 19972 < vectorized AG+slice 29045 <
  AG+slice loop+cat 40760; hardware 30.0 / 24.2 / 38.0 ms -> AG+T+RS beats the paper's loop+cat pack-and-gather at this
  scope (both sim and HW).
- OLMoE step scope (cap*DM ~ 20480 elements per pair, 224 experts): sim vectorized AG+slice 131603 < loop+cat 143416 <
  AG+T+RS 271144; hardware pack-and-gather 3127 ms vs AG+T+RS 4338 ms (1.39x) -> pack-and-gather wins.
The inversion is driven by the AG+RS form's second collective over the full ws x ws x cap x DM buffer plus its dense
permute copy, which scale with payload, versus pack-and-gather's slice/cat extraction whose per-slice cost does not.

## Leak-free non-LLM search vs the simulator (C-Q2, reward-hack check)
Against the shipped simulator, random/GA/beam "found" 60x sim wins on PP/TP MLP/FSDP: the baseline loop unrolled by two
(`range(0, M, 2)`), which the old graph-boundary detector did not recognize. Hardware: TP MLP 11.65-11.90 ms vs baseline
11.75 (1.00x), FSDP 15.25 vs 14.5-15.3 (1.00x), PP variants crash (harness-incompatible CPU tensors). The fixed detector
(any iterable mentioning a microbatch quantity) and harness-scope scoring remove the exploit; the search is being re-run
against the fixed simulator (results appended below when done).
