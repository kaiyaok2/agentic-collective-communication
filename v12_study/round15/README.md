# Round 15: 10 challenging comm problems (real optimization tension)

## Motivation

Round 12's 10 comm problems (`sum_across_ranks_comm`, etc.) all had ONE
obvious solution: `xm.all_reduce(SUM, x)` or similar single-collective. Kiss
and strat both converged there. Round 13 RT confirmed ~5.1ms tie on all
10 comm problems.

Not a useful tie-breaker. Real challenges — like OverlayCCL's grad_ar
naive vs bucketed — have multiple plausible strategies with non-obvious
HW/sim trade-offs. Round 15 adds 10 such problems.

## Design principle

Each problem has AT LEAST TWO plausible strategies where the "optimal"
choice depends on non-obvious factors: dispatch overhead vs bandwidth,
memory-copy vs launch amortization, single vs multiple collectives, etc.
Similar to how grad_ar breaks kiss v11 (naive per-tensor 8× AR) but strat
finds bucketed cat+AR+split (7.4× faster in sim).

## The 10 challenging problems

1. **`multi_grad_ar_chal`** — 8 mixed-size gradients, sum across ranks.
   Naive: 8 per-tensor AR. Smart: cat+AR+split. Same shape as `grad_ar`
   but different sizes/counts.

2. **`ag_then_rs_chal`** — (K, W) local matrix, sum + column-slice per rank.
   Options: AR full then slice vs reduce_scatter along dim 1 (W-1× less
   data moved).

3. **`multi_layer_ar_chal`** — 4-layer gradient AR with local compute
   between. Options: 4 sequential ARs vs bucketed AR + split. Amortization
   (α1, α2, α3 tiers) makes back-to-back cheaper.

4. **`double_reduction_chal`** — need BOTH sum and max across ranks. Options:
   2 ARs (SUM + MAX) vs single 2x-payload AR with local extremes.

5. **`hierarchical_ar_chal`** — 2-node topology. Options: flat 64-rank AR
   (inter-node BW dominates) vs intra-node AR first + inter-node exchange
   (NeuronLink is ~15× faster than EFA).

6. **`sparse_topk_chal`** — global top-K. Options: all_gather full N vs
   local top-K then all_gather K (world*K bytes vs world*N bytes, but
   extra sort ops).

7. **`weighted_mean_chal`** — weighted mean needing numerator + denominator.
   Options: 2 ARs, or 1 AR of cat(num, den) then split.

8. **`layered_matmul_chal`** — Q @ K^T where K distributed. Options: all_gather K
   then matmul, or reduce_scatter matmul chunks.

9. **`mixed_precision_ar_chal`** — upcast then AR vs AR then upcast (payload
   size differs).

10. **`rotating_shuffle_chal`** — each rank needs 2 neighbors' data (r+1, r+2).
    Options: all_gather + 2 slices vs 2 collective_permutes + cat.

11. (Bonus) **`batched_ar_scale_chal`** — 5 tensors each scaled by per-rank
    scalar. Options: 5 per-tensor scaled ARs vs local scale + batch AR
    vs cat all + 1 AR + split.

## Expected outcomes

- Kiss with v14 prompt (bucketing hint) may find better strategies than
  the naive baseline for problems 1, 3, 6, 10, 11.
- Strat's strategy-enumerate may cover different subsets — hierarchical_ar
  in particular has a well-known "hierarchical all-reduce" template that
  strat might find while kiss might not (unless the prompt hints it).
- Sim vs HW may disagree in these regimes (grad_ar-like patterns): sim
  may under-charge cat/split HBM cost.

Each problem is a real tie-breaker.

## Setup for autonomous run tomorrow

- **CB**: cr-0d7ee22e9c58ec7b3, us-east-1c, 24h from 2026-08-14 11:30 UTC.
- **Launch script**: `bootstrap_v6/launch_tomorrow.sh`. Provisions cluster,
  applies patches, verifies baselines, launches strat sweep on all
  32 problems (2 + 10 comm + 11 challenge + 8 OverlayCCL).

## What tomorrow's autonomous run should do

1. Launch cluster into placement group Kaiyao (per round 12 EFA fix).
2. Bootstrap via `bootstrap_v6/apply.sh` + `launch_tomorrow.sh`.
3. Full strat sweep on all 32 problems (~3.5 hours).
4. Rescore kiss v13/v14 winners under new sim.
5. RT 2-node verify on challenge problems.
6. Reward-hack audit.
7. Commit + push results, RT scorecard, and winner codes.

## Files staged for tomorrow

- `bootstrap_v6/search/problems_challenge_v8.py` — 10 challenge problems
- `bootstrap_v6/prompts/generic_evolution_v14.md` — v14 with bucketing hint
- `bootstrap_v6/launch_tomorrow.sh` — automated launch + sweep
- Updated `bootstrap_v6/apply.sh` — includes new catalog + v14 prompt
- All previous rounds' patched sim + phase-1 fix + primitive-viability
