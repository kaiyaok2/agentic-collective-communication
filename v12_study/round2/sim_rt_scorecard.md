# Sim + RT scorecard — patched sim, kiss v11 vs v12 (2026-08-11)

## Direct HW microbench (Neuron 2-node 64-rank, 200 iters/mark_step)

| Case | HW us/mark_step |
|---|---|
| Constant tensor emit (16x16) | 122 |
| Arith float chain (bit-decomp on 16x16) | 913 |
| Arith int64 chain | 988 |
| Bitwise XOR chain | 963 |
| 1 add after arange | 343 |
| 3 adds after arange | 754 |
| 7 adds after arange | 818 |
| 15 adds after arange | 805 |

## Sim results (12 problems, patched cost model)

| Problem | v11 sim (us) | v12 sim (us) | verdict |
|---|---|---|---|
| xor_grid_bcast | 802 | 120 | **v12 wins 6.7x** |
| gray_code_bcast | 120 | 120 | tied (both constant-fold) |
| piecewise_bcast | 469 | 120 | **v12 wins 3.9x** |
| triangle_num_bcast | 540 | 540 | tied (3-op arith baseline) |
| popcount_bcast | 120 | 120 | tied |
| hamming_dist_bcast | 120 | 120 | tied |
| cond_xor_bcast | 120 | 120 | tied |
| sum_popcount_bcast | 120 | 120 | tied |
| sign_alt_bcast | 642 | 642 | tied (4-op arith) |
| perm_shuffle_bcast | 440 | 440 | tied |
| mod_sq_bcast | 440 | 440 | tied |
| nested_mod_bcast | 120 | 120 | tied |

**v12 clean sim wins: 2. No v11 wins. Tied: 10.**

## Real-training (ms/iter, 300 iters, 2-node 64-rank)

| Problem | v11 RT (ms) | v12 RT (ms) | verdict |
|---|---|---|---|
| xor_grid_bcast | 3.25 | 2.87 | v12 wins 13% |
| hamming_dist_bcast | 2.46 | 2.50 | tied |
| nested_mod_bcast | 2.33 | 2.18 | v12 wins 7% |
| sum_popcount_bcast | 2.76 | 2.77 | tied |
| piecewise_bcast | 2.79 | 2.26 | v12 wins 23% |

Under patched sim, kiss v12 converges to constant-fold everywhere the pattern
applies. v12 wins RT on the 2 sim-win problems and 1 tied-sim problem
(nested_mod). Nothing regresses.

## Comparison to session round 1 (unpatched sim)

Round-1 conclusion was v12 hints hurt RT by 39-97% because unpatched sim
misled kiss v12 into arithmetic bit-decomposition. With the patched sim,
kiss v12 correctly uses constant-fold everywhere and matches or beats v11 RT.

## The load-bearing improvement claimed on top of OverlayCCL

1. Cost floor for collective-free graphs (standalone arithmetic chains) via
   saturating dispatch model calibrated from 2-node HW measurements.
2. Constant-fold graph cost calibrated to ~120 us.

Together these terms are what let the sim / RT ranking now agree for _bcast
position-based problems where OverlayCCL's model was arithmetic-friendly.
