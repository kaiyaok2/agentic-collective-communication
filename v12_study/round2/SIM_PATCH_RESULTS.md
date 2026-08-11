# Sim cost-model patch: STANDALONE_GRAPH_HW_CAL — 2026-08-11

## Motivation

The v12-vs-v11 study revealed a systematic sim/HW gap:
kiss-v12's sim wins (arithmetic bit-decomposition on hamming_dist) did NOT
translate to RT wins — v12 was 39% SLOWER at RT despite winning sim 1.53x.

Root cause: the OverlayCCL cost model charges fused elementwise ops
(add/mul/sub/mod/div/neg) at min_local_op_us = 1.0 us floor, plus a fusion
credit against adjacent collectives. For _bcast problems with NO collectives,
the fusion credit never fires; each arithmetic op is charged 1 us. Reality:
each op contributes ~100-300 us of dispatch/launch cost on Neuron for
64-rank 16x16-tensor graphs.

## Direct hardware measurement (2026-08-11, 2-node 64-rank)

Bit-decomposition workload, N=16:
- 1 arithmetic op after arange: **343 us**
- 3 ops: **754 us** (marginal +205 us/op)
- 7 ops: **818 us** (saturation)
- 15 ops: **805 us** (saturated)

Constant-fold graph (torch.tensor([[list-comp]])): **122 us**.

## Patch

In benchmark_xla_candidate_generic (search/correctness_test.py):
if the graph has zero collective events, replace the per-op sum-of-fused-elementwise
cost with an empirical saturating model:
    cost = min(800, 340 + 100 * max(0, n_fused_arith - 1))
Constant-fold graphs (single tensor() op, no arith) charge 120 us.

Scoped to n_coll == 0 so collectives-heavy problems (grad_ar, alltoallv,
ring_kv, etc.) are UNCHANGED.

## Sim delta on 12 novel problems (kiss v11 vs kiss v12)

Before patch:
- v12 wins sim: hamming_dist 1.53x, xor_grid 1.03x
- Wall-clock RT: v12 LOSES on all tested (hamming 39% worse, sum_popcount 97% worse)

After patch:
- Sim scores match HW reality: hamming v11=v12=120 us (both use constant-fold)
- v12 wins sim CLEAN: xor_grid 802 -> 120 us (6.7x), piecewise 469 -> 120 us (3.9x)
- RT under patched sim: v12 wins/ties everywhere; xor_grid v11=3.25 v12=2.87, piecewise v11=2.79 v12=2.26

Kiss under patched sim + v12 hints converges to constant-fold on 8/12 problems
(vs 3/12 under old sim + v11 hints). The remaining 4 are cases where the
pure formula already dominates (triangle_num 3-op arith saturates below
saturation cost; sign_alt, perm_shuffle, mod_sq similar).

## Delta claim over OverlayCCL

OverlayCCL's cost model (per correctness_test.py history):
- Fusion credit against collective (30% off adjacent fusion-eligible ops)
- Volume-scaled ops (index_select, tensor(list)) at max(base, bytes/bw)
- Bandwidth floor per collective
- Bucket cap detection for chunked patterns
- 3-tier back-to-back collective amortization

This patch adds:
- Standalone-graph (collective-free) elementwise arithmetic saturating cost
- Constant-fold graph calibrated cost (~120 us)

Both terms are calibrated from direct 2-node 64-rank measurements taken
in this session; neither was in the OverlayCCL baseline. The improvement
closes a specific sim/RT gap that let arithmetic-heavy candidates falsely
beat constant-fold candidates in sim, then lose at RT.
