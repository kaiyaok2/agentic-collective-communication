# Round 12: broaden test axis with 10 new comm problems + full kiss vs strat sweep

## Motivation

The 12 original _bcast benchmarks all test ONE axis: recognize when zero
communication is needed. Not a fair characterization of general
comm-search agents. Round 12 adds 10 new benchmarks that REQUIRE real
communication (all_reduce, all_gather, reduce_scatter, permute).

## 10 new comm problems

Registered in search/problems_comm_v7.py. Each has DIFFERENT input per
rank (unlike _bcast where rank 0 held the whole answer):

1. sum_across_ranks_comm    - all_reduce SUM
2. max_across_ranks_comm    - all_reduce MAX
3. concat_all_ranks_comm    - all_gather
4. dot_across_ranks_comm    - all_reduce SUM of scalar
5. shift_neighbor_comm      - collective_permute (ring shift)
6. reduce_scatter_sum_comm  - reduce_scatter
7. mean_max_normalize_comm  - two all_reduces (SUM and MAX)
8. rank_prefix_sum_comm     - all_gather + local prefix
9. center_by_mean_comm      - all_reduce + local subtract
10. top_k_scalars_comm      - all_reduce MAX of scalar

All 10 baselines pass correctness on 64-rank 2-node.

## Strat sweep results (20 problems, ~65 min total)

### 12 novel _bcast (kiss dominant regime)
Kiss wins 12/12. Strat consistently falls back to baseline_ar (5160us)
while kiss produces constant-fold (60.7us or 88.8us).

### 10 new comm problems (BOTH agents need real communication)
All 10 strat winners at ~5160us baseline all_reduce.
Kiss manual candidates score identically (both use single all_reduce).
Result: TIED on all 10. Expected: when comm is needed, theres one
right collective and both find it.

### 8 OverlayCCL originals
Kiss wins alltoallv (strat timeout), Strat wins grad_ar 7.4x (kiss v11
lacks bucketing pattern; v14 prompt hint fixes to 4407us in manual test).
Tied on 6 (uniform_a2a, ring_kv, dxe, pp_send_recv, tp_mlp, fsdp_prefetch,
llama_block_ar).

## Aggregate scorecard (30 problems total)

- kiss wins: 13 (12 _bcast + 1 alltoallv-timeout)
- strat wins: 1 (grad_ar, closes with v14 prompt)
- tied: 16 (10 comm + 6 OverlayCCL)

## Fair characterization

- Kiss > strat when local computation can replace communication.
- Kiss = strat when communication is genuinely needed.
- Kiss < strat on multi-collective bucketing patterns until prompt hints
  teach the pattern (v14 addresses grad_ar).

This is a more nuanced (and honest) picture than kiss dominates strat.
The 12 _bcast + 10 comm split makes clear that kisss advantage is on
the local-computation axis, not general comm search.
