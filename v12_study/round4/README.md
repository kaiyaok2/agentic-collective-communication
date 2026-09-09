# Round 4: de-hardcoded auto-fit sim + v13 prompt fix

## Changes vs Round 3

1. **De-hardcoded round-3 constants** (task #180): the 5 model
   parameters (const_fold_base_us, const_fold_bw_bytes_per_us,
   arith_saturating_us, arith_marginal_first_us, arith_marginal_next_us)
   were previously literal constants inside
   `_HARDWARE_MEASUREMENTS["standalone_graph_cost_us"]`. Now they
   are auto-fit at Phase-1 tool-call time from raw HW-microbench points
   in `raw_1d` and `raw_2d`, matching the alpha1/alpha2/alpha3
   auto-fit pattern used by `measure_back_to_back_amortization`.

   Fit logic in `measure_standalone_graph_cost` handler:
   - cf_base = min(us) over all const-fold points with output <= 1KB.
   - cf_bw   = mean(bytes / (us - cf_base)) over points with output >= 4KB.
   - arith_marg_first = mean(us) over all nops=1 measurements across shapes.
   - arith_marg_next  = (mean(us at nops=3) - marg_first) / 2.
   - arith_sat       = mean(us) over all nops>=7 measurements.

2. **Extended HW measurement sweep** (tasks #181, #182):
   - Added 1D N=[16..1024] const-fold and arith x nops=[1,3,7,15] points.
   - Added 2D N=[8..256] const-fold and arith x nops=[1,3,7] points.
   - Added mixed graphs (tensor+arith) and broadcast-only patterns.
   Full raw sweep in `hw_measurements_round1_raw.txt` (base sweep from
   round 3) and `hw_measurements_round2_raw.txt` (extension for
   N=192,256 and mixed/broadcast shapes).

3. **auto_probe in no-LLM path**: when phase1_profiling(use_llm=False),
   the pipeline now deterministically calls
   `measure_memory_copy_throughput`, `measure_graph_launch_overhead`,
   `measure_back_to_back_amortization` at depths {1,2,4,8,16}, and
   `measure_standalone_graph_cost` so that `agent_sim.knowledgebase`
   and `agent_sim.config` receive their calibrated values. Same effect
   as if the LLM had probed but deterministic.

4. **v13 prompt** (task #185): built on v11, replaces the biased
   "constant fold is a trade-off — always try arithmetic first" hint
   with a **neutral "measure both, pick the cheaper one"** rule. The
   v11 bias was written when the sim itself under-charged arithmetic;
   with the round-3 sim now honest about arith cost, v11's bias pushes
   kiss away from constant-fold on 4 problems where const-fold is
   correct. v12's opposite hint (push toward bit-decomp arith) was
   even worse — REGRESSED on 3 problems. v13 removes the bias in
   both directions and lets the LLM measure both approaches. No answer
   leakage, no problem-specific values, no reference constants.

## Sim scorecard (12 novel _bcast problems)

Under round-3 sim with auto-fit constants:

| Problem            | v11    | v12    | v13    |
|--------------------|--------|--------|--------|
| xor_grid_bcast     | 895.9  | 895.9  | **88.8**  |
| gray_code_bcast    | 60.7   | 60.7   | 60.7   |
| piecewise_bcast    | 60.7   | 60.7   | 60.7   |
| triangle_num_bcast | 60.7   | 861.9  | 60.7   |
| popcount_bcast     | 60.7   | 60.7   | 60.7   |
| hamming_dist_bcast | 60.7   | 895.9  | 60.7   |
| cond_xor_bcast     | 60.7   | 60.7   | 60.7   |
| sum_popcount_bcast | 88.8   | 88.8   | 88.8   |
| sign_alt_bcast     | 895.9  | 895.9  | **88.8**  |
| perm_shuffle_bcast | 786.4  | 786.4  | **60.7**  |
| mod_sq_bcast       | 824.2  | 824.2  | **60.7**  |
| nested_mod_bcast   | 60.7   | 893.9  | 60.7   |

- **v13 wins clean: 4** (xor_grid, sign_alt, perm_shuffle, mod_sq).
- v13 never regresses vs v11 or v12.
- v12 REGRESSED vs v11 on 3 problems (triangle_num, hamming_dist,
  nested_mod). Explanation: v12's Step-6 bit-decomposition hint was
  originally motivated by the round-1 broken sim; under the accurate
  round-3 sim it is a counterproductive suggestion. v13 removes it.

## 8 OverlayCCL problems: no regressions

All 8 collective-heavy problems (alltoallv, uniform_a2a, ring_kv,
grad_ar, dxe, pp_send_recv, tp_mlp, fsdp_prefetch, llama_block_ar)
tied kiss_v11 == kiss_v12 in sim (patched code has n_coll > 0
branch = untouched). Confirms the round-3 delta is scoped strictly to
collective-free graphs.

## What remains

- Run strat-enum on the 12 novel problems under round-3 sim; verify
  kiss v13 matches / beats strat's sim numbers.
- Real-training verify the v13 wins at 64-rank.
- Push before CB 2026-08-12 11:30 UTC.
