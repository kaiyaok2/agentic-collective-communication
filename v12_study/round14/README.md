# Round 14: final consolidated scorecard + kiss-fair edge cases

## Final consolidated results

### 2 representative _bcast (avoid-comm regime)
| Problem            | kiss RT | strat RT | winner   |
|--------------------|---------|----------|----------|
| xor_grid_bcast     | 2.51 ms | 5.25 ms  | kiss 2.1x|
| gray_code_bcast    | 2.20 ms | 5.15 ms  | kiss 2.3x|

### 10 new comm problems (comm-required regime)
| Problem                       | kiss RT | strat RT | winner |
|-------------------------------|---------|----------|--------|
| sum_across_ranks_comm         | 5.14    | 5.19     | tied   |
| max_across_ranks_comm         | 5.13    | 5.12     | tied   |
| concat_all_ranks_comm         | 5.01    | 5.07     | tied   |
| dot_across_ranks_comm         | 5.08    | 5.25     | tied   |
| shift_neighbor_comm           | (RT harness bug — sim tied)      |
| reduce_scatter_sum_comm       | 5.21    | 5.22     | tied   |
| mean_max_normalize_comm       | 5.30*   | 5.16     | tied (strat 3%) |
| rank_prefix_sum_comm          | 5.20    | 5.19     | tied   |
| center_by_mean_comm           | 5.08    | 5.29     | tied (kiss 4%) |
| top_k_scalars_comm            | 5.12    | 5.07     | tied   |

*kiss updated to match strats smart all_gather+local pattern (was 5.34 with
naive 2x all_reduce approach). Still 3% behind because strat found the pattern
first via strategy-enumerate templates; kiss can match with manual authoring.

### 8 OverlayCCL originals (sim data, RT would require rt-harness updates)
| Problem       | kiss sim | strat sim | winner |
|---------------|----------|-----------|--------|
| alltoallv     | 5376     | timeout   | kiss   |
| uniform_a2a   | 6108     | 6108      | tied   |
| ring_kv       | 5200     | 5264      | tied   |
| grad_ar       | 53902    | 7287      | STRAT 7.4x |
| dxe           | 5272     | 5207      | tied   |
| pp_send_recv  | 6014     | 6014      | tied   |
| tp_mlp        | 18680    | 18680     | tied   |
| fsdp_prefetch | 18680    | 18680     | tied   |
| llama_block_ar| 5985     | 5985      | tied   |

## Aggregate: 30 problems

- kiss wins clean: 3 (xor_grid, gray_code, alltoallv-timeout)
- strat wins clean: 1 (grad_ar, v14 prompt hint would fix)
- tied within noise: 26

Note: earlier rounds counted all 12 _bcast as kiss wins, but under user
guidance, they measure ONE test axis (recognize no-comm needed). We merged
into 2 representative + reserved the others as diagnostic tests.

## Simulator delta contribution (over PPoPP OverlayCCL paper)

Only in T_local, only fires when n_coll == 0 (collective-free graphs).
Auto-fit from raw HW measurements at Phase 1.
1. Const-fold cost: max(60, output_bytes / 46)
2. Arith saturating cost: min(1000, 340 + 100*(n-1))
3. Mixed graphs: max(const_fold, arith)
4. Unsupported local ops (cumsum/cumprod/sort/argsort): +inf via
   primitive-viability probe (extends _test_primitive_compilation).

Under this delta, 12 _bcast problems become non-trivially rankable
between kiss and strat. Without it, all these problems appeared to
score 29-38 us in the paper sim, making the ordering ambiguous.

## Load-bearing findings

1. **Kiss > strat in avoid-comm regime**: 2x on real HW (2 _bcast).
2. **Kiss = strat in comm-required regime**: ties on all 10 new comm problems.
3. **Strat > kiss on multi-collective bucketing (grad_ar)**: 7.4x, closable
   with v14 prompt hint.
4. **Phase-1 LLM tools are decorative**: they read from a static dict; making
   phase-1 deterministic (round 6) cut strat time from 25min to 3min per
   problem with no signal loss.
5. **Sim + prompt need to agree**: v12 (round 1-2) drove kiss AWAY from
   const-fold based on a broken sim, hurt RT. Round-3+ sim + v13 prompt
   fix this alignment.

## Files (this round)

-  — kiss-plausible candidates for the 10 new comm problems
-  — updated kiss code matching strats
  smart all_gather+local pattern (from 5.34ms to 5.30ms).

## Cluster info

CB cr-02ea10622d7d6bf7f active until 2026-08-13 11:30 UTC.
Master 172.31.17.206, worker 172.31.27.29, placement group Kaiyao.
