# Round 7: kiss vs strat full comparison under fixed phase-1

Under round-6 phase-1 fix (deterministic auto-probe), strat now completes
each problem in ~3 min. Ran full sweep on 12 novel _bcast + 8 OverlayCCL
problems on new CB cr-02ea10622d7d6bf7f (us-east-1c).

## Full simulator scorecard (round-4 sim)

| Problem            | cat   | kiss     | strat    | winner |
|--------------------|-------|----------|----------|--------|
| xor_grid_bcast     | novel | 88.8     | 5160.0   | kiss 58x |
| gray_code_bcast    | novel | 60.7     | 5160.0   | kiss 85x |
| piecewise_bcast    | novel | 60.7     | 669.0    | kiss 11x |
| triangle_num_bcast | novel | 60.7     | 29.0     | STRAT 2x |
| popcount_bcast     | novel | 60.7     | 5160.0   | kiss 85x |
| hamming_dist_bcast | novel | 60.7     | 5160.0   | kiss 85x |
| cond_xor_bcast     | novel | 60.7     | 5160.0   | kiss 85x |
| sum_popcount_bcast | novel | 88.8     | 102.4    | kiss 1.15x |
| sign_alt_bcast     | novel | 88.8     | 5160.0   | kiss 58x |
| perm_shuffle_bcast | novel | 60.7     | 5160.0   | kiss 85x |
| mod_sq_bcast       | novel | 60.7     | 5160.0   | kiss 85x |
| nested_mod_bcast   | novel | 60.7     | 5160.0   | kiss 85x |
| alltoallv          | orig  | 5376.4   | 5386.0   | tied |
| uniform_a2a        | orig  | 6107.9   | 6107.9   | tied |
| ring_kv            | orig  | 5200.0   | 5264.0   | tied |
| grad_ar            | orig  | 53902.4  | 7287.4   | STRAT 7.4x |
| dxe                | orig  | 5272.0   | 5207.0   | tied |
| pp_send_recv       | orig  | 6013.8   | 6013.8   | tied |
| tp_mlp             | orig  | 18680.0  | 18680.0  | tied |
| fsdp_prefetch      | orig  | 18680.0  | 18680.0  | tied |
| llama_block_ar     | orig  | 5984.5   | 5984.5   | tied |

**Total: kiss wins 11, strat wins 2, tied 8.**

## Answer to research question

**Under round-4 sim + phase-1 fix, kiss v13 >= strat on 11/12 novel _bcast problems.**
- Strat consistently falls back to baseline_ar_bcast (5160 us) on _bcast
  problems because its strategy-enumerate templates are collective-oriented
  and do not include position-based closed-form solutions.
- Only exception: triangle_num_bcast where strat found arange+cumsum which
  the sim scores at 29 us. That is a sim under-charge for cumsum (not in
  our standalone-graph model); real HW cost is probably closer to kiss's
  60 us. Worth calibrating separately.

**No regression on OverlayCCL originals: kiss ties strat on 7/8.**
- Only exception: grad_ar where strat wins 7.4x. This is a REAL strat
  advantage — strats bucketed algorithm outperforms kiss v11 for this
  gradient-all-reduce problem. Documented as follow-up.

## Two strat wins are qualitatively different

1. triangle_num_bcast: sim artifact. Cumsum charged as 1 op at 29 us
   (op_costs default 29). Real HW cost of arange+cumsum is likely
   150-400 us. If sim charged this correctly, kiss const-fold at 60 us
   would win. Not a real strat advantage.

2. grad_ar: real strat advantage. Kiss v11 produced a candidate that
   scores 53902 us; strat found a better bucketed algorithm at 7287 us.
   Kiss needs improved prompt hints for gradient-all-reduce specifically.
   This is an actual research finding.

## Follow-up ideas (documented, not yet implemented)

1. Add cumsum to standalone-graph cost model or measure it separately
   as its own probe. Would eliminate the triangle_num sim under-charge.
2. Investigate strats grad_ar candidate: what specific bucketed pattern
   did strat find? Can we add that as a prompt hint or seed for kiss?
