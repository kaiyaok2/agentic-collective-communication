# Sorcar vs Strat: Family Taxonomy of the 142-Problem Pool

**Scope**: the full RT-verified problem pool, organized into **7
optimization families**. For each family this doc gives (1) what the
family is, (2) Sorcar's general solution, (3) why strat-enumerate stays
at baseline, and (4) the complete per-problem warm-cache RT results.

**Measurement setup**: 7× trn1.32xlarge (224 NeuronCores), CB
`cr-0f2c701080c291ea8`, us-east-1c, run 2026-08-23/24. Warm-cache RT:
each variant run 2× back-to-back, second measurement reported; 100
iters per run. `Base ms` is the baseline/strat candidate (strat's
enumeration output equals the baseline template on every problem in
this pool); `Sorcar ms` is Sorcar's rewrite. 143 RT-verified entries
(a few are same-shape re-measurements, e.g. perrowmaxM256/―big), 137
wins ≥5%, 6 borderline ties, 0 losses.

## Family index

| # | Family | Problems | Win range | Representative |
|---|---|---|---|---|
| F1 | Sequential-AR linearity | 21 | 1.05–1.39× | six_ar_arith (1.28×) |
| F2 | CSE across redundant ARs | 25 | 1.08–4.30× | sixtyfourinline (4.30×) |
| F3 | Dead-collective elimination & algebraic zero | 15 | 1.04–8.82× | ag_slice_use (8.82×) |
| F4 | Per-row/col/batch dispatch collapse | 68 | 1.13–174.60× | perrowM2048 (174.60×) |
| F5 | Collective-type conversion & data-flow narrowing | 9 | 1.02–1.21× | four_scaled_plus_bcast_ar (1.21×) |
| F6 | Mixed-reduction-op extraction | 2 | 1.66–13.50× | perrow_mixed_bigM (13.50×) |
| F7 | Slab/chunk payload fusion | 3 | 1.07–1.41× | eightslab (1.41×) |

Total: 143 RT-verified entries, 137 wins ≥5%, 6 borderline ties, 0 losses.

---

## F1. Sequential-AR linearity (21 problems)

**What it is.** A chain of all-reduces whose results are combined
linearly: `y = c1*AR(x1) + c2*AR(x2) + ... + ck*AR(xk)`, where each `xi`
is a locally-computable transform of the input (often `x*si` or a
sequential dependency that unrolls to one). Includes the scaled-input
variant `AR(x*c) = c*AR(x)`.

Members: all 18 Cat-A problems (triple_ar_linear, ar_scalar_chain,
seq_dep_chain4/5, four/five/six/seven/eight_ar_* , chained_ar_nested,
sequential_ar_chain) + extras seven_scaled_input, mixedscaledseq,
seven_scaled_diff, plus the N=1M Cat-B scaled pair.

**Sorcar's general solution.** Recognize that all-reduce with
`REDUCE_SUM` is a linear operator: `sum_i c_i * AR(x_i) =
AR(sum_i c_i * x_i)`. Fold the whole chain into **one AR of a locally
pre-combined payload** plus scalar post-math. K collectives → 1.

```python
# baseline: k dispatches            # sorcar: 1 dispatch
y  = xm.all_reduce(SUM, x) * 2.0    ar = xm.all_reduce(SUM, x)
y -= xm.all_reduce(SUM, x) * 0.5    return ar * 3.5   # 2-0.5+1.5-0.25+0.75
y += xm.all_reduce(SUM, x) * 1.5
...
```

**Why strat stays baseline.** Strat's 5-strategy enumeration operates at
the level of *collective structure* (which primitive, what payload
layout, what ordering) — "apply the algebraic linearity of the reduction
operator" is not one of its strategy axes. Its candidates re-arrange the
same k ARs; the simulator correctly ranks them all ≈ equal, and the
refinement rounds mutate within the k-AR shape. RT win: 1.05–1.39×
(dispatch overhead of k−1 saved ARs on a latency-bound payload).


**Per-problem warm-cache RT results (224 ranks):**

| Problem | Base ms | Sorcar ms | Ratio |
|---|---|---|---|
| seven_scaled_diff (7 scaled ARs w/ zeros) | 11.16 | 8.06 | **1.39×** |
| eight_ar_half_ints | 11.31 | 8.37 | **1.35×** |
| seven_scaled_input (7 diff-scale ARs) | 10.90 | 8.13 | **1.34×** |
| seven_ar_seq | 10.75 | 8.20 | **1.31×** |
| six_ar_altsign | 10.56 | 8.19 | **1.29×** |
| six_ar_arith | 10.28 | 8.05 | **1.28×** |
| six_ar_seq | 10.55 | 8.32 | **1.27×** |
| five_ar_mixed_sign | 10.37 | 8.18 | **1.27×** |
| mixedscaledseq (5 mixed-scale ARs) | 10.14 | 8.05 | **1.26×** |
| four_ar_N224 | 10.00 | 8.15 | **1.23×** |
| seq_dep_chain5 | 9.85 | 8.26 | **1.19×** |
| seq_dep_chain4_scaled | 9.69 | 8.17 | **1.19×** |
| five_ar_arith_prog | 9.84 | 8.29 | **1.19×** |
| three_ar_frac_dep | 9.21 | 7.97 | **1.16×** |
| four_ar_pow2 | 9.46 | 8.24 | **1.15×** |
| four_ar_mixed_coef | 9.46 | 8.26 | **1.15×** |
| four_ar_evens | 9.22 | 8.26 | **1.12×** |
| chained_ar_nested | 8.98 | 8.15 | **1.10×** |
| triple_ar_linear | 8.85 | 8.09 | **1.09×** |
| sequential_ar_chain | 8.38 | 7.89 | **1.06×** |
| ar_scalar_chain | 8.80 | 8.37 | **1.05×** |

## F2. CSE across redundant ARs of the same input (25 problems)

**What it is.** N syntactically distinct calls `AR(x)` on the *same*
unmodified input, combined arithmetically — inline in one expression, or
assigned to N variables. The N results are identical; N−1 collectives
are pure waste.

Members: Cat-C1 (four/five/seven/nine_ar_same_input, three_inline_ars,
three_scaled_x_ars, alternating_indep_ars, five_ar_indep_sumatend,
six_ar_indep_pool, four_ar_indep_large_N, ar_via_two_paths) + extras
twelve/sixteen/twenty/twentyfour/twentyeight/thirty/thirtysix/forty/
fifty/sixtyfour-inline, tenariindep, large_N_4ar, thirtytwoalt,
eightyaltsum.

**Sorcar's general solution.** Hoist to a single `AR(x)`, replace every
other call with the hoisted variable, and collapse the arithmetic
combination to one scalar multiplier where possible
(`sum_{i=1..N} i = N(N+1)/2`, alternating-sign sums, etc.).

**Why strat stays baseline.** XLA's HLO CSE pass catches *some*
assigned-first cases but not inline-call chains (each `xm.all_reduce`
call site materializes its own token-ordered collective). Strat's
enumeration proposes payload/bucketing re-layouts of N collectives — it
never proposes "these N collectives are the same value." The win grows
with N: 1.08× at N=3 → 4.30× at N=64 (dispatch cost ~0.4ms per redundant
AR at 224 ranks).


**Per-problem warm-cache RT results (224 ranks):**

| Problem | Base ms | Sorcar ms | Ratio |
|---|---|---|---|
| eightyaltsum (80 alt-sign ARs) | 41.23 | 8.26 | **4.99×** |
| sixtyfourinline (64 inline ARs) | 34.15 | 7.95 | **4.30×** |
| fiftyinline (50 inline ARs) | 28.92 | 8.20 | **3.53×** |
| fortyinline (40 inline ARs) | 24.93 | 7.96 | **3.13×** |
| thirtysixinline (36 inline ARs) | 23.00 | 7.97 | **2.89×** |
| thirtytwoalt (32 alt-sign ARs) | 21.39 | 7.89 | **2.71×** |
| thirtyinline (30 inline ARs) | 20.37 | 8.02 | **2.54×** |
| twentyeightinline (28 inline ARs) | 19.51 | 7.84 | **2.49×** |
| twentyfourinline (24 inline ARs) | 18.54 | 8.21 | **2.26×** |
| twentyinline (20 inline ARs) | 16.50 | 8.16 | **2.02×** |
| sixteeninlin (16 inline ARs) | 14.94 | 8.21 | **1.82×** |
| twelveinlin (12 inline ARs) | 13.03 | 8.21 | **1.59×** |
| tenariindep (10 CSE-ready ARs) | 12.40 | 8.03 | **1.55×** |
| nine_ar_same_input | 11.78 | 7.89 | **1.49×** |
| seven_ar_same_input | 10.98 | 8.06 | **1.36×** |
| six_ar_indep_pool | 10.36 | 8.19 | **1.27×** |
| five_ar_scaled_same_input | 9.92 | 8.11 | **1.22×** |
| alternating_indep_ars | 10.10 | 8.54 | **1.18×** |
| five_ar_indep_sumatend | 9.83 | 8.30 | **1.18×** |
| four_ar_indep_large_N | 9.54 | 8.13 | **1.17×** |
| large_N_4ar (65K, four ARs merged) | 9.36 | 8.02 | **1.17×** |
| four_ar_same_input | 9.38 | 8.06 | **1.16×** |
| three_scaled_x_ars | 8.98 | 8.23 | 1.09× |
| three_inline_ars | 8.88 | 8.20 | 1.08× |
| ar_via_two_paths | 8.47 | 8.15 | 1.04× (tie) |

## F3. Dead-collective elimination & algebraic zero (15 problems)

**What it is.** Collectives whose results are provably unused,
mathematically canceled, or reducible to a constant: alternating-sign
sums that telescope to zero, `max` of an already-maxed tensor
(idempotence), gather-then-verify patterns where the verify branch is
dead, sums that cancel pairwise.

Members: Cat-C2 (ag_slice_use*, max_reduce_redundant,
idempotent_reduce_max, max_min_with_dead, mixed_reduce_dead_sum,
ar_dead_gather_verify, min_neg_max_dead_verify, three_ars_two_zero,
four_ar_sum_zero, ten_ar_alt_sign_zero, ar_scaled_by_worldsize,
pow_ar_double_verify) + extras algzero16, algzero20, csescaleddiff.
(*ag_slice_use also fits F5; its dead-AG elimination core places it here.)

**Sorcar's general solution.** Prove the algebraic identity and emit the
closed form — `torch.zeros_like(x)` for telescoping sums, drop the dead
collective entirely, or fold the constant. The extreme case replaces 20
ARs with **zero communication**: 16.41ms → 1.95ms (8.44×), where the
remaining 1.95ms is pure graph-launch floor.

**Why strat stays baseline.** Strat's strategies must *implement the
computation described*; nothing in its enumeration axis allows "the
answer is analytically zero, skip the network." Its correctness gate
(MockTorch) additionally rejects zero-collective candidates for several
of these problems (documented root cause: `sorcar_vs_strat_root_cause`,
2026-08-17), so even when the LLM proposes local recompute, the
candidate dies before the simulator ranks it. Sorcar recovers by
iterating against `score_candidate` feedback.


**Per-problem warm-cache RT results (224 ranks):**

| Problem | Base ms | Sorcar ms | Ratio |
|---|---|---|---|
| ag_slice_use | 15.72 | 1.78 | **8.82×** |
| algzero20 (20-AR algebraic zero) | 16.41 | 1.95 | **8.44×** |
| algzero16 (16-AR algebraic zero) | 14.82 | 1.90 | **7.80×** |
| ten_ar_alt_sign_zero (algebraic zero) | 11.94 | 1.85 | **6.45×** |
| four_ar_sum_zero (algebraic zero) | 9.28 | 2.13 | **4.36×** |
| ar_dead_gather_verify | 16.81 | 8.04 | **2.09×** |
| csescaleddiff (5 scaled ARs w/ zeros) | 9.90 | 7.97 | **1.24×** |
| three_ars_two_zero | 9.18 | 8.03 | **1.14×** |
| idempotent_reduce_max | 8.92 | 7.94 | **1.12×** |
| max_min_with_dead | 9.25 | 8.52 | 1.09× |
| max_reduce_redundant | 8.52 | 7.87 | 1.08× |
| min_neg_max_dead_verify | 8.61 | 7.95 | 1.08× |
| mixed_reduce_dead_sum | 8.99 | 8.53 | 1.05× |
| ar_scaled_by_worldsize | 8.48 | 8.13 | 1.04× (tie) |
| pow_ar_double_verify | 8.49 | 8.14 | 1.04× (tie) |

## F4. Per-row/col/batch dispatch collapse (68 problems — the largest family)

**What it is.** A loop (list comprehension) that all-reduces each row /
column / batch-slice of a 2D/3D tensor separately, then stacks:
`torch.stack([AR(x[m]) for m in range(M)])`. The baseline issues M
dispatches of N elements each; collectives are latency-bound at these
payload sizes, so cost ≈ M × dispatch-overhead.

Members: Cat-C3 (per_row_ar_M8..M1024, per_row_max/min_ar*,
per_column_ar*, per_batch_ar_3d) + ~42 extras (perrowM28..M3072,
perrowmax/min at 8 scales, percolC16..C512, perbatch* 2D/3D SUM/MAX/MIN,
perslice2d, perslice3dM32/64/96).

**Sorcar's general solution.** One line: `AR(x)` on the full tensor —
all-reduce is elementwise across ranks, so reducing the whole 2D/3D
tensor at once is semantically identical to reducing each slice.
M dispatches → 1. Wins scale linearly with M and are the largest in the
pool:

| M (rows) | RT win |
|---|---|
| 28 | 2.48× |
| 128 | 7.62× |
| 512 | 23.91× |
| 1024 | 46.27× |
| 2048 | **174.60×** |
| 3072 | 126.97× (baseline compile-time limited) |

**Why strat stays baseline.** Strat's per-strategy implementation is
anchored to the baseline template it is shown: a per-row loop. Its five
structural strategies vary *how the M collectives are laid out*
(bucketed, pipelined, reordered) — the simulator ranks these ≈ equal
because all keep M dispatches. "Reduce the enclosing tensor instead of
its slices" requires reading the *semantics* of the loop rather than its
structure — the freeform rewrite Sorcar performs and enumeration never
reaches. This family also breaks the Neuron compiler's own fusion: XLA
does not merge M token-ordered collectives into one.


**Per-problem warm-cache RT results (224 ranks):**

| Problem | Base ms | Sorcar ms | Ratio |
|---|---|---|---|
| perrowM2048 (2048, 32) SUM | 1538.25 | 8.81 | **174.60×** |
| perrowM3072 (3072, 24) SUM | 1019.56 | 8.03 | **126.97×** |
| perrowM1536 (1536, 42) SUM | 542.81 | 7.88 | **68.90×** |
| per_row_ar_M1024 (1024, 64) | 367.94 | 7.95 | **46.27×** |
| perrowM768 (768, 96) SUM | 283.73 | 8.25 | **34.42×** |
| percolC512 (256, 512) SUM | 231.90 | 8.06 | **28.76×** |
| per_row_ar_M512 (512, 128) | 196.88 | 8.23 | **23.91×** |
| perrowM384 (384, 192) SUM | 168.90 | 8.13 | **20.77×** |
| percolC384 (128, 384) SUM | 148.75 | 8.03 | **18.52×** |
| perrowmaxM256 (256, 256) MAX | 118.11 | 8.03 | **14.71×** |
| per_row_ar_M256 (256, 256) | 117.76 | 8.11 | **14.52×** |
| perrowmaxM256big (256, 256) MAX | 117.18 | 8.16 | **14.37×** |
| perrowminM256 (256, 256) MIN | 119.07 | 8.29 | **14.37×** |
| percolC256 (128, 256) SUM | 99.54 | 8.06 | **12.35×** |
| perrowM192 (192, 256) SUM | 89.71 | 8.07 | **11.12×** |
| percolC192 (256, 192) SUM | 88.66 | 8.17 | **10.86×** |
| perrowM144 (144, 384) SUM | 68.86 | 8.00 | **8.61×** |
| perrowmaxM128big (128, 512) MAX | 61.96 | 7.87 | **7.88×** |
| perrowmaxM128 (128, 512) MAX | 62.28 | 8.03 | **7.76×** |
| percolC128Big (256, 128) SUM | 61.47 | 8.00 | **7.68×** |
| percolC128Bigger (512, 128) SUM | 61.73 | 8.06 | **7.66×** |
| perrowminM128 (128, 512) MIN | 62.10 | 8.14 | **7.63×** |
| per_row_ar_M128 (128, 512) | 61.46 | 8.07 | **7.62×** |
| percolC128 (64, 128) | 53.01 | 7.97 | **6.65×** |
| perrowM112 (112, 512) SUM | 54.37 | 8.17 | **6.65×** |
| perrowmaxM96 (96, 512) MAX | 47.89 | 7.90 | **6.06×** |
| percolC96 (192, 96) SUM | 47.81 | 8.00 | **5.98×** |
| perrowmaxM96N1K (96, 1024) MAX | 48.12 | 8.26 | **5.83×** |
| per_row_ar_M96 (96, 512) | 47.55 | 8.19 | **5.81×** |
| perslice3dM96 (96, 8, 512) SUM | 48.40 | 8.42 | **5.75×** |
| perrowM80 (80, 256) SUM | 42.13 | 8.07 | **5.22×** |
| perrowmaxM64 (64, 1024) MAX | 34.23 | 7.87 | **4.35×** |
| per_row_ar_M64 (64, 1024) | 34.88 | 8.14 | **4.29×** |
| perslice2d (64, 1024) full-slice | 34.49 | 8.03 | **4.29×** |
| perrowminM64 (64, 1024) MIN | 34.66 | 8.23 | **4.21×** |
| percolC64Big (256, 64) SUM | 34.36 | 8.16 | **4.21×** |
| perrowM64N4K (64, 4096) SUM | 34.45 | 8.29 | **4.15×** |
| perslice3dM64 (64, 16, 512) SUM | 34.68 | 8.36 | **4.15×** |
| perrowM56 (56, 1024) SUM | 31.79 | 8.21 | **3.87×** |
| per_column_ar_C64 (128, 64) | 30.72 | 7.97 | **3.85×** |
| perrowminM48 (48, 512) MIN | 28.27 | 7.94 | **3.56×** |
| perrowmaxM48N2K (48, 2048) MAX | 28.44 | 8.05 | **3.53×** |
| per_row_ar_M48 (48, 1024) | 27.99 | 8.05 | **3.48×** |
| percolC48 (128, 48) SUM | 24.92 | 8.02 | **3.11×** |
| perrowM40 (40, 512) SUM | 24.80 | 8.07 | **3.07×** |
| per_column_ar_C32 (256, 32) | 21.30 | 7.96 | **2.68×** |
| per_row_max_ar_M32 (32, 2048) MAX | 21.30 | 8.00 | **2.66×** |
| per_row_ar_M32 (32, 2048) | 21.17 | 8.22 | **2.58×** |
| per_row_min_ar_M32 (32, 2048) MIN | 20.92 | 8.11 | **2.58×** |
| perrowM32N8K (32, 8192) SUM | 21.49 | 8.32 | **2.58×** |
| perbatch3dmaxM32 (32, 16, 512) MAX | 21.36 | 8.32 | **2.57×** |
| perslice3dM32 (32, 32, 512) SUM | 21.46 | 8.39 | **2.56×** |
| perbatchM32 (32, 16, 512) SUM | 21.42 | 8.41 | **2.55×** |
| perrowM28 (28, 1024) SUM | 19.60 | 7.90 | **2.48×** |
| percolC24 (192, 24) SUM | 17.80 | 8.09 | **2.20×** |
| per_column_max_ar (512, 16) MAX | 14.72 | 7.93 | **1.86×** |
| perbatchmin (16, 16, 512) MIN | 14.76 | 8.00 | **1.85×** |
| per_column_ar_C16 (512, 16) | 14.60 | 7.92 | **1.84×** |
| per_row_min_ar (16, 4096) MIN | 14.67 | 8.01 | **1.83×** |
| perbatchmaxM16big (16, 32, 512) MAX | 14.71 | 8.16 | **1.80×** |
| per_row_max_ar (16, 4096) MAX | 14.60 | 8.17 | **1.79×** |
| perbatchM12 (12, 32, 256) SUM | 12.85 | 8.12 | **1.58×** |
| per_row_ar_M8 (8, 8192) | 11.30 | 8.06 | **1.40×** |
| perbatchmax (8, 16, 512) MAX | 11.38 | 8.13 | **1.40×** |
| per_column_ar (1024, 8) | 11.16 | 8.04 | **1.39×** |
| perbatchmaxBig (8, 32, 512) MAX | 11.19 | 8.16 | **1.37×** |
| per_batch_ar_3d (8, 16, 512) | 11.23 | 8.24 | **1.36×** |
| perbatchM4big (4, 64, 512) SUM | 9.32 | 8.28 | 1.13× (tie) |

## F5. Collective-type conversion & data-flow narrowing (9 problems)

**What it is.** The result of a collective is only partially consumed —
a slice, a scalar reduction, or the local shard — so a cheaper
collective (or none) suffices. Includes AR→reduce_scatter conversion,
allgather-then-slice → direct use, local-reduce-before-AR (shrink
payload before crossing EFA), collective_permute cycles that net to
identity, and broadcast-with-mask fusions.

Members: Cat-C4–C6 (ar_before_local_reduce_M128,
ar_then_scalar_reduce_largeN, ar_4chunk_pattern, conditional_ars,
reduce_scatter_from_ar, cp_double_swap, four_scaled_plus_bcast_ar,
three_group_dead_verify, compare_two_ars) + ag_slice_use overlap from F3.

**Sorcar's general solution.** Track which bytes of the collective
output are live, then substitute the narrowest primitive that produces
exactly those bytes: `AR(x)[shard]` → `reduce_scatter(x)`;
`AG(x)[i]` → `x` local; permute∘permute = id → drop both.

**Why strat stays baseline.** This is the family where strat is
*closest* — its enumeration does include "AG+RS chain" style strategies,
and 3 of the 6 pool ties live here. But it applies conversions
structurally (as layout alternatives with identical liveness), not
data-flow-analytically, so it misses the cases where the narrowing is
only visible by tracing which slice the caller consumes. Wins are
mid-range: 1.06–1.21×. (ag_slice_use, 8.82×, has an F5-style AG→local
narrowing core but is tabulated under F3 since dead-AG elimination is
its dominant mechanism.)


**Per-problem warm-cache RT results (224 ranks):**

| Problem | Base ms | Sorcar ms | Ratio |
|---|---|---|---|
| four_scaled_plus_bcast_ar | 9.88 | 8.16 | **1.21×** |
| reduce_scatter_from_ar | 10.51 | 8.79 | **1.20×** |
| cp_double_swap | 2.48 | 2.12 | **1.17×** |
| ar_4chunk_pattern | 9.35 | 8.10 | **1.15×** |
| conditional_ars | 8.92 | 7.94 | **1.12×** |
| three_group_dead_verify | 10.15 | 9.08 | **1.12×** |
| ar_before_local_reduce_M128 | 8.56 | 8.08 | 1.06× |
| ar_then_scalar_reduce_largeN | 8.14 | 7.73 | 1.05× |
| compare_two_ars | 8.45 | 8.27 | 1.02× (tie) |

## F6. Mixed-reduction-op extraction (2 problems — NEW family)

**What it is.** A computation that interleaves *different* reduction
ops on the same input — e.g. 8 alternating `AR_MAX(x)` and `AR_MIN(x)`
calls scaled and summed, or a per-row loop computing
`AR_MAX(x[m]) − AR_MIN(x[m])`. Neither pure CSE (ops differ) nor pure
dispatch collapse (two distinct collectives are genuinely needed).

Members: extras mixmaxmin (1.66×), perrow_mixed_bigM (13.50×).

**Sorcar's general solution.** Extract each *distinct* reduction once —
`xmax = AR_MAX(x); xmin = AR_MIN(x)` — then rebuild the combination
locally (scalar coefficient sums, elementwise `xmax − xmin` on the full
2D tensor). 16 collectives → 2. The compound case (perrow_mixed_bigM)
stacks family F4's row-collapse on top: 2M per-row collectives → 2
full-tensor collectives, hence the 13.50×.

**Why strat stays baseline.** It compounds two rewrites strat already
misses individually: value-identity across call sites (F2) and
slice-to-tensor semantic lifting (F4). An enumerated strategy would need
to name both simultaneously; the strategies strat actually proposes
re-schedule the 2M collectives without deduplicating them.


**Per-problem warm-cache RT results (224 ranks):**

| Problem | Base ms | Sorcar ms | Ratio |
|---|---|---|---|
| perrow_mixed_bigM (128, 256) MAX-MIN | 115.81 | 8.58 | **13.50×** |
| mixmaxmin (max+min×8) | 14.54 | 8.76 | **1.66×** |

## F7. Slab/chunk payload fusion (3 problems — NEW family)

**What it is.** Contiguous slabs of one tensor all-reduced separately
and re-concatenated: `cat([AR(x[i*N:(i+1)*N]) for i in range(8)])`,
optionally with per-slab scaling. Structurally the 1-D sibling of F4,
but the payloads are *slices of a single buffer*, so the rewrite is
about payload re-layout, not loop semantics.

Members: extras batchedar8 (1.35×), eightslab (1.41×), catsplitar
(1.07× tie).

**Sorcar's general solution.** All-reduce the whole buffer once, then
apply per-slab scaling to the *output* slices:
`ar = AR(x); cat([ar[i*N:(i+1)*N] * (i+1) ...])`. 8 dispatches → 1;
slicing is free (metadata view ops).

**Why strat stays baseline.** This is the paper's own "one stacked AR
over packed payload" strategy — strat *does* enumerate it — but its
implementations packed via `stack`/copy rather than recognizing the
slabs are already contiguous in the source buffer, and the resulting
extra memcpy erased the sim margin, so refinement reverted to baseline.
The tie on catsplitar (1.07×) shows the Neuron compiler partially
recovers this one on its own: with only 2 slabs, XLA fuses the pair of
ARs, leaving no dispatch saving for either agent. Wins here are the
smallest of any family — the dispatch saving (8→1) is real but the
baseline was already only ~11ms.

**Per-problem warm-cache RT results (224 ranks):**

| Problem | Base ms | Sorcar ms | Ratio |
|---|---|---|---|
| eightslab (8 slab AR chunks) | 11.32 | 8.04 | **1.41×** |
| batchedar8 (8 chunks) | 11.27 | 8.37 | **1.35×** |
| catsplitar (split-cat baseline) | 8.57 | 8.04 | 1.07× (tie) |


---

## Cross-family observations

1. **One mechanism, many families.** Families F1, F2, F4, F6, F7 are
   all ultimately *dispatch-count reduction* — the per-dispatch cost at
   224 ranks (~0.4–0.5 ms measured; graph-launch + CCOM rendezvous
   dominated) is the single largest lever on latency-bound collectives.
   The families differ in **what analysis is needed to see the
   reduction**: operator linearity (F1), value numbering (F2), loop
   semantics (F4), op-set partitioning (F6), buffer contiguity (F7).

2. **Strat's blind spot is semantic, not structural.** Every family
   where strat loses requires reasoning about *what the code computes*
   (algebra, liveness, value identity). The one family where strat is
   competitive (F5, 3 of 6 pool ties) is where the rewrite is
   expressible as a structural layout alternative — exactly what
   enumeration is built to cover.

3. **The compiler recovers structure, not semantics, too.** The 6
   borderline ties are all cases where XLA/Neuron auto-fuses
   (ar_via_two_paths, compare_two_ars, catsplitar, ...). No case was
   found where the compiler performs an F1/F2/F3-style algebraic
   rewrite on its own.

4. **Wins scale with problem size within a family** (M in F4, N in F2),
   so family membership predicts *scaling behavior*, not just a fixed
   ratio — the basis for choosing one representative per family in the
   e2e training validation.

---

## End-to-end training validation

The family rewrites were validated inside real LLM training loops (all
7 sites in one step, real wikitext, loss parity):

| Experiment | Scale | Regime | Speedup | Doc |
|---|---|---|---|---|
| Dense Llama pure-DP | 26M | replicated-Adam-dominated | 5.8× (3 seeds) | `SORCAR_E2E_FAMILIES.md` |
| Expert-choice MoE | 9.4B | a2av-exchange-dominated (fixed both backends) | 1.02–1.04× loss-neutral | logs: `session_logs_2026_08_29/` |
| Dense Llama TP=32×DP=7 | 9.75B | grad-sync + optimizer-dominated | **2.02–2.47×** | `SORCAR_E2E_10B_TP.md` |
| Dense GPT-3-class TP=32×DP=7 | 9.70B | same | **1.90–2.21×** | `SORCAR_E2E_10B_TP.md` |
