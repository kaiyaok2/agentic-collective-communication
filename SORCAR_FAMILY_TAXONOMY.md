# Sorcar vs Strat vs Baseline: Family Taxonomy of the 55-Problem Set

**Scope**: the **55 problems** — every problem in the taxonomy
pool where Sorcar's searched rewrite beats OverlayCCL strat-enumeration
by >5% in the calibrated simulator.

The 55 span **6 optimization families** (F5, collective-type conversion /
ZeRO-1 data-flow narrowing, is exercised only in the E2E optimizer path,
not as a standalone micro-anchor).

**Three columns, three distinct code paths on every problem:**
- **baseline** — naive textbook-DDP source (one collective per logical op).
- **strat** — OverlayCCL 5-strategy enumeration output. On these 55 it emits
  *distinct source* (accumulate loops, per-tensor AR loops, re-bucketed
  payloads) but reaches the *same collective schedule* as baseline — it
  finds no fusion. This is why strat_ms ≈ baseline_ms at RT (see tables).
- **sorcar** — the searched family rewrite that fuses/eliminates collectives.

**Measurement**: 7× trn1.32xlarge (224 NeuronCores), us-east-1c. Simulator
columns from `taxonomy_3col_results/three_col.json`; warm-cache RT columns
(each variant run 2× back-to-back, 2nd reported, 100 iters) from
`taxonomy_3col_results/RT_THREE_COL_RESULTS.json`.

## Summary

| | count |
|---|---|
| Anchors (sim, sorcar > strat by >5%) | 55 |
| RT-confirmed Sorcar wins (≥1.05× warm-cache) | 45 |
| At RT dispatch floor (sim gap < RT noise) | 8 |
| Sorcar sim-pass / HW-abort (F3 total-cancel edge) | 2 |
| Strat RT wins over baseline | 0 |
| Strat sim wins over baseline (on these 55) | 0 |

## Family index

| # | Family | Anchors | Sim ratio range | Best RT (sorcar vs strat) |
|---|---|---|---|---|
| F1 | Sequential-AR linearity | 39 | 1.07-4.24× | 3.39× |
| F2 | CSE across redundant ARs of the same input | 7 | 1.14-1.34× | 1.34× |
| F3 | Dead-collective elimination & algebraic zero | 2 |  + ∞(total-cancel) | (sim-only) |
| F4 | Per-row/col/batch dispatch collapse | 5 | 1.43-4.51× | 2.32× |
| F6 | Mixed-reduction-op extraction | 1 | 1.53-1.53× | 1.28× |
| F7 | Slab/chunk payload fusion | 1 | 1.28-1.28× | 1.19× |

Total: 55 anchors across 6 families.

---

## F1. Sequential-AR linearity (39 anchors)

**What it is.** A chain of all-reduces combined linearly: y = c1*AR(x1)+...+ck*AR(xk), where each xi is a locally-computable transform of the input. all_reduce(SUM) is a linear operator, so the whole chain folds into ONE AR of a locally pre-combined payload plus scalar post-math. K collectives -> 1.

**Why strat stays at baseline's schedule.** Strat-enum operates at the level of collective STRUCTURE (which primitive, what payload layout) not collective ALGEBRA. It never proves the k ARs are linearly combinable, so it keeps baseline's k-dispatch schedule (its emitted source is a distinct accumulate loop, but the collective count is identical).

**Per-problem data** (sim µs; warm-cache RT ms, 224 ranks):

| Problem | sim strat | sim sorcar | sim × | RT base | RT strat | RT sorcar | RT sorcar/strat |
|---|---|---|---|---|---|---|---|
| eightyaltsum | 21960 | 5178 | 4.24× | 17.13 | 17.16 | 5.07 | **3.39×** |
| sixtyfourinline | 18733 | 5178 | 3.62× | 14.41 | 14.28 | 5.16 | **2.77×** |
| fiftyinline | 15910 | 5178 | 3.07× | 12.20 | 12.34 | 5.34 | **2.31×** |
| fortyinline | 13875 | 5178 | 2.68× | 10.92 | 10.85 | 4.97 | **2.18×** |
| thirtytwoalt | 12244 | 5178 | 2.37× | 9.85 | 9.78 | 4.71 | **2.08×** |
| thirtysixinline | 13060 | 5178 | 2.52× | 10.03 | 10.15 | 4.95 | **2.05×** |
| thirtyinline | 11836 | 5178 | 2.29× | 9.21 | 9.44 | 5.28 | **1.79×** |
| twentyeightinline | 11428 | 5178 | 2.21× | 8.74 | 9.26 | 5.31 | **1.75×** |
| twentyfourinline | 10613 | 5178 | 2.05× | 8.30 | 8.26 | 5.12 | **1.61×** |
| sixteeninlin | 8982 | 5178 | 1.74× | 7.27 | 7.47 | 4.86 | **1.54×** |
| twentyinline | 9797 | 5178 | 1.89× | 7.82 | 7.84 | 5.24 | **1.50×** |
| twelveinlin | 7959 | 5178 | 1.54× | 6.95 | 6.44 | 4.92 | **1.31×** |
| four_scaled_plus_bcast_ar_chal | 6168 | 5178 | 1.19× | 5.71 | 5.90 | 4.79 | **1.23×** |
| large_N_4ar | 5914 | 5178 | 1.14× | 5.49 | 5.72 | 4.64 | **1.23×** |
| tenariindep | 7448 | 5178 | 1.44× | 6.37 | 6.33 | 5.30 | **1.20×** |
| mixedscaledseq | 6169 | 5178 | 1.19× | 5.97 | 5.92 | 5.03 | **1.18×** |
| five_ar_indep_sumatend_chal | 6166 | 5180 | 1.19× | 5.86 | 5.70 | 4.88 | **1.17×** |
| six_ar_arith_edge_chal | 6167 | 5167 | 1.19× | 5.81 | 5.69 | 4.88 | **1.17×** |
| seq_dep_chain4_scaled_edge_chal | 5765 | 5165 | 1.12× | 5.72 | 5.51 | 4.80 | **1.15×** |
| six_ar_indep_pool_chal | 6421 | 5177 | 1.24× | 5.87 | 6.06 | 5.28 | **1.15×** |
| seven_ar_seq_edge_chal | 6368 | 5168 | 1.23× | 5.65 | 5.63 | 4.98 | **1.13×** |
| six_ar_seq_edge_chal | 6167 | 5167 | 1.19× | 5.74 | 5.41 | 4.80 | **1.13×** |
| chained_ar_nested_edge_chal | 5563 | 5161 | 1.08× | 5.34 | 5.09 | 4.60 | **1.11×** |
| ar_scalar_chain_edge_chal | 5564 | 5164 | 1.08× | 5.13 | 5.49 | 5.00 | **1.10×** |
| four_ar_N224_edge_chal | 5765 | 5165 | 1.12× | 5.16 | 5.47 | 4.95 | **1.10×** |
| alternating_indep_ars_chal | 6166 | 5180 | 1.19× | 5.69 | 5.87 | 5.36 | **1.09×** |
| three_scaled_x_ars_chal | 5657 | 5178 | 1.09× | 5.39 | 5.47 | 5.02 | **1.09×** |
| eight_ar_half_ints_edge_chal | 6570 | 5170 | 1.27× | 5.77 | 6.03 | 5.59 | **1.08×** |
| six_ar_altsign_edge_chal | 6168 | 5168 | 1.19× | 5.71 | 5.44 | 5.08 | **1.07×** |
| five_ar_mixed_sign_edge_chal | 5966 | 5166 | 1.16× | 5.20 | 5.34 | 5.04 | **1.06×** |
| seq_dep_chain5_edge_chal | 5966 | 5166 | 1.16× | 5.37 | 5.27 | 4.98 | **1.06×** |
| ar_before_local_reduce_M128_chal | 5526 | 5160 | 1.07× | - | - | - | _floor_ |
| conditional_ars_chal | 5656 | 5178 | 1.09× | - | - | - | _floor_ |
| four_ar_evens_edge_chal | 5765 | 5165 | 1.12× | - | - | - | _floor_ |
| four_ar_indep_large_N_chal | 6128 | 5202 | 1.18× | - | - | - | _floor_ |
| four_ar_pow2_edge_chal | 5764 | 5164 | 1.12× | - | - | - | _floor_ |
| three_ar_frac_dep_edge_chal | 5564 | 5164 | 1.08× | - | - | - | _floor_ |
| three_inline_ars_chal | 5657 | 5178 | 1.09× | - | - | - | _floor_ |
| triple_ar_linear_edge_chal | 5564 | 5163 | 1.08× | - | - | - | _floor_ |

## F2. CSE across redundant ARs of the same input (7 anchors)

**What it is.** N syntactically distinct AR(x) calls on the SAME unmodified input, combined arithmetically. The N results are identical; N-1 collectives are pure waste. Sorcar hoists to a single AR(x) and replaces every other call with the hoisted value, collapsing the arithmetic to one scalar multiplier.

**Why strat stays at baseline's schedule.** XLA HLO CSE catches some assigned-first cases but not inline-call chains. Strat proposes payload/bucketing re-layouts of the N collectives; it never proposes 'these N collectives are the same value.' Same N-AR schedule as baseline.

**Per-problem data** (sim µs; warm-cache RT ms, 224 ranks):

| Problem | sim strat | sim sorcar | sim × | RT base | RT strat | RT sorcar | RT sorcar/strat |
|---|---|---|---|---|---|---|---|
| nine_ar_same_input_chal | 6948 | 5166 | 1.34× | 6.17 | 6.15 | 4.58 | **1.34×** |
| seven_scaled_diff | 6680 | 5178 | 1.29× | 5.73 | 6.16 | 4.66 | **1.32×** |
| csescaleddiff | 6169 | 5178 | 1.19× | 5.74 | 5.75 | 4.74 | **1.21×** |
| seven_ar_same_input_chal | 6680 | 5178 | 1.29× | 6.13 | 5.97 | 4.96 | **1.20×** |
| seven_scaled_input | 6680 | 5178 | 1.29× | 6.28 | 5.90 | 4.97 | **1.19×** |
| five_ar_scaled_same_input_chal | 6169 | 5178 | 1.19× | 5.66 | 6.33 | 5.34 | **1.18×** |
| four_ar_same_input_chal | 5914 | 5178 | 1.14× | 5.87 | 5.66 | 5.12 | **1.11×** |

## F3. Dead-collective elimination & algebraic zero (2 anchors)

**What it is.** Collectives whose results are provably unused, mathematically canceled, or reducible to a constant: alternating-sign sums that telescope to zero, gather-then-verify with a dead verify branch. Sorcar proves the cancellation and removes ALL collectives (sim cost -> 0).

**Why strat stays at baseline's schedule.** Strat scores collective structure, not the algebraic value, so it never proves the alternating sum cancels. It keeps every dispatch. Baseline schedule preserved.

**Per-problem data** (sim µs; warm-cache RT ms, 224 ranks):

| Problem | sim strat | sim sorcar | sim × | RT base | RT strat | RT sorcar | RT sorcar/strat |
|---|---|---|---|---|---|---|---|
| sequential_ar_chain_edge_chal | 5361 | 0 | ∞ | - | - | - | _HW-abort_ |
| three_group_dead_verify_chal | 6419 | 0 | ∞ | - | - | - | _HW-abort_ |

## F4. Per-row/col/batch dispatch collapse (5 anchors)

**What it is.** A per-row / per-col / per-batch / per-slice loop that issues one AR per slice of a 2D/3D tensor. Sorcar stacks the slices and issues ONE AR over the whole tensor (or a single reshaped AR), collapsing M dispatches to 1.

**Why strat stays at baseline's schedule.** Strat can re-bucket or re-order the per-slice ARs but does not fuse across the loop iteration space (the slices are separate SSA values). Baseline's M-dispatch loop is preserved.

**Per-problem data** (sim µs; warm-cache RT ms, 224 ranks):

| Problem | sim strat | sim sorcar | sim × | RT base | RT strat | RT sorcar | RT sorcar/strat |
|---|---|---|---|---|---|---|---|
| perslice3dM96 | 24446 | 5417 | 4.51× | 17.79 | 16.90 | 7.30 | **2.32×** |
| perrowM64N4K | 17938 | 5309 | 3.38× | 12.74 | 12.93 | 6.75 | **1.92×** |
| perrowM32N8K | 11538 | 5309 | 2.17× | 9.61 | 9.47 | 6.45 | **1.47×** |
| perbatchM32 | 11538 | 5309 | 2.17× | 9.26 | 9.31 | 6.37 | **1.46×** |
| perbatchM12 | 7418 | 5189 | 1.43× | 6.52 | 6.11 | 5.02 | **1.22×** |

## F6. Mixed-reduction-op extraction (1 anchors)

**What it is.** A payload reduced under one op (SUM) alongside the same or related payload reduced under a different op (MAX/MIN), issued as separate collectives. Sorcar extracts the mixed-op structure into the minimum distinct collectives.

**Why strat stays at baseline's schedule.** Strat keeps the SUM and MAX/MIN collectives as emitted; it does not extract the shared payload. Baseline schedule preserved.

**Per-problem data** (sim µs; warm-cache RT ms, 224 ranks):

| Problem | sim strat | sim sorcar | sim × | RT base | RT strat | RT sorcar | RT sorcar/strat |
|---|---|---|---|---|---|---|---|
| mixmaxmin | 8199 | 5362 | 1.53× | 6.95 | 6.77 | 5.30 | **1.28×** |

## F7. Slab/chunk payload fusion (1 anchors)

**What it is.** A tensor split into slabs/chunks, each all-reduced separately then recombined. Sorcar fuses the slabs into one contiguous payload and issues a single AR.

**Why strat stays at baseline's schedule.** Strat keeps the per-slab AR loop; it does not fuse the slab payloads. Baseline schedule preserved.

**Per-problem data** (sim µs; warm-cache RT ms, 224 ranks):

| Problem | sim strat | sim sorcar | sim × | RT base | RT strat | RT sorcar | RT sorcar/strat |
|---|---|---|---|---|---|---|---|
| eightslab | 6603 | 5164 | 1.28× | 5.91 | 5.87 | 4.94 | **1.19×** |

