# Sorcar vs Strat: Family Taxonomy of the 142-Problem Pool

**Scope**: the full RT-verified pool from `SORCAR_VS_STRAT_7NODE_RESULTS.md`
(18 Cat-A + 51 Cat-C + ~74 extras, warm-cache RT on 7× trn1.32xlarge, 224
ranks). This doc re-classifies every problem — including all extras — into
**7 optimization families**, and for each family documents (1) what the
family is, (2) Sorcar's general solution, and (3) why strat-enumerate stays
at baseline.

The original results docs used 6 buckets (Cat-A, C1, C2, C3, C4–C6,
Extras). The "Extras" bucket was a chronological catch-all, not a
structural family: on inspection, ~62 of the 74 extras are scale/op
variants of the existing five families, and the remaining ~12 form **two
genuinely new families** (F6 mixed-reduction extraction, F7 slab fusion).

## Family index

| # | Family | Problems | Win range | Representative |
|---|---|---|---|---|
| F1 | Sequential-AR linearity | 22 | 1.05–1.39× | six_ar_arith (1.28×) |
| F2 | CSE across redundant ARs | 26 | 1.08–4.30× | sixtyfourinline (4.30×) |
| F3 | Dead-collective elimination & algebraic zero | 17 | 1.04–8.44× | algzero20 (8.44×) |
| F4 | Per-row/col/batch dispatch collapse | 61 | 1.36–174.60× | perrowM2048 (174.60×) |
| F5 | Collective-type conversion & data-flow narrowing | 11 | 1.02–8.82× | ag_slice_use (8.82×) |
| F6 | Mixed-reduction-op extraction | 2 | 1.66–13.50× | perrow_mixed_bigM (13.50×) |
| F7 | Slab/chunk payload fusion | 3 | 1.07–1.41× | eightslab (1.41×) |

Total: 142 problems, 137 wins ≥5%, 6 borderline ties, 0 losses.

---

## F1. Sequential-AR linearity (22 problems)

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

## F2. CSE across redundant ARs of the same input (26 problems)

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

## F3. Dead-collective elimination & algebraic zero (17 problems)

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

## F4. Per-row/col/batch dispatch collapse (61 problems — the largest family)

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

## F5. Collective-type conversion & data-flow narrowing (11 problems)

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
mid-range: 1.06–1.21×, plus 8.82× on ag_slice_use where the entire AG is
droppable.

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
