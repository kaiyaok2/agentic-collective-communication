# Fair-comparison divergence hunt — findings

**Question asked:** design VERY HARD problems (harder/trickier than the 55-set)
where SorcarCCL (kiss) and OverlayCCL (strat) REALLY diverge under a FAIR
comparison, so the gap is a genuine search-shape effect, not a gate artifact.

**Fair = identical correctness gate for both pipelines.** Both call one
`score_service_fair.py` with a fixed `GATE_MODE=fp32` (`test_xla_candidate_generic`,
atol=1e-5). This neutralizes the bf16-gate asymmetry (strat ran an extra
`test_xla_candidate_bf16` that kiss's service did not) that manufactured the
old 55-problem "strat==baseline" divergence. A guard also symmetrically rejects
the `reduce_scatter+all_gather` scorer artifact (sim≈0µs, below the physical
single-collective floor) so neither side can win by stumbling into it.

## Result: under a fair gate, they TIE on 12 of 13 hard problems.

| problem | overlay µs | kiss µs | ratio | diverge |
|---|---:|---:|---:|:--:|
| hd10_staged_shard_norm | 6172.2 | 5204.0 | **1.186** | **YES** |
| hd8_blockwise_rs_bcast | 5394.5 | 5194.3 | 1.039 | no |
| hd9_topk_mask_sum | 5381.0 | 5360.7 | 1.004 | no |
| hd1_staged_dead_fold | 5160.7 | 5160.7 | 1.0 | no |
| hd2_perblock_mixed_fold | 5161.0 | 5161.0 | 1.0 | no |
| hd3_rs_ladder | 5160.0 | 5160.0 | 1.0 | no |
| hd4_dead_slab_linear | 5176.0 | 5173.6 | 1.0 | no |
| hd5_telescoping | 5160.7 | 5160.7 | 1.0 | no |
| hd6_sharded_gram_diag | 5160.7 | 5160.7 | 1.0 | no |
| hd11_transpose_gather_interleave | 5163.7 | 5162.0 | 1.0 | no |
| hd12_colsum_scatter_2d | 5174.3 | 5174.3 | 1.0 | no |
| hd14_big_rs_bcast | 7026.9 | 7026.9 | 1.0 | no |
| hd15_gather_redundant_reduce | 5177.8 | 5177.8 | 1.0 | no |

**13 problems, 1 divergence ≥1.05× under the fair gate.**

## Why they tie almost everywhere

These problems were built around every structural lever I could find that
*should* separate an open-ended ReAct loop (kiss) from enumerate-K→refine-top-2
(overlay): dead-term cancellation (hd1), 48-collective per-row folds (hd2),
telescoping prefix sums (hd5), error-prone transpose/interleave reshapes
(hd11), reduce_scatter scatter_dim traps (hd12), and large bandwidth-bound
payloads (hd14, hd15).

On all of them **Sonnet 4.5 one-shots the optimum inside OverlayCCL's
enumerate+implement step.** Whether the optimum is a clean structural idea
(hd2: 48→1 collective), an algebraic fold (hd1 dead terms, hd5 telescoping),
or an error-prone reshape (hd11), the model gets it right on the first
implementation, so there is nothing left for kiss's iteration to recover. The
model is simply good enough that the *shape* of the search (open ReAct vs.
enumerate-once) stops mattering once both see the same correctness gate.

## The one real divergence — hd10, and why it's honest

`hd10_staged_shard_norm`: baseline is a 3-stage AR chain with a per-shard
scaling coefficient applied between stages. The true optimum collapses it to
**one all_reduce times a local per-shard scale vector** (5204 vs 6172µs).

Under the identical fp32 gate, OverlayCCL enumerated 5 strategies. Four of them
were the *optimizing* ones (fused two-stage, all-gather+local, reduce-scatter
decomposition, single-AR-with-prescale) — and **all four failed their cold
one-shot implementation** (max_diff 3.7–4.7: the model got the per-shard scale
indexing wrong on the first try). OverlayCCL's rule discards any strategy whose
first implementation fails the gate (phase3 line 357-359), so only the
*baseline* survived into refine-top-2 — and refinement made it slightly worse
(6711µs by round 2). Kiss, seeing the same gate's error text, iterated the
tricky single-AR+local-scale idea to correctness over 23 score calls (19
passed) and landed the 5204µs optimum.

This is the **L3 "route-around-rejection" lever** and it's the only one that
survives a fair gate: when the optimum is *hard to implement correctly on the
first attempt*, enumerate-once can lose the whole strategy while an iterative
loop repairs it. hd8 (1.039×, kiss deletes a redundant shard round-trip the 5
strategies all preserved) and hd9 (1.004×, CSE) are the same mechanism but
sub-threshold because the sim floors small collectives at ~5160µs.

**hd15 was designed to amplify hd8/hd10 with a large payload (1.40× headroom
in principle) — but it tied.** With a big buffer the redundant-reduce optimum
became *obvious* enough that overlay one-shot it too. The lever only fires when
the optimum is simultaneously (a) cheaper and (b) error-prone to implement
cold. Large payloads make (a) bigger but tend to kill (b).

## Bottom line

Under a genuinely fair (identical) correctness gate, **SorcarCCL ≈ OverlayCCL
on hard problems.** The only reproducible divergence is the route-around-
rejection effect (hd10, 1.19×): a real, defensible search-shape advantage for
open-ended iteration, but narrow — it needs an optimum that is both cheaper and
first-attempt-error-prone. This is consistent with, and strengthens, the
earlier finding that the 55-problem "divergence" was a bf16-gate artifact
rather than a search-shape effect.

I did **not** touch the paper (per instruction). Recommendation for the
check-in: hd10 is a clean, honest, fair-gate example if a single-problem
divergence illustration is wanted; the broader honest headline is parity.

## Artifacts
- Harness: `/private/tmp/fair_diverge/score_service_fair.py` (symmetric gate + artifact guard)
- Drivers: `run_kiss_fair.py`, `run_overlay_fair.py`; orchestrators `orchestrate_v{1..4}.py`
- Problems: `/private/tmp/acc_verify/search/problems_hard_diverge_v{1..4}.py`
- Reports: `results_v{1..4}/report.json`; hd10 traces in `results_v2/hd10_staged_shard_norm/`
- hd3 clean re-run: `results_hd3_clean/` (all 3 seeds 5160.0 = tie; original 0.0 was the guarded artifact)
