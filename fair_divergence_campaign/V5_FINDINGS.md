# v5 hard-cancellation family — findings

**Goal:** design problems Sonnet 4.5 CANNOT get right on the first try, to
produce MORE fair-gate divergences (ideally a family), following hd10's recipe:
a deep chain of scaled all_reduces whose true optimum is
`coeff_vector * all_reduce(SUM, x)` (ONE collective), where reconstructing the
fused coefficient is error-prone.

## Result (fair fp32 gate, best-of-3 kiss seeds, 7-node)

| problem | chain | overlay | kiss | ratio | verdict |
|---|---|---:|---:|---:|---|
| hd16_product_scale_chain | 5 AR, coeff = ∏ per-stage scales | 5775.8 | 6160.7 | **0.938** | **kiss LOSES** |
| hd18_sign_telescope_chain | 4 AR, coeff = block-parity sign × mag | 6000.2 | 5883.5 | 1.020 | near-tie |
| hd19_mixed_sum_max_chain | SUM/MAX/SUM (MAX-of-replicated = identity) | 5235.3 | 5175.0 | 1.012 | near-tie |
| hd20_modular_coeff_chain | 3 AR, coeff[i]=base[i%7], period∤S | 5205.7 | 5204.0 | 1.000 | tie |

(hd17_intrablock_ramp_chain dropped pre-run: its 1-collective optimum is
*slower* in sim than the baseline — no headroom.)

**No new ≥1.05× divergence. One problem (hd16) is actually a kiss LOSS.**

## Why — the sharp lesson about what made hd10 diverge

hd10 was not "hard." It was **asymmetrically hard**: hard enough that
OverlayCCL's *enumerate-once* failed every optimizing strategy on its cold
one-shot, but *recoverable by iteration* so kiss's ReAct loop repaired it. That
gap is the entire divergence.

The v5 problems miss that window in three distinct ways:

1. **hd16 — too hard for BOTH → kiss loses.** The optimum coefficient is a
   4-stage cumulative *product* per block. Neither system reasoned it through:
   overlay's "fused" strategy failed the gate and it refined the baseline to
   5776; kiss (all 3 seeds) also failed to collapse and stayed at ~6160,
   *worse* than overlay's refined baseline. When difficulty exceeds BOTH
   systems' reach, the outcome is noise — and here it favored overlay's
   refinement. This is an honest anti-result: cranking difficulty does NOT
   monotonically favor the iterative loop.

2. **hd19 / hd20 — crackable by BOTH → tie.** The MAX-of-replicated=identity
   trick (hd19) and the misaligned modular coefficient (hd20) look tricky but
   are single realizations. Overlay one-shot or one-refine'd them (5235, 5205);
   kiss vectorized them cleanly (5175, 5204). Both land at the optimum → tie.
   The kiss edge (1.01×) is within sim noise, not structural.

3. **hd18 — partial for BOTH → near-tie.** The block-parity sign pattern is
   moderately hard; both systems found *partial* collapses (overlay 6000, kiss
   5883, optimum 5204) but neither reached it. Partial-vs-partial is a coin
   flip, not a divergence.

## The refined model of when Sorcar > Overlay under a fair gate

Divergence needs the optimum to sit in a **narrow difficulty band**:

    cold one-shot FAILS  (so enumerate-once discards the strategy)
        AND
    iterative repair SUCCEEDS  (so the ReAct loop recovers it)

hd10 hit that band. Making problems *uniformly harder* (v5) overshoots it: the
optimum becomes unreachable for the iterative loop too, collapsing the gap (or
even inverting it, hd16). The lever is real but **narrow and hard to target on
purpose** — it depends on the optimum being right at the edge of Sonnet's
one-shot ability, which is not something a problem's structural depth controls
monotonically.

## Recommendation

The honest, well-supported headline stands: under a fair gate SorcarCCL ≈
OverlayCCL, with a single narrow structural lever (route-around-rejection,
hd10, 1.19×) rather than a family. v5 is strong negative evidence that the
lever does not generalize into a family by increasing difficulty. If a *family*
is needed for the paper, it would have to be engineered by calibrating each
problem to Sonnet's exact one-shot boundary (fragile, model-version-dependent)
— I'd advise against resting a claim on it. Paper untouched (per instruction).

## Artifacts
- Problems: `/private/tmp/acc_verify/search/problems_hard_diverge_v5.py`
- Reports: `/private/tmp/fair_diverge/results_v5/report.json`
- Optimum validation: `/private/tmp/fair_diverge/validate_v5.py` (headroom check)
- Per-problem traces: `results_v5/<problem>/{overlay,kiss_s*}/`
