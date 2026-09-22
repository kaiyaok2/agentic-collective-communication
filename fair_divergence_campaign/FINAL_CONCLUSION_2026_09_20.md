# Sorcar (kiss) vs Overlay (strat) — 15-Hour Fair-Gate Divergence Campaign

**Window:** 2026-09-20 08:30:11Z → 23:30:11Z (15 h autonomous; user +5h extension).
**Setup:** identical fp32 correctness gate on both pipelines; SYMMETRIC best-of-N
(both systems get N independent temperature draws; no RNG seeding, no response
caching); Sonnet 4.5 on both. CONFIRMED DIVERGENCE = best-of-N ratio ≥ 1.05 AND
Mann-Whitney U p < 0.05 (one-sided, tie-corrected) AND bootstrap 95% CI lower > 1.0.

## VERDICT: Sorcar > Overlay — a real, one-directional asymmetry (NOT parity)

The prior "SorcarCCL ≈ OverlayCCL" headline was a **problem-variety artifact** of
the earlier shallow/single-collapse problem set. Under a fair gate + symmetric
best-of-8 (headline wins re-confirmed at best-of-16) across 26 designed rounds:

- **15 CONFIRMED Sorcar > Overlay divergences. 0 confirmed Overlay > Sorcar.**
- Largest: `r22_su8_count16` best **2.40×** (overlay never escapes baseline on any
  of 9 seeds); `r9_deep8_big` best **2.54×** / median 2.35× CI[1.86, 2.37];
  `r2_deep8` best **2.18×** (best-of-16 **2.50×**) CI[1.87, 2.39].
- A dedicated reverse-direction round (r13) that deliberately BAITED kiss into
  over-engineering an already-minimal baseline produced **0 reverse signal** —
  kiss held at the optimum just like overlay. A reverse-lever probe (r18) on the
  one family where Overlay is *distributionally* more reliable (additive zero-sum
  folds) still **ties on best-of-N** — Overlay's edge never survives the symmetric
  criterion.

## Mechanism (established by code inspection + controls, refined across rounds)

Overlay enumerates K strategies **from the baseline framing** and refines the top
2 for a bounded R=3 rounds; any strategy failing its first correctness check is
discarded. This keeps Overlay **anchored to the baseline's collective structure**.
Kiss's open ReAct loop (~30 steps) is free to reframe.

The divergence lever is **algebraic depth-of-insight**, not payload, not syntactic
depth alone, not the collective primitive (Lessons L1–L8):

- **L4:** payload scaling does NOT widen the sim gap; the sim floors any single
  small collective at ~5160 µs, so headroom comes only from reducing collective
  COUNT/DEPTH.
- **L5:** divergence scales with **framing depth** — a baseline written as D
  dependent all_reduce stages that collapses to 1. Overlay stays pinned at ~D
  collectives; kiss collapses to 1–2. deep4 1.15× → deep8 2.18×.
- **L8 (keystone):** divergence requires the fused optimum to rest on a
  **globally-distributive** collapse that is INVISIBLE from the baseline's local
  framing (e.g. per-shard scale commuting through AR(SUM); ∑δ=0 cancellation;
  linearity telescoping). **Locally-visible identities always tie** because
  Overlay's best-of-8 sees through them: r8 telescope, r10 re-max, r10
  AG-roundtrip, r11 max-offset, r12 mixed-primitive all TIED.
- **Boundaries:** best-of-8 divergence has a depth threshold ~6–7 (r9_deep5 ties);
  the gate caps testable depth at 8 (resolve_passes=8); when the collective count
  is genuinely irreducible (r13 hetero SUM+MAX+MIN) both systems tie at the floor.

## Causal mechanism: CODE DEPTH is the anchor, description AMPLIFIES (r16 → r20 → r21 → r22 → r23 → r24 → r25 → r26 → r27)

A five-round causal chain, each holding the computation **byte-identical** and
varying **only the docstring** (r16/r20/r21/r22) then **de-confounding code vs
description** (r23), isolates what traps Overlay:

- **r16** — three problems, byte-identical confirmed deep-8 chain, docstring-only:
  `deepdoc` (narrates the 8 stages) pins overlay at the untouched baseline
  **12919 µs on every seed** → 2.38× (CONFIRMED best 1.374/med 1.908); `neutral`
  (result only) and `hintdoc` (hints fusion) → tie. **The trap is
  DESCRIPTION-driven, not code-driven.**
- **r20** — 2×2 (family × doc) at deep headroom: narration reproduces the trap on
  BOTH the multiplicative scale/unscale family (su8_narr CONFIRMED 1.56/1.94) AND,
  at the *distributional* level, on the additive zero-sum family (zs8_narr median
  flips from r18's *reverse* to *forward* 1.27, p<.05) — but the additive family
  ties on best-of-N because overlay's best seed still escapes.
- **r21** — decompose the narration into its two cues (the collective COUNT vs the
  per-stage PROCEDURE): naming the count ("8 all_reduce operations") is the STRONG
  trap (countonly CONFIRMED median 2.38); narrating the procedure without a count
  is a WEAK, bimodal trap (proconly CONFIRMED but median only 1.04).
- **r22** — the count cue is a **monotone dial** *when it matches the code*:
  `count1` (truthful minimal, code IS 1-AR) FREES overlay (screen 0.75, folds);
  `count8` traps (CONFIRMED 2.14 median); `count16` (overstated — code has 8) traps
  MAXIMALLY (CONFIRMED best **2.40**, overlay pinned at baseline on all 9 seeds). On
  the additive family the count cue is distributional-only (zs8_count8: median
  forward, best-of-N tie).
- **r23** — the DECISIVE de-confounding round: make the docstring count DISAGREE
  with the code. (a) `minimal_says8` — docstring lies "8 all_reduce operations" over
  the truly-minimal 1-AR code: overlay is **not fooled** (screen 1.04, stays at
  floor). (b) `deep8_says1` — docstring lies "a SINGLE all_reduce" over the deep-8
  chain: the false-low count rescues only **2/9** seeds → best-of-N ties (0.996) but
  the deep CODE keeps the median trapped (1.87, p<.05, overlay pinned on 7/9).

**The primary anchor is the ACTUAL CODE DEPTH; the stated collective count is a
SECONDARY modulator that only bites when it AGREES with the code.** Overlay's
enumerate-from-baseline is trapped by deep code that its bounded R=3 refinement
can't escape; a truthful high-count description AMPLIFIES the trap (r21/r22), a
truthful low count FREES it (r22 count1, truly-minimal code), but a description that
CONTRADICTS the code does not override what the code plainly shows (r23). Kiss's
open ReAct escapes regardless of both code framing and stated count. This is the
campaign's most precisely-bounded causal claim.

**Generality + layout (r24 → r25 → r26).** A three-round generality probe bounds
*which* deep code traps Overlay and shows the count cue's reach:
- **r24** — the trap is NOT any deep distributive collapse. Global-scalar homogeneity
  (c pulls out of AR(SUM)) and permutation-invariance (a cancelling roll) BOTH fold
  for Overlay at screen (ratio 1.0): these collapses are globally symmetric, so
  Overlay's enumerate factors them out. Only the RANK-HETEROGENEOUS per-shard
  scale/unscale chain (different a[r] per shard, invisible from any one rank) traps.
- **r25** — even among rank-heterogeneous multiplicative chains, LAYOUT matters under
  a result-only docstring: permuted contiguous shards keep Overlay's MEDIAN trapped
  (1.866), but INTERLEAVED lanes x[r::W] fold at screen (0.954) — Overlay recognizes
  the strided op as vectorizable and fuses it.
- **r26** — a TRUTHFUL count cue RE-ANCHORS Overlay onto fusible code: both layouts
  confirm at best-of-8 (~2.39×) once the docstring names "8 dependent all_reduce
  operations", including the interleaved layout that folded when unnarrated. But the
  best-of-16 recheck (r28) SPLITS them: the CONTIGUOUS permuted layout (perm_count8)
  HOLDS at best-of-16 (median 2.056, CI lower 1.862 — robust), while the INTERLEAVED
  layout (strided_count8) ESCAPES at best-of-16 (median stays 2.353 but Overlay's best
  of 16 draws reaches the floor). So the count cue's re-anchoring is real but the
  strided win is DRAW-FRAGILE — consistent with the interleaved layout being the one
  Overlay can fuse (it just needs enough draws). The robust count-cue win is the
  contiguous rank-heterogeneous layout. (No contradiction with r23: there the count
  was a LIE over MINIMAL code; here it is TRUE over deep code.)

- **r27** — the count cue's re-anchoring is BOUNDED to rank-heterogeneous code.
  Adding the same truthful "8 all_reduce" count to the r24 GLOBALLY-SYMMETRIC families
  does NOT rescue them (globalscale_count8 best-of-8 escapes, median 0.985 p .78;
  pairwise_count8 folds 0.853). So Overlay's fold on symmetric collapses is a
  code-structure fact — it sees the single global factor / the cancelling term from
  one rank, and no description overrides that — whereas its fusion of a
  rank-heterogeneous strided op IS description-suppressible (r26). This cleanly
  separates the two escape mechanisms and confirms the r24 symmetric nulls are real.

**Second-family probe (r33) — the trap is the MULTIPLICATIVE algebra, not "any deep
rank-heterogeneous collapse".** To test whether a *structurally distinct* collapse (not
per-shard diagonal scale) also traps Overlay, r33 held the deep-8 AR skeleton
byte-identical and swapped only the per-stage reversible operator:
- **`r33_shear8_count8`** — a purely OFF-DIAGONAL linear op (each stage couples adjacent
  shard pairs, even ← even+odd; NO diagonal scale). **TIE**: best-of-8 best 2.004 but
  median **1.0**, p 0.39, CI[0.93, 2.07]. Kiss folded on only 2/8 seeds; its *median*
  draw stayed at the baseline too. The order-dependent un-shear is hard for BOTH systems,
  so there is no robust asymmetry — the distinct-algebra probe does NOT reproduce the win.
- **`r33_permscale8_count8`** — a MONOMIAL op (per-shard multiplicative scale a[r] AND a
  half-rotation of shard positions). **CONFIRMED** at best-of-8 (best 2.012 / med 2.213 /
  p .0002 / CI[1.751, 2.379]) AND at best-of-16 (best 1.091 / med 1.866 / p .0082 /
  CI[1.019, 2.355]) → a **19th robust win**. But it confirms because it *retains the
  per-shard multiplicative scale*: it is family-1 WITH ROUTING, not a distinct algebra.

**Net: there is NO genuine second family.** Adding a permutation on top of the
multiplicative collapse PRESERVES the trap (permscale confirms); removing the diagonal
scale and using a purely off-diagonal linear op BREAKS it (shear ties). This SHARPENS the
r24 boundary rather than widening it: Sorcar's confirmed dominance is specific to the
rank-heterogeneous **multiplicative (diagonal) per-shard scale/unscale** collapse — the
one collapse invisible from any single rank yet linear through AR(SUM). The confirmed-win
roster is thus **19 robust / 0 reverse**, all one computational family (probed along six
axes: payload, depth, description, layout, scale constants, and now operator algebra).

So the robust confirmed-win regime is: **deep + rank-heterogeneous + multiplicative +
CONTIGUOUS-BLOCK collapse**. With a result-only docstring the median traps; a truthful
count cue also confirms at best-of-N. The interleaved/strided layout is the boundary:
a count cue confirms it at best-of-8 but Overlay escapes at best-of-16 (r28) — so it
is NOT counted among the robust wins. Globally-symmetric collapses stay
code-escapable regardless of description (r27). The mechanism thus has a clean
gradient of escapability — symmetric (escapes always) < strided rank-heterogeneous
(escapes at high draw count) < contiguous rank-heterogeneous (robustly trapped) — and
Sorcar wins robustly only at the trapped end.

## Distinct-family battery (r34–r37) — 5 genuinely different algebras, bo16-adjudicated

After the r33 shear probe (one distinct mechanism) the "no second family" claim was
justly criticized as under-powered. So a **breadth-first battery of five genuinely
distinct collapse mechanisms** was designed and screened, each a different algebra/
primitive (not a docstring/layout/constant variant of family-1), with survivors promoted
to the strict best-of-16 gate:

- **r34 tropical (max-plus / min-plus semiring)** — AR(MAX/MIN) + per-block additive
  offset. DIFFERENT semiring AND primitive; the reversible op is `+`, not `×`.
  **bo8 NOT confirmed** (max best 1.06, min best 0.79) — the additive inverse is trivial,
  Overlay folds. Behaves like the r14 additive family.
- **r35 masked partition-of-unity** — disjoint masks tile the vector, so D masked
  partial-sums accumulate to one AR(SUM). Combinatorial, not arithmetic scale.
  **bo8 TIE** (best 1.0 / med 1.06 / p .28) — the partition is locally visible; Overlay sees
  it and folds.
- **r36 affine (mult + additive)** — per-shard `a[r]·block + c[r]`. **bo16 CONFIRMED**
  (best 1.291 / med 2.162 / p=0.0 / CI[2.160, 2.169]) — HOLDS robustly. **But affine
  RETAINS a multiplicative factor**: its inverse still requires a division by `a[r]`. So
  this is not a new family — it is the multiplicative trap with an additive term riding
  along.
- **r37 group-theoretic block-rotation (NO scale)** — a D-deep chain of block rotations
  composing to one net rotation (symmetric-group product), collapsing to 1 AR(SUM) + a net
  permutation. This was the ONLY candidate with **no multiplicative component** — the sharp
  test. It passed bo8 (best 1.324 / CI[1.104, 1.705]) but **ESCAPES at best-of-16**
  (best 1.399 / med 1.704 / p 0.0061 / **CI_lo = 1.0** → CONFIRMED=False). With 16 draws
  Overlay reliably folds the pure permutation.

**Decisive outcome: the distinct-family battery VINDICATES the multiplicative-component
boundary rather than overturning it.** The only bo16 survivor (r36 affine) retains a
multiplicative factor; the one genuinely scale-free mechanism (r37 rotation) escapes the
strict gate, exactly like the pure-shear (r33) and pure-additive (r14/r34) probes. Across
**five distinct algebras — tropical semiring, combinatorial partition, mixed affine, pure
permutation, off-diagonal shear — no scale-free collapse yields a robust (bo16 CI_lo>1.0)
second family.** The r37 bo8→bo16 reversal is also a clean reminder that bo8 confirmation
is not robust and the strict CI_lo>1.0 gate is doing real work.

**Answer to the user's question — does a second divergence family exist?** Under strict,
symmetric, best-of-16 confirmation across genuinely distinct mechanisms: **No.** Sorcar's
robust dominance is specific to the rank-heterogeneous per-shard collapse with a
**multiplicative component** whose inverse is a division — now bounded not by one probe but
by a five-algebra battery. This is a stronger, better-powered version of the r33 conclusion.

**r39 (tuned pure-permutation) — the DECISIVE MECHANISTIC reason.** To test whether r37's
"distributionally faster but bo16-fragile" win could be tuned into a robust second family,
r39 made the net block permutation NON-closed-form: each stage rotates by k then applies a
fixed non-affine PI=[3,0,5,7,1,6,2,4], so Overlay's obvious "shift by c / apply PI once"
folds FAIL the fp32 gate (single-PI guess → max_diff≈4.4). At strict best-of-8: **count8**
Overlay pinned at baseline 6769 on 7/8, kiss folds to the true 5202 on 3/8 (p=0.0015,
CI_lo=1.005 — kiss distributionally faster) BUT Overlay's ONE folding seed kept 7 *local*
permutes (5424) → best-ratio 1.043 < 1.05 → NOT confirmed; **res** TIE (both fold ~2/8).
The crux: Overlay's fold reasons "after the first AR all ranks hold IDENTICAL data, so every
later AR-of-identical-data is the identity — keep the permutes LOCAL" — a *locally-visible*
identity (the L8 tie class). **Permutation complexity is irrelevant** because a pure
permutation between collectives operates on rank-identical data, so the fold is trivially
visible no matter how non-closed-form the net map is. Family-1 traps ONLY because per-shard
multiplicative scale makes the intermediate data *rank-heterogeneous*, breaking the
identity-shortcut and forcing a genuine non-local algebraic insight (scale distributes
through SUM). This structurally explains why r37 escaped bo16 and why the single-family
result is not sample-size-limited: no scale-free (pure-permutation) collapse is reachable as
a second family by tuning the permutation. Ledger key round_r39_pure_permutation_tuned;
artifacts in CONFIRMED_WINS_ARTIFACTS/second_family_candidate/.

**Artifact archive.** CONFIRMED_WINS_ARTIFACTS/ curates per-seed best_code + pipeline JSON +
.log trajectories for all 19 robust family-1 wins (robust_family1/), the 2 bo16-escape
draw-fragile cases (bo16_escape/), and the r37/r39 pure-permutation candidates
(second_family_candidate/) — 2275+ files, with MANIFEST_sims.json and README.md.

## Two-sided asymmetry — unequal in strength (r14, L9)

The asymmetry is **not** strictly one-directional at the *distributional* level:
- **Sorcar more reliable** on multiplicative scale/unscale depth (r1, r8) AND
  wins outright best-of-N on deep multiplicative chains (r2, r9).
- **Overlay more reliable** on additive zero-sum folds (r14 zerosum): overlay's
  enumerate naturally lists "drop the additive per-rank term"; kiss over-analyzes
  and stays at ~2 collectives on ~half its seeds.

**But the two sides are unequal in strength, and this is the crux:**
- Sorcar's advantage **survives best-of-N** — 9 CONFIRMED divergences where even
  Overlay's *best* of 8–16 seeds stays trapped.
- Overlay's advantage is **distributional-only** — 0 confirmed; on zerosum,
  best-of-N = 1.0 in both directions because Kiss's best seed always reaches the
  fold (reverse median 1.10–1.14, p 0.013–0.075, but best-of-N ties).

So: on the paper's **best-of-N criterion, Sorcar strictly dominates** (15–0). On a
**single-shot reliability criterion**, each system is more reliable on a different
problem class (Sorcar: multiplicative depth; Overlay: additive folds).

**Framing and algebra are two separate, composing factors (L10, r20/r22).** The
description's stated count sets the *median* trap on either algebraic family; the
algebra (multiplicative vs additive) sets whether that trap is *escapable at
best-of-N*. Multiplicative scale/unscale traps even overlay's best seed → confirms;
additive zero-sum is escapable by overlay's best seed no matter how strongly framed
→ distributional-only. This unifies the framing result (r16/r21/r22) with the
two-sided algebraic asymmetry (r14/L9) into one mechanism: **median = framing,
best-of-N escapability = algebra.**

## Honest scope of the claim

Sorcar's *confirmed* dominance holds **specifically and only** when a problem's
optimum requires collapsing removable collective depth to a **single** collective,
where the collapse rests on a non-locally-visible **multiplicative/distributive**
identity, and (r16) the problem description does not itself hand over the fusion.
On problems with no removable count (r13), removable-but-to-a-2-collective-floor
(r15), removable by a locally-obvious identity (r8/r10/r11/r12), or removable via
an additive zero-sum term (r14 — Overlay actually more reliable), the two systems
either tie on best-of-N or Overlay leads distributionally. There is **no** regime
where Overlay beats Sorcar on the best-of-N criterion. This is far sharper and
more defensible than either "they're equal" or "Sorcar always wins."

## Artifacts
- Harness: `/private/tmp/fair_diverge/{campaign.py,confirm_only.py,recompute_ci.py,synthesize.py}`
- Problems: `/private/tmp/acc_verify/search/problems_diverge_r{1..27,29,31}.py`
  (r29/r31 generality rounds run LIVE on Bedrock — see r29+ note below)
- Bedrock routing fix: `search/_anthropic_route.py` (`_post_bedrock`) +
  `experiments/ablation_kiss_vs_cc/kiss_token_shim.py` (AnthropicBedrock swap)
- Ledger: `campaign_ledger.json` — `robust_confirmed_divergences` (18) + `best_of16_recheck` + `generality_rounds_r29_r31`
- Per-round ledger + lessons (L1–L15): `CAMPAIGN_LOG.md`; snapshots `SYNTHESIS_SNAPSHOT_*.txt`
- Paper: UNTOUCHED (explicit constraint).

## On the r29+ "API cap" — a self-misdiagnosis, since corrected

New-round discovery briefly stalled at r29 when LLM calls returned HTTP 400
"workspace API usage limits… regain 2026-10-01". I initially recorded this as an
external hard blocker. **That was wrong.** The *direct* `ANTHROPIC_API_KEY` was
capped, but this session runs on **Bedrock** (`CLAUDE_CODE_USE_BEDROCK=1`), which is
NOT capped — the harness simply wasn't using it. Two code paths were hard-wired to
the direct API and both were fixed: overlay (`search/_anthropic_route._post_anthropic`
→ added a `_post_bedrock` branch) and kiss (its own `Anthropic()` client →
`kiss_token_shim.py` now swaps in `AnthropicBedrock` with a full recursive
`cache_control` strip). Both verified live on Bedrock. The r1–r28 verdict is
unaffected (those rounds predate the cap). Rounds r29/r31 were then re-run live —
results below.

## Generality rounds r29 / r31 (run live on Bedrock) — the trap GENERALIZES

To test whether the confirmed trap is tied to the specific 3-level {1, 1.5, 2}
scale constants or is a property of the *class* "contiguous rank-heterogeneous
multiplicative depth", two new constant sets were run live at symmetric best-of-8
(both pipelines genuinely on Bedrock):

- **`r31_lin5_count8`** (5 levels {1,1.25,1.5,1.75,2}, additive-linear, + count cue):
  **CONFIRMED at best-of-8** (best 2.494 / med 1.90 / p .0002) AND **at best-of-16**
  (best 2.395 / med 1.87 / p 0 / CI[1.862, 2.382]) → a genuine **18th robust win**.
- **`r31_lin5_res`** (same code, result-only doc): CONFIRMED at best-of-8
  (best 1.322 / med 1.16 / p .0006) but **escapes at best-of-16** (best .955 /
  med 1.004) — same draw-fragility as `r26_strided_count8`: a result-only docstring
  traps the median but overlay's best of 16 draws finds the fold.
- **`r29_pow2_count8`** (4 levels {0.5,1,2,4}, powers of 2, + count cue): median-forward
  at both best-of-8 (best 2.395 / med 1.935 / p .0053) and best-of-16 (med 1.87 / p .0006)
  but **NOT confirmed at either** — overlay's best of 16 draws ESCAPES (best 0.965). The
  per-shard unscale here is division by exact powers of 2 (0.5/1/2/4), a clean bit-shift
  overlay occasionally recognizes and fuses; the non-power-of-2 linear constants of r31
  do not fuse. **Mechanistic refinement: the trap needs an AWKWARD (non-power-of-2)
  per-shard unscale** — too-clean arithmetic lets overlay's best seed collapse it.
- **`r29_pow2_res`**: TIE (best .965 / med 1.438 / p .54).

**Conclusion: the confirmed trap is a property of the CLASS "deep + contiguous +
rank-heterogeneous + multiplicative scale/unscale with a non-trivially-fusible (non-
power-of-2) unscale," not of the specific 3-level {1,1.5,2} constants** — a different
5-level linear set (r31) reproduces a robust best-of-16 win, while an
exact-power-of-2 set (r29) is median-forward but best-of-N-escapable. This both kills
the single-constant-artifact objection AND sharpens the boundary: the arithmetic must
be awkward enough that overlay's enumerate-from-baseline won't one-shot the unscale.

## Confirmed-divergence roster (19 robust, best-of-N ratio / median / MW-p)
1.  r1_fold_s256        1.053 / 1.076 / .008
2.  r1_fold_s1024       1.153 / 1.161 / .001
3.  r1_fold_s4096       1.064 / 1.070 / .002
4.  r1_fold_s256_deep   1.101 / 1.062 / .0004
5.  r2_deep4            1.147 / 1.201 / .0005
6.  r2_deep8            2.179 / 1.870 / .0003   (best-of-16: 2.495)
7.  r2_deep6_big        1.245 / 1.640 / .0004
8.  r9_deep7            1.067 / 1.755 / .0018
9.  r9_deep8_big        2.542 / 2.350 / .0002
10. r16_deepdoc         1.374 / 1.908 / .0003   (framing causal)
11. r20_su8_narr        1.561 / 1.944 / .0002   (framing x deep family)
12. r21_su8_countonly   1.287 / 2.382 / .0002   (count cue, strong)
13. r21_su8_proconly    1.157 / 1.042 / .0076   (procedure cue, weak)
14. r22_su8_count16     2.395 / 1.939 / .0006   (overstated count, max trap)
15. r22_su8_count8      1.287 / 2.137 / .0002   (count=8 replicate)
16. r23_deep8_count8    2.482 / 2.382 / .0002   (de-confound control)
17. r26_perm_count8      1.140 / 2.353 / .0022   (permuted-shard layout + count;
    best-of-16 recheck (r28): best 1.10 / med 2.056 / p 0.0 CI[1.862,2.382] — HOLDS)
18. r31_lin5_count8      2.494 / 1.900 / .0002   (5-level {1,1.25,1.5,1.75,2} linear +
    count, live on Bedrock; best-of-16: best 2.395 / med 1.87 / p 0 CI[1.862,2.382] —
    HOLDS. Generality: trap is class-level, not tied to the 3-level {1,1.5,2} constants)
19. r33_permscale8_count8 2.012 / 2.213 / .0002  (monomial = multiplicative scale +
    half-rotation of shard positions, live on Bedrock; best-of-16: best 1.091 / med 1.866
    / p .0082 CI[1.019,2.353] — HOLDS. SECOND-FAMILY PROBE: confirms because it retains
    the multiplicative scale (family-1 + routing); the pure off-diagonal shear operator
    with NO diagonal scale, r33_shear8_count8, TIES — so this is NOT a distinct family)

Passed best-of-8 but ESCAPES at best-of-16 — reported for honesty, NOT counted:
  r26_strided_count8  bo8 2.393/2.382/.0007 CONFIRMED; bo16 best 1.0/med 2.353/.0001
    CONFIRMED=False. Interleaved layout: overlay's median stays trapped (2.35) but with
    16 draws its BEST seed finds the fusing solution -> fails best-of-N at N=16. This is
    itself evidence for L13'/L14: strided is precisely the layout overlay CAN fuse.
  r31_lin5_res        bo8 1.322/1.16/.0006 CONFIRMED; bo16 best .955/med 1.004/.207
    CONFIRMED=False. Result-only docstring (no count cue) over the 5-level linear chain:
    median traps at bo8 but overlay's best of 16 draws finds the fold — same draw-
    fragility as any result-only variant (cf. r24_deep8_res, r25). Reinforces that the
    TRUTHFUL COUNT CUE is what makes the rank-heterogeneous win best-of-16-robust.

Not confirmed but distributionally forward (median>1, p<.05, best-of-N ties —
these SUPPORT the two-factor mechanism, they are NOT reverse signals):
  r20_zs8_narr    med 1.273 p .0013  (narration flips additive family's median)
  r22_zs8_count8  med 1.029 p .0019  (count cue on additive family)
  r23_deep8_says1 med 1.873 p .0048  (deep code traps median; lying "1" rescues 2/9)
  r24_deep8_res   med 1.267 p .1995  (scale/unscale, result-only doc; best-of-N escapes)
  r25_perm_scale8 med 1.866 p .1139  (permuted layout, result-only doc; median trapped)

Generality / scope-bounding nulls (r24/r25 screens — SHARPEN the claim, not reverse):
  r24_globalscale8  screen 1.0   (global-scalar homogeneity — overlay factors it out)
  r24_pairwise8     screen 1.0   (permutation-invariance — overlay sees the roll cancel)
  r25_strided_scale8 screen 0.954 (interleaved lanes, result-only — overlay fuses)
  r27_globalscale_count8 best 1.0 med .985 p .78 (symmetric + count cue — still escapes)
  r27_pairwise_count8    screen 0.853 (symmetric + count cue — overlay folds)
  r33_shear8_count8      bo8 best 2.004 med 1.0 p .39 CI[.93,2.07] (SECOND-FAMILY probe:
     pure OFF-DIAGONAL linear op, NO diagonal scale — TIE. kiss folds only 2/8, median
     stays at baseline. The order-dependent un-shear is hard for BOTH systems -> no
     robust asymmetry. Removing the multiplicative scale BREAKS the trap; cf. r33_permscale
     which keeps the scale + adds routing and CONFIRMS. Confirms the trap is the
     MULTIPLICATIVE algebra specifically, not any deep rank-het collapse.)
  -> confirmed regime = deep + rank-heterogeneous + multiplicative collapse; when the
     docstring is result-only the layout must also be contiguous-block for the MEDIAN
     to trap, but a TRUTHFUL count cue (r26) makes the win layout-robust FOR RANK-
     HETEROGENEOUS code. The count cue does NOT rescue globally-symmetric collapses
     (r27): overlay's fold on those is a code-structure fact, not description-driven.
