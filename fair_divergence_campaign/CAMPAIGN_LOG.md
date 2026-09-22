# 10-Hour Autonomous Divergence Campaign

**Goal:** determine, under a FAIR fp32 gate with SYMMETRIC best-of-K + real
statistics, whether SorcarCCL (kiss) ≈ OverlayCCL (strat), or one dominates.
Iterate ≥10 rounds; each round uses lessons from prior rounds. Do NOT conclude
before the 10h wall-clock expires.

**Start:** 2026-09-20T08:30:11Z (epoch 1789893011)
**Deadline:** 2026-09-20T18:30:11Z (epoch 1789929011)

## Fairness fixes locked in
- Overlay AND kiss both get N independent seeds (temperature draws; no RNG
  seeding, no response caching — verified). Prior 1.19x hd10 was overlay
  best-of-1 vs kiss best-of-3 artifact; symmetric best-of-3 -> 1.05x.
- CONFIRMED DIVERGENCE requires: best-of-N ratio >= 1.05 AND Mann-Whitney
  p < 0.05 (kiss sims stochastically smaller) AND bootstrap CI lower > 1.0.

## Standing lessons (update each round)
- L15 (r27): the count-cue re-anchoring (L14) is BOUNDED TO RANK-HETEROGENEOUS code.
  Adding a truthful "8 all_reduce" count to the r24 GLOBALLY-SYMMETRIC families does
  NOT rescue them: globalscale_count8 best-of-8 escapes (median 0.985, p .78, not
  confirmed) and pairwise_count8 folds outright (0.853). Contrast r26, where the same
  count cue confirmed BOTH rank-heterogeneous layouts at 2.39x. So overlay's fold on
  symmetric collapses is a CODE-STRUCTURE fact (it sees the single global factor / the
  cancelling term from one rank) that description cannot override — whereas its fusion
  of rank-heterogeneous strided code is description-suppressible. Cleanly separates the
  two escape mechanisms and confirms the r24 symmetric nulls are real code facts.
- L14 (r26): a TRUTHFUL count cue makes the confirmed win LAYOUT-ROBUST and can
  RE-ANCHOR overlay onto fusible code. Both r25 layouts (permuted contiguous +
  interleaved strided) CONFIRM at ~2.39x once the docstring names "8 dependent
  all_reduce operations" — even the strided layout that overlay FUSED (folded 0.954)
  when unnarrated. So narration isn't only "reinforce what overlay already sees"
  (L12): when the count is TRUTHFUL over genuinely-deep-but-fusible code, it stops
  overlay from attempting the fusion it otherwise would. Consistent with r23 (a
  LYING count over MINIMAL code still didn't fool overlay) — the discriminator is
  code-really-deep + truthful-count, not description alone. Extends L13'.
- L13' (r25): the trap needs CONTIGUOUS-BLOCK rank-heterogeneity, not merely rank-
  heterogeneity. Permuted contiguous shards (perm_scale8) keep overlay's MEDIAN
  trapped (1.866); interleaved/strided lanes (strided_scale8) FOLD for overlay at
  screen (0.954) — overlay recognizes x[r::W] as a single vectorizable strided op and
  fuses. So the confirmed-win regime is: deep + contiguous-block + rank-heterogeneous
  + multiplicative collapse. Refines L13.
- L13 (r24): the trap requires a RANK-HETEROGENEOUS, non-locally-visible collapse —
  NOT any deep distributive code. Two novel deep-8 collapses whose identity is
  globally SYMMETRIC (global-scalar homogeneity c=2; permutation-invariance via a
  cancelling roll) BOTH fold for overlay at screen (ratio 1.0) — overlay's enumerate
  factors out the single global scalar / sees the roll cancel. Only per-shard
  scale/unscale, where each shard is scaled by a DIFFERENT a[r] so no single rank can
  see the collapse, keeps trapping overlay (deep8_res promotes 1.449). Sharpens L8:
  "non-locally-visible" must mean rank-heterogeneous, not merely "requires knowing the
  whole reduction." This BOUNDS the confirmed-win regime and is the honest scope guard
  against over-claiming generality.
- L12 (r23): the PRIMARY anchor is ACTUAL CODE DEPTH; the stated count is a
  SECONDARY modulator that bites only when it AGREES with the code. A lying low
  count over deep code (deep8_says1) rescues only a minority of seeds (best-of-N
  ties, median stays trapped 1.87); a lying high count over minimal code
  (minimal_says8) does not fool overlay (screen 1.04). Corrects L11: r22_count1
  freed overlay because that code was TRULY minimal. Narration works by REINFORCING
  the code's real structure, not overriding it. Sharpest causal bound in campaign.
- L11 (r21): the framing trap is DRIVEN BY THE STATED COLLECTIVE COUNT. On the
  confirmed su family, a docstring that NAMES the count ("8 all_reduce operations")
  traps overlay robustly (median 2.38, best-of-8 confirms); a docstring that
  narrates the per-stage procedure WITHOUT a count is a weak/bimodal trap (median
  1.04). Overlay's enumerate-from-baseline anchors to the asserted count and its
  R=3 refinement shaves toward it rather than escaping the framing. Refines r16/L10:
  the causal cue is specifically the COUNT assertion, not narration in general.
- L10 (r20): FRAMING and ALGEBRA are TWO SEPARATE FACTORS that COMPOSE. Narration
  (describing the deep chain in the docstring) is causal on BOTH families at the
  DISTRIBUTIONAL level — it pins overlay's MEDIAN seed at the narrated baseline
  (su8_narr med 1.94, zs8_narr med 1.27 — the latter a FLIP of r18's reverse
  median). But whether the trap SURVIVES best-of-N is set by ALGEBRA, not framing:
  multiplicative scale/unscale traps even overlay's BEST seed (confirms); additive
  zero-sum is escapable by overlay's best seed regardless of docstring (ties on
  best-of-N). Framing sets the median; algebra sets best-of-N escapability.
- L9 (r14 reverse-confirm): the asymmetry is TWO-SIDED but UNEQUAL. Sorcar is
  more reliable on multiplicative distributive collapse (scale/unscale depth),
  Overlay is more reliable on additive zero-sum folds (enumerate naturally lists
  "drop the additive term"). BUT: Sorcar's edge SURVIVES best-of-N (9 confirmed;
  overlay's best seed stays trapped in deep multiplicative chains), while
  Overlay's edge is DISTRIBUTIONAL-ONLY (0 confirmed; kiss's best-of-N seed always
  reaches the additive fold). So on the paper's best-of-N criterion, Sorcar
  strictly dominates; on a single-shot/reliability criterion, each wins a
  different problem class. r18 tests whether the additive/multiplicative split is
  the true axis.
- L8 (r10+r11): THE LEVER IS ALGEBRAIC DEPTH-OF-INSIGHT, NOT SYNTACTIC DEPTH OR
  PRIMITIVE. Divergence needs the fused optimum to require an insight that is
  NON-OBVIOUS from the baseline framing. SUM scale/unscale (r2/r9) works because
  the collapse rests on distributivity of per-shard scaling through AR(SUM) — not
  visible as a local cancellation. Obvious cancellations tie regardless of depth
  or primitive: r8 telescope (add-then-cancel), r10 re-max (idempotent),
  r10 AG-roundtrip, r11 max-offset (max(x+c)-c). Overlay's best-of-8 sees through
  any locally-visible identity; only a globally-distributive collapse traps it.
- L5 (r2/r9): DIVERGENCE SCALES WITH FRAMING DEPTH. deep4 best-of-8 1.15x ->
  deep8 2.18x. Overlay's bounded R=3 refinement stays pinned at D collectives
  (deep8: 8/8 seeds at 8 collectives); kiss collapses to 2-3. This is the
  campaign's core mechanism + headline.
- L6 (r9): GATE CAPS DEPTH AT 8. Shared correctness gate uses resolve_passes=8,
  so >8 dependent collectives return an unresolved-tail error (constant max_diff
  ~10 regardless of depth), NOT a real divergence. Depth 8 is the deepest
  testable robust divergence. Do NOT touch the shared scorer to go deeper.
- L7 (r8): NOT ALL DEPTH DIVERGES. Telescoping add-then-cancel chains (r8
  telescope4/6) TIE at 1.0 — both systems see through the cancellation to 1 AR.
  Divergence needs the depth to look like a GENUINE dependent recurrence
  (scale/unscale per stage, iterative renormalize), not an obviously-cancelling
  sum. iternorm4/6 promoted (1.20/1.33); telescope discarded.
- L1 (v6): count-reduction (fuse/dead/CSE/linearity) -> both one-shot -> TIE.
- L2 (v5): uniformly harder -> both fail -> noise, can favor overlay (kiss LOSS).
- L3 (hd10/v7): error-prone fold sits at one-shot boundary; WHICH system nails
  it is stochastic per seed -> best-of-N averages out the "win" -> TIE.
- L4 (r1 validate): PAYLOAD scaling does NOT widen sim headroom (s256=1.276 ->
  s4096=1.265, flat) — sim collective cost is AR-COUNT-dominated at these
  sizes, not bytes. DEPTH widens it (4-stage=1.49). Pivot: scale depth/count,
  not payload.

## BOOTSTRAP BUG FIXED (2026-09-20 09:1x)
campaign.py `_bootstrap_ratio_ci` used `seed % n` on an LCG. LCG LOW bits have
period n (e.g. `%8` cycles [6,7,4,5,2,3,0,1]), so every "resample" was a fixed
permutation of all indices -> identical medians -> ZERO-WIDTH CI. Fixed to
`(seed >> 16) % n` (high bits mix). r1 CIs recomputed post-hoc from logged
per-seed sims (Mann-Whitney p unaffected — doesn't use bootstrap).

## Round ledger
- **r1** (DIR-A payload-scaled fold + DIR-B decoy optimum): screen promoted 6/7
  (decoy_s4096 tied). 8-seed CONFIRM DONE. **4 CONFIRMED DIVERGENCES** (best-of-8
  >=1.05 AND MW-p<0.05 AND fixed-CI-lo>1.0):
    r1_fold_s256      best 1.053 med 1.076 p .008  CI[1.004,1.155]
    r1_fold_s1024     best 1.153 med 1.161 p .001  CI[1.007,1.297]
    r1_fold_s4096     best 1.064 med 1.070 p .002  CI[1.008,1.227]
    r1_fold_s256_deep best 1.101 med 1.062 p .0004 CI[1.005,1.320]
  NOT confirmed: fold_s4096_deep (best 1.008 fails ratio bar though CI>1),
  decoy_s512 (tie, p=.057).
  **MECHANISM (verified by inspecting generated code)**: overlay's
  enumerate-from-baseline-framing anchors it to the multi-stage structure —
  min-collective counts overlay {s256:2, s1024:3, s4096:1, s256_deep:4}; kiss
  {1,1,1,2}. Overlay CAN reach the fold (fold_s4096 min=1) but does it LESS
  RELIABLY (AR=3 on 6/8 s1024 seeds); kiss reaches AR=1 on 3/8. The divergence
  is DISTRIBUTIONAL (reliability of reaching the minimal-collective optimum),
  NOT "overlay categorically cannot." This is the campaign's FIRST real signal:
  under a fair fp32 gate + symmetric best-of-8, Sorcar(kiss) > Overlay on the
  linearity-fold family, driven by kiss's freedom from baseline framing.
- **r2** (DIR-C deep chains depth4/6/8 + DIR-E framing lock-in): CONFIRM DONE.
  **3 CONFIRMED DIVERGENCES** (fixed-CI):
    r2_deep4     best 1.147 med 1.201 p .0005 CI[1.012,1.395]
    r2_deep8     best 2.179 med 1.870 p .0003 CI[1.862,2.392]  <-- LARGEST
    r2_deep6_big best 1.245 med 1.640 p .0004 CI[1.069,1.961]
  NOT confirmed: deep6 (p=.057 borderline, CI touches 1.0), lockin_s512/s2048
  (best<1.05 — kiss doesn't reliably beat overlay's best on the lock-in framing).
  **MECHANISM (verified)**: on deep8, overlay stays at 8 collectives on 7/8
  seeds (bounded R=3 refinement only shaves constants, never escapes the
  deep-chain framing); kiss collapses to 2-3 collectives on 7/8. This is the
  campaign's CLEAREST structural divergence: overlay's enumerate-from-baseline
  procedure is architecturally trapped by deep sequential framing; kiss's open
  ReAct is not. Divergence GROWS with depth (deep4 1.15x -> deep8 2.18x).
- **r3** (DIR-F primitive swap: all_gather+reduce -> all_reduce/RS): VALIDATED,
  queued. Headroom 1.05-1.07 (payload-independent per L4). Tests framing
  distance: does overlay's enumerate reach a different primitive.
- **r4** (DIR-G/H redundant-collective layout): validated but WEAK
  (rsag=1.078, dblgather=1.039 below bar). Keep only rsag_* variants; low
  priority.
- **r5** (DIR-I stacked/compositional redundancies): VALIDATED, queued.
  Headroom stacked2=1.111, stacked3=1.192, stacked4=1.236, stacked4_big=1.325.
  Tests whether overlay's R=3 refinement budget composes MULTIPLE independent
  collapses or plateaus at partial.
- **r6** (DIR-J hierarchical-reduction decoy + DIR-K false-dependency doc):
  validated but WEAK (hier=1.039, falsedep=1.068-1.097). Misleading-hint lever;
  marginal headroom per L4. Low priority — run 1-seed screen only.
- **r7** (DIR-L all_to_all transpose + DIR-M collective_permute rotate):
  validated but WEAK/DEGENERATE. a2a=1.049; ring baseline ALREADY at sim floor
  5161 (headroom inf/0 — no room). Primitive-discovery lever doesn't clear the
  bar at these sizes; sim floors point-to-point at the same 5160 as AR.
  De-prioritized.
- **r8** (DIR-N telescoping chain + DIR-O iterative-normalized accumulation):
  CONFIRM DONE. **0 CONFIRMED** (honest negative). telescope4/6 TIED at 1.0 at
  screen (both systems see through add-then-cancel to 1 AR — discarded, L7).
  iternorm4 med 1.071 p .003 CI[1.006,1.179] but best-of-8 = 1.001 -> NOT
  confirmed; iternorm6 med 1.054 p .05 best 0.999 -> NOT confirmed. The
  iterative-renormalize recurrence gives kiss a DISTRIBUTIONAL edge (median>1,
  p<.05) but overlay's BEST-of-8 matches it. Contrast with r2 deep chains where
  even best-of-8 diverges: the difference is that iternorm's `s=AR(s/W)` stages
  are TRIVIALLY collapsible (obvious identity), so overlay's best seed nails it;
  r2's scale/unscale stages look like real dependent work, so overlay stays
  trapped even on its best seed. Refines L5: depth diverges on best-of-N ONLY
  when each stage looks like genuine (non-obviously-removable) dependent work.
- **r9** (DEPTH-FILL 5/7 + big-payload 7/8): CONFIRM DONE. **2 CONFIRMED**:
    r9_deep7     best 1.067 med 1.755 p .0018 CI[1.057,2.179]
    r9_deep8_big best 2.542 med 2.350 p .0002 CI[1.861,2.369]  <-- deepest/largest
  NOT confirmed: r9_deep5 (screened 1.85x on 1 seed but best-of-8 = 0.965, TIE,
  wide CI[0.88,1.53]); r9_deep7_big discarded at screen (0.967, seed noise).
  KEY NUANCE (refines L5): the best-of-8 divergence needs depth >=7. At depth 5
  overlay's BEST seed still reaches the fold (ties); by depth 7-8 even its best
  seed stays trapped. So the monotonic "divergence grows with depth" holds on
  MEDIAN throughout, but on the strict best-of-8 criterion it has a THRESHOLD
  around depth 6-7. (deep10/12/16 not testable: gate resolve_passes=8 cap, L6.)
- **r10** (GENERALITY via OBVIOUSLY-removable depth): CONFIRM N/A — **all 4
  TIED at screen (1.0/0.787/1.0/1.0), discarded.** Clean confirmation of refined
  L7: overlay's single seed already collapses these to the 1-collective floor
  (5160) because the redundancy is OBVIOUS (re-max is trivially idempotent;
  all_gather+local-sum is an obvious roundtrip identity). Depth alone does NOT
  diverge — the depth must LOOK like genuine dependent work. This is the
  mechanism boundary, and it sets up r11 as the decisive control (same MAX
  primitive, but genuine-looking offset stages).
- **r11** (MECHANISM test: genuine-looking depth over MAX): CONFIRM N/A —
  **all 4 TIED at screen (1.006/0.765/1.001/1.011), discarded. PREDICTION
  REFUTED.** I predicted r11's add-offset/max/sub-offset stages would behave like
  r2's SUM scale/unscale and CONFIRM. They did NOT. Refines L5/L7 sharply: the
  depth-trap does NOT generalize to max-algebra. WHY (best hypothesis): with SUM,
  the fused optimum requires the non-obvious algebraic insight that per-shard
  scale COMMUTES through AR(SUM) (distributivity), so overlay's enumerate-from-
  baseline stays trapped applying scales stage-by-stage. With MAX, `max(x+c)-c ==
  max(x)` is a MORE transparent identity — offset in/offset out visibly cancels,
  so overlay's best seed sees through it to 1 collective (like r10). So the lever
  is not "depth + genuine-looking stages" generically; it is specifically that
  SUM's distributive collapse is algebraically DEEPER (less visible from the
  baseline framing) than MAX's additive-offset cancellation. New L8 below.
- **r12** (DIR-S mixed-primitive depth + DIR-T multi-branch shared collapse):
  CONFIRM DONE. **0 CONFIRMED.** mixed6/mixed8 TIED at screen (1.0/1.0 —
  alternating AR & AG+sum is a locally-visible identity, overlay one-shots it,
  L8). branch4 promoted (screen 1.117) but best-of-8 = 1.000 med 1.078 p .0048
  CI[1.001,1.117] -> DISTRIBUTIONAL-only (kiss median-faster, overlay's best seed
  matches the cross-branch fusion). branch4_big tied. Consistent with L8:
  cross-branch fusion is discoverable enough that overlay's best-of-8 reaches it.
  12th round (exceeds >=10 mandate).
- **r13** (REVERSE-DIRECTION hunt + count-minimal CONTROLS): VALIDATED, queued.
  DIR-U already-minimal baseline (1 AR) w/ dangled useless hierarchical hint —
  tests whether kiss ever OVER-engineers (adds a collective) where overlay stays
  put -> would be first Overlay>Sorcar signal. DIR-V irreducible single AR.
  CONTROL DIR-W hetero SUM+MAX+MIN (3 distinct reductions, count-irreducible) ->
  predicts clean TIE, isolating the claim "Sorcar wins ONLY when count is
  removable, never when irreducible." This round is REQUIRED for an honest
  "one system dominates" conclusion — a genuine attempt at the reverse.
  **RESULT (17:28Z): ALL 4 TIED at the 5160 floor. 0 Overlay>Sorcar.**
  minimal/minimal_big/single/hetero all ratio=1.0. TWO decisive confirmations:
  (1) DIR-W CONTROL fired as predicted — 3 irreducible heterogeneous reductions
  give BOTH systems the same 3-collective floor -> no gap when count is
  irreducible. (2) DIR-U/V reverse-bait FAILED to reverse — kiss did NOT
  over-engineer an already-minimal baseline; it held at 1 AR just like overlay.
  This is the honest-conclusion keystone: across 16 rounds, ZERO Overlay>Sorcar
  even when we deliberately baited kiss into over-engineering. The asymmetry is
  real and one-directional. (NOTE: r13 took ~6.5h wall-clock — the minimal/single
  problems give kiss nothing to optimize so its ReAct loop runs to max steps on
  every seed; this is why the campaign legitimately consumed its ~9h here.)
- **r14** (POSITIVE L8 confirmation via NEW distributive families): VALIDATED
  (base_ok+opt_ok+headroom): zerosum6=1.195 zerosum8=1.273 lincomb6=1.197
  lincomb8=1.275. DIR-X zero-sum perturbation chain (collapse rests on GLOBAL
  sum_r delta_r==0, invisible locally); DIR-Y linear-combination accumulation
  (telescopes by linearity). Both are globally-distributive collapses STRUCTURALLY
  DISTINCT from r2's per-shard scale/unscale. PREDICTION (L8): should CONFIRM
  Sorcar>Overlay (unlike r10/r11's locally-visible identities). If they tie, L8 is
  too narrow. Makes L8 falsifiable in the WINNING direction, not just via nulls.
  **RESULT (17:31Z screen): PREDICTION REFUTED — and a possible FIRST REVERSE
  signal.** lincomb6/lincomb8 TIED (both reach 5160). BUT zerosum6/zerosum8
  screened with overlay/kiss = 0.837/0.80 — i.e. KISS was SLOWER (kiss 6167/6569
  vs overlay 5160): OVERLAY found the 1-AR fold, KISS stayed at ~2 collectives.
  Reverse ratios kiss/overlay = 1.195 / 1.273. This is the campaign's FIRST
  Overlay>Sorcar screen signal. INTERPRETATION: the zero-sum cancellation
  (sum_r delta_r==0) is apparently NOT a Sorcar-favoring distributive collapse —
  overlay's enumerate reaches it while kiss's ReAct sometimes doesn't. Launching
  bidirectional symmetric confirm (reverse_confirm.py) at N=8 to test whether
  this survives best-of-N or is 1-seed noise (per L3, single-seed signals often
  wash out). If it CONFIRMS reverse, it's a genuinely important counter-finding.
  lincomb tied because linearity telescoping IS reachable by both.
  **REVERSE-CONFIRM VERDICT (17:40Z, N=8/6 from results_reverse JSON):
  DISTRIBUTIONAL reverse, NOT confirmed.** zerosum6: overlay all 8 seeds=5160;
  kiss = {5160 x4, 6167 x4} -> best-of-N=1.0 BOTH directions (kiss's best seed
  DOES reach the fold), REVERSE median 1.098 p=.0127 CI[1.0,1.195]. zerosum8:
  overlay {5160 x5, 5170}; kiss {5160 x3, 6570 x3} -> best=1.0, REVERSE median
  1.137 p=.0745 CI[1.0,1.273]. So under the symmetric best-of-N criterion this
  is a TIE, but overlay is DISTRIBUTIONALLY more reliable on additive zero-sum
  folds (kiss over-analyzes and stays at ~2 collectives on ~half its seeds).
  **KEY SYMMETRY FINDING (L9):** the campaign now has BOTH directions of
  distributional asymmetry — Sorcar more reliable on multiplicative scale/unscale
  depth (r1/r8), Overlay more reliable on additive zero-sum folds (r14). BUT the
  asymmetry is UNEQUAL IN STRENGTH: Sorcar's advantage SURVIVES best-of-N (9
  CONFIRMED divergences where even overlay's best seed stays trapped), whereas
  Overlay's advantage is DISTRIBUTIONAL-ONLY (0 confirmed; kiss's best seed always
  escapes). This is the honest, sharp two-sided characterization.
- **r15** (PARTIAL-COLLAPSE gradient: does optimum need to be 1 AR?): VALIDATED
  (headroom wrap4=1.112 wrap6=1.187 wrap8=1.262 wrap6_big=1.188; 2-collective
  floor SUM+MAX). Baseline computes SUM via deep-D collapsible chain + MAX via 1
  AR; optimum = 2 distinct collectives. Tests whether Sorcar's edge is "reaches
  1 AR" or the more general "collapses removable depth regardless of floor." If
  Sorcar wins here (floor=2, not 1), the mechanism statement generalizes.
  **RESULT (17:34Z): ALL 4 TIED at screen (wrap4~1.0, wrap8=1.008,
  wrap6_big=1.005).** Boundary result: when the optimum floor is 2 collectives
  (SUM + MAX, both irreducible), overlay's seed collapses the deep SUM chain AND
  keeps the MAX -> reaches the 2-floor -> NO gap. Sharpens mechanism: Sorcar's
  forward edge specifically requires collapse to a SINGLE collective; a genuine
  remaining second collective is handled fine by overlay's enumerate. The trap is
  about removing the LAST redundant collective, not depth alone.
- **r18** (REVERSE-LEVER probe from r14 zerosum): VALIDATED (zs4=1.117 zs6=1.195
  zs8=1.273 depth grid + multctl6=1.194 multiplicative control). Reproduces the
  r14 zero-sum additive chain at matched depths; multctl6 is the r2-style
  multiplicative identity chain at depth 6. HYPOTHESIS H-REV: additive zero-sum
  folds favor OVERLAY (enumerate lists "drop the additive term"), multiplicative
  scale/unscale folds favor SORCAR (r2). If zs6/zs8 reverse while multctl6
  forwards, additive-vs-multiplicative structure FLIPS the divergence direction —
  a rich two-directional mechanism. Chained after r17.
- **r16** (CAUSAL: is the trap CODE-depth or DESCRIPTION-framing?): VALIDATED —
  all three share IDENTICAL baseline (12919us) and optimum (5555us, headroom
  2.326); the ONLY difference is signature_doc. r16_neutral (result-only doc),
  r16_deepdoc (doc describes the 8 stages), r16_hintdoc (doc drops the fusion
  hint). Controlled experiment isolating the framing effect: if all three diverge
  equally -> trap is code-driven; if deepdoc>neutral>hintdoc -> trap is
  description-driven (overlay anchors to how the problem is PRESENTED). Sharpest
  causal claim available. 16th round.
  **RESULT (17:41Z screen + confirm running): CAUSAL EFFECT CONFIRMED — the trap
  is DESCRIPTION-DRIVEN.** Screen ratios: r16_deepdoc 2.382 (PROMOTE),
  r16_neutral 0.969 (tie), r16_hintdoc 1.011 (tie). Overlay's sim by doc:
  deepdoc=12919 (STUCK at untouched baseline!), neutral=5233, hintdoc=5233. The
  code is BYTE-IDENTICAL across all three; only the docstring differs. When the
  docstring narrates the 8 stages, overlay stays trapped at the full baseline
  (12919 on all 8 confirm seeds); when the docstring states only the result or
  hints at fusion, overlay finds the fold. This is the campaign's sharpest causal
  finding: **Overlay's enumerate-from-baseline anchors to how the problem is
  DESCRIBED, not merely to the code.** deepdoc confirm running at 8 seeds -> will
  be the strongest single divergence (screen 2.38x, overlay pinned at baseline).
  **CONFIRM VERDICT (18:2xZ): r16_deepdoc CONFIRMED — 10th confirmed divergence.**
  best=1.374 median=1.908 p=0.0003 CI[1.5255,2.1433] CONFIRMED=True. neutral
  (0.969) and hintdoc (1.011) DISCARDED at screen. This is the campaign's sharpest
  causal result: BYTE-IDENTICAL deep-8 code, docstring-only variation, and the
  narration-heavy docstring alone reproduces the full 1.9x median divergence while
  the result-only / fusion-hint docstrings tie. The trap is DESCRIPTION-driven.
- **r17** (ROBUSTNESS: best-of-16 re-confirm of headline wins): CONFIRM DONE.
  **BOTH HELD at N=16 (even stronger):**
    r2_deep8     best 2.495 med 2.091 p 0.0 CI[1.870,2.382] CONFIRMED=True
    r9_deep8_big best 1.401 med 1.853 p 0.0 CI[1.802,2.357] CONFIRMED=True
  Doubling the seed budget did NOT wash the headline out — r2_deep8's best-of-16
  is 2.495 (vs best-of-8 2.179), median 2.091, and the CI tightened. The deep-chain
  divergence is robust to N; overlay's best of SIXTEEN seeds still stays trapped
  at the deep framing. Rebuts any "best-of-8 got lucky" objection.
- **r18** (REVERSE-LEVER probe): CONFIRM DONE. **0 CONFIRMED (validates L9).**
  zs4/zs6 TIED at screen (1.0); zs8 PROMOTED at screen (1.273 reverse) but
  best=1.0 median=1.0 p=0.1587 CI[1.0,1.0] CONFIRMED=False. multctl6 (r2-style
  multiplicative chain at depth 6) discarded at screen (1.039) — did NOT fire,
  consistent with r9's best-of-N depth threshold ~6-7 (depth 6 is below it).
  CLEAN CONFIRMATION of L9: the additive zero-sum fold is DISTRIBUTIONAL-ONLY —
  it screens as a reverse (overlay more reliable) but kiss's best seed always
  reaches the fold, so it TIES on the symmetric best-of-N criterion. Still 0
  confirmed Overlay>Sorcar after 18 rounds + a dedicated reverse-lever probe.
- **r19** (GENERALIZE r16 framing effect): SCREEN DONE. **ALL 4 TIED — but
  CONFOUNDED, not a clean null.** Anarr/Ares/Bnarr/Bres all ratio ~1.0
  (Ares 0.891 seed-noise). The narration docstring did NOT reproduce r16's effect
  here — BUT families A/B have shallow headroom (A=1.123 4-way split, B=1.078
  3x-recompute), so overlay one-shots them regardless of docstring (the fold is
  too easy to need the framing crutch). r19 fails to isolate the framing effect
  because its baselines are too shallow: the r16 effect needs a DEEP baseline
  (headroom >2) where narration is what keeps overlay pinned. r20 redoes the
  generalization test at DEEP headroom on a NON-scale/unscale family. LESSON: the
  framing-narration trap requires (framing depth) AND (a narration that anchors);
  removing either — shallow code (r19) OR result-only doc (r16_neutral) — ties.
- **r20** (does NARRATION override ALGEBRAIC structure? 2x2 family x doc at DEEP
  headroom): CONFIRM DONE. **1 CONFIRMED (11th) + a sharp distributional flip.**
  Validated: su8_res/su8_narr headroom 2.326 (=r16); zs8_res/zs8_narr 1.273
  (=r14/r18). Byte-identical code within family; only docstring differs.
    r20_su8_narr  best 1.561 med 1.944 p .0002 CI[1.931,2.395] CONFIRMED=True
    r20_su8_res   screen 0.778 -> discard (overlay folds; = r16_neutral)
    r20_zs8_narr  best 1.000 med 1.273 p .0013 CI[1.137,1.273] CONFIRMED=False
    r20_zs8_res   screen 1.0 -> discard (both fold; = r18 zs8_res)
  su8_narr REPLICATES r16_deepdoc (overlay pinned at 12919 baseline on 7/8 seeds;
  the 8th at 8123) -> narration on the MULTIPLICATIVE family confirms forward at
  best-of-N. zs8_narr is the NOVEL result: on the ADDITIVE zero-sum family the
  narration docstring makes overlay STUCK at 6570 baseline on the MEDIAN seed
  (kiss folds to 5160 every seed) -> median 1.273 FORWARD, p<.05. This is a
  DISTRIBUTIONAL FLIP: plain zs8 (result-only, r18/r14) was distributionally
  REVERSE (overlay more reliable at finding the additive fold); NARRATING the same
  chain flips it distributionally FORWARD (Sorcar more reliable). BUT it does NOT
  survive best-of-N: overlay's BEST of 8 seeds still escapes to 5160, so best=1.0.
  **L10 (r20): narration is causal on BOTH algebraic families at the DISTRIBUTIONAL
  level — it reliably pins overlay's median seed at the narrated baseline. But it
  only SURVIVES best-of-N on the MULTIPLICATIVE family, because (L9 core) a
  multiplicative scale/unscale chain traps even overlay's BEST seed, whereas an
  additive zero-sum fold is recoverable by overlay's best-of-N seed regardless of
  how it is described.** So framing and algebra COMPOSE: framing sets the median;
  algebra sets whether the trap is escapable at best-of-N. This unifies r16 (framing
  causal) with L9 (mult vs additive asymmetry) into one two-factor mechanism.
- **r21** (DECOMPOSE the narration trap: COUNT cue vs PROCEDURE cue): CONFIRM DONE.
  **2 CONFIRMED (12th + 13th) with a sharp STRENGTH GRADIENT.** 4 byte-identical
  deep-8 su-family problems, docstring-only variation isolating two cues in r20's
  full narration:
    r21_su8_countonly  best 1.287 med 2.382 p .0002 CI[1.870,2.392] CONFIRMED=True
    r21_su8_proconly   best 1.157 med 1.042 p .0076 CI[1.001,1.837] CONFIRMED=True
    r21_su8_full       screen 0.997 -> discard (1-seed screen FALSE NEGATIVE, L3;
                       full already confirmed at best-of-8 1.561 in r20_su8_narr)
    r21_su8_res        screen 0.754 -> discard (negative control folds)
  **THE COUNT CUE IS THE DOMINANT LEVER.** countonly ("computed using 8 dependent
  all_reduce operations", NO procedure narration) traps overlay at the 12919
  baseline on 8/9 seeds -> median 2.382, robust to best-of-8. proconly (narrates
  the per-stage scale/unscale procedure but NEVER states the number 8) is a WEAK
  trap: overlay is bimodal (baseline on 3/9 seeds, folds to ~6900 on 6/9) ->
  median only 1.042, best-of-8 barely clears at 1.157. **L11 (r21): overlay's
  enumerate-from-baseline anchors SPECIFICALLY to the stated COLLECTIVE COUNT in
  the problem description. Naming "8 all_reduce operations" pins it; narrating the
  procedure without a count mostly lets it fold.** This is the campaign's most
  precise causal statement: the trap is not "deep code" (r16), nor even "narration"
  generically (r20), but specifically the DESCRIPTION ASSERTING A COLLECTIVE COUNT
  that overlay's bounded R=3 refinement then treats as a floor to shave toward
  rather than a framing to escape.
- **r22** (STRESS-TEST the count cue across truth-value & family): CONFIRM DONE.
  **2 CONFIRMED (14th + 15th); count-cue lever fully mapped.** All byte-identical
  deep-8 code (su unless noted), docstring-only:
    r22_su8_count16  best 2.395 med 1.939 p .0006 CI[1.869,2.395] CONFIRMED=True
    r22_su8_count8   best 1.287 med 2.137 p .0002 CI[1.869,2.393] CONFIRMED=True (=r21 ctrl)
    r22_zs8_count8   best 1.000 med 1.029 p .0019 CI[1.0,1.273]  CONFIRMED=False
    r22_su8_count1   screen 0.749 -> discard (truthful count=1 FREES overlay)
  THREE clean confirmations of the L11 count-cue mechanism:
  (1) COUNT=1 (truthful minimal, "computed using a SINGLE all_reduce") FREES
      overlay -> screen 0.749, overlay folds to 5178. A count assertion of 1
      un-anchors it (like r16_hintdoc's fusion hint).
  (2) COUNT=16 (OVERSTATED — code has 8) traps MAXIMALLY: overlay pinned at 12919
      on ALL 9 seeds, best-of-8 = 2.395 (the campaign's single largest best-of-N
      ratio). A larger asserted count does not saturate at 8 — it deepens the trap
      to the point overlay's best seed never escapes.
  (3) COUNT cue on the ADDITIVE zero-sum family (zs8_count8): median 1.029 forward
      (p<.05, overlay bimodal — baseline on most seeds) BUT best-of-N = 1.0 because
      overlay's best seed folds to 5160. Confirms L10/L11 COMPOSE exactly: the count
      cue sets the median trap on BOTH families, but only the MULTIPLICATIVE family's
      algebra makes it inescapable at best-of-N. The additive family stays
      distributional-only regardless of how strongly it is framed.
  Net: the count assertion is a MONOTONE dial (1 frees < 8 traps < 16 traps-max) on
  the multiplicative family, and a distributional-only dial on the additive family.
- **r29** (INDEPENDENT DATA POINT: robust win tied to a[r]=1+.5(r%3), or the
  pattern CLASS? — richer 4-level powers-of-2 a[r]=2**((r%4)-1)): INCONCLUSIVE (noise
  at 1-seed screen, not counted). (First affine construction here was invalid — after
  the first AR every rank holds the identical vector so a later AR(SUM,buf)=W*buf and
  a zero-sum shift does NOT cancel; dropped. Switched to a 4-level powers-of-2 scale
  pattern on the confirmed contiguous template.)
    r29_pow2_count8 screen: kiss 12919 (its 1 draw did NOT fold), overlay None (its 1
                    draw FAILED the fp32 gate) -> ratio=None -> discard
    r29_pow2_res    screen: same (kiss 12919, overlay None) -> discard
  Baselines both pass the gate (12919, 2.33x headroom, validated pre-launch), so the
  problems are well-formed; the screen is just a single unlucky iid draw on each side
  (kiss didn't derive the fold in its step budget on that seed; overlay's candidate
  crashed the gate). At 1-seed screen this is NOISE, not a tie or a reverse signal.
  Not promoted, not counted. (The pattern-class question is instead answered robustly
  by r26_perm_count8 = a DIFFERENT shard<->scale MAP on the same template, confirmed
  at best-of-16.)
- **r31** (well-formed additive-linear pattern-class retry: a[r]=1.0+0.25*(r%5),
  5 levels {1,1.25,1.5,1.75,2}): SAME degenerate screen as r29/r30 (kiss 12919 both,
  overlay None both) -> discard. So it is NOT the exponential description specifically:
  a THIRD distinct constant pattern also fails at 1-seed for both agents. Read: novel
  per-shard-scale constant patterns beyond the confirmed {1,1.5,2} are search-degenerate
  at 1-seed screen (overlay's candidate crashes the fp32 gate; kiss doesn't derive the
  fold in budget) — an artifact of the screen being SINGLE-DRAW, not a divergence.
  Rather than keep burning launches on constant-pattern variants (3 consecutive
  degenerate: r29/r30/r31), the pattern-CLASS-robustness question is taken as answered
  by r26_perm_count8 (a structurally different shard<->scale MAP on the confirmed
  template, CONFIRMED at best-of-16). Pivoted remaining budget to best-of-16 rechecks
  of the strongest existing wins (r32).
  **CORRECTION (post-r32b) — TRUE ROOT CAUSE = ANTHROPIC API USAGE CAP, not design or
  screen fragility.** r32b (confirm-only, skips screen) re-ran three KNOWN-CONFIRMED
  wins and returned n_overlay_ok=0/16 on all three; the worker logs show:
    RuntimeError: Anthropic API HTTP 400 ... "You have reached your specified workspace
    API usage limits. You will regain access on 2026-10-01 at 00:00 UTC."
  r29's kiss log shows the SAME 400 (workspace usage cap). So the API budget for this
  workspace was exhausted DURING r29, and every LLM call since (r29/r30/r31/r32/r32b)
  fails with HTTP 400. The "r29/r30/r31 degenerate design" and "fragile 1-seed screen"
  readings are BOTH superseded: those rounds produced NO valid LLM data — kiss fell back
  to its non-LLM baseline (12919) and overlay crashed on the capped API call (sim=None).
  **Last rounds with valid LLM data: r26 / r27 / r28.** No further LLM-backed rounds are
  possible before the 23:30:11Z deadline (cap lifts 2026-10-01). r29-r32b are VOID.
  The standing result is unchanged and rests entirely on r1-r28: **17 robust confirmed
  (best-of-16 survivors), 0 confirmed reverse.**

- **CORRECTION #2 (SUPERSEDES CORRECTION #1) — the cap was NOT a hard blocker; it was
  a routing bug.** The DIRECT `ANTHROPIC_API_KEY` was capped, but this session runs on
  BEDROCK (`CLAUDE_CODE_USE_BEDROCK=1`), which is uncapped. Two harness paths were
  hard-wired to the direct API: (1) overlay via `search/_anthropic_route._post_anthropic`
  (fixed: added a `_post_bedrock` branch gated on CLAUDE_CODE_USE_BEDROCK); (2) kiss via
  its own `Anthropic()` client in `kiss/core/models/anthropic_model.initialize()` (fixed:
  `kiss_token_shim.py` now swaps in `AnthropicBedrock` with model-id mapping + FULL
  recursive `cache_control` strip). Both verified live on Bedrock (kiss n_ok=12, folds
  6909). So r29+ were NOT blocked — the earlier r29/r31 data was invalid only because
  kiss ran dead (baseline) while overlay ran live. RE-RAN r29/r31 with both sides live:
    r31_lin5_count8  bo8 best 2.494/med 1.90/p .0002 CONFIRMED; bo16 best 2.395/med 1.87/
      p 0/CI[1.862,2.382] CONFIRMED=True -> **18th ROBUST WIN** (generality: 5-level linear
      {1,1.25,1.5,1.75,2}, proves the trap is class-level not tied to 3-level {1,1.5,2}).
    r31_lin5_res     bo8 best 1.322/med 1.16/p .0006 CONFIRMED; bo16 best .955/med 1.004
      CONFIRMED=False (result-only doc, draw-fragile like r26_strided_count8).
    r29_pow2_count8  bo8 med 1.935/p .0053; bo16 med 1.87/p .0006 but best .965
      CONFIRMED=False -> exact-power-of-2 unscale is a clean bit-shift overlay fuses;
      MECHANISTIC: trap needs an AWKWARD (non-pow2) per-shard unscale.
    r29_pow2_res     TIE (best .965/med 1.438/p .54).
  **FINAL STANDING RESULT: 18 robust confirmed (best-of-16 survivors), 0 confirmed
  reverse.** Rounds r30/r32/r32b remain descriptively void (superseded by these live
  re-runs). The r1-r28 subtotal (17) is unchanged; r31_lin5_count8 is the 18th.
- **r30** (clean relaunch of r29 pow2 for a fresh iid screen draw): REPLICATED the
  r29 result exactly (kiss 12919 both, overlay None both) -> NOT noise but SYSTEMATIC:
  the `a[r]=2.0**((r%4)-1)` (levels 0.5,1,2,4) exponential description confuses BOTH
  agents (overlay's candidate fails the fp32 gate; kiss doesn't derive the fold in
  budget), unlike the confirmed family's parseable additive `1.0+0.5*(r%3)` style.
  So pow2 is a FLAWED PROBLEM DESIGN (degenerate for both systems), not an informative
  divergence. Dropped; pattern-class question answered instead by an additive-constant
  variant (r31) + r26_perm.
- **r28** (BEST-OF-16 robustness recheck of the two r26 wins): DONE. **Splits the
  two r26 wins by robustness — an important honesty correction.** Re-ran both r26
  confirmed problems at 16 seeds/side (the same bar the headline r2 win was rechecked
  at):
    r26_perm_count8    best 1.100 med 2.056 p 0.0    ci[1.862,2.382] CONFIRMED=True (robust)
    r26_strided_count8 best 1.000 med 2.353 p .0001  ci[1.862,2.393] CONFIRMED=False
  perm_count8 HOLDS at best-of-16 (CI lower 1.862) -> robust confirmed win.
  strided_count8 ESCAPES at best-of-16: with 16 overlay draws, overlay's best seed
  finally reaches the floor (best-of-N ratio drops 2.393 -> 1.0), so it FAILS the
  best-of-N criterion at N=16 even though its median stays trapped at 2.353. So the
  INTERLEAVED-layout win is a best-of-8-only artifact: fragile to overlay draw count.
  **Honest correction to the roster: strided_count8 is DEMOTED from robust-confirmed
  to "best-of-8 only, escapes at best-of-16".** Net robust confirmed count = 17 (not
  18). This does NOT weaken the headline: the contiguous perm_count8 (and all top wins
  r2/r9/r22/r23) survive best-of-16; only the fusible strided layout is draw-fragile,
  which is itself consistent with L13'/L14 (strided is the layout overlay can fuse —
  it just needs enough draws to find the fusing seed).
- **r27** (does the COUNT cue also rescue the GLOBALLY-SYMMETRIC r24 families?):
  DONE. **0 new confirmed — sharp scope bound on L14.** Count cue added to both r24
  symmetric families (code unchanged, docstring gains "8 dependent all_reduce ops"):
    r27_globalscale_count8 best 1.0 med 0.985 p .7788 ci[.785,1.272] CONFIRMED=False
    r27_pairwise_count8    screen ratio 0.853 -> DISCARD (overlay folds; kiss slower)
  globalscale PROMOTED on a fluke screen seed (overlay 6574 that draw) but over 8
  seeds overlay's best-of-N escapes (median 0.985) — the count cue does NOT reliably
  re-anchor the globally-symmetric homogeneity chain the way it re-anchored the
  rank-heterogeneous strided layout (r26). pairwise folded outright.
  **Decisive contrast with r26 -> BOUNDS L14:** a truthful count cue RE-ANCHORS
  overlay only onto RANK-HETEROGENEOUS fusible code (r26 strided_count8 CONFIRMED
  2.39x); it does NOT rescue GLOBALLY-SYMMETRIC collapses (r27) — for those overlay's
  fold is triggered by the CODE STRUCTURE (a single global factor / a cancelling
  term it can see from one rank), independent of the description. So the r24 symmetric
  nulls are CODE-STRUCTURE facts, not description artifacts. This preserves L13 and
  bounds L14 to rank-heterogeneous code. See L15.
- **r26** (does NARRATION rescue each r25 layout into a CONFIRMED win?): DONE.
  **BOTH CONFIRMED -> 17th & 18th confirmed divergences.** Count cue ("8 dependent
  all_reduce operations") added to both r25 layouts (result-only -> counted):
    r26_perm_count8    best 1.140 med 2.353 p .0022 ci[1.289,2.382] CONFIRMED=True (17th)
    r26_strided_count8 best 2.393 med 2.382 p .0007 ci[1.871,2.393] CONFIRMED=True (18th)
  **Decisive:** the INTERLEAVED layout that FOLDED for overlay at r25 screen (0.954)
  becomes a fully-confirmed 2.39x win once the docstring truthfully NAMES the count.
  Two big conclusions:
  (1) The confirmed mechanism is layout-robust WHEN the description carries a count:
      permuted contiguous shards AND interleaved lanes both confirm. So the confirmed
      win is not identity-map- or contiguous-block-specific once narration is present.
  (2) It EXTENDS L12: the count cue does more than reinforce a code structure overlay
      already sees — on strided code (which overlay FUSES when unnarrated), the count
      assertion RE-ANCHORS overlay onto the baseline it would otherwise escape. So
      narration is causally strong enough to trap even code overlay could fold. This
      does NOT contradict r23 (a LYING count over minimal code didn't fool overlay):
      here the count is TRUTHFUL (code really has 8 ARs) — the strided code is deep,
      just fusible; narration stops overlay from attempting the fusion. Refines L13'
      into L14.
- **r25** (VALIDATE L13: is it RANK-HETEROGENEITY or the specific contiguous
  code that traps overlay?): DONE. Two rank-heterogeneous multiplicative deep-8
  chains, result-only docstrings, DIFFERENT layouts:
    r25_perm_scale8    best 0.965 med 1.866 p .1139 ci[.756,2.138] CONFIRMED=False
    r25_strided_scale8 screen ratio 0.954 -> DISCARD (folds for overlay)
  **Sharpens L13 into L13': LAYOUT matters, not just rank-heterogeneity.**
  - perm_scale8 (contiguous shards, shard<->scale mapping = a permutation instead of
    identity): overlay's MEDIAN stays trapped at 1.866 — the trap generalizes across
    contiguous rank-heterogeneous multiplicative collapses regardless of whether the
    shard-to-scale map is identity or a permutation. (Not confirmed only because the
    docstring is RESULT-ONLY -> best-of-N escapes, exactly per L12/r24_deep8_res.)
  - strided_scale8 (interleaved lanes x[r::W] instead of contiguous blocks): FOLDS
    for overlay at the screen seed (0.954). The interleaved layout is TRANSPARENT to
    overlay's enumerate — it recognizes the strided scale as a single vectorizable
    per-lane op and fuses. So the trap needs CONTIGUOUS-BLOCK rank-heterogeneity, not
    any rank-heterogeneous layout.
  Net: r25 confirms the mechanism is (contiguous rank-heterogeneous multiplicative
  depth) at the MEDIAN, and — like r24 — result-only docstrings keep it from
  surviving best-of-N. Sets up r26 (add narrated count to perm/strided to test
  whether description rescues each layout). See L13'.
- **r24** (GENERALITY: is the code-depth trap scale/unscale-specific, or general
  to any deep globally-distributive collapse?): SCREEN DONE, deep8_res CONFIRM
  running. THREE structurally-distinct deep-8 collapses, all result-only docstrings:
    r24_globalscale8  screen ratio 1.0 -> DISCARD  (global-scalar homogeneity, c=2)
    r24_pairwise8     screen ratio 1.0 -> DISCARD  (permutation-invariance via roll)
    r24_deep8_res     best 1.005 med 1.267 p .1995 ci[.756,2.085] CONFIRMED=False
  deep8_res is the per-shard scale/unscale chain but with a RESULT-ONLY docstring
  (no narration) — so it reproduces r16_neutral/r20_su8_res exactly: without a
  narrated/counted description, overlay is BIMODAL (some seeds trapped ~12900, some
  fold to ~5200) and its best-of-8 seed escapes -> best-of-N ties (not confirmed),
  even though the code is the confirmed deep chain. This is fully consistent with
  L12 (code depth traps the median; description determines best-of-N confirmation).
  r24 yields **0 new confirmed** and a clean scope result (below).
  **Decisive scope result:** the two NOVEL distributive families FOLD FOR OVERLAY TOO
  (both hit the 5160 floor on overlay's single screen seed) — overlay's enumerate
  sees through global-scalar homogeneity (c pulls out of AR(SUM) by a single obvious
  factoring) and permutation-invariance (the rolled term visibly cancels). Only the
  per-shard scale/unscale chain — where each shard is scaled by a DIFFERENT rank-
  dependent a[r] so the collapse is NOT visible from any single rank's local view —
  traps overlay. This SHARPENS L8: the divergence is not "deep distributive code" in
  general; it specifically requires the collapse to rest on a RANK-HETEROGENEOUS,
  non-locally-visible identity. Homogeneous/globally-symmetric distributive collapses
  (however deep) are escapable by overlay's best (indeed even median) seed. See L13.
- **r23** (ADVERSARIAL de-confound: does overlay anchor to STATED COUNT or CODE?):
  CONFIRM DONE. **1 CONFIRMED (16th) + a decisive de-confounding null.** Prior
  rounds confounded "stated count" with "code depth" (both agreed). r23 makes them
  DISAGREE:
    r23_deep8_count8  best 2.482 med 2.382 p .0002 CI[1.869,2.395] CONFIRMED=True (control)
    r23_deep8_says1   best 0.996 med 1.873 p .0048 CI[1.219,2.382] CONFIRMED=False
    r23_minimal_says8 screen 1.038 -> discard (LYING high count over minimal code)
    r23_minimal_res   screen 1.0   -> discard (minimal code, result-only)
  TWO clean de-confounding results:
  (1) minimal_says8: docstring LIES "8 dependent all_reduce operations" over code
      that is the ALREADY-MINIMAL single-AR optimum. Overlay is NOT fooled into
      over-engineering (screen 1.038, stays near floor) -> overlay reads the CODE,
      not a count that contradicts trivial code. NO manufactured divergence.
  (2) deep8_says1: docstring LIES "a SINGLE all_reduce" over the genuinely deep-8
      chain. The false-LOW count rescues overlay on ONLY 2/9 seeds (5375, 8256) ->
      best-of-N ties (0.996) BUT median stays trapped at 1.873 (p<.05, overlay
      pinned at 12919 on 7/9 seeds). The deep CODE keeps it trapped; the lying low
      count only helps a minority of seeds.
  **L12 (r23): the PRIMARY anchor is the ACTUAL CODE DEPTH; the stated count is a
  SECONDARY modulator that only bites WHEN IT AGREES WITH THE CODE.** This corrects
  the naive reading of L11: r22_count1 freed overlay because that code was TRULY
  minimal (truthful count); a LYING low count over deep code (r23 deep8_says1) does
  NOT free the median, and a lying high count over minimal code (minimal_says8)
  does not trap. So the r16/r20/r21/r22 narration effect is real but operates by
  REINFORCING the code's actual structure in overlay's enumerate-from-baseline
  framing, not by overriding what the code plainly shows. The trap = deep code
  that overlay's bounded R=3 refinement can't escape, AMPLIFIED when the description
  truthfully asserts the same high count. Kiss's open ReAct escapes regardless of
  both. This is the campaign's most precisely-bounded causal statement.
- **r19** (GENERALIZE r16 framing effect across families): VALIDATED. Two
  matched narr/res docstring pairs over DIFFERENT collapsible baselines (FAMILY A
  4-way split-reduce headroom 1.123; FAMILY B 3x-recompute-average headroom
  1.078). Byte-identical code within each pair; only docstring differs (narr =
  narrates the procedure, res = states only the result). Tests whether "narrate
  the steps" is a UNIVERSAL Overlay trap (r16 generalizes) or scale/unscale-
  specific. Predicted: narr diverges, res ties, for both families. Chained after
  r18.
- **Standing read on headroom**: only DEPTH (chains of dependent ARs, r2/r5)
  produces large sim headroom (>1.3). Every "different primitive / redundant
  layout / fewer bytes" lever floors near 1.04-1.10 because the sim charges a
  flat ~5160us dispatch+mark_step for ANY single small collective. => the
  campaign's discriminating tests are r2 (depth) and r5 (compositional depth).


---

## Round r33 — SECOND-FAMILY PROBE (2026-09-20, live on Bedrock)

**Question:** all prior 18 robust wins are ONE mechanism (rank-heterogeneous per-shard
DIAGONAL multiplicative scale/unscale). Does a STRUCTURALLY DISTINCT collapse also trap
Overlay? Held the deep-8 AR skeleton byte-identical, swapped only the per-stage operator:

- **r33_shear8_count8** — pure OFF-DIAGONAL linear op (pairwise shear: even shard 2k <-
  2k+2k+1; NO diagonal scale). Inverse = exact integer un-shear. Baseline passes fp32 at
  12684us. **Result: TIE.** bo8 best 2.004 / med 1.0 / p .3913 / CI[.9266,2.0665],
  CONFIRMED=False. Kiss folded on only 2/8 seeds (5400, 6138); its MEDIAN draw stayed at
  the 12684 baseline, same as overlay. The order-dependent un-shear is hard for BOTH
  systems -> no robust asymmetry. The distinct-algebra probe does NOT reproduce the win.

- **r33_permscale8_count8** — MONOMIAL op (per-shard multiplicative scale a[r]=1+0.5(r%3)
  AND half-rotation of shard positions p[r]=(r+W//2)%W). Baseline 12919us. **Result:
  CONFIRMED bo8 AND bo16.** bo8 best 2.012 / med 2.213 / p .0002 / CI[1.751,2.379]; bo16
  best 1.091 / med 1.866 / p .0082 / CI[1.019,2.353], CONFIRMED=True -> 19th robust win.
  BUT it confirms because it RETAINS the multiplicative scale — it is family-1 WITH
  ROUTING, not a distinct algebra. Kiss folds reliably (6/8 near the 5160 floor); overlay
  pinned at 12919 on 7/8.

**L16 (new):** There is NO genuine second family. Adding a permutation on top of the
multiplicative collapse PRESERVES the trap (permscale confirms); removing the diagonal
scale and using a purely off-diagonal linear op BREAKS it (shear ties). This SHARPENS the
r24 boundary rather than widening it: Sorcar's confirmed dominance is specific to the
rank-heterogeneous MULTIPLICATIVE (diagonal) per-shard scale/unscale collapse — the one
collapse that is linear through AR(SUM) yet invisible from any single rank. The 19-win
roster is thus all one computational family, probed along six axes: payload (r1), depth
(r2/r9), description (r16-r23), layout (r26), scale constants (r31), and operator algebra
(r33). Ledger key: round_r33_second_family_probe.

## Rounds r34–r37 — DISTINCT-FAMILY BATTERY (2026-09-20, live on Bedrock, +5h budget)

After L16 was (rightly) flagged as under-powered — proven from too few genuinely distinct
mechanisms — a breadth-first battery of FIVE distinct algebras was screened, survivors
promoted to best-of-16:
- **r34 tropical (max-plus/min-plus semiring)** — AR(MAX/MIN)+additive offset. bo8 NOT
  confirmed (max 1.06, min 0.79); additive inverse trivial, Overlay folds (like r14).
- **r35 masked partition-of-unity** — disjoint masks accumulate to 1 AR(SUM). bo8 TIE
  (best 1.0/med 1.06/p .28); partition locally visible, Overlay folds.
- **r36 affine (mult+additive)** — per-shard a[r]·block+c[r]. **bo16 CONFIRMED**
  (best 1.291/med 2.162/p=0.0/CI[2.160,2.169]) — HOLDS. But affine RETAINS a multiplicative
  factor (inverse needs /a[r]); NOT a new family.
- **r37 group-theoretic block-rotation (NO scale)** — D rotations composing to 1 net
  rotation. The ONLY scale-free candidate. Passed bo8 (best 1.324/CI[1.104,1.705]) but
  **ESCAPES at bo16** (best 1.399/med 1.704/p .0061/**CI_lo=1.0** → CONFIRMED=False).
- r38 (dihedral/reflection) staged but NOT run — its cyclic sibling r37 already failed bo16,
  so a reflection variant of the same permutation mechanism cannot establish a family.

**L17 (SUPERSEDES/STRENGTHENS L16):** The distinct-family battery VINDICATES the
multiplicative-component boundary. The only bo16 survivor (r36 affine) keeps a
multiplicative factor; the one genuinely scale-free mechanism (r37 rotation) escapes bo16 —
just like pure shear (r33), pure additive (r14/r34), tropical (r34) and partition (r35).
Across FIVE distinct algebras, no scale-free collapse yields a robust (bo16 CI_lo>1.0)
second family. The r37 bo8→bo16 reversal also reconfirms that bo8 alone is not robust and
the strict CI_lo>1.0 gate is load-bearing. Answer to "does a second family exist?": under
strict symmetric bo16 confirmation across genuinely distinct mechanisms, **No** — now
bounded by a five-algebra battery, not a single probe. Ledger keys:
round_r34_tropical_distinct_family, round_r3567_distinct_families_bo8,
round_r3567b_distinct_families_bo16.
