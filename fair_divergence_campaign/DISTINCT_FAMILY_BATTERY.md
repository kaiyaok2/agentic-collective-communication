# FAM-4/FAM-5 70-MIN WINDOW — FINAL VERDICT (2026-09-21)

**Deliverable: two candidate axes designed + screened; both are CLEAN NEGATIVES, and the third
(rescue) attempt is provably degenerate. The negatives are the finding — they close the last open
degree of freedom in the trap boundary.** No cloud tokens spent on the rescue (proven degenerate
by faithful pre-screen, so a best-of-4 would tie by construction). Three attempts, one law:

| Attempt | Axis | Result | Why |
|---------|------|--------|-----|
| r49 fam-4 | K parallel AR on different fns of x, concat | TIE (best-of-4, both Bedrock) | parallel algebraic identity → one-shot foldable by BOTH |
| r50 fam-5 | K parallel AR(w_k·x), rank-HOMOGENEOUS weights, sum | TIE (best-of-4, both Bedrock) | AR-linearity is one-shot foldable by BOTH |
| r51 fam-4-rescue | DEPTH-D dependent chain + zero-sum additive bias beta[r]=(r-(W-1)/2)·G | **DEGENERATE (not run)** | Σ_r beta[r]=0 ⇒ bias vanishes in every AR ⇒ whole chain collapses to `W·AR(SUM,x)`; faithful pre-screen: the one-line `world_size*all_reduce(x)` fold PASSES the gate (sim 5160us floor). No genuine reduction ⇒ no heterogeneous-buffer trap ⇒ would tie by construction. |

**THE LAW (now complete).** A trap exists iff **Overlay's obvious one-shot guess FAILS the fp32
gate**. That happens iff the *intermediate buffers entering a collective are genuinely
rank-heterogeneous in a way the fold must undo*:
- fam-1: per-RANK static scale a[r] — heterogeneous, needs /a. ✓
- fam-2: rank-INDEXED routing count c[b] — heterogeneous selection. ✓
- fam-3: DATA-DEPENDENT continuous scale on a DEPENDENT chain — heterogeneous + non-obvious
  telescoping stability. ✓
- **NEGATIVES** (obvious guess PASSES ⇒ not a family): off-diagonal coupling (r42); discrete top-k
  SELECTION; PARALLEL algebraic identities — linearity/concat (r49/r50); and **zero-sum additive
  injection (r51)**, which is worse than a tie — it's *identically zero net effect*, collapsing to a
  scaled single AR. **Additivity that cancels globally can never trap: cancellation = the fold.**
  A trap needs a per-rank factor that SURVIVES the reduction (multiplicative/selective), not one
  that sums away.

**Conclusion for the campaign.** Three confirmed families (rank-multiplicative, rank-routing,
data-dependent-diagonal) exhaust the mechanisms that force a gate-failing guess at W=224. fam-4 and
fam-5 as fresh *parallel-fusion* axes do not yield families; the only path to a 4th family is a NEW
kind of surviving rank-heterogeneity (not additive, not the three above) — e.g. rank-dependent
PERMUTATION/gather-index that a scale-free fold can't collapse, or mixed-primitive chains where the
gate-failing step is a non-linear (max/min) reduction. Logged as the next-session lead; not
pursued in this window.

---

# FAM-4 & FAM-5 IDEATION + SCREEN (r49/r50, 2026-09-21, strict 70-min window)

Two NEW axes designed, both distinct from families 1-3 and both real training patterns. Confirmed
trap boundary = a fold Overlay's enumerate can't reach in one shot (fam-1 per-rank scale, fam-2
routing count, fam-3 data-dependent scale + cross-primitive RS+AG). fam-4/fam-5 attack a DIFFERENT
fold: COLLECTIVE-COUNT reduction via input-shape algebra of AR(SUM), not scale.

**fam-4 (r49) — STACKED MULTI-STATISTIC AR fusion.** Baseline issues K separate AR(SUM), each on a
different elementwise fn of x (x, x², |x|, relu, x³) and concatenates — the LayerNorm/moment-stats
pattern. AR(SUM,cat[f0,f1,...]) == cat[AR(f0),AR(f1),...] so K collectives fuse to ONE AR of the
concatenated inputs. Fold by INPUT CONCATENATION, multi-output. Pre-screen headroom modest: k2 1.044
(sub-gate), k3 1.081, k4 1.117, k5 1.155 — grows with K but shallow (concat only adds bytes, not
collectives-per-stage). 7/8 viable; distinct ref output per K and per payload.

**fam-5 (r50) — AR-LINEARITY additive fusion (per-ELEMENT rank-HOMOGENEOUS weights).** Baseline =
sum of K terms AR(SUM, w_k·x) with w_k a per-block weight that is the SAME on every rank. By
linearity: Σ_k AR(SUM,w_k·x) == AR(SUM,(Σ_k w_k)·x) → K collectives fuse to ONE. Fold by INPUT
SUMMATION; the rank-HOMOGENEOUS weight is the key distinction from fam-1's per-RANK a[r] (no /a
needed to invert; the naive "K·AR(x)" guess fails because w_k differ). Pre-screen headroom scales
strongly with K: k4 1.121, k6 1.203, k8 1.284, k10 1.364. All 8 viable; distinct ref per K/payload.
=> fam-5 is the STRONGER candidate (deeper headroom). Both gate-safe (all SUM).

**SCREEN RESULT (best-of-4, both pipelines Bedrock): CLEAN NEGATIVE — both families TIE.** Every
problem best-ratio ~1.0 / median ~1.0 (r49_stat_k3/k4/k5 all 1.0; r50_lin_k4/k6/k8/k10 all
1.00-1.006), despite headroom up to 1.36×. **Both pipelines find the fusion.** Mechanistic reason:
fam-4/fam-5 use K PARALLEL, INDEPENDENT collectives, so the fold is a ONE-STEP algebraic identity
(AR-linearity / concat-linearity) that Overlay's enumerate reaches in one shot — Overlay's naive
guess PASSES the gate and IS optimal, so there is no trap. This SHARPENS the boundary further:
> The robust trap needs Overlay's obvious guess to FAIL the fp32 gate. That requires either
> rank-HETEROGENEOUS intermediates (fam-1 per-rank scale, fam-2 routing) OR a DEPENDENT-chain
> whose telescoping stability is non-obvious (fam-3 data-dependent recompute). A PARALLEL
> algebraic identity (linearity, concatenation) is one-shot-foldable by BOTH pipelines → not a family.
Parallel-fusion axes therefore join off-diagonal coupling and discrete top-k selection as
NEGATIVES that map the trap boundary. Next-attempt direction (see r51 below if run): convert the
fusion to a DEPENDENT chain (each AR feeds the next term's weight) so the fold requires proving
telescoping stability — the fam-3 property that made data-dependent scale trap.

---

# FAMILY-3 EXPANSION @ 9 SEEDS (r48, 2026-09-21) — +3 CONFIRMED → FAMILY-3 = 7 TOTAL

Expanded family-3 on its two confirmed axes. Pre-screen (prescreen_r48) dropped 4 gate-fails that
map the mechanism's HARD LIMITS: **relu_d9/meanabs_d9** (depth-9 recompute drifts past atol,
max_diff ~2.3 → the data-dependent depth CEILING is 8) and **xc_r5/xc_r6** (W^(rounds-1)=224^4≈2.5e9
destroys float precision, max_diff 481/1927 → cross-collective can't exceed r4 at W=224). 9 viable →
9-seed confirm (r48fam3):

| Problem | scale fn | best | median | p | CI | CONFIRMED |
|---------|----------|------|--------|------|--------|-----------|
| r48_dd_meansq_d8 (mean of squares) | continuous | 1.206 | 1.204 | 0.0001 | [1.204,1.205] | **✓ tightest CI in campaign** |
| r48_dd_square_d8 (mean²) | continuous | 1.157 | 1.033 | 0.0015 | [1.001,1.034] | **✓** |
| r48_dd_relu_d6 (relu @ depth 6) | continuous | 1.097 | 1.017 | 0.0107 | [1.001,1.056] | **✓** |
| r48_dd_absdev_d8 (abs deviation) | continuous | 1.036 | 1.186 | 0.0001 | [1.186,1.225] | ✗ best<1.05 (median huge) |
| r48_dd_halfabs_d8 (½·abs) | continuous | 1.000 | 1.033 | 0.0008 | [1.032,1.267] | ✗ best<1.05 |
| r48_dd_meanabs_d6 (abs @ d6) | continuous | 1.091 | 1.002 | 0.0132 | [0.988,1.153] | ✗ CI_lo<1 |
| r48_dd_shift_d8 (shifted abs) | continuous | 0.963 | 1.033 | 0.0098 | [0.929,1.267] | ✗ overlay won best-of |
| r48_dd_square_d8_res | continuous | 1.047 | 0.890 | 0.9667 | [0.853,1.000] | ✗ TIE (framing) |
| r48_dd_relu_d8_res | continuous | 0.787 | 1.003 | 0.3779 | [0.799,1.024] | ✗ TIE (framing) |

**Findings.** (1) **Family-3 is now 7 confirmed** (4 r47 + 3 r48). The strongest expansion member,
meansq_d8, has the TIGHTEST CI of the whole campaign [1.204,1.205] — heavier per-stage compute
(squaring) gives a bigger, more consistent fold. (2) **A distinct WIN MODE surfaced**: on meansq_d8
Sorcar keeps the same collective count but VECTORIZES the per-block Python loop (`.mean(dim=1)` over
a [B,S] view) while Overlay keeps the explicit block-loop → Sorcar wins on cheaper local ops, not
only on collective-count fold (the xc_r4 mode). Both are legitimate under the identical fair gate.
(3) **Framing matters more here than in fam-1/2**: both `_res` (result-only) controls TIE — without
the "N dependent all_reduce" count cue, Overlay's best-of-9 folds the deep chain as often as Sorcar.
The count cue is load-bearing for the data-dependent family (unlike fam-2, where routing structure
alone carried it). (4) The confirmed continuous fns (meansq, square, relu, meanabs) all trap;
abs-deviation, half-abs, shifted-abs escape best-of despite huge medians — the win is FRAGILE at the
strict best-of boundary but distributionally robust (all p ≤ 0.013). Ledger round r48fam3; artifacts
CONFIRMED_WINS_ARTIFACTS/family3_9seed_r48.

---

# FAMILY-3 (NEW) — DATA-DEPENDENT diagonal collapse + cross-collective (r46/r47, 2026-09-21)

**The old "family-3" (off-diagonal coupling, r42) had 0 confirmed → retired as a NEGATIVE, not a
family.** This is the genuinely NEW family-3, on a distinct axis from families 1 & 2.

**Mechanism (distinct from both prior families):** the per-block diagonal scale is a DATA-DEPENDENT
function g(·) of the REDUCED VALUES — computed at runtime from AR(SUM,x), NOT from the rank index
(family-2) and NOT a static multiplicative constant (family-1). The baseline is a depth-D chain
that RECOMPUTES g each stage and unscales; it telescopes to a single AR(SUM,x) + ONE application of
g. The fold requires a value-dependent TELESCOPING insight (each intermediate returns the plain
sum), which Overlay's enumerate misses. Sub-axis B is a cross-collective cluster: RS+AG rounds ≡
AR·W^(r-1) — a primitive-EQUIVALENCE fold (RS+AG across primitive types == AR), closest to real
FSDP/TP training.

**Pre-screen (r46 bo4 + r47 faithful gate/headroom/distinctness):** all baselines + ideal folds
run through the REAL fp32 scorer at W=224. r46 bo4 already CONFIRMED two members
(dd_meanabs_d8 best 1.267/CI[1.032,1.267]; xc_r2 best 1.116/CI[1.075,1.116]) — surprising, since
I'd predicted data-dependent scale would tie (rank-HOMOGENEOUS intermediates). It traps via the
DEPTH-8 recompute TELESCOPE, not rank-heterogeneity → mechanistically distinct from fam-1/2.

**r47 pre-screen DROPPED 4 as gate-FAILS (honest):** norm/norm_res (`.norm()`+tensor-sum not
traceable by MockTorch → TrackedTensor AttributeError) and meanabs_d10/topk_d10 (depth-10 recompute
drifts past atol, max_diff 2.4–4.2). **10 VIABLE** kept (gate+fold, headroom 1.12–1.31×, 8 unique
ref outputs + 2 by-design controls): dd_meanabs_d8, dd_relu_d8, dd_topk_d8, dd_meanabs_d7,
dd_meanabs_d8_p384, dd_topk_d8_p384, dd_topk_d8_res, xc_r2, xc_r3, xc_r4.

**FAMILY-3 = 4 CONFIRMED @ strict best-of-9** (both pipelines Bedrock Sonnet-4.5, ledger r47fam3):

| Problem | sub-axis | best | median | p | CI | CONFIRMED |
|---------|----------|------|--------|------|--------|-----------|
| r47_xc_r4 (RS+AG×4 ≡ AR·W³) | cross-collective | 1.271 | 1.272 | 0.0003 | [1.139,1.272] | **✓ strongest** |
| r47_dd_relu_d8 | data-dep scale | 1.267 | 1.267 | 0.0001 | [1.009,1.267] | **✓** |
| r47_dd_meanabs_d7 | data-dep scale | 1.187 | 1.033 | 0.0005 | [1.003,1.060] | **✓** |
| r47_dd_meanabs_d8 | data-dep scale | 1.100 | 1.032 | 0.0010 | [1.022,1.267] | **✓** |
| r47_dd_topk_d8 | data-dep SELECT | 1.001 | 1.209 | 0.0109 | [1.000,1.264] | ✗ overlay folds best-of |
| r47_xc_r3 | cross-collective | 1.194 | 1.000 | 0.0853 | [1.000,1.194] | ✗ CI_lo=1.0, p>.05 |
| r47_xc_r2 | cross-collective | 1.116 | 1.075 | 0.0533 | [1.000,1.116] | ✗ just misses p/CI |
| r47_dd_topk_d8_p384 | data-dep SELECT | 0.962 | 1.044 | 0.0018 | [1.007,1.265] | ✗ overlay won best-of |
| r47_dd_meanabs_d8_p384 | data-dep scale | 1.039 | 1.033 | 0.0108 | [0.941,1.265] | ✗ best<1.05 |
| r47_dd_topk_d8_res | data-dep SELECT | 0.998 | 0.929 | 0.8003 | [0.792,1.044] | ✗ TIE |

**Findings.** (1) **A genuine third family exists** on a distinct axis from families 1 & 2: the
diagonal scale is a runtime function of the REDUCED VALUES (data-dependent), not a static constant
(fam-1) or rank-index routing count (fam-2). It traps Overlay via a value-dependent TELESCOPING
insight over the depth-D recompute chain, NOT via rank-heterogeneous intermediates — overturning my
pre-run prediction that data-homogeneous intermediates would tie. (2) **CONTINUOUS scale functions
trap** (meanabs, relu confirm); **DISCRETE top-k SELECTION does NOT** (all 3 topk variants fail —
Overlay recognizes the MoE-like selection mask is stable across the chain and folds it). This
sharpens the boundary: the trap needs a data-dependent *continuous* scale, not a *combinatorial*
selection. (3) **Cross-collective RS+AG≡AR is real but depth-gated**: xc_r4 (8 collectives→1)
confirms strongly; xc_r3/r2 just miss the strict gate (headroom too thin at shallow depth). The
xc_r4 median code is a CLEAN interpretable divergence — Sorcar folds 4×(RS+AG) into one AR·W³;
Overlay keeps all 8 dispatches, explicitly commenting "no fusion." Closest of any confirmed problem
to real FSDP/TP training. (4) E2E implication: family-3 deltas ~1.03–1.27× (modest, like fam-2),
median artifacts curated for E2E in CONFIRMED_WINS_ARTIFACTS/family3_9seed/. Ledger round r47fam3.

## Three-family map (final, all @ strict best-of-9)
- **family-1** (11 confirmed): rank-het per-shard MULTIPLICATIVE diagonal scale a[r] — STATIC constant, ~2.3× E2E.
- **family-2** (7 confirmed): rank-INDEXED routing net = per-block COUNT c[b] — STATIC from routing overlap, ~1.0–1.24×.
- **family-3** (7 confirmed: r47 relu_d8/meanabs_d7/meanabs_d8/xc_r4 + r48 meansq_d8/square_d8/relu_d6): DATA-DEPENDENT continuous diagonal scale g(reduced values) + cross-collective RS+AG≡AR — RUNTIME function, ~1.02–1.27×. Depth ceiling 8 (float drift); xcoll ceiling r4 (W^n overflow); discrete top-k SELECTION and result-only framing both escape.
- **NEGATIVE (retired, not a family)**: off-diagonal coupling (r42, 0/8) — foldable banded matmul; off-diagonal structure is NOT a trap axis.
Unifying boundary: **the robust trap requires a DIAGONAL net** whose per-block factor Overlay's
enumerate can't reach in one shot — whether static-multiplicative (1), static-routing-count (2), or
data-dependent-continuous (3). Off-diagonal coupling and discrete selection both escape.

---

# FAMILY-2 EXPANDED + RE-RUN @ 9 SEEDS (r43, 2026-09-21) — 7/15 CONFIRMED

---

# FAMILY-2 EXPANDED + RE-RUN @ 9 SEEDS (r43, 2026-09-21) — 7/15 CONFIRMED

**Why 9 seeds:** odd N → the median run is a CONCRETE 5th-ranked realized code artifact
(retrievable per system: overlay.json['final_code'], kiss best_code.py) usable for a fair
median-Sorcar vs median-Overlay real E2E training comparison. All three families redone at 9.

**COUNT-VECTOR CORRECTION (supersedes the r40 "per-block count IS the scale" story):** the gate
runs at world_size W = num_nodes*32 = 224. 224 is divisible by B=8, so r40's net count vector is
UNIFORM [84,84,...] — r40's divergence is driven by the DEPTH-8 rank-indexed MASKING CHAIN
(rank-heterogeneous INTERMEDIATE buffers defeat Overlay's identity-shortcut), NOT by
heterogeneity of the net. r43 adds B∈{6,9,10,11,12} (224 NOT divisible → genuinely non-uniform
net at W=224) to test whether net-heterogeneity adds anything beyond the masking-chain effect.

Family-2 = 15 problems (4 r40 anchors + 11 distinct r43 variants: block-count B sweep, payload
384, strided topology, res framing). 9 unique reference outputs + 2 framing controls (verified by
output hashing). ALL 15 baselines AND their folds pass the real fp32 gate at W=224.

| Problem | best | median | p | CI | CONFIRMED |
|---------|------|--------|------|--------|-----------|
| r43_route_B11_d8_L3_count8 | 1.249 | 1.020 | 0.0002 | [1.008,1.249] | **✓** |
| r43_route_B10_strided_count8 | 1.243 | 1.017 | 0.0004 | [1.008,1.247] | **✓** |
| r43_route_B9_d8_L3_count8 | 1.124 | 1.244 | 0.0008 | [1.009,1.245] | **✓** |
| r40_route_d8_L3_count8 | 1.105 | 1.236 | 0.0011 | [1.005,1.243] | **✓** |
| r40_route_d6_L3_count8 | 1.081 | 1.162 | 0.0006 | [1.011,1.203] | **✓** |
| r43_route_B10_p384_count8 | 1.082 | 1.014 | 0.0030 | [1.008,1.248] | **✓** |
| r43_route_B12_d8_L3_res | 1.084 | 1.013 | 0.0020 | [1.012,1.251] | **✓** |
| r43_route_B10_d8_L3_count8 | 1.083 | 1.197 | 0.0028 | [1.000,1.247] | ✗ CI_lo=1.0 not >1 |
| r40_route_d8_L3_count8b | 1.063 | 1.014 | 0.0160 | [0.876,1.244] | ✗ (confirmed @8, dropped @9) |
| r43_route_B8_strided_count8 | 1.057 | 1.225 | 0.0117 | [0.902,1.243] | ✗ CI_lo<1 |
| r43_route_B6_d8_L3_count8 | 1.079 | 1.011 | 0.0127 | [0.882,1.239] | ✗ CI_lo<1 |
| r43_route_B12_d8_L3_count8 | 1.043 | 1.013 | 0.0006 | [1.012,1.251] | ✗ best<1.05 |
| r40_route_d8_L3_res | 1.038 | 1.015 | 0.0137 | [0.882,1.243] | ✗ best<1.05 |
| r43_route_B10_d8_L3_res | 1.023 | 1.017 | 0.0135 | [0.850,1.247] | ✗ best<1.05 |
| r43_route_B8_p384_count8 | 0.965 | 1.198 | 0.0085 | [1.005,1.244] | ✗ overlay won best-of |

**Findings:** (1) Family-2 generalizes — confirmed members now span B9/B10-strided/B10-p384/B11/
B12-res, NOT just the original B=8 anchors → rank-indexed routing is a real family across
block-count AND topology, not a one-config artifact. (2) Confirmation is FRAGILE at the threshold:
all wins in a thin ~1.05–1.25× band; two r40 anchors that confirmed at 8 seeds dropped at 9 (extra
seed shifted min/CI across the strict boundary). The distributional signal is ROBUST (all p
0.0002–0.016, medians consistently >1) but the best-of MARGIN is small. (3) E2E implication:
family-2 real-training deltas will be MODEST (~1.0–1.24×), unlike family-1's ~2.3× — the honest
"small-but-consistent win" story. Ledger round r43fam2.

# SECOND FAMILY CONFIRMED — rank-indexed ROUTING (r40, 2026-09-20)

**This overturns the "no second family" conclusion recorded below.** The prior close was
correct that no PURE-PERMUTATION (rank-independent) collapse can trap Overlay — after the
first AR every rank holds identical data, so a subsequent permute+AR is a locally-visible
identity. But the user's suggested lever — make the inter-collective state rank-heterogeneous
WITHOUT a multiplicative scale, via RANK-INDEXED routing — succeeds.

Family-2 mechanism (problems_diverge_r40.py): each stage, rank r masks a contiguous WINDOW of
L blocks starting at (r+OFF)%B; the window DEPENDS ON `rank`, so each rank contributes a
DIFFERENT masked buffer → AR(SUM) is a genuine reduction, NOT an identity → Overlay's
identity-shortcut is FALSE (naive AR-only guess fails fp32 gate, max_diff 6.5). Net over the
D-deep chain = per-block scale by c[b] = #ranks whose window covers b (a fixed count vector
from routing OVERLAP), with /c each intermediate stage to stay bounded. NO multiplicative
constant in the code — the "scale" emerges from routing counts. Distinct algebra from
family-1 (explicit diagonal a[r]).

Strict best-of-8 (8 seeds, both pipelines on Bedrock):

| Problem | best | median | p | CI | CONFIRMED |
|---------|------|--------|------|--------|-----------|
| r40_route_d8_L3_count8 | 1.188 | 1.243 | 0.0019 | [1.014,1.265] | **✓** |
| r40_route_d6_L3_count8 | 1.081 | 1.162 | 0.0058 | [1.001,1.204] | **✓** |
| r40_route_d7_L3_count8 | 0.965 | 1.098 | 0.0224 | [0.965,1.203] | ✗ (bo8 draw folds at d7) |
| r40_route_d8_L2_count8 | 1.024 | 1.096 | 0.0471 | [0.884,1.217] | ✗ |
| r40_route_d8_L4_count8 | 1.036 | 1.163 | 0.0224 | [0.874,1.247] | ✗ |

On d8_L3: overlay pinned at min 6143 (folds 0/8), kiss folds to 5173 on 6/8. Window-length is
a real knob (L=3 traps; L=2/L=4 escape). Headroom is thin (~1.29×) because the sim PIPELINES
light-compute AR chains (~217us marginal per AR vs r37's ~934us with non-contiguous cat), but
1.29× clears the ≥1.05 gate. Ledger round r40a.

## r40b (2026-09-20): framing/payload/depth robustness — 2 MORE CONFIRMED (family-2 = 4 total)

| Problem | best | median | p | CI | CONFIRMED |
|---------|------|--------|------|--------|-----------|
| r40_route_d8_L3_count8b (512 payload) | 1.068 | 1.014 | 0.0022 | [1.009,1.244] | **✓** |
| r40_route_d8_L3_res (result-only doc) | 1.081 | 1.010 | 0.0022 | [1.010,1.198] | **✓** |
| r40_route_d8_L3_big (1024 payload) | 1.082 | 0.946 | 0.1981 | [0.875,1.098] | ✗ (big payload → less reliable fold) |
| r40_route_d6_L3_big | 1.053 | 0.970 | 0.1717 | [0.913,1.164] | ✗ |
| r40_route_d5_L3_count8 | 1.017 | 1.098 | 0.0098 | [0.967,1.122] | ✗ |
| r40_route_d6_L3_res | 1.001 | 1.024 | 0.1098 | [0.896,1.162] | ✗ |
| r40_route_d8_L2_res | 1.023 | 1.049 | 0.0448 | [0.848,1.242] | ✗ |
| r40_route_d8_L4_res | 1.124 | 1.118 | 0.0290 | [0.916,1.292] | ✗ (CI_lo<1) |
| r40_route_d4_L3_res | 1.076 | 1.003 | 0.0736 | [0.999,1.006] | ✗ |
| r40_route_d4_L3_count8 | 1.107 | 1.002 | 0.1847 | [0.998,1.081] | ✗ |

**Family-2 = 4 CONFIRMED members** (all d8_L3 except d6_L3_count8): d8_L3_count8, d6_L3_count8,
d8_L3_count8b, d8_L3_res. Pattern reconfirms family-1's shape: **D=8 depth robustly traps
Overlay; shallow depths (d4/d5/d6) and window L≠3 tie**. Unlike family-1, the win is INDEPENDENT
of the count cue (count8/count8b/res all confirm at d8_L3) — routing structure alone carries it.
Larger payload (`_big`, 1024) makes kiss's fold LESS reliable per-seed (median<1), so the thin
headroom needs the smaller payload to hold bo8. Ledger rounds r40a+r40b. r42 (third-family
off-diagonal probe) running.

# THIRD FAMILY PROBE — rank-indexed OFF-DIAGONAL coupling (r42, 2026-09-20)

Both confirmed families have a DIAGONAL net (family-1 mult scale; family-2 count scale). r42
tests an OFF-DIAGONAL (banded/Toeplitz) net: each stage rank r adds its left-neighbor block
only at its own block b=r%B; net = M^(D-1)/W^(D-2) where M = W*I + subdiagonal(p), a
diagonal-dominant lower-bidiagonal band that is NOT doubly-stochastic (so it does NOT converge
to the uniform average that makes conservative rotation/permutation TIE). Validated locally:
baseline 6583 / fold 5377 (headroom 1.22×), fold passes fp32 gate, naive AR-only fails
(coupling is ~93% of signal). problems_diverge_r42.py, 8 problems (depth sweep + res +
big). Status: staged/running.

--- prior battery (superseded conclusion) below ---

# Distinct-Family Divergence Battery (2026-09-20, +5h budget)

Goal (user): find divergences in DIFFERENT families than the known rank-heterogeneous
MULTIPLICATIVE per-shard collapse (family-1, 19 robust wins). Do NOT re-iterate family-1
variants or known-tie reduce_scatter-floor structures. Screen distinct MECHANISMS cheaply,
promote survivors to best-of-16. Criterion unchanged: best-of-N ratio >=1.05 AND MW-U
p<.05 AND bootstrap CI_lo>1.0.

Each round is a genuinely DISTINCT collapse mechanism (different algebra/primitive), not a
docstring/layout/constant variant of an existing win.

| Round | Mechanism | Distinct axis vs family-1 | Baseline sim (us) | Status |
|-------|-----------|---------------------------|-------------------|--------|
| r34_maxplus8 / minplus8 | tropical telescope: AR(MAX/MIN) + per-block additive offset | DIFFERENT semiring (max-plus/min-plus) AND primitive (MAX/MIN not SUM); reversible op is + not x | 11821 | **bo8 NOT confirmed** — forward but escapable (max best 1.06, min best 0.79); behaves like r14 additive |
| r35_masksum8 | disjoint masked partial-sums accumulate to one AR(SUM) | combinatorial partition-of-unity, NOT arithmetic scale | 7101 (low headroom ~1.38x) | **bo8 TIE** (best 1.0/med 1.06/p.28) — overlay sees the partition & folds |
| r36_affine8 | per-shard AFFINE a[r]*b+c[r] chain | MIXED mult+additive (neither family-1 pure-mult nor r14 pure-additive) | 14957 (~2.9x) | **bo16 CONFIRMED** (best 1.291/med 2.162/p=0.0/CI[2.160,2.169]) — HOLDS; but affine RETAINS a multiplicative factor (still needs /a to invert) |
| r37_rotsum8 | block-ROTATION chain composing to 1 net rotation | GROUP-THEORETIC (symmetric group composition), NOT scalar | 11746 (~2.3x) | **bo16 NOT confirmed** (best 1.399/med 1.704/p=0.0061/CI[**1.0**,2.257]) — CI_lo fails strict gate. The scale-free candidate ESCAPES at bo16 → NOT a second family |
| r38_reflect8 | reverse-then-rotate (dihedral) permutation chain | DIFFERENT group action than r37 (reflection/involution) | 8444 (~1.6x) | NOT RUN — superseded by r39 which directly tuned the permutation family |
| r39_permsum8 | rotate-by-k ∘ FIXED non-affine PI=[3,0,5,7,1,6,2,4]; net map has NO closed form (part=4096, baseline 6769 / fold 5202) | tuned r37: kills the closed-form-shift shortcut so overlay can't one-shot the fold | 6769 (~1.30x) | **bo8 NOT confirmed** — count8: overlay pinned 7/8 at baseline, kiss folds 3/8 (p .0015, CI_lo 1.005) BUT overlay's 1 folding seed keeps 7 LOCAL permutes → best-ratio 1.043 < 1.05. res: TIE (both fold ~2/8). **DECISIVE NEGATIVE — see below** |

## Sharpening mechanism map (what traps vs what escapes) — FINAL, bo16-adjudicated
The bo16 recheck REVERSED the bo8 over-read. bo8 confirmation is NOT robust: BOTH r37
(scale-free rotation) and r33_permscale earlier passed bo8 then behaved differently under
the strict bo16 gate. The decisive adjudication:
- ROBUST TRAPS (bo16, CI_lo>1.0): mult scale/SUM (family-1, division to invert);
  affine/SUM (r36, best=1.291/med=2.162/CI[2.160,2.169]). **Both retain a multiplicative
  factor** — affine is mult+additive, and its inverse still requires a division by a[r].
- ESCAPES AT bo16: block-rotation/SUM (r37 — best=1.399 but CI_lo=1.0, so NOT confirmed).
  This is the sharp result: the ONE candidate with NO multiplicative component escapes
  the strict gate. Overlay's best-of-16 draw folds the permutation reliably enough that
  the CI touches 1.0.
- ESCAPES/TIES AT bo8 already: tropical additive/MAX (r34 — additive inverse trivial);
  partition mask/SUM (r35 — locally-visible); pure shear (r33 — order-dependent);
  additive zero-sum/SUM (r14 — overlay MORE reliable).
=> FINAL claim (bo16-supported): the robust trap requires a per-shard op with a
MULTIPLICATIVE component whose inverse is a division. Pure-permutation and pure-additive
collapses do NOT yield a robust (bo16 CI_lo>1.0) second family. The r34–r37 battery of
genuinely distinct mechanisms therefore VINDICATES the multiplicative-component boundary
rather than overturning it: no second family survives the confirmation criterion. This
directly answers the user's "does a second family exist?" — under strict, symmetric,
best-of-16 confirmation across 5 distinct algebras, NO.

## r39 (tuned pure-permutation) — the DECISIVE MECHANISTIC reason there is no 2nd family
The user asked whether r37 could be tuned so Sorcar > Overlay holds at BOTH best-of-8 and
avg-sim (which would make it a genuine 2nd family). r37 escaped bo16 because its net map is
a closed-form cyclic shift by sum(1..7)=28. r39 removed the closed form: each stage rotates
by k THEN applies a fixed non-affine, non-involutive PI=[3,0,5,7,1,6,2,4], so the composed
net permutation has no shortcut and overlay's obvious "apply PI once / shift by c" guesses
FAIL the fp32 gate (verified: single-PI guess → max_diff≈4.4). Result at strict best-of-8:
- **count8**: overlay pinned at baseline 6769 on 7/8; kiss folds to the true 5202 on 3/8.
  p=0.0015, CI_lo=1.005 (kiss distributionally faster) — BUT overlay's ONE folding seed
  kept 7 LOCAL permutes after 1 AR (5424), so best-ratio = 5424/5202 = **1.043 < 1.05**.
- **res**: TIE (overlay folds 2/8, kiss 2/8).

**Why tuning the permutation can't work — the crux.** Overlay's folding seed reasoned:
"after the first all_reduce all ranks hold identical data, so every subsequent
all_reduce(SUM,·)/W of identical data is the identity — keep the permutes LOCAL." That is a
LOCALLY-VISIBLE identity (the L8 tie class). The permutation's complexity is IRRELEVANT:
between collectives a pure permutation operates on data that is identical across ranks, so
the "AR-of-identical-data = identity" fold is trivially visible no matter how non-closed-form
the net map is. Family-1 traps precisely because per-shard MULTIPLICATIVE scale makes the
intermediate data RANK-HETEROGENEOUS, so that shortcut is FALSE and the collapse (scale
distributes through SUM) is a genuine non-local algebraic insight overlay's enumerate can't
reach. This mechanistically explains WHY r37 escaped bo16 and confirms the single-family
result is STRUCTURAL, not sample-size-limited. Any pure-permutation collapse (no per-shard
scale) reduces to the identity-shortcut and ties → no second family is reachable by tuning
the permutation. Ledger key: round_r39_pure_permutation_tuned; artifacts in
CONFIRMED_WINS_ARTIFACTS/second_family_candidate/.

## Distinctness rationale (why these are NOT family-1 variants)
- family-1 collapse: `sum_r a[r]*x[r]` — a[r] MULTIPLICATIVE, distributes through AR(SUM).
- r34: `max_r (x[r] + b)` — + commutes through MAX. Different semiring entirely. If it
  confirms, "multiplicative-specific" is FALSE; the trap is "any reversible per-shard op
  commuting through its reduction." Sharpest test of last turn's over-claim.
- r35: `sum_k mask_k ⊙ AR(x) = AR(x)` — support-disjointness, not scaling. Analogous to
  r10 idempotence BUT non-locally-visible (each stage masks a different block).
- r36: affine mixes the family-1 axis (mult, confirms) with the r14 axis (additive, ties)
  — tests which axis dominates when combined.
- r37: composition in the symmetric group, reduction is order-independent so rotations
  telescope. Distinct from r24_pairwise (self-cancelling no-op that tied) because no
  adjacent pair cancels.

## Prior distinct-mechanism attempts (for honesty — most TIED)
r8 telescope, r10 re-max/AG-roundtrip, r11 max-offset, r12 mixed-primitive, r13 hetero
SUM+MAX+MIN, r14 additive zero-sum (overlay MORE reliable), r24 global-scalar/permutation-
invariance, r33 pure shear (order-dependent, tied). Only family-1 (mult scale) + r33
permscale (family-1 + routing) confirmed. This battery widens the distinct-mechanism net.


---
## r53/r54 MAX/MIN max-plus semiring — NEGATIVE (recorded 2026-09-21)

**Verdict: NOT a divergence family. Best-of-4 TIE (r53 best 1.0-1.27 but medians ~1.0,
no p<0.05; r54 even weaker, best 0.999-1.009).**

Root cause (from cost-model audit, correctness_test.py):
- Scoring cost is COLLECTIVE-TYPE-AGNOSTIC: `_coll_bandwidth_floor_us` (line 2860) = bytes/cluster_bw,
  uniform across AR/AG/RS. `all_gather` records INPUT bytes (line 880) = same floor as `all_reduce`.
- => gather-then-local achieves 1 collective AND passes fp32 gate: a UNIVERSAL escape for any
  deterministic function of all ranks' inputs. Cost can NEVER be the divergence lever.
- Divergence comes ONLY from overlay's enumerate firing a CONFIDENTLY-WRONG linear template and
  failing the correctness gate (always-fail bucket), while Sorcar iterates to a passing fold.
- MAX/MIN has NO wrong-firing linear template in overlay's enumerate => overlay falls straight to
  the correct general gather-then-local and passes => TIE.

**LESSON: the trap law is SUM-SPECIFIC. It only bites when overlay has a confidently-wrong
SUM-linear template. fam-1/2/3 are the known SUM survivors (multiplicative scale, selective count,
data-dependent diagonal). A 4th family must be a NEW SUM survivor overlay's linear enumerate
mis-folds — NOT a new reduction op.**

Next axis (r55): RATIO/NORMALIZATION survivor. Weighted mean (Σ w_r x_r)/(Σ w_r); overlay's linear
template produces AR(SUM, w·x) and forgets the denominator => wrong => gate fail => baseline.
Correct fold = bundle [w·x, w] into ONE AR(SUM) then divide. Distinct survivor: a coupled ratio.


---
## CORRECTED TRAP LAW (2026-09-21, from fam-1/fam-3 overlay-code audit)

Prior theory ("overlay fires a confidently-WRONG linear template and FAILS the gate") is WRONG.
Ground truth from confirmed wins:
- fam-1 r23_deep8_count8: overlay final_sim=12919 == baseline EXACTLY (gate PASSED); kiss=6191.
- fam-3 r48_dd_absdev_d8: overlay final_sim=7969 == baseline EXACTLY (gate PASSED); kiss lower.

Overlay does NOT fail correctness. It writes CORRECT but per-rank / per-block PYTHON SLICE-ASSIGN
LOOPS (`for r in range(W): buf[r*S:(r+1)*S] = a[r]*s[...]`) that (1) do NOT vectorize on Neuron
(baseline cost) and (2) never COLLAPSE the depth-D all_reduce chain. Kiss reliably (a) VECTORIZES
the per-index factor into a fused weight tensor / matmul and (b) COLLAPSES D all_reduces -> 1 by
pulling the linear transform out.

=> The divergence lever is EFFICIENCY (vectorization + AR-collapse), driven by a structure that
   TEMPTS overlay into a per-rank/per-block Python loop. NOT a gate failure.
=> Why MAX/MIN tied: a broadcast-add shift does NOT tempt a per-rank Python loop; overlay writes
   vectorized code with no chain to collapse -> tie.
=> A 4th family = a NEW per-index structure that (i) strongly tempts the Python loop AND (ii) has a
   deep telescoping AR-chain kiss can collapse. Must keep data rank-HETEROGENEOUS between ARs
   (scale/unscale pairing) so depth does real work (weighted-MEAN normalization idempotently
   reaches a fixed point -> NO depth -> weak; deprioritized).

r56 candidate: PER-RANK BLOCK-PERMUTATION + scale => cross-block COUPLING MATRIX M[b,b'].
  stage: buf_r[b] = (a[r]/W) * cur_r[pi_r(b)];  m = AR(SUM, buf);  cur = m
  m1[b] = sum_r (a[r]/W) x_r[pi_r(b)]  (genuine, per-rank x); later stages m_{t+1}=M@m_t,
  M[b,b'] = (1/W) sum_{r: pi_r(b)=b'} a[r]  => m_D = M^(D-1) @ m1.
  FOLD (kiss): precompute M (BxB), ONE AR for m1 (vectorized gather+scale), matmul M^(D-1). 1 coll.
  OVERLAY temptation: D ARs each preceded by nested `for b:` permute+scale Python loops. Distinct
  surface (permutation coupling, not diagonal scale); deep chain; gate-exact (SUM + local).
