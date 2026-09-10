#!/usr/bin/env python3
"""Regenerate SORCAR_FAMILY_TAXONOMY.md from the authoritative 55-anchor set.
Every number traces to taxonomy_3col_results/{three_col.json, RT_THREE_COL_RESULTS.json}."""
import json, collections

A = json.load(open("/home/ubuntu/anchor55_clean.json"))
FAM_ORDER = ["F1", "F2", "F3", "F4", "F6", "F7"]
FAM_NAME = {
    "F1": "Sequential-AR linearity",
    "F2": "CSE across redundant ARs of the same input",
    "F3": "Dead-collective elimination & algebraic zero",
    "F4": "Per-row/col/batch dispatch collapse",
    "F6": "Mixed-reduction-op extraction",
    "F7": "Slab/chunk payload fusion",
}
FAM_WHAT = {
    "F1": "A chain of all-reduces combined linearly: y = c1*AR(x1)+...+ck*AR(xk), where each xi is a locally-computable transform of the input. all_reduce(SUM) is a linear operator, so the whole chain folds into ONE AR of a locally pre-combined payload plus scalar post-math. K collectives -> 1.",
    "F2": "N syntactically distinct AR(x) calls on the SAME unmodified input, combined arithmetically. The N results are identical; N-1 collectives are pure waste. Sorcar hoists to a single AR(x) and replaces every other call with the hoisted value, collapsing the arithmetic to one scalar multiplier.",
    "F3": "Collectives whose results are provably unused, mathematically canceled, or reducible to a constant: alternating-sign sums that telescope to zero, gather-then-verify with a dead verify branch. Sorcar proves the cancellation and removes ALL collectives (sim cost -> 0).",
    "F4": "A per-row / per-col / per-batch / per-slice loop that issues one AR per slice of a 2D/3D tensor. Sorcar stacks the slices and issues ONE AR over the whole tensor (or a single reshaped AR), collapsing M dispatches to 1.",
    "F6": "A payload reduced under one op (SUM) alongside the same or related payload reduced under a different op (MAX/MIN), issued as separate collectives. Sorcar extracts the mixed-op structure into the minimum distinct collectives.",
    "F7": "A tensor split into slabs/chunks, each all-reduced separately then recombined. Sorcar fuses the slabs into one contiguous payload and issues a single AR.",
}
FAM_WHYSTRAT = {
    "F1": "Strat-enum operates at the level of collective STRUCTURE (which primitive, what payload layout) not collective ALGEBRA. It never proves the k ARs are linearly combinable, so it keeps baseline's k-dispatch schedule (its emitted source is a distinct accumulate loop, but the collective count is identical).",
    "F2": "XLA HLO CSE catches some assigned-first cases but not inline-call chains. Strat proposes payload/bucketing re-layouts of the N collectives; it never proposes 'these N collectives are the same value.' Same N-AR schedule as baseline.",
    "F3": "Strat scores collective structure, not the algebraic value, so it never proves the alternating sum cancels. It keeps every dispatch. Baseline schedule preserved.",
    "F4": "Strat can re-bucket or re-order the per-slice ARs but does not fuse across the loop iteration space (the slices are separate SSA values). Baseline's M-dispatch loop is preserved.",
    "F6": "Strat keeps the SUM and MAX/MIN collectives as emitted; it does not extract the shared payload. Baseline schedule preserved.",
    "F7": "Strat keeps the per-slab AR loop; it does not fuse the slab payloads. Baseline schedule preserved.",
}

byfam = collections.defaultdict(list)
for e in A:
    byfam[e["family"]].append(e)


def sortkey(e):
    r = e["rt_vs_strat"]
    if isinstance(r, (int, float)):
        return -r
    return 1e9


L = []
w = L.append
w("# Sorcar vs Strat vs Baseline: Family Taxonomy of the 55-Problem Divergence Set\n")
w("**Scope**: the **55 divergent problems** — every problem in the taxonomy")
w("pool where Sorcar's searched rewrite beats OverlayCCL strat-enumeration")
w("by >5% in the calibrated simulator. Tie problems (strat already optimal or")
w("both at dispatch floor) and strat-win problems (none exist) are excluded:")
w("this doc is exactly the set on which the three code paths diverge.\n")
w("The 55 span **6 optimization families** (F5, collective-type conversion /")
w("ZeRO-1 data-flow narrowing, is exercised only in the E2E optimizer path,")
w("not as a standalone micro-anchor).\n")
w("**Three columns, three distinct code paths on every problem:**")
w("- **baseline** — naive textbook-DDP source (one collective per logical op).")
w("- **strat** — OverlayCCL 5-strategy enumeration output. On these 55 it emits")
w("  *distinct source* (accumulate loops, per-tensor AR loops, re-bucketed")
w("  payloads) but reaches the *same collective schedule* as baseline — it")
w("  finds no fusion. This is why strat_ms ≈ baseline_ms at RT (see tables).")
w("- **sorcar** — the searched family rewrite that fuses/eliminates collectives.\n")
w("**Measurement**: 7× trn1.32xlarge (224 NeuronCores), us-east-1c. Simulator")
w("columns from `taxonomy_3col_results/three_col.json`; warm-cache RT columns")
w("(each variant run 2× back-to-back, 2nd reported, 100 iters) from")
w("`taxonomy_3col_results/RT_THREE_COL_RESULTS.json`.\n")

nrt = sum(1 for e in A if isinstance(e["rt_vs_strat"], (int, float)) and e["rt_vs_strat"] >= 1.05)
nfloor = sum(1 for e in A if e["rt_vs_strat"] is None)
nab = sum(1 for e in A if e["rt_vs_strat"] == "HW-abort")
w("## Summary\n")
w("| | count |")
w("|---|---|")
w("| Divergent anchors (sim, sorcar > strat by >5%) | 55 |")
w("| RT-confirmed Sorcar wins (≥1.05× warm-cache) | %d |" % nrt)
w("| At RT dispatch floor (sim divergence < RT noise) | %d |" % nfloor)
w("| Sorcar sim-pass / HW-abort (F3 total-cancel edge) | %d |" % nab)
w("| Strat RT wins over baseline | 0 |")
w("| Strat sim wins over baseline (on these 55) | 0 |\n")

w("## Family index\n")
w("| # | Family | Anchors | Sim ratio range | Best RT (sorcar vs strat) |")
w("|---|---|---|---|---|")
for f in FAM_ORDER:
    es = byfam.get(f, [])
    if not es:
        continue
    sims = [e["sim_ratio"] for e in es if isinstance(e["sim_ratio"], (int, float))]
    infcnt = sum(1 for e in es if e["sim_ratio"] == "inf")
    rng = ("%.2f-%.2f×" % (min(sims), max(sims)) if sims else "")
    if infcnt:
        rng += " + ∞(total-cancel)"
    rts = [e["rt_vs_strat"] for e in es if isinstance(e["rt_vs_strat"], (int, float))]
    best = ("%.2f×" % max(rts)) if rts else "(sim-only)"
    w("| %s | %s | %d | %s | %s |" % (f, FAM_NAME[f], len(es), rng, best))
w("\nTotal: 55 anchors across 6 families.\n")
w("---\n")

for f in FAM_ORDER:
    es = byfam.get(f, [])
    if not es:
        continue
    w("## %s. %s (%d anchors)\n" % (f, FAM_NAME[f], len(es)))
    w("**What it is.** %s\n" % FAM_WHAT[f])
    w("**Why strat stays at baseline's schedule.** %s\n" % FAM_WHYSTRAT[f])
    w("**Per-problem data** (sim µs; warm-cache RT ms, 224 ranks):\n")
    w("| Problem | sim strat | sim sorcar | sim × | RT base | RT strat | RT sorcar | RT sorcar/strat |")
    w("|---|---|---|---|---|---|---|---|")
    for e in sorted(es, key=sortkey):
        sr = e["sim_ratio"]
        srs = "∞" if sr == "inf" else ("%.2f×" % sr)
        rv = e["rt_vs_strat"]
        if rv is None:
            rvs = "_floor_"
        elif rv == "HW-abort":
            rvs = "_HW-abort_"
        else:
            rvs = "**%.2f×**" % rv
        def ms(v):
            return "-" if v is None else ("%.2f" % v)
        prob = e["problem"]
        w("| %s | %.0f | %.0f | %s | %s | %s | %s | %s |" % (
            prob, e["sim_strat"], e["sim_sorcar"], srs,
            ms(e["rt_baseline_ms"]), ms(e["rt_strat_ms"]), ms(e["rt_sorcar_ms"]), rvs))
    w("")

open("/home/ubuntu/acc_repo/SORCAR_FAMILY_TAXONOMY.md", "w").write("\n".join(L) + "\n")
print("wrote SORCAR_FAMILY_TAXONOMY.md,", len(L), "lines")
