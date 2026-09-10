#!/usr/bin/env python3
"""Aggregate the two-arm faithful-ablation cost comparison.
full  = research-discovery loop + adversarial methodology ENABLED
ablated = both DISABLED. Same 8 problems, same scorer, same model."""
import json, os

PROBS = ["sixtyfourinline", "eightyaltsum", "nine_ar_same_input_chal",
         "sequential_ar_chain_edge_chal", "perslice3dM96", "perrowM64N4K",
         "mixmaxmin", "eightslab"]
ROOT = "/home/ubuntu/ablation_cost"


def tok(path):
    it = ot = cc = cr = n = 0
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            it += r["input_tokens"]; ot += r["output_tokens"]
            cc += r["cache_creation_input_tokens"]
            cr += r["cache_read_input_tokens"]; n += 1
    return it, ot, cc, cr, n


def arm(a):
    T = dict(it=0, ot=0, cc=0, cr=0, n=0, calls=0, wall=0.0)
    rows = []
    for p in PROBS:
        d = "%s/%s/%s" % (ROOT, a, p)
        it, ot, cc, cr, n = tok(d + "/tokens.jsonl")
        s = json.load(open(d + "/kiss_summary.json"))
        rows.append((p, n, s["n_score_calls"], round(s["wall_seconds"], 1),
                     it, ot, cc, cr, round(s["baseline_sim_time_us"]),
                     round(s["best_sim_time_us"])))
        T["it"] += it; T["ot"] += ot; T["cc"] += cc; T["cr"] += cr; T["n"] += n
        T["calls"] += s["n_score_calls"]; T["wall"] += s["wall_seconds"]
    return rows, T


summ = {}
for a in ("full", "ablated"):
    rows, T = arm(a)
    print("===== ARM: %s =====" % a)
    hdr = ("%-32s%5s%5s%7s%8s%8s%9s%9s%8s%8s"
           % ("problem", "LLMc", "scr", "wall", "in", "out", "cwr", "crd",
              "base", "best"))
    print(hdr)
    for r in rows:
        print("%-32s%5d%5d%7.1f%8d%8d%9d%9d%8d%8d"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))
    tot_in = T["it"] + T["cc"] + T["cr"]
    print("%-32s%5d%5d%7d%8d%8d%9d%9d"
          % ("TOTAL", T["n"], T["calls"], round(T["wall"]),
             T["it"], T["ot"], T["cc"], T["cr"]))
    print("  in+cache = %d ; output = %d ; total tokens = %d"
          % (tot_in, T["ot"], tot_in + T["ot"]))
    print()
    summ[a] = dict(llm_calls=T["n"], score_calls=T["calls"],
                   wall_s=round(T["wall"], 1), in_tok=T["it"],
                   out_tok=T["ot"], cache_create=T["cc"], cache_read=T["cr"],
                   total_tok=tot_in + T["ot"])

f, ab = summ["full"], summ["ablated"]


def ratio(k):
    return (ab[k] / f[k]) if f[k] else float("nan")


print("===== ABLATED vs FULL (cost multiplier, >1 = ablation costs more) =====")
for k in ("llm_calls", "score_calls", "wall_s", "in_tok", "out_tok",
          "total_tok"):
    print("  %-14s full=%-10s ablated=%-10s  x%.3f"
          % (k, f[k], ab[k], ratio(k)))

json.dump({"full": f, "ablated": ab,
           "ablated_over_full": {k: (ab[k] / f[k] if f[k] else None)
                                 for k in f}},
          open(ROOT + "/ablation_cost_summary.json", "w"), indent=2)
print("\nwrote", ROOT + "/ablation_cost_summary.json")
