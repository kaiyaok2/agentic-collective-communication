#!/usr/bin/env python3
"""Sim-vs-measured analysis for the rebuttal (A-Q2 / C-Q1).
Inputs (pulled from master): sweep_results/*.json (measured avg_ms per candidate),
sweep_results/sim_default.json (hand-set-calibration sim), /tmp/ap7/sim_paper.json (recorded
LLM-calibrated sim where available). Outputs Spearman (overall + within-problem), top-1 recall, regret.
Usage: python3 analyze_sweep.py <dir with sweep_results>
"""
import json, glob, os, sys
from collections import defaultdict

def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
    i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and v[s[j+1]] == v[s[i]]: j += 1
        for k in range(i, j+1): r[s[k]] = (i+j)/2 + 1
        i = j+1
    return r
def spearman(x, y):
    n = len(x)
    if n < 3: return None
    rx, ry = rank(x), rank(y); mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = (sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry)) ** .5
    return num/den if den else None

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ap7/pull/sweep_results"
meas = {}
for f in glob.glob(os.path.join(d, "*__*.json")):
    j = json.load(open(f))
    v = j.get("measured_ms", j.get("avg_ms"))
    if v is not None: meas[f"{j['problem']}__{j['cand']}"] = float(v)
sim_def = json.load(open(os.path.join(d, "sim_default.json"))) if os.path.exists(os.path.join(d, "sim_default.json")) else {}
sim_pap = {k: v["sim_paper_us"] for k, v in json.load(open("/tmp/ap7/sim_paper.json")).items()}
print(f"measured candidates: {len(meas)}")
for name, sim in [("hand-set default calibration", sim_def), ("recorded LLM calibration", sim_pap)]:
    keys = [k for k in meas if sim.get(k) is not None]
    print(f"\n=== {name}: {len(keys)} (sim, measured) pairs ===")
    rho = spearman([sim[k] for k in keys], [meas[k] for k in keys]); print(f"overall Spearman rho = {rho}")
    byp = defaultdict(list)
    for k in keys: byp[k.split("__")[0]].append(k)
    top1 = 0; nprob = 0; regrets = []
    for p, ks in sorted(byp.items()):
        if len(ks) < 2: continue
        nprob += 1
        s = [sim[k] for k in ks]; m = [meas[k] for k in ks]
        r = spearman(s, m)
        sim_pick = ks[s.index(min(s))]; meas_best = ks[m.index(min(m))]
        reg = (meas[sim_pick] - meas[meas_best]) / meas[meas_best] * 100
        regrets.append(reg); top1 += (sim_pick == meas_best)
        print(f"  {p:15s} n={len(ks)} rho={r if r is None else round(r,2)}  sim-pick={sim_pick.split('__')[1]} "
              f"({meas[sim_pick]:.2f} ms) measured-best={meas_best.split('__')[1]} ({meas[meas_best]:.2f} ms) regret={reg:.1f}%")
    if nprob:
        print(f"top-1 recall {top1}/{nprob}; mean regret {sum(regrets)/len(regrets):.1f}%; max regret {max(regrets):.1f}%")
print("\n=== per-candidate table ===")
for k in sorted(meas): print(f"{k:50s} meas={meas[k]:9.2f} ms  sim_def={sim_def.get(k)}  sim_paper={sim_pap.get(k)}")
