#!/usr/bin/env python3
"""Merge sweep pass 1 and pass 2: per-candidate mean, spread, and per-problem ranking stability."""
import json, glob, os, sys
from collections import defaultdict
root = "/tmp/ap7/pull"
runs = defaultdict(list)
for d in ("sweep_results", "sweep_results_pass2"):
    for f in glob.glob(os.path.join(root, d, "*__*.json")):
        j = json.load(open(f)); v = j.get("measured_ms")
        if v is not None: runs[f"{j['problem']}__{j['cand']}"].append(float(v))
sim_pap = {k: v["sim_paper_us"] for k, v in json.load(open("/tmp/ap7/sim_paper.json")).items()}
sim_def = json.load(open(os.path.join(root, "sweep_results", "sim_default.json")))
byp = defaultdict(list)
for k, vs in sorted(runs.items()):
    byp[k.split("__")[0]].append((k, vs))
print(f"{'candidate':50s} {'n':>2s} {'mean ms':>9s} {'spread%':>8s} {'sim_paper':>10s} {'sim_def':>8s}")
for p, items in sorted(byp.items()):
    items.sort(key=lambda t: sum(t[1]) / len(t[1]))
    for k, vs in items:
        m = sum(vs) / len(vs); spread = (max(vs) - min(vs)) / m * 100 if len(vs) > 1 else float("nan")
        print(f"{k:50s} {len(vs):2d} {m:9.2f} {spread:8.1f} {str(sim_pap.get(k))[:9]:>10s} {str(sim_def.get(k))[:7]:>8s}")
    # rank stability: does pass1 ordering equal pass2 ordering?
    p1 = [k for k, vs in sorted(items, key=lambda t: t[1][0])]
    p2 = [k for k, vs in sorted([t for t in items if len(t[1]) > 1], key=lambda t: t[1][1])]
    if len(p2) > 1: print(f"   {p}: pass1 best={p1[0].split('__')[1]}  pass2 best={p2[0].split('__')[1]}  same={p1[0]==p2[0]}")
