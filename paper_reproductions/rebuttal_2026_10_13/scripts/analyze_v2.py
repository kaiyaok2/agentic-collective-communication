#!/usr/bin/env python3
"""Pool-v2 simulator-vs-measured analysis (A-Q2 / C-Q1), all 8 problems, 3 repeats.
Measured: step-based measured_ms (median of steady steps) and, where harvested, per-call collective time
from the PERCALL probe. Sim: default-calibration sim_default_v2.json (consistent across the whole pool)
and recorded LLM-calibrated scores where they exist.
Outputs: per-candidate mean ± sd, within-problem Spearman (8 problems), pooled Spearman on
ratio-to-baseline (unit-free), top-1 recall, regret. Usage: python3 analyze_v2.py [/tmp/ap7/pull/sweep_v2]
"""
import json, glob, os, sys, statistics
from collections import defaultdict
root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ap7/pull/sweep_v2"

def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
        for k in range(i, j + 1): r[s[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def spearman(x, y):
    n = len(x)
    if n < 3: return None
    rx, ry = rank(x), rank(y); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else None

runs = defaultdict(list); percall = defaultdict(list)
EXTRA = [d for d in os.environ.get("EXTRA_DIRS", "/tmp/ap7/pull/sweep_a2av,/tmp/ap7/pull/sweep_found").split(",") if d and os.path.isdir(d)]
for rep in sorted(glob.glob(os.path.join(root, "rep*"))) + EXTRA:
    for f in glob.glob(os.path.join(rep, "*__*.json")):
        if f.endswith("_percall.json"):
            try:
                d = json.load(open(f)); key = os.path.basename(f)[:-len("_percall.json")]
                vals = [v for k, v in d.items() if isinstance(v, list) and v and isinstance(v[0], (int, float))]
                if vals: percall[key].append(statistics.median(vals[0]))
            except Exception: pass
            continue
        j = json.load(open(f)); v = j.get("measured_ms")
        if v is not None: runs[f"{j['problem']}__{j['cand']}"].append(float(v))
SIMF = os.environ.get("SIM", os.path.join(root, "sim_default_v2.json"))
sim = json.load(open(SIMF)) if os.path.exists(SIMF) else {}
print("sim file:", SIMF)
# AllToAllV per-problem harness calls the runtime's fixed helper, so score helper variants by the algorithm the
# helper implements (grammar genome sims), not by the file's evolved_alltoallv
gpath = "/tmp/ap7/pull/grammar_search.json"
if os.path.exists(gpath) and "sim_default" in SIMF:
    ga = json.load(open(gpath))["alltoallv"]["all"]
    def gsim(structure, extract):
        for k, v in ga.items():
            d = json.loads(k)
            if d.get("structure") == structure and d.get("extract") == extract: return v
    sim["alltoallv__helper_AGRS"] = gsim("ag_rs", "loop_cat")
    sim["alltoallv__helper_loop_cat_paper"] = gsim("ag_slice", "loop_cat")
    sim["alltoallv__helper_view_col_public"] = gsim("ag_slice", "view_col")
    sim["alltoallv__harness_baseline"] = gsim("ag_rs", "loop_cat")
print(f"candidates measured: {len(runs)}  (percall records: {len(percall)})")
byp = defaultdict(list)
for k, vs in runs.items(): byp[k.split("__")[0]].append(k)

pooled_sim, pooled_meas = [], []
print(f"\n{'problem':15s} {'n_cands':>7s} {'rho(step)':>9s} {'top1':>5s} {'regret%':>8s}  sim-pick -> measured-best")
for p, ks in sorted(byp.items()):
    ks = [k for k in ks if sim.get(k) is not None]
    if len(ks) < 2: print(f"{p:15s} {len(ks):7d}   (too few scored candidates)"); continue
    m = {k: statistics.mean(runs[k]) for k in ks}; s_ = {k: sim[k] for k in ks}
    rho = spearman([s_[k] for k in ks], [m[k] for k in ks])
    sp = min(ks, key=lambda k: s_[k]); mb = min(ks, key=lambda k: m[k])
    reg = (m[sp] - m[mb]) / m[mb] * 100
    base = f"{p}__harness_baseline"
    if base in m and base in s_:
        for k in ks:
            pooled_sim.append(s_[base] / s_[k]); pooled_meas.append(m[base] / m[k])   # speedup ratios vs baseline
    print(f"{p:15s} {len(ks):7d} {rho if rho is None else round(rho,2):>9} {'Y' if sp==mb else 'n':>5s} {reg:8.1f}  {sp.split('__')[1][:28]} -> {mb.split('__')[1][:28]}")
print(f"\npooled Spearman on speedup-vs-baseline ratios (sim vs measured), n={len(pooled_sim)}: {spearman(pooled_sim, pooled_meas)}")
# sign agreement: does sim predict the direction (faster/slower than baseline) correctly?
pairs = [(a, b) for a, b in zip(pooled_sim, pooled_meas) if abs(a - 1) > 0.02]   # sim makes a call
agree = sum(1 for a, b in pairs if (a > 1) == (b > 1.02))
print(f"direction agreement where the sim predicts a change vs baseline: {agree}/{len(pairs)}  (ties excluded: {len(pooled_sim)-len(pairs)})")
print(f"\n{'candidate':58s} {'n':>2s} {'mean ms':>9s} {'sd':>7s} {'sim_def':>9s} {'percall ms':>10s}")
for p, ks in sorted(byp.items()):
    for k in sorted(ks, key=lambda k: statistics.mean(runs[k])):
        vs = runs[k]; sd = statistics.pstdev(vs) if len(vs) > 1 else float("nan")
        pc = statistics.mean(percall[k]) if percall.get(k) else float("nan")
        print(f"{k:58s} {len(vs):2d} {statistics.mean(vs):9.2f} {sd:7.2f} {str(sim.get(k))[:9]:>9s} {pc:10.3f}")
