#!/usr/bin/env python3
"""sweep_series.py — for every one-dimensional sweep series (step graphs), the compile-time dynamic range along the series
and the largest consecutive jumps (evidence of shape thresholds / non-monotone cost). Reads dataset.jsonl."""
import json, collections, math, os, sys
path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/work/ncc/data/dataset.jsonl")
rows = [json.loads(l) for l in open(path)]
sw = [r for r in rows if r["source"] == "sweep" and r["module_is_new"] and (r.get("bench_wall_s") or r.get("capture_compile_s"))]
series = collections.defaultdict(list)
for r in sw: series[(r["parent"].split("__")[0] + ":" + r["parent"].split("__")[1][:28], r["sweep_key"])].append((r["sweep_value"], r.get("bench_wall_s") or r.get("capture_compile_s"), "bench" if r.get("bench_wall_s") else "capture"))
print("| series | n | min s | max s | swing | largest consecutive jumps (value: s -> value: s) |\n|---|---|---|---|---|---|")
for k, v in sorted(series.items(), key=lambda kv: -max(t for _, t, _ in kv[1]) / min(t for _, t, _ in kv[1])):
    v.sort(); ts = [t for _, t, _ in v]; jumps = [((v[i + 1][1] / v[i][1]) if v[i][1] else 0, v[i], v[i + 1]) for i in range(len(v) - 1)]
    big = sorted([j for j in jumps if j[0] > 1.5 or j[0] < 1 / 1.5], key=lambda j: -abs(math.log(j[0])))
    print(f"| {k[0]}|{k[1]} | {len(v)} | {min(ts):.1f} | {max(ts):.1f} | {max(ts) / min(ts):.1f}× | " + "; ".join(f"{a[0]}: {a[1]:.0f} -> {b[0]}: {b[1]:.0f} ({r:.2f}×)" for r, a, b in big[:3]) + " |")
