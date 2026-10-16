#!/usr/bin/env python3
"""select_subset.py — pick a family-balanced subset of HLO modules from a bench result file for the flag / version /
cold-cache arms. Keeps PASS modules whose default compile wall lies in [lo, hi] seconds, samples up to k per family.
Usage: select_subset.py --bench results/bench_default.jsonl --manifest results/manifest.jsonl --out results/manifest_subset.jsonl --k 6 --lo 2 --hi 400
"""
import argparse, json, random, collections

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bench", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--lo", type=float, default=2.0); ap.add_argument("--hi", type=float, default=400.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-total", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    wall = {}
    import os
    if os.path.exists(a.bench):
        for l in open(a.bench):
            r = json.loads(l)
            if r.get("status") == "PASS" and r.get("repeat") == 0: wall[r["hlo_id"]] = r["wall_s"]
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if not wall:   # no bench yet: use the framework-wrapper compile time recorded at capture (step graphs only)
        for r in rows:
            c = (r.get("capture") or {}).get("wrapper_compile_time_s") or []
            if r.get("module_is_new") and c: wall[r["hlo_id"]] = max(c)
        print(f"[subset] using capture-time compile times for {len(wall)} modules")
    byfam = collections.defaultdict(list)
    for r in rows:
        w = wall.get(r["hlo_id"])
        if w is not None and a.lo <= w <= a.hi and r.get("module_is_new", True): byfam[r["family"]].append((w, r))
    out = []
    for fam, lst in sorted(byfam.items()):
        lst.sort(key=lambda t: t[0])
        # spread across the wall-time range: take evenly spaced quantiles
        if len(lst) <= a.k: pick = lst
        else: pick = [lst[int(i * (len(lst) - 1) / (a.k - 1))] for i in range(a.k)]
        out += [r for _, r in pick]
    if a.max_total and len(out) > a.max_total: out = rng.sample(out, a.max_total)
    with open(a.out, "w") as fo:
        for r in out: fo.write(json.dumps(r) + "\n")
    print(f"[subset] {len(out)} modules from {len(byfam)} families -> {a.out}; est default wall sum {sum(wall[r['hlo_id']] for r in out):.0f}s")

if __name__ == "__main__":
    main()
