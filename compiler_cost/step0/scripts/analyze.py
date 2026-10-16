#!/usr/bin/env python3
"""analyze.py — Step-0 premise analysis of the compile-cost corpus.

Inputs (pulled from the node): manifest.jsonl, bench_default.jsonl, bench_cold.jsonl, bench_flags.jsonl, bench_versions.jsonl
Outputs: a markdown report (stdout) with
  1. compile-cost distribution per source/family (wall, RSS, NEFF size), failures/timeouts/OOM with codes
  2. repeat noise (CV across warm repeats), cold vs warm ratio, bench vs framework-wrapper compile time
  3. flag and compiler-version effects (ratio to default / to installed version), including status changes
  4. mutation pairs: compile-time ratio child/parent vs structural deltas
  5. cheap-feature baselines for log(wall), log(RSS), PASS/FAIL: ridge + gradient boosting, random CV and leave-one-family-out
  6. execution latency (neuron-profile) vs framework ExecuteTime
"""
import argparse, json, math, os, collections, statistics as st
import numpy as np

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")

def fmt(x, d=1): return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"

FEATS = ["n_instructions", "n_computations", "n_entry_instructions", "n_params", "param_bytes", "output_bytes", "total_tensor_bytes", "largest_tensor_bytes", "max_rank",
         "n_distinct_shapes", "dot_flops", "conv_flops", "elementwise_bytes", "n_dot", "n_conv", "n_reduce", "n_while", "n_custom_call", "n_collective", "n_gather", "n_scatter",
         "n_sort", "n_dynamic_slice", "n_transpose", "n_reshape", "n_broadcast", "n_concat", "n_slice", "n_pad", "n_iota", "n_rng", "n_dtypes", "frac_bf16", "frac_f32", "proto_bytes"]

def featvec(f):
    v = []
    for k in FEATS:
        x = f.get(k, 0) or 0
        v.append(math.log1p(x) if ("bytes" in k or "flops" in k or k.startswith("n_")) else x)
    return v

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args(); D = a.dir
    man = {r["hlo_id"]: r for r in load(f"{D}/manifest.jsonl")}; bd = load(f"{D}/bench_default.jsonl"); bc = load(f"{D}/bench_cold.jsonl"); bf = load(f"{D}/bench_flags.jsonl"); bv = load(f"{D}/bench_versions.jsonl")
    print(f"# Step 0 — compile-cost premise check\n\nmanifest modules: {len(man)}; bench rows: default {len(bd)}, cold {len(bc)}, flags {len(bf)}, versions {len(bv)}\n")
    # ---------------------------------------------------------------- 1. distribution
    r0 = [r for r in bd if r["repeat"] == 0]
    byfam = collections.defaultdict(list)
    for r in r0: byfam[(man.get(r["hlo_id"], {}).get("source", "?"), r.get("family", "?"))].append(r)
    print("## 1. Compile cost by family (default flags, installed compiler, repeat 0)\n")
    print("| source | family | n | PASS | FAIL | TIMEOUT | OOM | wall median s | p90 | max | RSS median GB | RSS max GB | NEFF median MB | NEFF max MB |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (src, fam), rs in sorted(byfam.items()):
        stc = collections.Counter(r["status"] for r in rs); w = [r["wall_s"] for r in rs if r["status"] == "PASS"]; m = [r.get("max_rss_kb", 0) / 1e6 for r in rs]; nf = [r["neff_bytes"] / 1e6 for r in rs if r["status"] == "PASS"]
        print(f"| {src} | {fam} | {len(rs)} | {stc['PASS']} | {stc['FAIL']} | {stc['TIMEOUT']} | {stc['OOM']} | {fmt(st.median(w)) if w else '-'} | {fmt(q(w,.9)) if w else '-'} | {fmt(max(w)) if w else '-'} | {fmt(st.median(m),2) if m else '-'} | {fmt(max(m),2) if m else '-'} | {fmt(st.median(nf),2) if nf else '-'} | {fmt(max(nf),2) if nf else '-'} |")
    allw = [r["wall_s"] for r in r0 if r["status"] == "PASS"]
    if allw:
        print(f"\nAll PASS modules: n={len(allw)}, wall min {min(allw):.1f} s, median {st.median(allw):.1f}, p90 {q(allw,.9):.1f}, p99 {q(allw,.99):.1f}, max {max(allw):.1f}; "
              f"total compile time {sum(allw)/60:.1f} min; share > 60 s: {sum(w>60 for w in allw)/len(allw):.1%}; > 300 s: {sum(w>300 for w in allw)/len(allw):.1%}; "
              f"max/min ratio {max(allw)/min(allw):.0f}×")
    # 1b. framework-side compile time recorded at capture, all step graphs (one compile each, under capture-time load)
    cap = collections.defaultdict(list)
    for m in man.values():
        c = (m.get("capture") or {}).get("wrapper_compile_time_s") or []
        if m.get("module_is_new") and c: cap[(m.get("source", "?"), m.get("family", "?"))].append(max(c))
    if cap:
        print("\n### 1b. Framework-side compile time at capture (every step graph compiled once by the torch-xla wrapper)\n")
        print("| source | family | n | min s | median | p90 | max |\n|---|---|---|---|---|---|---|")
        for (src, fam), v in sorted(cap.items()): print(f"| {src} | {fam} | {len(v)} | {min(v):.1f} | {st.median(v):.1f} | {q(v,.9):.1f} | {max(v):.1f} |")
        allc = [x for v in cap.values() for x in v]
        print(f"\nAll step graphs n={len(allc)}: median {st.median(allc):.1f} s, p90 {q(allc,.9):.1f}, max {max(allc):.1f}; > 60 s: {sum(x>60 for x in allc)/len(allc):.0%}; > 120 s: {sum(x>120 for x in allc)/len(allc):.0%}; max/min {max(allc)/min(allc):.0f}×")
    fails = [r for r in r0 if r["status"] != "PASS"]
    if fails:
        print(f"\n### Failures ({len(fails)})\n\n| hlo_id | status | code | wall s | RSS GB | log tail |\n|---|---|---|---|---|---|")
        for r in fails[:40]: print(f"| {r['hlo_id'][:60]} | {r['status']} | {r.get('ncc_code','')} | {fmt(r.get('wall_s'))} | {fmt(r.get('max_rss_kb',0)/1e6,2)} | {r.get('error_line', r.get('log_tail',''))[:140].replace('|','/')} |")
    # ---------------------------------------------------------------- 2. noise
    print("\n## 2. Noise: warm repeats, cold cache, bench vs framework wrapper\n")
    reps = collections.defaultdict(list)
    for r in bd:
        if r["status"] == "PASS": reps[r["hlo_id"]].append(r["wall_s"])
    cvs = [(st.pstdev(v) / st.mean(v), max(v) / min(v), st.mean(v)) for v in reps.values() if len(v) >= 2]
    if cvs:
        print(f"- warm repeats: {len(cvs)} modules with ≥2 repeats; CV median {st.median([c for c,_,_ in cvs]):.3f}, p90 {q([c for c,_,_ in cvs],.9):.3f}, max {max(c for c,_,_ in cvs):.3f}; max/min ratio median {st.median([m for _,m,_ in cvs]):.3f}, max {max(m for _,m,_ in cvs):.2f}")
        big = [(c, m, w) for c, m, w in cvs if w > 10]; small = [(c, m, w) for c, m, w in cvs if w <= 10]
        if big and small: print(f"  - CV for compiles > 10 s: median {st.median([c for c,_,_ in big]):.3f} (n={len(big)}); ≤ 10 s: {st.median([c for c,_,_ in small]):.3f} (n={len(small)})")
    warm0 = {r["hlo_id"]: r["wall_s"] for r in r0 if r["status"] == "PASS"}
    cold = collections.defaultdict(list)
    for r in bc:
        if r["status"] == "PASS": cold[r["hlo_id"]].append(r["wall_s"])
    ratios = [(st.mean(v) / warm0[k], warm0[k]) for k, v in cold.items() if k in warm0]
    if ratios:
        print(f"- cold page cache vs warm repeat 0: {len(ratios)} modules; ratio median {st.median([x for x,_ in ratios]):.3f}, p90 {q([x for x,_ in ratios],.9):.3f}, max {max(x for x,_ in ratios):.3f}; absolute extra median {st.median([ (x-1)*w for x,w in ratios]):.2f} s")
    wr = []
    for r in r0:
        c = man.get(r["hlo_id"], {}).get("capture", {}); wct = c.get("wrapper_compile_time_s") or []
        if r["status"] == "PASS" and len(wct) == 1 and man[r["hlo_id"]].get("n_modules_in_prog", 0) <= 2 and man[r["hlo_id"]].get("module_is_new"): wr.append((wct[0] / r["wall_s"], r["wall_s"]))
    if wr: print(f"- framework-wrapper CompileTime vs standalone CLI wall (single-step-graph programs, n={len(wr)}): ratio median {st.median([x for x,_ in wr]):.3f}, p10 {q([x for x,_ in wr],.1):.3f}, p90 {q([x for x,_ in wr],.9):.3f}")
    # ---------------------------------------------------------------- 3. flags & versions
    def arm_table(rows, keyf, title, base):
        print(f"\n## {title}\n"); groups = collections.defaultdict(list)
        for r in rows: groups[keyf(r)].append(r)
        print("| arm | n | PASS | FAIL/TIMEOUT/OOM | wall ratio to default: median | p10 | p90 | max | RSS ratio median | NEFF ratio median | status changes |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for k, rs in sorted(groups.items()):
            rat = []; rss = []; nf = []; ch = 0; stc = collections.Counter(r["status"] for r in rs)
            for r in rs:
                b = base.get(r["hlo_id"])
                if not b: continue
                if (r["status"] == "PASS") != (b["status"] == "PASS"): ch += 1
                if r["status"] == "PASS" and b["status"] == "PASS":
                    rat.append(r["wall_s"] / b["wall_s"]); rss.append(r.get("max_rss_kb", 1) / max(1, b.get("max_rss_kb", 1))); nf.append(r["neff_bytes"] / max(1, b["neff_bytes"]))
            print(f"| {k} | {len(rs)} | {stc['PASS']} | {stc['FAIL']}/{stc['TIMEOUT']}/{stc['OOM']} | {fmt(st.median(rat),3) if rat else '-'} | {fmt(q(rat,.1),3) if rat else '-'} | {fmt(q(rat,.9),3) if rat else '-'} | {fmt(max(rat),2) if rat else '-'} | {fmt(st.median(rss),3) if rss else '-'} | {fmt(st.median(nf),3) if nf else '-'} | {ch} |")
    base = {r["hlo_id"]: r for r in r0}
    if bf: arm_table(bf, lambda r: r["flagset"], "3a. Compiler flags (subset; ratio to default flags, same module)", base)
    if bv: arm_table(bv, lambda r: r["cc_tag"], "3b. Compiler versions (subset; ratio to installed 2.21, default flags)", base)
    # ---------------------------------------------------------------- 4. mutation pairs
    print("\n## 4. Near-identical pairs (one parameter changed)\n")
    prog_wall = collections.defaultdict(list)
    for r in r0:
        m = man.get(r["hlo_id"], {})
        if r["status"] == "PASS" and m.get("module_is_new"): prog_wall[m["prog_id"]].append(r["wall_s"])
    pw = {k: max(v) for k, v in prog_wall.items()}   # step graph = the largest new module of the program
    n_bench = len(pw)
    for m in man.values():   # programs without bench rows yet: use the capture-time compile of the step graph
        c = (m.get("capture") or {}).get("wrapper_compile_time_s") or []
        if m.get("module_is_new") and c and m["prog_id"] not in pw: pw[m["prog_id"]] = max(c)
    print(f"(compile times: {n_bench} programs from the bench, {len(pw)-n_bench} from capture-time framework compiles)\n")
    pairs = []
    for hid, m in man.items():
        if m.get("source") == "mutation" and m.get("parent") in pw and m["prog_id"] in pw:
            pairs.append((m["parent"], m["mutation"], pw[m["prog_id"]] / pw[m["parent"]], pw[m["parent"]], pw[m["prog_id"]]))
    seen = set(); pairs = [p for p in pairs if not (p[:2] in seen or seen.add(p[:2]))]
    if pairs:
        print("| parent | mutation | parent wall s | child wall s | ratio |\n|---|---|---|---|---|")
        for par, mut, rat, pwl, cwl in sorted(pairs, key=lambda t: -abs(math.log(t[2]))): print(f"| {par[:55]} | {mut} | {pwl:.1f} | {cwl:.1f} | {rat:.2f} |")
        rs = [p[2] for p in pairs]; print(f"\n{len(pairs)} pairs; |log ratio| median {st.median([abs(math.log(x)) for x in rs]):.3f}; pairs with >20% change: {sum(abs(math.log(x))>math.log(1.2) for x in rs)}; >2×: {sum(abs(math.log(x))>math.log(2) for x in rs)}")
        pm = [p for p in pairs if p[1].endswith("plus1") or p[1].endswith("minus1")]
        if pm: print(f"±1 dimension changes: {len(pm)} pairs, ratio median {st.median([p[2] for p in pm]):.3f}, max {max(p[2] for p in pm):.2f}, min {min(p[2] for p in pm):.2f}")
    # ---------------------------------------------------------------- 5. cheap-feature baselines
    print("\n## 5. Cheap-feature baselines (default flags, repeat 0)\n")
    X = []; yw = []; yr = []; yp = []; fams = []
    for r in r0:
        m = man.get(r["hlo_id"]); f = (m or {}).get("features") or {}
        if not m or "n_instructions" not in f: continue
        X.append(featvec(f)); yw.append(math.log(r["wall_s"]) if r["status"] == "PASS" else float("nan")); yr.append(math.log(max(1, r.get("max_rss_kb", 1)))); yp.append(1 if r["status"] == "PASS" else 0); fams.append(m["family"])
    X = np.array(X); yw = np.array(yw); yr = np.array(yr); yp = np.array(yp); fams = np.array(fams)
    try:
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.model_selection import KFold
        from scipy.stats import spearmanr
        def cv_eval(model_fn, Xa, ya, groups=None, k=5):
            pred = np.full(len(ya), np.nan)
            if groups is None:
                for tr, te in KFold(k, shuffle=True, random_state=0).split(Xa):
                    mdl = model_fn(); mdl.fit(Xa[tr], ya[tr]); pred[te] = mdl.predict(Xa[te])
            else:
                for g in np.unique(groups):
                    tr = groups != g; te = groups == g
                    if tr.sum() < 5: continue
                    mdl = model_fn(); mdl.fit(Xa[tr], ya[tr]); pred[te] = mdl.predict(Xa[te])
            ok = ~np.isnan(pred); e = pred[ok] - ya[ok]
            return dict(n=int(ok.sum()), r2=1 - np.sum(e**2) / np.sum((ya[ok] - ya[ok].mean())**2), mape=float(np.mean(np.abs(np.exp(e) - 1))), medape=float(np.median(np.abs(np.exp(e) - 1))),
                        spearman=float(spearmanr(pred[ok], ya[ok]).correlation), within2x=float(np.mean(np.abs(e) < math.log(2))))
        ok = ~np.isnan(yw); Xw, ywv, fw = X[ok], yw[ok], fams[ok]
        print(f"n={len(ywv)} PASS modules with features; targets: log(compile wall s), log(peak RSS)\n")
        print("| target | model | CV | n | R² | MAPE | median APE | Spearman | within 2× |\n|---|---|---|---|---|---|---|---|---|")
        for tname, Xa, ya, ga in (("wall", Xw, ywv, fw), ("rss", X, yr, fams)):
            for mname, mf in (("ridge", lambda: Ridge(alpha=1.0)), ("gbdt", lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0))):
                for cvn, g in (("random 5-fold", None), ("leave-one-family-out", ga)):
                    e = cv_eval(mf, Xa, ya, g); print(f"| {tname} | {mname} | {cvn} | {e['n']} | {e['r2']:.3f} | {e['mape']:.2f} | {e['medape']:.2f} | {e['spearman']:.3f} | {e['within2x']:.2f} |")
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(Xw, ywv)
        imp = sorted(zip(gb.feature_importances_, FEATS), reverse=True)[:12]; print("\nGBDT top features for log(wall): " + ", ".join(f"{n} {v:.2f}" for v, n in imp))
        # single-feature baselines
        for k in ("n_instructions", "proto_bytes", "total_tensor_bytes", "dot_flops"):
            i = FEATS.index(k); c = spearmanr(Xw[:, i], ywv).correlation; print(f"- Spearman(log wall, {k}) = {c:.3f}")
        if yp.min() != yp.max():
            from sklearn.model_selection import cross_val_predict
            pr = cross_val_predict(GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0), X, yp, cv=5, method="predict_proba")[:, 1]
            from sklearn.metrics import roc_auc_score; print(f"- PASS/FAIL classifier (GBDT, 5-fold): AUC {roc_auc_score(yp, pr):.3f}; failures {int((yp==0).sum())}/{len(yp)}")
    except Exception as e:
        print(f"(sklearn analysis skipped: {e})")
    # ---------------------------------------------------------------- 6. latency
    print("\n## 6. Execution latency: neuron-profile vs framework ExecuteTime\n")
    lat = []
    for r in r0:
        p = r.get("profile") or {}; m = man.get(r["hlo_id"], {}); et = (m.get("capture") or {}).get("execute_time_us") or []
        tt = p.get("total_time") or p.get("total_time_us") or p.get("neuroncore_cycle_count")
        if tt and et and m.get("module_is_new") and m.get("n_modules_in_prog", 0) <= 2: lat.append((float(tt), st.median(et), r["hlo_id"]))
    if lat:
        from scipy.stats import spearmanr
        print(f"n={len(lat)} single-step-graph modules; Spearman(profile total_time, median ExecuteTime) = {spearmanr([a for a,_,_ in lat],[b for _,b,_ in lat]).correlation:.3f}")
        keys = sorted({k for r in r0 for k in (r.get('profile') or {})})[:60]; print(f"profile summary keys available ({len(keys)} shown): {', '.join(keys)}")
    else:
        keys = sorted({k for r in r0 for k in (r.get('profile') or {})}); print(f"no paired latency rows; profile keys: {', '.join(keys[:40])}")

if __name__ == "__main__":
    main()
