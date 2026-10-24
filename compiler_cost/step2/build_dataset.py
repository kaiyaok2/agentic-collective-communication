#!/usr/bin/env python3
"""build_dataset.py — one row per HLO module: manifest metadata + Step-2 features + measured targets.

Targets (default flags, neuronx-cc 2.21, warm cache): bench_wall_s (median over repeats), bench_rss_gb (max), bench_status
(PASS/FAIL/OOM/TIMEOUT), neff_bytes, exec_total_time_us (neuron-profile), job/pass times; capture_compile_s = framework-side
compile time of the capture run (for step graphs of single-step programs) as a fallback target; framework_done/capture_ok
= compile failure labels from the capture. usage: build_dataset.py --data ~/work/ncc/data --out dataset.jsonl
"""
import argparse, glob, json, os, statistics, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features2

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.expanduser("~/work/ncc/data")); ap.add_argument("--out", default="dataset.jsonl")
    ap.add_argument("--pb-root", default=None, help="local root holding corpus/<prog>/cache/.../model.hlo_module.pb (default <data>/corpus_pb)")
    a = ap.parse_args(); pb_root = a.pb_root or os.path.join(a.data, "corpus_pb")
    rows = [json.loads(l) for l in open(os.path.join(a.data, "manifest_step1.jsonl")) if l.strip()]
    bench = collections.defaultdict(list)
    for f in glob.glob(os.path.join(a.data, "bench_*.jsonl")):
        if "bogus" in f: continue
        for l in open(f):
            if not l.strip(): continue
            r = json.loads(l); bench[(r["hlo_id"], r["flagset"], r["cc_tag"], bool(r["cold"]))].append(r)
    n_feat_err = 0; out = []
    for r in rows:
        pb_local = os.path.join(pb_root, r["hlo_pb"].replace("/home/ubuntu/", ""))
        try: feats = features2.features2_from_pb(pb_local) if os.path.exists(pb_local) else r.get("features", {})
        except Exception as e: feats = dict(r.get("features", {})); feats["feat2_err"] = str(e)[:100]; n_feat_err += 1
        feats = {k: v for k, v in feats.items() if k not in ("op_hist", "dtype_hist")}
        d = dict(hlo_id=r["hlo_id"], prog_id=r["prog_id"], family=r["family"], source=r["source"], params=r.get("params"), parent=r.get("parent"), mutation=r.get("mutation"),
                 sweep_key=r.get("sweep_key"), sweep_value=r.get("sweep_value"), module_is_new=bool(r.get("module_is_new")), n_modules_in_prog=r.get("n_modules_in_prog"),
                 framework_done=r.get("framework_done", True), capture_ok=r.get("capture_ok", True), framework_neff_bytes=r.get("framework_neff_bytes"))
        cap = r.get("capture") or {}; wct = cap.get("wrapper_compile_time_s") or []
        d["capture_compile_s"] = (wct[-1] if r.get("module_is_new") else wct[0]) if wct and r.get("n_modules_in_prog", 0) <= 2 else None
        d["exec_time_us_capture"] = statistics.median(cap["execute_time_us"]) if cap.get("execute_time_us") else None
        bs = bench.get((r["hlo_id"], "default", "2.21.33363.0", False), [])
        ok = [b for b in bs if b.get("status") == "PASS"]
        d["bench_n"] = len(bs); d["bench_status"] = (collections.Counter(b["status"] for b in bs).most_common(1)[0][0] if bs else None)
        d["bench_wall_s"] = statistics.median(b["wall_s"] for b in ok) if ok else (statistics.median(b["wall_s"] for b in bs) if bs else None)
        d["bench_user_s"] = statistics.median(b.get("user_s", 0) for b in ok) if ok else None
        d["bench_rss_gb"] = max(b.get("max_rss_kb", 0) for b in bs) / 1e6 if bs else None
        d["neff_bytes"] = max((b.get("neff_bytes") or 0) for b in ok) if ok else None
        prof = next((b.get("profile") for b in ok if isinstance(b.get("profile"), dict)), None)
        d["exec_total_time_us"] = (prof or {}).get("total_time")
        d["job_times_s"] = next((b.get("job_times_s") for b in ok if b.get("job_times_s")), None)
        d["pass_times_s"] = next((b.get("pass_times_s") for b in ok if b.get("pass_times_s")), None)
        d["note"] = next(((b.get("host") or {}).get("note") for b in bs), None)
        # other arms (for later): per (flagset, cc_tag) median wall
        arms = {}
        for (hid, fs, tag, cold), lst in bench.items():
            if hid != r["hlo_id"] or cold or (fs == "default" and tag == "2.21.33363.0"): continue
            okl = [b for b in lst if b["status"] == "PASS"]; arms[f"{fs}|{tag}"] = dict(status=collections.Counter(b["status"] for b in lst).most_common(1)[0][0], wall_s=statistics.median(b["wall_s"] for b in okl) if okl else None)
        d["arms"] = arms
        d["features"] = feats; out.append(d)
    with open(a.out, "w") as fo:
        for d in out: fo.write(json.dumps(d) + "\n")
    nb = sum(1 for d in out if d["bench_wall_s"] is not None); nc = sum(1 for d in out if d["capture_compile_s"])
    print(f"[dataset] {len(out)} modules -> {a.out}; bench target on {nb}, capture-time target on {nc}; feature errors {n_feat_err}; statuses {collections.Counter(d['bench_status'] for d in out)}")

if __name__ == "__main__":
    main()
