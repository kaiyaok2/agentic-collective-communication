#!/usr/bin/env python3
"""capture.py — drive capture_one.py over a grid of program specs, one subprocess + private cache per spec.
Usage: capture.py --grids synth,real,mutation --corpus /home/ubuntu/corpus [--filter regex] [--timeout 1800] [--nproc 2 for collective]
Resumable: specs with an existing meta.json are skipped.
"""
import argparse, json, os, subprocess, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import programs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grids", default="synth"); ap.add_argument("--corpus", default="/home/ubuntu/corpus")
    ap.add_argument("--filter", default=None); ap.add_argument("--timeout", type=int, default=2400); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--python", default=sys.executable); ap.add_argument("--nproc", type=int, default=1, help=">1 runs under torchrun (collective grid)")
    ap.add_argument("--extra-cc-flags", default="", help="appended to NEURON_CC_FLAGS at capture time")
    a = ap.parse_args()
    specs = []
    for g in a.grids.split(","): specs += programs.GRIDS[g]()
    if a.filter: specs = [s for s in specs if re.search(a.filter, s["id"])]
    if a.limit: specs = specs[: a.limit]
    print(f"[capture] {len(specs)} specs", flush=True); here = os.path.dirname(os.path.abspath(__file__))
    for i, s in enumerate(specs):
        d = os.path.join(a.corpus, s["id"]); os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "meta.json")):
            continue
        json.dump(s, open(os.path.join(d, "spec.json"), "w"))
        env = dict(os.environ); env["NEURON_CC_FLAGS"] = f"--cache_dir={d}/cache {a.extra_cc_flags}".strip(); env["NEURON_COMPILE_CACHE_URL"] = f"{d}/cache"
        env.setdefault("XLA_IR_DEBUG", "0"); env.setdefault("PJRT_DEVICE", "NEURON")
        if a.nproc > 1: cmd = ["torchrun", "--nproc_per_node", str(a.nproc), os.path.join(here, "capture_one.py"), os.path.join(d, "spec.json"), d]
        else: cmd = [a.python, os.path.join(here, "capture_one.py"), os.path.join(d, "spec.json"), d]
        t0 = time.time()
        with open(os.path.join(d, "capture.log"), "w") as lf:
            try: rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=a.timeout).returncode
            except subprocess.TimeoutExpired: rc = "TIMEOUT"
        if not os.path.exists(os.path.join(d, "meta.json")):
            json.dump(dict(spec=s, ok=False, error=f"subprocess rc={rc}", wall_s=round(time.time() - t0, 1)), open(os.path.join(d, "meta.json"), "w"))
        m = json.load(open(os.path.join(d, "meta.json")))
        print(f"[capture] {i+1}/{len(specs)} {s['id'][:70]:70s} ok={m.get('ok')} mods={len(m.get('modules', []))} wall={round(time.time()-t0,1)}s {str(m.get('error',''))[:80]}", flush=True)
        subprocess.run("pkill -9 -f capture_one.py >/dev/null 2>&1; true", shell=True)

if __name__ == "__main__":
    main()
