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
    ap.add_argument("--extra-cc-flags", default="", help="appended to NEURON_CC_FLAGS at capture time"); ap.add_argument("--reverse", action="store_true", help="process specs in reverse order (second driver on the other core)"); ap.add_argument("--min-free-gb", type=float, default=9.0); ap.add_argument("--oom-kill-gb", type=float, default=1.5, help="kill the running capture when MemAvailable drops below this")
    a = ap.parse_args()
    specs = []
    for g in a.grids.split(","): specs += programs.GRIDS[g]()
    if a.filter: specs = [s for s in specs if re.search(a.filter, s["id"])]
    if a.limit: specs = specs[: a.limit]
    if a.reverse: specs = specs[::-1]
    print(f"[capture] {len(specs)} specs", flush=True); here = os.path.dirname(os.path.abspath(__file__))
    def mem_avail_gb():
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable"): return int(line.split()[1]) / 1e6
        except Exception: return 99.0
        return 99.0
    for i, s in enumerate(specs):
        d = os.path.join(a.corpus, s["id"]); os.makedirs(d, exist_ok=True)
        waited = 0
        while mem_avail_gb() < a.min_free_gb and waited < 1800: time.sleep(15); waited += 15   # memory guard: don't start while another compile holds the RAM
        if os.path.exists(os.path.join(d, "meta.json")) or os.path.exists(os.path.join(d, "spec.json")):
            continue   # done, or claimed by the other driver
        json.dump(s, open(os.path.join(d, "spec.json"), "w"))
        env = dict(os.environ); env["NEURON_CC_FLAGS"] = f"--cache_dir={d}/cache {a.extra_cc_flags}".strip(); env["NEURON_COMPILE_CACHE_URL"] = f"{d}/cache"
        env.setdefault("XLA_IR_DEBUG", "0"); env.setdefault("PJRT_DEVICE", "NEURON")
        if a.nproc > 1: cmd = ["torchrun", "--nproc_per_node", str(a.nproc), os.path.join(here, "capture_one.py"), os.path.join(d, "spec.json"), d]
        else: cmd = [a.python, os.path.join(here, "capture_one.py"), os.path.join(d, "spec.json"), d]
        t0 = time.time()
        with open(os.path.join(d, "capture.log"), "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            import signal; rc = None
            while True:   # poll with a host-memory watchdog: a compile that exhausts RAM is killed (OOM) before the node thrashes
                rc = proc.poll()
                if rc is not None: break
                if time.time() - t0 > a.timeout: os.killpg(proc.pid, signal.SIGKILL); proc.wait(); rc = "TIMEOUT"; break
                if mem_avail_gb() < a.oom_kill_gb: os.killpg(proc.pid, signal.SIGKILL); proc.wait(); rc = "OOM"; break
                time.sleep(1)
        if not os.path.exists(os.path.join(d, "meta.json")):   # timeout / crash: keep the protos already written to the cache (labelled timeouts)
            import glob as _g
            mods = sorted(set(_g.glob(os.path.join(d, "cache", "neuronxcc-*", "MODULE_*"))))
            json.dump(dict(spec=s, ok=False, error=f"subprocess rc={rc}", wall_s=round(time.time() - t0, 1), timeout_s=a.timeout if rc == "TIMEOUT" else None, oom_killed=(rc == "OOM"),
                           modules=[dict(dir=m, new=True, done=os.path.exists(os.path.join(m, "model.done")),
                                         hlo_pb_bytes=os.path.getsize(os.path.join(m, "model.hlo_module.pb")) if os.path.exists(os.path.join(m, "model.hlo_module.pb")) else 0,
                                         neff_bytes=0, compile_flags=None) for m in mods]), open(os.path.join(d, "meta.json"), "w"))
        m = json.load(open(os.path.join(d, "meta.json")))
        print(f"[capture] {i+1}/{len(specs)} {s['id'][:70]:70s} ok={m.get('ok')} mods={len(m.get('modules', []))} wall={round(time.time()-t0,1)}s {str(m.get('error',''))[:80]}", flush=True)

if __name__ == "__main__":
    main()
