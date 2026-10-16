#!/usr/bin/env python3
"""write_meta.py <outdir> <spec_json_str> <rc> <wall_s> [log_file] — write a capture meta.json for an externally run
program (e.g. a torchrun harness) from the modules found in <outdir>/cache."""
import glob, json, os, sys, re

def main():
    outdir, spec, rc, wall = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], float(sys.argv[4]); log = sys.argv[5] if len(sys.argv) > 5 else None
    mods = sorted(set(glob.glob(os.path.join(outdir, "cache", "neuronxcc-*", "MODULE_*"))))
    meta = dict(spec=spec, ok=(rc == "0" and len(mods) > 0), rc=rc, wall_s=wall, modules=[])
    for m in mods:
        pb = os.path.join(m, "model.hlo_module.pb"); nf = os.path.join(m, "model.neff"); cf = os.path.join(m, "compile_flags.json")
        meta["modules"].append(dict(dir=m, new=True, done=os.path.exists(os.path.join(m, "model.done")), hlo_pb_bytes=os.path.getsize(pb) if os.path.exists(pb) else 0,
                                    neff_bytes=os.path.getsize(nf) if os.path.exists(nf) else 0, compile_flags=open(cf).read() if os.path.exists(cf) else None))
    if log and os.path.exists(log):
        txt = open(log, errors="ignore").read()
        m = re.findall(r"[Ss]teady[^\n]{0,80}?([0-9.]+)\s*ms", txt); meta["steady_step_ms_from_log"] = [float(x) for x in m[:5]]
        if rc != "0": meta["error"] = txt[-600:]
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=1)
    print(f"[write_meta] {spec['id']} ok={meta['ok']} modules={len(mods)} rc={rc} wall={wall}")

if __name__ == "__main__":
    main()
