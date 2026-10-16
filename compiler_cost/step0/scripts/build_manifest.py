#!/usr/bin/env python3
"""build_manifest.py — scan the corpus directory (one subdir per captured program) and write a JSONL manifest with one
row per HLO module: hlo_id, hlo_pb path, program spec/source/family/params/parent/mutation, capture-time metrics
(wrapper compile time, execute time, step wall), the framework's compile flags, and structural features.
Usage: build_manifest.py --corpus /home/ubuntu/corpus --out /home/ubuntu/results/manifest.jsonl
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import features as F

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", default="/home/ubuntu/corpus"); ap.add_argument("--out", required=True); ap.add_argument("--no-features", action="store_true")
    a = ap.parse_args(); n = 0; n_prog = 0; n_fail = 0
    with open(a.out, "w") as fo:
        for d in sorted(glob.glob(os.path.join(a.corpus, "*"))):
            mp = os.path.join(d, "meta.json")
            if not os.path.exists(mp): continue
            m = json.load(open(mp)); spec = m.get("spec", {}); n_prog += 1
            if not m.get("ok"): n_fail += 1
            mods = [x for x in m.get("modules", []) if x.get("done") and x.get("hlo_pb_bytes", 0) > 0]
            txt = os.path.join(d, "step.hlo.txt")
            for x in mods:
                mdir = x["dir"]; pb = os.path.join(mdir, "model.hlo_module.pb"); mh = os.path.basename(mdir).split("+")[0].replace("MODULE_", "")
                row = dict(hlo_id=f"{spec.get('id', os.path.basename(d))}__M{mh[-10:]}", hlo_pb=pb, module_dir=mdir, module_hash=mh, prog_id=spec.get("id"), family=spec.get("family"), source=spec.get("source"),
                           params=spec.get("params"), parent=spec.get("parent"), mutation=spec.get("mutation"), n_modules_in_prog=len(mods), module_is_new=x.get("new"),
                           framework_compile_flags=x.get("compile_flags"), framework_neff_bytes=x.get("neff_bytes"), capture=dict(
                               wrapper_compile_time_s=m.get("wrapper_compile_time_s"), first_step_wall_s=m.get("first_step_wall_s"), steady_step_wall_s=m.get("steady_step_wall_s"),
                               execute_time_us=m.get("execute_time_us"), n_params=m.get("n_params"), capture_wall_s=m.get("wall_s")))
                if not a.no_features: row["features"] = F.features(hlo_pb=pb, hlo_txt=txt if len(mods) == 1 else None)
                fo.write(json.dumps(row) + "\n"); n += 1
    print(f"[manifest] programs={n_prog} (failed {n_fail}) modules={n} -> {a.out}  HAVE_PB={F.HAVE_PB}")

if __name__ == "__main__":
    main()
