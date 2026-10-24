#!/usr/bin/env python3
"""Install HLO modules captured from a production NeuronX Distributed Inference compile cache (deepseek-coder-33b-instruct,
TP=32, continuous batching) into the corpus as programs of family nxd_inference, so build_manifest/ncc_bench treat them like
captured programs. usage: import_nxd.py <extracted cache dir containing neuronxcc-*/MODULE_*> <corpus dir> <tag>"""
import glob, json, os, shutil, sys
src, corpus, tag = sys.argv[1], sys.argv[2], sys.argv[3]; n = 0
for mdir in sorted(glob.glob(os.path.join(src, "neuronxcc-*", "MODULE_*"))):
    pb = os.path.join(mdir, "model.hlo_module.pb")
    if not os.path.exists(pb): continue
    mh = os.path.basename(mdir).split("+")[0].replace("MODULE_", ""); pid = f"{tag}__{mh[-12:]}"
    d = os.path.join(corpus, pid); cache_m = os.path.join(d, "cache", os.path.basename(os.path.dirname(mdir)), os.path.basename(mdir)); os.makedirs(cache_m, exist_ok=True)
    shutil.copy(pb, cache_m)
    flags = None
    if os.path.exists(os.path.join(mdir, "compile_flags.json")):
        shutil.copy(os.path.join(mdir, "compile_flags.json"), cache_m); flags = open(os.path.join(mdir, "compile_flags.json")).read()
    spec = {"id": pid, "family": "nxd_inference", "source": "real", "params": {"model": "deepseek-coder-33b-instruct", "tp": 32, "framework": "neuronx-distributed-inference", "batching": "continuous(ctx1,tkg8)", "module_hash": mh}}
    json.dump(spec, open(os.path.join(d, "spec.json"), "w"))
    json.dump(dict(spec=spec, ok=True, wall_s=0, imported=True, modules=[dict(dir=cache_m, new=True, done=True, hlo_pb_bytes=os.path.getsize(pb), neff_bytes=0, compile_flags=flags)]), open(os.path.join(d, "meta.json"), "w"), indent=1)
    n += 1
print(f"[import_nxd] {n} modules installed under {corpus}/{tag}__*")
