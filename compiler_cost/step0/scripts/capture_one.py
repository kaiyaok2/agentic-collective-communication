#!/usr/bin/env python3
"""capture_one.py — run one program spec on the XLA device with a private Neuron compile cache, so that every HLO module
the framework sends to neuronx-cc lands in <cache>/neuronxcc-*/MODULE_*/model.hlo_module.pb. Records wrapper compile time,
first-step wall, steady-step wall, XLA ExecuteTime samples and the HLO text of the step graph. Writes <outdir>/meta.json.
Usage: capture_one.py <spec.json> <outdir>   (env NEURON_CC_FLAGS must already point --cache_dir at <outdir>/cache)
"""
import json, os, sys, time, glob, traceback
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")   # one core per capture driver; the second driver sets 1

def main():
    spec = json.load(open(sys.argv[1])); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
    meta = dict(spec=spec, ok=False, t0=time.time())
    try:
        import torch, torch_xla, torch_xla.core.xla_model as xm, torch_xla.debug.metrics as met
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import programs
        dev = xm.xla_device(); torch.manual_seed(0)
        step, keep = programs.build(spec, dev)
        xm.mark_step(); xm.wait_device_ops(); met.clear_all()   # materialise params/inputs; don't count their transfer graph
        cache_before = set(glob.glob(os.path.join(outdir, "cache", "neuronxcc-*", "MODULE_*")))
        t = time.time(); outs = step()
        try:
            hlo_txt = torch_xla._XLAC._get_xla_tensors_hlo([o for o in outs if isinstance(o, torch.Tensor)] + [k for k in keep if isinstance(k, torch.Tensor)])
            open(os.path.join(outdir, "step.hlo.txt"), "w").write(hlo_txt); meta["hlo_text_bytes"] = len(hlo_txt)
        except Exception as e: meta["hlo_text_err"] = str(e)[:200]
        xm.mark_step(); xm.wait_device_ops(); meta["first_step_wall_s"] = round(time.time() - t, 3)
        ct = met.metric_data("CompileTime"); meta["wrapper_compile_time_s"] = [round(s[1] / 1e9, 3) for s in (ct[2] if ct else [])]
        n_steps = int(spec.get("n_steps", 5)); walls = []
        for i in range(n_steps):
            t = time.time(); outs = step(); xm.mark_step(); xm.wait_device_ops(); walls.append(round(time.time() - t, 4))
        meta["steady_step_wall_s"] = walls
        et = met.metric_data("ExecuteTime"); meta["execute_time_us"] = [round(s[1] / 1e3, 1) for s in (et[2] if et else [])]
        meta["compile_count"] = int(met.counter_value("CachedCompile") or 0) if hasattr(met, "counter_value") else None
        mods = sorted(set(glob.glob(os.path.join(outdir, "cache", "neuronxcc-*", "MODULE_*"))))
        meta["modules"] = [dict(dir=m, new=(m not in cache_before), done=os.path.exists(os.path.join(m, "model.done")),
                                hlo_pb_bytes=os.path.getsize(os.path.join(m, "model.hlo_module.pb")) if os.path.exists(os.path.join(m, "model.hlo_module.pb")) else 0,
                                neff_bytes=os.path.getsize(os.path.join(m, "model.neff")) if os.path.exists(os.path.join(m, "model.neff")) else 0,
                                compile_flags=(open(os.path.join(m, "compile_flags.json")).read() if os.path.exists(os.path.join(m, "compile_flags.json")) else None)) for m in mods]
        meta["n_params"] = int(sum(k.numel() for k in keep if hasattr(k, "numel"))); meta["ok"] = True
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {str(e)[:600]}"; meta["traceback"] = traceback.format_exc()[-2000:]
    meta["wall_s"] = round(time.time() - meta["t0"], 2)
    if os.environ.get("RANK", "0") != "0":   # torchrun: only rank 0 writes the meta (concurrent writers corrupted the file)
        print(f"[capture_one] rank {os.environ.get('RANK')} done ok={meta['ok']}", flush=True); return
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=1)
    print(f"[capture_one] {spec['id']} ok={meta['ok']} modules={len(meta.get('modules', []))} first={meta.get('first_step_wall_s')} err={meta.get('error', '')[:120]}", flush=True)

if __name__ == "__main__":
    main()
