"""Generic taxonomy warm-cache RT probe (7 nodes / 224 ranks).

Env:
  PROBLEM       taxonomy problem name (must be in the full catalog)
  RUNTIME_FILE  a .py exposing exactly one evolved_* / *_fn function (the
                candidate code: baseline template, strat winner, or Sorcar)
  RT_WORLD      logical world_size to feed generate_test_case (default 224)
  RT_ITERS      timed warm iterations (default 20)
  RT_TAG        label for output (baseline|strat|sorcar)

Reuses the problem's own generate_test_case()/call_candidate() so the on-HW
inputs match the correctness/scoring path exactly. Reports cold (first
mark_step, includes compile/load) and warm (median of RT_ITERS) ms for rank 0.
"""
import os
import sys
import time
import json
import importlib.util

sys.path.insert(0, "/home/ubuntu/acc")
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist

import search.problems_all_catalogs  # register full catalog
from search.problems import get_problem

PROBLEM = os.environ["PROBLEM"]
RUNTIME_FILE = os.environ["RUNTIME_FILE"]
RT_WORLD = int(os.environ.get("RT_WORLD", "224"))
RT_ITERS = int(os.environ.get("RT_ITERS", "20"))
RT_TAG = os.environ.get("RT_TAG", "cand")

dist.init_process_group("xla")
rank = xr.global_ordinal()
world = xr.world_size()
device = xm.xla_device()
num_devices = max(world // 2, 1)
cpd = 2
num_nodes = max(world // 32, 1)

spec = importlib.util.spec_from_file_location("rt_cand", RUNTIME_FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = None
for nm in dir(mod):
    if nm.startswith("evolved_") or nm.endswith("_fn"):
        fn = getattr(mod, nm)
        break
if fn is None:
    if rank == 0:
        print(json.dumps({"problem": PROBLEM, "tag": RT_TAG,
                          "error": "no candidate fn in runtime file"}))
    sys.exit(2)

p = get_problem(PROBLEM)
tc = p.generate_test_case(RT_WORLD, "uniform", 16 if RT_WORLD > 32 else 32, seed=99)
lr = rank % RT_WORLD
rank_args = tc["per_rank_args"][lr]
shared_args = tc.get("shared_args", {})


def to_dev(a):
    if isinstance(a, torch.Tensor):
        return a.to(device)
    if isinstance(a, dict):
        return {k: to_dev(v) for k, v in a.items()}
    if isinstance(a, (list, tuple)):
        return type(a)(to_dev(v) for v in a)
    return a


rank_args = to_dev(rank_args)
shared_args = to_dev(shared_args)


def one_call():
    return p.call_candidate(fn, rank_args, shared_args,
                            rank, world, num_devices, cpd,
                            xm, torch, num_nodes=num_nodes)


def materialize(out):
    xm.mark_step()
    if isinstance(out, (list, tuple)):
        if out and isinstance(out[0], torch.Tensor):
            _ = out[0].cpu()
    elif isinstance(out, torch.Tensor):
        _ = out.cpu()


# cold: first execution (compile + load + run)
t0 = time.time()
out = one_call()
materialize(out)
dist.barrier()
cold_ms = (time.time() - t0) * 1000.0

# warm: timed iterations
times = []
for _ in range(RT_ITERS):
    t = time.time()
    out = one_call()
    materialize(out)
    times.append((time.time() - t) * 1000.0)
dist.barrier()

times.sort()
warm_med = times[len(times) // 2]
warm_min = times[0]

if rank == 0:
    rec = {"problem": PROBLEM, "tag": RT_TAG, "world": world,
           "cold_ms": round(cold_ms, 4), "warm_ms": round(warm_med, 4),
           "warm_min_ms": round(warm_min, 4)}
    print("RT_RESULT " + json.dumps(rec), flush=True)
    outdir = os.environ.get("RT_OUTDIR", "/home/ubuntu/rt_sweep")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, PROBLEM + "." + RT_TAG + ".json"), "w") as f:
        json.dump(rec, f)
