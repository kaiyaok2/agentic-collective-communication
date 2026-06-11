#!/usr/bin/env python3
"""7-node 224-rank training comparison for AllToAllV (variable-length).

  baseline: developer inline AG+T+RS (uniform packed layout). Matches
            make_baseline_alltoallv in train_olmoe10b.py line-for-line.
  agent:    runtime.trainium_alltoallv_7node.alltoallv (deployed
            Phase-5 wrapper).
"""
import argparse, os, sys, time, json, math, statistics
import time as _t

os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_RT_STOCHASTIC_ROUNDING_EN', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CAP = 4
DM = 256
LAYERS = 1


def _baseline_a2av(x, ws, mc):
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    reshaped = gathered.view(ws, ws, mc)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                             scale=1.0 / ws, scatter_dim=0, shard_count=ws)


def _load_agent_fn(ws):
    from runtime.trainium_alltoallv_7node import alltoallv, init_alltoallv
    init_alltoallv()
    return lambda x, mc: alltoallv(x, ws, mc)


class _A2AVLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mc, fn, ws):
        ctx.mc = mc
        ctx.fn = fn
        ctx.ws = ws
        xm.mark_step()
        out = fn(x, mc) if fn is not None else _baseline_a2av(x, ws, mc)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        gx = ctx.fn(g.contiguous(), ctx.mc) if ctx.fn is not None \
             else _baseline_a2av(g.contiguous(), ctx.ws, ctx.mc)
        xm.mark_step()
        return gx, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["baseline", "agent"], required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = xr.world_size()
    mc = CAP * DM

    if rank == 0:
        print(f"[init] alltoallv backend={args.backend} ws={ws} CAP={CAP} D={DM} LAYERS={LAYERS} steps={args.steps}", flush=True)

    fn = _load_agent_fn(ws) if args.backend == "agent" else None

    inputs = [torch.randn(ws * mc, device=dev, dtype=torch.bfloat16, requires_grad=True) * 0.01
              for _ in range(args.steps + args.warmup)]
    experts = [nn.Linear(DM, DM, bias=False, dtype=torch.bfloat16).to(dev)
               for _ in range(LAYERS)]
    all_params = []
    for e in experts:
        all_params.extend(list(e.parameters()))

    def step_once(x):
        h = x
        for L in range(LAYERS):
            h_out = _A2AVLayer.apply(h, mc, fn, ws)
            h_recv = h_out.view(ws, CAP, DM)
            h_proc = torch.matmul(h_recv, experts[L].weight.t())
            h = h_proc.view(-1)
        loss = (h ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for p in all_params:
                if p.grad is not None:
                    p.data -= 1e-4 * p.grad
                    p.grad = None
        return loss

    for s in range(args.warmup):
        _ = step_once(inputs[s]).item()
    if rank == 0: print("[warmup] done", flush=True)

    times = []
    t_total = time.time()
    for s in range(args.steps):
        xm.mark_step()
        t0 = time.time()
        _ = step_once(inputs[args.warmup + s]).item()
        times.append((time.time() - t0) * 1000.0)
        if rank == 0 and (s + 1) % 50 == 0:
            print(f"  step {s+1}/{args.steps}: median_ms={statistics.median(times[-50:]):.2f}", flush=True)

    wall = time.time() - t_total
    if rank == 0:
        steady = times[args.steps // 2:]
        print(f"  Problem     : alltoallv", flush=True)
        print(f"  Backend     : {args.backend}", flush=True)
        print(f"  Wall        : {wall:.2f} s", flush=True)
        print(f"  Avg step    : {statistics.mean(times):.2f} ms", flush=True)
        print(f"  Steady step : {statistics.median(steady):.2f} ms (median of last half)", flush=True)
        os.makedirs("/tmp/tp_search", exist_ok=True)
        with open(f"/tmp/tp_search/alltoallv_{args.backend}.json", "w") as f:
            json.dump({"backend": args.backend, "wall_s": wall,
                       "avg_ms": statistics.mean(times),
                       "steady_median_ms": statistics.median(steady),
                       "all_ms": times,
                       "ws": ws, "CAP": CAP, "DM": DM, "LAYERS": LAYERS}, f)


if __name__ == "__main__":
    main()
