#!/usr/bin/env python3
"""
Microbenchmark: agent-evolved AllToAllV vs AG+ReduceScatter on Trainium.

Measures per-call latency of each AllToAllV implementation in a realistic
MoE token-dispatch pattern. Both forward and backward (reverse) calls.

    torchrun --nproc_per_node=32 --nnodes=2 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
        bench_alltoallv.py --iters 100
"""

import argparse
import os
import time
import json

os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


DM = 128
NEXP = 64
TOPK = 2
NTOK = 128   # tokens per rank


def _evolved(x, sc, rc, mc):
    rank = xr.global_ordinal()
    ws = xr.world_size()
    ps = ws * mc
    packed = torch.zeros(ps, device=x.device, dtype=x.dtype)
    off = 0
    for i in range(ws):
        s = sc[i]
        if s > 0:
            packed[i * mc:i * mc + s] = x[off:off + s]
        off += s
    gathered = xm.all_gather(packed.unsqueeze(0), dim=0).view(-1)
    idx = []
    for src in range(ws):
        c = rc[src]
        if c > 0:
            base = src * ps + rank * mc
            idx.extend(range(base, base + c))
    if idx:
        return torch.index_select(
            gathered, 0, torch.tensor(idx, device=x.device, dtype=torch.long))
    return torch.zeros(0, device=x.device, dtype=x.dtype)


def _agrs(x, sc, rc, mc):
    rank = xr.global_ordinal()
    ws = xr.world_size()
    ps = ws * mc
    packed = torch.zeros(ps, device=x.device, dtype=x.dtype)
    off = 0
    for i in range(ws):
        s = sc[i]
        if s > 0:
            packed[i * mc:i * mc + s] = x[off:off + s]
        off += s
    gathered = xm.all_gather(packed.unsqueeze(0), dim=0)
    reshaped = gathered.view(ws, ws, mc)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    shard = xm.reduce_scatter(
        xm.REDUCE_SUM, transposed,
        scale=1.0 / ws, scatter_dim=0, shard_count=ws)
    idx = []
    for src in range(ws):
        c = rc[src]
        base = src * mc
        idx.extend(range(base, base + c))
    return torch.index_select(
        shard, 0, torch.tensor(idx, device=x.device, dtype=torch.long))


def make_routing(rank, ws, ntok, topk, nexp, dm):
    rng = torch.Generator().manual_seed(rank * 997 + 42)
    eids = torch.randint(0, nexp, (ntok, topk), generator=rng)
    flat = eids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_flat = flat[order]
    counts = [(sorted_flat == e).sum().item() for e in range(ws)]
    sc = [c * dm for c in counts]
    return sc


def exchange_counts(sc, ws):
    dev = xm.xla_device()
    t = torch.tensor(sc, device=dev, dtype=torch.int32)
    g = xm.all_gather(t.unsqueeze(0), dim=0)
    xm.mark_step()
    g = g.view(ws, ws)
    return g[:, xr.global_ordinal()].cpu().tolist()


def bench_one(fn, x, sc, rc, mc, warmup, iters, name, rank):
    for _ in range(warmup):
        out = fn(x, sc, rc, mc)
        xm.mark_step()

    xm.wait_device_ops()
    t0 = time.time()
    for _ in range(iters):
        out = fn(x, sc, rc, mc)
        xm.mark_step()
    xm.wait_device_ops()
    elapsed = time.time() - t0

    avg_ms = elapsed / iters * 1000
    if rank == 0:
        total_recv = sum(rc)
        print(f"  {name:20s}: {avg_ms:.3f} ms/call  "
              f"({iters} iters, recv={total_recv} elems)")
    return avg_ms


def run(args):
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()

    if rank == 0:
        print(f"AllToAllV Microbenchmark  world_size={ws}  "
              f"DM={DM}  NEXP={NEXP}  TOPK={TOPK}  NTOK={NTOK}")

    sc = make_routing(rank, ws, NTOK, TOPK, NEXP, DM)
    rc = exchange_counts(sc, ws)
    mc = max(max(sc), max(rc), 1)

    x = torch.randn(sum(sc), device=dev, dtype=torch.bfloat16)

    # Also prepare reverse direction (for backward pass analog)
    x_rev = torch.randn(sum(rc), device=dev, dtype=torch.bfloat16)

    if rank == 0:
        print(f"send_total={sum(sc)}  recv_total={sum(rc)}  max_chunk={mc}")
        print(f"warmup={args.warmup}  iters={args.iters}")
        print()
        print("Forward (dispatch):")

    fwd_evolved = bench_one(_evolved, x, sc, rc, mc,
                            args.warmup, args.iters, "Agent-Evolved", rank)
    fwd_agrs = bench_one(_agrs, x, sc, rc, mc,
                         args.warmup, args.iters, "AG+ReduceScatter", rank)

    if rank == 0:
        print()
        print("Backward (combine):")

    bwd_evolved = bench_one(_evolved, x_rev, rc, sc, mc,
                            args.warmup, args.iters, "Agent-Evolved", rank)
    bwd_agrs = bench_one(_agrs, x_rev, rc, sc, mc,
                         args.warmup, args.iters, "AG+ReduceScatter", rank)

    if rank == 0:
        total_evolved = fwd_evolved + bwd_evolved
        total_agrs = fwd_agrs + bwd_agrs

        # Per MoE layer: 2 AllToAllV (dispatch + combine) in fwd, 2 in bwd = 4
        calls_per_layer = 4
        nlayers = 12   # DeepSeek-MoE-Lite projection
        calls_per_step = calls_per_layer * nlayers

        overhead_evolved = total_evolved * calls_per_layer / 2 * nlayers
        overhead_agrs = total_agrs * calls_per_layer / 2 * nlayers

        print()
        print(f"{'='*60}")
        print(f"  Per-call (fwd+bwd round-trip):")
        print(f"    Agent-Evolved  : {total_evolved:.3f} ms")
        print(f"    AG+ReduceScatter: {total_agrs:.3f} ms")
        print(f"    Difference     : {total_agrs - total_evolved:+.3f} ms")
        print(f"  Projected per training step ({calls_per_step} calls, {nlayers} MoE layers):")
        print(f"    Agent-Evolved  : {overhead_evolved:.1f} ms")
        print(f"    AG+ReduceScatter: {overhead_agrs:.1f} ms")
        print(f"    Savings        : {overhead_agrs - overhead_evolved:.1f} ms/step")
        print(f"{'='*60}")

        out = os.environ.get(
            'RESULTS_DIR', '/home/ubuntu/trainium-llm-search/training/results')
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, 'alltoallv_bench.json')
        with open(path, 'w') as f:
            json.dump(dict(
                world_size=ws, dm=DM, nexp=NEXP, topk=TOPK, ntok=NTOK,
                iters=args.iters,
                fwd_evolved_ms=round(fwd_evolved, 3),
                fwd_agrs_ms=round(fwd_agrs, 3),
                bwd_evolved_ms=round(bwd_evolved, 3),
                bwd_agrs_ms=round(bwd_agrs, 3),
                total_evolved_ms=round(total_evolved, 3),
                total_agrs_ms=round(total_agrs, 3),
                projected_per_step_evolved_ms=round(overhead_evolved, 1),
                projected_per_step_agrs_ms=round(overhead_agrs, 1),
            ), f, indent=2)
        print(f"  Saved -> {path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--warmup', type=int, default=5)
    run(p.parse_args())
