#!/usr/bin/env python3
"""Per-call Uniform-AllToAll probe sweep.

Measures wall-clock per-call latency of two ua2a compositions at a
configurable (ws, chunk) shape.

  baseline (AG+T+RS): one all_gather, one strided permute/contiguous,
                      one reduce_scatter — two collective dispatches.

  agent (AG+slice):   one all_gather over the rank-major buffer, take
                      this rank's row by metadata-only view + slice —
                      one collective dispatch, no strided permute.

Run with torchrun --nproc_per_node=<ws> ... (single or multi node).
"""
import argparse, os, sys, time, json, statistics, pathlib
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')


def _ua2a_baseline(x, ws, chunk):
    # AG + transpose/contiguous + RS. x is (chunk,) flat per rank.
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)        # (ws, chunk)
    reshaped = gathered.view(ws, ws, chunk // ws)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                             scale=1.0 / ws, scatter_dim=0, shard_count=ws)


def _ua2a_agent(x, ws, chunk):
    # Pack-and-gather: one AG over (ws, chunk) buffer per rank.
    # The output is this rank's row from the gathered tensor — a metadata-only
    # slice, no strided copy, no reduce_scatter.
    x2 = x.view(ws, chunk // ws) if x.numel() == ws * (chunk // ws) else x.view(1, chunk).expand(ws, chunk).contiguous()
    gathered = xm.all_gather(x2, dim=1)
    rank = xr.global_ordinal()
    return gathered[rank].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['baseline', 'agent', 'both'], default='both')
    ap.add_argument('--chunk', type=int, default=16384,
                    help='per-rank flat element count (must be divisible by ws)')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device()
    ws = xr.world_size()
    rank = xr.global_ordinal()
    chunk = args.chunk
    assert chunk % ws == 0, f'chunk={chunk} must be divisible by ws={ws}'

    if rank == 0:
        print(f'[ua2a-probe] ws={ws} chunk={chunk} backend={args.backend} '
              f'iters={args.iters} warmup={args.warmup}', flush=True)

    def time_one(fn, label):
        x = torch.randn(chunk, device=dev, dtype=torch.bfloat16)
        # warmup
        for _ in range(args.warmup):
            y = fn(x, ws, chunk)
            xm.mark_step()
        # measure
        latencies = []
        for _ in range(args.iters):
            t0 = time.time()
            y = fn(x, ws, chunk)
            _ = y.sum().item()  # forces materialization
            latencies.append((time.time() - t0) * 1000.0)
        return latencies

    results = {}
    if args.backend in ('baseline', 'both'):
        lat = time_one(_ua2a_baseline, 'baseline')
        results['baseline'] = lat
        if rank == 0:
            print(f'  baseline: median={statistics.median(lat):.3f} ms/call '
                  f'mean={statistics.mean(lat):.3f} ms/call (n={len(lat)})', flush=True)

    if args.backend in ('agent', 'both'):
        lat = time_one(_ua2a_agent, 'agent')
        results['agent'] = lat
        if rank == 0:
            print(f'  agent:    median={statistics.median(lat):.3f} ms/call '
                  f'mean={statistics.mean(lat):.3f} ms/call (n={len(lat)})', flush=True)

    if rank == 0 and results:
        pathlib.Path('/tmp/tp_search').mkdir(parents=True, exist_ok=True)
        tag = args.tag or f'ws{ws}_c{chunk}'
        out = {
            'ws': ws, 'chunk': chunk, 'iters': args.iters,
            **{k: {'median_ms': statistics.median(v), 'mean_ms': statistics.mean(v),
                   'all_ms': v} for k, v in results.items()}
        }
        path = f'/tmp/tp_search/ua2a_sweep_{tag}.json'
        with open(path, 'w') as f:
            json.dump(out, f)
        print(f'[ua2a-probe] results written to {path}', flush=True)


if __name__ == '__main__':
    main()
