#!/usr/bin/env python3
"""
7-node 224-rank per-call training-scope micro-runner for TP-MLP all-reduce.

Mirrors the shape constants of experiments/model_extension/train_llama_e2e_amp3.py
(DM=2048, HID=5376, M=4 microbatches, N_LAYERS_PER_STAGE=1, B=1, S=2048, ws=224)
so the per-collective timing is taken under the same byte budget that the Llama
end-to-end amp3 sees.

Backend semantics:
  baseline: per-microbatch xm.all_reduce(SUM, partial) inside a `for m in
            range(N_MB): for L in range(N_LAYERS):` loop — matches the
            tp_fsdp_block reference call in train_llama_e2e_amp3.py.
  agent:    runtime.trainium_tp_mlp_7node.evolved_tp_mlp — single fused
            all_reduce across the (M, N_LAYERS) stack.

Per-call probing is gated by training._percall_probe.in_window() so we can
restrict measurement to a late window (PROBE_START_STEP..PROBE_END_STEP) and
keep the NEFF cache stable. The collective itself is wrapped in
`torch.autograd.Function` with `xm.mark_step()` before/after the call, so the
graph-boundary semantics that real autograd training imposes are reproduced.
"""
import argparse, os, sys, time, json, math, statistics
import time as _t

os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_RT_STOCHASTIC_ROUNDING_EN', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
from training import _percall_probe as probe

# Mirror train_llama_e2e_amp3.py shape constants exactly.
DM = 2048
HID = 5376
N_LAYERS_PER_STAGE = 1
N_MB = 4
B = 1
S = 1024
WARMUP_DEFAULT = 5
SEED = 42
PROB = 'tp_mlp'


def _baseline_per_mb(partials, M, N_LAYERS, ws):
    """Per-microbatch all_reduce, one dispatch per (m, L). Copy of the
    canonical tp_fsdp_block AR call applied separately to each partial."""
    outs = [[None] * N_LAYERS for _ in range(M)]
    for m in range(M):
        for L in range(N_LAYERS):
            outs[m][L] = xm.all_reduce(xm.REDUCE_SUM, partials[m][L]) / ws
    return outs


def _load_agent():
    from runtime.trainium_tp_mlp_7node import evolved_tp_mlp, init_tp_mlp
    init_tp_mlp()
    return evolved_tp_mlp


class _TPMLP(torch.autograd.Function):
    """Wrap the (M, N_LAYERS) tp_mlp AR call in an autograd.Function so the
    XLA graph sees the same `mark_step / collective / mark_step` boundaries
    that real autograd training imposes."""

    @staticmethod
    def forward(ctx, flat_in, M, N_LAYERS, ws, backend, rank, fn):
        # flat_in has shape (M*N_LAYERS, B, S, DM); rebuild list-of-lists.
        ctx.M, ctx.N_LAYERS, ctx.ws, ctx.backend = M, N_LAYERS, ws, backend
        ctx.fn = fn
        ctx.rank = rank
        partials = [[flat_in[m * N_LAYERS + L] for L in range(N_LAYERS)]
                    for m in range(M)]
        xm.mark_step()
        if probe.in_window():
            t0 = _t.time()
            if backend == 'baseline':
                outs = _baseline_per_mb(partials, M, N_LAYERS, ws)
            else:
                outs = fn(partials, M, N_LAYERS, rank, ws,
                          16, 2, xm, torch, num_nodes=7)
            flat_out = torch.stack(
                [outs[m][L] for m in range(M) for L in range(N_LAYERS)], dim=0)
            _ = flat_out.float().sum().item()
            probe.record(f'{PROB}_fwd', (_t.time() - t0) * 1000.0)
        else:
            if backend == 'baseline':
                outs = _baseline_per_mb(partials, M, N_LAYERS, ws)
            else:
                outs = fn(partials, M, N_LAYERS, rank, ws,
                          16, 2, xm, torch, num_nodes=7)
            flat_out = torch.stack(
                [outs[m][L] for m in range(M) for L in range(N_LAYERS)], dim=0)
        xm.mark_step()
        return flat_out

    @staticmethod
    def backward(ctx, g):
        # AR is its own transpose: backward is another AR-sum across ranks.
        M, N_LAYERS, ws = ctx.M, ctx.N_LAYERS, ctx.ws
        backend, fn, rank = ctx.backend, ctx.fn, ctx.rank
        partials = [[g[m * N_LAYERS + L].contiguous() for L in range(N_LAYERS)]
                    for m in range(M)]
        xm.mark_step()
        if probe.in_window():
            t0 = _t.time()
            if backend == 'baseline':
                outs = _baseline_per_mb(partials, M, N_LAYERS, ws)
            else:
                outs = fn(partials, M, N_LAYERS, rank, ws,
                          16, 2, xm, torch, num_nodes=7)
            flat_out = torch.stack(
                [outs[m][L] for m in range(M) for L in range(N_LAYERS)], dim=0)
            _ = flat_out.float().sum().item()
            probe.record(f'{PROB}_bwd', (_t.time() - t0) * 1000.0)
        else:
            if backend == 'baseline':
                outs = _baseline_per_mb(partials, M, N_LAYERS, ws)
            else:
                outs = fn(partials, M, N_LAYERS, rank, ws,
                          16, 2, xm, torch, num_nodes=7)
            flat_out = torch.stack(
                [outs[m][L] for m in range(M) for L in range(N_LAYERS)], dim=0)
        xm.mark_step()
        return flat_out, None, None, None, None, None, None


def run(args):
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()
    assert HID % ws == 0
    shard_hid = HID // ws

    fn = _load_agent() if args.backend == 'agent' else None
    if rank == 0:
        print(f"[init] backend={args.backend} ws={ws} steps={args.steps} "
              f"DM={DM} HID={HID} M={N_MB} N_LAYERS={N_LAYERS_PER_STAGE} "
              f"B={B} S={S}")

    torch.manual_seed(SEED)
    # Create the (B, S, DM) "partial" inputs and a dummy parameter chain so
    # we have real gradients to AR over.
    w_in = nn.Parameter(torch.randn(DM, DM, dtype=torch.bfloat16, device=dev) * 0.01)
    inputs = [torch.randn(B, S, DM, dtype=torch.bfloat16, device=dev)
              for _ in range(N_MB * N_LAYERS_PER_STAGE)]
    opt = torch.optim.SGD([w_in], lr=1e-4)

    def step(s):
        probe.set_step(s)
        opt.zero_grad()
        # produce M*N_LAYERS bf16 (B,S,DM) partials sourced from a real matmul
        partials = []
        for k in range(N_MB * N_LAYERS_PER_STAGE):
            partials.append(torch.matmul(inputs[k], w_in))
        flat_in = torch.stack(partials, dim=0)
        flat_out = _TPMLP.apply(flat_in, N_MB, N_LAYERS_PER_STAGE, ws,
                                args.backend, rank, fn)
        loss = flat_out.float().pow(2).mean()
        loss.backward()
        opt.step()
        xm.mark_step()
        return loss

    if rank == 0:
        print("[warmup] compiling XLA graphs...")
    for _ in range(args.warmup):
        step(0)
    if rank == 0:
        print("[warmup] done")
    xm.rendezvous('pre_measure')

    wall_start = time.time()
    times = []
    for s in range(args.steps):
        t0 = time.time()
        loss = step(s)
        xm.wait_device_ops()
        dt = time.time() - t0
        times.append(dt)
        if rank == 0 and (s + 1) % 50 == 0:
            recent = times[-50:]
            print(f"  step {s+1:>5}/{args.steps}  median_ms="
                  f"{statistics.median(recent)*1000:.1f}")

    wall = time.time() - wall_start
    if rank == 0:
        avg_ms = sum(times) / len(times) * 1000
        steady = sum(times[200:]) / max(1, len(times) - 200) * 1000 \
                 if len(times) > 200 else None
        print(f"\n  Problem     : {PROB}")
        print(f"  Backend     : {args.backend}")
        print(f"  Wall        : {wall:.2f} s")
        print(f"  Avg step    : {avg_ms:.1f} ms")
        print(f"  Steady step : {steady:.1f} ms (from step 200)"
              if steady else "")
        out = os.environ.get('RESULTS_DIR',
                             '/home/ubuntu/agentic-collective-communication/training/results/percall_r22')
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f'{PROB}_{args.backend}_step.json'), 'w') as f:
            json.dump(dict(
                problem=PROB, backend=args.backend, steps=args.steps,
                world_size=ws, wall_s=round(wall, 2),
                avg_ms=round(avg_ms, 1),
                steady_ms=round(steady, 1) if steady else None,
                step_times_ms=[round(t*1000, 1) for t in times],
            ), f, indent=2)
        probe.dump(os.path.join(out, f'{PROB}_{args.backend}_percall.json'),
                   extra={'backend': args.backend, 'steps': args.steps,
                          'problem': PROB})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['baseline', 'agent'], required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--warmup', type=int, default=WARMUP_DEFAULT)
    run(p.parse_args())
