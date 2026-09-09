#!/usr/bin/env python3
"""
7-node 224-rank per-call training-scope micro-runner for Llama block AR
(parallel-attention: attn + mlp partials).

Mirrors the shape constants of experiments/model_extension/train_llama_e2e_amp3.py
(DM=2048, HID=5376, M=4 microbatches, N_LAYERS_PER_STAGE=1, B=1, S=2048,
ws=224). Each "dispatch" reduces a (B, S, DM) partial across the TP group.

Backend semantics:
  baseline: two separate xm.all_reduce(SUM, attn_partial) and
            xm.all_reduce(SUM, mlp_partial) — i.e. 2 dispatches per (m, L).
            Called M*N_LAYERS times per step.
  agent:    runtime.trainium_llama_block_ar_7node.evolved_llama_block —
            stack(attn, mlp) + single AR. 1 dispatch per (m, L).
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

# Mirror train_llama_e2e_amp3.py shapes.
DM = 2048
HID = 5376
N_LAYERS_PER_STAGE = 1
N_MB = 4
B = 1
S = 1024
WARMUP_DEFAULT = 5
SEED = 42
PROB = 'llama_block_ar'


def _baseline_two_ar(attn, mlp, ws):
    """Two separate AR dispatches."""
    a = xm.all_reduce(xm.REDUCE_SUM, attn) / ws
    m = xm.all_reduce(xm.REDUCE_SUM, mlp) / ws
    return a, m


def _load_agent():
    from runtime.trainium_llama_block_ar_7node import (
        evolved_llama_block, init_llama_block_ar)
    init_llama_block_ar()
    return evolved_llama_block


class _LlamaBlockAR(torch.autograd.Function):
    """One forward call carries M*N_LAYERS (attn, mlp) pairs to keep the
    measurement under the same byte budget as Llama amp3 step. Per-call
    probing records the WHOLE batched call so that dividing by
    (M*N_LAYERS * dispatches_per_call) gives a per-AR latency.

    In baseline mode we issue 2*M*N_LAYERS separate AR dispatches in a loop
    (matching the per-mb llama_e2e_amp3 baseline). In agent mode we call
    evolved_llama_block once per (m, L) which does 1 fused AR per pair —
    so M*N_LAYERS dispatches total.
    """

    @staticmethod
    def forward(ctx, flat_in, M, N_LAYERS, ws, backend, rank, fn):
        # flat_in: (M*N_LAYERS, 2, B, S, DM); flat_in[k, 0] = attn,
        #          flat_in[k, 1] = mlp.
        ctx.M, ctx.N_LAYERS, ctx.ws = M, N_LAYERS, ws
        ctx.backend, ctx.fn, ctx.rank = backend, fn, rank
        K = M * N_LAYERS
        xm.mark_step()
        if probe.in_window():
            t0 = _t.time()
            outs = []
            for k in range(K):
                attn = flat_in[k, 0]
                mlp = flat_in[k, 1]
                if backend == 'baseline':
                    a, m = _baseline_two_ar(attn, mlp, ws)
                else:
                    a, m = fn(attn, mlp, rank, ws, 16, 2, xm, torch,
                              num_nodes=7)
                outs.append(torch.stack([a, m], dim=0))
            flat_out = torch.stack(outs, dim=0)
            _ = flat_out.float().sum().item()
            probe.record(f'{PROB}_fwd', (_t.time() - t0) * 1000.0)
        else:
            outs = []
            for k in range(K):
                attn = flat_in[k, 0]
                mlp = flat_in[k, 1]
                if backend == 'baseline':
                    a, m = _baseline_two_ar(attn, mlp, ws)
                else:
                    a, m = fn(attn, mlp, rank, ws, 16, 2, xm, torch,
                              num_nodes=7)
                outs.append(torch.stack([a, m], dim=0))
            flat_out = torch.stack(outs, dim=0)
        xm.mark_step()
        return flat_out

    @staticmethod
    def backward(ctx, g):
        # AR is its own transpose.
        M, N_LAYERS, ws = ctx.M, ctx.N_LAYERS, ctx.ws
        backend, fn, rank = ctx.backend, ctx.fn, ctx.rank
        K = M * N_LAYERS
        xm.mark_step()
        if probe.in_window():
            t0 = _t.time()
            outs = []
            for k in range(K):
                attn = g[k, 0].contiguous()
                mlp = g[k, 1].contiguous()
                if backend == 'baseline':
                    a, m = _baseline_two_ar(attn, mlp, ws)
                else:
                    a, m = fn(attn, mlp, rank, ws, 16, 2, xm, torch,
                              num_nodes=7)
                outs.append(torch.stack([a, m], dim=0))
            flat_out = torch.stack(outs, dim=0)
            _ = flat_out.float().sum().item()
            probe.record(f'{PROB}_bwd', (_t.time() - t0) * 1000.0)
        else:
            outs = []
            for k in range(K):
                attn = g[k, 0].contiguous()
                mlp = g[k, 1].contiguous()
                if backend == 'baseline':
                    a, m = _baseline_two_ar(attn, mlp, ws)
                else:
                    a, m = fn(attn, mlp, rank, ws, 16, 2, xm, torch,
                              num_nodes=7)
                outs.append(torch.stack([a, m], dim=0))
            flat_out = torch.stack(outs, dim=0)
        xm.mark_step()
        return flat_out, None, None, None, None, None, None


def run(args):
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()

    fn = _load_agent() if args.backend == 'agent' else None
    if rank == 0:
        K = N_MB * N_LAYERS_PER_STAGE
        nd_base = 2 * K
        nd_agent = K
        print(f"[init] backend={args.backend} ws={ws} steps={args.steps} "
              f"DM={DM} M={N_MB} N_LAYERS={N_LAYERS_PER_STAGE} B={B} S={S} "
              f"dispatches/step: baseline={nd_base} agent={nd_agent}")

    torch.manual_seed(SEED)
    w = nn.Parameter(torch.randn(DM, DM, dtype=torch.bfloat16, device=dev) * 0.01)
    inp = torch.randn(B, S, DM, dtype=torch.bfloat16, device=dev)
    opt = torch.optim.SGD([w], lr=1e-4)

    def step(s):
        probe.set_step(s)
        opt.zero_grad()
        K = N_MB * N_LAYERS_PER_STAGE
        attn_partials = [torch.matmul(inp, w) for _ in range(K)]
        mlp_partials = [torch.matmul(inp, w) for _ in range(K)]
        # Stack (K, 2, B, S, DM).
        flat_in = torch.stack(
            [torch.stack([attn_partials[k], mlp_partials[k]], dim=0)
             for k in range(K)], dim=0)
        flat_out = _LlamaBlockAR.apply(flat_in, N_MB, N_LAYERS_PER_STAGE, ws,
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
