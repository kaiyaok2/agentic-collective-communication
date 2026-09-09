#!/usr/bin/env python3
"""
7-node 224-rank per-call training-scope micro-runner for PP cross-stage
send/recv (masked AR through paired-rank groups).

Mirrors the shape constants of experiments/model_extension/train_llama_e2e_amp3.py
(DM=2048, M=4 microbatches, B=1, S=2048, ws=224 -> half=112). The cross-stage
transfer is implemented as a paired-group AR sum, identical to the `transfer`
reference call in train_llama_e2e_amp3.py.

Backend semantics:
  baseline: per-microbatch transfer() call inside `for m in range(N_MB):` —
            one AR dispatch per microbatch (M total). Matches the
            train_llama_e2e_amp3.py step_per_mb path.
  agent:    runtime.trainium_pp_send_recv_7node.evolved_pp_send_recv —
            stacks all M activations and does one paired-group AR.
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
N_MB = 4
B = 1
S = 1024
WARMUP_DEFAULT = 5
SEED = 42
PROB = 'pp_send_recv'


def _baseline_per_mb(activations, src_stage, half, M, stage, pair_id, dev):
    """Per-mb masked all-rank AR (Llama-amp transfer style). Avoids sub-group
    AR which hits NRT_FAILURE on Neuron at 224 ranks."""
    outs = []
    a0 = activations[0]
    for m in range(M):
        buf = torch.zeros(half, *a0.shape, dtype=a0.dtype, device=dev)
        if stage == src_stage:
            buf = buf.clone()
            buf[pair_id] = activations[m]
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        outs.append(ared[pair_id])
    return outs


def _load_agent():
    from runtime.trainium_pp_send_recv_7node import (
        evolved_pp_send_recv, init_pp_send_recv)
    init_pp_send_recv()
    return evolved_pp_send_recv


class _PPSendRecv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, flat_in, M, ws, half, stage, pair_id, src_stage,
                backend, rank, fn, dev):
        ctx.M, ctx.ws, ctx.half = M, ws, half
        ctx.stage, ctx.pair_id, ctx.src_stage = stage, pair_id, src_stage
        ctx.backend, ctx.fn, ctx.rank, ctx.dev = backend, fn, rank, dev
        # flat_in: (M, B, S, DM)
        activations = [flat_in[m] for m in range(M)]
        xm.mark_step()
        if probe.in_window():
            t0 = _t.time()
            if backend == 'baseline':
                outs = _baseline_per_mb(activations, src_stage, half, M,
                                        stage, pair_id, dev)
            else:
                outs = fn(activations, src_stage, half, M, rank, ws,
                          16, 2, xm, torch, num_nodes=7)
            flat_out = torch.stack(outs, dim=0)
            _ = flat_out.float().sum().item()
            probe.record(f'{PROB}_fwd', (_t.time() - t0) * 1000.0)
        else:
            if backend == 'baseline':
                outs = _baseline_per_mb(activations, src_stage, half, M,
                                        stage, pair_id, dev)
            else:
                outs = fn(activations, src_stage, half, M, rank, ws,
                          16, 2, xm, torch, num_nodes=7)
            flat_out = torch.stack(outs, dim=0)
        xm.mark_step()
        return flat_out

    @staticmethod
    def backward(ctx, g):
        # Symmetric send/recv: backward reverses direction (dst_stage->src).
        M, ws, half = ctx.M, ctx.ws, ctx.half
        stage, pair_id = ctx.stage, ctx.pair_id
        backend, fn, rank, dev = ctx.backend, ctx.fn, ctx.rank, ctx.dev
        # In real autograd through transfer(), the backward AR mirrors the
        # forward AR with reversed src_stage.
        rev_src = 1 - ctx.src_stage
        activations = [g[m].contiguous() for m in range(M)]
        xm.mark_step()
        if probe.in_window():
            t0 = _t.time()
            if backend == 'baseline':
                outs = _baseline_per_mb(activations, rev_src, half, M,
                                        stage, pair_id, dev)
            else:
                outs = fn(activations, rev_src, half, M, rank, ws,
                          16, 2, xm, torch, num_nodes=7)
            flat_out = torch.stack(outs, dim=0)
            _ = flat_out.float().sum().item()
            probe.record(f'{PROB}_bwd', (_t.time() - t0) * 1000.0)
        else:
            if backend == 'baseline':
                outs = _baseline_per_mb(activations, rev_src, half, M,
                                        stage, pair_id, dev)
            else:
                outs = fn(activations, rev_src, half, M, rank, ws,
                          16, 2, xm, torch, num_nodes=7)
            flat_out = torch.stack(outs, dim=0)
        xm.mark_step()
        return (flat_out, None, None, None, None, None, None, None,
                None, None, None)


def run(args):
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()
    half = ws // 2
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    src_stage = 0  # always 0 -> 1 transfer

    fn = _load_agent() if args.backend == 'agent' else None
    if rank == 0:
        print(f"[init] backend={args.backend} ws={ws} half={half} "
              f"steps={args.steps} DM={DM} M={N_MB} B={B} S={S} "
              f"dispatches/step: baseline={N_MB} agent=1")

    torch.manual_seed(SEED)
    w = nn.Parameter(torch.randn(DM, DM, dtype=torch.bfloat16, device=dev) * 0.01)
    inp = torch.randn(B, S, DM, dtype=torch.bfloat16, device=dev)
    opt = torch.optim.SGD([w], lr=1e-4)

    def step(s):
        probe.set_step(s)
        opt.zero_grad()
        if stage == src_stage:
            acts = [torch.matmul(inp, w) for _ in range(N_MB)]
        else:
            acts = [torch.zeros(B, S, DM, dtype=torch.bfloat16, device=dev)
                    for _ in range(N_MB)]
        flat_in = torch.stack(acts, dim=0)
        flat_out = _PPSendRecv.apply(
            flat_in, N_MB, ws, half, stage, pair_id, src_stage,
            args.backend, rank, fn, dev)
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
