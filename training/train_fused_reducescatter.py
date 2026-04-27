#!/usr/bin/env python3
"""
DeepSeek-MoE-Lite training on Trainium with Fused ReduceScatter.

Compares evolved fused ReduceScatter (1 dispatch on concatenated flat tensor)
vs baseline (N separate reduce_scatters, one per shard).

    torchrun --nproc_per_node=32 --nnodes=2 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
        training/train_fused_reducescatter.py --backend evolved --steps 5000
"""

import argparse
import os
import sys
import time
import json
import math

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
from runtime.trainium_fused_reducescatter import fused_reducescatter_flat, init_fused_reducescatter

# ======================== Config ========================

VOCAB  = 32768
DM     = 2048
HEADS  = 16
LAYERS = 12
NEXP   = 64
TOPK   = 6
EXDIM  = 1408
N_SHARDS = 8
SEQLEN = 256
BSZ    = 1
SEED   = 42
N_BATCHES = 1000


# ======================== ReduceScatter implementations ========================

def _frs_evolved(x, ws, n_shards, shard_size):
    """Evolved: single reduce_scatter on entire flat tensor (1 dispatch)."""
    return fused_reducescatter_flat(x, ws)


def _frs_baseline(x, ws, n_shards, shard_size):
    """Baseline: N separate reduce_scatters, one per shard (N dispatches)."""
    results = []
    for i in range(n_shards):
        shard = x[i * shard_size:(i + 1) * shard_size]
        rs = xm.reduce_scatter(
            xm.REDUCE_SUM, shard, scale=1.0,
            scatter_dim=0, shard_count=ws)
        results.append(rs)
    return torch.cat(results, dim=0)


# ======================== Autograd Wrapper ========================

class _FRS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ws, fn, n_shards, shard_size):
        ctx.ws, ctx.fn, ctx.n_shards, ctx.shard_size = ws, fn, n_shards, shard_size
        ctx.input_size = x.numel()
        xm.mark_step()
        out = fn(x, ws, n_shards, shard_size)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        gathered = xm.all_gather(g, dim=0)
        xm.mark_step()
        return gathered, None, None, None, None


# ======================== Model ========================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt().to(x.dtype) * self.w


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.register_buffer('mask', torch.triu(torch.ones(SEQLEN, SEQLEN, dtype=torch.bool), 1))

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        a = F.softmax(a.masked_fill(self.mask[:S, :S], -1e9), dim=-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, S, D))


class Expert(nn.Module):
    def __init__(self, d, ed):
        super().__init__()
        self.up = nn.Linear(d, ed, bias=False)
        self.down = nn.Linear(ed, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class MoE(nn.Module):
    """MoE block using reduce_scatter for output partitioning."""
    def __init__(self, d, ne, k, ed, fn, ws, cap, n_shards):
        super().__init__()
        self.d, self.k, self.ne, self.ws, self.fn = d, k, ne, ws, fn
        self.cap = cap
        self.n_shards = n_shards
        self.gate = nn.Linear(d, ne, bias=False)
        self.expert = Expert(d, ed)

    def forward(self, x):
        B, S, D = x.shape
        T = B * S
        xf = x.reshape(T, D)

        gp = F.softmax(self.gate(xf), dim=-1)
        _, top_idx = gp.topk(self.k, dim=-1)
        ew = torch.gather(gp, 1, top_idx)
        ew = ew / (ew.sum(-1, keepdim=True) + 1e-9)

        xe = xf.unsqueeze(1).expand(-1, self.k, -1).reshape(T * self.k, D)
        processed = self.expert(xe)

        elem_count = processed.numel()
        shard_elem = (elem_count + self.ws - 1) // self.ws
        total_needed = shard_elem * self.ws * self.n_shards

        flat = processed.reshape(-1).repeat(self.n_shards)
        if flat.numel() < total_needed:
            flat = F.pad(flat, (0, total_needed - flat.numel()))
        else:
            flat = flat[:total_needed]

        per_shard = flat.numel() // self.n_shards
        xm.mark_step()

        recv = _FRS.apply(flat, self.ws, self.fn, self.n_shards, per_shard)

        recv_total = recv.numel()
        usable = min(recv_total, T * self.k * D)
        combined = recv[:usable]
        if usable < T * self.k * D:
            combined = F.pad(combined, (0, T * self.k * D - usable))
        combined = combined.reshape(T * self.k, D)

        return (combined.reshape(T, self.k, D) * ew.unsqueeze(-1)).sum(1).reshape(B, S, D)


class Block(nn.Module):
    def __init__(self, d, h, ne, k, ed, fn, ws, cap, n_shards):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), Attn(d, h)
        self.n2, self.moe = RMSNorm(d), MoE(d, ne, k, ed, fn, ws, cap, n_shards)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.moe(self.n2(x))
        return x


class MoEModel(nn.Module):
    def __init__(self, fn, ws, cap):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.layers = nn.ModuleList(
            [Block(DM, HEADS, NEXP, TOPK, EXDIM, fn, ws, cap, N_SHARDS) for _ in range(LAYERS)])
        self.norm = RMSNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


# ======================== LR Schedule ========================

def get_lr(step, total_steps, max_lr=3e-4, min_lr=3e-5, warmup_steps=200):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_ratio))


# ======================== Training ========================

def run(args):
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()

    tokens_per_step = BSZ * SEQLEN * ws

    init_fused_reducescatter()
    fn = _frs_evolved if args.backend == 'evolved' else _frs_baseline

    if rank == 0:
        print(f"[init] backend={args.backend}  world_size={ws}  steps={args.steps}")
        labels = {'evolved': f'evolved (1 dispatch on flat tensor)',
                  'baseline': f'baseline ({N_SHARDS} separate reduce_scatters)'}
        print(f"[init] Fused ReduceScatter: {labels[args.backend]}")
        print(f"[init] n_shards={N_SHARDS}")

    T = BSZ * SEQLEN
    cap = (T * TOPK + ws - 1) // ws

    torch.manual_seed(SEED)
    model = MoEModel(fn, ws, cap).to(dev).to(torch.bfloat16)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rep = [p for n, p in model.named_parameters() if '.expert.' not in n]

    rng = torch.Generator().manual_seed(SEED + rank)
    data_inp = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=rng) for _ in range(N_BATCHES)]
    data_tgt = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=rng) for _ in range(N_BATCHES)]

    cur_inp = data_inp[0].to(dev)
    cur_tgt = data_tgt[0].to(dev)

    def step(s):
        nonlocal cur_inp, cur_tgt
        bi = s % N_BATCHES
        if bi == 0 and s > 0:
            cur_inp = data_inp[0].to(dev)
            cur_tgt = data_tgt[0].to(dev)
        elif s > 0:
            cur_inp = data_inp[bi].to(dev)
            cur_tgt = data_tgt[bi].to(dev)

        lr = get_lr(s, args.steps)
        for pg in opt.param_groups:
            pg['lr'] = lr

        logits = model(cur_inp)
        loss = F.cross_entropy(logits.view(-1, VOCAB), cur_tgt.view(-1))
        loss.backward()
        for p in rep:
            if p.grad is not None:
                p.grad.data = xm.all_reduce('sum', p.grad.data) / ws
        opt.step()
        opt.zero_grad()
        xm.mark_step()
        return loss

    if rank == 0:
        print("[warmup] compiling XLA graphs...")
    for i in range(args.warmup):
        step(0)
    if rank == 0:
        print("[warmup] done")

    xm.rendezvous('pre_measure')
    wall_start = time.time()
    times = []
    losses = []
    log_interval = 50

    for s in range(args.steps):
        t0 = time.time()
        loss = step(s)
        xm.wait_device_ops()
        dt = time.time() - t0
        times.append(dt)
        loss_val = loss.item()
        losses.append(loss_val)

        if rank == 0 and (s + 1) % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            avg_ms = sum(times[-log_interval:]) / log_interval * 1000
            lr = get_lr(s, args.steps)
            print(f"  step {s+1:>5}/{args.steps}  loss={avg_loss:.4f}  "
                  f"lr={lr:.2e}  avg={avg_ms:.0f}ms")

    wall_total = time.time() - wall_start
    avg_ms = sum(times) / len(times) * 1000
    final_loss = sum(losses[-100:]) / min(100, len(losses))

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  Collective     : Fused ReduceScatter (FSDP gradient partition)")
        print(f"  Backend        : {args.backend}")
        print(f"  Steps          : {args.steps}")
        print(f"  Final loss     : {final_loss:.4f}")
        print(f"  Wall clock     : {wall_total:.2f} s  ({wall_total/60:.1f} min)")
        print(f"  Avg step       : {avg_ms:.1f} ms")
        print(f"  Throughput     : {args.steps * tokens_per_step / wall_total:.0f} tok/s")
        print(f"{'='*65}")

        out = os.environ.get(
            'RESULTS_DIR', '/home/ubuntu/trainium-llm-search/training/results')
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f'frs_{args.backend}_converge.json')
        with open(path, 'w') as f:
            json.dump(dict(
                collective='fused_reducescatter',
                backend=args.backend,
                steps=args.steps,
                world_size=ws,
                n_shards=N_SHARDS,
                final_loss_avg100=round(final_loss, 4),
                wall_clock_s=round(wall_total, 2),
                avg_step_ms=round(avg_ms, 1),
                min_step_ms=round(min(times) * 1000, 1),
                max_step_ms=round(max(times) * 1000, 1),
                throughput_toks=round(args.steps * tokens_per_step / wall_total),
                losses=[round(l, 4) for l in losses],
                step_times_ms=[round(t * 1000, 1) for t in times],
            ), f, indent=2)
        print(f"  Saved → {path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['evolved', 'baseline'], required=True)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--warmup', type=int, default=5)
    run(p.parse_args())
