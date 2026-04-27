#!/usr/bin/env python3
"""
DeepSeek-MoE-Lite training on Trainium with sequence-parallel attention
using Ring KV distribution.

Compares evolved ring_kv_gather (all_gather + view, 1 dispatch) vs baseline
(per-head all_gather, HEADS separate dispatches per K or V tensor).

    torchrun --nproc_per_node=32 --nnodes=2 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
        training/train_ring_kv.py --backend evolved --steps 5000
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
from runtime.trainium_ring_kv import ring_kv_gather as _kv_gather_evolved, init_ring_kv

# ======================== Config ========================

VOCAB  = 32768
DM     = 2048
HEADS  = 16
LAYERS = 8
NEXP   = 64
TOPK   = 6
EXDIM  = 1408
SEQLEN = 128
BSZ    = 1
SEED   = 42
N_BATCHES = 1000


# ======================== Baseline implementation ========================

def _kv_gather_baseline(kv_chunk):
    """Baseline: per-head all_gather (HEADS separate dispatches per K or V)."""
    n_heads = HEADS
    kv_size = kv_chunk.numel()
    head_sz = kv_size // n_heads
    parts = []
    for h in range(n_heads):
        head_data = kv_chunk[h * head_sz:(h + 1) * head_sz]
        gathered = xm.all_gather(head_data.unsqueeze(0), dim=0)
        parts.append(gathered.view(-1))
    return torch.cat(parts)


# ======================== KV gather autograd wrapper ========================

class _KVGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, fn, ws):
        ctx.fn, ctx.ws = fn, ws
        ctx.kv_size = kv.numel()
        xm.mark_step()
        out = fn(kv)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rank = xr.global_ordinal()
        kv_size = ctx.kv_size
        xm.mark_step()
        grad_local = grad_out[rank * kv_size:(rank + 1) * kv_size]
        xm.mark_step()
        return grad_local, None, None


# ======================== AllToAllV for MoE (slice+cat) ========================

def _a2av(x, ws, mc):
    from runtime.trainium_alltoallv import alltoallv
    return alltoallv(x, ws, mc)


class _A2AV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ws, mc, fn):
        ctx.ws, ctx.mc, ctx.fn = ws, mc, fn
        xm.mark_step()
        out = fn(x, ws, mc)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        out = ctx.fn(g.contiguous(), ctx.ws, ctx.mc)
        xm.mark_step()
        return out, None, None, None


# ======================== Model ========================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt().to(x.dtype) * self.w


class SeqParallelAttn(nn.Module):
    """Sequence-parallel attention: gathers K,V from all ranks."""
    def __init__(self, d, h, kv_gather_fn, ws):
        super().__init__()
        self.h, self.hd = h, d // h
        self.ws = ws
        self.kv_gather_fn = kv_gather_fn
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, S, D = x.shape

        q = self.q_proj(x).reshape(B, S, self.h, self.hd).permute(0, 2, 1, 3)

        k_local = self.k_proj(x).reshape(B, S, self.h, self.hd).permute(0, 2, 1, 3)
        v_local = self.v_proj(x).reshape(B, S, self.h, self.hd).permute(0, 2, 1, 3)

        k_flat = k_local.reshape(-1)
        v_flat = v_local.reshape(-1)

        k_all = _KVGather.apply(k_flat, self.kv_gather_fn, self.ws)
        v_all = _KVGather.apply(v_flat, self.kv_gather_fn, self.ws)

        S_global = S * self.ws
        k_global = k_all.reshape(B, self.h, S_global, self.hd)
        v_global = v_all.reshape(B, self.h, S_global, self.hd)

        a = (q @ k_global.transpose(-2, -1)) * (self.hd ** -0.5)
        a = F.softmax(a, dim=-1)
        out = a @ v_global

        return self.o(out.transpose(1, 2).reshape(B, S, D))


class Expert(nn.Module):
    def __init__(self, d, ed):
        super().__init__()
        self.up = nn.Linear(d, ed, bias=False)
        self.down = nn.Linear(ed, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class MoE(nn.Module):
    def __init__(self, d, ne, k, ed, ws, cap):
        super().__init__()
        self.d, self.k, self.ne, self.ws = d, k, ne, ws
        self.cap = cap
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
        pad_to = self.ws * self.cap * D
        if xe.reshape(-1).shape[0] < pad_to:
            send = F.pad(xe.reshape(-1), (0, pad_to - xe.reshape(-1).shape[0]))
        else:
            send = xe.reshape(-1)[:pad_to]
        xm.mark_step()
        mc = self.cap * D
        recv = _A2AV.apply(send, self.ws, mc, _a2av)
        n_recv_tok = self.ws * self.cap
        processed = self.expert(recv.view(n_recv_tok, D))
        xm.mark_step()
        combined = _A2AV.apply(processed.reshape(-1), self.ws, mc, _a2av)
        combined = combined[:T * self.k * D].reshape(T * self.k, D)
        return (combined.reshape(T, self.k, D) * ew.unsqueeze(-1)).sum(1).reshape(B, S, D)


class Block(nn.Module):
    def __init__(self, d, h, ne, k, ed, kv_gather_fn, ws, cap):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = SeqParallelAttn(d, h, kv_gather_fn, ws)
        self.n2, self.moe = RMSNorm(d), MoE(d, ne, k, ed, ws, cap)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.moe(self.n2(x))
        return x


class MoEModel(nn.Module):
    def __init__(self, kv_gather_fn, ws, cap):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.layers = nn.ModuleList(
            [Block(DM, HEADS, NEXP, TOPK, EXDIM, kv_gather_fn, ws, cap) for _ in range(LAYERS)])
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

    if rank == 0:
        print(f"[init] backend={args.backend}  world_size={ws}  steps={args.steps}")
        labels = {'evolved': 'evolved (all_gather + view, 1 dispatch)',
                  'baseline': f'baseline (per-head all_gather, {HEADS} dispatches per K/V)'}
        print(f"[init] KV gather: {labels[args.backend]}")
        kv_elems = BSZ * SEQLEN * DM
        print(f"[init] KV size per rank: {kv_elems} elements, "
              f"gathered: {kv_elems * ws} elements")
        print(f"[init] KV gathers per step: {LAYERS * 2} (K+V per layer)")

    init_ring_kv()
    kv_fn = _kv_gather_evolved if args.backend == 'evolved' else _kv_gather_baseline

    T = BSZ * SEQLEN
    cap = (T * TOPK + ws - 1) // ws

    torch.manual_seed(SEED)
    model = MoEModel(kv_fn, ws, cap).to(dev).to(torch.bfloat16)
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
        print(f"  Collective     : Ring KV (sequence-parallel attention)")
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
        path = os.path.join(out, f'rkv_{args.backend}_converge.json')
        with open(path, 'w') as f:
            json.dump(dict(
                collective='ring_kv',
                backend=args.backend,
                steps=args.steps,
                world_size=ws,
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
