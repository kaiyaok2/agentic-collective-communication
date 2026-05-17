#!/usr/bin/env python3
"""
Minimal MoE training using native xm.all_to_all (compiles reliably on neuron-cc).
Measures baseline step time, then AllToAllV overhead difference is added separately.

    torchrun --nproc_per_node=32 --nnodes=2 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
        train_native.py --steps 50
"""

import argparse
import os
import time
import json

os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

VOCAB  = 1024
DM     = 128
HEADS  = 4
LAYERS = 2
NEXP   = 64
TOPK   = 2
EXDIM  = 64
SEQLEN = 64
BSZ    = 2


def native_alltoall(x, ws):
    """Use xm.all_to_all with equal splits (uniform routing approximation)."""
    return xm.all_to_all(x, split_dimension=0, concat_dimension=0, split_count=ws)


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

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
        a = F.softmax(a.masked_fill(mask, -1e9), dim=-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, S, D))


class Expert(nn.Module):
    def __init__(self, d, ed):
        super().__init__()
        self.up = nn.Linear(d, ed, bias=False)
        self.down = nn.Linear(ed, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class MoE(nn.Module):
    def __init__(self, d, ne, k, ed, ws):
        super().__init__()
        self.d, self.k, self.ws = d, k, ws
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

        # Uniform all_to_all: pad to divisible by world_size
        xe = xf.unsqueeze(1).expand(-1, self.k, -1).reshape(T * self.k, D)
        pad_to = ((T * self.k + self.ws - 1) // self.ws) * self.ws
        if xe.shape[0] < pad_to:
            xe = F.pad(xe, (0, 0, 0, pad_to - xe.shape[0]))
        dispatched = native_alltoall(xe, self.ws)
        processed = self.expert(dispatched)
        combined = native_alltoall(processed, self.ws)
        combined = combined[:T * self.k]

        return (combined.reshape(T, self.k, D) * ew.unsqueeze(-1)).sum(1).reshape(B, S, D)


class Block(nn.Module):
    def __init__(self, d, h, ne, k, ed, ws):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), Attn(d, h)
        self.n2, self.moe = RMSNorm(d), MoE(d, ne, k, ed, ws)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.moe(self.n2(x))
        return x


class MoEModel(nn.Module):
    def __init__(self, ws):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.layers = nn.ModuleList(
            [Block(DM, HEADS, NEXP, TOPK, EXDIM, ws) for _ in range(LAYERS)])
        self.norm = RMSNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


def run(args):
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()

    if rank == 0:
        print(f"[init] world_size={ws}  steps={args.steps}  warmup={args.warmup}")

    model = MoEModel(ws).to(dev).to(torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[model] {n:,} params  layers={LAYERS}  experts={NEXP}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    inp = torch.randint(0, VOCAB, (BSZ, SEQLEN), device=dev)
    tgt = torch.randint(0, VOCAB, (BSZ, SEQLEN), device=dev)

    def step():
        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, VOCAB), tgt.view(-1))
        loss.backward()
        xm.all_reduce('sum', [p.grad for p in model.parameters() if p.grad is not None])
        opt.step()
        opt.zero_grad()
        xm.mark_step()
        return loss

    if rank == 0:
        print("[warmup] compiling...")
    for _ in range(args.warmup):
        step()
    if rank == 0:
        print("[warmup] done")

    xm.wait_device_ops()
    wall_start = time.time()
    times = []
    for s in range(args.steps):
        t0 = time.time()
        step()
        xm.wait_device_ops()
        times.append(time.time() - t0)
        if rank == 0 and (s + 1) % 10 == 0:
            avg = sum(times[-10:]) / min(10, len(times))
            print(f"  step {s+1:>4}/{args.steps}  avg={avg*1000:.1f}ms")

    wall_total = time.time() - wall_start
    avg = sum(times) / len(times) * 1000

    if rank == 0:
        print(f"\n{'='*55}")
        print(f"  Steps       : {args.steps}")
        print(f"  Total       : {wall_total:.2f} s")
        print(f"  Avg step    : {avg:.1f} ms")
        print(f"  Min step    : {min(times)*1000:.1f} ms")
        print(f"  Max step    : {max(times)*1000:.1f} ms")
        print(f"{'='*55}")

        out = os.environ.get(
            'RESULTS_DIR', '/home/ubuntu/trainium-llm-search/training/results')
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, 'native_training.json')
        with open(path, 'w') as f:
            json.dump(dict(
                steps=args.steps,
                total_s=round(wall_total, 2),
                avg_ms=round(avg, 1),
                min_ms=round(min(times) * 1000, 1),
                max_ms=round(max(times) * 1000, 1),
                step_times_ms=[round(t * 1000, 1) for t in times],
            ), f, indent=2)
        print(f"  Saved -> {path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--warmup', type=int, default=3)
    run(p.parse_args())
