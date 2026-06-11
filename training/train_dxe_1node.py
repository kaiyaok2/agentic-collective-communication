#!/usr/bin/env python3
"""
1-node 32-rank DeepSeek-MoE-Lite training comparing baseline vs agent
distributed cross-entropy on a vocab-sharded head.

Baseline : all_gather(logits_local, dim=-1) + F.cross_entropy on the
           materialised (B*S, V) tensor (developer-written).
Agent    : runtime.trainium_dxe.dxe_loss (2 all_reduces, no max-shift).

The MoE block uses the baseline AG+T+RS AllToAllV; gradient sync uses
plain xm.all_reduce. The only swap is the cross-entropy.
"""
import argparse, os, sys, time, json, math

os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_RT_STOCHASTIC_ROUNDING_EN', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

VOCAB = 32768  # divisible by ws=32 -> V_local = 1024
DM, HEADS = 2048, 16
LAYERS, NEXP, TOPK, EXDIM = 12, 64, 6, 1408
SEQLEN, BSZ, SEED, N_BATCHES = 256, 1, 42, 1000


def _ua2a_baseline(x, ws, chunk_size):
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    reshaped = gathered.view(ws, ws, chunk_size)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                             scale=1.0 / ws, scatter_dim=0, shard_count=ws)


class _UA2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ws, chunk_size, fn):
        ctx.ws, ctx.chunk_size, ctx.fn = ws, chunk_size, fn
        xm.mark_step(); out = fn(x, ws, chunk_size); xm.mark_step()
        return out
    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        out = ctx.fn(g.contiguous(), ctx.ws, ctx.chunk_size)
        xm.mark_step()
        return out, None, None, None


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt().to(x.dtype) * self.w


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3*d, bias=False); self.o = nn.Linear(d, d, bias=False)
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
        self.up = nn.Linear(d, ed, bias=False); self.down = nn.Linear(ed, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class MoE(nn.Module):
    def __init__(self, d, ne, k, ed, ws, cap):
        super().__init__()
        self.d, self.k, self.ne, self.ws, self.cap = d, k, ne, ws, cap
        self.gate = nn.Linear(d, ne, bias=False); self.expert = Expert(d, ed)
    def forward(self, x):
        B, S, D = x.shape
        T = B * S
        xf = x.reshape(T, D)
        gp = F.softmax(self.gate(xf), dim=-1)
        _, top_idx = gp.topk(self.k, dim=-1)
        ew = torch.gather(gp, 1, top_idx); ew = ew / (ew.sum(-1, keepdim=True) + 1e-9)
        xe = xf.unsqueeze(1).expand(-1, self.k, -1).reshape(T * self.k, D)
        pad_to = self.ws * self.cap * D
        send = F.pad(xe.reshape(-1), (0, pad_to - xe.reshape(-1).shape[0])) \
                if xe.reshape(-1).shape[0] < pad_to else xe.reshape(-1)[:pad_to]
        xm.mark_step()
        chunk_size = self.cap * D
        recv = _UA2A.apply(send, self.ws, chunk_size, _ua2a_baseline)
        processed = self.expert(recv.view(self.ws * self.cap, D))
        xm.mark_step()
        combined = _UA2A.apply(processed.reshape(-1), self.ws, chunk_size, _ua2a_baseline)
        combined = combined[:T * self.k * D].reshape(T * self.k, D)
        return (combined.reshape(T, self.k, D) * ew.unsqueeze(-1)).sum(1).reshape(B, S, D)


class Block(nn.Module):
    def __init__(self, d, h, ne, k, ed, ws, cap):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), Attn(d, h)
        self.n2, self.moe = RMSNorm(d), MoE(d, ne, k, ed, ws, cap)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.moe(self.n2(x))
        return x


class ShardedHeadMoEModel(nn.Module):
    def __init__(self, ws, cap, v_local):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.layers = nn.ModuleList(
            [Block(DM, HEADS, NEXP, TOPK, EXDIM, ws, cap) for _ in range(LAYERS)])
        self.norm = RMSNorm(DM)
        self.head_shard = nn.Linear(DM, v_local, bias=False)
    def forward(self, ids):
        x = self.emb(ids)
        for L in self.layers:
            x = L(x)
        return self.head_shard(self.norm(x))


def _ce_baseline(logits_local, targets, v_local):
    """All-gather the local logit shard and run F.cross_entropy on full (N, V)."""
    ll = logits_local.reshape(-1, v_local).contiguous()
    gathered = xm.all_gather(ll, dim=1)
    return F.cross_entropy(gathered, targets.reshape(-1))


def get_lr(step, total, max_lr=3e-4, min_lr=3e-5, warmup=200):
    if step < warmup:
        return max_lr * step / warmup
    r = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * r))


def run(args):
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()
    assert VOCAB % ws == 0
    v_local = VOCAB // ws
    T = BSZ * SEQLEN
    cap = (T * TOPK + ws - 1) // ws

    if args.backend == 'agent':
        from runtime.trainium_dxe import dxe_loss, init_dxe
        init_dxe()
        ce = lambda ll, tg: dxe_loss(ll.reshape(-1, v_local).contiguous(),
                                     tg.reshape(-1), v_local)
    else:
        ce = lambda ll, tg: _ce_baseline(ll, tg, v_local)

    if rank == 0:
        print(f"[init] backend={args.backend} ws={ws} steps={args.steps} v_local={v_local}")
        print(f"[init] cross-entropy: "
              f"{'agent (2 ARs, no max-shift)' if args.backend=='agent' else 'baseline (all_gather + F.cross_entropy)'}")

    torch.manual_seed(SEED)
    model = ShardedHeadMoEModel(ws, cap, v_local).to(dev).to(torch.bfloat16)
    rep = [p for n, p in model.named_parameters()
           if '.expert.' not in n and 'head_shard' not in n]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = torch.Generator().manual_seed(SEED + rank)
    data_inp = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=rng) for _ in range(N_BATCHES)]
    data_tgt = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=rng) for _ in range(N_BATCHES)]

    def step(s):
        bi = s % N_BATCHES
        cur_inp = data_inp[bi].to(dev); cur_tgt = data_tgt[bi].to(dev)
        for pg in opt.param_groups: pg['lr'] = get_lr(s, args.steps)
        logits_local = model(cur_inp)
        loss = ce(logits_local, cur_tgt)
        loss.backward()
        for p in rep:
            if p.grad is not None:
                p.grad.data = xm.all_reduce(xm.REDUCE_SUM, p.grad.data) / ws
        opt.step(); opt.zero_grad()
        xm.mark_step()
        return loss

    if rank == 0: print("[warmup] compiling...")
    for _ in range(args.warmup): step(0)
    if rank == 0: print("[warmup] done")
    xm.rendezvous('pre_measure')
    wall_start = time.time()
    times, losses = [], []
    for s in range(args.steps):
        t0 = time.time()
        loss = step(s); xm.wait_device_ops()
        dt = time.time() - t0
        times.append(dt); losses.append(loss.item())
        if rank == 0 and (s + 1) % 50 == 0:
            avg = sum(times[-50:]) / 50 * 1000
            al = sum(losses[-50:]) / 50
            print(f"  step {s+1:>5}/{args.steps}  loss={al:.4f}  avg={avg:.0f}ms")

    wall = time.time() - wall_start
    if rank == 0:
        avg_ms = sum(times) / len(times) * 1000
        steady = sum(times[200:]) / max(1, len(times) - 200) * 1000 if len(times) > 200 else None
        final = sum(losses[-100:]) / max(1, min(100, len(losses)))
        print(f"\n  Backend     : {args.backend}")
        print(f"  Steps       : {args.steps}")
        print(f"  Wall        : {wall:.2f} s ({wall/60:.1f} min)")
        print(f"  Avg step    : {avg_ms:.1f} ms")
        print(f"  Steady step : {steady:.1f} ms (from step 200)" if steady else "")
        print(f"  Final loss  : {final:.4f}")
        out = os.environ.get('RESULTS_DIR', '/home/ubuntu/agentic-collective-communication/training/results/dxe_1node')
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f'dxe_{args.backend}.json'), 'w') as f:
            json.dump(dict(
                collective='dxe', backend=args.backend, steps=args.steps, world_size=ws,
                wall_s=round(wall, 2), avg_ms=round(avg_ms, 1),
                steady_ms=round(steady, 1) if steady else None,
                final_loss=round(final, 4),
                step_times_ms=[round(t*1000, 1) for t in times],
                losses=[round(l, 4) for l in losses],
            ), f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['baseline', 'agent'], required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--warmup', type=int, default=5)
    run(p.parse_args())
