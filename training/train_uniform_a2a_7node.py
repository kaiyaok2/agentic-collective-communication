#!/usr/bin/env python3
"""
7-node 224-rank training comparison for Uniform AllToAll.

Smaller-shape than the 1-node script (train_uniform_a2a.py) so the
AG+T+RS gathered buffer (ws^2 * cap * D bf16) fits Trainium HBM
together with the model itself.

Shape: LAYERS=6, DM=2048, NEXP=224 (=ws), TOPK=4, EXDIM=512,
SEQLEN=128, BSZ=1, VOCAB=32256. cap = ceil(128*4/224) = 3 ->
chunk_size = 3*2048 = 6144 elements -> gathered = 224^2 * 6144
~= 308M bf16 elements ~= 615 MB per dispatch.
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


VOCAB, DM, HEADS = 32256, 2048, 16
LAYERS, NEXP, TOPK, EXDIM = 6, 224, 4, 512
SEQLEN, BSZ, SEED, N_BATCHES = 128, 1, 42, 500


def _ua2a_baseline(x, ws, chunk_size):
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    reshaped = gathered.view(ws, ws, chunk_size)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                             scale=1.0 / ws, scatter_dim=0, shard_count=ws)


def _load_agent(ws):
    from runtime.trainium_uniform_a2a import uniform_a2a, init_uniform_a2a
    init_uniform_a2a()
    return lambda x, ws_, cs: uniform_a2a(x, cs)


import time as _t
class _UA2ATimer:
    def __init__(self):
        self.curr = None
        self.per_step = []
    def begin_step(self, idx):
        if self.curr is not None:
            self.per_step.append(self.curr)
        self.curr = {'idx': idx, 'ua2a_fwd_ms': 0.0, 'ua2a_bwd_ms': 0.0, 'fwd_calls': 0, 'bwd_calls': 0}
    def record(self, direction, dt_s):
        if self.curr is None:
            self.curr = {'idx': -1, 'ua2a_fwd_ms': 0.0, 'ua2a_bwd_ms': 0.0, 'fwd_calls': 0, 'bwd_calls': 0}
        if direction == 'fwd':
            self.curr['ua2a_fwd_ms'] += dt_s * 1000.0
            self.curr['fwd_calls'] += 1
        else:
            self.curr['ua2a_bwd_ms'] += dt_s * 1000.0
            self.curr['bwd_calls'] += 1
    def finalize(self):
        if self.curr is not None:
            self.per_step.append(self.curr)
            self.curr = None
_UA2A_TIMER = _UA2ATimer()


class _UA2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ws, chunk_size, fn):
        ctx.ws, ctx.chunk_size, ctx.fn = ws, chunk_size, fn
        xm.mark_step()
        _t0 = _t.time()
        out = fn(x, ws, chunk_size)
        _ = out.sum().item()
        _UA2A_TIMER.record('fwd', _t.time() - _t0)
        xm.mark_step()
        return out
    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        _t0 = _t.time()
        out = ctx.fn(g.contiguous(), ctx.ws, ctx.chunk_size)
        _ = out.sum().item()
        _UA2A_TIMER.record('bwd', _t.time() - _t0)
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
    def __init__(self, d, ne, k, ed, fn, ws, cap):
        super().__init__()
        self.d, self.k, self.ne, self.ws, self.fn, self.cap = d, k, ne, ws, fn, cap
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
        recv = _UA2A.apply(send, self.ws, chunk_size, self.fn)
        processed = self.expert(recv.view(self.ws * self.cap, D))
        xm.mark_step()
        combined = _UA2A.apply(processed.reshape(-1), self.ws, chunk_size, self.fn)
        combined = combined[:T * self.k * D].reshape(T * self.k, D)
        return (combined.reshape(T, self.k, D) * ew.unsqueeze(-1)).sum(1).reshape(B, S, D)


class Block(nn.Module):
    def __init__(self, d, h, ne, k, ed, fn, ws, cap):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), Attn(d, h)
        self.n2, self.moe = RMSNorm(d), MoE(d, ne, k, ed, fn, ws, cap)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.moe(self.n2(x))
        return x


class MoEModel(nn.Module):
    def __init__(self, fn, ws, cap):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.layers = nn.ModuleList(
            [Block(DM, HEADS, NEXP, TOPK, EXDIM, fn, ws, cap) for _ in range(LAYERS)])
        self.norm = RMSNorm(DM); self.head = nn.Linear(DM, VOCAB, bias=False)
    def forward(self, ids):
        x = self.emb(ids)
        for L in self.layers: x = L(x)
        return self.head(self.norm(x))


def get_lr(step, total, max_lr=3e-4, min_lr=3e-5, warmup=100):
    if step < warmup: return max_lr * step / warmup
    r = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * r))


def run(args):
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()
    T = BSZ * SEQLEN
    cap = (T * TOPK + ws - 1) // ws

    fn = _load_agent(ws) if args.backend == 'agent' else _ua2a_baseline

    if rank == 0:
        chunk_size = cap * DM
        print(f"[init] backend={args.backend} ws={ws} steps={args.steps}")
        print(f"[init] cap={cap} chunk_size={chunk_size} gathered={ws*ws*chunk_size*2/1e6:.1f} MB bf16")

    torch.manual_seed(SEED)
    model = MoEModel(fn, ws, cap).to(dev).to(torch.bfloat16)
    rep = [p for n, p in model.named_parameters() if '.expert.' not in n]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = torch.Generator().manual_seed(SEED + rank)
    data_inp = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=rng) for _ in range(N_BATCHES)]
    data_tgt = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=rng) for _ in range(N_BATCHES)]

    def step(s):
        _UA2A_TIMER.begin_step(s)
        bi = s % N_BATCHES
        cur_inp = data_inp[bi].to(dev); cur_tgt = data_tgt[bi].to(dev)
        for pg in opt.param_groups: pg['lr'] = get_lr(s, args.steps)
        logits = model(cur_inp)
        loss = F.cross_entropy(logits.view(-1, VOCAB), cur_tgt.view(-1))
        loss.backward()
        for p in rep:
            if p.grad is not None:
                p.grad.data = xm.all_reduce(xm.REDUCE_SUM, p.grad.data) / ws
        opt.step(); opt.zero_grad()
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
        print(f"  Wall        : {wall:.2f} s")
        print(f"  Avg step    : {avg_ms:.1f} ms")
        print(f"  Steady step : {steady:.1f} ms (from step 200)" if steady else "")
        print(f"  Final loss  : {final:.4f}")
        out = os.environ.get('RESULTS_DIR', '/home/ubuntu/agentic-collective-communication/training/results/ua2a_7node')
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f'ua2a_7node_{args.backend}.json'), 'w') as f:
            json.dump(dict(
                collective='uniform_a2a', backend=args.backend, steps=args.steps, world_size=ws,
                wall_s=round(wall, 2), avg_ms=round(avg_ms, 1),
                steady_ms=round(steady, 1) if steady else None,
                final_loss=round(final, 4),
                step_times_ms=[round(t*1000, 1) for t in times],
                losses=[round(l, 4) for l in losses],
            ), f, indent=2)
        _UA2A_TIMER.finalize()
        tpath = os.path.join(out, f'ua2a_7node_{args.backend}_ua2a_phase.json')
        with open(tpath, 'w') as f:
            json.dump(dict(per_step=_UA2A_TIMER.per_step), f)
        print(f'  UA2A per-call timer -> {tpath}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['baseline', 'agent'], required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--warmup', type=int, default=5)
    run(p.parse_args())
