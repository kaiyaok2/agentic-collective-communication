#!/usr/bin/env python3
"""
OLMoE-architectural-style MoE training on Trainium across 7 nodes,
with expert-choice routing so the AllToAllV call has uniform per-rank
send/receive counts.

Three pluggable agent backends (each independently selectable):
  --backend  baseline|agent   AllToAllV (MoE expert dispatch + combine)
  --grad-sync baseline|agent  Replicated-gradient AllReduce
  --ce       baseline|agent   Distributed cross-entropy on vocab-sharded
                              logits (only differs when --ce agent)

The agent backends import the 7-node search outputs:
  runtime/trainium_alltoallv_7node.py
  runtime/trainium_grad_ar_7node.py
  runtime/trainium_dxe_7node.py

Architecture: OLMoE-architectural-style with LAYERS=8, DM=2048, HEADS=16,
NEXP=ws=224 (1 expert per rank), TOPK=8, EXDIM=1024, SwiGLU MoE, RoPE,
RMSNorm, expert-choice routing with per-(src,dst) capacity
cap = ceil(SEQLEN * TOPK / NEXP) * EXPANSION (=13 at SEQLEN=256, TOPK=8,
NEXP=224, EXPANSION=1.25). Vocab is sharded along the head:
V_local = VOCAB / ws.

Cluster params: ~11.3 B (replicated 197 M + expert 50 M per rank * 224).
Active params per token: ~0.65 B.
"""
import argparse, os, sys, time, json, math

os.environ.setdefault("NEURON_NUM_RECENT_MODELS_TO_KEEP", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
os.environ.setdefault("NEURON_COMPILE_CACHE_URL", "/tmp/neuron_cache")

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ==================== Phase StepTimer via .item() sync ====================
import time as _t
class _StepTimer:
    def __init__(self):
        self.per_step = []
        self.curr = None
    def begin(self, idx, sync_tensor=None):
        if sync_tensor is not None:
            _ = sync_tensor.float().sum().item()
        self.curr = {"idx": idx}
        self._t0 = _t.time()
    def mark(self, label, sync_tensor):
        _ = sync_tensor.float().sum().item()
        now = _t.time()
        self.curr[label] = now - self._t0
        self._t0 = now
    def end(self):
        if self.curr is not None:
            self.per_step.append(self.curr)
            self.curr = None
STEP_TIMER = _StepTimer()

# ==========================================================================


VOCAB     = 32256      # = 224 * 144; divisible by ws so dxe shards cleanly
DM        = 2048
HEADS     = 16
HDIM      = DM // HEADS
LAYERS    = 8
TOPK      = 8          # average target expert calls per token
EXDIM     = 1024
SEQLEN    = 256
BSZ       = 1
SEED      = 42
EXPANSION = 1.0       # expert-choice per-(src,dst) buffer slack
N_BATCHES = 500


def make_baseline_alltoallv(ws):
    """Canonical AG + reshape + RS-SUM with scale=1/ws (2 dispatches)."""
    def fn(x, mc):
        gathered = xm.all_gather(x.unsqueeze(0), dim=0)
        reshaped = gathered.view(ws, ws, mc)
        transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
        return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                                 scale=1.0 / ws, scatter_dim=0, shard_count=ws)
    return fn


def load_agent_alltoallv(ws):
    """Load the agent's alltoallv runtime, preferring the 7-node search output."""
    try:
        import runtime.trainium_alltoallv_7node as _mod
        if hasattr(_mod, "init_alltoallv"):
            _mod.init_alltoallv()
        fn = _mod.alltoallv
        print(f"  using 7-node agent alltoallv runtime")
    except ModuleNotFoundError:
        import runtime.trainium_alltoallv as _mod
        if hasattr(_mod, "init_alltoallv"):
            _mod.init_alltoallv()
        fn = _mod.alltoallv
        print(f"  using 1-node agent alltoallv runtime (fallback)")
    return lambda x, mc: fn(x, ws, mc)


class _A2AV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mc, fn):
        ctx.mc, ctx.fn = mc, fn
        xm.mark_step()
        out = fn(x, mc)
        xm.mark_step()
        return out
    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        out = ctx.fn(g.contiguous(), ctx.mc)
        xm.mark_step()
        return out, None, None


def precompute_rope(seqlen, hdim, device):
    half = hdim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seqlen, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype)
        return x * v * self.w


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DM, 3 * DM, bias=False)
        self.o = nn.Linear(DM, DM, bias=False)
        self.register_buffer("mask",
                              torch.triu(torch.ones(SEQLEN, SEQLEN, dtype=torch.bool), 1))
    def forward(self, x, cos, sin):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, HEADS, HDIM).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        a = (q @ k.transpose(-2, -1)) * (HDIM ** -0.5)
        a = F.softmax(a.masked_fill(self.mask[:S, :S], -1e4), dim=-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, S, D))


class SwiGLUExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_gate = nn.Linear(DM, EXDIM, bias=False)
        self.w_up   = nn.Linear(DM, EXDIM, bias=False)
        self.w_down = nn.Linear(EXDIM, DM, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoEBlock(nn.Module):
    """Expert-choice MoE: each (src rank, dst expert) carries cap tokens —
    the top-cap local tokens by that expert's gate score. With NEXP=ws,
    every rank owns one expert, and per-rank send/receive counts are
    fixed at ws*cap*D elements regardless of routing skew."""
    def __init__(self, ws, cap, mc, a2av_fn):
        super().__init__()
        self.ws, self.cap, self.mc = ws, cap, mc
        self.gate = nn.Linear(DM, ws, bias=False)
        self.expert = SwiGLUExpert()
        self.a2av_fn = a2av_fn

    def forward(self, x):
        B, S, D = x.shape
        T = B * S
        xf = x.reshape(T, D)
        gp = F.softmax(self.gate(xf), dim=-1)              # (T, NEXP=ws)
        gp_t = gp.transpose(0, 1).contiguous()              # (ws, T)
        scores, token_idx = gp_t.topk(self.cap, dim=1)      # each (ws, cap)
        # Gather the chosen tokens for each destination expert.
        sel = xf[token_idx]                                  # (ws, cap, D)
        send = sel.reshape(-1)
        xm.mark_step()
        recv = _A2AV.apply(send, self.mc, self.a2av_fn)     # (ws*cap*D,)
        # Local expert processes this rank's ws*cap incoming tokens.
        proc = self.expert(recv.view(self.ws * self.cap, D))
        xm.mark_step()
        combined = _A2AV.apply(proc.reshape(-1), self.mc, self.a2av_fn)
        # combined is back in (dest, slot, dim) layout matching `send`.
        weighted = combined.view(self.ws, self.cap, D) * scores.unsqueeze(-1)
        # Scatter-add each (dest, slot) result back to its source token.
        out = torch.zeros(T, D, device=xf.device, dtype=xf.dtype)
        out = out.index_add(0, token_idx.reshape(-1), weighted.reshape(-1, D))
        return out.reshape(B, S, D)


class Block(nn.Module):
    def __init__(self, ws, cap, mc, a2av_fn):
        super().__init__()
        self.n1 = RMSNorm(DM)
        self.attn = Attn()
        self.n2 = RMSNorm(DM)
        self.moe = MoEBlock(ws, cap, mc, a2av_fn)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.moe(self.n2(x))
        return x


class Model(nn.Module):
    def __init__(self, ws, cap, mc, a2av_fn, v_local):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.layers = nn.ModuleList(
            [Block(ws, cap, mc, a2av_fn) for _ in range(LAYERS)])
        self.norm = RMSNorm(DM)
        # Sharded head: each rank owns V_local = VOCAB/ws of the vocab.
        self.head_shard = nn.Linear(DM, v_local, bias=False)
    def forward(self, ids, cos, sin):
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.head_shard(self.norm(x))


def make_loss_fn(args, ws, v_local):
    """Return a (logits_local, targets) -> scalar loss closure."""
    if args.ce == "baseline":
        # Reference: all-gather full logits then F.cross_entropy.
        def baseline_ce(logits_local, targets):
            # logits_local: (B, S, V_local) -> (B*S, V_local)
            ll = logits_local.reshape(-1, v_local).contiguous()
            gathered = xm.all_gather(ll, dim=1)             # (B*S, V)
            return F.cross_entropy(gathered, targets.reshape(-1))
        return baseline_ce
    else:
        from runtime.trainium_dxe_7node import dxe_loss, init_dxe
        init_dxe()
        def agent_ce(logits_local, targets):
            ll = logits_local.reshape(-1, v_local).contiguous()
            return dxe_loss(ll, targets.reshape(-1), v_local)
        return agent_ce


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
    cos, sin = precompute_rope(SEQLEN, HDIM, dev)
    cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)

    # Expert-choice per-(src,dst) capacity. Uniform per-rank receive shape.
    cap = max(1, int(math.ceil(math.ceil(SEQLEN * TOPK / ws) * EXPANSION)))
    mc = cap * DM
    assert VOCAB % ws == 0, f"VOCAB={VOCAB} must be divisible by ws={ws} for dxe"
    v_local = VOCAB // ws

    a2av_fn = make_baseline_alltoallv(ws) if args.backend == "baseline" \
              else load_agent_alltoallv(ws)

    def _grad_sync_baseline(rep_params, ws):
        for p in rep_params:
            if p.grad is not None:
                p.grad.data = xm.all_reduce(xm.REDUCE_SUM, p.grad.data) / ws

    def _grad_sync_agent_factory():
        from runtime.trainium_grad_ar_7node import grad_ar_sync, init_grad_ar
        init_grad_ar()
        return lambda rep_params, ws: grad_ar_sync(rep_params, ws)

    _grad_sync = _grad_sync_baseline if args.grad_sync == "baseline" \
                 else _grad_sync_agent_factory()

    loss_fn = make_loss_fn(args, ws, v_local)

    torch.manual_seed(SEED)
    model = Model(ws, cap, mc, a2av_fn, v_local).to(dev).to(torch.bfloat16)
    # Replicated params: everything except local expert and sharded head.
    rep_params = [p for n, p in model.named_parameters()
                  if ".expert." not in n and "head_shard" not in n]

    if rank == 0:
        n_total = sum(p.numel() for p in model.parameters())
        n_rep = sum(p.numel() for p in rep_params)
        n_sharded = n_total - n_rep
        cluster_total = n_rep + n_sharded * ws
        print(f"[init] backend={args.backend} grad_sync={args.grad_sync} ce={args.ce}")
        print(f"[init] ws={ws} steps={args.steps}")
        print(f"[init] LAYERS={LAYERS} DM={DM} HEADS={HEADS} TOPK={TOPK}"
              f" EXDIM={EXDIM} NEXP={ws} VOCAB={VOCAB} v_local={v_local}")
        print(f"[init] expert-choice cap={cap} (ceil(SEQLEN*TOPK/ws)*{EXPANSION})"
              f" mc={mc} elements/(src,dst)")
        print(f"[init] per-rank: rep={n_rep/1e6:.1f}M  sharded={n_sharded/1e6:.1f}M"
              f"  total/rank={n_total/1e6:.1f}M"
              f"  cluster_total={cluster_total/1e9:.2f}B")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    g = torch.Generator().manual_seed(SEED + rank)
    data_inp = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=g) for _ in range(N_BATCHES)]
    data_tgt = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=g) for _ in range(N_BATCHES)]

    def step(s):
        bi = s % N_BATCHES
        cur_inp = data_inp[bi].to(dev)
        cur_tgt = data_tgt[bi].to(dev)
        lr = get_lr(s, args.steps)
        for pg in opt.param_groups: pg["lr"] = lr
        STEP_TIMER.begin(s)
        logits_local = model(cur_inp, cos, sin)
        STEP_TIMER.mark("fwd", logits_local)
        loss = loss_fn(logits_local, cur_tgt)
        STEP_TIMER.mark("ce", loss)
        loss.backward()
        # Backward graph affects rep param grads; use one as sync probe.
        STEP_TIMER.mark("bwd", rep_params[0].grad)
        _grad_sync(rep_params, ws)
        STEP_TIMER.mark("grad_ar", rep_params[0].grad)
        opt.step(); opt.zero_grad()
        xm.mark_step()
        STEP_TIMER.end()
        return loss

    if rank == 0: print("[warmup] compiling XLA graphs...")
    for _ in range(args.warmup): step(0)
    if rank == 0: print("[warmup] done")

    xm.rendezvous("pre_measure")
    wall_start = time.time()
    times, losses = [], []
    log_interval = 50
    for s in range(args.steps):
        t0 = time.time()
        loss = step(s)
        xm.wait_device_ops()
        dt = time.time() - t0
        times.append(dt)
        losses.append(loss.item())
        if rank == 0 and (s + 1) % log_interval == 0:
            avg = sum(times[-log_interval:]) / log_interval * 1000
            finite = [l for l in losses[-log_interval:] if l == l]
            al = sum(finite) / len(finite) if finite else float("nan")
            print(f"  step {s+1:>4}/{args.steps}  loss={al:.4f}  avg={avg:.0f}ms")

    wall = time.time() - wall_start
    if rank == 0:
        avg_ms = sum(times) / len(times) * 1000
        finite_losses = [l for l in losses if l == l]
        final = sum(finite_losses[-100:]) / max(1, min(100, len(finite_losses))) \
                if finite_losses else float("nan")
        steady_avg = round(sum(times[200:]) / max(1, len(times) - 200) * 1000, 1) \
                     if len(times) > 200 else None
        print("\n" + "=" * 65)
        print(f"  Backend       : {args.backend}")
        print(f"  Grad sync     : {args.grad_sync}")
        print(f"  CE            : {args.ce}")
        print(f"  Steps         : {args.steps}")
        print(f"  Wall clock    : {wall:.2f} s  ({wall/60:.1f} min)")
        print(f"  Avg step      : {avg_ms:.1f} ms")
        print(f"  Steady step   : {steady_avg} ms (from step 200)")
        print(f"  Final loss    : {final:.4f}  ({len(finite_losses)}/{len(losses)} finite)")
        print("=" * 65)
        out = os.environ.get("RESULTS_DIR", "/tmp")
        os.makedirs(out, exist_ok=True)
        tag = f"a2av-{args.backend}_gs-{args.grad_sync}_ce-{args.ce}"
        path = os.path.join(out, f"olmoe10b_{tag}.json")
        with open(path, "w") as f:
            json.dump(dict(
                backend=args.backend, grad_sync=args.grad_sync, ce=args.ce,
                steps=args.steps, world_size=ws,
                arch=dict(DM=DM, HEADS=HEADS, LAYERS=LAYERS, TOPK=TOPK,
                          EXDIM=EXDIM, NEXP=ws, VOCAB=VOCAB,
                          SEQLEN=SEQLEN, BSZ=BSZ, cap=cap, expansion=EXPANSION,
                          routing="expert-choice"),
                wall_clock_s=round(wall, 2),
                avg_step_ms=round(avg_ms, 1),
                steady_avg_step_ms=steady_avg,
                final_loss=None if final != final else round(final, 4),
                finite_loss_count=len(finite_losses),
                losses=[round(l, 4) if l == l else None for l in losses],
                step_times_ms=[round(t * 1000, 1) for t in times],
            ), f, indent=2)
        print(f"  Saved -> {path}")
        tpath = path.replace('.json', '_phase.json')
        with open(tpath, 'w') as f:
            json.dump(dict(per_step=STEP_TIMER.per_step), f)
        print(f"  Phase timer -> {tpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["baseline", "agent"], required=True,
                   help="AllToAllV backend for MoE dispatch/combine")
    p.add_argument("--grad-sync", choices=["baseline", "agent"], default="baseline",
                   help="Replicated-gradient AllReduce backend")
    p.add_argument("--ce", choices=["baseline", "agent"], default="baseline",
                   help="Distributed cross-entropy backend over vocab-sharded logits")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=5)
    run(p.parse_args())
