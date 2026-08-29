#!/usr/bin/env python3
"""
OLMoE-10B-class communication-heavy MoE training, 7-node 224-rank, with
ALL 7 Sorcar-vs-strat optimization families wired into the natural
communication paths of the training step.

Architecture (same class as training/train_olmoe10b.py, the OverlayCCL
e2e model): DM=2048, HEADS=16, LAYERS=8, expert-choice MoE with
NEXP=ws=224 (1 SwiGLU expert per rank, EXDIM=1024), RoPE, RMSNorm,
vocab-sharded head (VOCAB=32256, V_local=144). Cluster params ~11.3B
(replicated ~197M + 50M expert × 224 ranks). bf16.

Communication per step (what makes this comm-heavy):
  - 2 AllToAllV (AG+RS) per MoE layer × 8 layers × fwd+bwd
  - replicated-grad sync of ~131M elements (the family sites below)
  - vocab-sharded CE all-gather
  - metric/checksum collectives

Backends (--backend):
  baseline : strat-enumerate outcome at every family site (per-tensor /
             per-loop collective schedule).
  sorcar   : Sorcar family-general rewrite at each site. Mathematically
             exact → loss must track within bf16 noise.

Family → site mapping (all on the replicated-parameter gradient sync,
the dominant DP communication at this scale):
  F1 dispatch linearity : embedding-table grad (66.0M) accumulated over
                          N_MB=2 microbatches.
       baseline: AR(g_mb1) + AR(g_mb2); sorcar: AR(g_mb1 + g_mb2).
  F2 CSE                : step-loss metric consumed by logger, LR
                          controller, anomaly detector.
       baseline: 3× AR(loss); sorcar: 1 AR reused.
  F3 algebraic zero     : 8-AR telescoping sync checksum (≡ 0).
       baseline: 8 dispatches; sorcar: zeros_like.
  F4 dispatch collapse  : (a) 17 RMSNorm weight grads (2/layer + final),
                          per-tensor AR vs stack→1 AR; (b) the remaining
                          replicated attn/gate grads (~52M in 23
                          tensors), per-tensor AR loop (the paper's
                          grad_ar baseline) vs flat-concat → 1 AR →
                          split (the paper's grad_ar agent solution).
  F5 collective narrowing: attn o-proj grads across layers (33.6M flat).
       baseline: AR full + every rank runs full-size Adam (plain DP).
       sorcar  : reduce_scatter → 1/224-shard Adam → all_gather(update)
                 (ZeRO-1).
  F6 mixed-reduce       : grad-clip diagnostics: global max & min over
                          per-tensor |g|max vector.
       baseline: per-entry AR_MAX + AR_MIN; sorcar: stacked, 1+1.
  F7 slab fusion        : layer-0 fused-QKV grad (12.6M) synced as 8
                          contiguous slabs vs 1 AR + views.

The MoE AllToAllV itself uses the canonical AG+RS in BOTH backends (it
is one of the paper's original 8 problems, not one of the 7 post-paper
families — keeping it fixed isolates the family effect).

Data: wikitext-103-raw bytes (values 0..255 inside the 32256 vocab),
disjoint per-rank stripes, deterministic across backends.
"""
import argparse, os, sys, time, json, math

os.environ.setdefault("NEURON_NUM_RECENT_MODELS_TO_KEEP", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
os.environ.setdefault("NEURON_COMPILE_CACHE_URL", "/tmp/neuron_cache")
# -O1: the full 8-layer MoE step graph OOMs neuronx-cc at default optlevel
# when several modules compile concurrently on one host ([F137]).
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + " --optlevel=1"

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

VOCAB   = 16128      # divisible by 224 and by 32 (1-node smoke)
DM      = 2048
HEADS   = 16
HDIM    = DM // HEADS
LAYERS  = 7
TOPK    = 8
EXDIM   = 960
SEQLEN  = 192
BSZ     = 1
N_MB    = 2          # F1 microbatches
SEED    = 42
N_CHECKSUM = 8       # F3
N_QKV_SLABS = 8      # F7


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------- model
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


def make_a2av(ws):
    """Canonical AG+RS AllToAllV — identical in both backends."""
    def fn(x, mc):
        gathered = xm.all_gather(x.unsqueeze(0), dim=0)
        reshaped = gathered.view(ws, ws, mc)
        transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
        return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                                 scale=1.0 / ws, scatter_dim=0, shard_count=ws)
    return fn


class _CEGather(torch.autograd.Function):
    """all_gather along vocab dim with a correct backward (Neuron
    xm.all_gather's autograd silently drops gradients). The backward
    extracts the local shard via a one-hot mask TENSOR rather than a
    Python-int slice — slicing by `rank` bakes a per-rank constant into
    the HLO, producing 32 distinct graphs per node and 32 simultaneous
    neuronx-cc compiles (host OOM, [F137])."""
    @staticmethod
    def forward(ctx, ll, shard_mask, v_local):
        ctx.v_local = v_local
        ctx.save_for_backward(shard_mask)
        return xm.all_gather(ll, dim=1)
    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors            # (ws,) one-hot at rank
        v = ctx.v_local
        T = g.shape[0]
        gv = g.view(T, -1, v)                  # (T, ws, v)
        local = (gv * mask.view(1, -1, 1).to(g.dtype)).sum(dim=1)
        return local.contiguous(), None, None


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


class MoEBlock(nn.Module):
    """Expert-choice MoE — uniform per-(src,dst) capacity, 1 expert/rank."""
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
        gp = F.softmax(self.gate(xf), dim=-1)
        gp_t = gp.transpose(0, 1).contiguous()
        scores, token_idx = gp_t.topk(self.cap, dim=1)
        sel = xf[token_idx]
        send = sel.reshape(-1)
        xm.mark_step()
        recv = _A2AV.apply(send, self.mc, self.a2av_fn)
        proc = self.expert(recv.view(self.ws * self.cap, D))
        xm.mark_step()
        combined = _A2AV.apply(proc.reshape(-1), self.mc, self.a2av_fn)
        weighted = combined.view(self.ws, self.cap, D) * scores.unsqueeze(-1)
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
        self.head_shard = nn.Linear(DM, v_local, bias=False)
    def forward(self, ids, cos, sin):
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.head_shard(self.norm(x))


# ------------------------------------------------- family-site helpers
def f2_loss_metrics(loss_vec, ws, backend):
    if backend == 'baseline':
        return (xm.all_reduce(xm.REDUCE_SUM, loss_vec) / ws,
                xm.all_reduce(xm.REDUCE_SUM, loss_vec) / ws,
                xm.all_reduce(xm.REDUCE_SUM, loss_vec) / ws)
    m = xm.all_reduce(xm.REDUCE_SUM, loss_vec) / ws
    return m, m, m


def f3_checksum(probe_vec, backend):
    if backend == 'baseline':
        acc = None
        for i in range(N_CHECKSUM):
            r = xm.all_reduce(xm.REDUCE_SUM, probe_vec) * ((-1.0) ** i)
            acc = r if acc is None else acc + r
        return acc
    return torch.zeros_like(probe_vec)


def f4_norm_sync(norm_grads, ws, backend):
    if backend == 'baseline':
        return [xm.all_reduce(xm.REDUCE_SUM, g) / ws for g in norm_grads]
    stacked = torch.stack(norm_grads, dim=0)
    red = xm.all_reduce(xm.REDUCE_SUM, stacked) / ws
    return [red[i] for i in range(red.shape[0])]


F4B_BUCKET = 16 * 1024 * 1024   # 16M elements ≈ 32MB bf16 (grad_ar v6)


def f4_flat_sync(grads, ws, backend):
    """grad_ar site: per-tensor AR loop (13 dispatches) vs 32MB-bucketed
    concat + AR + split (5 dispatches at 78.7M elements). Unbucketed
    concat (1 dispatch) OOMs device HBM at this scale — same finding as
    the production grad_ar v6 runtime."""
    if backend == 'baseline':
        return [xm.all_reduce(xm.REDUCE_SUM, g) / ws for g in grads]
    outs = [None] * len(grads)
    bucket, idxs, acc = [], [], 0
    def flush():
        nonlocal bucket, idxs, acc
        if not bucket:
            return
        flat = torch.cat([g.reshape(-1) for g in bucket])
        red = xm.all_reduce(xm.REDUCE_SUM, flat) / ws
        off = 0
        for i, g in zip(idxs, bucket):
            n = g.numel()
            outs[i] = red[off:off + n].view_as(g)
            off += n
        bucket, idxs, acc = [], [], 0
    for i, g in enumerate(grads):
        if acc + g.numel() > F4B_BUCKET and bucket:
            flush()
        bucket.append(g)
        idxs.append(i)
        acc += g.numel()
    flush()
    return outs


def f5_grad_sync(flat_grad, ws, backend):
    if backend == 'baseline':
        return xm.all_reduce(xm.REDUCE_SUM, flat_grad) / ws
    return xm.reduce_scatter(xm.REDUCE_SUM, flat_grad, scale=1.0 / ws,
                             scatter_dim=0, shard_count=ws)


def f6_clip_stats(absmax_vec, ws, backend):
    if backend == 'baseline':
        gmax = [xm.all_reduce(xm.REDUCE_MAX, absmax_vec[i])
                for i in range(absmax_vec.shape[0])]
        gmin = [xm.all_reduce(xm.REDUCE_MIN, absmax_vec[i])
                for i in range(absmax_vec.shape[0])]
        return torch.stack(gmax).max(), torch.stack(gmin).min()
    gmax = xm.all_reduce(xm.REDUCE_MAX, absmax_vec)
    gmin = xm.all_reduce(xm.REDUCE_MIN, absmax_vec)
    return gmax.max(), gmin.min()


def f7_qkv_sync(qkv_flat, ws, backend):
    n = qkv_flat.numel() // N_QKV_SLABS
    if backend == 'baseline':
        parts = [xm.all_reduce(xm.REDUCE_SUM, qkv_flat[i * n:(i + 1) * n]) / ws
                 for i in range(N_QKV_SLABS)]
        return torch.cat(parts, dim=0)
    return xm.all_reduce(xm.REDUCE_SUM, qkv_flat) / ws


# ------------------------------------------------------------------ data
def load_data(rank, ws, path='/home/ubuntu/wiki.train.raw'):
    with open(path, 'rb') as f:
        raw = f.read()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    stripe = data.numel() // ws
    return data[rank * stripe:(rank + 1) * stripe]


def get_batch(data, step, mb):
    n_tok = SEQLEN + 1
    per_step = N_MB * BSZ * n_tok
    off = (step * per_step + mb * BSZ * n_tok) % (data.numel() - per_step - 1)
    chunk = data[off:off + n_tok]
    return chunk[:SEQLEN].view(1, SEQLEN), chunk[1:SEQLEN + 1].view(1, SEQLEN)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['baseline', 'sorcar'], required=True)
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--warmup', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--seqlen', type=int, default=None,
                    help='override SEQLEN (short-seq = grad-sync-dominated regime)')
    args = ap.parse_args()
    global SEQLEN
    if args.seqlen:
        SEQLEN = args.seqlen
    backend = args.backend

    dist.init_process_group('xla', init_method='xla://')
    rank = xr.global_ordinal()
    ws = xr.world_size()
    device = torch_xla.device()
    torch.manual_seed(args.seed)

    assert VOCAB % ws == 0
    v_local = VOCAB // ws
    cap = max(1, int(math.ceil(SEQLEN * TOPK / ws)))
    mc = cap * DM
    a2av_fn = make_a2av(ws)

    cos, sin = precompute_rope(SEQLEN, HDIM, device)
    cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
    shard_mask = torch.zeros(ws, device=device)
    shard_mask[rank] = 1.0

    model = Model(ws, cap, mc, a2av_fn, v_local).to(device).to(torch.bfloat16)

    # ---------------- parameter partition over family sites ----------
    emb_param = model.emb.weight                       # F1
    norm_params, qkv0, oproj_params, rest_rep = [], None, [], []
    expert_or_head = []
    for name, p in model.named_parameters():
        if '.expert.' in name or 'head_shard' in name:
            expert_or_head.append(p)                   # rank-local, no sync
        elif name == 'emb.weight':
            pass
        elif '.n1.' in name or '.n2.' in name or name.startswith('norm.'):
            norm_params.append(p)                      # F4a
        elif name == 'layers.0.attn.qkv.weight':
            qkv0 = p                                   # F7
        elif '.attn.o.' in name and int(name.split('.')[1]) < 4:
            oproj_params.append(p)                     # F5 (layers 0-3;
            # full group's fp32 Adam chain exceeds HBM at 224 ranks)
        else:
            rest_rep.append(p)                         # F4b (grad_ar site)

    n_total = sum(p.numel() for p in model.parameters())
    n_local = sum(p.numel() for p in expert_or_head)
    n_rep = n_total - n_local
    cluster = n_rep + n_local * ws
    log(rank, f'[10b] backend={backend} ws={ws} rep={n_rep/1e6:.1f}M '
              f'local/rank={n_local/1e6:.1f}M cluster={cluster/1e9:.2f}B '
              f'cap={cap} mc={mc}')
    log(rank, f'[10b] sites: emb(F1)={emb_param.numel()/1e6:.1f}M '
              f'norms(F4a)={sum(p.numel() for p in norm_params)} '
              f'qkv0(F7)={qkv0.numel()/1e6:.1f}M '
              f'oproj(F5)={sum(p.numel() for p in oproj_params)/1e6:.1f}M '
              f'rest(F4b)={sum(p.numel() for p in rest_rep)/1e6:.1f}M '
              f'in {len(rest_rep)} tensors')

    data = load_data(rank, ws)

    # Optimizers: AdamW for everything except the F5 group (manual Adam,
    # sharded in sorcar / replicated in baseline).
    opt_params = ([emb_param] + norm_params + [qkv0] + rest_rep
                  + expert_or_head)
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=0.01)

    f5_numel = sum(p.numel() for p in oproj_params)
    pad = (ws - f5_numel % ws) % ws
    f5_flat = torch.cat([p.detach().reshape(-1).float() for p in oproj_params]
                        + [torch.zeros(pad, device=device)])
    m_state = v_state = None
    adam_t = 0

    step_times, losses = [], []
    for step in range(args.steps):
        t0 = time.time()
        # ---------- microbatched forward/backward (F1 site inline) ----
        # baseline (strat): AR each microbatch's grad increment as it
        #   appears — N_MB dispatches on the 66M emb grad.
        # sorcar: no per-mb work; AR once on the accumulated grad after
        #   the loop (linearity: sum AR(g_mb) == AR(sum g_mb)).
        f1_total = torch.zeros_like(emb_param) if backend == 'baseline' \
            else None
        prev = torch.zeros_like(emb_param) if backend == 'baseline' \
            else None
        step_loss = None
        for mb in range(N_MB):
            x, y = get_batch(data, step, mb)
            x, y = x.to(device), y.to(device)
            logits_local = model(x, cos, sin)          # (B,S,v_local)
            full = _CEGather.apply(
                logits_local.reshape(-1, v_local).contiguous(),
                shard_mask, v_local)
            loss = F.cross_entropy(full.float(), y.reshape(-1))
            (loss / N_MB).backward()
            if backend == 'baseline':
                cur = emb_param.grad.detach()
                f1_total = f1_total + \
                    xm.all_reduce(xm.REDUCE_SUM, cur - prev) / ws
                prev = cur.clone()
            step_loss = loss.detach() if step_loss is None \
                else step_loss + loss.detach()
        step_loss = step_loss / N_MB

        # ---------------- family-site synchronization -----------------
        xm.mark_step()   # split sync graph from fwd/bwd graph (compile size)
        if backend == 'baseline':
            emb_synced = f1_total                                     # F1
            del f1_total, prev
        else:
            emb_synced = xm.all_reduce(
                xm.REDUCE_SUM, emb_param.grad.detach()) / ws          # F1
        lv = step_loss.reshape(1)
        m_log, m_sched, m_anom = f2_loss_metrics(lv, ws, backend)    # F2
        chk = f3_checksum(lv, backend)                                # F3
        norm_synced = f4_norm_sync([p.grad for p in norm_params],
                                   ws, backend)                       # F4a
        rest_synced = f4_flat_sync([p.grad for p in rest_rep],
                                   ws, backend)                       # F4b
        absmax = torch.stack([g.abs().max().float() for g in
                              [p.grad for p in norm_params]])
        gmax, gmin = f6_clip_stats(absmax, ws, backend)               # F6
        qkv_synced = f7_qkv_sync(qkv0.grad.reshape(-1), ws, backend)  # F7

        f5_grad = torch.cat([p.grad.reshape(-1).float() for p in oproj_params]
                            + [torch.zeros(pad, device=device)])
        gsync = f5_grad_sync(f5_grad, ws, backend)                    # F5
        if m_state is None:
            m_state = torch.zeros_like(gsync)
            v_state = torch.zeros_like(gsync)
        adam_t += 1
        m_state.mul_(0.9).add_(gsync, alpha=0.1)
        v_state.mul_(0.999).addcmul_(gsync, gsync, value=0.001)
        mhat = m_state / (1 - 0.9 ** adam_t)
        vhat = v_state / (1 - 0.999 ** adam_t)
        upd = args.lr * mhat / (vhat.sqrt() + 1e-8)
        del mhat, vhat, gsync, f5_grad
        if backend == 'sorcar':
            upd = xm.all_gather(upd, dim=0)
        f5_flat = f5_flat - upd
        del upd
        off = 0
        with torch.no_grad():
            for p in oproj_params:
                n = p.numel()
                p.copy_(f5_flat[off:off + n].view_as(p).to(p.dtype))
                off += n

        with torch.no_grad():
            emb_param.grad.copy_(emb_synced)
            for p, g in zip(norm_params, norm_synced):
                p.grad.copy_(g)
            for p, g in zip(rest_rep, rest_synced):
                p.grad.copy_(g)
            qkv0.grad.copy_(qkv_synced.view_as(qkv0.grad))
            for p in oproj_params:
                p.grad.zero_()
        opt.step()
        model.zero_grad(set_to_none=False)
        xm.mark_step()

        loss_host = float(m_log.item())
        chk_host = float(chk.sum().item())
        dt = (time.time() - t0) * 1000.0
        if step >= args.warmup:
            step_times.append(dt)
        losses.append(loss_host)
        log(rank, f'[10b] step {step:3d} loss={loss_host:.4f} '
                  f'chk={chk_host:.2e} gmax={float(gmax.item()):.3e} '
                  f'ms={dt:.1f}')

    if rank == 0:
        med = sorted(step_times)[len(step_times) // 2] if step_times else -1
        print('RESULT_JSON ' + json.dumps({
            'arch': 'olmoe10b', 'backend': backend, 'ws': ws,
            'cluster_params_b': cluster / 1e9,
            'median_ms_per_step': med,
            'mean_ms_per_step': sum(step_times) / max(1, len(step_times)),
            'first_loss': losses[0], 'final_loss': losses[-1],
            'losses': losses}), flush=True)


if __name__ == '__main__':
    main()
