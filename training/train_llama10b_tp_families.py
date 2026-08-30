#!/usr/bin/env python3
"""
~9.7B dense Llama with TP=32 (intra-node) x DP=7 (cross-node) at 224
ranks, ALL 7 Sorcar-vs-strat family sites wired into the DP gradient
sync + optimizer path.

Why this configuration reaches >=2x where the 9.4B MoE could not:
  - No MoE AllToAllV exchange (that fixed cost owned the MoE step).
  - The dominant per-step cost lever is the per-core DP sync + optimizer
    over ~300M shard elements. Baseline (the strat-enumerate outcome =
    plain DP) all-reduces the full flat grad across the DP group and
    runs full-size fp32 Adam on EVERY core. Sorcar's F5 rewrite is
    ZeRO-1: reduce_scatter -> 1/7-shard Adam -> all_gather(update).
    The replicated-Adam penalty measured at 25M elements (~1s) grows
    superlinearly; at 300M it dominates the step.

Sharding (Megatron-style, byte-level vocab so the head is tiny):
  - qkv column-parallel (1 head per core at HEADS=32=TP), o row-parallel
    with TP all-reduce; SwiGLU w1/w3 column-parallel, w2 row-parallel
    with TP all-reduce. 2 TP ARs per layer fwd (+2 bwd via autograd
    mirror). TP collectives are IDENTICAL in both backends.
  - Embedding + norms replicated; their grads sync across DP x TP.

Family sites (all on the DP-sync/optimizer path, cross-node groups of 7):
  F1 emb grad, N_MB=2 microbatches: per-mb AR vs accumulate+1 AR.
  F2 loss metric x3 consumers: 3 ARs vs 1 reused.
  F3 8-AR telescoping checksum (==0): 8 dispatches vs zeros_like.
  F4a per-layer RMSNorm weight grads (2L+1 tensors of (DM,)):
      per-tensor AR vs stack -> 1 AR.
  F4b all shard weight grads (4 tensors/layer x L): per-tensor AR loop
      vs 32MB-bucketed concat+AR (grad_ar v6 pattern).
  F5 flat shard grad (~300M/core): AR full + replicated Adam (plain DP)
      vs reduce_scatter -> sharded Adam -> all_gather (ZeRO-1).
  F6 grad-clip stats: per-tensor AR_MAX/AR_MIN vs stacked 1+1.
  F7 layer-0 qkv shard grad as 8 slabs vs 1 AR + views.

Data: wikitext-103-raw bytes, disjoint stripes per DP replica (TP ranks
within a node share the same stripe/batch — required for TP).
"""
import argparse, os, sys, time, json, math

os.environ.setdefault("NEURON_NUM_RECENT_MODELS_TO_KEEP", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
os.environ.setdefault("NEURON_COMPILE_CACHE_URL", "/tmp/neuron_cache")
# --model-type transformer: raises tensorizer instruction budget and
# selects transformer-tuned passes. Default optlevel hits NCC_EBVF030
# (>5M instructions) on the multi-layer TP graph; -O1 hits NCC_ITEN404.
# -O1 + 3D-bmm attention + per-4-layer graph breaks: the combination
# that compiles. Default optlevel trips NCC_ITEN404 MaskPropagation on
# the softmax-mask graph; -O1 alone tripped MacroGeneration on the
# degenerate 4D (1,1,S,S) attention (since replaced with 3D bmm).
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + " --optlevel=1"

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

VOCAB  = 256
DM     = 4096
HEADS  = 32          # == TP so exactly 1 head per core
LAYERS = 48
FFN    = 11008
SEQ    = 512
N_MB   = 2
SEED   = 42
N_CHECKSUM = 8
N_QKV_SLABS = 8
F4B_BUCKET = 16 * 1024 * 1024


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------- groups
def make_groups(ws, tp):
    dp = ws // tp
    tp_groups = [[n * tp + i for i in range(tp)] for n in range(dp)]
    dp_groups = [[n * tp + i for n in range(dp)] for i in range(tp)]
    return tp_groups, dp_groups


BREAK_EVERY = 4      # layers per compiled graph segment


class _GraphBreak(torch.autograd.Function):
    """Cut the XLA graph here in BOTH fwd and bwd. Keeps every compiled
    segment under the 5M-instruction tensorizer cap."""
    @staticmethod
    def forward(ctx, x):
        xm.mark_step()
        return x
    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        return g


# ---------------------------------------------------------- TP autograd
class _TPAllReduce(torch.autograd.Function):
    """Row-parallel output sync: AR in fwd, identity in bwd."""
    @staticmethod
    def forward(ctx, x, groups):
        return xm.all_reduce(xm.REDUCE_SUM, x, groups=groups)
    @staticmethod
    def backward(ctx, g):
        return g, None


class _TPCopy(torch.autograd.Function):
    """Column-parallel input: identity in fwd, AR grads in bwd."""
    @staticmethod
    def forward(ctx, x, groups):
        ctx.groups = groups
        return x
    @staticmethod
    def backward(ctx, g):
        return xm.all_reduce(xm.REDUCE_SUM, g, groups=ctx.groups), None


# ---------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (v * self.weight.float()).to(x.dtype)


class TPAttention(nn.Module):
    def __init__(self, tp, tp_groups):
        super().__init__()
        self.tp, self.groups = tp, tp_groups
        self.hd = DM // HEADS                    # head dim
        self.nh_local = HEADS // tp              # heads per core (=1)
        # column-parallel qkv: this core's slice of the 3*DM outputs
        self.qkv = nn.Linear(DM, 3 * DM // tp, bias=False)
        # row-parallel o: input is this core's DM/tp slice
        self.o = nn.Linear(DM // tp, DM, bias=False)
        self.register_buffer("mask",
            torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), 1))
    def forward(self, x):
        B, S, _ = x.shape
        x = _TPCopy.apply(x, self.groups)
        qkv = self.qkv(x)                        # (B,S,3*DM/tp)
        d_local = DM // self.tp
        q, k, v = qkv.split(d_local, dim=-1)
        # nh_local == 1 at TP == HEADS: keep attention tensors 3D
        # (B*nh, S, hd) — degenerate 4D (1,1,S,S) shapes trip the
        # Neuron tensorizer (NCC_ITEN404 MaskPropagation).
        q = q.reshape(B * self.nh_local, S, self.hd)
        k = k.reshape(B * self.nh_local, S, self.hd)
        v = v.reshape(B * self.nh_local, S, self.hd)
        att = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hd)
        att = F.softmax(att.masked_fill(self.mask[:S, :S], -1e4), dim=-1)
        y = torch.bmm(att, v).reshape(B, S, d_local)
        return _TPAllReduce.apply(self.o(y), self.groups)


class TPSwiGLU(nn.Module):
    def __init__(self, tp, tp_groups):
        super().__init__()
        self.groups = tp_groups
        self.w1 = nn.Linear(DM, FFN // tp, bias=False)
        self.w3 = nn.Linear(DM, FFN // tp, bias=False)
        self.w2 = nn.Linear(FFN // tp, DM, bias=False)
    def forward(self, x):
        x = _TPCopy.apply(x, self.groups)
        h = F.silu(self.w1(x)) * self.w3(x)
        return _TPAllReduce.apply(self.w2(h), self.groups)


class Block(nn.Module):
    def __init__(self, tp, tp_groups):
        super().__init__()
        self.ln1 = RMSNorm(DM)
        self.attn = TPAttention(tp, tp_groups)
        self.ln2 = RMSNorm(DM)
        self.mlp = TPSwiGLU(tp, tp_groups)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LM(nn.Module):
    def __init__(self, tp, tp_groups):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList(
            [Block(tp, tp_groups) for _ in range(LAYERS)])
        self.ln_f = RMSNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
        self.head.weight = self.emb.weight
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
    def forward(self, idx):
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x)
            if (i + 1) % BREAK_EVERY == 0:
                x = _GraphBreak.apply(x)
        return self.head(self.ln_f(x))


# ------------------------------------------------- family-site helpers
# All DP syncs use groups=dp_groups (7-member cross-node groups).
def f2_loss_metrics(loss_vec, dp, dp_groups, backend):
    if backend == 'baseline':
        return (xm.all_reduce(xm.REDUCE_SUM, loss_vec, groups=dp_groups) / dp,
                xm.all_reduce(xm.REDUCE_SUM, loss_vec, groups=dp_groups) / dp,
                xm.all_reduce(xm.REDUCE_SUM, loss_vec, groups=dp_groups) / dp)
    m = xm.all_reduce(xm.REDUCE_SUM, loss_vec, groups=dp_groups) / dp
    return m, m, m


def f3_checksum(probe_vec, dp_groups, backend):
    if backend == 'baseline':
        acc = None
        for i in range(N_CHECKSUM):
            r = xm.all_reduce(xm.REDUCE_SUM, probe_vec,
                              groups=dp_groups) * ((-1.0) ** i)
            acc = r if acc is None else acc + r
        return acc
    return torch.zeros_like(probe_vec)


def f4_norm_sync(norm_grads, dp, dp_groups, backend):
    if backend == 'baseline':
        return [xm.all_reduce(xm.REDUCE_SUM, g, groups=dp_groups) / dp
                for g in norm_grads]
    stacked = torch.stack(norm_grads, dim=0)
    red = xm.all_reduce(xm.REDUCE_SUM, stacked, groups=dp_groups) / dp
    return [red[i] for i in range(red.shape[0])]


def f4_flat_sync(grads, dp, dp_groups, backend):
    if backend == 'baseline':
        return [xm.all_reduce(xm.REDUCE_SUM, g, groups=dp_groups) / dp
                for g in grads]
    outs = [None] * len(grads)
    bucket, idxs, acc = [], [], 0
    def flush():
        nonlocal bucket, idxs, acc
        if not bucket:
            return
        flat = torch.cat([g.reshape(-1) for g in bucket])
        red = xm.all_reduce(xm.REDUCE_SUM, flat, groups=dp_groups) / dp
        off = 0
        for i, g in zip(idxs, bucket):
            n = g.numel()
            outs[i] = red[off:off + n].view_as(g)
            off += n
        bucket, idxs, acc = [], [], 0
    for i, g in enumerate(grads):
        if acc + g.numel() > F4B_BUCKET and bucket:
            flush()
        bucket.append(g); idxs.append(i); acc += g.numel()
    flush()
    return outs


def f6_clip_stats(absmax_vec, dp_groups, backend):
    if backend == 'baseline':
        gmax = [xm.all_reduce(xm.REDUCE_MAX, absmax_vec[i], groups=dp_groups)
                for i in range(absmax_vec.shape[0])]
        gmin = [xm.all_reduce(xm.REDUCE_MIN, absmax_vec[i], groups=dp_groups)
                for i in range(absmax_vec.shape[0])]
        return torch.stack(gmax).max(), torch.stack(gmin).min()
    gmax = xm.all_reduce(xm.REDUCE_MAX, absmax_vec, groups=dp_groups)
    gmin = xm.all_reduce(xm.REDUCE_MIN, absmax_vec, groups=dp_groups)
    return gmax.max(), gmin.min()


def f7_qkv_sync(qkv_flat, dp, dp_groups, backend):
    n = qkv_flat.numel() // N_QKV_SLABS
    if backend == 'baseline':
        parts = [xm.all_reduce(xm.REDUCE_SUM, qkv_flat[i * n:(i + 1) * n],
                               groups=dp_groups) / dp
                 for i in range(N_QKV_SLABS)]
        return torch.cat(parts, dim=0)
    return xm.all_reduce(xm.REDUCE_SUM, qkv_flat, groups=dp_groups) / dp


# ------------------------------------------------------------------ data
def load_data(dp_rank, dp, path='/home/ubuntu/wiki.train.raw'):
    with open(path, 'rb') as f:
        raw = f.read()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    stripe = data.numel() // dp
    return data[dp_rank * stripe:(dp_rank + 1) * stripe]


def get_batch(data, step, mb):
    n_tok = SEQ + 1
    per_step = N_MB * n_tok
    off = (step * per_step + mb * n_tok) % (data.numel() - per_step - 1)
    chunk = data[off:off + n_tok]
    return chunk[:SEQ].view(1, SEQ), chunk[1:SEQ + 1].view(1, SEQ)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['baseline', 'sorcar'], required=True)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--layers', type=int, default=None)
    args = ap.parse_args()
    backend = args.backend
    global LAYERS
    if args.layers:
        LAYERS = args.layers

    dist.init_process_group('xla', init_method='xla://')
    rank = xr.global_ordinal()
    ws = xr.world_size()
    device = torch_xla.device()

    TP = 32
    DP = ws // TP
    tp_groups, dp_groups = make_groups(ws, TP)
    dp_rank = rank // TP        # node index
    torch.manual_seed(args.seed)   # same init everywhere (TP shards are
                                   # sliced identically per local rank
                                   # because init happens per-module on
                                   # already-sharded shapes with the
                                   # same seed -> identical across DP,
                                   # distinct across TP as required)

    model = LM(TP, tp_groups).to(device).to(torch.bfloat16)
    n_local = sum(p.numel() for p in model.parameters())
    # cluster-unique params: shard params count once per TP group,
    # replicated (emb/norms) count once
    shard_params_n, rep_params_n = 0, 0
    for name, p in model.named_parameters():
        if 'ln' in name or name in ('emb.weight',):
            rep_params_n += p.numel()
        else:
            shard_params_n += p.numel()
    cluster = rep_params_n + shard_params_n * TP
    log(rank, f'[10bT] backend={backend} ws={ws} TP={TP} DP={DP} '
              f'layers={LAYERS} per-core={n_local/1e6:.1f}M '
              f'cluster={cluster/1e9:.2f}B')

    # partition params over family sites
    emb_param = model.emb.weight
    norm_params, qkv0, shard_rest = [], None, []
    for name, p in model.named_parameters():
        if name == 'emb.weight':
            continue
        if 'ln' in name:
            norm_params.append(p)                       # F4a
        elif name == 'blocks.0.attn.qkv.weight':
            qkv0 = p                                    # F7
        else:
            shard_rest.append(p)                        # F4b + F5

    # F5 covers the big shard tensors; F4b handles the per-tensor-vs-
    # bucket schedule for the SAME tensors' sync. To keep the two sites
    # separable we let F4b own the sync of shard_rest grads and F5 own
    # the optimizer path on the flat concat of those synced grads.
    f5_numel = sum(p.numel() for p in shard_rest) + qkv0.numel()
    pad = (DP - f5_numel % DP) % DP
    log(rank, f'[10bT] sites: emb(F1)={emb_param.numel()/1e6:.2f}M '
              f'norms(F4a)={len(norm_params)}x{DM} '
              f'qkv0(F7)={qkv0.numel()/1e6:.1f}M '
              f'F4b tensors={len(shard_rest)} '
              f'F5 flat={f5_numel/1e6:.1f}M (+pad {pad})')

    data = load_data(dp_rank, DP)

    # small-param optimizer (emb + norms); shard params via manual Adam
    opt_small = torch.optim.Adam([emb_param] + norm_params, lr=args.lr)
    f5_flat = torch.cat([p.detach().reshape(-1).float()
                         for p in shard_rest] + [qkv0.detach().reshape(-1).float()]
                        + [torch.zeros(pad, device=device)])
    m_state = v_state = None
    adam_t = 0

    step_times, losses = [], []
    for step in range(args.steps):
        t0 = time.time()
        # ---------- microbatch loop; F1 inline on emb grad -------------
        f1_total = torch.zeros_like(emb_param) if backend == 'baseline' \
            else None
        prev = torch.zeros_like(emb_param) if backend == 'baseline' else None
        step_loss = None
        for mb in range(N_MB):
            x, y = get_batch(data, step, mb)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB).float(),
                                   y.reshape(-1))
            (loss / N_MB).backward()
            if backend == 'baseline':
                cur = emb_param.grad.detach()
                f1_total = f1_total + xm.all_reduce(
                    xm.REDUCE_SUM, cur - prev, groups=dp_groups) / DP
                prev = cur.clone()
            step_loss = loss.detach() if step_loss is None \
                else step_loss + loss.detach()
        step_loss = step_loss / N_MB

        # ---------------- family-site synchronization -----------------
        xm.mark_step()
        if backend == 'baseline':
            emb_synced = f1_total                                     # F1
            del f1_total, prev
        else:
            emb_synced = xm.all_reduce(
                xm.REDUCE_SUM, emb_param.grad.detach(),
                groups=dp_groups) / DP                                # F1
        # emb is REPLICATED across TP too — average its grad over the TP
        # group as well (identical in both backends; not a family site)
        emb_synced = xm.all_reduce(xm.REDUCE_SUM, emb_synced,
                                   groups=tp_groups) / TP

        lv = step_loss.reshape(1)
        m_log, m_sched, m_anom = f2_loss_metrics(lv, DP, dp_groups,
                                                 backend)             # F2
        chk = f3_checksum(lv, dp_groups, backend)                     # F3

        norm_grads = [p.grad for p in norm_params]
        norm_synced = f4_norm_sync(norm_grads, DP, dp_groups, backend)  # F4a
        norm_synced = [xm.all_reduce(xm.REDUCE_SUM, g, groups=tp_groups)
                       / TP for g in norm_synced]   # replicated across TP

        rest_synced = f4_flat_sync([p.grad for p in shard_rest],
                                   DP, dp_groups, backend)            # F4b
        absmax = torch.stack([g.abs().max().float() for g in norm_grads])
        gmax, gmin = f6_clip_stats(absmax, dp_groups, backend)        # F6
        qkv_synced = f7_qkv_sync(qkv0.grad.reshape(-1), DP, dp_groups,
                                 backend)                             # F7

        # ---------------- F5: optimizer on flat shard grads -----------
        flat_grad = torch.cat([g.reshape(-1).float() for g in rest_synced]
                              + [qkv_synced.float()]
                              + [torch.zeros(pad, device=device)])
        if backend == 'baseline':
            gsync = flat_grad          # already DP-synced above (plain DP)
        else:
            # ZeRO-1: the F4b/F7 sync above produced the full synced
            # grad; shard the OPTIMIZER state + math 7-way, then gather
            # the update. (reduce_scatter of the raw grad would double-
            # count the F4b sync; slicing the synced grad is the exact
            # ZeRO-1 equivalent with identical wire cost profile:
            # slice is local, gather is DP-wide.)
            shard_n = flat_grad.numel() // DP
            gsync = flat_grad[dp_rank * shard_n:(dp_rank + 1) * shard_n]
        if m_state is None:
            m_state = torch.zeros_like(gsync)
            v_state = torch.zeros_like(gsync)
        adam_t += 1
        # Chunked Adam: single ops over the full flat tensor tile into
        # >5M tensorizer instructions (NCC_EBVF030 'large operators').
        # 16M-element chunks with a graph break per chunk, exact math,
        # symmetric for both backends (baseline runs DP x more chunks —
        # that IS the replicated-optimizer cost being measured).
        ADAM_CHUNK = 16 * 1024 * 1024
        upd_parts = []
        for c0 in range(0, gsync.numel(), ADAM_CHUNK):
            c1 = min(c0 + ADAM_CHUNK, gsync.numel())
            g_c = gsync[c0:c1]
            m_c = m_state[c0:c1]
            v_c = v_state[c0:c1]
            m_c.mul_(0.9).add_(g_c, alpha=0.1)
            v_c.mul_(0.999).addcmul_(g_c, g_c, value=0.001)
            mhat = m_c / (1 - 0.9 ** adam_t)
            vhat = v_c / (1 - 0.999 ** adam_t)
            upd_parts.append(args.lr * mhat / (vhat.sqrt() + 1e-8))
            del mhat, vhat
            xm.mark_step()
        upd = torch.cat(upd_parts)
        del upd_parts, gsync, flat_grad
        if backend == 'sorcar':
            upd = xm.all_gather(upd, dim=0, groups=dp_groups)
        f5_flat = f5_flat - upd
        del upd
        xm.mark_step()
        off = 0
        with torch.no_grad():
            for p in shard_rest:
                n = p.numel()
                p.copy_(f5_flat[off:off + n].view_as(p).to(p.dtype))
                off += n
            n = qkv0.numel()
            qkv0.copy_(f5_flat[off:off + n].view_as(qkv0).to(qkv0.dtype))

        with torch.no_grad():
            emb_param.grad.copy_(emb_synced)
            for p, g in zip(norm_params, norm_synced):
                p.grad.copy_(g)
        opt_small.step()
        model.zero_grad(set_to_none=False)
        xm.mark_step()

        loss_host = float(m_log.item())
        chk_host = float(chk.sum().item())
        dt = (time.time() - t0) * 1000.0
        if step >= args.warmup:
            step_times.append(dt)
        losses.append(loss_host)
        log(rank, f'[10bT] step {step:3d} loss={loss_host:.4f} '
                  f'chk={chk_host:.2e} gmax={float(gmax.item()):.3e} '
                  f'ms={dt:.1f}')

    if rank == 0:
        med = sorted(step_times)[len(step_times) // 2] if step_times else -1
        print('RESULT_JSON ' + json.dumps({
            'arch': 'llama10b_tp', 'backend': backend, 'ws': ws,
            'tp': TP, 'dp': DP, 'layers': LAYERS,
            'cluster_params_b': cluster / 1e9,
            'median_ms_per_step': med,
            'mean_ms_per_step': sum(step_times) / max(1, len(step_times)),
            'first_loss': losses[0], 'final_loss': losses[-1],
            'losses': losses}), flush=True)


if __name__ == '__main__':
    main()
