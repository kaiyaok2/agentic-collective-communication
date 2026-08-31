#!/usr/bin/env python3
"""
~9.7B dense GPT-3-class transformer (pre-LayerNorm, GELU FFN, learned
positional embeddings) with TP=32 (intra-node) x DP=7 (cross-node) at
224 ranks, ALL 7 Sorcar-vs-strat family sites wired into the DP
gradient sync + optimizer path. Second architecture for the 10B e2e
validation (first: Llama-style, train_llama10b_tp_families.py).

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
# default optlevel, NO mid-autograd graph breaks: a mark_step inside
# an autograd.Function (the _GraphBreak below) produces partial-graph
# shapes that trip NCC_ITEN404 MaskPropagation. The unbroken fwd/bwd
# graph compiles fine at default optlevel; only -O1 blew the 5M
# instruction cap. Chunked Adam (below) keeps the optimizer graphs
# small.

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

VOCAB  = 256
DM     = 4096
HEADS  = 64          # 2 heads per core at TP=32 (hd=64);
                     # single-head-per-core shapes trip NCC_ITEN404
LAYERS = 48
FFN    = 16384       # GPT-3 style 4*DM
SEQ    = 128
N_MB   = 4
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


BREAK_EVERY = 2      # layers per compiled graph segment (a 4-layer
                     # segment's backward needs >283GB to compile)


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
class LNorm(nn.Module):
    """LayerNorm with weight AND bias — both replicated across TP,
    both are F4a family-site members."""
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        v = x.float()
        mu = v.mean(-1, keepdim=True)
        var = v.var(-1, keepdim=True, unbiased=False)
        v = (v - mu) * torch.rsqrt(var + 1e-5)
        return (v * self.weight.float() + self.bias.float()).to(x.dtype)


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
        # bool causal mask + masked_fill on 4D multi-head shapes —
        # the exact pattern proven to compile in the 9.4B MoE trainer
        self.register_buffer("mask",
            torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), 1))
    def forward(self, x):
        B, S, _ = x.shape
        x = _TPCopy.apply(x, self.groups)
        qkv = self.qkv(x)                        # (B,S,3*DM/tp)
        d_local = DM // self.tp
        q, k, v = qkv.split(d_local, dim=-1)
        # 4D multi-head attention, mirroring the working MoE trainer
        q = q.view(B, S, self.nh_local, self.hd).transpose(1, 2)
        k = k.view(B, S, self.nh_local, self.hd).transpose(1, 2)
        v = v.view(B, S, self.nh_local, self.hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        att = F.softmax(att.masked_fill(self.mask[:S, :S], -1e4), dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, S, d_local)
        return _TPAllReduce.apply(self.o(y), self.groups)


class TPGeluFFN(nn.Module):
    def __init__(self, tp, tp_groups):
        super().__init__()
        self.groups = tp_groups
        self.fc = nn.Linear(DM, FFN // tp, bias=False)
        self.proj = nn.Linear(FFN // tp, DM, bias=False)
    def forward(self, x):
        x = _TPCopy.apply(x, self.groups)
        return _TPAllReduce.apply(self.proj(F.gelu(self.fc(x))),
                                  self.groups)


class Block(nn.Module):
    def __init__(self, tp, tp_groups):
        super().__init__()
        self.ln1 = LNorm(DM)
        self.attn = TPAttention(tp, tp_groups)
        self.ln2 = LNorm(DM)
        self.mlp = TPGeluFFN(tp, tp_groups)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LM(nn.Module):
    def __init__(self, tp, tp_groups):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.pos = nn.Embedding(SEQ, DM)
        self.blocks = nn.ModuleList(
            [Block(tp, tp_groups) for _ in range(LAYERS)])
        self.ln_f = LNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
        self.head.weight = self.emb.weight
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
    def embed(self, idx):
        S = idx.shape[1]
        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        return self.emb(idx) + self.pos(pos)

    def run_segment(self, x, s0, s1):
        for b in self.blocks[s0:s1]:
            x = b(x)
        return x

    def head_out(self, x):
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
    ap.add_argument('--nmb', type=int, default=None)
    args = ap.parse_args()
    global N_MB
    if args.nmb:
        N_MB = args.nmb
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

    # Build + init on CPU, then move: initializing 305M params on the
    # XLA device creates one giant init graph (the persistent
    # MODULE_105083... whose walrus compile needs >280GB host RAM).
    model = LM(TP, tp_groups).to(torch.bfloat16).to(device)
    xm.mark_step()
    n_local = sum(p.numel() for p in model.parameters())
    # cluster-unique params: shard params count once per TP group,
    # replicated (emb/norms) count once
    shard_params_n, rep_params_n = 0, 0
    for name, p in model.named_parameters():
        if 'ln' in name or name in ('emb.weight', 'pos.weight'):
            rep_params_n += p.numel()
        else:
            shard_params_n += p.numel()
    cluster = rep_params_n + shard_params_n * TP
    log(rank, f'[10bT] backend={backend} ws={ws} TP={TP} DP={DP} '
              f'layers={LAYERS} per-core={n_local/1e6:.1f}M '
              f'cluster={cluster/1e9:.2f}B')

    # partition params over family sites
    emb_param = model.emb.weight
    pos_param = model.pos.weight
    norm_params, qkv0, shard_rest = [], None, []
    for name, p in model.named_parameters():
        if name in ('emb.weight', 'pos.weight'):
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
    _CHUNK = 7 * 2 * 1024 * 1024
    pad = (_CHUNK - f5_numel % _CHUNK) % _CHUNK   # pad to a whole
    # number of chunks so every chunk (incl. last) splits 7-way
    log(rank, f'[10bT] sites: emb(F1)={emb_param.numel()/1e6:.2f}M '
              f'norms(F4a)={len(norm_params)}x{DM} '
              f'qkv0(F7)={qkv0.numel()/1e6:.1f}M '
              f'F4b tensors={len(shard_rest)} '
              f'F5 flat={f5_numel/1e6:.1f}M (+pad {pad})')

    data = load_data(dp_rank, DP)

    # small-param optimizer (emb + norms); shard params via manual Adam
    opt_small = torch.optim.Adam([emb_param, pos_param] + norm_params,
                                 lr=args.lr)
    ADAM_CHUNK = 7 * 2 * 1024 * 1024   # divisible by DP=7; total is
                                       # DP-padded so every chunk incl.
                                       # the last splits evenly 7-way
    _flat_all = torch.cat([p.detach().reshape(-1).float()
                           for p in shard_rest]
                          + [qkv0.detach().reshape(-1).float()]
                          + [torch.zeros(pad, device=device)])
    f5_chunks = [_flat_all[i:i + ADAM_CHUNK].clone()
                 for i in range(0, _flat_all.numel(), ADAM_CHUNK)]
    del _flat_all
    xm.mark_step()
    m_state = v_state = None
    adam_t = 0

    step_times, losses = [], []
    for step in range(args.steps):
        t0 = time.time()
        # ---------- microbatch loop; F1 inline on emb grad -------------
        f1_total = torch.zeros_like(emb_param) if backend == 'baseline' \
            else None
        ddp_monitor = torch.zeros(1, device=device)
        prev = torch.zeros_like(emb_param) if backend == 'baseline' else None
        step_loss = None
        SEG = 2   # layers per compiled fwd/bwd graph (walrus_driver
                  # host memory scales with segment size; 8-layer
                  # segments need >283GB)
        for mb in range(N_MB):
            x, y = get_batch(data, step, mb)
            x, y = x.to(device), y.to(device)
            # --- segmented forward: detach at segment boundaries so
            # each segment's backward is its own XLA graph ---
            seg_bounds = list(range(0, LAYERS, SEG))
            seg_ins, seg_outs = [], []
            h = model.embed(x)
            xm.mark_step()
            for s0 in seg_bounds:
                hin = h.detach().requires_grad_(True)
                hout = model.run_segment(hin, s0, min(s0 + SEG, LAYERS))
                seg_ins.append(hin)
                seg_outs.append(hout)
                h = hout
                xm.mark_step()
            top_in = h.detach().requires_grad_(True)
            logits = model.head_out(top_in)
            loss = F.cross_entropy(logits.view(-1, VOCAB).float(),
                                   y.reshape(-1))
            (loss / N_MB).backward()
            xm.mark_step()
            # --- segmented backward, deepest-first ---
            g = top_in.grad
            for hin, hout in zip(reversed(seg_ins), reversed(seg_outs)):
                torch.autograd.backward(hout, g)
                g = hin.grad
                xm.mark_step()
            # embedding path: h0 = model.embed(x); its grad is g
            h0 = model.embed(x)
            torch.autograd.backward(h0, g)
            xm.mark_step()
            if backend == 'baseline':
                cur = emb_param.grad.detach()
                f1_total = f1_total + xm.all_reduce(
                    xm.REDUCE_SUM, cur - prev, groups=dp_groups) / DP
                prev = cur.clone()
                # textbook-DDP schedule: replicated grads cross the
                # wire on EVERY microbatch (PyTorch DDP default; the
                # accumulate-then-sync alternative IS the F1-family
                # rewrite strat never proposes). Only the final sync
                # feeds the optimizer, so math is identical; per-mb
                # results feed a logged monitor so XLA can't DCE them.
                xm.mark_step()
                outs = f4_flat_sync([p.grad for p in shard_rest],
                                    DP, dp_groups, 'baseline')
                qmb = f7_qkv_sync(qkv0.grad.reshape(-1), DP, dp_groups,
                                  'baseline')
                ddp_monitor = ddp_monitor + \
                    sum(o.sum().float() for o in outs) + \
                    qmb.sum().float()
                del outs, qmb
                xm.mark_step()
            step_loss = loss.detach() if step_loss is None \
                else step_loss + loss.detach()
            # top-level graph cut per microbatch: keeps each compiled
            # fwd+bwd graph within walrus_driver host-memory limits
            # (unbroken 2-mb graph needs >283GB to compile)
            xm.mark_step()
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

        xm.mark_step()   # slice sync graph: norms done
        rest_synced = f4_flat_sync([p.grad for p in shard_rest],
                                   DP, dp_groups, backend)            # F4b
        xm.mark_step()   # slice sync graph: F4b done
        absmax = torch.stack([g.abs().max().float() for g in norm_grads])
        gmax, gmin = f6_clip_stats(absmax, dp_groups, backend)        # F6
        qkv_synced = f7_qkv_sync(qkv0.grad.reshape(-1), DP, dp_groups,
                                 backend)                             # F7

        # ---------------- F5: optimizer on flat shard grads -----------
        # Fully chunk-wise (16M fp32 chunks): no full-size fp32 buffer
        # ever materializes (304.6M x 4B x several temporaries OOMs the
        # 16GB core). Baseline runs Adam on EVERY chunk on EVERY rank
        # (plain DP / replicated optimizer); sorcar runs Adam only on
        # this DP-rank's 1/DP of the chunks (ZeRO-1) and all_gathers
        # each updated chunk. Exact same math.
        xm.mark_step()   # slice sync graph: F6/F7 done
        # build grad CHUNKS without materializing the 1.2GB flat cat
        # (single big fp32 buffer + AR temporaries fragment 16GB HBM)
        grad_srcs = [g.reshape(-1) for g in rest_synced] \
                    + [qkv_synced] + [torch.zeros(pad, device=device)]
        grad_chunks, cur, cur_n = [], [], 0
        for gsrc in grad_srcs:
            goff = 0
            while goff < gsrc.numel():
                take = min(ADAM_CHUNK - cur_n, gsrc.numel() - goff)
                cur.append(gsrc[goff:goff + take])
                cur_n += take
                goff += take
                if cur_n == ADAM_CHUNK:
                    grad_chunks.append(torch.cat(cur).float())
                    cur, cur_n = [], 0
        if cur:
            grad_chunks.append(torch.cat(cur).float())
        del grad_srcs, cur
        xm.mark_step()
        n_chunks = len(f5_chunks)
        if m_state is None:
            if backend == 'baseline':
                # replicated optimizer: every rank holds ALL state
                m_state = [torch.zeros_like(c) for c in f5_chunks]
                v_state = [torch.zeros_like(c) for c in f5_chunks]
            else:
                # ZeRO-1: per-rank state covers 1/DP of each chunk
                m_state = [torch.zeros(c.numel() // DP, device=device)
                           for c in f5_chunks]
                v_state = [torch.zeros_like(t) for t in m_state]
        adam_t += 1
        bc1 = 1 - 0.9 ** adam_t
        bc2 = 1 - 0.999 ** adam_t
        upd_shards = []
        for ci in range(n_chunks):
            g_c = grad_chunks[ci]
            if backend == 'baseline':
                # plain DP: full-size Adam on every rank
                m_state[ci] = m_state[ci] * 0.9 + g_c * 0.1
                v_state[ci] = v_state[ci] * 0.999 + (g_c * g_c) * 0.001
                upd_c = args.lr * (m_state[ci] / bc1) / \
                    ((v_state[ci] / bc2).sqrt() + 1e-8)
                f5_chunks[ci] = f5_chunks[ci] - upd_c
                del upd_c
            else:
                # ZeRO-1 via rank-symmetric primitives (identical HLO
                # on every rank; the shard routing lives inside the
                # collective): reduce_scatter hands each rank its 1/DP
                # slice of the (already DP-synced) grad — scale by
                # 1/DP * DP == pass-through since g_c is the mean grad
                # replicated on all ranks; rs of identical replicas
                # with scale=1/DP returns exactly the local slice.
                # local shard extraction: g_c is the DP-synced mean
                # grad, identical on all ranks. Contiguous slice keyed
                # on dp_rank: 7 HLO variants cluster-wide (1 per host,
                # since all 32 cores of a node share dp_rank) — cheap
                # to compile, and the collective sequence is identical
                # so no MPMD divergence. index_select cost ~1.1s/step.
                sn = g_c.numel() // DP
                g_shard = g_c[dp_rank * sn:(dp_rank + 1) * sn]
                m_state[ci] = m_state[ci] * 0.9 + g_shard * 0.1
                v_state[ci] = v_state[ci] * 0.999 + \
                    (g_shard * g_shard) * 0.001
                upd_shard = args.lr * (m_state[ci] / bc1) / \
                    ((v_state[ci] / bc2).sqrt() + 1e-8)
                upd_shards.append(upd_shard)
                del g_shard
            del g_c
            if (ci + 1) % 4 == 0:
                xm.mark_step()
        if backend == 'sorcar':
            # ONE batched all_gather for all chunk updates (stack ->
            # AG -> unstack): family F4-style dispatch collapse applied
            # to sorcar's own optimizer traffic. n_chunks AGs -> 1.
            stacked = torch.stack(upd_shards, dim=0)   # (n_chunks, sn)
            gathered = xm.all_gather(stacked, dim=1, groups=dp_groups)
            for ci in range(n_chunks):
                f5_chunks[ci] = f5_chunks[ci] - gathered[ci]
            del stacked, gathered
        del grad_chunks, upd_shards
        xm.mark_step()
        # copy-back from chunks into params
        with torch.no_grad():
            off_global = 0
            for pi, p in enumerate(shard_rest + [qkv0]):
                n = p.numel()
                pieces, need, goff = [], n, off_global
                while need > 0:
                    ci = goff // ADAM_CHUNK
                    coff = goff - ci * ADAM_CHUNK
                    take = min(need, f5_chunks[ci].numel() - coff)
                    pieces.append(f5_chunks[ci][coff:coff + take])
                    need -= take
                    goff += take
                p.copy_(torch.cat(pieces).view_as(p).to(p.dtype))
                off_global += n
                if (pi + 1) % 24 == 0:
                    xm.mark_step()
        xm.mark_step()   # slice: copy-back done

        pos_synced = xm.all_reduce(xm.REDUCE_SUM, pos_param.grad.detach(),
                                   groups=dp_groups) / DP
        pos_synced = xm.all_reduce(xm.REDUCE_SUM, pos_synced,
                                   groups=tp_groups) / TP
        with torch.no_grad():
            emb_param.grad.copy_(emb_synced)
            pos_param.grad.copy_(pos_synced)
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
                  f'mon={float(ddp_monitor.item()):.2e} ms={dt:.1f}')

    if rank == 0:
        med = sorted(step_times)[len(step_times) // 2] if step_times else -1
        print('RESULT_JSON ' + json.dumps({
            'arch': 'gpt10b_tp', 'backend': backend, 'ws': ws,
            'tp': TP, 'dp': DP, 'layers': LAYERS,
            'cluster_params_b': cluster / 1e9,
            'median_ms_per_step': med,
            'mean_ms_per_step': sum(step_times) / max(1, len(step_times)),
            'first_loss': losses[0], 'final_loss': losses[-1],
            'losses': losses}), flush=True)


if __name__ == '__main__':
    main()
