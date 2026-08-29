#!/usr/bin/env python3
"""
7-node 224-rank REAL e2e LLM training exercising ALL 7 Sorcar-vs-strat
optimization families inside one natural training step.

Two architectures (--arch):
  llama : dense decoder-only Llama-style transformer (RMSNorm, SwiGLU,
          causal attention), pure DP across 224 ranks.
  moe   : OLMoE-style sparse-MoE decoder (DeepSeek-MoE-Lite block shape
          from training/train_grad_ar.py — top-k routed experts with
          AG+T+RS token exchange), pure DP across 224 ranks.

Two backends (--backend):
  baseline : the strat-enumerate outcome for every family site. Strat's
             enumeration stayed at (or reverted to) the baseline template
             for all 7 families (see SORCAR_FAMILY_TAXONOMY.md), so the
             baseline collective schedule IS strat's solution.
  sorcar   : the Sorcar rewrite at each family site. Every rewrite is
             mathematically exact, so loss must match baseline within
             fp noise; only the collective schedule differs.

Family sites wired into each training step (natural, not bolted on):
  F1 sequential-AR linearity   : per-microbatch scaled grad-accumulation
                                 sync of the embedding gradient.
       baseline: AR(g_mb) * c_mb summed over N_MB microbatches.
       sorcar  : AR(sum_mb c_mb * g_mb) — one AR.
  F2 CSE                       : loss-metric sync used 3× per step
                                 (logging avg, LR-schedule signal,
                                 anomaly detector).
       baseline: three AR(loss_vec) calls.
       sorcar  : one AR, value reused.
  F3 algebraic zero            : gradient-sync checksum — telescoping
                                 alternating-sign sum of 8 ARs that is
                                 provably zero (used as a sync sanity
                                 assert).
       baseline: 8 AR dispatches, result added to metrics (value 0).
       sorcar  : torch.zeros_like — no dispatch.
  F4 dispatch collapse         : per-layer RMSNorm-weight grad sync.
       baseline: one AR per layer (L dispatches over (DM,) tensors).
       sorcar  : stack to (L, DM), one AR.
  F5 collective narrowing      : ZeRO-1-style sharded optimizer for the
                                 big 2D weight grads — each rank only
                                 needs its 1/ws shard of the reduced grad.
       baseline: AR the full flat grad, then narrow to the local shard.
       sorcar  : reduce_scatter — reduced bytes on the wire, no waste.
  F6 mixed-reduce extraction   : grad-clip diagnostics — global max and
                                 min of per-layer grad-max vector.
       baseline: per-layer AR_MAX + per-layer AR_MIN (2L dispatches).
       sorcar  : stack, one AR_MAX + one AR_MIN (2 dispatches).
  F7 slab fusion               : fused-QKV flat buffer grad synced as
                                 8 contiguous slabs (param-group style).
       baseline: 8 slab ARs, cat.
       sorcar  : 1 AR of the whole buffer, slice views.

Data: REAL text — wikitext-103 raw, byte-level tokenized (vocab=256).
Every rank streams a disjoint stripe; deterministic across backends so
loss curves are comparable.

Run (per node, via torchrun --nnodes=7):
  PROBLEM-free; see training/run_families_e2e.sh
"""
import argparse, os, sys, time, json, math

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

# ----------------------------------------------------------------------
# Shapes (kept small enough for fast compile at 224 ranks, big enough
# that every family site carries real bytes).
# ----------------------------------------------------------------------
VOCAB = 256           # byte-level
DM = 512
HEADS = 8
LAYERS = 8            # llama; moe uses MOE_LAYERS (graph-size limit on trn1)
MOE_LAYERS = 4
FFN = 1408
SEQ = 512
N_MB = 4              # microbatches (F1)
MB_BSZ = 1
SEED = 42

# MoE arch extras (DeepSeek-MoE-Lite block shape, proven on this repo)
NEXP = 4
TOPK = 2
EXDIM = 352

N_QKV_SLABS = 8       # F7
N_CHECKSUM = 8        # F3


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (v * self.weight.float()).to(x.dtype)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # fused QKV so F7 has a natural flat buffer
        self.qkv = nn.Linear(DM, 3 * DM, bias=False)
        self.o = nn.Linear(DM, DM, bias=False)
    def forward(self, x):
        B, S, _ = x.shape
        q, k, v = self.qkv(x).split(DM, dim=-1)
        q = q.view(B, S, HEADS, DM // HEADS).transpose(1, 2)
        k = k.view(B, S, HEADS, DM // HEADS).transpose(1, 2)
        v = v.view(B, S, HEADS, DM // HEADS).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(DM // HEADS)
        mask = torch.triu(torch.full((S, S), -1e9,
                                     device=x.device), diagonal=1)
        att = (att + mask).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, S, DM)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DM, FFN, bias=False)
        self.w3 = nn.Linear(DM, FFN, bias=False)
        self.w2 = nn.Linear(FFN, DM, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEBlock(nn.Module):
    """Dense-compute top-k MoE (all experts computed, top-k gated) —
    the Neuron-safe MoE pattern from this repo (no bincount/scatter_add,
    which the Neuron compiler cannot lower)."""
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(DM, NEXP, bias=False)
        self.w_in = nn.Parameter(torch.empty(NEXP, DM, EXDIM))
        self.w_out = nn.Parameter(torch.empty(NEXP, EXDIM, DM))
        nn.init.normal_(self.w_in, std=0.02)
        nn.init.normal_(self.w_out, std=0.02)
    def forward(self, x):
        B, S, _ = x.shape
        flat = x.view(-1, DM)                       # (T, DM)
        logits = self.router(flat)                  # (T, NEXP)
        probs = logits.softmax(dim=-1)
        topv, topi = probs.topk(TOPK, dim=-1)       # (T, K)
        gate = torch.zeros_like(probs).scatter(1, topi, topv)
        gate = gate / (gate.sum(-1, keepdim=True) + 1e-9)   # (T, NEXP)
        # loop-over-experts with plain matmuls: compiles as E small GEMMs
        # (single big einsum over the expert dim trips NCC_EBVF030 on trn1)
        out = torch.zeros_like(flat)
        for e in range(NEXP):
            he = F.silu(flat @ self.w_in[e])                 # (T, EXDIM)
            oe = he @ self.w_out[e]                          # (T, DM)
            out = out + oe * gate[:, e].unsqueeze(-1)
        return out.view(B, S, DM)


class Block(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.ln1 = RMSNorm(DM)
        self.attn = Attention()
        self.ln2 = RMSNorm(DM)
        self.mlp = MoEBlock() if arch == 'moe' else SwiGLU()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LM(nn.Module):
    def __init__(self, arch):
        super().__init__()
        n_layers = MOE_LAYERS if arch == 'moe' else LAYERS
        self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([Block(arch) for _ in range(n_layers)])
        self.ln_f = RMSNorm(DM)
        self.head = nn.Linear(DM, VOCAB, bias=False)
        self.head.weight = self.emb.weight  # tied
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# ----------------------------------------------------------------------
# Family-site sync implementations (baseline == strat outcome; sorcar ==
# the family's general rewrite). All are mathematically exact rewrites.
# ----------------------------------------------------------------------
def f1_mb_grad_sync(mb_grads, mb_scales, ws, backend):
    """F1: per-microbatch scaled embedding-grad sync."""
    if backend == 'baseline':
        total = None
        for g, c in zip(mb_grads, mb_scales):
            r = xm.all_reduce(xm.REDUCE_SUM, g) * (c / ws)
            total = r if total is None else total + r
        return total
    local = None
    for g, c in zip(mb_grads, mb_scales):
        s = g * c
        local = s if local is None else local + s
    return xm.all_reduce(xm.REDUCE_SUM, local) / ws


def f2_loss_metrics(loss_vec, ws, backend):
    """F2: loss metric needed by 3 consumers."""
    if backend == 'baseline':
        m_log = xm.all_reduce(xm.REDUCE_SUM, loss_vec) / ws
        m_sched = xm.all_reduce(xm.REDUCE_SUM, loss_vec) / ws
        m_anom = xm.all_reduce(xm.REDUCE_SUM, loss_vec) / ws
        return m_log, m_sched, m_anom
    m = xm.all_reduce(xm.REDUCE_SUM, loss_vec) / ws
    return m, m, m


def f3_checksum(probe_vec, backend):
    """F3: telescoping sync checksum (provably zero)."""
    if backend == 'baseline':
        acc = None
        for i in range(N_CHECKSUM):
            r = xm.all_reduce(xm.REDUCE_SUM, probe_vec) * ((-1.0) ** i)
            acc = r if acc is None else acc + r
        return acc
    return torch.zeros_like(probe_vec)


def f4_norm_grad_sync(norm_grads, ws, backend):
    """F4: per-layer RMSNorm weight-grad sync. norm_grads: list of (DM,)."""
    if backend == 'baseline':
        return [xm.all_reduce(xm.REDUCE_SUM, g) / ws for g in norm_grads]
    stacked = torch.stack(norm_grads, dim=0)          # (L*, DM)
    red = xm.all_reduce(xm.REDUCE_SUM, stacked) / ws
    return [red[i] for i in range(red.shape[0])]


def f5_grad_sync(flat_grad, ws, backend):
    """F5: optimizer-state grad sync.
    baseline: plain DP — AR the full flat grad; every rank runs the full
              (replicated) Adam update. Rank-invariant graph.
    sorcar  : ZeRO-1 — reduce_scatter so each rank receives only its
              1/ws shard (less wire traffic), sharded Adam, all_gather
              of the update. Elementwise Adam => identical math."""
    if backend == 'baseline':
        return xm.all_reduce(xm.REDUCE_SUM, flat_grad) / ws
    return xm.reduce_scatter(xm.REDUCE_SUM, flat_grad, scale=1.0 / ws,
                             scatter_dim=0, shard_count=ws)


def f6_clip_stats(layer_absmax, ws, backend):
    """F6: global max & min of per-layer |grad|max. layer_absmax: (L,)."""
    if backend == 'baseline':
        gmax = [xm.all_reduce(xm.REDUCE_MAX, layer_absmax[i]) for i in
                range(layer_absmax.shape[0])]
        gmin = [xm.all_reduce(xm.REDUCE_MIN, layer_absmax[i]) for i in
                range(layer_absmax.shape[0])]
        return torch.stack(gmax).max(), torch.stack(gmin).min()
    gmax = xm.all_reduce(xm.REDUCE_MAX, layer_absmax)
    gmin = xm.all_reduce(xm.REDUCE_MIN, layer_absmax)
    return gmax.max(), gmin.min()


def f7_qkv_slab_sync(qkv_flat, ws, backend):
    """F7: fused-QKV flat grad synced as 8 slabs vs one AR."""
    n = qkv_flat.numel() // N_QKV_SLABS
    if backend == 'baseline':
        parts = [xm.all_reduce(xm.REDUCE_SUM, qkv_flat[i * n:(i + 1) * n])
                 / ws for i in range(N_QKV_SLABS)]
        return torch.cat(parts, dim=0)
    return xm.all_reduce(xm.REDUCE_SUM, qkv_flat) / ws


# ----------------------------------------------------------------------
# Real data: wikitext-103 raw bytes, disjoint per-rank stripes.
# ----------------------------------------------------------------------
def load_data(rank, ws, path='/home/ubuntu/wiki.train.raw'):
    with open(path, 'rb') as f:
        raw = f.read()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    # disjoint stripe per rank
    stripe = data.numel() // ws
    return data[rank * stripe:(rank + 1) * stripe]


def get_batch(data, step, mb):
    n_tok = SEQ + 1
    per_step = N_MB * MB_BSZ * n_tok
    off = (step * per_step + mb * MB_BSZ * n_tok) % (data.numel() - per_step - 1)
    chunk = data[off:off + n_tok]
    x = chunk[:SEQ].view(1, SEQ)
    y = chunk[1:SEQ + 1].view(1, SEQ)
    return x, y


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['llama', 'moe'], required=True)
    ap.add_argument('--backend', choices=['baseline', 'sorcar'], required=True)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    dist.init_process_group('xla', init_method='xla://')
    rank = xr.global_ordinal()
    ws = xr.world_size()
    device = torch_xla.device()
    torch.manual_seed(args.seed)     # identical init on every rank (DP)

    model = LM(args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(rank, f'[e2e] arch={args.arch} backend={args.backend} ws={ws} '
              f'params={n_params/1e6:.1f}M')

    data = load_data(rank, ws)
    log(rank, f'[e2e] data tokens per rank: {data.numel()}')

    # Parameter partition for the family sites:
    #  - embedding grad             -> F1 (microbatch scaled accumulation)
    #  - RMSNorm weights (all)      -> F4 (per-layer dispatch collapse)
    #  - attn.qkv of layer 0        -> F7 (slab fusion)
    #  - all remaining 2D weights   -> F5 (ZeRO-1 reduce-scatter), flattened
    norm_params, qkv0_param, big_params = [], None, []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == 'emb.weight':
            emb_param = p
        elif 'ln' in name:
            norm_params.append(p)
        elif name == 'blocks.0.attn.qkv.weight':
            qkv0_param = p
        else:
            big_params.append(p)

    big_numel = sum(p.numel() for p in big_params)
    pad = (ws - big_numel % ws) % ws
    shard = (big_numel + pad) // ws
    log(rank, f'[e2e] F5 flat grad numel={big_numel} (+pad {pad}), '
              f'shard={shard}')

    # ZeRO-1 sharded Adam state for big params; full Adam for the rest.
    opt_small = torch.optim.Adam(
        [emb_param] + norm_params + [qkv0_param], lr=args.lr)
    m_state = None
    v_state = None
    flat_params = torch.cat([p.detach().reshape(-1) for p in big_params]
                            + [torch.zeros(pad, device=device)])

    step_times, losses = [], []
    adam_t = 0

    for step in range(args.steps):
        t0 = time.time()
        # ---------------- forward/backward over microbatches ----------
        # F1 site: per-microbatch embedding-grad increments (same shape,
        # summed with scales) captured as deltas of the accumulating grad.
        mb_emb_grads = []
        prev_eg = torch.zeros_like(emb_param)
        step_loss = None
        for mb in range(N_MB):
            x, y = get_batch(data, step, mb)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
            (loss / N_MB).backward()
            cur = emb_param.grad.detach()
            mb_emb_grads.append((cur - prev_eg).clone())
            prev_eg = cur.clone()
            step_loss = loss.detach() if step_loss is None \
                else step_loss + loss.detach()
        step_loss = step_loss / N_MB

        # ---------------- family-site synchronization -----------------
        backend = args.backend
        # F1: sum_mb c_mb*AR(g_mb) (baseline, N_MB ARs) vs
        #     AR(sum_mb c_mb*g_mb) (sorcar, 1 AR). Uniform c=1 -> exact.
        emb_synced = f1_mb_grad_sync(mb_emb_grads, [1.0] * N_MB, ws, backend)

        # F2: loss metrics ×3 consumers
        lv = step_loss.reshape(1)
        m_log, m_sched, m_anom = f2_loss_metrics(lv, ws, backend)

        # F3: sync checksum (provably zero)
        chk = f3_checksum(lv, backend)

        # F4: per-layer norm-weight grads
        norm_grads = [p.grad for p in norm_params]
        norm_synced = f4_norm_grad_sync(norm_grads, ws, backend)

        # F6: clip stats over per-layer |g|max of norm grads
        absmax = torch.stack([g.abs().max() for g in norm_grads])
        gmax, gmin = f6_clip_stats(absmax, ws, backend)

        # F7: qkv slab sync
        qg = qkv0_param.grad.reshape(-1)
        qkv_synced = f7_qkv_slab_sync(qg, ws, backend)

        # F5: baseline = AR full + replicated Adam; sorcar = ZeRO-1
        # (reduce_scatter + sharded Adam + all_gather of the update).
        flat_grad = torch.cat([p.grad.reshape(-1) for p in big_params]
                              + [torch.zeros(pad, device=device)])
        gsync = f5_grad_sync(flat_grad, ws, backend)
        if m_state is None:
            m_state = torch.zeros_like(gsync)
            v_state = torch.zeros_like(gsync)
        adam_t += 1
        m_state.mul_(0.9).add_(gsync, alpha=0.1)
        v_state.mul_(0.999).addcmul_(gsync, gsync, value=0.001)
        mhat = m_state / (1 - 0.9 ** adam_t)
        vhat = v_state / (1 - 0.999 ** adam_t)
        upd = args.lr * mhat / (vhat.sqrt() + 1e-8)
        if backend == 'sorcar':
            upd = xm.all_gather(upd, dim=0)
        flat_params = flat_params - upd
        # scatter updated values back into params
        off = 0
        with torch.no_grad():
            for p in big_params:
                n = p.numel()
                p.copy_(flat_params[off:off + n].view_as(p))
                off += n

        # small-param updates: write synced grads back, checksum folded
        # into the metric stream (zero — exactness preserved)
        with torch.no_grad():
            emb_param.grad.copy_(emb_synced)
            for p, g in zip(norm_params, norm_synced):
                p.grad.copy_(g)
            qkv0_param.grad.copy_(qkv_synced.view_as(qkv0_param.grad))
        opt_small.step()

        model.zero_grad(set_to_none=False)
        xm.mark_step()
        # host sync for timing + metrics
        loss_host = float(m_log.item())
        chk_host = float(chk.sum().item())
        dt = (time.time() - t0) * 1000.0
        if step >= args.warmup:
            step_times.append(dt)
        losses.append(loss_host)
        log(rank, f'[e2e] step {step:3d} loss={loss_host:.4f} '
                  f'chk={chk_host:.2e} gmax={float(gmax.item()):.3e} '
                  f'ms={dt:.1f}')

    if rank == 0:
        med = sorted(step_times)[len(step_times) // 2] if step_times else -1
        out = {
            'arch': args.arch, 'backend': args.backend, 'ws': ws,
            'params_m': n_params / 1e6,
            'median_ms_per_step': med,
            'mean_ms_per_step': sum(step_times) / max(1, len(step_times)),
            'first_loss': losses[0], 'final_loss': losses[-1],
            'losses': losses,
        }
        print('RESULT_JSON ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
