#!/usr/bin/env python3
"""
Toy 7-node 224-rank Uniform AllToAll training script.

A scaled-down version of train_uniform_a2a_7node.py whose gathered
buffer fits comfortably within the per-rank HBM allocation budget at
224 ranks so that the agent path can complete a full training loop
(the full-shape script SIGABRTs during compile-group setup at this
scale, see Table 2 footnote).

Shape: LAYERS=1, DM=512, NEXP=224 (=ws), TOPK=2, EXDIM=128,
SEQLEN=64, BSZ=1, VOCAB=8064. cap = ceil(64*2/224) = 1 ->
chunk_size = 1*512 = 512 elements -> gathered = 224**2 * 512
~= 26M bf16 elements ~= 51 MB per dispatch. About 12x smaller than
the full-shape script.

Backends are the same as the full script:
  baseline: developer inline AG+T+RS
  agent:    runtime.trainium_uniform_a2a.uniform_a2a (whatever Phase 5
            deployed; bit-identical to main strategy-enumerate when
            run after backup_main_runtimes()).
"""
import argparse, os, sys, time, json, math, statistics
import time as _t

os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_RT_STOCHASTIC_ROUNDING_EN', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

VOCAB, DM, HEADS = 8064, 512, 4
LAYERS, NEXP, TOPK, EXDIM = 1, 224, 2, 128
SEQLEN, BSZ = 64, 1


def _ua2a_baseline(x, ws, chunk_size):
    """Developer inline AG+T+RS, identical algorithm to train_uniform_a2a_7node.py."""
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    reshaped = gathered.view(ws, ws, chunk_size)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                             scale=1.0 / ws, scatter_dim=0, shard_count=ws)


def _load_agent():
    from runtime.trainium_uniform_a2a import uniform_a2a, init_uniform_a2a
    init_uniform_a2a()
    return lambda x, ws_, cs: uniform_a2a(x, cs)


class _UA2ALayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ws, chunk_size, fn):
        ctx.ws = ws
        ctx.chunk_size = chunk_size
        ctx.fn = fn
        return fn(x, ws, chunk_size)

    @staticmethod
    def backward(ctx, grad_out):
        # Symmetric: ua2a is its own transpose; apply the same fn to the gradient.
        return ctx.fn(grad_out.contiguous(), ctx.ws, ctx.chunk_size), None, None, None


class ToyMoE(nn.Module):
    def __init__(self, dm, exdim, ws, cap, chunk_size, fn):
        super().__init__()
        self.dm = dm
        self.exdim = exdim
        self.ws = ws
        self.cap = cap
        self.chunk_size = chunk_size
        self.fn = fn
        # One expert per rank; expert is a simple linear layer.
        self.expert_w = nn.Parameter(torch.randn(dm, exdim, dtype=torch.bfloat16) * 0.01)

    def forward(self, h):
        # h: (BSZ, SEQLEN, DM). Pretend each rank dispatches `cap` tokens
        # to each of `ws` destinations. Pad to (ws * cap, DM), flatten to a
        # 1D send buffer of length ws*cap*DM, ua2a, then sum-reduce on the
        # received chunks as the "expert output".
        B, S, D = h.shape
        flat = h.reshape(B * S, D)
        pad_n = self.ws * self.cap
        if flat.shape[0] < pad_n:
            pad = torch.zeros(pad_n - flat.shape[0], D, dtype=flat.dtype, device=flat.device)
            flat = torch.cat([flat, pad], dim=0)
        else:
            flat = flat[:pad_n]
        send_buf = flat.reshape(-1).contiguous()  # (ws*cap*D,)
        recv_buf = _UA2ALayer.apply(send_buf, self.ws, self.chunk_size, self.fn)
        # Reshape back to (ws*cap, D) and run the expert layer.
        recv = recv_buf.view(self.ws * self.cap, D)
        processed = recv @ self.expert_w  # (ws*cap, exdim)
        # Send results back (re-pack as send buffer) — symmetric ua2a.
        send2 = processed.contiguous().view(-1)
        # Adjust chunk_size for the return trip (now per-token EXDIM, not DM).
        return_chunk = self.cap * self.exdim
        recv2_buf = _UA2ALayer.apply(send2, self.ws, return_chunk, self.fn)
        recv2 = recv2_buf.view(self.ws * self.cap, self.exdim)
        # Sum-reduce per source into a single (BSZ, SEQLEN, EXDIM)
        # representation that we project back to DM.
        out_flat = recv2.sum(dim=0).expand(B * S, self.exdim)
        return out_flat


class ToyBlock(nn.Module):
    def __init__(self, dm, exdim, ws, cap, chunk_size, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dm, dtype=torch.bfloat16)
        self.moe = ToyMoE(dm, exdim, ws, cap, chunk_size, fn)
        # Project EXDIM back to DM
        self.proj = nn.Parameter(torch.randn(exdim, dm, dtype=torch.bfloat16) * 0.01)

    def forward(self, x):
        # x: (B, S, DM)
        h = self.norm(x)
        moe_out = self.moe(h)            # (B*S, EXDIM)
        out_dm = moe_out @ self.proj     # (B*S, DM)
        return x + out_dm.view(x.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["baseline", "agent"], required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = xr.world_size()
    assert ws == NEXP, f"NEXP({NEXP}) must equal world_size({ws})"
    cap = max(1, math.ceil(SEQLEN * TOPK / ws))
    chunk_size = cap * DM
    gathered_mb = (ws * ws * chunk_size * 2) / 1e6
    if rank == 0:
        print(f"[init] backend={args.backend} ws={ws} steps={args.steps}", flush=True)
        print(f"[init] cap={cap} chunk_size={chunk_size} gathered={gathered_mb:.1f} MB bf16", flush=True)

    fn = _ua2a_baseline if args.backend == "baseline" else _load_agent()

    blocks = nn.ModuleList(
        [ToyBlock(DM, EXDIM, ws, cap, chunk_size, fn) for _ in range(LAYERS)]
    ).to(dev)
    all_params = list(blocks.parameters())

    inputs = [torch.randn(BSZ, SEQLEN, DM, device=dev, dtype=torch.bfloat16) * 0.01
              for _ in range(args.steps + args.warmup)]
    targets = [torch.randint(0, VOCAB, (BSZ, SEQLEN), device=dev, dtype=torch.int64)
               for _ in range(args.steps + args.warmup)]

    def step_once(x, tgt):
        h = x
        for b in blocks:
            h = b(h)
        # Random target loss (toy)
        loss = (h ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for p in all_params:
                if p.grad is not None:
                    p.data -= 1e-4 * p.grad
                    p.grad = None
        return loss

    for s in range(args.warmup):
        _ = step_once(inputs[s], targets[s]).item()
    if rank == 0:
        print("[warmup] done", flush=True)

    times = []
    t_total = time.time()
    for s in range(args.steps):
        xm.mark_step()
        t0 = time.time()
        _ = step_once(inputs[args.warmup + s], targets[args.warmup + s]).item()
        times.append((time.time() - t0) * 1000.0)
        if rank == 0 and (s + 1) % 50 == 0:
            print(f"  step {s+1}/{args.steps}: median_ms={statistics.median(times[-50:]):.2f}", flush=True)

    wall = time.time() - t_total
    if rank == 0:
        steady = times[args.steps // 2:]
        print(f"\n  Problem     : uniform_a2a (toy)", flush=True)
        print(f"  Backend     : {args.backend}", flush=True)
        print(f"  Wall        : {wall:.2f} s", flush=True)
        print(f"  Avg step    : {statistics.mean(times):.2f} ms", flush=True)
        print(f"  Steady step : {statistics.median(steady):.2f} ms (median of last half)", flush=True)
        with open(f"/tmp/tp_search/ua2a_toy_{args.backend}.json", "w") as f:
            json.dump({"backend": args.backend, "wall_s": wall,
                       "avg_ms": statistics.mean(times),
                       "steady_median_ms": statistics.median(steady),
                       "all_ms": times,
                       "ws": ws, "cap": cap, "chunk_size": chunk_size,
                       "gathered_mb": gathered_mb}, f)


if __name__ == "__main__":
    main()
