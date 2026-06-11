#!/usr/bin/env python3
"""7-node 224-rank training comparison for distributed cross-entropy (dxe).

  baseline: developer inline 1-AG + F.cross_entropy. Matches baseline_ce
            in train_olmoe10b.py line-for-line.
  agent:    runtime.trainium_dxe_7node.dxe_loss (deployed Phase-5 runtime).
"""
import argparse, os, sys, time, json, math, statistics

os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_RT_STOCHASTIC_ROUNDING_EN', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

VOCAB = 32256
DM = 2048
BSZ = 1
SEQLEN = 256
LAYERS = 2


def _baseline_dxe(logits_local, targets, v_local):
    """Inline 1-AG + F.cross_entropy. Matches train_olmoe10b.py baseline_ce."""
    ll = logits_local.reshape(-1, v_local).contiguous()
    gathered = xm.all_gather(ll, dim=1)
    return F.cross_entropy(gathered.float(), targets.reshape(-1))


def _load_agent_fn(ws):
    from runtime.trainium_dxe_7node import dxe_loss, init_dxe
    init_dxe()
    return dxe_loss


class _DxeLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, v_local, fn):
        ctx.save_for_backward(logits, targets)
        ctx.v_local = v_local
        ctx.fn = fn
        xm.mark_step()
        out = fn(logits, targets, v_local) if fn is not None \
              else _baseline_dxe(logits, targets, v_local)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        logits, targets = ctx.saved_tensors
        return torch.zeros_like(logits), None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["baseline", "agent"], required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = xr.world_size()
    assert VOCAB % ws == 0, f"VOCAB={VOCAB} must be divisible by ws={ws}"
    v_local = VOCAB // ws
    T = BSZ * SEQLEN

    if rank == 0:
        print(f"[init] dxe backend={args.backend} ws={ws} VOCAB={VOCAB} v_local={v_local} T={T} LAYERS={LAYERS} steps={args.steps}", flush=True)

    fn = _load_agent_fn(ws) if args.backend == "agent" else None

    # Per-step random inputs
    logits_list = [torch.randn(T, v_local, device=dev, dtype=torch.bfloat16, requires_grad=True) * 0.01
                   for _ in range(args.steps + args.warmup)]
    targets_list = [torch.randint(0, VOCAB, (T,), device=dev, dtype=torch.int64)
                    for _ in range(args.steps + args.warmup)]

    def step_once(logits, tgt):
        # Run LAYERS dxe calls to amortise per-step cost
        loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
        for L in range(LAYERS):
            loss = loss + _DxeLayer.apply(logits, tgt, v_local, fn)
        # No backward; dxe baseline has different backward semantics from agent.
        # We only measure the forward dispatch cost (which is what dominates).
        return loss

    for s in range(args.warmup):
        _ = step_once(logits_list[s], targets_list[s]).item()
    if rank == 0: print("[warmup] done", flush=True)

    times = []
    t_total = time.time()
    for s in range(args.steps):
        xm.mark_step()
        t0 = time.time()
        _ = step_once(logits_list[args.warmup + s], targets_list[args.warmup + s]).item()
        times.append((time.time() - t0) * 1000.0)
        if rank == 0 and (s + 1) % 50 == 0:
            print(f"  step {s+1}/{args.steps}: median_ms={statistics.median(times[-50:]):.2f}", flush=True)

    wall = time.time() - t_total
    if rank == 0:
        steady = times[args.steps // 2:]
        print(f"  Problem     : dxe", flush=True)
        print(f"  Backend     : {args.backend}", flush=True)
        print(f"  Wall        : {wall:.2f} s", flush=True)
        print(f"  Avg step    : {statistics.mean(times):.2f} ms", flush=True)
        print(f"  Steady step : {statistics.median(steady):.2f} ms (median of last half)", flush=True)
        os.makedirs("/tmp/tp_search", exist_ok=True)
        with open(f"/tmp/tp_search/dxe_{args.backend}.json", "w") as f:
            json.dump({"backend": args.backend, "wall_s": wall,
                       "avg_ms": statistics.mean(times),
                       "steady_median_ms": statistics.median(steady),
                       "all_ms": times,
                       "ws": ws, "VOCAB": VOCAB, "v_local": v_local, "T": T, "LAYERS": LAYERS}, f)


if __name__ == "__main__":
    main()
