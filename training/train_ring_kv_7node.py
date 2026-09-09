#!/usr/bin/env python3
"""7-node 224-rank per-call training-scope micro-runner for Ring KV.

Mirrors a small sequence-parallel attention setup that exercises the Ring KV
collective (all_gather of K and V across all ranks for distributed attention).

Backend semantics:
  baseline: per-head xm.all_gather of K and V (HEADS×2 dispatches per layer per step)
  agent:    runtime.trainium_ring_kv_7node.ring_kv_gather — single fused all_gather

Per-call probing via training._percall_probe.in_window() in a late window.
"""
import argparse, os, sys, time, json, math, statistics
import time as _t

os.environ.setdefault("NEURON_NUM_RECENT_MODELS_TO_KEEP", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
os.environ.setdefault("NEURON_COMPILE_CACHE_URL", "/home/ubuntu/neuron_cache")

import torch
import torch.nn as nn
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, "/home/ubuntu/agentic-collective-communication")
from training import _percall_probe as probe

# Small focused config
DM = 1024
HEADS = 8
HEAD_DIM = DM // HEADS  # 128
SEQ_PER_RANK = 16  # tiny, just to exercise the collective
KV = 2  # K and V
WARMUP_DEFAULT = 5
SEED = 42
PROB = "ring_kv"


def _baseline_per_head(kv):
    """HEADS×KV separate all_gathers along seq dim. kv shape (KV, HEADS, SEQ_PER_RANK, HEAD_DIM)."""
    parts = []
    for slot in range(KV):
        for h in range(HEADS):
            gathered = xm.all_gather(kv[slot, h], dim=0)  # (SEQ_GLOBAL, HEAD_DIM)
            parts.append(gathered.reshape(-1))
    return torch.cat(parts)


def _load_agent():
    from runtime.trainium_ring_kv_7node import ring_kv_gather as _kv, init_ring_kv
    init_ring_kv()
    return _kv


class _RingKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, backend, rank, fn):
        ctx.backend = backend
        ctx.fn = fn
        ctx.rank = rank
        xm.mark_step()
        if probe.in_window():
            _t0 = _t.time()
            if backend == "baseline":
                out = _baseline_per_head(kv)
            else:
                out = fn(kv)
                if isinstance(out, (list, tuple)):
                    out = torch.cat([o.reshape(-1) for o in out])
            _ = out.sum().item()
            probe.record("ring_kv", (_t.time() - _t0) * 1000.0)
        else:
            if backend == "baseline":
                out = _baseline_per_head(kv)
            else:
                out = fn(kv)
                if isinstance(out, (list, tuple)):
                    out = torch.cat([o.reshape(-1) for o in out])
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        return None, None, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["baseline", "agent"], required=True)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    args = p.parse_args()

    dist.init_process_group("xla")
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get("WORLD_SIZE", xr.world_size()))
    torch.manual_seed(SEED)

    if args.backend == "baseline":
        agent_fn = None
    else:
        agent_fn = _load_agent()

    # Synthetic K/V tensor exercising the gather
    kv = (torch.randn(KV, HEADS, SEQ_PER_RANK, HEAD_DIM, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    kv.requires_grad_(False)
    if rank == 0:
        print(f"[init] ring_kv backend={args.backend} ws={ws} HEADS={HEADS} SEQ_PER_RANK={SEQ_PER_RANK} HEAD_DIM={HEAD_DIM}", flush=True)

    for s in range(args.warmup):
        out = _RingKV.apply(kv, args.backend, rank, agent_fn)
        xm.wait_device_ops()
    if rank == 0: print("[init] warmup done", flush=True)

    times = []
    for s in range(args.steps):
        probe.set_step(s)
        xm.mark_step()
        t0 = time.time()
        out = _RingKV.apply(kv, args.backend, rank, agent_fn)
        _ = out.sum().item()
        times.append((time.time() - t0) * 1000)
        if rank == 0 and (s + 1) % 50 == 0:
            print(f"  step {s+1}: median_ms={statistics.median(times[-50:]):.2f}", flush=True)

    if rank == 0:
        STEADY = 50
        steady = times[STEADY:] if len(times) > STEADY else times
        out_dir = os.environ.get("RESULTS_DIR", "/tmp/h7_bench")
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/ring_kv_7node_{args.backend}.json", "w") as f:
            json.dump({"backend": args.backend, "ws": ws, "steady_mean_ms": statistics.mean(steady),
                       "steady_median_ms": statistics.median(steady), "all_ms": times}, f)
        probe.dump(f"{out_dir}/ring_kv_7node_{args.backend}_percall.json",
                   extra={"backend": args.backend, "steps": args.steps})
        print(f"[bench] ring_kv {args.backend} steady_mean={statistics.mean(steady):.2f}ms steady_median={statistics.median(steady):.2f}ms", flush=True)


if __name__ == "__main__":
    main()
