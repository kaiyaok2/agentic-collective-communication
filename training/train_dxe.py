#!/usr/bin/env python3
"""
Distributed cross-entropy training comparison on AWS Trainium.

Tiny vocab-sharded model exercising the distributed-CE collective.
--backend baseline: 3-AR straightforward distributed CE (developer-written).
--backend agent:    uses runtime.trainium_dxe.dxe_loss (agent-evolved).

Run on 7 nodes with the launcher in training/run_dxe_compare.sh.
"""
import os, sys, time, json, math, argparse

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

VOCAB = 32256  # 224 * 144 for clean vocab-sharding across ws=224
DM = 1024
SEQLEN = 256
BSZ = 1
SEED = 42
N_BATCHES = 500


# ============ Distributed CE: developer baseline ============

def _ce_baseline(logits_local, targets, V_local, ws, rank):
    """Straightforward 3-AR distributed cross-entropy."""
    local_max = logits_local.max(dim=-1).values
    global_max = xm.all_reduce(xm.REDUCE_MAX, local_max)
    shifted = logits_local - global_max.unsqueeze(-1)
    local_sum_exp = shifted.exp().sum(dim=-1)
    global_sum_exp = xm.all_reduce(xm.REDUCE_SUM, local_sum_exp)
    log_sum_exp = global_sum_exp.log() + global_max
    lo, hi = rank * V_local, (rank + 1) * V_local
    target_local = targets - lo
    in_shard = (targets >= lo) & (targets < hi)
    target_local_safe = target_local.clamp(0, V_local - 1)
    local_target_logit = logits_local.gather(
        1, target_local_safe.unsqueeze(1)).squeeze(1)
    local_target_logit = torch.where(
        in_shard, local_target_logit, torch.zeros_like(local_target_logit))
    global_target_logit = xm.all_reduce(xm.REDUCE_SUM, local_target_logit)
    return (log_sum_exp - global_target_logit).mean()


def _ce_agent_factory():
    from runtime.trainium_dxe import dxe_loss, init_dxe
    init_dxe()
    rank = xr.global_ordinal()
    return lambda logits_local, targets, V_local, ws, _rank: dxe_loss(
        logits_local, targets, V_local)


class TinyModel(nn.Module):
    def __init__(self, V_local):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.head = nn.Linear(DM, V_local, bias=False)
        self.V_local = V_local

    def forward(self, ids):
        return self.head(self.emb(ids))


def get_lr(step, total, max_lr=3e-4, min_lr=3e-5, warm=200):
    if step < warm:
        return max_lr * step / warm
    decay = (step - warm) / max(total - warm, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay))


def run(args):
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()
    assert VOCAB % ws == 0, f"VOCAB={VOCAB} not divisible by ws={ws}"
    V_local = VOCAB // ws

    ce_fn = _ce_baseline if args.backend == "baseline" else _ce_agent_factory()

    if rank == 0:
        print(f"[init] backend={args.backend} ws={ws} steps={args.steps}")
        print(f"[init] VOCAB={VOCAB} DM={DM} SEQLEN={SEQLEN} V_local={V_local}")

    torch.manual_seed(SEED)
    model = TinyModel(V_local).to(dev).to(torch.bfloat16)
    rep_params = list(model.emb.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(SEED + rank)
    data_inp = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=g) for _ in range(N_BATCHES)]
    data_tgt = [torch.randint(0, VOCAB, (BSZ, SEQLEN), generator=g) for _ in range(N_BATCHES)]

    def step(s):
        bi = s % N_BATCHES
        cur_inp = data_inp[bi].to(dev)
        cur_tgt = data_tgt[bi].to(dev)
        lr = get_lr(s, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        logits = model(cur_inp).view(-1, V_local)
        targets_flat = cur_tgt.view(-1)
        loss = ce_fn(logits, targets_flat, V_local, ws, rank)
        loss.backward()
        # grad sync on rep_params using the WINNING async pattern from Routine A
        inv = 1.0 / ws
        out = []
        for p in rep_params:
            if p.grad is not None:
                out.append((p, xm.all_reduce(xm.REDUCE_SUM, p.grad.data)))
        for p, gg in out:
            p.grad.data = gg * inv
        opt.step()
        opt.zero_grad()
        xm.mark_step()
        return loss

    if rank == 0:
        print("[warmup] compiling XLA graphs...")
    for _ in range(args.warmup):
        step(0)
    if rank == 0:
        print("[warmup] done")

    xm.rendezvous("pre_measure")
    t0 = time.time()
    times, losses = [], []
    for s in range(args.steps):
        ts = time.time()
        loss = step(s)
        xm.wait_device_ops()
        times.append(time.time() - ts)
        losses.append(loss.item())
        if rank == 0 and (s + 1) % 25 == 0:
            print(f"  step {s+1:>4}/{args.steps}  loss={losses[-1]:.4f}  "
                  f"avg={sum(times[-25:])/25*1000:.0f}ms")
    wall = time.time() - t0
    avg_ms = sum(times) / len(times) * 1000
    steady = sum(times[-30:]) / min(30, len(times)) * 1000 if len(times) >= 30 else None

    if rank == 0:
        print(f"  Backend     : {args.backend}")
        print(f"  Wall clock  : {wall:.2f} s")
        print(f"  Avg step    : {avg_ms:.1f} ms")
        if steady:
            print(f"  Steady (last 30): {steady:.1f} ms")
        out = os.environ.get("RESULTS_DIR",
                             "/home/ubuntu/agentic-collective-communication/training/results")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"dxe_{args.backend}.json")
        with open(path, "w") as f:
            json.dump(dict(
                backend=args.backend, steps=args.steps, world_size=ws,
                wall_clock_s=round(wall, 2),
                avg_step_ms=round(avg_ms, 1),
                steady_step_ms=round(steady, 1) if steady else None,
                losses=[round(l, 4) if l == l else None for l in losses],
                step_times_ms=[round(t * 1000, 1) for t in times],
            ), f, indent=2)
        print(f"  Saved -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["baseline", "agent"], required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    run(p.parse_args())
