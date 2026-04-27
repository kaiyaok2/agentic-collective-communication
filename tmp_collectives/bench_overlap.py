#!/usr/bin/env python3
"""
Benchmark #3: Overlapped Communication-Computation
Baseline: all_reduce after all backward computation completes (sequential)
Optimized: chunk the all_reduce and interleave with computation via mark_step
"""
import os, time, json
os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache_overlap')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

DM = 2048
LAYERS = 6
FFN_DIM = 4096
SEQLEN = 256
BSZ = 1
VOCAB = 8192


class SimpleBlock(nn.Module):
    def __init__(self, d, fd):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, fd, bias=False)
        self.down = nn.Linear(fd, d, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.down(F.gelu(self.up(h)))


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.blocks = nn.ModuleList([SimpleBlock(DM, FFN_DIM) for _ in range(LAYERS)])
        self.head = nn.Linear(DM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def baseline_sequential(model, inp, tgt, opt, ws):
    """Standard: forward, backward, then all_reduce all grads, step."""
    logits = model(inp)
    loss = F.cross_entropy(logits.view(-1, VOCAB), tgt.view(-1))
    loss.backward()
    # All-reduce ALL grads at once
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    xm.all_reduce('sum', grads)
    for g in grads:
        g.data /= ws
    opt.step()
    opt.zero_grad()
    xm.mark_step()
    return loss


def optimized_chunked_overlap(model, inp, tgt, opt, ws):
    """Chunked: after backward, all_reduce grads in buckets with mark_step between.
    This lets XLA pipeline the all_reduce of bucket N with any pending compute."""
    logits = model(inp)
    loss = F.cross_entropy(logits.view(-1, VOCAB), tgt.view(-1))
    loss.backward()

    # Bucket grads by layer — each layer's grads get their own all_reduce + mark_step
    params = list(model.parameters())
    bucket_size = len(params) // 3  # 3 buckets
    for i in range(0, len(params), bucket_size):
        bucket = [p.grad for p in params[i:i + bucket_size] if p.grad is not None]
        if bucket:
            xm.all_reduce('sum', bucket)
            for g in bucket:
                g.data /= ws
            xm.mark_step()

    opt.step()
    opt.zero_grad()
    xm.mark_step()
    return loss


def optimized_per_layer_overlap(model, inp, tgt, opt, ws):
    """Per-layer: all_reduce each layer's grads separately with mark_step.
    Maximum overlap opportunity but more dispatches."""
    logits = model(inp)
    loss = F.cross_entropy(logits.view(-1, VOCAB), tgt.view(-1))
    loss.backward()

    for block in model.blocks:
        block_grads = [p.grad for p in block.parameters() if p.grad is not None]
        if block_grads:
            xm.all_reduce('sum', block_grads)
            for g in block_grads:
                g.data /= ws
            xm.mark_step()

    # Embedding + head grads
    remaining = [p.grad for n, p in model.named_parameters()
                 if p.grad is not None and 'blocks' not in n]
    if remaining:
        xm.all_reduce('sum', remaining)
        for g in remaining:
            g.data /= ws

    opt.step()
    opt.zero_grad()
    xm.mark_step()
    return loss


def run():
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()

    warmup = 3
    iters = 20

    if rank == 0:
        print(f"[overlap] ws={ws}  model={LAYERS}L x {DM}d x {FFN_DIM}ffn  seq={SEQLEN}")

    results = {}
    for name, step_fn_factory in [
        ('baseline_sequential', lambda m, i, t, o: lambda: baseline_sequential(m, i, t, o, ws)),
        ('chunked_3bucket', lambda m, i, t, o: lambda: optimized_chunked_overlap(m, i, t, o, ws)),
        ('per_layer_overlap', lambda m, i, t, o: lambda: optimized_per_layer_overlap(m, i, t, o, ws)),
    ]:
        torch.manual_seed(42)
        model = SimpleModel().to(dev).to(torch.bfloat16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        inp = torch.randint(0, VOCAB, (BSZ, SEQLEN), device=dev)
        tgt = torch.randint(0, VOCAB, (BSZ, SEQLEN), device=dev)

        step_fn = step_fn_factory(model, inp, tgt, opt)

        for _ in range(warmup):
            step_fn()

        xm.wait_device_ops()
        t0 = time.time()
        for _ in range(iters):
            step_fn()
        xm.wait_device_ops()
        avg_ms = (time.time() - t0) / iters * 1000
        results[name] = round(avg_ms, 3)
        if rank == 0:
            print(f"  {name:30s}: {avg_ms:.3f} ms/step")

    if rank == 0:
        path = '/home/ubuntu/trainium-llm-search/tmp_collectives/results_overlap.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {path}")


if __name__ == '__main__':
    run()
