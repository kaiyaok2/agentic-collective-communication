"""Clean NXD-primitive Llama harness for real-OWT descent validation.

Single-stage TP=world_size, no PP, no hand-rolled FSDP. All collectives are
provided by neuronx_distributed.parallel_layers, which wrap the backward in
autograd-correct primitives — sidestepping the silent-zero pattern that
killed gradients in train_llama_nxd_mb.py.

Defaults: amp1-shape Llama-block (DM=2048, HID=5376, N_LAYERS=4, S=1024).

Two backends compared:
  baseline: per-microbatch mark_step (M separate dispatches per step)
  agent:    bundled (single mark_step graph for M stacked microbatches)
"""
import argparse, os, sys, time, json, statistics, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_RT_STOCHASTIC_ROUNDING_EN', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache')

from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    parallel_state,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy

# Defaults: amp1-shape (Llama-block) — small enough for fast iteration on
# a single trn1.32xlarge, large enough that real-OWT descent is meaningful.
DM = 2048
HID = 5376
N_LAYERS = 4
B = 1
S = 1024
VOCAB = 32256
LR_MAX = 5e-5
LR_WARMUP_STEPS = 30
SEED = 42


class LlamaMLP(nn.Module):
    def __init__(self, dm, hid):
        super().__init__()
        # gate + up: column-parallel (shard output dim HID), keep output sharded.
        self.gate = ColumnParallelLinear(dm, hid, bias=False, gather_output=False, dtype=torch.bfloat16)
        self.up   = ColumnParallelLinear(dm, hid, bias=False, gather_output=False, dtype=torch.bfloat16)
        # down: row-parallel (shard input dim HID); reduce_output=True does the all-reduce.
        self.down = RowParallelLinear(hid, dm, bias=False, input_is_parallel=True, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LlamaModel(nn.Module):
    def __init__(self, vocab, dm, hid, n_layers):
        super().__init__()
        self.embed  = ParallelEmbedding(vocab, dm, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([LlamaMLP(dm, hid) for _ in range(n_layers)])
        # LM head: column-parallel, keep sharded → parallel_cross_entropy consumes vocab-parallel logits.
        self.lm_head = ColumnParallelLinear(dm, vocab, bias=False, gather_output=False, dtype=torch.bfloat16)

    def forward(self, tokens):
        h = self.embed(tokens)
        for layer in self.layers:
            h = h + layer(h)
        return self.lm_head(h)


def step_per_mb(model, batches_in, batches_tgt, M):
    loss_total = 0.0
    for m in range(M):
        logits = model(batches_in[m])
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_tgts   = batches_tgt[m].reshape(-1)
        loss = parallel_cross_entropy(flat_logits, flat_tgts).mean()
        loss_total = loss_total + loss
        xm.mark_step()
    return loss_total / M


def step_bundled(model, batches_in, batches_tgt, M):
    inp = torch.stack(batches_in, dim=0).reshape(-1, batches_in[0].shape[-1])
    tgt = torch.stack(batches_tgt, dim=0).reshape(-1, batches_tgt[0].shape[-1])
    logits = model(inp)
    loss = parallel_cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        tgt.reshape(-1),
    ).mean()
    xm.mark_step()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['baseline', 'agent'], default='baseline')
    ap.add_argument('--microbatches', type=int, default=2)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--realtok', action='store_true')
    ap.add_argument('--n-batches', type=int, default=64)
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group('xla', init_method='xla://')
    ws = xr.world_size()
    rank = xr.global_ordinal()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=ws,
        pipeline_model_parallel_size=1,
    )

    dev = xm.xla_device()
    torch.manual_seed(SEED)

    if rank == 0:
        print(f'[init] ws={ws} backend={args.backend} M={args.microbatches} steps={args.steps} '
              f'DM={DM} HID={HID} N_LAYERS={N_LAYERS} S={S}', flush=True)

    model = LlamaModel(VOCAB, DM, HID, N_LAYERS).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.01)

    if args.realtok:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('gpt2')
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        ds = load_dataset('Skylion007/openwebtext', split='train', streaming=True,
                          trust_remote_code=False)
        it = iter(ds)
        needed = args.n_batches * args.microbatches * B * (S + 1)
        buf = []
        while sum(len(x) for x in buf) < needed:
            try: ex = next(it)
            except StopIteration: break
            ids = tok(ex['text'], add_special_tokens=False, truncation=False)['input_ids']
            buf.append(ids)
        flat = [t % VOCAB for chunk in buf for t in chunk][:needed]
        windows = S + 1
        data_inp, data_tgt, offset = [], [], 0
        for bi in range(args.n_batches):
            mb_in, mb_tg = [], []
            for m in range(args.microbatches):
                toks = flat[offset:offset + B * windows]
                offset += B * windows
                t = torch.tensor(toks, dtype=torch.int64).view(B, windows)
                mb_in.append(t[:, :S].to(dev))
                mb_tg.append(t[:, 1:S+1].to(dev))
            data_inp.append(mb_in)
            data_tgt.append(mb_tg)
        if rank == 0:
            print(f'[data] real-OWT: {len(data_inp)} batches, M={args.microbatches}, B={B}, S={S}', flush=True)
    else:
        data_inp = [[torch.randint(0, VOCAB, (B, S), device=dev, dtype=torch.int64) for _ in range(args.microbatches)] for _ in range(args.n_batches)]
        data_tgt = [[torch.randint(0, VOCAB, (B, S), device=dev, dtype=torch.int64) for _ in range(args.microbatches)] for _ in range(args.n_batches)]

    fn = step_per_mb if args.backend == 'baseline' else step_bundled

    for _ in range(args.warmup):
        opt.zero_grad()
        loss = fn(model, data_inp[0], data_tgt[0], args.microbatches)
        loss.backward()
        opt.step()
    if rank == 0: print('[warmup] done', flush=True)

    times, losses = [], []
    for s in range(args.steps):
        bi = s % len(data_inp)
        # Linear warmup over LR_WARMUP_STEPS, then constant.
        lr = LR_MAX * min(1.0, (s + 1) / LR_WARMUP_STEPS)
        for pg in opt.param_groups: pg['lr'] = lr
        t0 = time.time()
        opt.zero_grad()
        loss = fn(model, data_inp[bi], data_tgt[bi], args.microbatches)
        loss.backward()
        # Grad clipping at L2-norm 1.0 across replicated params (cross-rank).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        xm.wait_device_ops()
        times.append((time.time() - t0) * 1000.0)
        losses.append(loss.item())
        if rank == 0 and (s+1) % 10 == 0:
            print(f'  step {s+1:4d}: median_ms={statistics.median(times[-10:]):.1f} '
                  f'loss={statistics.mean(losses[-10:]):.4f} lr={lr:.2e}', flush=True)

    if rank == 0:
        steady = times[20:] if len(times) > 20 else times
        steady_l = losses[20:] if len(losses) > 20 else losses
        print(f'[bench] {args.backend} steady_median={statistics.median(steady):.2f}ms '
              f'final_loss={losses[-1]:.4f}', flush=True)
        pathlib.Path('/tmp/tp_search').mkdir(parents=True, exist_ok=True)
        out = f'/tmp/tp_search/llama_nxd_clean_M{args.microbatches}_{args.backend}.json'
        with open(out, 'w') as f:
            json.dump({'backend': args.backend, 'M': args.microbatches, 'steps': args.steps,
                       'steady_median_ms': statistics.median(steady),
                       'losses': losses, 'step_times_ms': times,
                       'shape': dict(DM=DM, HID=HID, N_LAYERS=N_LAYERS, B=B, S=S, VOCAB=VOCAB)}, f)
        print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
