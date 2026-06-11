"""Llama-style training with sequential vs parallel attention+MLP.

Two backends:
  sequential : standard Llama block (attn -> AR -> residual+norm -> MLP -> AR)
  parallel   : PaLM/GPT-J style (attn || MLP both consume norm(x); stacked AR)

Each step is forward+backward+SGD on a small Llama transformer.
Measures steady-state step time over WARMUP-skipped iterations.
"""
import os, sys, time, json, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

DM = 2240
HID = 5376
NUM_HEADS = 16
HEAD_DIM = DM // NUM_HEADS
N_LAYERS = 2
B = 1
S = 512
VOCAB = 224 * 32
LR = 1e-4
WARMUP = 3
STEADY_FROM = 50


def main():
    dev = xm.xla_device()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    rank = xr.global_ordinal()
    assert HID % ws == 0 and DM % ws == 0
    shard_hid = HID // ws

    # Replicated embedding + LM head for simplicity (the architecture
    # difference under test is attention+MLP arrangement, not embed/head)
    embed = nn.Embedding(VOCAB, DM, dtype=torch.bfloat16).to(dev)
    lm_head = nn.Linear(DM, VOCAB, bias=False, dtype=torch.bfloat16).to(dev)

    # Per-layer params, all TP-sharded along HID dim
    # Attention: column-parallel QKV (DM, 3*shard_head), row-parallel out_proj (shard_head, DM)
    # We simplify: just shard DM into 'shard_head' = DM/ws for col-parallel attention QKV
    shard_dm = DM // ws  # for col-parallel
    Wqkv_shards = [nn.Parameter(torch.randn(DM, 3 * shard_dm, dtype=torch.bfloat16) * 0.01).to(dev)
                   for _ in range(N_LAYERS)]
    Wo_shards   = [nn.Parameter(torch.randn(shard_dm, DM, dtype=torch.bfloat16) * 0.01).to(dev)
                   for _ in range(N_LAYERS)]
    # MLP: column-parallel gate+up, row-parallel down
    Wg_shards = [nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01).to(dev)
                 for _ in range(N_LAYERS)]
    Wu_shards = [nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01).to(dev)
                 for _ in range(N_LAYERS)]
    Wd_shards = [nn.Parameter(torch.randn(shard_hid, DM, dtype=torch.bfloat16) * 0.01).to(dev)
                 for _ in range(N_LAYERS)]
    norm_w = [nn.Parameter(torch.ones(DM, dtype=torch.bfloat16)).to(dev) for _ in range(N_LAYERS)]
    norm_w2 = [nn.Parameter(torch.ones(DM, dtype=torch.bfloat16)).to(dev) for _ in range(N_LAYERS)]  # used only by sequential

    params = list(embed.parameters()) + list(lm_head.parameters()) \
             + Wqkv_shards + Wo_shards + Wg_shards + Wu_shards + Wd_shards \
             + norm_w + norm_w2

    inputs = [torch.randint(0, VOCAB, (B, S), device=dev, dtype=torch.int64) for _ in range(3)]
    targets = [torch.randint(0, VOCAB, (B, S), device=dev, dtype=torch.int64) for _ in range(3)]

    def rms_norm(x, w):
        v = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() / torch.sqrt(v + 1e-6)).to(x.dtype) * w

    def attn_partial(x_norm, L):
        qkv = x_norm @ Wqkv_shards[L]  # (B, S, 3*shard_dm) per rank
        q, k, v = qkv.chunk(3, dim=-1)
        # Attention with shard_dm/HEAD_PER_RANK heads (simplified scaled dot-product)
        # Flatten the head dim and just do plain attention with shape (B, S, shard_dm)
        scores = q @ k.transpose(-1, -2) / (shard_dm ** 0.5)
        attn = F.softmax(scores, dim=-1)
        ctx = attn @ v
        out_partial = ctx @ Wo_shards[L]  # (B, S, DM) partial
        return out_partial

    def mlp_partial(x_norm, L):
        gate = x_norm @ Wg_shards[L]
        up   = x_norm @ Wu_shards[L]
        h = F.silu(gate) * up
        return h @ Wd_shards[L]  # (B, S, DM) partial

    def sequential_block(x, L):
        # standard Llama: attn -> AR -> residual+norm -> MLP -> AR -> residual
        x_n = rms_norm(x, norm_w[L])
        ap = attn_partial(x_n, L)
        a = xm.all_reduce(xm.REDUCE_SUM, ap)
        x = x + a
        x_n2 = rms_norm(x, norm_w2[L])
        mp = mlp_partial(x_n2, L)
        m = xm.all_reduce(xm.REDUCE_SUM, mp)
        return x + m

    def parallel_block(x, L):
        # PaLM-style: attn || MLP, both consume norm(x). 1 stacked AR.
        x_n = rms_norm(x, norm_w[L])
        ap = attn_partial(x_n, L)
        mp = mlp_partial(x_n, L)
        stacked = torch.stack([ap, mp], dim=0)              # (2, B, S, DM)
        ared = xm.all_reduce(xm.REDUCE_SUM, stacked)        # 1 AR
        return x + ared[0] + ared[1]

    def step(s, backend):
        block_fn = sequential_block if backend == 'sequential' else parallel_block
        for p in params:
            p.grad = None
        idx = s % len(inputs)
        x = embed(inputs[idx]).to(torch.bfloat16)
        for L in range(N_LAYERS):
            x = block_fn(x, L)
        logits = lm_head(x).float()
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets[idx].view(-1))
        loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    g = xm.all_reduce(xm.REDUCE_SUM, p.grad) / ws
                    p.data = p.data - LR * g.to(p.dtype)
        return loss

    backend = sys.argv[1] if len(sys.argv) > 1 else 'sequential'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    if rank == 0:
        print(f'[init] ws={ws} N_LAYERS={N_LAYERS} DM={DM} HID={HID} '
              f'shard_dm={shard_dm} shard_hid={shard_hid} B={B} S={S}', flush=True)
        print(f'[init] backend={backend} steps={steps}', flush=True)

    for _ in range(WARMUP):
        loss = step(0, backend)
        _ = loss.item()
    if rank == 0:
        print('[init] warmup done', flush=True)

    times = []
    losses = []
    for s in range(steps):
        xm.mark_step()
        t0 = time.time()
        loss = step(s, backend)
        lv = loss.item()
        losses.append(lv)
        times.append((time.time() - t0) * 1000)
        if rank == 0 and (s + 1) % 50 == 0:
            recent = times[-25:]
            print(f'  step {s+1}: median_ms={statistics.median(recent):.2f} loss={lv:.3f}', flush=True)

    if rank == 0:
        steady = times[STEADY_FROM:] if len(times) > STEADY_FROM else times
        print(f'[bench] {backend} steady_mean={statistics.mean(steady):.2f}ms '
              f'steady_median={statistics.median(steady):.2f}ms '
              f'final_loss={losses[-1]:.3f}', flush=True)
        with open(f'/tmp/tp_search/llama_parattn_{backend}.json', 'w') as f:
            json.dump({'backend': backend,
                       'steady_mean_ms': statistics.mean(steady),
                       'steady_median_ms': statistics.median(steady),
                       'final_loss': losses[-1], 'steps': steps,
                       'all_ms': times, 'losses': losses[-100:]}, f)


if __name__ == '__main__':
    main()
