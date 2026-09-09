"""Block-timed Llama e2e: no per-step .item(), time K-step blocks instead.

Tests whether bundling forward + backward + optimizer into one graph
(no intermediate .item() boundary) gives more speedup beyond the
1.10x baseline measured in train_llama_e2e.py.
"""
import os, sys, time, json, statistics
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

DM = 2048; HID = 5376; N_LAYERS_PER_STAGE = 2; N_MB = 4
B = 1; S = 256; VOCAB = 224 * 32; LR = 1e-4
WARMUP_STEPS = 5
N_BLOCKS = 20            # how many timing blocks
STEPS_PER_BLOCK = 50     # steps per block (one .item() per block)


def main():
    dev = xm.xla_device()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    rank = xr.global_ordinal()
    half = ws // 2
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    shard_hid = HID // ws
    shard_vocab = VOCAB // ws

    embed = nn.Embedding(VOCAB, DM, dtype=torch.bfloat16).to(dev)
    w_gate = nn.ParameterList([nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01)
                               for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    w_up = nn.ParameterList([nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01)
                             for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    w_down = nn.ParameterList([nn.Parameter(torch.randn(shard_hid, DM, dtype=torch.bfloat16) * 0.01)
                               for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    lm_head_shard = nn.Parameter(torch.randn(DM, shard_vocab, dtype=torch.bfloat16) * 0.01).to(dev)
    rep_params = list(embed.parameters()) + [lm_head_shard]
    sharded_params = list(w_gate) + list(w_up) + list(w_down)
    all_params = rep_params + sharded_params

    inputs = [torch.randint(0, VOCAB, (B, S), device=dev, dtype=torch.int64) for _ in range(N_MB)]
    targets = [torch.randint(0, VOCAB, (B, S), device=dev, dtype=torch.int64) for _ in range(N_MB)]

    def block(x, L):
        w1 = xm.all_gather(w_gate[L], dim=1); w2 = xm.all_gather(w_up[L], dim=1)
        w3 = xm.all_gather(w_down[L], dim=0)
        return x + xm.all_reduce(xm.REDUCE_SUM, ((x @ w1) * (x @ w2)) @ w3) / ws

    def transfer(act, src):
        buf = torch.zeros(half, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src: buf = buf.clone(); buf[pair_id] = act
        return xm.all_reduce(xm.REDUCE_SUM, buf)[pair_id]

    def transfer_batched(acts, src):
        buf = torch.zeros(half, N_MB, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src: buf = buf.clone(); buf[pair_id] = acts
        return xm.all_reduce(xm.REDUCE_SUM, buf)[pair_id]

    def loss_fn(h, t):
        logits = (h @ lm_head_shard).float()
        sum_exp = xm.all_reduce(xm.REDUCE_SUM, torch.exp(logits).sum(dim=-1))
        lo = rank * shard_vocab; hi = lo + shard_vocab
        mask = (t >= lo) & (t < hi)
        idx = (t - lo).clamp(0, shard_vocab - 1)
        tlogit = torch.where(mask, logits.gather(-1, idx.unsqueeze(-1)).squeeze(-1), torch.zeros_like(sum_exp))
        return (torch.log(sum_exp + 1e-20) - xm.all_reduce(xm.REDUCE_SUM, tlogit)).mean()

    def one_step_per_mb():
        for p in all_params: p.grad = None
        loss_total = 0.0
        for m in range(N_MB):
            if stage == 0:
                h = embed(inputs[m]).to(torch.bfloat16)
                for L in range(N_LAYERS_PER_STAGE): h = block(h, L)
            else:
                h = torch.zeros(B, S, DM, device=dev, dtype=torch.bfloat16)
            h_t = transfer(h, src=0)
            if stage == 1:
                h2 = h_t
                for L in range(N_LAYERS_PER_STAGE): h2 = block(h2, L)
                loss_total = loss_total + loss_fn(h2, targets[m])
            xm.mark_step()
        if stage == 1:
            (loss_total / N_MB).backward()
        else:
            d = sum(transfer(embed(inputs[m]).to(torch.bfloat16), src=0).sum() * 0 for m in range(N_MB))
            d.backward()
        with torch.no_grad():
            for p in all_params:
                if p.grad is not None:
                    g = xm.all_reduce(xm.REDUCE_SUM, p.grad) / ws
                    p.data = p.data - LR * g.to(p.dtype)

    def one_step_bundled():
        for p in all_params: p.grad = None
        if stage == 0:
            outs = []
            for m in range(N_MB):
                h = embed(inputs[m]).to(torch.bfloat16)
                for L in range(N_LAYERS_PER_STAGE): h = block(h, L)
                outs.append(h)
            h_all = torch.stack(outs, dim=0)
        else:
            h_all = torch.zeros(N_MB, B, S, DM, device=dev, dtype=torch.bfloat16)
        h_all = transfer_batched(h_all, src=0)
        loss_total = 0.0
        if stage == 1:
            for m in range(N_MB):
                h2 = h_all[m]
                for L in range(N_LAYERS_PER_STAGE): h2 = block(h2, L)
                loss_total = loss_total + loss_fn(h2, targets[m])
            (loss_total / N_MB).backward()
        else:
            (h_all.sum() * 0).backward()
        with torch.no_grad():
            for p in all_params:
                if p.grad is not None:
                    g = xm.all_reduce(xm.REDUCE_SUM, p.grad) / ws
                    p.data = p.data - LR * g.to(p.dtype)

    backend = sys.argv[1] if len(sys.argv) > 1 else 'per_mb'
    one_step = one_step_per_mb if backend == 'per_mb' else one_step_bundled

    bench_rank = half
    if rank == bench_rank:
        print(f'[init] backend={backend} N_BLOCKS={N_BLOCKS} STEPS_PER_BLOCK={STEPS_PER_BLOCK}', flush=True)

    # Warmup
    for _ in range(WARMUP_STEPS):
        one_step()
    if rank == bench_rank:
        _ = all_params[0].sum().item()
        print('[init] warmup done', flush=True)
    else:
        _ = all_params[0].sum().item()

    per_step_times = []
    for blk in range(N_BLOCKS):
        xm.mark_step()
        t0 = time.time()
        for _ in range(STEPS_PER_BLOCK):
            one_step()
        # ONE blocking sync per block: total time / STEPS_PER_BLOCK = per-step time
        _ = all_params[0].sum().item()
        wall_ms = (time.time() - t0) * 1000
        per_step = wall_ms / STEPS_PER_BLOCK
        per_step_times.append(per_step)
        if rank == bench_rank:
            print(f'  block {blk+1}/{N_BLOCKS}: per_step={per_step:.2f}ms (block_wall={wall_ms:.0f}ms)', flush=True)

    if rank == bench_rank:
        # Skip first block as warmup-equivalent
        steady = per_step_times[2:]
        print(f'[bench] {backend} steady_mean={statistics.mean(steady):.2f}ms '
              f'steady_median={statistics.median(steady):.2f}ms n={len(steady)}', flush=True)
        with open(f'/tmp/tp_search/llama_e2e_block_{backend}.json', 'w') as f:
            json.dump({'backend': backend,
                       'per_step_ms': per_step_times,
                       'steady_mean_ms': statistics.mean(steady),
                       'steady_median_ms': statistics.median(steady),
                       'N_BLOCKS': N_BLOCKS,
                       'STEPS_PER_BLOCK': STEPS_PER_BLOCK}, f)


if __name__ == '__main__':
    main()
