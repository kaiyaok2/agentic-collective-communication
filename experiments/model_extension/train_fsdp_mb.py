"""FSDP-style sharded weight prefetch with microbatching.

5th-problem variant in the FSDP direction. Each rank holds 1/ws of
each MLP weight; per layer the agent AG's the weight, runs the matmul,
and frees the gathered copy. With N layers and M microbatches per step:
  baseline (per-mb mark_step): M graphs × (XLA fuses N AGs each) = M
    structurally unfused AG calls per step
  agent  (bundled): all M μbatches in one graph
"""
import os, sys, time, json, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

DM = 2048
HID = 5504    # llama-7b ratio; not divisible by 224 → use 5376
N_LAYERS = 4
N_MB = 4
B = 1
S = 512
WARMUP = 5


def main():
    dev = xm.xla_device()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    rank = xr.global_ordinal()
    # Use HID divisible by ws=224. 224*24=5376, llama-7b-ish.
    hid_eff = 5376
    assert hid_eff % ws == 0, f'hid={hid_eff} must divide ws={ws}'
    shard_hid = hid_eff // ws

    # Sharded along HID dim. w1: (DM, shard_hid). w2: (shard_hid, DM).
    sharded_w1 = [torch.randn(DM, shard_hid, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(N_LAYERS)]
    sharded_w2 = [torch.randn(shard_hid, DM, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(N_LAYERS)]

    inputs = [torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(N_MB)]

    def fsdp_block(x, L):
        # Gather full weight then matmul. Releases gathered weight after layer.
        w1_full = xm.all_gather(sharded_w1[L], dim=1)   # (DM, hid_eff)
        h = F.silu(torch.matmul(x, w1_full))
        w2_full = xm.all_gather(sharded_w2[L], dim=0)   # (hid_eff, DM)
        return torch.matmul(h, w2_full)

    def step_per_mb():
        outs = []
        for m in range(N_MB):
            h = inputs[m]
            for L in range(N_LAYERS):
                h = h + fsdp_block(h, L)
            outs.append(h)
            xm.mark_step()                              # per-μbatch barrier
        return outs

    def step_bundled():
        outs = []
        for m in range(N_MB):
            h = inputs[m]
            for L in range(N_LAYERS):
                h = h + fsdp_block(h, L)
            outs.append(h)
        return outs

    backend = sys.argv[1] if len(sys.argv) > 1 else 'per_mb'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    fn = step_per_mb if backend == 'per_mb' else step_bundled

    if rank == 0:
        print(f'[init] ws={ws} N_LAYERS={N_LAYERS} N_MB={N_MB} DM={DM} hid={hid_eff} shard_hid={shard_hid}', flush=True)
        print(f'[init] backend={backend} steps={steps}', flush=True)

    for _ in range(WARMUP):
        outs = fn()
        _ = outs[-1].sum().item()
    if rank == 0:
        print('[init] warmup done', flush=True)

    times = []
    for s in range(steps):
        xm.mark_step()
        t0 = time.time()
        outs = fn()
        _ = outs[-1].sum().item()
        times.append((time.time() - t0) * 1000)

    if rank == 0:
        s2 = times[10:] if len(times) > 10 else times
        print(f'[bench] {backend} mean={statistics.mean(s2):.2f}ms median={statistics.median(s2):.2f}ms', flush=True)
        with open(f'/tmp/tp_search/fsdp_mb_result_{backend}.json', 'w') as f:
            json.dump({'backend': backend, 'mean_ms': statistics.mean(s2),
                       'med_ms': statistics.median(s2), 'all': times,
                       'DM': DM, 'hid': hid_eff, 'N_LAYERS': N_LAYERS, 'N_MB': N_MB, 'B': B, 'S': S}, f)


if __name__ == '__main__':
    main()
